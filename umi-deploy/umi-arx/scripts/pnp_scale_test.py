#!/usr/bin/env python3
"""
Collect and analyze ArUco PnP distance-scale samples.

The camera path intentionally uses direct OpenCV V4L2 capture:

  cv2.VideoCapture(device, cv2.CAP_V4L2)

No UvcCamera, shared memory, recorder, or threaded camera buffer is used.
"""

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml

from utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


DEFAULT_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0"
)
DEFAULT_INTRINSICS = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "calibration",
        "cam0_sensor_intrinsics.json",
    )
)
DEFAULT_ARUCO_YAML = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "hand_eye_tags",
        "aruco_config_tag12_147mm.yaml",
    )
)
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "hand_eye_recalib_640",
        "pnp_scale_test",
    )
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def marker_object_points(marker_size_m):
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)


def project_marker_corners(rvec, tvec, marker_size_m, intr):
    projected, _ = cv2.fisheye.projectPoints(
        marker_object_points(marker_size_m),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        intr["K"],
        intr["D"],
    )
    return projected.reshape(-1, 2)


def project_marker_corners_pinhole(rvec, tvec, marker_size_m, K):
    projected, _ = cv2.projectPoints(
        marker_object_points(marker_size_m),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def project_marker_corners_model(rvec, tvec, marker_size_m, intr, projection_model):
    if projection_model == "rectified_pinhole":
        return project_marker_corners_pinhole(rvec, tvec, marker_size_m, intr["K"])
    return project_marker_corners(rvec, tvec, marker_size_m, intr)


def reprojection_error_px(corners, rvec, tvec, marker_size_m, intr, projection_model="fisheye_raw"):
    projected = project_marker_corners_model(rvec, tvec, marker_size_m, intr, projection_model)
    return np.linalg.norm(projected - np.asarray(corners, dtype=np.float64).reshape(4, 2), axis=1)


def corner_pixel_metrics(corners):
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    bbox = corners.max(axis=0) - corners.min(axis=0)
    return {
        "corner_side_px": float(sides.mean()),
        "bbox_width_px": float(bbox[0]),
        "bbox_height_px": float(bbox[1]),
    }


def parse_distance_mm(text):
    value = text.strip().lower().replace("mm", "")
    return float(value)


def prompt_distance_mm(previous=None):
    suffix = "" if previous is None else f" [{previous:.1f} mm]"
    while True:
        raw = input(f"Enter measured tag-to-camera distance in mm{suffix}: ").strip()
        if raw == "" and previous is not None:
            return previous
        try:
            return parse_distance_mm(raw)
        except ValueError:
            print("Invalid distance. Examples: 150, 200mm, 250.5")


def serializable_record(record):
    metrics = corner_pixel_metrics(record["corners"])
    return {
        "sample_idx": int(record["sample_idx"]),
        "timestamp": float(record["timestamp"]),
        "measured_distance_mm": float(record["measured_distance_mm"]),
        "image_path": record["image_path"],
        "raw_image_path": record.get("raw_image_path"),
        "projection_model": record.get("projection_model", "fisheye_raw"),
        "corners": np.asarray(record["corners"], dtype=float).tolist(),
        "rvec": np.asarray(record["rvec"], dtype=float).reshape(3).tolist(),
        "tvec": np.asarray(record["tvec"], dtype=float).reshape(3).tolist(),
        "tvec_z_m": float(record["tvec_z_m"]),
        "tvec_norm_m": float(record["tvec_norm_m"]),
        "reprojection_error_px": np.asarray(record["reprojection_error_px"], dtype=float).tolist(),
        "reprojection_mean_px": float(record["reprojection_mean_px"]),
        "reprojection_max_px": float(record["reprojection_max_px"]),
        "corner_side_px": float(record.get("corner_side_px", metrics["corner_side_px"])),
        "bbox_width_px": float(record.get("bbox_width_px", metrics["bbox_width_px"])),
        "bbox_height_px": float(record.get("bbox_height_px", metrics["bbox_height_px"])),
    }


def write_outputs(output_dir, records, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pnp_scale_samples.pkl", "wb") as f:
        pickle.dump(records, f)

    json_records = [serializable_record(r) for r in records]
    with open(output_dir / "pnp_scale_samples.json", "w") as f:
        json.dump({"samples": json_records, "summary": summary}, f, indent=2)

    with open(output_dir / "pnp_scale_samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "timestamp",
                "measured_distance_mm",
                "tvec_z_mm",
                "tvec_norm_mm",
                "reprojection_mean_px",
                "reprojection_max_px",
                "corner_side_px",
                "bbox_width_px",
                "bbox_height_px",
                "projection_model",
                "image_path",
                "raw_image_path",
            ],
        )
        writer.writeheader()
        for r in records:
            metrics = corner_pixel_metrics(r["corners"])
            writer.writerow(
                {
                    "sample_idx": int(r["sample_idx"]),
                    "timestamp": float(r["timestamp"]),
                    "measured_distance_mm": float(r["measured_distance_mm"]),
                    "tvec_z_mm": float(r["tvec_z_m"] * 1000.0),
                    "tvec_norm_mm": float(r["tvec_norm_m"] * 1000.0),
                    "reprojection_mean_px": float(r["reprojection_mean_px"]),
                    "reprojection_max_px": float(r["reprojection_max_px"]),
                    "corner_side_px": float(r.get("corner_side_px", metrics["corner_side_px"])),
                    "bbox_width_px": float(r.get("bbox_width_px", metrics["bbox_width_px"])),
                    "bbox_height_px": float(r.get("bbox_height_px", metrics["bbox_height_px"])),
                    "projection_model": r.get("projection_model", "fisheye_raw"),
                    "image_path": r["image_path"],
                    "raw_image_path": r.get("raw_image_path"),
                }
            )

    with open(output_dir / "pnp_scale_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_record_image_bgr(record):
    image_path = record.get("image_path")
    if image_path:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is not None:
            return np.ascontiguousarray(img)
    if "image" not in record:
        raise RuntimeError(f"sample {record.get('sample_idx')} has no readable image_path or embedded image")
    image_rgb = np.asarray(record["image"])
    return np.ascontiguousarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def build_rectifier(raw_intr, resolution, balance=0.0, fov_scale=1.0):
    intr = convert_fisheye_intrinsics_resolution(raw_intr, resolution)
    size = (int(resolution[0]), int(resolution[1]))
    K = np.asarray(intr["K"], dtype=np.float64)
    D = np.asarray(intr["D"], dtype=np.float64)
    rectified_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        size,
        np.eye(3),
        balance=float(balance),
        new_size=size,
        fov_scale=float(fov_scale),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        rectified_K,
        size,
        cv2.CV_16SC2,
    )
    return {
        "map1": map1,
        "map2": map2,
        "intr": {
            "DIM": np.asarray(size, dtype=np.int64),
            "K": np.asarray(rectified_K, dtype=np.float64),
            "D": np.zeros((4, 1), dtype=np.float64),
        },
    }


def rectify_frame(frame_bgr, rectifier):
    return cv2.remap(
        frame_bgr,
        rectifier["map1"],
        rectifier["map2"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def detect_localize_aruco_tags_pinhole(img, aruco_dict, marker_size_map, K, refine_subpix=True):
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(image=img, dictionary=aruco_dict, parameters=param)
    if ids is None or len(corners) == 0:
        return {}

    tag_dict = {}
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        marker_size_m = marker_size_map[this_id]
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            this_corners,
            marker_size_m,
            np.asarray(K, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64),
        )
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


def make_detection_record(
    sample_idx,
    timestamp,
    measured_distance_mm,
    image_bgr,
    image_path,
    corners,
    rvec,
    tvec,
    reproj,
    projection_model,
    raw_image_path=None,
):
    metrics = corner_pixel_metrics(corners)
    return {
        "sample_idx": int(sample_idx),
        "timestamp": float(timestamp),
        "measured_distance_mm": float(measured_distance_mm),
        "image": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy(),
        "image_path": str(image_path),
        "raw_image_path": None if raw_image_path is None else str(raw_image_path),
        "projection_model": projection_model,
        "corners": np.asarray(corners, dtype=np.float64).reshape(4, 2).copy(),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).copy(),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3).copy(),
        "tvec_z_m": float(np.asarray(tvec, dtype=np.float64).reshape(3)[2]),
        "tvec_norm_m": float(np.linalg.norm(np.asarray(tvec, dtype=np.float64).reshape(3))),
        "reprojection_error_px": np.asarray(reproj, dtype=np.float64).reshape(-1).copy(),
        "reprojection_mean_px": float(np.asarray(reproj, dtype=np.float64).mean()),
        "reprojection_max_px": float(np.asarray(reproj, dtype=np.float64).max()),
        "corner_side_px": metrics["corner_side_px"],
        "bbox_width_px": metrics["bbox_width_px"],
        "bbox_height_px": metrics["bbox_height_px"],
    }


def detect_sample(frame_bgr, aruco_config, tag_id, marker_size_m, intr, rectifier=None):
    if rectifier is None:
        detect_bgr = frame_bgr
        detect_rgb = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
        tag_dict = detect_localize_aruco_tags(
            detect_rgb,
            aruco_config["aruco_dict"],
            aruco_config["marker_size_map"],
            intr,
        )
        projection_model = "fisheye_raw"
    else:
        detect_bgr = rectify_frame(frame_bgr, rectifier)
        detect_rgb = cv2.cvtColor(detect_bgr, cv2.COLOR_BGR2RGB)
        tag_dict = detect_localize_aruco_tags_pinhole(
            detect_rgb,
            aruco_config["aruco_dict"],
            aruco_config["marker_size_map"],
            rectifier["intr"]["K"],
        )
        intr = rectifier["intr"]
        projection_model = "rectified_pinhole"

    if tag_id not in tag_dict:
        return {
            "detected": False,
            "image_bgr": detect_bgr,
            "intr": intr,
            "projection_model": projection_model,
        }

    tag = tag_dict[tag_id]
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    reproj = reprojection_error_px(corners, rvec, tvec, marker_size_m, intr, projection_model)
    projected = project_marker_corners_model(rvec, tvec, marker_size_m, intr, projection_model)
    return {
        "detected": True,
        "image_bgr": detect_bgr,
        "intr": intr,
        "projection_model": projection_model,
        "corners": corners,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_error_px": reproj,
        "projected": projected,
    }


def draw_overlays(output_dir, records, marker_size_m, intr, projection_model, overlay_subdir):
    overlays_dir = output_dir / overlay_subdir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}

    for record in records:
        sample_idx = int(record["sample_idx"])
        measured = float(record["measured_distance_mm"])
        frame_bgr = load_record_image_bgr(record)
        corners = np.asarray(record["corners"], dtype=np.float64).reshape(4, 2)
        rvec = np.asarray(record["rvec"], dtype=np.float64).reshape(3)
        tvec = np.asarray(record["tvec"], dtype=np.float64).reshape(3)
        projected = project_marker_corners_model(rvec, tvec, marker_size_m, intr, projection_model)
        metrics = corner_pixel_metrics(corners)
        text = (
            f"{projection_model} sample={sample_idx:04d} d={measured:.1f}mm "
            f"z={tvec[2] * 1000.0:.1f}mm norm={np.linalg.norm(tvec) * 1000.0:.1f}mm "
            f"reproj={float(record['reprojection_mean_px']):.2f}px side={metrics['corner_side_px']:.1f}px"
        )
        vis = draw_detection(frame_bgr, corners, projected, text)
        out_path = overlays_dir / f"sample_{sample_idx:04d}_{measured:.1f}mm_overlay.png"
        cv2.imwrite(str(out_path), vis)
        grouped.setdefault(measured, []).append((sample_idx, vis))

    for measured, items in sorted(grouped.items()):
        thumbs = []
        for sample_idx, vis in sorted(items, key=lambda item: item[0]):
            thumb = cv2.resize(vis, (320, 240), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
        cols = min(5, len(thumbs))
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = np.full((rows * 240, cols * 320, 3), 32, dtype=np.uint8)
        for idx, thumb in enumerate(thumbs):
            row = idx // cols
            col = idx % cols
            sheet[row * 240 : (row + 1) * 240, col * 320 : (col + 1) * 320] = thumb
        cv2.imwrite(str(overlays_dir / f"contact_sheet_{measured:.1f}mm.png"), sheet)

    print(f"overlays_dir={overlays_dir}")


def parse_exclude_samples(value):
    if value is None or value.strip() == "":
        return set()
    result = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        result.add(int(token))
    return result


def filter_records(records, exclude_samples):
    if not exclude_samples:
        return list(records), []
    filtered = []
    excluded = []
    for record in records:
        sample_idx = int(record["sample_idx"])
        if sample_idx in exclude_samples:
            excluded.append(sample_idx)
        else:
            filtered.append(record)
    return filtered, sorted(excluded)


def recompute_rectified_records(
    records,
    output_dir,
    raw_intr,
    aruco_config,
    tag_id,
    marker_size_m,
    args,
    images_dir_name="rectified_images",
    failed_dir_name="rectified_failed_images",
    label="rectified_pinhole",
):
    rectified_images_dir = output_dir / images_dir_name
    failed_images_dir = output_dir / failed_dir_name
    rectified_images_dir.mkdir(parents=True, exist_ok=True)
    failed_images_dir.mkdir(parents=True, exist_ok=True)
    rectifier = build_rectifier(
        raw_intr,
        (args.width, args.height),
        balance=args.rectify_balance,
        fov_scale=args.rectify_fov_scale,
    )
    print(f"detection_pipeline={label}")
    print(f"rectified_K={rectifier['intr']['K'].tolist()}")

    out_records = []
    failed = []
    for record in records:
        sample_idx = int(record["sample_idx"])
        measured = float(record["measured_distance_mm"])
        frame_bgr = load_record_image_bgr(record)
        detection = detect_sample(
            frame_bgr,
            aruco_config,
            tag_id,
            marker_size_m,
            convert_fisheye_intrinsics_resolution(raw_intr, (args.width, args.height)),
            rectifier=rectifier,
        )
        if not detection["detected"]:
            failed.append(sample_idx)
            failed_vis = detection["image_bgr"].copy()
            cv2.putText(
                failed_vis,
                f"{label} sample={sample_idx:04d} d={measured:.1f}mm tag {tag_id} not detected",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                failed_vis,
                f"{label} sample={sample_idx:04d} d={measured:.1f}mm tag {tag_id} not detected",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(failed_images_dir / f"sample_{sample_idx:04d}_{measured:.1f}mm_processed_failed.png"), failed_vis)
            continue
        image_path = rectified_images_dir / f"sample_{sample_idx:04d}_{measured:.1f}mm_processed.png"
        cv2.imwrite(str(image_path), detection["image_bgr"])
        out_records.append(
            make_detection_record(
                sample_idx,
                record.get("timestamp", 0.0),
                measured,
                detection["image_bgr"],
                image_path,
                detection["corners"],
                detection["rvec"],
                detection["tvec"],
                detection["reprojection_error_px"],
                detection["projection_model"],
                raw_image_path=record.get("image_path"),
            )
        )
    if failed:
        print(f"{label} detection failed sample indices: {failed}")
    return out_records, rectifier["intr"]


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "max": float(values.max()),
        "min": float(values.min()),
    }


def analyze_records(records):
    grouped = {}
    for record in records:
        measured = float(record["measured_distance_mm"])
        grouped.setdefault(measured, []).append(record)

    distance_summaries = []
    for measured in sorted(grouped):
        group = grouped[measured]
        tvec_z_mm = np.asarray([r["tvec_z_m"] * 1000.0 for r in group], dtype=np.float64)
        tvec_norm_mm = np.asarray([r["tvec_norm_m"] * 1000.0 for r in group], dtype=np.float64)
        corner_side_px = np.asarray(
            [r.get("corner_side_px", corner_pixel_metrics(r["corners"])["corner_side_px"]) for r in group],
            dtype=np.float64,
        )
        bbox_width_px = np.asarray(
            [r.get("bbox_width_px", corner_pixel_metrics(r["corners"])["bbox_width_px"]) for r in group],
            dtype=np.float64,
        )
        bbox_height_px = np.asarray(
            [r.get("bbox_height_px", corner_pixel_metrics(r["corners"])["bbox_height_px"]) for r in group],
            dtype=np.float64,
        )
        reproj_all = np.concatenate(
            [np.asarray(r["reprojection_error_px"], dtype=np.float64).reshape(-1) for r in group]
        )
        distance_summaries.append(
            {
                "measured_distance_mm": float(measured),
                "count": len(group),
                "tvec_z_mm": stats(tvec_z_mm),
                "tvec_norm_mm": stats(tvec_norm_mm),
                "reprojection_px": stats(reproj_all),
                "corner_side_px": stats(corner_side_px),
                "bbox_width_px": stats(bbox_width_px),
                "bbox_height_px": stats(bbox_height_px),
                "scale_error_z": float(tvec_z_mm.mean() / measured - 1.0),
                "scale_error_norm": float(tvec_norm_mm.mean() / measured - 1.0),
            }
        )

    delta_summaries = []
    for prev, cur in zip(distance_summaries[:-1], distance_summaries[1:]):
        measured_delta = cur["measured_distance_mm"] - prev["measured_distance_mm"]
        z_delta = cur["tvec_z_mm"]["mean"] - prev["tvec_z_mm"]["mean"]
        norm_delta = cur["tvec_norm_mm"]["mean"] - prev["tvec_norm_mm"]["mean"]
        delta_summaries.append(
            {
                "from_mm": prev["measured_distance_mm"],
                "to_mm": cur["measured_distance_mm"],
                "measured_delta_mm": float(measured_delta),
                "pnp_delta_z_mm": float(z_delta),
                "pnp_delta_norm_mm": float(norm_delta),
                "delta_scale_error_z": float(z_delta / measured_delta - 1.0),
                "delta_scale_error_norm": float(norm_delta / measured_delta - 1.0),
            }
        )

    return {
        "distances": distance_summaries,
        "adjacent_deltas": delta_summaries,
    }


def print_summary(summary):
    print("\nPNP_SCALE_SUMMARY")
    for item in summary["distances"]:
        print(
            f"distance={item['measured_distance_mm']:.1f} mm count={item['count']} "
            f"tvec_z mean={item['tvec_z_mm']['mean']:.3f} median={item['tvec_z_mm']['median']:.3f} "
            f"std={item['tvec_z_mm']['std']:.3f} mm "
            f"norm mean={item['tvec_norm_mm']['mean']:.3f} median={item['tvec_norm_mm']['median']:.3f} "
            f"std={item['tvec_norm_mm']['std']:.3f} mm "
            f"reproj mean={item['reprojection_px']['mean']:.3f} median={item['reprojection_px']['median']:.3f} "
            f"max={item['reprojection_px']['max']:.3f} px "
            f"side mean={item['corner_side_px']['mean']:.3f} px "
            f"bbox mean={item['bbox_width_px']['mean']:.3f}x{item['bbox_height_px']['mean']:.3f} px "
            f"scale_error_z={item['scale_error_z'] * 100.0:.2f}% "
            f"scale_error_norm={item['scale_error_norm'] * 100.0:.2f}%"
        )
    print("\nPNP_DELTA_SCALE")
    for item in summary["adjacent_deltas"]:
        print(
            f"{item['from_mm']:.1f}->{item['to_mm']:.1f} mm "
            f"measured_delta={item['measured_delta_mm']:.3f} mm "
            f"pnp_delta_z={item['pnp_delta_z_mm']:.3f} mm "
            f"pnp_delta_norm={item['pnp_delta_norm_mm']:.3f} mm "
            f"delta_scale_error_z={item['delta_scale_error_z'] * 100.0:.2f}% "
            f"delta_scale_error_norm={item['delta_scale_error_norm'] * 100.0:.2f}%"
        )


def distance_summary_map(summary):
    return {float(item["measured_distance_mm"]): item for item in summary.get("distances", [])}


def delta_summary_map(summary):
    return {
        (float(item["from_mm"]), float(item["to_mm"])): item
        for item in summary.get("adjacent_deltas", [])
    }


def write_compare_outputs(output_dir, raw_summary, processed_summary):
    raw_by_dist = distance_summary_map(raw_summary)
    processed_by_dist = distance_summary_map(processed_summary)
    distances = sorted(set(raw_by_dist) | set(processed_by_dist))
    rows = []
    for measured in distances:
        raw_item = raw_by_dist.get(measured)
        processed_item = processed_by_dist.get(measured)
        rows.append(
            {
                "distance_mm": measured,
                "raw_detection_count": 0 if raw_item is None else raw_item["count"],
                "processed_detection_count": 0 if processed_item is None else processed_item["count"],
                "raw_z_mean": None if raw_item is None else raw_item["tvec_z_mm"]["mean"],
                "processed_z_mean": None if processed_item is None else processed_item["tvec_z_mm"]["mean"],
                "raw_norm_mean": None if raw_item is None else raw_item["tvec_norm_mm"]["mean"],
                "processed_norm_mean": None if processed_item is None else processed_item["tvec_norm_mm"]["mean"],
                "raw_reproj_mean": None if raw_item is None else raw_item["reprojection_px"]["mean"],
                "processed_reproj_mean": None if processed_item is None else processed_item["reprojection_px"]["mean"],
                "raw_scale_error_z": None if raw_item is None else raw_item["scale_error_z"],
                "processed_scale_error_z": None if processed_item is None else processed_item["scale_error_z"],
                "raw_scale_error_norm": None if raw_item is None else raw_item["scale_error_norm"],
                "processed_scale_error_norm": None if processed_item is None else processed_item["scale_error_norm"],
            }
        )

    raw_by_delta = delta_summary_map(raw_summary)
    processed_by_delta = delta_summary_map(processed_summary)
    delta_keys = sorted(set(raw_by_delta) | set(processed_by_delta))
    delta_rows = []
    for key in delta_keys:
        raw_item = raw_by_delta.get(key)
        processed_item = processed_by_delta.get(key)
        from_mm, to_mm = key
        delta_rows.append(
            {
                "from_mm": from_mm,
                "to_mm": to_mm,
                "measured_delta_mm": to_mm - from_mm,
                "raw_delta_z": None if raw_item is None else raw_item["pnp_delta_z_mm"],
                "processed_delta_z": None if processed_item is None else processed_item["pnp_delta_z_mm"],
                "raw_delta_norm": None if raw_item is None else raw_item["pnp_delta_norm_mm"],
                "processed_delta_norm": None if processed_item is None else processed_item["pnp_delta_norm_mm"],
                "raw_delta_error": None if raw_item is None else raw_item["delta_scale_error_z"],
                "processed_delta_error": None if processed_item is None else processed_item["delta_scale_error_z"],
                "raw_delta_error_norm": None if raw_item is None else raw_item["delta_scale_error_norm"],
                "processed_delta_error_norm": None if processed_item is None else processed_item["delta_scale_error_norm"],
            }
        )

    payload = {
        "raw_summary": raw_summary,
        "processed_summary": processed_summary,
        "distance_comparison": rows,
        "delta_comparison": delta_rows,
    }
    with open(output_dir / "raw_vs_processed_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(output_dir / "raw_vs_processed_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["distance_mm"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(output_dir / "raw_vs_processed_delta_scale.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(delta_rows[0].keys()) if delta_rows else ["from_mm", "to_mm"],
        )
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(row)

    print("\nPNP_RAW_VS_PROCESSED_SCALE")
    for row in rows:
        def pct(value):
            return "N/A" if value is None else f"{value * 100.0:.2f}%"

        def val(value):
            return "N/A" if value is None else f"{value:.3f}"

        print(
            f"distance={row['distance_mm']:.1f}mm "
            f"raw_count={row['raw_detection_count']} processed_count={row['processed_detection_count']} "
            f"raw_z={val(row['raw_z_mean'])} processed_z={val(row['processed_z_mean'])} "
            f"raw_norm={val(row['raw_norm_mean'])} processed_norm={val(row['processed_norm_mean'])} "
            f"raw_reproj={val(row['raw_reproj_mean'])} processed_reproj={val(row['processed_reproj_mean'])} "
            f"raw_scale_z={pct(row['raw_scale_error_z'])} processed_scale_z={pct(row['processed_scale_error_z'])} "
            f"raw_scale_norm={pct(row['raw_scale_error_norm'])} processed_scale_norm={pct(row['processed_scale_error_norm'])}"
        )

    print("\nPNP_RAW_VS_PROCESSED_DELTA_SCALE")
    for row in delta_rows:
        def pct(value):
            return "N/A" if value is None else f"{value * 100.0:.2f}%"

        def val(value):
            return "N/A" if value is None else f"{value:.3f}"

        print(
            f"{row['from_mm']:.1f}->{row['to_mm']:.1f}mm "
            f"raw_delta_z={val(row['raw_delta_z'])} processed_delta_z={val(row['processed_delta_z'])} "
            f"raw_delta_error={pct(row['raw_delta_error'])} processed_delta_error={pct(row['processed_delta_error'])}"
        )


def make_combined_contact_sheet(src_dir, distances, out_path):
    sheets = []
    for distance in distances:
        path = src_dir / f"contact_sheet_{distance:.1f}mm.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            sheets.append(img)
    if not sheets:
        return
    width = max(img.shape[1] for img in sheets)
    padded = []
    for img in sheets:
        if img.shape[1] == width:
            padded.append(img)
            continue
        canvas = np.full((img.shape[0], width, 3), 32, dtype=np.uint8)
        canvas[:, : img.shape[1]] = img
        padded.append(canvas)
    cv2.imwrite(str(out_path), np.vstack(padded))


def write_contact_sheet_aliases(output_dir):
    aliases = [
        ("raw_detect_overlay", "raw_contact_sheet_150mm.png", [150.0]),
        ("processed_detect_overlay", "processed_contact_sheet_150mm.png", [150.0]),
        ("raw_detect_overlay", "raw_contact_sheet_250_300_350mm.png", [250.0, 300.0, 350.0]),
        ("processed_detect_overlay", "processed_contact_sheet_250_300_350mm.png", [250.0, 300.0, 350.0]),
    ]
    for subdir, alias_name, distances in aliases:
        src_dir = output_dir / subdir
        if len(distances) == 1:
            src = src_dir / f"contact_sheet_{distances[0]:.1f}mm.png"
            if src.exists():
                shutil.copyfile(src, output_dir / alias_name)
        else:
            make_combined_contact_sheet(src_dir, distances, output_dir / alias_name)


def draw_detection(frame_bgr, corners, projected, text):
    vis = frame_bgr.copy()
    detected_pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    projected_pts = np.asarray(projected, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [detected_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(vis, [projected_pts], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def open_capture(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.cap_buffer_size)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")
    for _ in range(10):
        cap.read()
        time.sleep(0.02)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"camera={args.camera}")
    print(f"opencv_capture=CAP_V4L2 fourcc={fourcc_str} resolution={width}x{height} fps={fps:.3f}")
    if (width, height) != (args.width, args.height):
        raise RuntimeError(f"Camera negotiated {width}x{height}; expected {args.width}x{args.height}")
    return cap


def run_capture(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / ("rectified_images" if args.detect_on_rectified else "images")
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_images_dir = output_dir / "raw_images"
    if args.detect_on_rectified:
        raw_images_dir.mkdir(parents=True, exist_ok=True)

    raw_intr = parse_fisheye_intrinsics(load_json(args.intr_json))
    intr = convert_fisheye_intrinsics_resolution(raw_intr, (args.width, args.height))
    rectifier = None
    if args.detect_on_rectified:
        rectifier = build_rectifier(
            raw_intr,
            (args.width, args.height),
            balance=args.rectify_balance,
            fov_scale=args.rectify_fov_scale,
        )
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"detection_pipeline={'rectified_pinhole' if args.detect_on_rectified else 'fisheye_raw'}")
    if args.detect_on_rectified:
        print(f"rectified_K={rectifier['intr']['K'].tolist()}")

    current_distance_mm = prompt_distance_mm()
    records = []
    cap = open_capture(args)
    try:
        while True:
            ret, frame_bgr = cap.read()
            timestamp = time.time()
            if not ret or frame_bgr is None:
                print("camera read failed, retrying...")
                time.sleep(0.02)
                continue
            frame_bgr = np.ascontiguousarray(frame_bgr)
            detection = detect_sample(
                frame_bgr,
                aruco_config,
                args.tag_id,
                marker_size_m,
                intr,
                rectifier=rectifier,
            )

            detected = detection["detected"]
            if detected:
                tvec = detection["tvec"]
                reproj = detection["reprojection_error_px"]
                metrics = corner_pixel_metrics(detection["corners"])
                text = (
                    f"{detection['projection_model']} d={current_distance_mm:.1f}mm z={tvec[2] * 1000.0:.1f}mm "
                    f"norm={np.linalg.norm(tvec) * 1000.0:.1f}mm reproj={reproj.mean():.2f}px "
                    f"side={metrics['corner_side_px']:.1f}px samples={len(records)}"
                )
                vis = draw_detection(
                    detection["image_bgr"],
                    detection["corners"],
                    detection["projected"],
                    text,
                )
            else:
                vis = detection["image_bgr"].copy()
                cv2.putText(
                    vis,
                    f"d={current_distance_mm:.1f}mm tag {args.tag_id} not detected samples={len(records)}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("PnP Scale Test", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D"), ord("n"), ord("N")):
                current_distance_mm = prompt_distance_mm(current_distance_mm)
                continue
            if key in (ord("s"), ord("S"), ord(" ")):
                if not detected:
                    print("Cannot save: tag not detected.")
                    continue
                sample_idx = len(records)
                image_path = images_dir / f"sample_{sample_idx:04d}_{current_distance_mm:.1f}mm.png"
                cv2.imwrite(str(image_path), detection["image_bgr"])
                raw_image_path = None
                if args.detect_on_rectified:
                    raw_image_path = raw_images_dir / f"sample_{sample_idx:04d}_{current_distance_mm:.1f}mm_raw.png"
                    cv2.imwrite(str(raw_image_path), frame_bgr)
                record = make_detection_record(
                    sample_idx,
                    timestamp,
                    current_distance_mm,
                    detection["image_bgr"],
                    image_path,
                    detection["corners"],
                    detection["rvec"],
                    detection["tvec"],
                    detection["reprojection_error_px"],
                    detection["projection_model"],
                    raw_image_path=raw_image_path,
                )
                records.append(record)
                filtered_records, excluded = filter_records(records, args.exclude_samples)
                summary = analyze_records(filtered_records)
                summary["excluded_samples"] = excluded
                write_outputs(output_dir, filtered_records, summary)
                print(
                    f"saved sample={sample_idx:04d} measured={current_distance_mm:.1f}mm "
                    f"z={record['tvec_z_m'] * 1000.0:.3f}mm norm={record['tvec_norm_m'] * 1000.0:.3f}mm "
                    f"reproj_mean={record['reprojection_mean_px']:.3f}px "
                    f"side={record['corner_side_px']:.3f}px "
                    f"bbox={record['bbox_width_px']:.3f}x{record['bbox_height_px']:.3f}px"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    filtered_records, excluded = filter_records(records, args.exclude_samples)
    print(f"excluded sample indices: {excluded}")
    summary = analyze_records(filtered_records) if filtered_records else {"distances": [], "adjacent_deltas": []}
    summary["excluded_samples"] = excluded
    write_outputs(output_dir, filtered_records, summary)
    if args.write_overlays and filtered_records:
        overlay_intr = rectifier["intr"] if args.detect_on_rectified else intr
        projection_model = "rectified_pinhole" if args.detect_on_rectified else "fisheye_raw"
        overlay_subdir = "rectified_detect_overlay" if args.detect_on_rectified else "raw_detect_overlay"
        draw_overlays(output_dir, filtered_records, marker_size_m, overlay_intr, projection_model, overlay_subdir)
    print_summary(summary)
    print(f"output_dir={output_dir}")


def run_analyze(args):
    input_pkl = Path(args.analyze_pkl).expanduser().resolve()
    records = pickle.load(open(input_pkl, "rb"))
    filtered_records, excluded = filter_records(records, args.exclude_samples)
    print(f"excluded sample indices: {excluded}")
    output_dir = Path(args.output_dir).expanduser().resolve()

    raw_intr = parse_fisheye_intrinsics(load_json(args.intr_json))
    intr = convert_fisheye_intrinsics_resolution(raw_intr, (args.width, args.height))
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"detection_pipeline={'rectified_pinhole' if args.detect_on_rectified else 'fisheye_raw'}")

    overlay_intr = intr
    projection_model = "fisheye_raw"
    overlay_subdir = "raw_detect_overlay"
    if args.detect_on_rectified:
        filtered_records, overlay_intr = recompute_rectified_records(
            filtered_records,
            output_dir,
            raw_intr,
            aruco_config,
            args.tag_id,
            marker_size_m,
            args,
        )
        projection_model = "rectified_pinhole"
        overlay_subdir = "rectified_detect_overlay"

    summary = analyze_records(filtered_records)
    summary["excluded_samples"] = excluded
    write_outputs(output_dir, filtered_records, summary)
    if args.write_overlays and filtered_records:
        draw_overlays(output_dir, filtered_records, marker_size_m, overlay_intr, projection_model, overlay_subdir)
    print_summary(summary)
    print(f"analyzed={input_pkl}")
    print(f"output_dir={output_dir}")


def run_analyze_raw_vs_processed(args):
    input_pkl = Path(args.analyze_pkl).expanduser().resolve()
    records = pickle.load(open(input_pkl, "rb"))
    filtered_records, excluded = filter_records(records, args.exclude_samples)
    print(f"excluded sample indices: {excluded}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.processed_mode != "rectified":
        raise ValueError(
            "processed_mode=policy_processed needs the policy sim_fov/out_res to define processed_K. "
            "Use processed_mode=rectified for fixed 640x480 OpenCV rectification."
        )

    raw_intr = parse_fisheye_intrinsics(load_json(args.intr_json))
    intr = convert_fisheye_intrinsics_resolution(raw_intr, (args.width, args.height))
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print("raw_pipeline=fisheye_raw")
    print("processed_pipeline=rectified_pinhole")

    raw_records = list(filtered_records)
    raw_summary = analyze_records(raw_records)
    raw_summary["excluded_samples"] = excluded
    raw_pipeline_dir = output_dir / "raw_pipeline"
    write_outputs(raw_pipeline_dir, raw_records, raw_summary)

    processed_records, processed_intr = recompute_rectified_records(
        raw_records,
        output_dir,
        raw_intr,
        aruco_config,
        args.tag_id,
        marker_size_m,
        args,
        images_dir_name="processed_images",
        failed_dir_name="processed_failed_images",
        label="processed_rectified_pinhole",
    )
    processed_summary = analyze_records(processed_records)
    processed_summary["excluded_samples"] = excluded
    processed_summary["processed_detection_failed_count"] = len(raw_records) - len(processed_records)
    processed_pipeline_dir = output_dir / "processed_pipeline"
    write_outputs(processed_pipeline_dir, processed_records, processed_summary)

    if args.write_raw_overlays or args.write_overlays:
        draw_overlays(output_dir, raw_records, marker_size_m, intr, "fisheye_raw", "raw_detect_overlay")
    if args.write_processed_overlays or args.write_overlays:
        draw_overlays(
            output_dir,
            processed_records,
            marker_size_m,
            processed_intr,
            "rectified_pinhole",
            "processed_detect_overlay",
        )
    if (args.write_raw_overlays or args.write_processed_overlays or args.write_overlays) and raw_records:
        write_contact_sheet_aliases(output_dir)

    write_compare_outputs(output_dir, raw_summary, processed_summary)
    if raw_records:
        raw_150 = [r for r in raw_records if abs(float(r["measured_distance_mm"]) - 150.0) < 1e-6]
        processed_150 = [
            r for r in processed_records if abs(float(r["measured_distance_mm"]) - 150.0) < 1e-6
        ]
        print(
            f"processed_150mm_detection_count={len(processed_150)}/{len(raw_150)} "
            f"(failed={len(raw_150) - len(processed_150)})"
        )
    print(f"analyzed={input_pkl}")
    print(f"output_dir={output_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=DEFAULT_DEVICE)
    parser.add_argument("--intr_json", default=DEFAULT_INTRINSICS)
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO_YAML)
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--cap_buffer_size", type=int, default=1)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analyze_pkl", default=None, help="Analyze a previously saved pkl instead of capturing.")
    parser.add_argument(
        "--exclude_samples",
        type=parse_exclude_samples,
        default=set(),
        help="Comma-separated sample_idx values to exclude from statistics, e.g. 35,36,37,38.",
    )
    parser.add_argument(
        "--write_overlays",
        action="store_true",
        help="Write per-sample green detected-corner overlays and per-distance contact sheets.",
    )
    parser.add_argument(
        "--write_raw_overlays",
        action="store_true",
        help="When analyzing with --detect_on_processed, write raw fisheye detection overlays.",
    )
    parser.add_argument(
        "--write_processed_overlays",
        action="store_true",
        help="When analyzing with --detect_on_processed, write processed/rectified detection overlays.",
    )
    parser.add_argument(
        "--detect_on_rectified",
        action="store_true",
        help="Undistort/rectify raw fisheye frames to 640x480, detect ArUco there, and solve PnP with rectified_K and zero distortion.",
    )
    parser.add_argument(
        "--detect_on_processed",
        action="store_true",
        help="Alias for processed-image analysis. In analyze mode this runs raw and processed pipelines side by side.",
    )
    parser.add_argument(
        "--processed_mode",
        choices=["rectified", "policy_processed"],
        default="rectified",
        help="Processed image definition. policy_processed requires explicit policy FOV support and is not inferred here.",
    )
    parser.add_argument(
        "--rectify_balance",
        type=float,
        default=0.0,
        help="balance passed to cv2.fisheye.estimateNewCameraMatrixForUndistortRectify.",
    )
    parser.add_argument(
        "--rectify_fov_scale",
        type=float,
        default=1.0,
        help="fov_scale passed to cv2.fisheye.estimateNewCameraMatrixForUndistortRectify.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.detect_on_processed:
        args.detect_on_rectified = True
    if args.analyze_pkl:
        if args.detect_on_processed:
            run_analyze_raw_vs_processed(args)
        else:
            run_analyze(args)
    else:
        run_capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
