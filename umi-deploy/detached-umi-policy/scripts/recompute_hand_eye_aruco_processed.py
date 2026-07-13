#!/usr/bin/env python3
"""
Recompute ArUco detections on processed/rectified hand-eye images.

This script preserves the original sample["aruco"] field and writes a new
sample["processed_aruco"] field. It does not touch robot poses, URDF, runtime
calibration, or hand-eye matrix conventions.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml

from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTR = REPO_ROOT / "umi-deploy/data_local/calibration/cam0_sensor_intrinsics.json"
DEFAULT_ARUCO = REPO_ROOT / "umi-deploy/data_local/hand_eye_tags/aruco_config_tag12_147mm.yaml"


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
        "K": np.asarray(rectified_K, dtype=np.float64),
        "D": np.zeros((5, 1), dtype=np.float64),
        "resolution": size,
    }


def detect_pinhole(img_rgb, aruco_config, tag_id, marker_size_m, K):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(
        img_rgb,
        aruco_config["aruco_dict"],
        parameters=params,
    )
    if ids is None:
        return None
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id != tag_id:
            continue
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            this_corners,
            marker_size_m,
            K,
            np.zeros((5, 1), dtype=np.float64),
        )
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        corner_arr = np.asarray(this_corners, dtype=np.float64).reshape(4, 2)
        projected, _ = cv2.projectPoints(
            marker_object_points(marker_size_m),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            K,
            np.zeros((5, 1), dtype=np.float64),
        )
        projected = projected.reshape(4, 2)
        reproj = np.linalg.norm(projected - corner_arr, axis=1)
        return {
            "tag_id": int(tag_id),
            "detected": True,
            "corners": corner_arr,
            "rvec": rvec,
            "tvec": tvec,
            "z_mm": float(tvec[2] * 1000.0),
            "norm_mm": float(np.linalg.norm(tvec) * 1000.0),
            "reprojection_error_px": reproj,
            "reprojection_mean_px": float(reproj.mean()),
            "reprojection_max_px": float(reproj.max()),
            "projected_corners": projected,
        }
    return None


def serializable_aruco(tag, rectifier, processed_mode, balance, fov_scale):
    if tag is None:
        return {
            "detected": False,
            "processed_mode": processed_mode,
            "processed_K": rectifier["K"].tolist(),
            "distortion_model": "zero_pinhole",
            "rectify_balance": float(balance),
            "rectify_fov_scale": float(fov_scale),
        }
    return {
        "tag_id": int(tag["tag_id"]),
        "detected": True,
        "corners": np.asarray(tag["corners"], dtype=np.float64).tolist(),
        "rvec": np.asarray(tag["rvec"], dtype=np.float64).reshape(3).tolist(),
        "tvec": np.asarray(tag["tvec"], dtype=np.float64).reshape(3).tolist(),
        "z_mm": float(tag["z_mm"]),
        "norm_mm": float(tag["norm_mm"]),
        "reprojection_error_px": np.asarray(tag["reprojection_error_px"], dtype=np.float64).tolist(),
        "reprojection_mean_px": float(tag["reprojection_mean_px"]),
        "reprojection_max_px": float(tag["reprojection_max_px"]),
        "processed_mode": processed_mode,
        "processed_K": rectifier["K"].tolist(),
        "distortion_model": "zero_pinhole",
        "rectify_balance": float(balance),
        "rectify_fov_scale": float(fov_scale),
    }


def draw_overlay(rectified_rgb, tag, sample_idx):
    vis = rectified_rgb.copy()
    if tag is None:
        cv2.putText(
            vis,
            f"sample={sample_idx:03d} processed tag not detected",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return vis
    detected = np.asarray(tag["corners"], dtype=np.int32).reshape(-1, 1, 2)
    projected = np.asarray(tag["projected_corners"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [detected], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.polylines(vis, [projected], True, (0, 180, 255), 1, cv2.LINE_AA)
    text = (
        f"sample={sample_idx:03d} processed z={tag['z_mm']:.1f}mm "
        f"norm={tag['norm_mm']:.1f}mm reproj={tag['reprojection_mean_px']:.2f}px"
    )
    cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
    return vis


def get_raw_aruco(sample, tag_id):
    tag = sample.get("aruco")
    if not tag or not tag.get("detected", True):
        return None
    if int(tag.get("tag_id", tag_id)) != tag_id:
        return None
    if not all(key in tag for key in ("rvec", "tvec", "corners")):
        return None
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    return {
        "z_mm": float(tag.get("z_mm", tvec[2] * 1000.0)),
        "norm_mm": float(tag.get("norm_mm", np.linalg.norm(tvec) * 1000.0)),
        "reprojection_mean_px": float(tag.get("reprojection_mean_px", np.nan)),
    }


def stat_line(name, values, unit):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return f"{name}: no data"
    return (
        f"{name}: mean={values.mean():.3f}{unit} median={np.median(values):.3f}{unit} "
        f"min={values.min():.3f}{unit} max={values.max():.3f}{unit}"
    )


def count_in_range(tags, lo, hi):
    return sum(1 for tag in tags if tag is not None and lo <= tag["norm_mm"] <= hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--output_pkl", required=True)
    parser.add_argument("--intr_json", default=str(DEFAULT_INTR))
    parser.add_argument("--aruco_yaml", default=str(DEFAULT_ARUCO))
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--output_overlay_dir", required=True)
    parser.add_argument("--processed_mode", choices=["rectified"], default="rectified")
    parser.add_argument("--rectify_balance", type=float, default=0.0)
    parser.add_argument("--rectify_fov_scale", type=float, default=1.0)
    args = parser.parse_args()

    samples = pickle.load(open(args.pkl, "rb"))
    if not samples:
        raise ValueError("input pkl contains no samples")
    raw_intr = parse_fisheye_intrinsics(json.load(open(args.intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(args.aruco_yaml, "r")))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    resolution = samples[0]["img"].shape[:2][::-1]
    rectifier = build_rectifier(
        raw_intr,
        resolution,
        balance=args.rectify_balance,
        fov_scale=args.rectify_fov_scale,
    )

    overlay_dir = Path(args.output_overlay_dir).expanduser().resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)

    raw_tags = []
    processed_tags = []
    failed = []
    out_samples = []
    for idx, sample in enumerate(samples):
        out_sample = dict(sample)
        img_rgb = np.asarray(sample["img"], dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        rectified_bgr = cv2.remap(
            img_bgr,
            rectifier["map1"],
            rectifier["map2"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        tag = detect_pinhole(rectified_rgb, aruco_config, args.tag_id, marker_size_m, rectifier["K"])
        if tag is None:
            failed.append(idx)
        out_sample["processed_aruco"] = serializable_aruco(
            tag,
            rectifier,
            args.processed_mode,
            args.rectify_balance,
            args.rectify_fov_scale,
        )
        overlay = draw_overlay(rectified_rgb, tag, idx)
        cv2.imwrite(str(overlay_dir / f"sample_{idx:03d}_processed_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        raw_tags.append(get_raw_aruco(sample, args.tag_id))
        processed_tags.append(None if tag is None else tag)
        out_samples.append(out_sample)

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out_samples, open(output_pkl, "wb"))

    raw_detected = sum(tag is not None for tag in raw_tags)
    processed_detected = sum(tag is not None for tag in processed_tags)
    raw_z = [tag["z_mm"] for tag in raw_tags if tag is not None]
    raw_norm = [tag["norm_mm"] for tag in raw_tags if tag is not None]
    raw_reproj = [tag["reprojection_mean_px"] for tag in raw_tags if tag is not None]
    processed_z = [tag["z_mm"] for tag in processed_tags if tag is not None]
    processed_norm = [tag["norm_mm"] for tag in processed_tags if tag is not None]
    processed_reproj = [tag["reprojection_mean_px"] for tag in processed_tags if tag is not None]

    summary = {
        "input_pkl": str(Path(args.pkl).expanduser().resolve()),
        "output_pkl": str(output_pkl),
        "overlay_dir": str(overlay_dir),
        "samples_total": len(samples),
        "raw_aruco_detected": raw_detected,
        "processed_aruco_detected": processed_detected,
        "processed_failed_indices": failed,
        "processed_K": rectifier["K"].tolist(),
        "processed_mode": args.processed_mode,
        "raw_250_350_count": count_in_range(raw_tags, 250.0, 350.0),
        "processed_250_350_count": count_in_range(processed_tags, 250.0, 350.0),
        "raw_300_350_count": count_in_range(raw_tags, 300.0, 350.0),
        "processed_300_350_count": count_in_range(processed_tags, 300.0, 350.0),
    }
    json.dump(summary, open(output_pkl.with_suffix(".summary.json"), "w"), indent=2)

    print(f"samples_total: {len(samples)}")
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"processed_mode={args.processed_mode}")
    print(f"processed_K={rectifier['K'].tolist()}")
    print(f"raw_aruco_detected={raw_detected}")
    print(f"processed_aruco_detected={processed_detected}")
    print(f"processed_failed_indices={failed}")
    print(stat_line("raw_z_mm", raw_z, "mm"))
    print(stat_line("processed_z_mm", processed_z, "mm"))
    print(stat_line("raw_norm_mm", raw_norm, "mm"))
    print(stat_line("processed_norm_mm", processed_norm, "mm"))
    print(stat_line("raw_reprojection_mean_px", raw_reproj, "px"))
    print(stat_line("processed_reprojection_mean_px", processed_reproj, "px"))
    print(f"raw_250_350_count={summary['raw_250_350_count']}")
    print(f"processed_250_350_count={summary['processed_250_350_count']}")
    print(f"raw_300_350_count={summary['raw_300_350_count']}")
    print(f"processed_300_350_count={summary['processed_300_350_count']}")
    print(f"output_pkl={output_pkl}")
    print(f"overlay_dir={overlay_dir}")


if __name__ == "__main__":
    main()
