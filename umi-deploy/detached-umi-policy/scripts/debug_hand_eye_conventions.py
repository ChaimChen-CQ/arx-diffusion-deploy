#!/usr/bin/env python3
"""
Exhaustively check robot-world/hand-eye frame conventions.

This script intentionally does not trust variable names.  It enumerates these
interpretations of the saved data and calibration JSON:

  tcp_pose_as:
    - T_base_tcp
    - T_tcp_base
  tx_gripper2camera_as:
    - T_camera_tcp
    - T_tcp_camera
  tx_base2world_as:
    - T_world_base
    - T_base_world

For each combination it predicts the observed tag pose as:

  T_camera_world_pred = T_camera_tcp @ T_tcp_base @ T_base_world

and compares it with the ArUco/PnP observed T_camera_world.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import pose_to_mat


DEFAULT_PKL = (
    "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/data_local/"
    "hand_eye_recalib_640/hand_eye_calib.pkl"
)
DEFAULT_JSON = (
    "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/data_local/"
    "hand_eye_recalib_640/robot_world_hand_eye_cam0_tag145.json"
)
DEFAULT_INTR = (
    "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/data_local/"
    "calibration/cam0_sensor_intrinsics.json"
)
DEFAULT_ARUCO = (
    "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/data_local/"
    "hand_eye_tags/aruco_config_tag12_145mm.yaml"
)


def as_transform_from_rvec_tvec(rvec, tvec):
    tx = np.eye(4, dtype=np.float64)
    tx[:3, :3] = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    tx[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return tx


def marker_object_points(marker_size):
    half = float(marker_size) / 2.0
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def project_marker_fisheye(tx_camera_marker, marker_size, intr):
    object_points = marker_object_points(marker_size).reshape(-1, 1, 3)
    rvec = R.from_matrix(tx_camera_marker[:3, :3]).as_rotvec().reshape(3, 1)
    tvec = tx_camera_marker[:3, 3].reshape(3, 1)
    projected, _ = cv2.fisheye.projectPoints(
        object_points,
        np.ascontiguousarray(rvec, dtype=np.float64),
        np.ascontiguousarray(tvec, dtype=np.float64),
        intr["K"],
        intr["D"],
    )
    return projected.reshape(-1, 2)


def project_marker_undistorted_pinhole(tx_camera_marker, marker_size, intr):
    object_points = marker_object_points(marker_size)
    rvec = R.from_matrix(tx_camera_marker[:3, :3]).as_rotvec().reshape(3, 1)
    tvec = tx_camera_marker[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(
        object_points,
        np.ascontiguousarray(rvec, dtype=np.float64),
        np.ascontiguousarray(tvec, dtype=np.float64),
        intr["K"],
        np.zeros((1, 5), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def relative_transform_error(tx_left, tx_right):
    tx_err = np.linalg.inv(tx_left) @ tx_right
    trans_mm = float(np.linalg.norm(tx_err[:3, 3]) * 1000.0)
    rot_deg = float(np.rad2deg(R.from_matrix(tx_err[:3, :3]).magnitude()))
    return trans_mm, rot_deg


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "max": float(np.max(arr)),
    }


def summarize_stats(label, summary, unit):
    return (
        f"{label}: mean={summary['mean']:.6g}{unit}, "
        f"median={summary['median']:.6g}{unit}, "
        f"rms={summary['rms']:.6g}{unit}, max={summary['max']:.6g}{unit}"
    )


def draw_polyline_rgb(img, points, color, thickness):
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_corner_points_rgb(img, points, color, radius, thickness):
    for pt in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        cv2.circle(
            img,
            (int(round(pt[0])), int(round(pt[1]))),
            radius,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def save_overlay(path, sample, combo_label, sample_error, marker_size, intr):
    img = sample["img_rgb"].copy()
    pnp_projected = project_marker_fisheye(sample["tx_camera_world_obs"], marker_size, intr)
    he_projected = project_marker_fisheye(sample_error["tx_camera_world_pred"], marker_size, intr)

    green = (0, 255, 0)
    orange = (255, 165, 0)
    cyan = (0, 255, 255)
    red = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)

    draw_polyline_rgb(img, sample["corners_raw"], green, 2)
    draw_polyline_rgb(img, pnp_projected, orange, 1)
    draw_corner_points_rgb(img, pnp_projected, cyan, 4, 2)
    draw_polyline_rgb(img, he_projected, red, 2)
    draw_corner_points_rgb(img, he_projected, red, 5, 2)

    lines = [
        f"sample={sample['idx']:03d} trans={sample_error['trans_mm']:.1f}mm rot={sample_error['rot_deg']:.1f}deg",
        "green=detected orange/cyan=PnP red=hand-eye",
        combo_label,
    ]
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, black, 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, white, 1, cv2.LINE_AA)
        y += 20

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_tcp_base_from_pose(tcp_pose, tcp_pose_as):
    raw = pose_to_mat(np.asarray(tcp_pose, dtype=np.float64))
    if tcp_pose_as == "T_base_tcp":
        return np.linalg.inv(raw)
    if tcp_pose_as == "T_tcp_base":
        return raw
    raise ValueError(tcp_pose_as)


def get_camera_tcp_from_json(raw_tx_gripper2camera, tx_gripper2camera_as):
    if tx_gripper2camera_as == "T_camera_tcp":
        return raw_tx_gripper2camera
    if tx_gripper2camera_as == "T_tcp_camera":
        return np.linalg.inv(raw_tx_gripper2camera)
    raise ValueError(tx_gripper2camera_as)


def get_world_base_from_json(raw_tx_base2world, tx_base2world_as):
    if tx_base2world_as == "T_world_base":
        return raw_tx_base2world
    if tx_base2world_as == "T_base_world":
        return np.linalg.inv(raw_tx_base2world)
    raise ValueError(tx_base2world_as)


def load_samples(input_pkl, intr, aruco_config, tag_id, mask=True):
    with open(input_pkl, "rb") as f:
        raw_samples = pickle.load(f)
    if len(raw_samples) < 3:
        raise ValueError(f"Need at least 3 samples, got {len(raw_samples)}")

    marker_size = aruco_config["marker_size_map"][tag_id]
    samples = []
    fisheye_reproj_all = []
    undist_reproj_all = []

    for idx, raw in enumerate(raw_samples):
        img_rgb = np.asarray(raw["img"])
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Sample {idx} img must be HxWx3, got {img_rgb.shape}")
        this_intr = convert_fisheye_intrinsics_resolution(intr, img_rgb.shape[:2][::-1])

        detect_img = img_rgb.copy()
        if mask:
            draw_predefined_mask(
                detect_img,
                color=(0, 0, 0),
                mirror=True,
                gripper=False,
                finger=False,
            )

        tag_dict = detect_localize_aruco_tags(
            img=detect_img,
            aruco_dict=aruco_config["aruco_dict"],
            marker_size_map=aruco_config["marker_size_map"],
            fisheye_intr_dict=this_intr,
        )
        if tag_id not in tag_dict:
            continue

        tag = tag_dict[tag_id]
        tx_camera_world_obs = as_transform_from_rvec_tvec(tag["rvec"], tag["tvec"])
        corners_raw = np.asarray(tag["corners"], dtype=np.float64).reshape(-1, 2)

        pnp_raw_projected = project_marker_fisheye(tx_camera_world_obs, marker_size, this_intr)
        fisheye_err = np.linalg.norm(pnp_raw_projected - corners_raw, axis=1)
        fisheye_reproj_all.extend(fisheye_err.tolist())

        corners_undist = cv2.fisheye.undistortPoints(
            corners_raw.reshape(1, -1, 2),
            this_intr["K"],
            this_intr["D"],
            P=this_intr["K"],
        ).reshape(-1, 2)
        pnp_undist_projected = project_marker_undistorted_pinhole(
            tx_camera_world_obs, marker_size, this_intr
        )
        undist_err = np.linalg.norm(pnp_undist_projected - corners_undist, axis=1)
        undist_reproj_all.extend(undist_err.tolist())

        samples.append(
            {
                "idx": idx,
                "img_rgb": img_rgb.copy(),
                "tcp_pose": np.asarray(raw["tcp_pose"], dtype=np.float64).reshape(6),
                "tx_camera_world_obs": tx_camera_world_obs,
                "corners_raw": corners_raw,
                "fisheye_reproj_err_px": fisheye_err,
                "undist_reproj_err_px": undist_err,
                "intr": this_intr,
            }
        )

    if not samples:
        raise RuntimeError(f"Tag {tag_id} was not detected in any sample")

    return raw_samples, samples, np.asarray(fisheye_reproj_all), np.asarray(undist_reproj_all)


def evaluate_combo(samples, raw_tx_base2world, raw_tx_gripper2camera, combo):
    tcp_pose_as, tx_gripper2camera_as, tx_base2world_as = combo
    tx_camera_tcp = get_camera_tcp_from_json(raw_tx_gripper2camera, tx_gripper2camera_as)
    tx_world_base = get_world_base_from_json(raw_tx_base2world, tx_base2world_as)
    tx_base_world = np.linalg.inv(tx_world_base)

    sample_errors = []
    trans = []
    rot = []
    for sample in samples:
        tx_tcp_base = get_tcp_base_from_pose(sample["tcp_pose"], tcp_pose_as)
        tx_left = sample["tx_camera_world_obs"] @ tx_world_base
        tx_right = tx_camera_tcp @ tx_tcp_base
        tx_camera_world_pred = tx_camera_tcp @ tx_tcp_base @ tx_base_world
        trans_mm, rot_deg = relative_transform_error(tx_left, tx_right)
        trans.append(trans_mm)
        rot.append(rot_deg)
        sample_errors.append(
            {
                "idx": sample["idx"],
                "trans_mm": trans_mm,
                "rot_deg": rot_deg,
                "tx_camera_world_pred": tx_camera_world_pred,
            }
        )

    return {
        "tcp_pose_as": tcp_pose_as,
        "tx_gripper2camera_as": tx_gripper2camera_as,
        "tx_base2world_as": tx_base2world_as,
        "translation_mm": stats(trans),
        "rotation_deg": stats(rot),
        "sample_errors": sample_errors,
    }


def write_summary_files(output_dir, args, results, reproj_summary, samples):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_pkl": str(args.pkl),
        "hand_eye_json": str(args.json_file),
        "intr_json": str(args.intr_json),
        "aruco_yaml": str(args.aruco_yaml),
        "tag_id": args.tag_id,
        "samples_total": args.samples_total,
        "samples_used": len(samples),
        "aruco_reprojection": reproj_summary,
        "convention_results": [
            {
                "combo_id": result["combo_id"],
                "tcp_pose_as": result["tcp_pose_as"],
                "tx_gripper2camera_as": result["tx_gripper2camera_as"],
                "tx_base2world_as": result["tx_base2world_as"],
                "translation_mm": result["translation_mm"],
                "rotation_deg": result["rotation_deg"],
                "worst_samples_by_translation": [
                    {
                        "sample": err["idx"],
                        "trans_mm": err["trans_mm"],
                        "rot_deg": err["rot_deg"],
                    }
                    for err in sorted(
                        result["sample_errors"], key=lambda x: x["trans_mm"], reverse=True
                    )[: args.worst_n]
                ],
            }
            for result in results
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "convention_results.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "combo_id",
                "tcp_pose_as",
                "tx_gripper2camera_as",
                "tx_base2world_as",
                "trans_mean_mm",
                "trans_median_mm",
                "trans_rms_mm",
                "trans_max_mm",
                "rot_mean_deg",
                "rot_median_deg",
                "rot_rms_deg",
                "rot_max_deg",
            ]
        )
        for result in results:
            t = result["translation_mm"]
            r = result["rotation_deg"]
            writer.writerow(
                [
                    result["combo_id"],
                    result["tcp_pose_as"],
                    result["tx_gripper2camera_as"],
                    result["tx_base2world_as"],
                    f"{t['mean']:.6f}",
                    f"{t['median']:.6f}",
                    f"{t['rms']:.6f}",
                    f"{t['max']:.6f}",
                    f"{r['mean']:.6f}",
                    f"{r['median']:.6f}",
                    f"{r['rms']:.6f}",
                    f"{r['max']:.6f}",
                ]
            )

    with open(output_dir / "per_sample_errors.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["combo_id", "sample", "trans_mm", "rot_deg"])
        for result in results:
            for err in result["sample_errors"]:
                writer.writerow(
                    [
                        result["combo_id"],
                        err["idx"],
                        f"{err['trans_mm']:.6f}",
                        f"{err['rot_deg']:.6f}",
                    ]
                )


def print_reprojection_report(samples, fisheye_reproj, undist_reproj, worst_n):
    print("aruco_reproj_error_px_fisheye_raw:")
    print(summarize_stats("  all_corners", stats(fisheye_reproj), " px"))
    print("aruco_reproj_error_px_undistorted_fit_space:")
    print(summarize_stats("  all_corners", stats(undist_reproj), " px"))
    print("per_frame_aruco_reproj_error_px_fisheye_raw:")
    per_frame = []
    for sample in samples:
        err = sample["fisheye_reproj_err_px"]
        per_frame.append((sample["idx"], float(np.mean(err)), float(np.max(err))))
        print(f"  sample={sample['idx']:03d} mean={np.mean(err):.4f} px max={np.max(err):.4f} px")
    print("worst_aruco_reproj_samples_by_frame_max:")
    for idx, mean_px, max_px in sorted(per_frame, key=lambda x: x[2], reverse=True)[:worst_n]:
        print(f"  sample={idx:03d} mean={mean_px:.4f} px max={max_px:.4f} px")
    return {
        "fisheye_raw_all_corners_px": stats(fisheye_reproj),
        "undistorted_fit_space_all_corners_px": stats(undist_reproj),
        "per_frame_fisheye_raw_px": [
            {"sample": idx, "mean": mean_px, "max": max_px}
            for idx, mean_px, max_px in per_frame
        ],
    }


def print_stability_report(samples):
    if len(samples) < 2:
        return

    tcp0 = pose_to_mat(samples[0]["tcp_pose"])
    tag0 = samples[0]["tx_camera_world_obs"]
    tcp_trans = []
    tcp_rot = []
    tag_trans = []
    tag_rot = []
    for sample in samples:
        tcp = pose_to_mat(sample["tcp_pose"])
        tag = sample["tx_camera_world_obs"]
        dtcp = np.linalg.inv(tcp0) @ tcp
        dtag = np.linalg.inv(tag0) @ tag
        tcp_trans.append(np.linalg.norm(dtcp[:3, 3]) * 1000.0)
        tcp_rot.append(np.rad2deg(R.from_matrix(dtcp[:3, :3]).magnitude()))
        tag_trans.append(np.linalg.norm(dtag[:3, 3]) * 1000.0)
        tag_rot.append(np.rad2deg(R.from_matrix(dtag[:3, :3]).magnitude()))

    print("stability_against_first_sample:")
    print(summarize_stats("  tcp_pose_delta_translation", stats(tcp_trans), " mm"))
    print(summarize_stats("  tcp_pose_delta_rotation", stats(tcp_rot), " deg"))
    print(summarize_stats("  aruco_tvec_delta_translation", stats(tag_trans), " mm"))
    print(summarize_stats("  aruco_rvec_delta_rotation", stats(tag_rot), " deg"))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default=DEFAULT_PKL, help="hand_eye_calib.pkl")
    parser.add_argument(
        "--json",
        "--json_file",
        dest="json_file",
        default=DEFAULT_JSON,
        help="robot_world_hand_eye JSON",
    )
    parser.add_argument("--intr_json", default=DEFAULT_INTR, help="fisheye intrinsics JSON")
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO, help="ArUco YAML config")
    parser.add_argument("--tag_id", type=int, default=12, help="ArUco tag id")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <pkl_dir>/debug_hand_eye_conventions",
    )
    parser.add_argument(
        "--overlay_top_k",
        type=int,
        default=10,
        help="Save worst K overlays per convention by translation error. <=0 saves all.",
    )
    parser.add_argument("--worst_n", type=int, default=5, help="Number of worst samples to print")
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Do not apply the same predefined mask used by calibration.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.pkl = Path(args.pkl).expanduser().resolve()
    args.json_file = Path(args.json_file).expanduser().resolve()
    args.intr_json = Path(args.intr_json).expanduser().resolve()
    args.aruco_yaml = Path(args.aruco_yaml).expanduser().resolve()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else args.pkl.parent / "debug_hand_eye_conventions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.intr_json, "r") as f:
        intr = parse_fisheye_intrinsics(json.load(f))
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))
    with open(args.json_file, "r") as f:
        hand_eye = json.load(f)

    raw_tx_base2world = np.asarray(hand_eye["tx_base2world"], dtype=np.float64)
    raw_tx_gripper2camera = np.asarray(hand_eye["tx_gripper2camera"], dtype=np.float64)

    raw_samples, samples, fisheye_reproj, undist_reproj = load_samples(
        args.pkl,
        intr,
        aruco_config,
        args.tag_id,
        mask=not args.no_mask,
    )
    args.samples_total = len(raw_samples)
    marker_size = aruco_config["marker_size_map"][args.tag_id]

    print(f"pkl: {args.pkl}")
    print(f"hand_eye_json: {args.json_file}")
    print(f"intrinsics: {args.intr_json}")
    print(f"aruco_yaml: {args.aruco_yaml}")
    print(f"image_resolution: {tuple(int(v) for v in intr['DIM'])}")
    print(f"tag_id: {args.tag_id}")
    print(f"tag_size_m: {marker_size}")
    print(f"samples_total: {len(raw_samples)}")
    print(f"samples_used_with_tag_{args.tag_id}: {len(samples)}")
    print()

    reproj_summary = print_reprojection_report(
        samples,
        fisheye_reproj,
        undist_reproj,
        worst_n=args.worst_n,
    )
    print()
    print_stability_report(samples)
    print()

    combos = []
    for tcp_pose_as in ("T_base_tcp", "T_tcp_base"):
        for tx_gripper2camera_as in ("T_camera_tcp", "T_tcp_camera"):
            for tx_base2world_as in ("T_world_base", "T_base_world"):
                combos.append((tcp_pose_as, tx_gripper2camera_as, tx_base2world_as))

    results = []
    print("convention_results:")
    for combo_id, combo in enumerate(combos):
        result = evaluate_combo(samples, raw_tx_base2world, raw_tx_gripper2camera, combo)
        result["combo_id"] = combo_id
        results.append(result)
        label = (
            f"combo={combo_id} tcp_pose_as={result['tcp_pose_as']} "
            f"tx_gripper2camera_as={result['tx_gripper2camera_as']} "
            f"tx_base2world_as={result['tx_base2world_as']}"
        )
        print(label)
        print("  " + summarize_stats("translation", result["translation_mm"], " mm"))
        print("  " + summarize_stats("rotation", result["rotation_deg"], " deg"))
        print("  worst_samples_by_translation:")
        for err in sorted(result["sample_errors"], key=lambda x: x["trans_mm"], reverse=True)[: args.worst_n]:
            print(f"    sample={err['idx']:03d} trans={err['trans_mm']:.3f} mm rot={err['rot_deg']:.3f} deg")
        print()

        selected = sorted(
            result["sample_errors"], key=lambda x: x["trans_mm"], reverse=True
        )
        if args.overlay_top_k > 0:
            selected = selected[: args.overlay_top_k]
        combo_dir = output_dir / "overlays" / f"combo_{combo_id:02d}"
        by_idx = {sample["idx"]: sample for sample in samples}
        for err in selected:
            sample = by_idx[err["idx"]]
            overlay_name = (
                f"sample_{err['idx']:03d}_trans_{err['trans_mm']:.1f}mm_"
                f"rot_{err['rot_deg']:.1f}deg.png"
            )
            save_overlay(combo_dir / overlay_name, sample, label, err, marker_size, sample["intr"])

    best = min(results, key=lambda x: x["translation_mm"]["median"])
    print("best_convention_by_median_translation:")
    print(
        f"  combo={best['combo_id']} tcp_pose_as={best['tcp_pose_as']} "
        f"tx_gripper2camera_as={best['tx_gripper2camera_as']} "
        f"tx_base2world_as={best['tx_base2world_as']}"
    )
    print("  " + summarize_stats("translation", best["translation_mm"], " mm"))
    print("  " + summarize_stats("rotation", best["rotation_deg"], " deg"))
    print()

    write_summary_files(output_dir, args, results, reproj_summary, samples)
    print(f"output_dir: {output_dir}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(f"convention_tsv: {output_dir / 'convention_results.tsv'}")
    print(f"per_sample_tsv: {output_dir / 'per_sample_errors.tsv'}")
    print(f"overlays_dir: {output_dir / 'overlays'}")


if __name__ == "__main__":
    main()
