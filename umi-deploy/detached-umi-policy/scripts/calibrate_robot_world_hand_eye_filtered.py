#!/usr/bin/env python3
"""
Filtered hand-eye calibration using ArUco PnP distance/reprojection diagnostics.

This is a wrapper for single-variant calibration. It preserves the existing
OpenCV robot-world/hand-eye matrix convention and only filters samples.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR))
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from calibrate_compare_variants import (
    filter_detections,
    load_detections,
    save_hand_eye_json,
    transform_from_rvec_tvec,
)
from umi.common.cv_util import parse_aruco_config, parse_fisheye_intrinsics
from umi.common.pose_util import mat_to_pose, pose_to_mat


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTR = REPO_ROOT / "umi-deploy/data_local/calibration/cam0_sensor_intrinsics.json"
DEFAULT_ARUCO = REPO_ROOT / "umi-deploy/data_local/hand_eye_tags/aruco_config_tag12_147mm.yaml"


def pose6_to_mat(pose, rotation_repr):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    if rotation_repr == "rotvec":
        return pose_to_mat(pose)
    if rotation_repr == "euler_xyz":
        tx = np.eye(4, dtype=np.float64)
        tx[:3, 3] = pose[:3]
        tx[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
        return tx
    raise ValueError(f"Unsupported rotation_repr: {rotation_repr}")


def calibrate_filtered(detections, pose_key, pose_rotation_repr):
    if len(detections) < 3:
        raise ValueError(f"Need at least 3 used samples, got {len(detections)}")
    r_world2cam = []
    t_world2cam = []
    r_base2gripper = []
    t_base2gripper = []
    for det in detections:
        sample = det["sample"]
        if pose_key not in sample:
            raise KeyError(f"Sample {det['idx']} is missing pose_key={pose_key!r}")
        tx_camera_world = transform_from_rvec_tvec(det["rvec"], det["tvec"])
        tx_base_gripper = pose6_to_mat(sample[pose_key], pose_rotation_repr)
        tx_gripper_base = np.linalg.inv(tx_base_gripper)
        tv_gripper_base = mat_to_pose(tx_gripper_base)
        r_world2cam.append(R.from_matrix(tx_camera_world[:3, :3]).as_rotvec())
        t_world2cam.append(tx_camera_world[:3, 3])
        r_base2gripper.append(tv_gripper_base[3:])
        t_base2gripper.append(tv_gripper_base[:3])

    r_b2w, t_b2w, r_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_world2cam=r_world2cam,
        t_world2cam=t_world2cam,
        R_base2gripper=r_base2gripper,
        t_base2gripper=t_base2gripper,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    tx_world_base = np.eye(4, dtype=np.float64)
    tx_world_base[:3, :3] = r_b2w
    tx_world_base[:3, 3] = np.asarray(t_b2w).reshape(3)
    tx_camera_gripper = np.eye(4, dtype=np.float64)
    tx_camera_gripper[:3, :3] = r_g2c
    tx_camera_gripper[:3, 3] = np.asarray(t_g2c).reshape(3)
    return tx_world_base, tx_camera_gripper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--intr_json", default=str(DEFAULT_INTR))
    parser.add_argument("--aruco_yaml", default=str(DEFAULT_ARUCO))
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--min_pnp_distance_mm", type=float, default=None)
    parser.add_argument("--max_pnp_distance_mm", type=float, default=None)
    parser.add_argument("--max_reproj_px", type=float, default=4.0)
    parser.add_argument("--pose_key", default="tcp_pose")
    parser.add_argument("--pose_rotation_repr", choices=["rotvec", "euler_xyz"], default="rotvec")
    args = parser.parse_args()

    samples = pickle.load(open(args.input, "rb"))
    raw_intr = parse_fisheye_intrinsics(json.load(open(args.intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(args.aruco_yaml, "r")))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"aruco_yaml={args.aruco_yaml}")

    samples_with_saved_aruco = sum(
        1
        for sample in samples
        if "aruco" in sample
        and sample["aruco"] is not None
        and sample["aruco"].get("detected", True)
        and int(sample["aruco"].get("tag_id", args.tag_id)) == args.tag_id
        and all(key in sample["aruco"] for key in ("rvec", "tvec", "corners"))
    )
    if samples_with_saved_aruco < len(samples):
        print(
            "WARNING: some samples do not have sample['aruco']; "
            "missing tags will be redetected with the provided intr_json and 147mm aruco_yaml."
        )

    detections = load_detections(samples, "raw", raw_intr, aruco_config, args.tag_id, marker_size_m)
    used, excluded = filter_detections(
        detections,
        args.min_pnp_distance_mm,
        args.max_pnp_distance_mm,
        args.max_reproj_px,
    )
    excluded_by_distance = [item for item in excluded if "distance" in item["reason"]]
    excluded_by_reprojection = [item for item in excluded if "reprojection" in item["reason"]]

    print(f"samples_total: {len(samples)}")
    print(f"samples_with_tag: {len(detections)}")
    print(f"samples_used: {len(used)}")
    print(f"excluded_by_distance: {len(excluded_by_distance)}")
    print(f"excluded_by_reprojection: {len(excluded_by_reprojection)}")
    print("used_samples:")
    for det in used:
        print(
            f"  sample={det['idx']:03d} z_mm={det['z_mm']:.3f} "
            f"norm_mm={det['norm_mm']:.3f} reproj={det['reprojection_mean_px']:.4f}"
        )
    print("excluded_samples:")
    for det in excluded:
        print(
            f"  sample={det['idx']:03d} reason={det['reason']} z_mm={det['z_mm']:.3f} "
            f"norm_mm={det['norm_mm']:.3f} reproj={det['reprojection_mean_px']:.4f}"
        )

    tx_world_base, tx_camera_gripper = calibrate_filtered(used, args.pose_key, args.pose_rotation_repr)
    variant = {
        "name": "filtered_single",
        "detection_mode": "raw",
        "distance_filter": f"{args.min_pnp_distance_mm}_{args.max_pnp_distance_mm}",
        "pose_source": args.pose_key,
        "runtime_safe": args.pose_key == "tcp_pose",
    }
    save_hand_eye_json(Path(args.output).expanduser().resolve(), variant, tx_world_base, tx_camera_gripper)
    payload = json.load(open(args.output, "r"))
    payload["pose_key"] = args.pose_key
    payload["pose_rotation_repr"] = args.pose_rotation_repr
    payload["filter_summary"] = {
        "samples_total": len(samples),
        "samples_with_tag": len(detections),
        "samples_used": len(used),
        "excluded_by_distance": len(excluded_by_distance),
        "excluded_by_reprojection": len(excluded_by_reprojection),
    }
    json.dump(payload, open(args.output, "w"), indent=2)

    print("matrix_semantics:")
    print("  tx_base2world key stores T_world_base (^w T_b), matching the existing legacy json key.")
    print("  tx_gripper2camera key stores T_camera_gripper (^c T_g).")
    print("  closed_loop: T_camera_world @ T_world_base == T_camera_gripper @ T_gripper_base")
    print(f"wrote: {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
