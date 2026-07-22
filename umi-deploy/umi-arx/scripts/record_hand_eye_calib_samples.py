#!/usr/bin/env python3
"""
Record ARX5 hand-eye calibration samples with both ARX pose conventions.

Each saved sample is:

  {
    "img": RGB uint8 HxWx3,
    "tcp_pose": rotvec_pose6,
    "ee_pose": euler_xyz_pose6,
    "joint_state": joint6,
    "timestamp": camera_host_timestamp,
    "robot_timestamp": robot_timestamp,
    "robot_host_timestamp": robot_host_timestamp,
    "frame_robot_delta_ms": robot_host_timestamp - camera_host_timestamp,
    "aruco": detected tag diagnostics
  }

Use this for tcp-vs-ee hand-eye A/B tests. Press Space or "s" to save the
current camera/state pair, Backspace to remove the last sample, and "q" to quit.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml

from modules.arx5_zmq_client import Arx5Client
from utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from utils.usb_util import get_sorted_v4l_paths, reset_all_elgato_devices


DEFAULT_INTRINSICS = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "calibration",
        "cam0_sensor_intrinsics.json",
    )
)
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "hand_eye_recalib_640",
        "hand_eye_calib_with_ee_live.pkl",
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
DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0"
)


@dataclass
class DetectionResult:
    aruco: dict | None
    status: str
    status_color: tuple
    warning: str


@dataclass
class CaptureSample:
    frame_bgr: np.ndarray
    img_rgb: np.ndarray
    detection: DetectionResult
    tcp_pose: np.ndarray
    ee_pose: np.ndarray
    joint_state: np.ndarray
    camera_host_timestamp: float
    fresh_frame_host_time_before: float
    fresh_frame_host_time_after: float
    robot_host_time_before: float
    robot_host_time_after: float
    robot_timestamp: float
    frame_robot_delta_ms: float


def load_resolution(path):
    with open(path, "r") as f:
        payload = json.load(f)
    if payload.get("intrinsic_type") != "FISHEYE":
        raise ValueError(f"Expected FISHEYE intrinsics, got {payload.get('intrinsic_type')}")
    return int(payload["image_width"]), int(payload["image_height"])


def load_fisheye_intrinsics(path):
    with open(path, "r") as f:
        return parse_fisheye_intrinsics(json.load(f))


def maybe_detect_aruco(img_rgb, raw_intrinsics, aruco_config, tag_id):
    if raw_intrinsics is None or aruco_config is None:
        return None
    intr = convert_fisheye_intrinsics_resolution(
        raw_intrinsics,
        img_rgb.shape[:2][::-1],
    )
    tag_dict = detect_localize_aruco_tags(
        img_rgb,
        aruco_config["aruco_dict"],
        aruco_config["marker_size_map"],
        intr,
    )
    if tag_id not in tag_dict:
        return None
    tag = tag_dict[tag_id]
    marker_size_m = float(aruco_config["marker_size_map"][tag_id])
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    half = marker_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    projected, _ = cv2.fisheye.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intr["K"],
        intr["D"],
    )
    reproj = np.linalg.norm(projected.reshape(4, 2) - corners, axis=1)
    return {
        "tag_id": int(tag_id),
        "detected": True,
        "corners": corners.copy(),
        "rvec": rvec.copy(),
        "tvec": tvec.copy(),
        "z_mm": float(tvec[2] * 1000.0),
        "norm_mm": float(np.linalg.norm(tvec) * 1000.0),
        "reprojection_error_px": reproj.copy(),
        "reprojection_mean_px": float(reproj.mean()),
        "reprojection_max_px": float(reproj.max()),
    }


def classify_norm(norm_mm):
    if 300.0 <= norm_mm <= 350.0:
        return "preferred", (0, 255, 0), "preferred 300-350mm"
    if 250.0 <= norm_mm < 300.0:
        return "contrast only", (0, 255, 255), "WARNING contrast only / not preferred"
    if norm_mm < 250.0:
        return "too close", (0, 0, 255), "STRONG WARNING too close; avoid for hand-eye"
    return "too far", (0, 0, 255), "STRONG WARNING too far; outside preferred range"


def detect_with_status(img_rgb, raw_intrinsics, aruco_config, tag_id):
    aruco = maybe_detect_aruco(img_rgb, raw_intrinsics, aruco_config, tag_id)
    if aruco is None:
        return DetectionResult(None, "no tag", (0, 0, 255), "WARNING no ArUco tag detected")
    status, color, warning = classify_norm(aruco["norm_mm"])
    return DetectionResult(aruco, status, color, warning)


def rotvec_to_rotm(rotvec):
    rotm, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return rotm


def rotvec_distance_deg(a, b):
    ra = rotvec_to_rotm(a)
    rb = rotvec_to_rotm(b)
    rel = ra.T @ rb
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.rad2deg(angle))


def max_pairwise_position_range_mm(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 2:
        return 0.0
    max_delta = 0.0
    for i in range(arr.shape[0] - 1):
        deltas = np.linalg.norm(arr[i + 1 :] - arr[i], axis=1)
        if deltas.size:
            max_delta = max(max_delta, float(np.max(deltas)))
    return float(max_delta * 1000.0)


def max_pairwise_rotvec_range_deg(rotvecs):
    arr = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] < 2:
        return 0.0
    max_delta = 0.0
    for i in range(arr.shape[0] - 1):
        for j in range(i + 1, arr.shape[0]):
            max_delta = max(max_delta, rotvec_distance_deg(arr[i], arr[j]))
    return float(max_delta)


def aruco_corner_margin_px(aruco, width_height):
    if aruco is None:
        return None
    width, height = width_height
    corners = np.asarray(aruco["corners"], dtype=np.float64).reshape(-1, 2)
    margins = np.stack(
        [
            corners[:, 0],
            (width - 1.0) - corners[:, 0],
            corners[:, 1],
            (height - 1.0) - corners[:, 1],
        ],
        axis=1,
    )
    return float(np.min(margins))


def capture_calib_sample(cap, robot, raw_intrinsics, aruco_config, tag_id):
    fresh_frame_host_time_before = time.time()
    fresh_ret, fresh_frame_bgr = cap.read()
    fresh_frame_host_time_after = time.time()
    if not fresh_ret or fresh_frame_bgr is None:
        return None
    fresh_frame_bgr = np.ascontiguousarray(fresh_frame_bgr)
    fresh_img_rgb = cv2.cvtColor(fresh_frame_bgr, cv2.COLOR_BGR2RGB)
    fresh_detection = detect_with_status(fresh_img_rgb, raw_intrinsics, aruco_config, tag_id)
    robot_host_time_before = time.time()
    robot.get_state()
    robot_host_time_after = time.time()
    frame_robot_delta_ms = (robot_host_time_after - fresh_frame_host_time_after) * 1000.0
    return CaptureSample(
        frame_bgr=fresh_frame_bgr,
        img_rgb=fresh_img_rgb,
        detection=fresh_detection,
        tcp_pose=np.asarray(robot.tcp_pose, dtype=np.float64).copy(),
        ee_pose=np.asarray(robot.ee_pose, dtype=np.float64).copy(),
        joint_state=np.asarray(robot.joint_pos, dtype=np.float64).copy(),
        camera_host_timestamp=float(fresh_frame_host_time_after),
        fresh_frame_host_time_before=float(fresh_frame_host_time_before),
        fresh_frame_host_time_after=float(fresh_frame_host_time_after),
        robot_host_time_before=float(robot_host_time_before),
        robot_host_time_after=float(robot_host_time_after),
        robot_timestamp=float(robot.timestamp),
        frame_robot_delta_ms=float(frame_robot_delta_ms),
    )


def summarize_stability(samples, args, width_height):
    stats = {
        "stability_window_sec": float(args.stability_window_sec),
        "stability_sample_count": int(len(samples)),
        "stability_joint_delta_deg": None,
        "stability_tcp_trans_delta_mm": None,
        "stability_tcp_rot_delta_deg": None,
        "stability_aruco_tvec_delta_mm": None,
        "stability_aruco_rvec_delta_deg": None,
        "stability_frame_robot_delta_ms": None,
        "stability_reproj_mean_px_max": None,
        "stability_norm_mm_min": None,
        "stability_norm_mm_max": None,
        "stability_corner_margin_px_min": None,
        "stability_pass": False,
        "stability_fail_reasons": [],
    }
    reasons = []
    if len(samples) < 2:
        reasons.append("not enough samples in stability window")
        stats["stability_fail_reasons"] = reasons
        return stats

    detections = [sample.detection for sample in samples]
    arucos = [detection.aruco for detection in detections]
    detected = [aruco is not None and aruco.get("detected", True) for aruco in arucos]
    if not all(detected):
        reasons.append("aruco_detected=False within stability window")

    joints = np.asarray([sample.joint_state for sample in samples], dtype=np.float64)
    tcp_poses = np.asarray([sample.tcp_pose for sample in samples], dtype=np.float64)
    frame_robot_deltas = np.asarray([sample.frame_robot_delta_ms for sample in samples], dtype=np.float64)
    stats["stability_joint_delta_deg"] = float(np.max(np.ptp(joints, axis=0)) * 180.0 / np.pi)
    stats["stability_tcp_trans_delta_mm"] = max_pairwise_position_range_mm(tcp_poses[:, :3])
    stats["stability_tcp_rot_delta_deg"] = max_pairwise_rotvec_range_deg(tcp_poses[:, 3:])
    stats["stability_frame_robot_delta_ms"] = float(np.max(frame_robot_deltas))

    if all(detected):
        tvecs = np.asarray([aruco["tvec"] for aruco in arucos], dtype=np.float64)
        rvecs = np.asarray([aruco["rvec"] for aruco in arucos], dtype=np.float64)
        norm_mms = np.asarray([aruco["norm_mm"] for aruco in arucos], dtype=np.float64)
        reprojs = np.asarray([aruco["reprojection_mean_px"] for aruco in arucos], dtype=np.float64)
        margins = np.asarray([aruco_corner_margin_px(aruco, width_height) for aruco in arucos], dtype=np.float64)
        stats["stability_aruco_tvec_delta_mm"] = max_pairwise_position_range_mm(tvecs)
        stats["stability_aruco_rvec_delta_deg"] = max_pairwise_rotvec_range_deg(rvecs)
        stats["stability_reproj_mean_px_max"] = float(np.max(reprojs))
        stats["stability_norm_mm_min"] = float(np.min(norm_mms))
        stats["stability_norm_mm_max"] = float(np.max(norm_mms))
        stats["stability_corner_margin_px_min"] = float(np.min(margins))

        if np.min(norm_mms) < args.min_norm_mm or np.max(norm_mms) > args.max_norm_mm:
            reasons.append(
                f"norm_mm outside [{args.min_norm_mm:.1f}, {args.max_norm_mm:.1f}] "
                f"(range {np.min(norm_mms):.3f}-{np.max(norm_mms):.3f})"
            )
        if np.max(reprojs) >= args.max_reproj_px:
            reasons.append(f"reproj_mean_px >= {args.max_reproj_px:.3f} (max {np.max(reprojs):.4f})")
        if np.min(margins) <= args.corner_margin_px:
            reasons.append(f"corner_margin_px <= {args.corner_margin_px:.3f} (min {np.min(margins):.3f})")
        if stats["stability_aruco_tvec_delta_mm"] >= args.max_aruco_tvec_delta_mm:
            reasons.append(
                f"aruco tvec range >= {args.max_aruco_tvec_delta_mm:.3f}mm "
                f"({stats['stability_aruco_tvec_delta_mm']:.4f}mm)"
            )
        if stats["stability_aruco_rvec_delta_deg"] >= args.max_aruco_rvec_delta_deg:
            reasons.append(
                f"aruco rvec range >= {args.max_aruco_rvec_delta_deg:.3f}deg "
                f"({stats['stability_aruco_rvec_delta_deg']:.4f}deg)"
            )

    if stats["stability_frame_robot_delta_ms"] >= 20.0:
        reasons.append(f"frame_robot_delta_ms >= 20ms (max {stats['stability_frame_robot_delta_ms']:.3f}ms)")
    if stats["stability_joint_delta_deg"] >= args.max_joint_delta_deg:
        reasons.append(
            f"joint_state range >= {args.max_joint_delta_deg:.3f}deg "
            f"({stats['stability_joint_delta_deg']:.4f}deg)"
        )
    if stats["stability_tcp_trans_delta_mm"] >= args.max_tcp_trans_delta_mm:
        reasons.append(
            f"tcp translation range >= {args.max_tcp_trans_delta_mm:.3f}mm "
            f"({stats['stability_tcp_trans_delta_mm']:.4f}mm)"
        )
    if stats["stability_tcp_rot_delta_deg"] >= args.max_tcp_rot_delta_deg:
        reasons.append(
            f"tcp rotation range >= {args.max_tcp_rot_delta_deg:.3f}deg "
            f"({stats['stability_tcp_rot_delta_deg']:.4f}deg)"
        )

    stats["stability_fail_reasons"] = reasons
    stats["stability_pass"] = len(reasons) == 0
    return stats


def print_stability_stats(stats, prefix="[CALIB]"):
    print(
        f"{prefix} stability pass={stats['stability_pass']} "
        f"n={stats['stability_sample_count']} "
        f"joint={stats['stability_joint_delta_deg']}deg "
        f"tcp_trans={stats['stability_tcp_trans_delta_mm']}mm "
        f"tcp_rot={stats['stability_tcp_rot_delta_deg']}deg "
        f"aruco_t={stats['stability_aruco_tvec_delta_mm']}mm "
        f"aruco_r={stats['stability_aruco_rvec_delta_deg']}deg "
        f"frame_robot_delta_max={stats['stability_frame_robot_delta_ms']}ms "
        f"reproj_max={stats['stability_reproj_mean_px_max']} "
        f"norm_range=({stats['stability_norm_mm_min']}, {stats['stability_norm_mm_max']}) "
        f"corner_margin_min={stats['stability_corner_margin_px_min']}"
    )
    for reason in stats["stability_fail_reasons"]:
        print(f"{prefix} stability reject: {reason}")


def collect_stable_sample(cap, robot, raw_intrinsics, aruco_config, args, width_height):
    print("[CALIB] release robot and wait for stability")
    window = collections.deque()
    last_report = 0.0
    while True:
        sample = capture_calib_sample(cap, robot, raw_intrinsics, aruco_config, args.tag_id)
        now = time.time()
        if sample is None:
            print("[CALIB] fresh cap.read failed during stability check")
            time.sleep(0.02)
            continue
        window.append(sample)
        while (
            len(window) > 2
            and (sample.camera_host_timestamp - window[1].camera_host_timestamp) >= args.stability_window_sec
        ):
            window.popleft()
        if (
            len(window) >= 2
            and (window[-1].camera_host_timestamp - window[0].camera_host_timestamp) >= args.stability_window_sec
        ):
            stats = summarize_stability(list(window), args, width_height)
            if stats["stability_pass"]:
                print_stability_stats(stats)
                return window[-1], stats
            if not args.wait_until_stable:
                print_stability_stats(stats)
                return None, stats
            if now - last_report >= 1.0:
                print_stability_stats(stats)
                last_report = now


def build_record_from_sample(sample, stats, args):
    record = {
        "img": sample.img_rgb,
        "tcp_pose": sample.tcp_pose,
        "ee_pose": sample.ee_pose,
        "joint_state": sample.joint_state,
        "timestamp": sample.camera_host_timestamp,
        "camera_host_timestamp": sample.camera_host_timestamp,
        "fresh_frame_host_time_before": sample.fresh_frame_host_time_before,
        "fresh_frame_host_time_after": sample.fresh_frame_host_time_after,
        "robot_host_time_before": sample.robot_host_time_before,
        "robot_host_time_after": sample.robot_host_time_after,
        "robot_host_timestamp": sample.robot_host_time_after,
        "frame_robot_delta_ms": sample.frame_robot_delta_ms,
        "time_delta_ms": sample.frame_robot_delta_ms,
        "robot_timestamp": sample.robot_timestamp,
        "stability_joint_delta_deg": stats["stability_joint_delta_deg"],
        "stability_tcp_trans_delta_mm": stats["stability_tcp_trans_delta_mm"],
        "stability_tcp_rot_delta_deg": stats["stability_tcp_rot_delta_deg"],
        "stability_aruco_tvec_delta_mm": stats["stability_aruco_tvec_delta_mm"],
        "stability_aruco_rvec_delta_deg": stats["stability_aruco_rvec_delta_deg"],
        "stability_pass": bool(stats["stability_pass"]),
        "stability": stats.copy(),
    }
    aruco = sample.detection.aruco
    if aruco is not None:
        record["aruco"] = aruco
    else:
        record["aruco"] = {
            "tag_id": int(args.tag_id),
            "detected": False,
        }
    return record


def draw_fisheye_axes(vis_bgr, aruco, intr, axis_length_m=0.05):
    if aruco is None:
        return
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, -axis_length_m],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    projected, _ = cv2.fisheye.projectPoints(
        axis_points,
        np.asarray(aruco["rvec"], dtype=np.float64).reshape(3, 1),
        np.asarray(aruco["tvec"], dtype=np.float64).reshape(3, 1),
        intr["K"],
        intr["D"],
    )
    pts = np.round(projected.reshape(-1, 2)).astype(int)
    origin = tuple(pts[0])
    cv2.line(vis_bgr, origin, tuple(pts[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(vis_bgr, origin, tuple(pts[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(vis_bgr, origin, tuple(pts[3]), (255, 0, 0), 2, cv2.LINE_AA)


def draw_detection_overlay(frame_bgr, detection, raw_intrinsics, width_height, saved_total, saved_preferred, tag_id):
    vis = frame_bgr.copy()
    aruco = detection.aruco
    if aruco is not None:
        corners = np.asarray(aruco["corners"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
        intr = convert_fisheye_intrinsics_resolution(raw_intrinsics, width_height)
        draw_fisheye_axes(vis, aruco, intr)
        lines = [
            f"tag_id={tag_id}",
            f"z={aruco['z_mm']:.1f} mm",
            f"norm={aruco['norm_mm']:.1f} mm",
            f"reproj={aruco['reprojection_mean_px']:.2f} px",
            f"status={detection.status}",
            f"saved_total={saved_total}",
            f"saved_preferred={saved_preferred}",
        ]
        color = detection.status_color
    else:
        lines = [
            f"NO TAG {tag_id} DETECTED",
            f"saved_total={saved_total}",
            f"saved_preferred={saved_preferred}",
        ]
        color = (0, 0, 255)

    y = 26
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        y += 24
    return vis


def count_preferred(records):
    count = 0
    for record in records:
        aruco = record.get("aruco")
        if (
            aruco is not None
            and aruco.get("detected", True)
            and 300.0 <= aruco.get("norm_mm", -1.0) <= 350.0
            and record.get("stability_pass", True)
        ):
            count += 1
    return count


def save_records(output, records):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(records, f)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Absolute V4L camera device path. Overrides --camera_reorder when set.")
    parser.add_argument("--camera_reorder", default="0")
    parser.add_argument("--gripper_fisheye_intrinsics", default=DEFAULT_INTRINSICS)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--capture_fourcc",
        default="MJPG",
        help="FourCC requested from V4L2, e.g. MJPG or YUYV. Use auto/none to skip forcing FourCC.",
    )
    parser.add_argument("--cap_buffer_size", type=int, default=1)
    parser.add_argument("--camera_warmup_frames", type=int, default=30)
    parser.add_argument("--robot_ip", default="127.0.0.1")
    parser.add_argument("--robot_port", type=int, default=8765)
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO_YAML)
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--skip_aruco_detection", action="store_true")
    parser.add_argument("--allow_save_without_tag", action="store_true")
    parser.add_argument("--strict_preferred", dest="strict_preferred", action="store_true", default=True)
    parser.add_argument("--no_strict_preferred", dest="strict_preferred", action="store_false")
    parser.add_argument("--wait_until_stable", action="store_true", default=True)
    parser.add_argument("--no_wait_until_stable", dest="wait_until_stable", action="store_false")
    parser.add_argument(
        "--hold_current_pose",
        action="store_true",
        help="Deprecated safety no-op unless --enable_hold_current_pose is also passed.",
    )
    parser.add_argument(
        "--enable_hold_current_pose",
        action="store_true",
        help="Actually execute --hold_current_pose. Unsafe for hand-guided collection.",
    )
    parser.add_argument(
        "--hold_on_start",
        action="store_true",
        help="Immediately hold current pose at startup. Do not use for hand-guided collection.",
    )
    parser.add_argument("--stability_window_sec", type=float, default=1.0)
    parser.add_argument("--max_joint_delta_deg", type=float, default=0.05)
    parser.add_argument("--max_tcp_trans_delta_mm", type=float, default=0.5)
    parser.add_argument("--max_tcp_rot_delta_deg", type=float, default=0.1)
    parser.add_argument("--max_aruco_tvec_delta_mm", type=float, default=2.0)
    parser.add_argument("--max_aruco_rvec_delta_deg", type=float, default=0.3)
    parser.add_argument("--max_reproj_px", type=float, default=1.5)
    parser.add_argument("--min_norm_mm", type=float, default=300.0)
    parser.add_argument("--max_norm_mm", type=float, default=350.0)
    parser.add_argument("--corner_margin_px", type=float, default=30.0)
    parser.add_argument("--reset_elgato", action="store_true")
    parser.add_argument("--no_preview", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.stability_window_sec <= 0:
        raise ValueError("--stability_window_sec must be > 0")
    if (args.strict_preferred or args.wait_until_stable) and args.skip_aruco_detection:
        raise ValueError(
            "stability/strict preferred gating requires ArUco detection; "
            "remove --skip_aruco_detection or pass --no_strict_preferred --no_wait_until_stable"
        )
    resolution = load_resolution(args.gripper_fisheye_intrinsics)
    raw_intrinsics = None
    aruco_config = None
    if not args.skip_aruco_detection:
        raw_intrinsics = load_fisheye_intrinsics(args.gripper_fisheye_intrinsics)
        with open(args.aruco_yaml, "r") as f:
            aruco_config = parse_aruco_config(yaml.safe_load(f))
    output = Path(args.output).expanduser().resolve()
    debug_overlay_dir = output.parent / "debug_saved_overlays"
    debug_overlay_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_elgato:
        reset_all_elgato_devices()
        time.sleep(0.1)

    if args.camera:
        dev_video_path = args.camera
    else:
        v4l_paths = get_sorted_v4l_paths()
        camera_indices = [int(token) for token in args.camera_reorder.split(",") if token.strip()]
        if not camera_indices:
            raise ValueError("--camera_reorder must contain at least one index")
        if max(camera_indices) >= len(v4l_paths):
            raise IndexError(
                f"--camera_reorder requested index {max(camera_indices)}, "
                f"but only {len(v4l_paths)} camera device(s) were found: {v4l_paths}"
            )
        dev_video_path = v4l_paths[camera_indices[0]]

    print(f"[CALIB] camera={dev_video_path}")
    print(f"[CALIB] resolution={resolution[0]}x{resolution[1]} fps={args.fps}")
    print(f"[CALIB] robot=tcp://{args.robot_ip}:{args.robot_port}")
    print("[CALIB] Space/s: save sample, Backspace: delete last, q: save and quit.")
    print(
        "[CALIB] Press s to request a save; the script will wait for no-touch stability "
        "before writing the sample."
    )
    print(
        "[CALIB] strict gates: "
        f"norm={args.min_norm_mm:.1f}-{args.max_norm_mm:.1f}mm "
        f"reproj<{args.max_reproj_px:.2f}px corner_margin>{args.corner_margin_px:.1f}px "
        "frame_robot_delta<20ms "
        f"joint<{args.max_joint_delta_deg:.3f}deg "
        f"tcp_trans<{args.max_tcp_trans_delta_mm:.3f}mm "
        f"tcp_rot<{args.max_tcp_rot_delta_deg:.3f}deg "
        f"aruco_t<{args.max_aruco_tvec_delta_mm:.3f}mm "
        f"aruco_r<{args.max_aruco_rvec_delta_deg:.3f}deg"
    )

    records = []
    cap = cv2.VideoCapture(dev_video_path, cv2.CAP_V4L2)
    requested_fourcc = str(args.capture_fourcc).strip()
    if requested_fourcc.lower() not in ("", "auto", "none", "skip"):
        if len(requested_fourcc) != 4:
            raise ValueError("--capture_fourcc must be 4 chars, or auto/none to skip forcing FourCC")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))
    else:
        print("[CALIB] capture_fourcc=auto; not forcing V4L2 FourCC")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.cap_buffer_size)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {dev_video_path}")

    # Warm up the direct OpenCV/V4L2 path and verify the negotiated mode.
    for _ in range(args.camera_warmup_frames):
        cap.read()
        time.sleep(0.02)

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        "[CALIB] opencv_capture="
        f"backend=CAP_V4L2 fourcc={actual_fourcc_str} "
        f"resolution={actual_width}x{actual_height} fps={actual_fps:.3f}"
    )
    if (actual_width, actual_height) != tuple(resolution):
        raise RuntimeError(
            f"Camera negotiated {actual_width}x{actual_height}, expected {resolution[0]}x{resolution[1]}"
        )

    robot = Arx5Client(args.robot_ip, args.robot_port)
    time.sleep(0.5)
    if args.hold_on_start and not args.enable_hold_current_pose:
        raise ValueError("--hold_on_start requires --enable_hold_current_pose because it can lock the robot")
    if args.hold_on_start:
        print("[CALIB] hold_current_pose before collection")
        robot.hold_current_pose()
    last_status_print = 0.0
    try:
        while True:
            ret, frame_bgr = cap.read()
            preview_timestamp = time.time()
            if not ret or frame_bgr is None:
                print("[CALIB] camera read failed, retrying...")
                time.sleep(0.02)
                continue

            # Keep preview on a safe contiguous copy of the raw BGR frame returned by OpenCV.
            frame_bgr = np.ascontiguousarray(frame_bgr)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detection = detect_with_status(img_rgb, raw_intrinsics, aruco_config, args.tag_id)
            saved_preferred = count_preferred(records)
            vis = draw_detection_overlay(
                frame_bgr,
                detection,
                raw_intrinsics,
                tuple(resolution),
                len(records),
                saved_preferred,
                args.tag_id,
            )
            if preview_timestamp - last_status_print >= 0.5:
                if detection.aruco is None:
                    print(f"[CALIB] current tag_id={args.tag_id} NOT DETECTED saved_total={len(records)}")
                else:
                    aruco = detection.aruco
                    print(
                        f"[CALIB] current z_mm={aruco['z_mm']:.3f} "
                        f"norm_mm={aruco['norm_mm']:.3f} "
                        f"reproj={aruco['reprojection_mean_px']:.4f} "
                        f"status={detection.status} saved_total={len(records)} "
                        f"saved_preferred={saved_preferred}"
                    )
                last_status_print = preview_timestamp
            if not args.no_preview:
                cv2.imshow("ARX5 Hand-Eye Calibration", vis)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (8, 127):  # Backspace on common OpenCV backends.
                if records:
                    deleted_idx = len(records) - 1
                    records.pop()
                    save_records(output, records)
                    overlay_path = debug_overlay_dir / f"sample_{deleted_idx:04d}.jpg"
                    if overlay_path.exists():
                        overlay_path.unlink()
                    print(f"[CALIB] deleted sample={deleted_idx:04d}, remaining={len(records)}")
                continue
            if key in (ord(" "), ord("s"), ord("S")):
                if args.hold_current_pose:
                    if args.enable_hold_current_pose:
                        print("[CALIB] hold_current_pose before stability gate")
                        robot.hold_current_pose()
                    else:
                        print(
                            "[CALIB] --hold_current_pose ignored for safety; "
                            "use keyboard/direct-control collection instead"
                        )

                if args.wait_until_stable or args.strict_preferred:
                    stable_sample, stability_stats = collect_stable_sample(
                        cap,
                        robot,
                        raw_intrinsics,
                        aruco_config,
                        args,
                        tuple(resolution),
                    )
                    if stable_sample is None or not stability_stats["stability_pass"]:
                        print("[CALIB] sample not saved: stability/quality gate rejected it")
                        continue
                    sample = stable_sample
                else:
                    sample = capture_calib_sample(cap, robot, raw_intrinsics, aruco_config, args.tag_id)
                    if sample is None:
                        print("[CALIB] fresh cap.read failed; sample not saved")
                        continue
                    stability_stats = {
                        "stability_window_sec": 0.0,
                        "stability_sample_count": 1,
                        "stability_joint_delta_deg": 0.0,
                        "stability_tcp_trans_delta_mm": 0.0,
                        "stability_tcp_rot_delta_deg": 0.0,
                        "stability_aruco_tvec_delta_mm": 0.0,
                        "stability_aruco_rvec_delta_deg": 0.0,
                        "stability_frame_robot_delta_ms": sample.frame_robot_delta_ms,
                        "stability_reproj_mean_px_max": (
                            None
                            if sample.detection.aruco is None
                            else sample.detection.aruco["reprojection_mean_px"]
                        ),
                        "stability_norm_mm_min": None if sample.detection.aruco is None else sample.detection.aruco["norm_mm"],
                        "stability_norm_mm_max": None if sample.detection.aruco is None else sample.detection.aruco["norm_mm"],
                        "stability_corner_margin_px_min": aruco_corner_margin_px(sample.detection.aruco, tuple(resolution)),
                        "stability_pass": True,
                        "stability_fail_reasons": [],
                    }

                fresh_detection = sample.detection
                if fresh_detection.aruco is None and not args.allow_save_without_tag:
                    print(f"[WARN] tag {args.tag_id} not detected, sample not saved")
                    continue
                if args.strict_preferred and not stability_stats["stability_pass"]:
                    print("[CALIB] sample not saved: strict preferred gate failed")
                    continue

                record = build_record_from_sample(sample, stability_stats, args)
                aruco = fresh_detection.aruco
                if aruco is not None:
                    distance_note = fresh_detection.warning
                else:
                    distance_note = "WARNING no ArUco tag detected"
                save_vis = draw_detection_overlay(
                    sample.frame_bgr,
                    fresh_detection,
                    raw_intrinsics,
                    tuple(resolution),
                    len(records) + 1,
                    count_preferred(records + [record]),
                    args.tag_id,
                )
                overlay_path = debug_overlay_dir / f"sample_{len(records):04d}.jpg"
                cv2.imwrite(str(overlay_path), save_vis)
                records.append(record)
                save_records(output, records)
                frame_robot_delta_ms = sample.frame_robot_delta_ms
                if frame_robot_delta_ms > 100.0:
                    sync_warning = "STRONG WARNING frame_robot_delta_ms > 100ms"
                elif frame_robot_delta_ms > 50.0:
                    sync_warning = "WARNING frame_robot_delta_ms > 50ms"
                else:
                    sync_warning = "sync ok"
                print(
                    f"[CALIB] saved sample={len(records)-1:03d} "
                    f"tcp_pose={np.round(sample.tcp_pose, 6).tolist()} "
                    f"ee_pose={np.round(sample.ee_pose, 6).tolist()} "
                    f"joint_state={np.round(sample.joint_state, 6).tolist()} "
                    f"aruco_detected={aruco is not None} "
                    f"z_mm={None if aruco is None else round(aruco['z_mm'], 3)} "
                    f"norm_mm={None if aruco is None else round(aruco['norm_mm'], 3)} "
                    f"reproj={None if aruco is None else round(aruco['reprojection_mean_px'], 4)} "
                    f"status={fresh_detection.status} "
                    f"frame_robot_delta_ms={frame_robot_delta_ms:.3f} "
                    f"stability_joint_delta_deg={stability_stats['stability_joint_delta_deg']} "
                    f"stability_tcp_trans_delta_mm={stability_stats['stability_tcp_trans_delta_mm']} "
                    f"stability_tcp_rot_delta_deg={stability_stats['stability_tcp_rot_delta_deg']} "
                    f"stability_aruco_tvec_delta_mm={stability_stats['stability_aruco_tvec_delta_mm']} "
                    f"stability_aruco_rvec_delta_deg={stability_stats['stability_aruco_rvec_delta_deg']} "
                    f"{sync_warning} {distance_note} "
                    f"debug_overlay={overlay_path}"
                )
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    save_records(output, records)
    print(f"[CALIB] wrote {len(records)} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
