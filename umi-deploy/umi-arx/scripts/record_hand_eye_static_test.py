#!/usr/bin/env python3
"""
Record a static hand-eye consistency test with direct OpenCV/V4L2 capture.

Keep the robot and tag still. The script saves repeated camera/state samples
and prints pose, ArUco, and frame-vs-robot timing stability.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from modules.arx5_zmq_client import Arx5Client
from utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0"
)
DEFAULT_INTRINSICS = os.path.abspath(
    os.path.join(ROOT_DIR, "..", "data_local", "calibration", "cam0_sensor_intrinsics.json")
)
DEFAULT_ARUCO_YAML = os.path.abspath(
    os.path.join(ROOT_DIR, "..", "data_local", "hand_eye_tags", "aruco_config_tag12_147mm.yaml")
)
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(ROOT_DIR, "..", "data_local", "hand_eye_recalib_640", "static_consistency_test.pkl")
)


def load_resolution(path):
    with open(path, "r") as f:
        payload = json.load(f)
    if payload.get("intrinsic_type") != "FISHEYE":
        raise ValueError(f"Expected FISHEYE intrinsics, got {payload.get('intrinsic_type')}")
    return int(payload["image_width"]), int(payload["image_height"])


def load_intr(path):
    with open(path, "r") as f:
        return parse_fisheye_intrinsics(json.load(f))


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


def detect_aruco(img_rgb, raw_intr, aruco_config, tag_id):
    intr = convert_fisheye_intrinsics_resolution(raw_intr, img_rgb.shape[:2][::-1])
    tag_dict = detect_localize_aruco_tags(
        img_rgb,
        aruco_config["aruco_dict"],
        aruco_config["marker_size_map"],
        intr,
    )
    if tag_id not in tag_dict:
        return {"tag_id": int(tag_id), "detected": False}
    tag = tag_dict[tag_id]
    marker_size_m = float(aruco_config["marker_size_map"][tag_id])
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    projected, _ = cv2.fisheye.projectPoints(
        marker_object_points(marker_size_m),
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


def open_capture(args, resolution):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    requested_fourcc = str(args.capture_fourcc).strip()
    if requested_fourcc.lower() not in ("", "auto", "none", "skip"):
        if len(requested_fourcc) != 4:
            raise ValueError("--capture_fourcc must be 4 chars, or auto/none to skip forcing FourCC")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))
    else:
        print("[STATIC] capture_fourcc=auto; not forcing V4L2 FourCC")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.cap_buffer_size)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")
    for _ in range(args.camera_warmup_frames):
        cap.read()
        time.sleep(0.02)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[STATIC] opencv_capture=CAP_V4L2 fourcc={fourcc_str} resolution={width}x{height} fps={fps:.3f}")
    if (width, height) != tuple(resolution):
        raise RuntimeError(f"Camera negotiated {width}x{height}, expected {resolution[0]}x{resolution[1]}")
    return cap


def pose_rot_error_deg_rotvec(a, b):
    ra = R.from_rotvec(np.asarray(a, dtype=np.float64).reshape(3))
    rb = R.from_rotvec(np.asarray(b, dtype=np.float64).reshape(3))
    return float(np.rad2deg((ra.inv() * rb).magnitude()))


def pose_rot_error_deg_euler(a, b):
    ra = R.from_euler("xyz", np.asarray(a, dtype=np.float64).reshape(3))
    rb = R.from_euler("xyz", np.asarray(b, dtype=np.float64).reshape(3))
    return float(np.rad2deg((ra.inv() * rb).magnitude()))


def summarize(name, values, unit):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        print(f"{name}: no data")
        return None
    print(
        f"{name}: mean={values.mean():.6g}{unit} median={np.median(values):.6g}{unit} "
        f"max={values.max():.6g}{unit}"
    )
    return float(values.max())


def summarize_static(records):
    if len(records) < 2:
        print("[STATIC] not enough samples for consistency summary")
        return False
    first = records[0]
    tcp_pos = []
    tcp_rot = []
    ee_pos = []
    ee_rot = []
    joint_deg = []
    aruco_t = []
    aruco_r = []
    reproj = []
    deltas = []
    for record in records:
        tcp_pos.append(np.linalg.norm(np.asarray(record["tcp_pose"][:3]) - np.asarray(first["tcp_pose"][:3])) * 1000.0)
        tcp_rot.append(pose_rot_error_deg_rotvec(first["tcp_pose"][3:], record["tcp_pose"][3:]))
        ee_pos.append(np.linalg.norm(np.asarray(record["ee_pose"][:3]) - np.asarray(first["ee_pose"][:3])) * 1000.0)
        ee_rot.append(pose_rot_error_deg_euler(first["ee_pose"][3:], record["ee_pose"][3:]))
        joint_deg.append(np.linalg.norm(np.asarray(record["joint_state"]) - np.asarray(first["joint_state"])) * 180.0 / np.pi)
        deltas.append(record["frame_robot_delta_ms"])
        aruco = record.get("aruco", {})
        first_aruco = first.get("aruco", {})
        if aruco.get("detected") and first_aruco.get("detected"):
            aruco_t.append(np.linalg.norm(np.asarray(aruco["tvec"]) - np.asarray(first_aruco["tvec"])) * 1000.0)
            aruco_r.append(pose_rot_error_deg_rotvec(first_aruco["rvec"], aruco["rvec"]))
            reproj.append(aruco["reprojection_mean_px"])
    tcp_pos_max = summarize("tcp position delta from first", tcp_pos, " mm")
    tcp_rot_max = summarize("tcp rotation delta from first", tcp_rot, " deg")
    summarize("ee position delta from first", ee_pos, " mm")
    summarize("ee rotation delta from first", ee_rot, " deg")
    joint_max = summarize("joint_state delta from first", joint_deg, " deg")
    aruco_t_max = summarize("aruco tvec delta from first", aruco_t, " mm")
    aruco_r_max = summarize("aruco rvec delta from first", aruco_r, " deg")
    reproj_max = summarize("reproj", reproj, " px")
    delta_max = summarize("frame_robot_delta_ms", deltas, " ms")

    fail_reasons = []
    if tcp_pos_max is not None and tcp_pos_max > 2.0:
        fail_reasons.append("tcp_pose position unstable")
    if tcp_rot_max is not None and tcp_rot_max > 0.5:
        fail_reasons.append("tcp_pose rotation unstable")
    if joint_max is not None and joint_max > 0.5:
        fail_reasons.append("joint_state unstable")
    if aruco_t_max is None:
        fail_reasons.append("tag not detected in enough static samples")
    elif aruco_t_max > 5.0:
        fail_reasons.append("aruco tvec unstable")
    if aruco_r_max is not None and aruco_r_max > 2.0:
        fail_reasons.append("aruco rvec unstable")
    if reproj_max is not None and reproj_max > 4.0:
        fail_reasons.append("aruco reprojection too high")
    if delta_max is not None and delta_max > 100.0:
        fail_reasons.append("frame_robot_delta_ms too large")

    if fail_reasons:
        print(f"STATIC_FAIL reasons={fail_reasons}")
        return False
    print("STATIC_PASS")
    return True


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--gripper_fisheye_intrinsics", default=DEFAULT_INTRINSICS)
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO_YAML)
    parser.add_argument("--tag_id", type=int, default=12)
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
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    resolution = load_resolution(args.gripper_fisheye_intrinsics)
    raw_intr = load_intr(args.gripper_fisheye_intrinsics)
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))

    print(f"[STATIC] camera={args.camera}")
    print(f"[STATIC] resolution={resolution[0]}x{resolution[1]} fps={args.fps}")
    print(f"[STATIC] robot=tcp://{args.robot_ip}:{args.robot_port}")
    print("[STATIC] Keep robot and tag physically fixed until recording finishes.")

    cap = open_capture(args, resolution)
    robot = Arx5Client(args.robot_ip, args.robot_port)
    time.sleep(0.5)
    records = []
    try:
        for idx in range(args.samples):
            frame_time_before = time.time()
            ret, frame_bgr = cap.read()
            frame_time_after = time.time()
            if not ret or frame_bgr is None:
                print(f"[STATIC] sample={idx:03d} camera read failed; skipping")
                time.sleep(args.interval)
                continue
            frame_bgr = np.ascontiguousarray(frame_bgr)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            robot_time_before = time.time()
            robot.get_state()
            robot_time_after = time.time()
            frame_robot_delta_ms = (robot_time_after - frame_time_after) * 1000.0
            aruco = detect_aruco(img_rgb, raw_intr, aruco_config, args.tag_id)
            record = {
                "img": img_rgb,
                "tcp_pose": np.asarray(robot.tcp_pose, dtype=np.float64).copy(),
                "ee_pose": np.asarray(robot.ee_pose, dtype=np.float64).copy(),
                "joint_state": np.asarray(robot.joint_pos, dtype=np.float64).copy(),
                "timestamp": float(frame_time_after),
                "camera_host_timestamp": float(frame_time_after),
                "fresh_frame_host_time_before": float(frame_time_before),
                "fresh_frame_host_time_after": float(frame_time_after),
                "robot_host_time_before": float(robot_time_before),
                "robot_host_time_after": float(robot_time_after),
                "robot_host_timestamp": float(robot_time_after),
                "frame_robot_delta_ms": float(frame_robot_delta_ms),
                "time_delta_ms": float(frame_robot_delta_ms),
                "robot_timestamp": float(robot.timestamp),
                "aruco": aruco,
            }
            records.append(record)
            print(
                f"[STATIC] sample={idx:03d} aruco_detected={aruco.get('detected')} "
                f"z_mm={aruco.get('z_mm')} norm_mm={aruco.get('norm_mm')} "
                f"reproj={aruco.get('reprojection_mean_px')} "
                f"frame_robot_delta_ms={frame_robot_delta_ms:.3f}"
            )
            with open(output, "wb") as f:
                pickle.dump(records, f)
            if idx + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        cap.release()

    with open(output, "wb") as f:
        pickle.dump(records, f)
    print(f"[STATIC] wrote {len(records)} samples to {output}")
    summarize_static(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
