#!/usr/bin/env python3
"""
Record robot-state-only static jitter from Arx5Client.

Do not touch the robot while this script is running. It does not open any
camera; it only samples robot state over ZMQ and writes a CSV plus summary.json.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np

from modules.arx5_zmq_client import Arx5Client, rotvec2rotm, rpy2rotm


DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "..", "data_local", "hand_eye_recalib_640", "robot_state_static_jitter")
)


def rotation_error_deg(rotm_a, rotm_b):
    rel = rotm_a.T @ rotm_b
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    return float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "range": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


def write_csv(path, records):
    fieldnames = [
        "idx",
        "host_time_before",
        "host_time_after",
        "host_timestamp",
        "robot_timestamp",
        "get_state_duration_ms",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_rx",
        "tcp_ry",
        "tcp_rz",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_roll",
        "ee_pitch",
        "ee_yaw",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records):
            row = {
                "idx": idx,
                "host_time_before": record["host_time_before"],
                "host_time_after": record["host_time_after"],
                "host_timestamp": record["host_timestamp"],
                "robot_timestamp": record["robot_timestamp"],
                "get_state_duration_ms": record["get_state_duration_ms"],
            }
            row.update({f"j{i + 1}": float(record["joint_state"][i]) for i in range(6)})
            row.update({name: float(value) for name, value in zip(["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"], record["tcp_pose"])})
            row.update({name: float(value) for name, value in zip(["ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw"], record["ee_pose"])})
            writer.writerow(row)


def summarize_records(records):
    if not records:
        return {"sample_count": 0}

    joints = np.asarray([record["joint_state"] for record in records], dtype=np.float64)
    tcp = np.asarray([record["tcp_pose"] for record in records], dtype=np.float64)
    ee = np.asarray([record["ee_pose"] for record in records], dtype=np.float64)
    host_times = np.asarray([record["host_timestamp"] for record in records], dtype=np.float64)
    robot_times = np.asarray([record["robot_timestamp"] for record in records], dtype=np.float64)
    get_state_ms = np.asarray([record["get_state_duration_ms"] for record in records], dtype=np.float64)

    joint_summary = {}
    for idx in range(joints.shape[1]):
        joint = joints[:, idx]
        joint_summary[f"j{idx + 1}"] = {
            "mean_rad": float(np.mean(joint)),
            "std_rad": float(np.std(joint)),
            "range_rad": float(np.max(joint) - np.min(joint)),
            "max_delta_from_first_rad": float(np.max(np.abs(joint - joint[0]))),
            "mean_deg": float(np.rad2deg(np.mean(joint))),
            "std_deg": float(np.rad2deg(np.std(joint))),
            "range_deg": float(np.rad2deg(np.max(joint) - np.min(joint))),
            "max_delta_from_first_deg": float(np.rad2deg(np.max(np.abs(joint - joint[0])))),
        }

    tcp_first_pos = tcp[0, :3]
    tcp_first_rot = rotvec2rotm(tcp[0, 3:])
    tcp_pos_delta_mm = np.linalg.norm(tcp[:, :3] - tcp_first_pos, axis=1) * 1000.0
    tcp_rot_delta_deg = np.asarray([rotation_error_deg(tcp_first_rot, rotvec2rotm(pose[3:])) for pose in tcp])

    ee_first_pos = ee[0, :3]
    ee_first_rot = rpy2rotm(ee[0, 3:])
    ee_pos_delta_mm = np.linalg.norm(ee[:, :3] - ee_first_pos, axis=1) * 1000.0
    ee_rot_delta_deg = np.asarray([rotation_error_deg(ee_first_rot, rpy2rotm(pose[3:])) for pose in ee])

    host_intervals = np.diff(host_times)
    robot_intervals = np.diff(robot_times)

    return {
        "sample_count": int(len(records)),
        "duration_host_sec": float(host_times[-1] - host_times[0]) if len(records) > 1 else 0.0,
        "joint_state": joint_summary,
        "tcp_pose": {
            "position_delta_from_first_mm": summarize_array(tcp_pos_delta_mm),
            "rotation_delta_from_first_deg": summarize_array(tcp_rot_delta_deg),
            "max_position_delta_from_first_mm": float(np.max(tcp_pos_delta_mm)),
            "max_rotation_delta_from_first_deg": float(np.max(tcp_rot_delta_deg)),
        },
        "ee_pose": {
            "position_delta_from_first_mm": summarize_array(ee_pos_delta_mm),
            "rotation_delta_from_first_deg": summarize_array(ee_rot_delta_deg),
            "max_position_delta_from_first_mm": float(np.max(ee_pos_delta_mm)),
            "max_rotation_delta_from_first_deg": float(np.max(ee_rot_delta_deg)),
        },
        "timestamp_intervals": {
            "host_sec": summarize_array(host_intervals),
            "robot_sec": summarize_array(robot_intervals),
            "get_state_duration_ms": summarize_array(get_state_ms),
        },
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_ip", default="127.0.0.1")
    parser.add_argument("--robot_port", type=int, default=8765)
    parser.add_argument("--duration_sec", type=float, default=15.0)
    parser.add_argument("--frequency_hz", type=float, default=100.0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hold_current_pose", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.duration_sec <= 0:
        raise ValueError("--duration_sec must be > 0")
    if args.frequency_hz <= 0:
        raise ValueError("--frequency_hz must be > 0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robot_state_static_jitter.csv"
    summary_path = output_dir / "summary.json"

    print(f"[ROBOT_JITTER] robot=tcp://{args.robot_ip}:{args.robot_port}")
    print(f"[ROBOT_JITTER] duration={args.duration_sec:.3f}s frequency={args.frequency_hz:.3f}Hz")
    print("[ROBOT_JITTER] Do not touch the robot until recording finishes.")

    robot = Arx5Client(args.robot_ip, args.robot_port)
    if args.hold_current_pose:
        print("[ROBOT_JITTER] hold_current_pose before sampling")
        robot.hold_current_pose()
    time.sleep(0.2)

    dt = 1.0 / args.frequency_hz
    deadline = time.monotonic() + args.duration_sec
    next_sample = time.monotonic()
    records = []
    idx = 0
    while time.monotonic() < deadline:
        host_time_before = time.time()
        robot.get_state()
        host_time_after = time.time()
        records.append(
            {
                "host_time_before": float(host_time_before),
                "host_time_after": float(host_time_after),
                "host_timestamp": float(host_time_after),
                "robot_timestamp": float(robot.timestamp),
                "get_state_duration_ms": float((host_time_after - host_time_before) * 1000.0),
                "joint_state": np.asarray(robot.joint_pos, dtype=np.float64).copy(),
                "tcp_pose": np.asarray(robot.tcp_pose, dtype=np.float64).copy(),
                "ee_pose": np.asarray(robot.ee_pose, dtype=np.float64).copy(),
            }
        )
        idx += 1
        next_sample += dt
        sleep_time = next_sample - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

    write_csv(csv_path, records)
    summary = summarize_records(records)
    summary.update(
        {
            "robot": f"tcp://{args.robot_ip}:{args.robot_port}",
            "requested_duration_sec": float(args.duration_sec),
            "requested_frequency_hz": float(args.frequency_hz),
            "csv_path": str(csv_path),
        }
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ROBOT_JITTER] wrote csv={csv_path}")
    print(f"[ROBOT_JITTER] wrote summary={summary_path}")
    for name, joint_stats in summary["joint_state"].items():
        print(
            f"[ROBOT_JITTER] {name}: "
            f"mean={joint_stats['mean_deg']:.6f}deg "
            f"std={joint_stats['std_deg']:.6f}deg "
            f"range={joint_stats['range_deg']:.6f}deg "
            f"max_delta_from_first={joint_stats['max_delta_from_first_deg']:.6f}deg"
        )
    print(f"[ROBOT_JITTER] host_interval_sec={summary['timestamp_intervals']['host_sec']}")
    print(f"[ROBOT_JITTER] robot_interval_sec={summary['timestamp_intervals']['robot_sec']}")
    print(f"[ROBOT_JITTER] get_state_duration_ms={summary['timestamp_intervals']['get_state_duration_ms']}")
    print(
        "[ROBOT_JITTER] max deltas: "
        f"tcp_pos={summary['tcp_pose']['max_position_delta_from_first_mm']:.6f}mm "
        f"tcp_rot={summary['tcp_pose']['max_rotation_delta_from_first_deg']:.6f}deg "
        f"ee_pos={summary['ee_pose']['max_position_delta_from_first_mm']:.6f}mm "
        f"ee_rot={summary['ee_pose']['max_rotation_delta_from_first_deg']:.6f}deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
