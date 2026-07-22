#!/usr/bin/env python3
"""
Keyboard joint jog and hand-eye sample recording in one process.

This script talks to the ARX5 SDK directly. Stop the ZMQ server before using it;
two processes must not own the CAN interface at the same time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDK_PYTHON_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "arx5-sdk", "python"))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SDK_PYTHON_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml

import arx5_interface as arx5
from modules.arx5_zmq_client import ee2tcp
from record_hand_eye_calib_samples import (
    DEFAULT_ARUCO_YAML,
    DEFAULT_CAMERA,
    DEFAULT_INTRINSICS,
    DEFAULT_OUTPUT,
    build_record_from_sample,
    collect_stable_sample,
    count_preferred,
    detect_with_status,
    draw_detection_overlay,
    load_fisheye_intrinsics,
    load_resolution,
    parse_aruco_config,
    save_records,
)


JOINT_KEYMAP = {
    "q": (0, 1.0),
    "a": (0, -1.0),
    "w": (1, 1.0),
    "s": (1, -1.0),
    "e": (2, 1.0),
    "d": (2, -1.0),
    "r": (3, 1.0),
    "f": (3, -1.0),
    "t": (4, 1.0),
    "g": (4, -1.0),
    "y": (5, 1.0),
    "h": (5, -1.0),
}


class DirectArx5Robot:
    def __init__(self, controller):
        self.controller = controller
        self.latest_state: dict[str, Any] = {}
        self.get_state()

    def get_state(self):
        joint_state = self.controller.get_joint_state()
        eef_state = self.controller.get_eef_state()
        self.latest_state = {
            "timestamp": float(eef_state.timestamp),
            "ee_pose": eef_state.pose_6d().copy(),
            "joint_pos": joint_state.pos().copy(),
            "joint_vel": joint_state.vel().copy(),
            "joint_torque": joint_state.torque().copy(),
            "gripper_pos": float(joint_state.gripper_pos),
            "gripper_vel": float(joint_state.gripper_vel),
            "gripper_torque": float(joint_state.gripper_torque),
        }
        return self.latest_state

    @property
    def timestamp(self):
        return float(self.latest_state["timestamp"])

    @property
    def ee_pose(self):
        return self.latest_state["ee_pose"]

    @property
    def tcp_pose(self):
        return ee2tcp(self.ee_pose)

    @property
    def joint_pos(self):
        return self.latest_state["joint_pos"]

    @property
    def joint_vel(self):
        return self.latest_state["joint_vel"]

    @property
    def joint_torque(self):
        return self.latest_state["joint_torque"]

    @property
    def gripper_pos(self):
        return float(self.latest_state["gripper_pos"])


def make_l5_robot_config(model):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if model == "L5":
        robot_config.urdf_path = os.path.abspath(
            os.path.join(ROOT_DIR, "..", "arx5-sdk", "models", "L5_assembly.urdf")
        )
        robot_config.base_link_name = "ARXR5_arm_only_no_gray_base_base_link_arm"
        robot_config.eef_link_name = "DAS_Controller_V3_with_flange_link_Flange"
    return robot_config


def set_gain(controller, kp, kd, gripper_kp, gripper_kd):
    gain = arx5.Gain(len(kp))
    gain.kp()[:] = kp
    gain.kd()[:] = kd
    gain.gripper_kp = float(gripper_kp)
    gain.gripper_kd = float(gripper_kd)
    controller.set_gain(gain)


def ramp_servo_gain(controller, kp_scale, kd_scale, ramp_sec):
    cfg = controller.get_controller_config()
    current = controller.get_gain()
    start_kp = current.kp().copy()
    start_kd = current.kd().copy()
    start_gripper_kp = float(current.gripper_kp)
    start_gripper_kd = float(current.gripper_kd)
    target_kp = np.asarray(cfg.default_kp, dtype=np.float64) * kp_scale
    target_kd = np.asarray(cfg.default_kd, dtype=np.float64) * kd_scale
    target_gripper_kp = float(cfg.default_gripper_kp) * kp_scale
    target_gripper_kd = float(cfg.default_gripper_kd) * kd_scale
    steps = max(int(ramp_sec / max(float(cfg.controller_dt), 0.002)), 1)
    for i in range(steps + 1):
        alpha = i / steps
        set_gain(
            controller,
            start_kp * (1.0 - alpha) + target_kp * alpha,
            start_kd * (1.0 - alpha) + target_kd * alpha,
            start_gripper_kp * (1.0 - alpha) + target_gripper_kp * alpha,
            start_gripper_kd * (1.0 - alpha) + target_gripper_kd * alpha,
        )
        time.sleep(max(float(cfg.controller_dt), 0.002))


def command_joint(controller, target_joint, gripper_pos, preview_sec):
    cmd = arx5.JointState(
        np.asarray(target_joint, dtype=np.float64).reshape(6),
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        float(gripper_pos),
    )
    cmd.timestamp = controller.get_timestamp() + float(preview_sec)
    controller.set_joint_cmd(cmd)


def ensure_joint_limits(target, robot_config):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    below = target < robot_config.joint_pos_min
    above = target > robot_config.joint_pos_max
    if np.any(below | above):
        print(
            "[KEYREC] blocked by joint limit "
            f"target_deg={np.round(np.rad2deg(target), 3).tolist()} "
            f"min_deg={np.round(np.rad2deg(robot_config.joint_pos_min), 3).tolist()} "
            f"max_deg={np.round(np.rad2deg(robot_config.joint_pos_max), 3).tolist()}"
        )
        return False
    return True


def open_capture(args, resolution):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    requested_fourcc = str(args.capture_fourcc).strip()
    if requested_fourcc.lower() not in ("", "auto", "none", "skip"):
        if len(requested_fourcc) != 4:
            raise ValueError("--capture_fourcc must be 4 chars, or auto/none to skip forcing FourCC")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))
    else:
        print("[KEYREC] capture_fourcc=auto; not forcing V4L2 FourCC")
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
    print(f"[KEYREC] opencv_capture=CAP_V4L2 fourcc={fourcc_str} resolution={width}x{height} fps={fps:.3f}")
    if (width, height) != tuple(resolution):
        raise RuntimeError(f"Camera negotiated {width}x{height}, expected {resolution[0]}x{resolution[1]}")
    return cap


def print_help(args):
    print("[KEYREC] keys: q/a J1, w/s J2, e/d J3, r/f J4, t/g J5, y/h J6")
    print("[KEYREC] Space: save after stability gate, Backspace: delete last, 0: sync target, Esc: damping + quit")
    print(
        f"[KEYREC] joint_step={args.joint_step_deg:.3f}deg "
        f"preview={args.preview_sec:.3f}s kp_scale={args.kp_scale:.3f} "
        f"max_tracking_error={args.max_tracking_error_deg:.3f}deg"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--gripper_fisheye_intrinsics", default=DEFAULT_INTRINSICS)
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO_YAML)
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--capture_fourcc", default="YUYV")
    parser.add_argument("--cap_buffer_size", type=int, default=1)
    parser.add_argument("--camera_warmup_frames", type=int, default=60)
    parser.add_argument("--model", default="L5")
    parser.add_argument("--interface", default="can1")
    parser.add_argument("--joint_step_deg", type=float, default=0.5)
    parser.add_argument("--preview_sec", type=float, default=0.10)
    parser.add_argument("--kp_scale", type=float, default=1.0)
    parser.add_argument("--kd_scale", type=float, default=1.0)
    parser.add_argument("--gain_ramp_sec", type=float, default=1.0)
    parser.add_argument("--gravity_compensation", action="store_true")
    parser.add_argument("--max_tracking_error_deg", type=float, default=2.0)
    parser.add_argument("--strict_preferred", action="store_true", default=True)
    parser.add_argument("--wait_until_stable", action="store_true", default=True)
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
    parser.add_argument("--no_preview", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if "arx5_ros" not in sys.executable:
        raise RuntimeError(
            "Direct-CAN keyboard collection must run with the arx5_ros Python, not umi-arx. "
            "Use /home/yd/anaconda3/envs/arx5_ros/bin/python for this script."
        )
    output = Path(args.output).expanduser().resolve()
    debug_overlay_dir = output.parent / "debug_saved_overlays"
    debug_overlay_dir.mkdir(parents=True, exist_ok=True)

    resolution = load_resolution(args.gripper_fisheye_intrinsics)
    raw_intrinsics = load_fisheye_intrinsics(args.gripper_fisheye_intrinsics)
    with open(args.aruco_yaml, "r") as f:
        aruco_config = parse_aruco_config(yaml.safe_load(f))

    print("[KEYREC] Stop zmq_server.py before running this direct-CAN script.")
    print(f"[KEYREC] output={output}")
    print(f"[KEYREC] camera={args.camera}")
    print(f"[KEYREC] robot={args.model} {args.interface} gravity_compensation={args.gravity_compensation}")
    print_help(args)

    cap = open_capture(args, resolution)
    robot_config = make_l5_robot_config(args.model)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.gravity_compensation = bool(args.gravity_compensation)
    controller = arx5.Arx5JointController(robot_config, controller_config, args.interface)
    robot = DirectArx5Robot(controller)

    joint_state = controller.get_joint_state()
    target_joint = joint_state.pos().copy()
    command_joint(controller, target_joint, joint_state.gripper_pos, args.preview_sec)
    print("[KEYREC] ramping servo gain at current joint target")
    ramp_servo_gain(controller, args.kp_scale, args.kd_scale, args.gain_ramp_sec)

    records = []
    last_status_print = 0.0
    finished_cleanly = False
    try:
        while True:
            ret, frame_bgr = cap.read()
            now = time.time()
            if not ret or frame_bgr is None:
                print("[KEYREC] camera read failed")
                time.sleep(0.02)
                continue

            frame_bgr = np.ascontiguousarray(frame_bgr)
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            robot.get_state()
            detection = detect_with_status(img_rgb, raw_intrinsics, aruco_config, args.tag_id)
            vis = draw_detection_overlay(
                frame_bgr,
                detection,
                raw_intrinsics,
                tuple(resolution),
                len(records),
                count_preferred(records),
                args.tag_id,
            )
            if now - last_status_print >= 0.5:
                tracking_error = np.rad2deg(target_joint - robot.joint_pos)
                if detection.aruco is None:
                    tag_msg = "tag NOT DETECTED"
                else:
                    tag_msg = (
                        f"norm_mm={detection.aruco['norm_mm']:.1f} "
                        f"reproj={detection.aruco['reprojection_mean_px']:.3f}"
                    )
                print(
                    f"[KEYREC] {tag_msg} saved={len(records)} "
                    f"joint_deg={np.round(np.rad2deg(robot.joint_pos), 3).tolist()} "
                    f"tracking_error_deg={np.round(tracking_error, 3).tolist()}"
                )
                last_status_print = now

            if not args.no_preview:
                cv2.imshow("ARX5 Keyboard Hand-Eye", vis)
                key_code = cv2.waitKey(1) & 0xFF
            else:
                key_code = 255

            if key_code == 255:
                continue
            if key_code == 27:
                print("[KEYREC] Esc pressed: set_to_damping and quit")
                controller.set_to_damping()
                finished_cleanly = True
                break
            if key_code in (8, 127):
                if records:
                    deleted_idx = len(records) - 1
                    records.pop()
                    save_records(output, records)
                    overlay_path = debug_overlay_dir / f"sample_{deleted_idx:04d}.jpg"
                    if overlay_path.exists():
                        overlay_path.unlink()
                    print(f"[KEYREC] deleted sample={deleted_idx:04d}, remaining={len(records)}")
                continue
            key = chr(key_code).lower() if key_code < 128 else ""
            if key == "?":
                print_help(args)
                continue
            if key == "0":
                robot.get_state()
                target_joint = robot.joint_pos.copy()
                command_joint(controller, target_joint, robot.gripper_pos, args.preview_sec)
                print(f"[KEYREC] synced target to actual {np.round(np.rad2deg(target_joint), 3).tolist()} deg")
                continue
            if key_code == ord(" "):
                robot.get_state()
                target_joint = robot.joint_pos.copy()
                command_joint(controller, target_joint, robot.gripper_pos, args.preview_sec)
                print("[KEYREC] save requested: waiting for stability")
                stable_sample, stability_stats = collect_stable_sample(
                    cap,
                    robot,
                    raw_intrinsics,
                    aruco_config,
                    args,
                    tuple(resolution),
                )
                if stable_sample is None or not stability_stats["stability_pass"]:
                    print("[KEYREC] sample not saved: stability/quality gate rejected it")
                    continue
                record = build_record_from_sample(stable_sample, stability_stats, args)
                save_vis = draw_detection_overlay(
                    stable_sample.frame_bgr,
                    stable_sample.detection,
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
                aruco = stable_sample.detection.aruco
                print(
                    f"[KEYREC] saved sample={len(records)-1:03d} "
                    f"joint_state={np.round(stable_sample.joint_state, 6).tolist()} "
                    f"tcp_pose={np.round(stable_sample.tcp_pose, 6).tolist()} "
                    f"aruco_detected={aruco is not None} "
                    f"norm_mm={None if aruco is None else round(aruco['norm_mm'], 3)} "
                    f"reproj={None if aruco is None else round(aruco['reprojection_mean_px'], 4)} "
                    f"stability_joint_delta_deg={stability_stats['stability_joint_delta_deg']} "
                    f"stability_tcp_trans_delta_mm={stability_stats['stability_tcp_trans_delta_mm']} "
                    f"debug_overlay={overlay_path}"
                )
                continue
            if key in JOINT_KEYMAP:
                joint_idx, direction = JOINT_KEYMAP[key]
                robot.get_state()
                actual = robot.joint_pos.copy()
                tracking_error_deg = np.max(np.abs(np.rad2deg(target_joint - actual)))
                if tracking_error_deg > args.max_tracking_error_deg:
                    print(
                        "[KEYREC] target synced because tracking error exceeded limit: "
                        f"{tracking_error_deg:.3f}deg"
                    )
                    target_joint = actual.copy()
                else:
                    target_joint = actual.copy()
                target_joint[joint_idx] += np.deg2rad(args.joint_step_deg) * direction
                if not ensure_joint_limits(target_joint, robot_config):
                    continue
                command_joint(controller, target_joint, robot.gripper_pos, args.preview_sec)
                print(
                    f"[KEYREC][JOINT] key={key} J{joint_idx + 1} "
                    f"target_deg={np.round(np.rad2deg(target_joint), 3).tolist()} "
                    f"actual_deg={np.round(np.rad2deg(actual), 3).tolist()}"
                )
                continue
            print(f"[KEYREC] ignored key={key_code}")
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if records:
            save_records(output, records)
            print(f"[KEYREC] wrote {len(records)} samples to {output}")
        elif finished_cleanly and not output.exists():
            save_records(output, records)
            print(f"[KEYREC] wrote 0 samples to {output}")
        else:
            print(f"[KEYREC] wrote 0 samples; left output unchanged: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
