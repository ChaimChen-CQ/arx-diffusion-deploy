#!/usr/bin/env python3
"""
Small-step keyboard jog for ARX5 over the existing ZMQ server.

Default mode is dry-run. Pass --run to send commands.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np

from modules.arx5_zmq_client import Arx5Client


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

CARTESIAN_KEYMAP = {
    "i": (0, 1.0, "x"),
    "k": (0, -1.0, "x"),
    "j": (1, 1.0, "y"),
    "l": (1, -1.0, "y"),
    "u": (2, 1.0, "z"),
    "o": (2, -1.0, "z"),
}

L5_JOINT_MIN_RAD = np.array([-3.14, -0.05, -0.1, -1.6, -1.57, -2.0], dtype=np.float64)
L5_JOINT_MAX_RAD = np.array([2.618, 3.50, 3.20, 1.55, 1.57, 2.0], dtype=np.float64)


def parse_vec3(text, name):
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 comma-separated values")
    return np.asarray(values, dtype=np.float64)


class RawTerminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not readable:
            return None
        return sys.stdin.read(1)


def print_help(args):
    run_state = "RUN" if args.run else "DRY-RUN"
    print(f"[JOG] mode={args.mode} state={run_state}")
    print("[JOG] Esc: emergency damping + quit, Ctrl-C: quit, ?: help, 0: sync joint target to actual")
    print("[JOG] joint keys: q/a J1, w/s J2, e/d J3, r/f J4, t/g J5, y/h J6")
    print("[JOG] cartesian keys: i/k X, j/l Y, u/o Z")
    print(
        f"[JOG] joint_step={args.joint_step_deg:.3f}deg "
        f"joint_reference={args.joint_reference} "
        f"max_joint_error={args.max_joint_error_deg:.3f}deg "
        f"sync_inactive={args.sync_inactive_joints} "
        f"sync_after={args.sync_after_command} "
        f"cart_step={args.cartesian_step_mm:.3f}mm"
    )


def ensure_joint_limits(target, args):
    if args.disable_joint_limits:
        return True
    below = target < L5_JOINT_MIN_RAD
    above = target > L5_JOINT_MAX_RAD
    if np.any(below | above):
        print(
            "[JOG] blocked by L5 joint limit: "
            f"target_deg={np.round(np.rad2deg(target), 3).tolist()} "
            f"min_deg={np.round(np.rad2deg(L5_JOINT_MIN_RAD), 3).tolist()} "
            f"max_deg={np.round(np.rad2deg(L5_JOINT_MAX_RAD), 3).tolist()}"
        )
        return False
    return True


def ensure_workspace(target_pose, workspace_min, workspace_max):
    pos = target_pose[:3]
    if np.any(pos < workspace_min) or np.any(pos > workspace_max):
        print(
            "[JOG] blocked by workspace limit: "
            f"pos={np.round(pos, 6).tolist()} "
            f"min={workspace_min.tolist()} max={workspace_max.tolist()}"
        )
        return False
    return True


def diagnose_cartesian_motion(axis, commanded_delta_m, actual_delta_m):
    axis_delta = actual_delta_m[axis]
    actual_norm = float(np.linalg.norm(actual_delta_m))
    commanded_abs = abs(commanded_delta_m)
    if actual_norm < max(0.0002, commanded_abs * 0.2):
        return "no visible motion: set_pose rejected, IK failed, or controller did not execute"
    if np.sign(axis_delta) != np.sign(commanded_delta_m):
        return "axis moved in opposite sign: likely frame direction mismatch"
    off_axis = actual_delta_m.copy()
    off_axis[axis] = 0.0
    if np.linalg.norm(off_axis) > max(abs(axis_delta), 1e-9):
        return "motion is mostly off-axis: likely frame coupling or IK solution issue"
    if abs(axis_delta) < commanded_abs * 0.5:
        return "axis motion is much smaller than command: likely IK/controller tracking limit"
    return "axis motion follows command"


def jog_joint(robot, key, args, joint_target):
    joint_idx, direction = JOINT_KEYMAP[key]
    robot.get_state()
    before_joint = np.asarray(robot.joint_pos, dtype=np.float64).copy()
    before_tcp = np.asarray(robot.tcp_pose, dtype=np.float64).copy()
    if joint_target is None or args.joint_reference == "actual":
        joint_target = before_joint.copy()
    step_rad = np.deg2rad(args.joint_step_deg) * direction
    if args.sync_inactive_joints:
        target = before_joint.copy()
        target[joint_idx] = joint_target[joint_idx]
    else:
        target = joint_target.copy()
    target[joint_idx] += step_rad
    if not ensure_joint_limits(target, args):
        return joint_target
    command_error = target - before_joint
    max_joint_error_rad = np.deg2rad(args.max_joint_error_deg)
    max_command_error_rad = float(np.max(np.abs(command_error)))
    if max_command_error_rad > max_joint_error_rad:
        print(
            "[JOG][JOINT] blocked by max joint command error: "
            f"max_error_deg={np.rad2deg(max_command_error_rad):.3f} "
            f"limit_deg={args.max_joint_error_deg:.3f} "
            f"actual_deg={np.round(np.rad2deg(before_joint), 3).tolist()} "
            f"target_deg={np.round(np.rad2deg(target), 3).tolist()}"
        )
        return before_joint.copy() if args.sync_target_on_block else joint_target
    print(
        f"[JOG][JOINT] key={key} J{joint_idx + 1} step_deg={np.rad2deg(step_rad):.3f} "
        f"before_deg={np.round(np.rad2deg(before_joint), 3).tolist()} "
        f"prev_cmd_deg={np.round(np.rad2deg(joint_target), 3).tolist()} "
        f"target_deg={np.round(np.rad2deg(target), 3).tolist()} "
        f"cmd_error_deg={np.round(np.rad2deg(command_error), 3).tolist()}"
    )
    if args.run:
        try:
            robot.set_joint_pos(
                target,
                duration=args.duration_sec,
                max_joint_step_rad=max_joint_error_rad,
            )
        except ValueError as exc:
            print(f"[JOG][JOINT] command rejected: {exc}")
            return joint_target
        time.sleep(args.settle_sec)
        robot.get_state()
        after_joint = np.asarray(robot.joint_pos, dtype=np.float64).copy()
        after_tcp = np.asarray(robot.tcp_pose, dtype=np.float64).copy()
        tracking_error = target - after_joint
        print(
            f"[JOG][JOINT] actual_after_deg={np.round(np.rad2deg(after_joint), 3).tolist()} "
            f"actual_delta_deg={np.round(np.rad2deg(after_joint - before_joint), 4).tolist()} "
            f"tracking_error_deg={np.round(np.rad2deg(tracking_error), 3).tolist()} "
            f"tcp_delta_mm={np.round((after_tcp[:3] - before_tcp[:3]) * 1000.0, 4).tolist()}"
        )
    if args.sync_after_command:
        return after_joint.copy() if args.run else before_joint.copy()
    return target


def jog_cartesian(robot, key, args, workspace_min, workspace_max):
    axis, direction, axis_name = CARTESIAN_KEYMAP[key]
    robot.get_state()
    if args.cartesian_pose == "tcp":
        before_pose = np.asarray(robot.tcp_pose, dtype=np.float64).copy()
    else:
        before_pose = np.asarray(robot.ee_pose, dtype=np.float64).copy()
    step_m = args.cartesian_step_mm * 0.001 * direction
    target_pose = before_pose.copy()
    target_pose[axis] += step_m
    if not ensure_workspace(target_pose, workspace_min, workspace_max):
        return
    print(
        f"[JOG][CART] key={key} axis={axis_name} step_mm={step_m * 1000.0:.3f} "
        f"pose_source={args.cartesian_pose}"
    )
    print(f"[JOG][CART] command_pose={np.round(target_pose, 6).tolist()}")
    print(f"[JOG][CART] actual_before={np.round(before_pose, 6).tolist()}")
    if args.run:
        try:
            if args.cartesian_pose == "tcp":
                robot.set_tcp_pose(target_pose)
            else:
                robot.set_ee_pose(target_pose)
        except ValueError as exc:
            print(f"[JOG][CART] command rejected: {exc}")
            print("[JOG][CART] diagnosis=IK failed or SET_EE_POSE safety gate rejected the pose")
            return
        time.sleep(args.settle_sec)
        robot.get_state()
        if args.cartesian_pose == "tcp":
            after_pose = np.asarray(robot.tcp_pose, dtype=np.float64).copy()
        else:
            after_pose = np.asarray(robot.ee_pose, dtype=np.float64).copy()
        actual_delta_m = after_pose[:3] - before_pose[:3]
        print(f"[JOG][CART] actual_after={np.round(after_pose, 6).tolist()}")
        print(f"[JOG][CART] actual_delta_mm={np.round(actual_delta_m * 1000.0, 4).tolist()}")
        print(f"[JOG][CART] diagnosis={diagnose_cartesian_motion(axis, step_m, actual_delta_m)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_ip", default="127.0.0.1")
    parser.add_argument("--robot_port", type=int, default=8765)
    parser.add_argument("--mode", choices=["joint", "cartesian"], default="joint")
    parser.add_argument("--run", action="store_true", help="Actually send robot commands. Default is dry-run.")
    parser.add_argument("--joint_step_deg", type=float, default=0.5)
    parser.add_argument(
        "--joint_reference",
        choices=["commanded", "actual"],
        default="commanded",
        help="Use accumulated command target or recompute every jog from actual joint state.",
    )
    parser.add_argument(
        "--sync_inactive_joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For joint jog, hold non-commanded joints at their latest actual positions.",
    )
    parser.add_argument(
        "--sync_after_command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each command, reset the accumulated joint target to the latest actual state.",
    )
    parser.add_argument(
        "--sync_target_on_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When max joint error blocks a command, reset target to actual to avoid repeated blocks.",
    )
    parser.add_argument(
        "--max_joint_error_deg",
        type=float,
        default=2.0,
        help="Maximum allowed difference between commanded target and actual joint state.",
    )
    parser.add_argument("--cartesian_step_mm", type=float, default=1.0)
    parser.add_argument("--cartesian_pose", choices=["tcp", "ee"], default="tcp")
    parser.add_argument("--duration_sec", type=float, default=0.15)
    parser.add_argument("--settle_sec", type=float, default=0.25)
    parser.add_argument(
        "--startup_settle_sec",
        type=float,
        default=0.2,
        help="Delay after switching/holding the controller at startup.",
    )
    parser.add_argument("--workspace_min", default="-0.8,-0.8,0.02")
    parser.add_argument("--workspace_max", default="0.8,0.8,0.8")
    parser.add_argument("--disable_joint_limits", action="store_true")
    parser.add_argument("--no_hold_on_start", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.joint_step_deg <= 0:
        raise ValueError("--joint_step_deg must be > 0")
    if args.max_joint_error_deg < args.joint_step_deg:
        raise ValueError("--max_joint_error_deg must be >= --joint_step_deg")
    if args.cartesian_step_mm <= 0:
        raise ValueError("--cartesian_step_mm must be > 0")
    if args.duration_sec < 0:
        raise ValueError("--duration_sec must be >= 0")
    if args.settle_sec < 0:
        raise ValueError("--settle_sec must be >= 0")
    if args.startup_settle_sec < 0:
        raise ValueError("--startup_settle_sec must be >= 0")

    workspace_min = parse_vec3(args.workspace_min, "--workspace_min")
    workspace_max = parse_vec3(args.workspace_max, "--workspace_max")
    if np.any(workspace_max <= workspace_min):
        raise ValueError("--workspace_max must be greater than --workspace_min")

    print_help(args)
    robot = Arx5Client(args.robot_ip, args.robot_port)
    if args.run and not args.no_hold_on_start:
        if args.mode == "joint":
            print("[JOG] switch to joint controller and hold current joints before jog")
            robot.get_state()
            current_joint = np.asarray(robot.joint_pos, dtype=np.float64).copy()
            robot.set_joint_pos(
                current_joint,
                duration=0.0,
                max_joint_step_rad=np.deg2rad(args.max_joint_error_deg),
            )
            time.sleep(args.startup_settle_sec)
        else:
            print("[JOG] hold_current_pose before jog")
            robot.hold_current_pose()

    try:
        joint_target = None
        with RawTerminal() as terminal:
            while True:
                key = terminal.read_key()
                if key is None:
                    continue
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == "\x1b":
                    print("[JOG] emergency stop key pressed")
                    if args.run:
                        robot.set_to_damping()
                    return 0
                if key == "?":
                    print_help(args)
                    continue
                if key == "0":
                    robot.get_state()
                    joint_target = np.asarray(robot.joint_pos, dtype=np.float64).copy()
                    print(
                        "[JOG][JOINT] synced command target to actual: "
                        f"{np.round(np.rad2deg(joint_target), 3).tolist()} deg"
                    )
                    continue
                if args.mode == "joint":
                    if key in JOINT_KEYMAP:
                        joint_target = jog_joint(robot, key, args, joint_target)
                    else:
                        print(f"[JOG] ignored key={key!r} in joint mode")
                else:
                    if key in CARTESIAN_KEYMAP:
                        jog_cartesian(robot, key, args, workspace_min, workspace_max)
                    else:
                        print(f"[JOG] ignored key={key!r} in cartesian mode")
    except KeyboardInterrupt:
        print("\n[JOG] interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
