#!/usr/bin/env python
"""
Eye-in-hand calibration for Flexiv Rizon + RealSense camera.

Workflow:
  1. [collect]    Robot enters free-drive mode (low joint impedance) so you can
                  drag the arm by hand. Press 's' to save image + EE pose.
  2. [calibrate]  Run hand-eye calibration from saved data (offline).

Usage:
  # Step 1 – collect data (uses free-drive, no need to switch pendant modes)
    python eye_in_hand_calibration.py collect --save_dir data/calibration/eye_in_hand

  # Step 2 – calibrate (offline, no hardware needed)
    python eye_in_hand_calibration.py calibrate --data_dir data/calibration/eye_in_hand
"""

import argparse
import math
import os
import glob
import time
import threading
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_project_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

# ──────────────────────────────────────────────────────────────────────
# Hardware serials (edit to match your setup)
# ──────────────────────────────────────────────────────────────────────
ROBOT_SERIAL = "Rizon4s-063191"
WRIST_CAMERA_SERIAL = "034522060350"
GLOBAL_CAMERA_SERIAL = "024122062052"

# ──────────────────────────────────────────────────────────────────────
# Checkerboard parameters
# Board 100×90 mm, square 5×5 mm  →  20×18 squares  →  19×17 inner corners
# ──────────────────────────────────────────────────────────────────────
BOARD_ROWS = 17
BOARD_COLS = 19
SQUARE_SIZE = 0.005  # metres


# ──────────────────────────────────────────────────────────────────────
# Pose conversions
# ──────────────────────────────────────────────────────────────────────
def flexiv_pose_to_matrix(pose):
    """Flexiv SDK [x, y, z, qw, qx, qy, qz] → 4×4 homogeneous matrix."""
    x, y, z, qw, qx, qy, qz = pose
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose7(T):
    """4×4 → [x, y, z, qx, qy, qz, qw] (scipy scalar-last)."""
    quat = Rot.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], quat])


# ──────────────────────────────────────────────────────────────────────
# Checkerboard detection
# ──────────────────────────────────────────────────────────────────────
def make_object_points():
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             + cv2.CALIB_CB_NORMALIZE_IMAGE
             + cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Force consistent ordering: first corner should be top-left.
    # If the first corner is below the last corner, the ordering is flipped 180°.
    if corners[0, 0, 1] > corners[-1, 0, 1]:
        corners = corners[::-1].copy()

    return corners


# ──────────────────────────────────────────────────────────────────────
# Hand-eye calibration  (AX = XB, eye-in-hand)
# ──────────────────────────────────────────────────────────────────────
HANDEYE_METHODS = {
    "tsai":       cv2.CALIB_HAND_EYE_TSAI,
    "park":       cv2.CALIB_HAND_EYE_PARK,
    "horaud":     cv2.CALIB_HAND_EYE_HORAUD,
    "andreff":    cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def calibrate_hand_eye(images, ee_poses, K, dist, method="tsai"):
    objp = make_object_points()
    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    for i, (img, T_ee) in enumerate(zip(images, ee_poses)):
        corners = detect_corners(img)
        if corners is None:
            print(f"  [warn] image {i}: corners not found, skipped")
            continue
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if not ret:
            print(f"  [warn] image {i}: solvePnP failed, skipped")
            continue
        R_target2cam.append(cv2.Rodrigues(rvec)[0])
        t_target2cam.append(tvec)
        R_gripper2base.append(T_ee[:3, :3])
        t_gripper2base.append(T_ee[:3, 3].reshape(3, 1))

    n = len(R_target2cam)
    assert n >= 3, f"Need >=3 valid pose pairs, got {n}"
    print(f"\n[HandEye] Using {n} valid pose pairs (method={method})")

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=HANDEYE_METHODS[method],
    )
    T = np.eye(4)
    T[:3, :3] = R_cam2ee
    T[:3, 3] = t_cam2ee.ravel()
    return T


# ──────────────────────────────────────────────────────────────────────
# RealSense camera
# ──────────────────────────────────────────────────────────────────────
def init_realsense(serial=None, resolution=(1280, 720), fps=30):
    import pyrealsense2 as rs

    if serial is None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device found")
        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"[RealSense] Auto-detected device: {serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    w, h = resolution
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color)
    intr = color_profile.as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    for _ in range(30):
        pipeline.wait_for_frames()

    print(f"[RealSense] Camera ready  serial={serial}  resolution={w}x{h}")
    print(f"[RealSense] K:\n{K}")
    return pipeline, K, dist


def get_realsense_frame(pipeline):
    frames = pipeline.wait_for_frames()
    return np.asanyarray(frames.get_color_frame().get_data())


# ──────────────────────────────────────────────────────────────────────
# Flexiv robot
# ──────────────────────────────────────────────────────────────────────
def init_flexiv(serial):
    import flexivrdk

    robot = flexivrdk.Robot(serial)
    if robot.fault():
        if not robot.ClearFault():
            raise RuntimeError("Cannot clear robot faults")
        time.sleep(1)
    robot.Enable()
    while not robot.operational():
        time.sleep(0.5)
    print(f"[Flexiv] Robot ready  serial={serial}")
    return robot


def get_ee_pose(robot):
    """Read current T_ee_in_base as 4×4."""
    pose = np.array(robot.states().tcp_pose, dtype=np.float64)
    return flexiv_pose_to_matrix(pose)


# ──────────────────────────────────────────────────────────────────────
# Free-drive (low joint impedance) — drag arm by hand while in auto mode
# ──────────────────────────────────────────────────────────────────────
class FreeDriveController:
    """Background thread that keeps the robot in low-impedance mode.

    Uses NRT_JOINT_IMPEDANCE with near-zero stiffness and continuously
    sends the current joint position as target, so the arm stays where
    you leave it but offers almost no resistance to external forces.
    """

    def __init__(self, robot, stiffness_ratio=0.5):
        import flexivrdk
        self.robot = robot
        self.stiffness_ratio = stiffness_ratio
        self._running = False
        self._thread = None
        self._flexivrdk = flexivrdk

    def start(self):
        robot = self.robot
        flexivrdk = self._flexivrdk

        robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
        K_q_nom = np.array(robot.info().K_q_nom)
        K_q = K_q_nom * self.stiffness_ratio
        robot.SetJointImpedance(K_q.tolist())
        print(f"[FreeDrive] Enabled  (stiffness_ratio={self.stiffness_ratio})")
        print(f"[FreeDrive] You can now drag the arm by hand.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        robot = self.robot
        DoF = robot.info().DoF
        zero = [0.0] * DoF
        max_vel = [2.0] * DoF
        max_acc = [3.0] * DoF
        while self._running:
            try:
                current_q = list(robot.states().q)
                robot.SendJointPosition(current_q, zero, zero, max_vel, max_acc)
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self.robot.SwitchMode(self._flexivrdk.Mode.IDLE)
        except Exception:
            pass
        print("[FreeDrive] Stopped")


# ──────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────
def collect_data(save_dir, robot_serial, camera_serial=None,
                 resolution=(1280, 720), freedrive=False):
    os.makedirs(save_dir, exist_ok=True)

    robot = init_flexiv(robot_serial)
    pipeline, K, dist = init_realsense(camera_serial, resolution)

    np.savez(os.path.join(save_dir, "intrinsics.npz"), K=K, dist=dist)
    print("[Saved] intrinsics.npz")

    fd = None
    if freedrive:
        fd = FreeDriveController(robot)
        fd.start()

    idx = len(glob.glob(os.path.join(save_dir, "*.png")))

    print("\n=== Data Collection ===")
    if freedrive:
        print("  FREE-DRIVE ON — drag the arm by hand to desired poses.")
    else:
        print("  Move robot via pendant / other means.")
    print("  Press 's' to save current frame + EE pose")
    print("  Press 'q' to quit")
    print(f"  Starting index: {idx}\n")

    try:
        while True:
            frame = get_realsense_frame(pipeline)
            vis = frame.copy()
            corners = detect_corners(frame)

            if corners is not None:
                cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
                cv2.putText(vis, "Corners OK", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "No corners", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(vis, f"Saved: {idx}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Eye-in-Hand Calibration", vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):
                if corners is None:
                    print("  [skip] no corners in current frame")
                    continue
                T_ee = get_ee_pose(robot)
                img_path = os.path.join(save_dir, f"{idx:04d}.png")
                pose_path = os.path.join(save_dir, f"{idx:04d}_pose.npy")
                cv2.imwrite(img_path, frame)
                np.save(pose_path, T_ee)
                p = T_ee[:3, 3]
                print(f"  [{idx:04d}] saved | EE = [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
                idx += 1
            elif key == ord('q'):
                break
    finally:
        if fd:
            fd.stop()
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\nCollected {idx} samples in {save_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Calibration (offline)
# ──────────────────────────────────────────────────────────────────────
def load_dataset(data_dir):
    img_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    images, poses = [], []
    for ip in img_paths:
        base = os.path.splitext(ip)[0]
        pose_path = base + "_pose.npy"
        if not os.path.exists(pose_path):
            print(f"  [warn] no pose for {ip}, skipped")
            continue
        images.append(cv2.imread(ip))
        poses.append(np.load(pose_path))
    assert len(images) >= 3, f"Need >=3 image/pose pairs, found {len(images)}"
    return images, poses


def run_calibration(data_dir, method="tsai"):
    images, ee_poses = load_dataset(data_dir)
    print(f"Loaded {len(images)} image/pose pairs from {data_dir}/")

    intr_path = os.path.join(data_dir, "intrinsics.npz")
    assert os.path.exists(intr_path), f"intrinsics.npz not found in {data_dir}"
    data = np.load(intr_path)
    K, dist = data["K"], data["dist"]
    print(f"Loaded intrinsics from {intr_path}")
    print(f"  K =\n{K}")

    print(f"\nRunning hand-eye calibration (method={method}) ...")
    T_cam_in_ee = calibrate_hand_eye(images, ee_poses, K, dist, method=method)

    print("\n" + "=" * 60)
    print("T_cam_in_ee (camera frame in end-effector frame):")
    print(T_cam_in_ee)

    pose7 = matrix_to_pose7(T_cam_in_ee)
    euler_deg = Rot.from_matrix(T_cam_in_ee[:3, :3]).as_euler('xyz', degrees=True)
    print(f"\n  translation  : {pose7[:3]}")
    print(f"  quat (xyzw)  : {pose7[3:]}")
    print(f"  euler XYZ    : {euler_deg} deg")

    T_ee_in_cam = np.linalg.inv(T_cam_in_ee)
    print("\nT_ee_in_cam (end-effector in camera frame):")
    print(T_ee_in_cam)
    print("=" * 60)

    out_path = os.path.join(data_dir, "hand_eye_result.npz")
    np.savez(
        out_path,
        T_cam_in_ee=T_cam_in_ee,
        T_ee_in_cam=T_ee_in_cam,
        K=K,
        dist=dist,
        method=method,
    )
    print(f"\nResults saved to {out_path}")

    verify_calibration(images, ee_poses, K, dist, T_cam_in_ee)
    all_methods_comparison(images, ee_poses, K, dist)

    return T_cam_in_ee


# ──────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────
def verify_calibration(images, ee_poses, K, dist, T_cam_in_ee):
    """Board is static → T_board_in_base should be consistent across images."""
    objp = make_object_points()
    positions = []

    for img, T_ee in zip(images, ee_poses):
        corners = detect_corners(img)
        if corners is None:
            continue
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if not ret:
            continue
        T_board_in_cam = np.eye(4)
        T_board_in_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_board_in_cam[:3, 3] = tvec.ravel()

        T_cam_in_base = T_ee @ T_cam_in_ee
        T_board_in_base = T_cam_in_base @ T_board_in_cam
        positions.append(T_board_in_base[:3, 3])

    if len(positions) < 2:
        print("\n[Verify] Not enough valid images.")
        return

    positions = np.array(positions)
    std = positions.std(axis=0)
    mean = positions.mean(axis=0)
    print(f"\n[Verify] Board origin in base frame:")
    print(f"         mean = {mean}")
    print(f"         std  = {std}  (metres)")
    if np.all(std < 0.005):
        print("         Result: GOOD (< 5 mm std)")
    elif np.all(std < 0.01):
        print("         Result: ACCEPTABLE (< 10 mm std)")
    else:
        print("         Result: POOR (> 10 mm std), check data quality")


def all_methods_comparison(images, ee_poses, K, dist):
    print("\n" + "=" * 60)
    print("Comparison of all solver methods:")
    print("-" * 60)
    for name in HANDEYE_METHODS:
        try:
            T = calibrate_hand_eye(images, ee_poses, K, dist, method=name)
            euler = Rot.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
            t = T[:3, 3]
            print(f"  {name:12s} | t=[{t[0]:+.5f}, {t[1]:+.5f}, {t[2]:+.5f}]  "
                  f"euler=[{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}] deg")
        except Exception as e:
            print(f"  {name:12s} | FAILED: {e}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Eye-in-hand calibration: Flexiv Rizon + RealSense + checkerboard"
    )
    sub = parser.add_subparsers(dest="command")

    p_collect = sub.add_parser("collect", help="Collect calibration data")
    p_collect.add_argument("--save_dir", default="data/calibration/eye_in_hand")
    p_collect.add_argument("--robot_serial", default=ROBOT_SERIAL)
    p_collect.add_argument("--camera_serial", default=WRIST_CAMERA_SERIAL,
                           help="RealSense serial (default: wrist camera)")
    p_collect.add_argument("--resolution", default="1280x720")
    p_collect.add_argument("--freedrive", action="store_true",
                           help="Enable free-drive mode (low joint impedance, drag arm by hand)")

    p_calib = sub.add_parser("calibrate", help="Run calibration from saved data")
    p_calib.add_argument("--data_dir", default="data/calibration/eye_in_hand")
    p_calib.add_argument("--method", default="tsai", choices=list(HANDEYE_METHODS.keys()))

    args = parser.parse_args()
    if hasattr(args, "save_dir"):
        args.save_dir = resolve_project_path(args.save_dir)
    if hasattr(args, "data_dir"):
        args.data_dir = resolve_project_path(args.data_dir)

    if args.command == "collect":
        w, h = map(int, args.resolution.split("x"))
        collect_data(args.save_dir, args.robot_serial, args.camera_serial,
                     (w, h), args.freedrive)
    elif args.command == "calibrate":
        run_calibration(args.data_dir, args.method)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
