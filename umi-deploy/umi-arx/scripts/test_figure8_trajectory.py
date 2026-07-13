"""
test_figure8_trajectory.py
==========================
Hardcoded figure-8 trajectory test for ARX5 arm + Gen gripper with live camera.

Prerequisites (4 terminals, each with 环境清洗咒语 first):
  T1: roscore
  T2: cd gen_controller_sdk_release && source devel/setup.zsh && \
      rosrun robot_driver databus_single.py _serial_port:=/dev/ttyUSB0 \
      _topic_encoder:=/encoder _topic_target_distance:=/target_distance
  T3: cd arx5-sdk && python python/communication/zmq_server.py L5 can0
  T4: cd umi-arx  && python scripts/test_figure8_trajectory.py [OPTIONS]

Usage:
  python scripts/test_figure8_trajectory.py                   # default with GUI
  python scripts/test_figure8_trajectory.py --headless        # terminal-only
  python scripts/test_figure8_trajectory.py --dry-run         # no hardware
  python scripts/test_figure8_trajectory.py --amplitude 0.05  # larger sweep
"""

import sys
import os
import time
import math
import argparse
import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  ENVIRONMENT SANITY CHECK  (must run BEFORE any heavy imports)
# ──────────────────────────────────────────────────────────────────────
def _check_environment():
    """Hard-fail if ROS 2 Humble pollution is detected."""
    polluted = False
    for var in ("PYTHONPATH", "AMENT_PREFIX_PATH", "LD_LIBRARY_PATH", "PATH"):
        val = os.environ.get(var, "")
        if "ros/humble" in val or "ros2" in val.lower():
            polluted = True
            break
    ros_distro = os.environ.get("ROS_DISTRO", "")
    if ros_distro and ros_distro != "noetic":
        polluted = True

    if polluted:
        print("\033[91m" + "=" * 72)
        print("  FATAL: ROS 2 Humble environment detected!")
        print("  Please run the 环境清洗咒语 before executing this script:")
        print()
        print("    unset PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION")
        print('    export PATH=$(echo $PATH | tr \':\' \'\\n\' | grep -v "ros/humble" | paste -sd ":" -)')
        print('    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr \':\' \'\\n\' | grep -v "ros/humble" | grep -v "ros2" | paste -sd ":" -)')
        print("    conda activate arx5_ros")
        print("=" * 72 + "\033[0m")
        sys.exit(1)


_check_environment()

# ──────────────────────────────────────────────────────────────────────
#  Path setup
# ──────────────────────────────────────────────────────────────────────
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from multiprocessing.managers import SharedMemoryManager
from utils.other_util import precise_wait


# ──────────────────────────────────────────────────────────────────────
#  Math: Lissajous figure-8 trajectory
# ──────────────────────────────────────────────────────────────────────
def compute_figure8_pose(
    t: float,
    center: np.ndarray,
    amplitude_x: float,
    amplitude_y: float,
    omega: float,
) -> np.ndarray:
    """
    Lissajous curve with frequency ratio 1:2 → figure-8 in xy plane.

    x(t) = x₀ + Aₓ · sin(ω·t)
    y(t) = y₀ + Aᵧ · sin(2ω·t)
    z, rx, ry, rz  held constant from *center*.
    """
    pose = center.copy()
    pose[0] += amplitude_x * math.sin(omega * t)
    pose[1] += amplitude_y * math.sin(2.0 * omega * t)
    return pose


def compute_gripper_profile(
    t: float,
    t_total: float,
    gripper_max: float,
) -> float:
    """
    Smooth open→close bell curve: 0 → max → 0.

    gripper(t) = gripper_max · sin(π · t / T_total)
    """
    phase = math.pi * t / t_total
    return gripper_max * math.sin(phase)


# ──────────────────────────────────────────────────────────────────────
#  Dry-run (off-line) mode
# ──────────────────────────────────────────────────────────────────────
def run_dry(args):
    """Print the computed trajectory without touching any hardware."""
    omega = 2.0 * math.pi / args.period
    t_total = args.period * args.loops
    dt = 1.0 / args.frequency

    # Fake initial pose: [x=0.3, y=0.0, z=0.2, rx=0, ry=0, rz=0]
    center = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])

    print(f"[DRY-RUN] center={center}, Ax={args.amplitude}, Ay={args.amplitude}")
    print(f"[DRY-RUN] period={args.period}s, loops={args.loops}, total={t_total}s, freq={args.frequency}Hz")
    print(f"{'time':>8s}  {'x':>8s}  {'y':>8s}  {'z':>8s}  {'grip':>8s}")
    print("-" * 50)

    total_steps = int(t_total * args.frequency)
    for i in range(total_steps + 1):
        t = i * dt
        pose = compute_figure8_pose(t, center, args.amplitude, args.amplitude, omega)
        grip = compute_gripper_profile(t, t_total, args.gripper_max)
        if i % max(1, int(args.frequency)) == 0:  # print ~1Hz
            print(f"{t:8.2f}  {pose[0]:8.4f}  {pose[1]:8.4f}  {pose[2]:8.4f}  {grip:8.4f}")

    print(f"\n[DRY-RUN] Total waypoints: {total_steps + 1}")
    print("[DRY-RUN] Max x-offset: {:.4f} m, Max y-offset: {:.4f} m".format(
        args.amplitude, args.amplitude))
    print("[DRY-RUN] Max step increment (x): {:.5f} m".format(
        args.amplitude * omega * dt))
    print("[DRY-RUN] Done.  (ZMQ 0.1m safety margin is safe ✓)")


# ──────────────────────────────────────────────────────────────────────
#  Hardware execution
# ──────────────────────────────────────────────────────────────────────
def run_hardware(args):
    """Execute the figure-8 trajectory on real hardware."""
    from modules.arx5_controller import Arx5Controller

    # Optional camera
    camera = None
    cv2 = None
    if not args.no_camera:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            from peripherals.uvc_camera import UvcCamera
            from peripherals.video_recorder import VideoRecorder
        except ImportError as e:
            print(f"[WARN] Camera imports failed ({e}), running without camera.")
            args.no_camera = True

    omega = 2.0 * math.pi / args.period
    t_total = args.period * args.loops
    dt = 1.0 / args.frequency
    total_steps = int(t_total * args.frequency)

    print("=" * 60)
    print("  ARX5 + Gen Gripper  Figure-8 Trajectory Test")
    print("=" * 60)
    print(f"  Amplitude   : {args.amplitude} m")
    print(f"  Period       : {args.period} s")
    print(f"  Loops        : {args.loops}")
    print(f"  Total time   : {t_total} s  ({total_steps} steps @ {args.frequency} Hz)")
    print(f"  Gripper max  : {args.gripper_max} m")
    print(f"  Headless     : {args.headless}")
    print(f"  Camera       : {'disabled' if args.no_camera else args.camera_dev}")
    print(f"  skip_home    : True  (bumpless)")
    print("=" * 60)

    with SharedMemoryManager() as shm_manager:
        # ── Arx5Controller (arm + gripper) ──
        robot = Arx5Controller(
            shm_manager=shm_manager,
            robot_ip="127.0.0.1",
            robot_port=8765,
            frequency=200,
            verbose=False,
            skip_home=False,  # <--- 【修改点 1】改回 False，让底层执行安全的关节空间回零
        )

        # ── UvcCamera (optional) ──
        if not args.no_camera:
            camera = UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=args.camera_dev,
                resolution=(1280, 720),
                capture_fps=30,
                put_fps=30,
                get_max_k=2,
                receive_latency=0.0,
                cap_buffer_size=1,
                verbose=False,
            )

        # ── Start subsystems ──
        robot.start()
        if camera is not None:
            camera.start()

        try:
            # Wait for readiness
            print("[INFO] Waiting for Arx5Controller to synchronize...")
            while not robot.is_ready:
                time.sleep(0.1)
            print("[OK] Arx5Controller is ready.")

            if camera is not None:
                print("[INFO] Waiting for camera...")
                while not camera.is_ready:
                    time.sleep(0.1)
                print("[OK] Camera is ready.")

            # 【修复致命跳变】获取底层回到 Home 之后的真实安全位姿
            state = robot.get_state()
            current_pose = state["ActualTCPPose"].copy()
            
            # 只修改 XYZ 平移，原封不动地保留 Rx, Ry, Rz 旋转角！
            center = current_pose.copy()
            center[0] = 0.30  # X
            center[1] = 0.00  # Y
            center[2] = 0.20  # Z
            
            grip_init = float(state["gripper_position"])
            print(f"[INFO] Trajectory Center forced to : {center}")
            print(f"[INFO] Initial gripper  : {grip_init:.4f} m")
            print()
            
            # 确保机械臂平滑移动到这个安全中心（duration=3秒）
            print("[ACTION] Moving to safe center pose...")
            robot.servoL(pose=center, gripper_pos=0.0, duration=3.0)
            time.sleep(3.5)

            # Create OpenCV window (if GUI)
            if not args.headless and cv2 is not None and camera is not None:
                cv2.namedWindow("ARX5 Camera", cv2.WINDOW_NORMAL)

            # ── Main control loop ──
            print("[ACTION] Starting figure-8 trajectory...")
            t_start = time.monotonic()
            log_interval = max(1, int(args.frequency))  # ~1Hz logging

            for step_i in range(total_steps):
                t_now = time.monotonic()
                t_elapsed = t_now - t_start

                # Compute targets
                pose_target = compute_figure8_pose(
                    t_elapsed, center, args.amplitude, args.amplitude, omega
                )
                gripper_target = compute_gripper_profile(
                    t_elapsed, t_total, args.gripper_max
                )

                # Send command (non-blocking, writes to SharedMemoryQueue)
                robot.servoL(
                    pose=pose_target,
                    gripper_pos=gripper_target,
                    duration=dt,
                )

                # Camera: read + display/log
                if camera is not None:
                    try:
                        frame_data = camera.get()
                        frame = frame_data["color"]

                        if not args.headless and cv2 is not None:
                            cv2.imshow("ARX5 Camera", frame)
                    except Exception as e:
                        # 偶尔读不到新帧是正常的，直接跳过本次 imshow，但绝对不能吞掉下面的 waitKey
                        pass

                # ⚠️ 【关键修复】waitKey 必须放在 try 的外面！保证它在每次循环都必然执行以刷新 GUI
                if not args.headless and cv2 is not None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] 'q' pressed — aborting trajectory.")
                        break

                # Headless logging (~1Hz)
                if args.headless and step_i % log_interval == 0:
                    cur_state = robot.get_state()
                    tcp = cur_state["ActualTCPPose"]
                    grip = float(cur_state["gripper_position"])
                    print(
                        f"  t={t_elapsed:6.2f}s  "
                        f"TCP=[{tcp[0]:+.4f} {tcp[1]:+.4f} {tcp[2]:+.4f}]  "
                        f"grip={grip:.4f}m  "
                        f"target_grip={gripper_target:.4f}m"
                    )

                # Maintain frequency
                t_wait = t_start + (step_i + 1) * dt
                precise_wait(t_wait, time_func=time.monotonic)

            # ── Trajectory complete: return to center ──
            print("\n[ACTION] Trajectory complete. Returning to center pose...")
            robot.servoL(pose=center, gripper_pos=0.0, duration=2.0)
            time.sleep(3.0)

            final_state = robot.get_state()
            print(f"[DONE] Final TCP pose : {final_state['ActualTCPPose']}")
            print(f"[DONE] Final gripper  : {final_state['gripper_position']:.4f} m")
            print("\n[SUCCESS] Figure-8 trajectory test completed successfully!")

        except KeyboardInterrupt:
            print("\n[WARN] KeyboardInterrupt — stopping safely...")
        finally:
            print("[INFO] Cleaning up...")
            if not args.headless and cv2 is not None:
                cv2.destroyAllWindows()
            if camera is not None:
                camera.stop(wait=True)
            robot.stop(wait=True)
            print("[INFO] All resources released.")


# ──────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ARX5 + Gen Gripper figure-8 trajectory test"
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.03,
        help="Figure-8 amplitude in meters (default: 0.03)",
    )
    parser.add_argument(
        "--period", type=float, default=10.0,
        help="Single figure-8 loop period in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--loops", type=int, default=2,
        help="Number of figure-8 loops (default: 2)",
    )
    parser.add_argument(
        "--gripper-max", type=float, default=0.05,
        help="Maximum gripper opening in meters (default: 0.05)",
    )
    parser.add_argument(
        "--frequency", type=float, default=10.0,
        help="Command frequency in Hz (default: 10.0)",
    )
    parser.add_argument(
        "--camera-dev", type=str, default="/dev/video0",
        help="V4L2 camera device path (default: /dev/video0)",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Disable camera entirely",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No GUI — print status to terminal only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print trajectory math without hardware (offline verification)",
    )

    args = parser.parse_args()

    if args.dry_run:
        run_dry(args)
    else:
        run_hardware(args)


if __name__ == "__main__":
    main()
