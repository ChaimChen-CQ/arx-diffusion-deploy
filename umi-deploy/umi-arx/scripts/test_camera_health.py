import argparse
import json
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from peripherals.uvc_camera import UvcCamera
from utils.usb_util import get_sorted_v4l_paths, reset_all_elgato_devices


DEFAULT_GRIPPER_FISHEYE_INTRINSICS = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "calibration",
        "cam0_sensor_intrinsics.json",
    )
)


def load_intrinsics_resolution(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("intrinsic_type") != "FISHEYE":
        raise ValueError(f"Expected FISHEYE intrinsics, got {payload.get('intrinsic_type')}")
    return int(payload["image_width"]), int(payload["image_height"])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run ARX5 gripper UVC camera health test without policy or video recording.")
    parser.add_argument("--duration", type=float, default=180.0, help="Test duration in seconds.")
    parser.add_argument("--camera_reorder", default="0", help="Comma-separated camera indices from sorted v4l paths. Uses the first index.")
    parser.add_argument("--gripper_fisheye_intrinsics", default=DEFAULT_GRIPPER_FISHEYE_INTRINSICS)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--capture_fourcc", default="MJPG")
    parser.add_argument("--cap_buffer_size", type=int, default=1)
    parser.add_argument("--max_frame_staleness", type=float, default=1.0)
    parser.add_argument("--reset_elgato", action="store_true", help="Reset Elgato capture devices before opening cameras.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    resolution = load_intrinsics_resolution(args.gripper_fisheye_intrinsics)

    if args.reset_elgato:
        reset_all_elgato_devices()
        time.sleep(0.1)

    v4l_paths = get_sorted_v4l_paths()
    if not v4l_paths:
        raise RuntimeError(
            "No V4L camera devices found. Check /dev/v4l/by-id visibility, "
            "USB connection, and whether this shell/session has camera device access."
        )
    camera_indices = [int(token) for token in args.camera_reorder.split(",") if token.strip()]
    if not camera_indices:
        raise ValueError("--camera_reorder must contain at least one index")
    if max(camera_indices) >= len(v4l_paths):
        raise IndexError(
            f"--camera_reorder requested index {max(camera_indices)}, "
            f"but only {len(v4l_paths)} camera device(s) were found: {v4l_paths}"
        )
    dev_video_path = v4l_paths[camera_indices[0]]

    print(f"[CAMERA_HEALTH] device={dev_video_path}")
    print(f"[CAMERA_HEALTH] resolution={resolution[0]}x{resolution[1]} fps={args.fps} fourcc={args.capture_fourcc}")
    print(f"[CAMERA_HEALTH] video_recording=disabled duration={args.duration}s")

    failed = False
    with SharedMemoryManager() as shm_manager:
        camera = UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=dev_video_path,
            resolution=resolution,
            capture_fps=args.fps,
            put_downsample=False,
            get_max_k=16,
            receive_latency=0.0,
            cap_buffer_size=args.cap_buffer_size,
            capture_fourcc=args.capture_fourcc,
            enable_video_recording=False,
            verbose=False,
        )
        camera.start(wait=True)
        try:
            deadline = time.time() + args.duration
            next_log = 0.0
            while time.time() < deadline:
                now = time.time()
                status = camera.get_health_status()
                stale_for = None
                if status["last_frame_time"] > 0:
                    stale_for = max(0.0, now - status["last_frame_time"])

                if now >= next_log:
                    stale_text = "n/a" if stale_for is None else f"{stale_for:.3f}s"
                    print(
                        "[CAMERA_HEALTH] "
                        f"alive={status['process_alive']} ready={status['ready']} "
                        f"recovering={status['recovering']} failed={status['failed']} "
                        f"count={status['ring_buffer_count']} stale={stale_text} "
                        f"failures={status['failure_count']}"
                    )
                    next_log = now + 5.0

                if (
                    (not status["process_alive"])
                    or status["recovering"]
                    or status["failed"]
                    or status["failure_count"] > 0
                    or stale_for is None
                    or stale_for > args.max_frame_staleness
                ):
                    failed = True
                    break

                time.sleep(0.1)
        finally:
            camera.stop(wait=True)

    if failed:
        print("[CAMERA_HEALTH] FAILED")
        return 1
    print("[CAMERA_HEALTH] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
