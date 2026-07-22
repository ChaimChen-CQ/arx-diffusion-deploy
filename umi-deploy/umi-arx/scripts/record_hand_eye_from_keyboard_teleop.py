#!/usr/bin/env python3
"""
Record wrist-camera hand-eye samples while driving ARX5 with SDK keyboard teleop.

This script talks to the ARX5 SDK directly. Stop zmq_server.py before using it;
two processes must not own the same CAN interface at the same time.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMI_DEPLOY_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
SDK_PYTHON_DIR = os.path.abspath(os.path.join(UMI_DEPLOY_DIR, "arx5-sdk", "python"))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SDK_PYTHON_DIR)
os.chdir(SDK_PYTHON_DIR)

import cv2
import numpy as np
import yaml
from pynput import keyboard

from arx5_interface import Arx5CartesianController, EEFState, LogLevel
from utils.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


SCRIPT_NAME = "record_hand_eye_from_keyboard_teleop.py"
DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0"
)
DEFAULT_INTRINSICS = os.path.join(
    UMI_DEPLOY_DIR,
    "data_local",
    "calibration",
    "cam0_sensor_intrinsics.json",
)
DEFAULT_ARUCO_YAML = os.path.join(
    UMI_DEPLOY_DIR,
    "data_local",
    "hand_eye_tags",
    "aruco_config_tag12_147mm.yaml",
)
DEFAULT_OUTPUT = os.path.join(
    UMI_DEPLOY_DIR,
    "data_local",
    "hand_eye_recalib_640",
    "hand_eye_calib_300_350_v5.pkl",
)


@dataclass(frozen=True)
class CameraSnapshot:
    frame_index: int
    frame_bgr: np.ndarray
    undistorted_bgr: np.ndarray
    frame_host_timestamp: float
    aruco: dict[str, Any]
    checkerboard: dict[str, Any]
    decode_mode: str


def fourcc_to_str(value: int) -> str:
    return "".join(chr((int(value) >> (8 * i)) & 0xFF) for i in range(4))


def yuyv_to_bgr(yuyv: np.ndarray) -> np.ndarray:
    yuyv = np.ascontiguousarray(yuyv)
    for code_name in ("COLOR_YUV2BGR_YUY2", "COLOR_YUV2BGR_YUYV"):
        code = getattr(cv2, code_name, None)
        if code is None:
            continue
        try:
            return cv2.cvtColor(yuyv, code)
        except cv2.error:
            continue
    raise RuntimeError("OpenCV cannot convert YUYV/YUY2 to BGR")


def try_imdecode(raw: np.ndarray) -> np.ndarray | None:
    if raw.dtype != np.uint8:
        return None
    decoded = cv2.imdecode(np.ascontiguousarray(raw).reshape(-1), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        return None
    return decoded


def decode_camera_frame(
    raw: np.ndarray,
    width: int,
    height: int,
    fourcc: str,
) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw)
    width = int(width)
    height = int(height)

    if raw.ndim == 3 and raw.shape[2] == 3:
        return np.ascontiguousarray(raw), "opencv_bgr_3ch"
    if raw.ndim == 3 and raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR), "opencv_bgra_to_bgr"
    if raw.ndim == 3 and raw.shape[2] == 2:
        return yuyv_to_bgr(raw), "raw_yuyv_hwc2_to_bgr"

    if raw.ndim == 2:
        h, w = raw.shape
        if h == height and w == width * 2:
            return yuyv_to_bgr(raw.reshape(height, width, 2)), "raw_yuyv_hw2_to_bgr"
        if h == height and w == width and raw.dtype == np.uint16:
            raw_u8 = np.ascontiguousarray(raw).view(np.uint8)
            return yuyv_to_bgr(raw_u8.reshape(height, width, 2)), "raw_yuyv_uint16_to_bgr"
        if raw.size == width * height * 2:
            return yuyv_to_bgr(raw.reshape(height, width, 2)), "raw_yuyv_flat_to_bgr"
        decoded = try_imdecode(raw)
        if decoded is not None:
            return decoded, "compressed_imdecode_to_bgr"
        if h == height and w == width:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), "gray_to_bgr"

    if raw.ndim == 1:
        if raw.size == width * height * 2:
            return yuyv_to_bgr(raw.reshape(height, width, 2)), "raw_yuyv_1d_to_bgr"
        decoded = try_imdecode(raw)
        if decoded is not None:
            return decoded, "compressed_imdecode_to_bgr"

    raise RuntimeError(
        f"Unsupported camera frame layout shape={raw.shape} dtype={raw.dtype} "
        f"fourcc={fourcc}"
    )


def load_fisheye_intrinsics(path: str) -> dict[str, np.ndarray]:
    with open(path, "r") as f:
        return parse_fisheye_intrinsics(json.load(f))


def load_resolution(path: str) -> tuple[int, int]:
    with open(path, "r") as f:
        payload = json.load(f)
    if payload.get("intrinsic_type") != "FISHEYE":
        raise ValueError(f"Expected FISHEYE intrinsics, got {payload.get('intrinsic_type')}")
    return int(payload["image_width"]), int(payload["image_height"])


def load_aruco_config(path: str) -> dict[str, Any]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing; install an OpenCV build with aruco support")
    with open(path, "r") as f:
        return parse_aruco_config(yaml.safe_load(f))


def build_fisheye_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    resolution: tuple[int, int],
    balance: float = 0.0,
    preview_mode: str = "keep_k",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = resolution
    dim = (int(width), int(height))
    if preview_mode == "keep_k":
        preview_new_K = np.asarray(K, dtype=np.float64).copy()
    elif preview_mode == "new_k":
        preview_new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            dim,
            np.eye(3),
            balance=float(balance),
            new_size=dim,
        )
    else:
        raise ValueError(f"Unsupported undistort preview mode: {preview_mode}")
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        preview_new_K,
        dim,
        cv2.CV_16SC2,
    )
    return map1, map2, preview_new_K


def empty_aruco_result(tag_id: int, tag_size_m: float) -> dict[str, Any]:
    return {
        "tag_id": int(tag_id),
        "tag_size_m": float(tag_size_m),
        "detected": False,
        "corners": None,
        "rvec": None,
        "tvec": None,
        "z_mm": None,
        "norm_mm": None,
        "reprojection_error_px": None,
        "reprojection_mean_px": None,
        "reprojection_max_px": None,
    }


def detect_aruco(
    frame_bgr: np.ndarray,
    raw_intrinsics: dict[str, np.ndarray],
    aruco_config: dict[str, Any],
    tag_id: int,
    tag_size_m: float,
) -> dict[str, Any]:
    intr = convert_fisheye_intrinsics_resolution(
        raw_intrinsics,
        frame_bgr.shape[:2][::-1],
    )
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tag_dict = detect_localize_aruco_tags(
        img_rgb,
        aruco_config["aruco_dict"],
        aruco_config["marker_size_map"],
        intr,
    )
    if int(tag_id) not in tag_dict:
        return empty_aruco_result(tag_id, tag_size_m)

    tag = tag_dict[int(tag_id)]
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    half = float(tag_size_m) / 2.0
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
        "tag_size_m": float(tag_size_m),
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


def empty_checkerboard_result(
    xx: int,
    yy: int,
    square_size_m: float,
) -> dict[str, Any]:
    return {
        "xx": int(xx),
        "yy": int(yy),
        "square_size_m": float(square_size_m),
        "detected": False,
        "detected_on": "raw",
        "raw_corners": None,
        "undistorted_corners": None,
        "corners": None,
        "rvec": None,
        "tvec": None,
        "norm_mm": None,
        "reprojection_error_px": None,
        "reprojection_mean_px": None,
        "reprojection_max_px": None,
    }


def make_checkerboard_object_points(
    xx: int,
    yy: int,
    square_size_m: float,
) -> np.ndarray:
    objp = np.zeros((int(xx) * int(yy), 3), np.float32)
    objp[:, :2] = np.mgrid[0:int(xx), 0:int(yy)].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def detect_checkerboard_raw(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    preview_new_K: np.ndarray,
    xx: int,
    yy: int,
    square_size_m: float,
) -> dict[str, Any]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pattern_size = (int(xx), int(yy))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return empty_checkerboard_result(xx, yy, square_size_m)

    raw_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if raw_corners is None:
        raw_corners = corners
    raw_corners = np.asarray(raw_corners, dtype=np.float64).reshape(-1, 1, 2)
    undistorted_corners = cv2.fisheye.undistortPoints(
        raw_corners,
        K,
        D,
        P=preview_new_K,
    )

    result = {
        "xx": int(xx),
        "yy": int(yy),
        "square_size_m": float(square_size_m),
        "detected": True,
        "detected_on": "raw",
        "raw_corners": raw_corners.reshape(-1, 2).copy(),
        "undistorted_corners": undistorted_corners.reshape(-1, 2).copy(),
        "corners": undistorted_corners.reshape(-1, 2).copy(),
        "rvec": None,
        "tvec": None,
        "norm_mm": None,
        "reprojection_error_px": None,
        "reprojection_mean_px": None,
        "reprojection_max_px": None,
    }

    objp = make_checkerboard_object_points(xx, yy, square_size_m)
    zero_dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        np.asarray(objp, dtype=np.float64).reshape(-1, 1, 3),
        undistorted_corners,
        preview_new_K,
        zero_dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        projected, _ = cv2.projectPoints(
            np.asarray(objp, dtype=np.float64).reshape(-1, 1, 3),
            rvec,
            tvec,
            preview_new_K,
            zero_dist,
        )
        reproj = np.linalg.norm(
            projected.reshape(-1, 2) - undistorted_corners.reshape(-1, 2),
            axis=1,
        )
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        result.update(
            {
                "rvec": rvec.copy(),
                "tvec": tvec.copy(),
                "norm_mm": float(np.linalg.norm(tvec) * 1000.0),
                "reprojection_error_px": reproj.copy(),
                "reprojection_mean_px": float(reproj.mean()),
                "reprojection_max_px": float(reproj.max()),
            }
        )
    return result


def draw_fisheye_axes(
    vis_bgr: np.ndarray,
    aruco: dict[str, Any],
    intr: dict[str, np.ndarray],
    axis_length_m: float = 0.05,
) -> None:
    if not aruco.get("detected"):
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


def draw_pinhole_axes(
    vis_bgr: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    axis_length_m: float,
    thickness: int = 2,
) -> None:
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, -axis_length_m],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(
        axis_points,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K,
        np.zeros((4, 1), dtype=np.float64),
    )
    pts = np.round(projected.reshape(-1, 2)).astype(int)
    origin = tuple(pts[0])
    cv2.line(vis_bgr, origin, tuple(pts[1]), (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.line(vis_bgr, origin, tuple(pts[2]), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(vis_bgr, origin, tuple(pts[3]), (255, 0, 0), thickness, cv2.LINE_AA)


def draw_undistorted_aruco(
    vis_bgr: np.ndarray,
    aruco: dict[str, Any],
    raw_intrinsics: dict[str, np.ndarray],
    preview_new_K: np.ndarray,
) -> None:
    if not aruco.get("detected"):
        return
    corners = np.asarray(aruco["corners"], dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(
        corners,
        raw_intrinsics["K"],
        raw_intrinsics["D"],
        P=preview_new_K,
    )
    poly = np.round(undistorted).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis_bgr, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
    draw_pinhole_axes(
        vis_bgr,
        aruco["rvec"],
        aruco["tvec"],
        preview_new_K,
        float(aruco["tag_size_m"]) * 0.35,
    )


def draw_checkerboard_pose_raw(
    vis_bgr: np.ndarray,
    checkerboard: dict[str, Any],
) -> None:
    if not checkerboard.get("detected"):
        return
    pattern_size = (int(checkerboard["xx"]), int(checkerboard["yy"]))
    cv2.drawChessboardCorners(
        vis_bgr,
        pattern_size,
        np.asarray(checkerboard["raw_corners"], dtype=np.float32).reshape(-1, 1, 2),
        True,
    )


def draw_checkerboard_pose_undistorted(
    vis_bgr: np.ndarray,
    checkerboard: dict[str, Any],
) -> None:
    if not checkerboard.get("detected"):
        return
    pattern_size = (int(checkerboard["xx"]), int(checkerboard["yy"]))
    cv2.drawChessboardCorners(
        vis_bgr,
        pattern_size,
        np.asarray(checkerboard["undistorted_corners"], dtype=np.float32).reshape(
            -1,
            1,
            2,
        ),
        True,
    )


def draw_text_lines(
    vis: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (12, 24),
) -> None:
    x, y = origin
    for text, color in lines:
        cv2.putText(
            vis,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20


def draw_preview_overlay(
    frame_bgr: np.ndarray,
    undistorted_bgr: np.ndarray,
    aruco: dict[str, Any],
    checkerboard: dict[str, Any],
    raw_intrinsics: dict[str, np.ndarray],
    preview_new_K: np.ndarray,
    resolution: tuple[int, int],
    saved_count: int,
    joint_pos: np.ndarray | None,
    detect_mode: str,
    decode_mode: str,
) -> np.ndarray:
    raw_vis = frame_bgr.copy()
    undistorted_vis = undistorted_bgr.copy()
    good = (0, 255, 0)
    bad = (0, 0, 255)
    white = (255, 255, 255)
    detect_mode = str(detect_mode)

    if detect_mode == "aruco" and aruco.get("detected"):
        corners = np.asarray(aruco["corners"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(raw_vis, [corners], True, good, 2, cv2.LINE_AA)
        intr = convert_fisheye_intrinsics_resolution(raw_intrinsics, resolution)
        draw_fisheye_axes(raw_vis, aruco, intr)
        draw_undistorted_aruco(undistorted_vis, aruco, raw_intrinsics, preview_new_K)
    elif detect_mode == "checkerboard":
        draw_checkerboard_pose_raw(raw_vis, checkerboard)
        draw_checkerboard_pose_undistorted(undistorted_vis, checkerboard)

    if joint_pos is None:
        joint_line = "joint deg: unavailable"
    else:
        joint_deg = np.rad2deg(np.asarray(joint_pos, dtype=np.float64).reshape(-1))
        joint_line = "joint deg: " + " ".join(f"{value:6.1f}" for value in joint_deg)

    left_lines = [
        ("RAW camera0", white),
        (f"samples: {saved_count}", white),
        (f"decode: {decode_mode}", white),
        (joint_line, white),
        (f"mode: {detect_mode}", white),
    ]
    draw_text_lines(raw_vis, left_lines)

    right_lines = [
        ("UNDISTORTED camera0", white),
        (f"mode: {detect_mode}", white),
    ]
    if detect_mode == "aruco":
        color = good if aruco.get("detected") else bad
        right_lines.append((f"aruco: {'yes' if aruco.get('detected') else 'no'}", color))
        if aruco.get("detected"):
            right_lines.append((f"norm: {aruco['norm_mm']:.1f} mm", color))
            right_lines.append((f"reproj: {aruco['reprojection_mean_px']:.3f} px", color))
    else:
        color = good if checkerboard.get("detected") else bad
        right_lines.append(
            (f"checkerboard: {'yes' if checkerboard.get('detected') else 'no'}", color)
        )
        if checkerboard.get("detected"):
            if checkerboard.get("norm_mm") is not None:
                right_lines.append((f"norm: {checkerboard['norm_mm']:.1f} mm", color))
            if checkerboard.get("reprojection_mean_px") is not None:
                right_lines.append(
                    (f"reproj: {checkerboard['reprojection_mean_px']:.3f} px", color)
                )
    draw_text_lines(undistorted_vis, right_lines)
    assert raw_vis.shape == undistorted_vis.shape
    return np.hstack([raw_vis, undistorted_vis])


class CameraWorker:
    def __init__(
        self,
        cap: cv2.VideoCapture,
        resolution: tuple[int, int],
        fourcc: str,
        raw_intrinsics: dict[str, np.ndarray],
        undistort_map1: np.ndarray,
        undistort_map2: np.ndarray,
        preview_new_K: np.ndarray,
        aruco_config: dict[str, Any],
        tag_id: int,
        tag_size_m: float,
        checkerboard_xx: int,
        checkerboard_yy: int,
        checkerboard_square_size: float,
        detect_mode: str,
        debug_camera_decode: bool,
    ):
        self.cap = cap
        self.width, self.height = resolution
        self.fourcc = fourcc
        self.raw_intrinsics = raw_intrinsics
        self.undistort_map1 = undistort_map1
        self.undistort_map2 = undistort_map2
        self.preview_new_K = preview_new_K
        self.aruco_config = aruco_config
        self.tag_id = int(tag_id)
        self.tag_size_m = float(tag_size_m)
        self.checkerboard_xx = int(checkerboard_xx)
        self.checkerboard_yy = int(checkerboard_yy)
        self.checkerboard_square_size = float(checkerboard_square_size)
        self.debug_camera_decode = bool(debug_camera_decode)
        self._lock = threading.Lock()
        self._detect_mode_lock = threading.Lock()
        self._detect_mode = str(detect_mode)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera0-reader", daemon=True)
        self._latest: CameraSnapshot | None = None
        self._frame_index = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> CameraSnapshot | None:
        with self._lock:
            return self._latest

    def set_detect_mode(self, detect_mode: str) -> None:
        if detect_mode not in ("aruco", "checkerboard"):
            raise ValueError(f"Unsupported detect mode: {detect_mode}")
        with self._detect_mode_lock:
            self._detect_mode = detect_mode

    def get_detect_mode(self) -> str:
        with self._detect_mode_lock:
            return self._detect_mode

    def _run(self) -> None:
        last_error_print = 0.0
        while not self._stop.is_set():
            ret, raw = self.cap.read()
            frame_time = time.time()
            if not ret or raw is None:
                now = time.time()
                if now - last_error_print >= 1.0:
                    print("[CALIB] camera0 read failed")
                    last_error_print = now
                time.sleep(0.01)
                continue
            try:
                frame_bgr, decode_mode = decode_camera_frame(
                    raw,
                    self.width,
                    self.height,
                    self.fourcc,
                )
                frame_bgr = np.ascontiguousarray(frame_bgr)
                undistorted_bgr = cv2.remap(
                    frame_bgr,
                    self.undistort_map1,
                    self.undistort_map2,
                    interpolation=cv2.INTER_LINEAR,
                )
                if self._frame_index < 10:
                    print(
                        "[CALIB] camera decode "
                        f"raw_shape={raw.shape} raw_dtype={raw.dtype} "
                        f"actual_fourcc={self.fourcc} "
                        f"decode_mode={decode_mode} "
                        f"frame_bgr_shape={frame_bgr.shape}"
                    )
                if self.debug_camera_decode and self._frame_index < 10:
                    debug_dir = Path("/tmp/arx5_camera_decode_debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(debug_dir / f"raw_or_decoded_frame_{self._frame_index:03d}.jpg"),
                        frame_bgr,
                    )
                    cv2.imwrite(
                        str(debug_dir / f"undistorted_frame_{self._frame_index:03d}.jpg"),
                        undistorted_bgr,
                    )
                detect_mode = self.get_detect_mode()
                aruco = empty_aruco_result(self.tag_id, self.tag_size_m)
                checkerboard = empty_checkerboard_result(
                    self.checkerboard_xx,
                    self.checkerboard_yy,
                    self.checkerboard_square_size,
                )
                if detect_mode == "aruco":
                    aruco = detect_aruco(
                        frame_bgr,
                        self.raw_intrinsics,
                        self.aruco_config,
                        self.tag_id,
                        self.tag_size_m,
                    )
                else:
                    checkerboard = detect_checkerboard_raw(
                        frame_bgr,
                        self.raw_intrinsics["K"],
                        self.raw_intrinsics["D"],
                        self.preview_new_K,
                        self.checkerboard_xx,
                        self.checkerboard_yy,
                        self.checkerboard_square_size,
                    )
            except Exception as exc:
                now = time.time()
                if now - last_error_print >= 1.0:
                    print(f"[CALIB] camera0 decode/detect failed: {exc}")
                    last_error_print = now
                time.sleep(0.01)
                continue

            self._frame_index += 1
            snapshot = CameraSnapshot(
                frame_index=self._frame_index,
                frame_bgr=frame_bgr,
                undistorted_bgr=undistorted_bgr,
                frame_host_timestamp=float(frame_time),
                aruco=aruco,
                checkerboard=checkerboard,
                decode_mode=decode_mode,
            )
            with self._lock:
                self._latest = snapshot


def open_capture(args: argparse.Namespace, resolution: tuple[int, int]) -> tuple[cv2.VideoCapture, str]:
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    requested_fourcc = str(args.capture_fourcc).strip()
    if requested_fourcc.lower() not in ("", "auto", "none", "skip"):
        if len(requested_fourcc) != 4:
            raise ValueError("--capture_fourcc must be 4 chars, or auto/none to skip forcing FourCC")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*requested_fourcc))
    else:
        print("[CALIB] capture_fourcc=auto; not forcing V4L2 FourCC")
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.cap_buffer_size)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.camera}")

    for _ in range(args.camera_warmup_frames):
        cap.read()
        time.sleep(0.01)

    actual_fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    convert_rgb = cap.get(cv2.CAP_PROP_CONVERT_RGB)
    print(
        "[CALIB] opencv_capture="
        f"backend=CAP_V4L2 fourcc={actual_fourcc} "
        f"resolution={width}x{height} fps={fps:.3f} "
        f"convert_rgb={convert_rgb}"
    )
    if (width, height) != tuple(resolution):
        raise RuntimeError(f"Camera negotiated {width}x{height}, expected {resolution[0]}x{resolution[1]}")
    return cap, actual_fourcc


def wait_for_first_frame(camera: CameraWorker, timeout_sec: float = 3.0) -> CameraSnapshot:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        snapshot = camera.latest()
        if snapshot is not None:
            return snapshot
        time.sleep(0.02)
    raise RuntimeError("Timed out waiting for first camera0 frame")


def rpy_to_rotm(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3)
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def ee_pose_to_tcp_pose(ee_pose: np.ndarray) -> np.ndarray:
    ee_pose = np.asarray(ee_pose, dtype=np.float64).reshape(6)
    ee_rot_mat = rpy_to_rotm(ee_pose[3:])
    ee_to_tcp_rot_mat = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    tcp_rot_mat = ee_rot_mat @ ee_to_tcp_rot_mat
    tcp_rotvec, _ = cv2.Rodrigues(tcp_rot_mat)
    tcp_rotvec = tcp_rotvec.reshape(3)
    angle_rad = float(np.linalg.norm(tcp_rotvec))
    if angle_rad > 1e-12:
        tcp_rotvec = -(tcp_rotvec / angle_rad) * (2.0 * np.pi - angle_rad)
    return np.concatenate([ee_pose[:3], tcp_rotvec])


def pose6_to_mat(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    rotm, _ = cv2.Rodrigues(pose6[3:].reshape(3, 1))
    tx = np.eye(4, dtype=np.float64)
    tx[:3, :3] = rotm
    tx[:3, 3] = pose6[:3]
    return tx


def read_robot_state(controller: Arx5CartesianController) -> dict[str, Any]:
    host_before = time.time()
    joint_state = controller.get_joint_state()
    eef_state = controller.get_eef_state()
    host_after = time.time()

    joint_pos = np.asarray(joint_state.pos(), dtype=np.float64).copy()
    joint_vel = np.asarray(joint_state.vel(), dtype=np.float64).copy()
    joint_torque = np.asarray(joint_state.torque(), dtype=np.float64).copy()
    ee_pose = np.asarray(eef_state.pose_6d(), dtype=np.float64).copy()
    tcp_pose = ee_pose_to_tcp_pose(ee_pose)
    return {
        "robot_host_time_before": float(host_before),
        "robot_host_timestamp": float(host_after),
        "robot_timestamp": float(eef_state.timestamp),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_torque": joint_torque,
        "gripper_pos": float(joint_state.gripper_pos),
        "gripper_vel": float(joint_state.gripper_vel),
        "gripper_torque": float(joint_state.gripper_torque),
        "ee_pose": ee_pose,
        "tcp_pose": tcp_pose,
        "tx_base_tcp": pose6_to_mat(tcp_pose),
    }


def build_record(
    args: argparse.Namespace,
    robot_state: dict[str, Any],
    snapshot: CameraSnapshot,
    raw_intrinsics: dict[str, np.ndarray],
    preview_new_K: np.ndarray,
    resolution: tuple[int, int],
    detect_mode: str,
) -> dict[str, Any]:
    aruco = snapshot.aruco
    checkerboard = snapshot.checkerboard
    record = {
        "timestamp": float(snapshot.frame_host_timestamp),
        "sample_host_timestamp": float(time.time()),
        "frame_host_timestamp": float(snapshot.frame_host_timestamp),
        "frame_index": int(snapshot.frame_index),
        "robot_timestamp": robot_state["robot_timestamp"],
        "robot_host_timestamp": robot_state["robot_host_timestamp"],
        "robot_host_time_before": robot_state["robot_host_time_before"],
        "frame_robot_delta_ms": float(
            (robot_state["robot_host_timestamp"] - snapshot.frame_host_timestamp) * 1000.0
        ),
        "joint_pos": robot_state["joint_pos"].copy(),
        "joint_vel": robot_state["joint_vel"].copy(),
        "joint_torque": robot_state["joint_torque"].copy(),
        "gripper_pos": robot_state["gripper_pos"],
        "gripper_vel": robot_state["gripper_vel"],
        "gripper_torque": robot_state["gripper_torque"],
        "ee_pose": robot_state["ee_pose"].copy(),
        "tcp_pose": robot_state["tcp_pose"].copy(),
        "tx_base_tcp": robot_state["tx_base_tcp"].copy(),
        "image_bgr": snapshot.frame_bgr.copy(),
        "image_shape": tuple(int(v) for v in snapshot.frame_bgr.shape),
        "camera_decode_mode": snapshot.decode_mode,
        "detect_mode": str(detect_mode),
        "intrinsics_path": str(args.gripper_fisheye_intrinsics),
        "K": raw_intrinsics["K"].copy(),
        "D": raw_intrinsics["D"].copy(),
        "undistort_balance": float(args.undistort_balance),
        "undistort_preview_mode": str(args.undistort_preview_mode),
        "preview_new_K": preview_new_K.copy(),
        "undistort_new_K": preview_new_K.copy(),
        "new_K": preview_new_K.copy(),
        "resolution": tuple(int(v) for v in resolution),
        "aruco_yaml": str(args.aruco_yaml),
        "tag_id": int(args.tag_id),
        "tag_size_m": float(aruco["tag_size_m"]),
        "aruco_detected": bool(aruco["detected"]),
        "aruco_corners": None if aruco["corners"] is None else aruco["corners"].copy(),
        "aruco_rvec": None if aruco["rvec"] is None else aruco["rvec"].copy(),
        "aruco_tvec": None if aruco["tvec"] is None else aruco["tvec"].copy(),
        "aruco_norm_mm": aruco["norm_mm"],
        "aruco_reprojection_mean_px": aruco["reprojection_mean_px"],
        "aruco_reprojection_max_px": aruco["reprojection_max_px"],
        "aruco_reprojection_error_px": (
            None
            if aruco["reprojection_error_px"] is None
            else aruco["reprojection_error_px"].copy()
        ),
        "aruco": {
            "tag_id": int(aruco["tag_id"]),
            "tag_size_m": float(aruco["tag_size_m"]),
            "detected": bool(aruco["detected"]),
            "corners": None if aruco["corners"] is None else aruco["corners"].copy(),
            "rvec": None if aruco["rvec"] is None else aruco["rvec"].copy(),
            "tvec": None if aruco["tvec"] is None else aruco["tvec"].copy(),
            "z_mm": aruco["z_mm"],
            "norm_mm": aruco["norm_mm"],
            "reprojection_mean_px": aruco["reprojection_mean_px"],
            "reprojection_max_px": aruco["reprojection_max_px"],
            "reprojection_error_px": (
                None
                if aruco["reprojection_error_px"] is None
                else aruco["reprojection_error_px"].copy()
            ),
        },
        "checkerboard_xx": int(args.checkerboard_xx),
        "checkerboard_yy": int(args.checkerboard_yy),
        "checkerboard_square_size": float(args.checkerboard_square_size),
        "checkerboard_detected": bool(checkerboard["detected"]),
        "checkerboard_detected_on": checkerboard["detected_on"],
        "checkerboard_raw_corners": (
            None
            if checkerboard["raw_corners"] is None
            else checkerboard["raw_corners"].copy()
        ),
        "checkerboard_undistorted_corners": (
            None
            if checkerboard["undistorted_corners"] is None
            else checkerboard["undistorted_corners"].copy()
        ),
        "checkerboard_corners": (
            None
            if checkerboard["corners"] is None
            else checkerboard["corners"].copy()
        ),
        "checkerboard_rvec": (
            None
            if checkerboard["rvec"] is None
            else checkerboard["rvec"].copy()
        ),
        "checkerboard_tvec": (
            None
            if checkerboard["tvec"] is None
            else checkerboard["tvec"].copy()
        ),
        "checkerboard_norm_mm": checkerboard["norm_mm"],
        "checkerboard_reprojection_mean_px": checkerboard["reprojection_mean_px"],
        "checkerboard_reprojection_max_px": checkerboard["reprojection_max_px"],
        "checkerboard_reprojection_error_px": (
            None
            if checkerboard["reprojection_error_px"] is None
            else checkerboard["reprojection_error_px"].copy()
        ),
        "checkerboard": {
            "xx": int(checkerboard["xx"]),
            "yy": int(checkerboard["yy"]),
            "square_size_m": float(checkerboard["square_size_m"]),
            "detected": bool(checkerboard["detected"]),
            "detected_on": checkerboard["detected_on"],
            "raw_corners": (
                None
                if checkerboard["raw_corners"] is None
                else checkerboard["raw_corners"].copy()
            ),
            "undistorted_corners": (
                None
                if checkerboard["undistorted_corners"] is None
                else checkerboard["undistorted_corners"].copy()
            ),
            "corners": (
                None
                if checkerboard["corners"] is None
                else checkerboard["corners"].copy()
            ),
            "rvec": (
                None
                if checkerboard["rvec"] is None
                else checkerboard["rvec"].copy()
            ),
            "tvec": (
                None
                if checkerboard["tvec"] is None
                else checkerboard["tvec"].copy()
            ),
            "norm_mm": checkerboard["norm_mm"],
            "reprojection_mean_px": checkerboard["reprojection_mean_px"],
            "reprojection_max_px": checkerboard["reprojection_max_px"],
            "reprojection_error_px": (
                None
                if checkerboard["reprojection_error_px"] is None
                else checkerboard["reprojection_error_px"].copy()
            ),
        },
        "model": args.model,
        "interface": args.interface,
        "script": SCRIPT_NAME,
    }
    return record


def load_existing_records(output: Path) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    with open(output, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise TypeError(f"Expected list in existing pkl, got {type(records)!r}: {output}")
    print(f"[CALIB] loaded existing records: {len(records)} from {output}")
    return records


def save_records(output: Path, records: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(records, f)
    os.replace(tmp_path, output)


def print_help() -> None:
    print("[CALIB] Movement keys:")
    print("[CALIB]   Up/Down: +x/-x")
    print("[CALIB]   Left/Right: +y/-y")
    print("[CALIB]   n/m: +z/-z")
    print("[CALIB]   q/a: +roll/-roll")
    print("[CALIB]   w/s: +pitch/-pitch")
    print("[CALIB]   e/d: +yaw/-yaw")
    print("[CALIB]   r/f: gripper open/close")
    print("[CALIB] Extra keys:")
    print("[CALIB]   Space: save current sample")
    print("[CALIB]   Backspace: delete previous sample")
    print("[CALIB]   p: print current robot and detection state")
    print("[CALIB]   ?: print help")
    print("[CALIB]   Esc: slowly return home, then damping and exit")
    print("[CALIB] Detection:")
    print("[CALIB]   --detect_mode aruco/checkerboard")
    print("[CALIB]   t: toggle detect mode")
    print("[CALIB]   Left preview: raw camera0")
    print("[CALIB]   Right preview: fisheye-undistorted camera0")


def print_state(
    robot_state: dict[str, Any],
    snapshot: CameraSnapshot | None,
    detect_mode: str,
) -> None:
    print(
        "[CALIB] robot "
        f"joint_deg={np.round(np.rad2deg(robot_state['joint_pos']), 3).tolist()} "
        f"ee_pose={np.round(robot_state['ee_pose'], 6).tolist()} "
        f"tcp_pose={np.round(robot_state['tcp_pose'], 6).tolist()} "
        f"gripper={robot_state['gripper_pos']:.6f} "
        f"robot_ts={robot_state['robot_timestamp']:.6f}"
    )
    if snapshot is None:
        print(f"[CALIB] detect_mode={detect_mode} camera0=no frame")
        return
    aruco = snapshot.aruco
    checkerboard = snapshot.checkerboard
    print(
        "[CALIB] detection "
        f"mode={detect_mode} "
        f"frame={snapshot.frame_index} "
        f"age_ms={(time.time() - snapshot.frame_host_timestamp) * 1000.0:.1f} "
        f"decode={snapshot.decode_mode}"
    )
    print(
        "[CALIB] aruco "
        f"detected={aruco['detected']} "
        f"norm_mm={aruco['norm_mm']} "
        f"reproj={aruco['reprojection_mean_px']}"
    )
    print(
        "[CALIB] checkerboard "
        f"detected={checkerboard['detected']} "
        f"norm_mm={checkerboard['norm_mm']} "
        f"reproj={checkerboard['reprojection_mean_px']}"
    )


def command_key_id(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.backspace:
        return "backspace"
    if key == keyboard.Key.esc:
        return "esc"
    char = getattr(key, "char", None)
    if char in ("p", "P"):
        return "p"
    if char in ("t", "T"):
        return "toggle"
    if char == "?":
        return "help"
    return None


def command_event_for_id(key_id: str | None) -> str | None:
    if key_id == "space":
        return "save"
    if key_id == "backspace":
        return "delete"
    if key_id == "esc":
        return "quit"
    if key_id == "p":
        return "print"
    if key_id == "toggle":
        return "toggle_mode"
    if key_id == "help":
        return "help"
    return None


def get_filtered_keyboard_output(
    key_pressed: dict[keyboard.Key | keyboard.KeyCode, bool],
    keyboard_queue: Queue,
) -> np.ndarray:
    state = np.zeros(6, dtype=np.float64)
    if key_pressed[keyboard.Key.up]:
        state[0] = 1
    if key_pressed[keyboard.Key.down]:
        state[0] = -1
    if key_pressed[keyboard.Key.left]:
        state[1] = 1
    if key_pressed[keyboard.Key.right]:
        state[1] = -1
    if key_pressed[keyboard.Key.page_up] or key_pressed[keyboard.KeyCode.from_char("n")]:
        state[2] = 1
    if key_pressed[keyboard.Key.page_down] or key_pressed[keyboard.KeyCode.from_char("m")]:
        state[2] = -1
    if key_pressed[keyboard.KeyCode.from_char("q")]:
        state[3] = 1
    if key_pressed[keyboard.KeyCode.from_char("a")]:
        state[3] = -1
    if key_pressed[keyboard.KeyCode.from_char("w")]:
        state[4] = 1
    if key_pressed[keyboard.KeyCode.from_char("s")]:
        state[4] = -1
    if key_pressed[keyboard.KeyCode.from_char("e")]:
        state[5] = 1
    if key_pressed[keyboard.KeyCode.from_char("d")]:
        state[5] = -1

    if keyboard_queue.maxsize > 0 and keyboard_queue._qsize() == keyboard_queue.maxsize:
        keyboard_queue._get()

    keyboard_queue.put(state)
    return np.mean(np.array(list(keyboard_queue.queue)), axis=0)


def save_current_sample(
    args: argparse.Namespace,
    controller: Arx5CartesianController,
    camera: CameraWorker,
    raw_intrinsics: dict[str, np.ndarray],
    preview_new_K: np.ndarray,
    resolution: tuple[int, int],
    output: Path,
    debug_overlay_dir: Path,
    records: list[dict[str, Any]],
    detect_mode: str,
) -> None:
    snapshot = camera.latest()
    if snapshot is None:
        print("[CALIB] save skipped: no camera0 frame yet")
        return
    robot_state = read_robot_state(controller)
    record = build_record(
        args,
        robot_state,
        snapshot,
        raw_intrinsics,
        preview_new_K,
        resolution,
        detect_mode,
    )
    sample_idx = len(records)
    overlay = draw_preview_overlay(
        snapshot.frame_bgr,
        snapshot.undistorted_bgr,
        snapshot.aruco,
        snapshot.checkerboard,
        raw_intrinsics,
        preview_new_K,
        resolution,
        sample_idx + 1,
        robot_state["joint_pos"],
        detect_mode,
        snapshot.decode_mode,
    )
    debug_overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = debug_overlay_dir / f"sample_{sample_idx:04d}.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    records.append(record)
    save_records(output, records)
    aruco = snapshot.aruco
    checkerboard = snapshot.checkerboard
    print(
        f"[CALIB] saved sample={sample_idx:04d} "
        f"mode={detect_mode} "
        f"joint_deg={np.round(np.rad2deg(robot_state['joint_pos']), 3).tolist()} "
        f"tcp_pose={np.round(robot_state['tcp_pose'], 6).tolist()} "
        f"aruco_detected={aruco['detected']} "
        f"norm_mm={None if aruco['norm_mm'] is None else round(aruco['norm_mm'], 3)} "
        f"reproj={None if aruco['reprojection_mean_px'] is None else round(aruco['reprojection_mean_px'], 4)} "
        f"checkerboard_detected={checkerboard['detected']} "
        f"checkerboard_norm_mm={None if checkerboard['norm_mm'] is None else round(checkerboard['norm_mm'], 3)} "
        f"checkerboard_reproj={None if checkerboard['reprojection_mean_px'] is None else round(checkerboard['reprojection_mean_px'], 4)} "
        f"frame_robot_delta_ms={record['frame_robot_delta_ms']:.3f} "
        f"output={output} overlay={overlay_path}"
    )


def delete_last_sample(
    output: Path,
    debug_overlay_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        print("[CALIB] delete skipped: no samples")
        return
    deleted_idx = len(records) - 1
    records.pop()
    save_records(output, records)
    overlay_path = debug_overlay_dir / f"sample_{deleted_idx:04d}.jpg"
    if overlay_path.exists():
        overlay_path.unlink()
    print(f"[CALIB] deleted sample={deleted_idx:04d}, remaining={len(records)}")


def safe_return_home_slow(
    controller: Arx5CartesianController,
    start_pose_6d: np.ndarray,
    home_pose_6d: np.ndarray,
    start_gripper_pos: float,
    home_gripper_pos: float,
    preview_time: float = 0.1,
    duration_sec: float = 4.0,
    cmd_dt: float = 0.01,
) -> None:
    print("[CALIB] safe exit requested: slowly returning to home before damping")

    start_pose_6d = np.asarray(start_pose_6d, dtype=np.float64).copy()
    home_pose_6d = np.asarray(home_pose_6d, dtype=np.float64).copy()
    start_gripper_pos = float(start_gripper_pos)
    home_gripper_pos = float(home_gripper_pos)
    duration_sec = max(float(duration_sec), cmd_dt)

    start_time = time.monotonic()
    loop_cnt = 0
    while True:
        elapsed = time.monotonic() - start_time
        alpha = min(elapsed / duration_sec, 1.0)
        alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)

        cmd_pose = (1.0 - alpha_smooth) * start_pose_6d + alpha_smooth * home_pose_6d
        cmd_gripper = (
            (1.0 - alpha_smooth) * start_gripper_pos
            + alpha_smooth * home_gripper_pos
        )

        current_timestamp = controller.get_timestamp()
        eef_cmd = EEFState()
        eef_cmd.pose_6d()[:] = cmd_pose
        eef_cmd.gripper_pos = cmd_gripper
        eef_cmd.timestamp = current_timestamp + preview_time
        controller.set_eef_cmd(eef_cmd)

        if alpha >= 1.0:
            break

        loop_cnt += 1
        target_t = start_time + loop_cnt * cmd_dt
        while time.monotonic() < target_t:
            time.sleep(0.001)

    print("[CALIB] slow return-home command finished")


def start_keyboard_collection(
    controller: Arx5CartesianController,
    camera: CameraWorker,
    raw_intrinsics: dict[str, np.ndarray],
    preview_new_K: np.ndarray,
    resolution: tuple[int, int],
    records: list[dict[str, Any]],
    output: Path,
    debug_overlay_dir: Path,
    args: argparse.Namespace,
) -> bool:
    ori_speed = args.ori_speed
    pos_speed = args.pos_speed
    gripper_speed = args.gripper_speed
    home_pose_6d = controller.get_home_pose().copy()
    home_gripper_pos = 0.0
    target_pose_6d = home_pose_6d.copy()

    target_gripper_pos = home_gripper_pos
    cmd_dt = 0.01
    preview_time = 0.1
    window_size = 5
    keyboard_queue = Queue(window_size)
    robot_config = controller.get_robot_config()
    controller.get_controller_config()

    print_help()
    print(
        f"[CALIB] teleop speed: pos_speed={pos_speed} m/s, "
        f"ori_speed={ori_speed} rad/s, gripper_speed={gripper_speed}"
    )
    print("[CALIB] Teleop tracking started.")

    key_pressed: dict[keyboard.Key | keyboard.KeyCode, bool] = {
        keyboard.Key.up: False,
        keyboard.Key.down: False,
        keyboard.Key.left: False,
        keyboard.Key.right: False,
        keyboard.Key.page_up: False,
        keyboard.Key.page_down: False,
        keyboard.KeyCode.from_char("n"): False,
        keyboard.KeyCode.from_char("m"): False,
        keyboard.KeyCode.from_char("q"): False,
        keyboard.KeyCode.from_char("a"): False,
        keyboard.KeyCode.from_char("w"): False,
        keyboard.KeyCode.from_char("s"): False,
        keyboard.KeyCode.from_char("e"): False,
        keyboard.KeyCode.from_char("d"): False,
        keyboard.KeyCode.from_char("r"): False,
        keyboard.KeyCode.from_char("f"): False,
    }
    key_lock = threading.Lock()
    command_queue: Queue[str] = Queue()
    command_keys_down: set[str] = set()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        with key_lock:
            if key in key_pressed:
                key_pressed[key] = True
            key_id = command_key_id(key)
            if key_id is not None and key_id not in command_keys_down:
                command_keys_down.add(key_id)
                event = command_event_for_id(key_id)
                if event is not None:
                    command_queue.put(event)

    def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
        with key_lock:
            if key in key_pressed:
                key_pressed[key] = False
            key_id = command_key_id(key)
            if key_id is not None:
                command_keys_down.discard(key_id)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    start_time = time.monotonic()
    loop_cnt = 0
    latest_robot_state: dict[str, Any] | None = None
    quit_requested = False
    safe_home_completed = False
    current_detect_mode = str(args.detect_mode)

    def request_safe_quit() -> bool:
        nonlocal safe_home_completed
        with key_lock:
            for key in key_pressed:
                key_pressed[key] = False
        if args.disable_safe_home_on_exit:
            print("[CALIB] safe home exit disabled; damping will run during cleanup")
            return True
        current_state = read_robot_state(controller)
        safe_return_home_slow(
            controller=controller,
            start_pose_6d=current_state["ee_pose"],
            home_pose_6d=home_pose_6d,
            start_gripper_pos=current_state["gripper_pos"],
            home_gripper_pos=home_gripper_pos,
            preview_time=preview_time,
            duration_sec=args.exit_home_duration,
            cmd_dt=cmd_dt,
        )
        safe_home_completed = True
        return True

    try:
        while not quit_requested:
            latest_robot_state = read_robot_state(controller)

            with key_lock:
                pressed_snapshot = dict(key_pressed)
            state = get_filtered_keyboard_output(pressed_snapshot, keyboard_queue)
            key_open = pressed_snapshot[keyboard.KeyCode.from_char("r")]
            key_close = pressed_snapshot[keyboard.KeyCode.from_char("f")]

            if key_open and not key_close:
                gripper_cmd = 1
            elif key_close and not key_open:
                gripper_cmd = -1
            else:
                gripper_cmd = 0

            target_pose_6d[:3] += state[:3] * pos_speed * cmd_dt
            target_pose_6d[3:] += state[3:] * ori_speed * cmd_dt
            target_gripper_pos += gripper_cmd * gripper_speed * cmd_dt
            if target_gripper_pos >= robot_config.gripper_width:
                target_gripper_pos = robot_config.gripper_width
            elif target_gripper_pos <= 0:
                target_gripper_pos = 0

            loop_cnt += 1
            while time.monotonic() < start_time + loop_cnt * cmd_dt:
                pass

            current_timestamp = controller.get_timestamp()
            eef_cmd = EEFState()
            eef_cmd.pose_6d()[:] = target_pose_6d
            eef_cmd.gripper_pos = target_gripper_pos
            eef_cmd.timestamp = current_timestamp + preview_time
            controller.set_eef_cmd(eef_cmd)

            snapshot = camera.latest()
            if not args.no_preview and snapshot is not None:
                overlay = draw_preview_overlay(
                    snapshot.frame_bgr,
                    snapshot.undistorted_bgr,
                    snapshot.aruco,
                    snapshot.checkerboard,
                    raw_intrinsics,
                    preview_new_K,
                    resolution,
                    len(records),
                    latest_robot_state["joint_pos"],
                    current_detect_mode,
                    snapshot.decode_mode,
                )
                cv2.imshow("ARX5 wrist camera0 hand-eye teleop", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    quit_requested = request_safe_quit()

            while True:
                try:
                    event = command_queue.get_nowait()
                except Empty:
                    break

                if event == "save":
                    save_current_sample(
                        args,
                        controller,
                        camera,
                        raw_intrinsics,
                        preview_new_K,
                        resolution,
                        output,
                        debug_overlay_dir,
                        records,
                        current_detect_mode,
                    )
                elif event == "delete":
                    delete_last_sample(output, debug_overlay_dir, records)
                elif event == "print":
                    latest_robot_state = read_robot_state(controller)
                    print_state(latest_robot_state, camera.latest(), current_detect_mode)
                elif event == "toggle_mode":
                    current_detect_mode = (
                        "checkerboard" if current_detect_mode == "aruco" else "aruco"
                    )
                    camera.set_detect_mode(current_detect_mode)
                    print(f"[CALIB] detect mode switched to {current_detect_mode}")
                elif event == "help":
                    print_help()
                elif event == "quit":
                    if not quit_requested:
                        quit_requested = request_safe_quit()

            schedule_lag = time.monotonic() - (start_time + loop_cnt * cmd_dt)
            if schedule_lag > cmd_dt * 2.0:
                start_time = time.monotonic()
                loop_cnt = 0

            if loop_cnt % 50 == 0 and latest_robot_state is not None:
                eef_state = latest_robot_state["ee_pose"]
                print(
                    f"Time elapsed: {time.monotonic() - start_time:.03f}s, "
                    f"x: {eef_state[0]:.03f}, y: {eef_state[1]:.03f}, z: {eef_state[2]:.03f}, "
                    f"samples: {len(records)}",
                    end="\r",
                )
    finally:
        listener.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print()
    return safe_home_completed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--gripper_fisheye_intrinsics", default=DEFAULT_INTRINSICS)
    parser.add_argument("--aruco_yaml", default=DEFAULT_ARUCO_YAML)
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--capture_fourcc", default="YUYV")
    parser.add_argument("--cap_buffer_size", type=int, default=1)
    parser.add_argument("--camera_warmup_frames", type=int, default=10)
    parser.add_argument("--model", default="L5")
    parser.add_argument("--interface", default="can1")
    parser.add_argument("--pos_speed", type=float, default=0.08)
    parser.add_argument("--ori_speed", type=float, default=0.25)
    parser.add_argument("--gripper_speed", type=float, default=0.02)
    parser.add_argument("--exit_home_duration", type=float, default=4.0)
    parser.add_argument("--disable_safe_home_on_exit", action="store_true")
    parser.add_argument("--undistort_balance", type=float, default=0.0)
    parser.add_argument(
        "--undistort_preview_mode",
        choices=["keep_k", "new_k"],
        default="keep_k",
    )
    parser.add_argument("--detect_mode", choices=["aruco", "checkerboard"], default="aruco")
    parser.add_argument("--checkerboard_xx", type=int, default=10)
    parser.add_argument("--checkerboard_yy", type=int, default=7)
    parser.add_argument("--checkerboard_square_size", type=float, default=0.025)
    parser.add_argument("--debug_camera_decode", action="store_true")
    parser.add_argument("--no_preview", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output = str(Path(args.output).expanduser().resolve())
    args.camera = str(Path(args.camera).expanduser())
    args.gripper_fisheye_intrinsics = str(
        Path(args.gripper_fisheye_intrinsics).expanduser().resolve()
    )
    args.aruco_yaml = str(Path(args.aruco_yaml).expanduser().resolve())

    np.set_printoptions(precision=4, suppress=True)
    resolution = load_resolution(args.gripper_fisheye_intrinsics)
    raw_intrinsics = load_fisheye_intrinsics(args.gripper_fisheye_intrinsics)
    raw_intrinsics = convert_fisheye_intrinsics_resolution(raw_intrinsics, resolution)
    undistort_map1, undistort_map2, preview_new_K = build_fisheye_undistort_maps(
        raw_intrinsics["K"],
        raw_intrinsics["D"],
        resolution,
        balance=args.undistort_balance,
        preview_mode=args.undistort_preview_mode,
    )
    aruco_config = load_aruco_config(args.aruco_yaml)
    tag_size_m = aruco_config["marker_size_map"].get(int(args.tag_id))
    if tag_size_m is None:
        raise KeyError(f"tag_id={args.tag_id} has no marker size in {args.aruco_yaml}")
    tag_size_m = float(tag_size_m)

    output = Path(args.output)
    debug_overlay_dir = output.parent / "debug_saved_overlays"
    records = load_existing_records(output)

    print(f"[CALIB] output={output}")
    print(f"[CALIB] camera0={args.camera}")
    print(f"[CALIB] resolution={resolution[0]}x{resolution[1]} fps={args.fps}")
    print("[CALIB] raw K=\n", raw_intrinsics["K"])
    print("[CALIB] raw D=", raw_intrinsics["D"].reshape(-1))
    print("[CALIB] undistort_preview_mode=", args.undistort_preview_mode)
    print("[CALIB] preview_new_K=\n", preview_new_K)
    print("[CALIB] frame resolution=", resolution)
    print(f"[CALIB] aruco_yaml={args.aruco_yaml} tag_id={args.tag_id} size={tag_size_m:.6f}m")
    print(
        f"[CALIB] detect_mode={args.detect_mode} "
        f"checkerboard={args.checkerboard_xx}x{args.checkerboard_yy} "
        f"square={args.checkerboard_square_size:.6f}m "
        f"undistort_balance={args.undistort_balance}"
    )
    print(f"[CALIB] model={args.model} interface={args.interface}")

    cap: cv2.VideoCapture | None = None
    camera: CameraWorker | None = None
    controller: Arx5CartesianController | None = None
    safe_home_returned = False
    try:
        cap, actual_fourcc = open_capture(args, resolution)
        camera = CameraWorker(
            cap,
            resolution,
            actual_fourcc,
            raw_intrinsics,
            undistort_map1,
            undistort_map2,
            preview_new_K,
            aruco_config,
            args.tag_id,
            tag_size_m,
            args.checkerboard_xx,
            args.checkerboard_yy,
            args.checkerboard_square_size,
            args.detect_mode,
            args.debug_camera_decode,
        )
        camera.start()
        first = wait_for_first_frame(camera)
        print(
            f"[CALIB] first camera0 frame decode={first.decode_mode} "
            f"index={first.frame_index} frame_bgr.shape={first.frame_bgr.shape} "
            f"undistorted_bgr.shape={first.undistorted_bgr.shape}"
        )

        controller = Arx5CartesianController(args.model, args.interface)
        controller.reset_to_home()
        controller.set_log_level(LogLevel.DEBUG)

        safe_home_returned = start_keyboard_collection(
            controller,
            camera,
            raw_intrinsics,
            preview_new_K,
            resolution,
            records,
            output,
            debug_overlay_dir,
            args,
        )
    except KeyboardInterrupt:
        print("\n[CALIB] KeyboardInterrupt")
    finally:
        if controller is not None:
            if safe_home_returned:
                print("[CALIB] set_to_damping after safe home return")
            else:
                print("[CALIB] set_to_damping and exit")
            try:
                controller.set_to_damping()
            except Exception as exc:
                print(f"[CALIB] set_to_damping failed: {exc}")
        if camera is not None:
            camera.stop()
        if cap is not None:
            cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"[CALIB] samples in memory: {len(records)}")
    print(f"[CALIB] pkl path: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
