#!/usr/bin/env python3
"""
Diagnose V4L2 camera formats for hand-eye calibration preview quality.

This script only opens the requested camera. It does not import the ARX SDK,
does not connect to CAN, and does not create any robot controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0"
)
DEFAULT_OUT_DIR = (
    "/home/yd/program/nyx/arx-difussion-deploy/umi-deploy/"
    "data_local/hand_eye_recalib_640/camera_format_debug"
)


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


def decode_frame(raw: np.ndarray, width: int, height: int, fourcc: str, convert_rgb: int) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw)
    fourcc_upper = str(fourcc).upper()
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
        if raw.size == width * height * 2 and ("YUYV" in fourcc_upper or "YUY2" in fourcc_upper):
            return yuyv_to_bgr(raw.reshape(height, width, 2)), "raw_yuyv_flat_to_bgr"
        decoded = try_imdecode(raw)
        if decoded is not None:
            return decoded, "compressed_imdecode_to_bgr"
        if h == height and w == width:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), "gray_to_bgr"

    if raw.ndim == 1:
        if raw.size == width * height * 2 and ("YUYV" in fourcc_upper or "YUY2" in fourcc_upper):
            return yuyv_to_bgr(raw.reshape(height, width, 2)), "raw_yuyv_1d_to_bgr"
        decoded = try_imdecode(raw)
        if decoded is not None:
            return decoded, "compressed_imdecode_to_bgr"

    raise RuntimeError(
        f"Unsupported raw frame layout shape={raw.shape} dtype={raw.dtype} "
        f"fourcc={fourcc} convert_rgb={convert_rgb}"
    )


def analyze_frame_quality(frame_bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_brightness = float(np.mean(gray))
    sharpness_laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return {
            "bad_frame": True,
            "stripe_score": float("inf"),
            "stripe_col_p95": float("inf"),
            "stripe_row_p95": 0.0,
            "stripe_high_col_fraction": 1.0,
            "mean_brightness": mean_brightness,
            "sharpness_laplacian_var": sharpness_laplacian_var,
        }

    col_edge = np.mean(np.abs(np.diff(gray, axis=1)), axis=0)
    row_edge = np.mean(np.abs(np.diff(gray, axis=0)), axis=1)
    col_p95 = float(np.percentile(col_edge, 95))
    row_p95 = float(np.percentile(row_edge, 95))
    col_median = float(np.median(col_edge))
    high_threshold = max(25.0, col_median * 3.0)
    high_fraction = float(np.mean(col_edge > high_threshold))
    stripe_score = float(col_p95 / (row_p95 + 1.0))
    bad_frame = bool(
        (col_p95 > 28.0 and stripe_score > 2.2 and high_fraction > 0.05)
        or (col_p95 > 45.0 and high_fraction > 0.10)
    )
    return {
        "bad_frame": bad_frame,
        "stripe_score": stripe_score,
        "stripe_col_p95": col_p95,
        "stripe_row_p95": row_p95,
        "stripe_high_col_fraction": high_fraction,
        "mean_brightness": mean_brightness,
        "sharpness_laplacian_var": sharpness_laplacian_var,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_raw_frames(combo_dir: Path, raw_frames: list[np.ndarray]) -> list[str]:
    if not raw_frames:
        return []
    first_shape = raw_frames[0].shape
    first_dtype = raw_frames[0].dtype
    if all(frame.shape == first_shape and frame.dtype == first_dtype for frame in raw_frames):
        np.save(str(combo_dir / "raw.npy"), np.stack(raw_frames, axis=0))
        return ["raw.npy"]
    filenames = []
    for idx, raw in enumerate(raw_frames):
        filename = f"raw_{idx:03d}.npy"
        np.save(str(combo_dir / filename), raw)
        filenames.append(filename)
    return filenames


def summarize_frame_metrics(frame_records: list[dict[str, Any]]) -> dict[str, Any]:
    decoded = [record for record in frame_records if record.get("decoded", False)]
    if not decoded:
        return {
            "decoded_count": 0,
            "bad_frame": True,
            "bad_fraction": 1.0,
            "stripe_score": float("inf"),
            "mean_brightness": None,
            "sharpness_laplacian_var": None,
        }
    bad_values = np.asarray([record["bad_frame"] for record in decoded], dtype=bool)
    return {
        "decoded_count": int(len(decoded)),
        "bad_frame": bool(np.mean(bad_values.astype(np.float64)) > 0.5),
        "bad_fraction": float(np.mean(bad_values.astype(np.float64))),
        "stripe_score": float(np.median([record["stripe_score"] for record in decoded])),
        "mean_brightness": float(np.median([record["mean_brightness"] for record in decoded])),
        "sharpness_laplacian_var": float(np.median([record["sharpness_laplacian_var"] for record in decoded])),
    }


def test_combo(
    camera: str,
    out_dir: Path,
    fmt: str,
    width: int,
    height: int,
    fps: int,
    convert_rgb: int,
    warmup_frames: int,
    save_frames: int,
) -> dict[str, Any]:
    combo_name = f"fmt_{fmt}_res_{width}x{height}_fps_{fps}_rgb_{convert_rgb}"
    combo_dir = out_dir / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if fmt != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, int(convert_rgb))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    meta: dict[str, Any] = {
        "combo_name": combo_name,
        "camera": camera,
        "requested": {
            "format": fmt,
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "convert_rgb": int(convert_rgb),
        },
        "opened": bool(cap.isOpened()),
        "frames": [],
    }
    if not cap.isOpened():
        meta["error"] = "VideoCapture failed to open"
        with open(combo_dir / "meta.json", "w") as f:
            json.dump(json_safe(meta), f, indent=2)
        return meta

    for _ in range(warmup_frames):
        cap.read()

    negotiated_fourcc = fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
    negotiated_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    negotiated_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    negotiated_fps = float(cap.get(cv2.CAP_PROP_FPS))
    negotiated_convert_rgb = float(cap.get(cv2.CAP_PROP_CONVERT_RGB))
    meta["negotiated"] = {
        "fourcc": negotiated_fourcc,
        "width": negotiated_width,
        "height": negotiated_height,
        "fps": negotiated_fps,
        "convert_rgb": negotiated_convert_rgb,
    }

    raw_frames: list[np.ndarray] = []
    for idx in range(save_frames):
        ok, raw = cap.read()
        record: dict[str, Any] = {
            "idx": int(idx),
            "read_ok": bool(ok),
            "timestamp": float(time.time()),
        }
        if ok and raw is not None:
            raw = np.ascontiguousarray(raw)
            raw_frames.append(raw)
            record["raw_shape"] = tuple(int(v) for v in raw.shape)
            record["raw_dtype"] = str(raw.dtype)
            try:
                decoded, decode_mode = decode_frame(
                    raw,
                    negotiated_width,
                    negotiated_height,
                    negotiated_fourcc,
                    int(round(negotiated_convert_rgb)),
                )
                decoded = np.ascontiguousarray(decoded)
                cv2.imwrite(str(combo_dir / f"decoded_{idx:03d}.jpg"), decoded)
                quality = analyze_frame_quality(decoded)
                record.update(
                    {
                        "decoded": True,
                        "decoded_shape": tuple(int(v) for v in decoded.shape),
                        "decoded_dtype": str(decoded.dtype),
                        "decode_mode": decode_mode,
                        **quality,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "decoded": False,
                        "decode_error": repr(exc),
                        "bad_frame": True,
                        "stripe_score": float("inf"),
                    }
                )
        meta["frames"].append(record)

    cap.release()
    meta["raw_files"] = save_raw_frames(combo_dir, raw_frames)
    meta["summary"] = summarize_frame_metrics(meta["frames"])
    with open(combo_dir / "meta.json", "w") as f:
        json.dump(json_safe(meta), f, indent=2)
    return meta


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n[CAM_DIAG] Summary sorted by usable frames and stripe_score")
    header = (
        "rank  bad  stripe   bad_frac  sharpness  meanY  req_fmt  req_res   "
        "req_fps  rgb  neg_fourcc  neg_res   neg_fps  decode"
    )
    print(header)
    for rank, row in enumerate(rows):
        print(
            f"{rank:>4}  "
            f"{str(row['bad_frame']):>5}  "
            f"{row['stripe_score']:>7.3f}  "
            f"{row['bad_fraction']:>8.2f}  "
            f"{row['sharpness_laplacian_var'] if row['sharpness_laplacian_var'] is not None else None!s:>9}  "
            f"{row['mean_brightness'] if row['mean_brightness'] is not None else None!s:>5}  "
            f"{row['requested_format']:>7}  "
            f"{row['requested_width']}x{row['requested_height']:<4}  "
            f"{row['requested_fps']:>7}  "
            f"{row['requested_convert_rgb']:>3}  "
            f"{row['negotiated_fourcc']:>10}  "
            f"{row['negotiated_width']}x{row['negotiated_height']:<4}  "
            f"{row['negotiated_fps']:>7.3f}  "
            f"{row['decode_mode']}"
        )


def build_summary_rows(metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for meta in metas:
        requested = meta.get("requested", {})
        negotiated = meta.get("negotiated", {})
        summary = meta.get("summary", {})
        first_decoded = next((frame for frame in meta.get("frames", []) if frame.get("decoded")), {})
        row = {
            "combo_name": meta.get("combo_name"),
            "opened": bool(meta.get("opened", False)),
            "requested_format": requested.get("format"),
            "requested_width": requested.get("width"),
            "requested_height": requested.get("height"),
            "requested_fps": requested.get("fps"),
            "requested_convert_rgb": requested.get("convert_rgb"),
            "negotiated_fourcc": negotiated.get("fourcc"),
            "negotiated_width": negotiated.get("width"),
            "negotiated_height": negotiated.get("height"),
            "negotiated_fps": negotiated.get("fps", 0.0),
            "negotiated_convert_rgb": negotiated.get("convert_rgb"),
            "decoded_count": summary.get("decoded_count", 0),
            "bad_frame": summary.get("bad_frame", True),
            "bad_fraction": summary.get("bad_fraction", 1.0),
            "stripe_score": summary.get("stripe_score", float("inf")),
            "mean_brightness": summary.get("mean_brightness"),
            "sharpness_laplacian_var": summary.get("sharpness_laplacian_var"),
            "raw_shape": first_decoded.get("raw_shape"),
            "raw_dtype": first_decoded.get("raw_dtype"),
            "decoded_shape": first_decoded.get("decoded_shape"),
            "decoded_dtype": first_decoded.get("decoded_dtype"),
            "decode_mode": first_decoded.get("decode_mode"),
            "combo_dir": meta.get("combo_name"),
        }
        rows.append(row)
    rows.sort(key=lambda row: (bool(row["bad_frame"]), float(row["stripe_score"])))
    return rows


def write_summary_files(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    with open(out_dir / "summary.json", "w") as f:
        json.dump(json_safe(rows), f, indent=2)
    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(json_safe(row))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--save_frames", type=int, default=10)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = ["auto", "MJPG", "YUYV"]
    resolutions = [(640, 480), (320, 240)]
    fps_values = [5, 10, 15, 30]
    convert_rgb_values = [1, 0]

    print(f"[CAM_DIAG] camera={args.camera}")
    print(f"[CAM_DIAG] out_dir={out_dir}")
    print("[CAM_DIAG] testing camera0 only")

    metas = []
    for fmt in formats:
        for width, height in resolutions:
            for fps in fps_values:
                for convert_rgb in convert_rgb_values:
                    print(
                        "[CAM_DIAG] test "
                        f"format={fmt} res={width}x{height} fps={fps} convert_rgb={convert_rgb}"
                    )
                    meta = test_combo(
                        args.camera,
                        out_dir,
                        fmt,
                        width,
                        height,
                        fps,
                        convert_rgb,
                        int(args.warmup_frames),
                        int(args.save_frames),
                    )
                    summary = meta.get("summary", {})
                    negotiated = meta.get("negotiated", {})
                    print(
                        "[CAM_DIAG] result "
                        f"opened={meta.get('opened')} "
                        f"neg_fourcc={negotiated.get('fourcc')} "
                        f"neg_res={negotiated.get('width')}x{negotiated.get('height')} "
                        f"neg_fps={negotiated.get('fps')} "
                        f"bad_frame={summary.get('bad_frame')} "
                        f"stripe_score={summary.get('stripe_score')}"
                    )
                    metas.append(meta)

    rows = build_summary_rows(metas)
    write_summary_files(out_dir, rows)
    print_summary(rows)
    print(f"\n[CAM_DIAG] wrote summary to {out_dir / 'summary.json'} and {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
