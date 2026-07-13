#!/usr/bin/env python3
"""
Diagnose hand-eye dataset consistency, clusters, outliers, and timing metadata.

This script is diagnostic only. It does not modify runtime calibration, URDF,
motor zeros, or matrix conventions.
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR))
os.chdir(ROOT_DIR)

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from calibrate_compare_variants import (
    calibrate_hand_eye,
    evaluate_hand_eye,
    filter_detections,
    load_detections,
    stats,
)
from umi.common.cv_util import parse_aruco_config, parse_fisheye_intrinsics


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PKL = REPO_ROOT / "umi-deploy/data_local/hand_eye_recalib_640/hand_eye_calib_300_350.pkl"
DEFAULT_INTR = REPO_ROOT / "umi-deploy/data_local/calibration/cam0_sensor_intrinsics.json"
DEFAULT_ARUCO = REPO_ROOT / "umi-deploy/data_local/hand_eye_tags/aruco_config_tag12_147mm.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "umi-deploy/data_local/hand_eye_recalib_640/dataset_consistency_diagnosis"


def distance_status(norm_mm):
    if 300.0 <= norm_mm <= 350.0:
        return "preferred"
    if 250.0 <= norm_mm < 300.0:
        return "contrast"
    if norm_mm < 250.0:
        return "too_close"
    return "too_far"


def load_hand_eye(path):
    payload = json.load(open(path, "r"))
    return (
        np.asarray(payload["tx_base2world"], dtype=np.float64),
        np.asarray(payload["tx_gripper2camera"], dtype=np.float64),
    )


def evaluate_per_sample(detections, tx_world_base, tx_camera_gripper):
    trans, rot = evaluate_hand_eye(detections, tx_world_base, tx_camera_gripper, "tcp_pose")
    return {
        int(det["idx"]): {
            "trans_mm": float(t),
            "rot_deg": float(r),
        }
        for det, t, r in zip(detections, trans, rot)
    }


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def group_ranges(n):
    default = [(0, 8), (9, 15), (16, 24), (25, n - 1)]
    return [(a, b) for a, b in default if a < n and b >= a]


def dets_in_range(detections, start, end):
    return [det for det in detections if start <= int(det["idx"]) <= end]


def summarize_group(name, detections, residual_by_idx):
    vals = [residual_by_idx[int(det["idx"])]["trans_mm"] for det in detections if int(det["idx"]) in residual_by_idx]
    rots = [residual_by_idx[int(det["idx"])]["rot_deg"] for det in detections if int(det["idx"]) in residual_by_idx]
    if not vals:
        return {"group": name, "count": 0}
    vals = np.asarray(vals, dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)
    return {
        "group": name,
        "count": int(vals.size),
        "trans_mean": float(vals.mean()),
        "trans_median": float(np.median(vals)),
        "trans_rms": float(np.sqrt(np.mean(vals * vals))),
        "trans_max": float(vals.max()),
        "rot_mean": float(rots.mean()),
        "rot_median": float(np.median(rots)),
        "rot_max": float(rots.max()),
    }


def write_samples_csv(path, samples, detections, residual_by_idx):
    det_by_idx = {int(det["idx"]): det for det in detections}
    fieldnames = [
        "idx",
        "z_mm",
        "norm_mm",
        "reproj",
        "status",
        "frame_robot_delta_ms",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_rx",
        "tcp_ry",
        "tcp_rz",
        "ee_x",
        "ee_y",
        "ee_z",
        "joint_state",
        "trans_mm",
        "rot_deg",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(samples):
            det = det_by_idx.get(idx)
            residual = residual_by_idx.get(idx, {})
            tcp = np.asarray(sample.get("tcp_pose", np.full(6, np.nan)), dtype=np.float64).reshape(-1)
            ee = np.asarray(sample.get("ee_pose", np.full(6, np.nan)), dtype=np.float64).reshape(-1)
            joint = np.asarray(sample.get("joint_state", []), dtype=np.float64).reshape(-1)
            norm_mm = None if det is None else det["norm_mm"]
            writer.writerow(
                {
                    "idx": idx,
                    "z_mm": None if det is None else det["z_mm"],
                    "norm_mm": norm_mm,
                    "reproj": None if det is None else det["reprojection_mean_px"],
                    "status": "no_tag" if det is None else distance_status(norm_mm),
                    "frame_robot_delta_ms": sample.get("frame_robot_delta_ms", sample.get("time_delta_ms")),
                    "tcp_x": tcp[0] if tcp.size >= 6 else None,
                    "tcp_y": tcp[1] if tcp.size >= 6 else None,
                    "tcp_z": tcp[2] if tcp.size >= 6 else None,
                    "tcp_rx": tcp[3] if tcp.size >= 6 else None,
                    "tcp_ry": tcp[4] if tcp.size >= 6 else None,
                    "tcp_rz": tcp[5] if tcp.size >= 6 else None,
                    "ee_x": ee[0] if ee.size >= 6 else None,
                    "ee_y": ee[1] if ee.size >= 6 else None,
                    "ee_z": ee[2] if ee.size >= 6 else None,
                    "joint_state": " ".join(f"{x:.8f}" for x in joint),
                    "trans_mm": residual.get("trans_mm"),
                    "rot_deg": residual.get("rot_deg"),
                }
            )


def write_simple_plot(path, x, y, xlabel, ylabel, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return False
    plt.figure(figsize=(7, 4))
    plt.scatter(x[mask], y[mask], s=24)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def leave_one_out(variant_name, detections, output_dir):
    rows = []
    if len(detections) < 4:
        return rows
    tx_world_base, tx_camera_gripper = calibrate_hand_eye(detections, "tcp_pose")
    base_trans, base_rot = evaluate_hand_eye(detections, tx_world_base, tx_camera_gripper, "tcp_pose")
    base_median = float(np.median(base_trans))
    base_by_idx = {
        int(det["idx"]): float(trans)
        for det, trans in zip(detections, base_trans)
    }
    for det in detections:
        kept = [d for d in detections if int(d["idx"]) != int(det["idx"])]
        if len(kept) < 3:
            continue
        tx_wb, tx_cg = calibrate_hand_eye(kept, "tcp_pose")
        trans, rot = evaluate_hand_eye(kept, tx_wb, tx_cg, "tcp_pose")
        median = float(np.median(trans))
        improvement = base_median - median
        reason = "unknown"
        if det["reprojection_mean_px"] > 4.0:
            reason = "bad PnP"
        elif base_by_idx[int(det["idx"])] > base_median * 1.5:
            reason = "geometric outlier"
        rows.append(
            {
                "variant": variant_name,
                "mode": "leave_one_out",
                "removed": int(det["idx"]),
                "removed_group": "",
                "base_median_mm": base_median,
                "new_median_mm": median,
                "improvement_mm": improvement,
                "new_rms_mm": float(np.sqrt(np.mean(trans * trans))),
                "new_max_mm": float(np.max(trans)),
                "suspected_reason": reason,
            }
        )

    sorted_base = sorted(base_by_idx.items(), key=lambda item: item[1], reverse=True)
    for k in range(1, min(5, len(sorted_base) - 2) + 1):
        remove = {idx for idx, _ in sorted_base[:k]}
        kept = [d for d in detections if int(d["idx"]) not in remove]
        if len(kept) < 3:
            continue
        tx_wb, tx_cg = calibrate_hand_eye(kept, "tcp_pose")
        trans, rot = evaluate_hand_eye(kept, tx_wb, tx_cg, "tcp_pose")
        median = float(np.median(trans))
        rows.append(
            {
                "variant": variant_name,
                "mode": f"drop_top_{k}_residual",
                "removed": " ".join(str(x) for x in sorted(remove)),
                "removed_group": "",
                "base_median_mm": base_median,
                "new_median_mm": median,
                "improvement_mm": base_median - median,
                "new_rms_mm": float(np.sqrt(np.mean(trans * trans))),
                "new_max_mm": float(np.max(trans)),
                "suspected_reason": "top residual group",
            }
        )

    n = max(int(det["idx"]) for det in detections) + 1
    for group_idx, (start, end) in enumerate(group_ranges(n)):
        remove = {int(det["idx"]) for det in detections if start <= int(det["idx"]) <= end}
        kept = [d for d in detections if int(d["idx"]) not in remove]
        if len(remove) == 0 or len(kept) < 3:
            continue
        tx_wb, tx_cg = calibrate_hand_eye(kept, "tcp_pose")
        trans, rot = evaluate_hand_eye(kept, tx_wb, tx_cg, "tcp_pose")
        median = float(np.median(trans))
        rows.append(
            {
                "variant": variant_name,
                "mode": "drop_index_group",
                "removed": " ".join(str(x) for x in sorted(remove)),
                "removed_group": f"group_{group_idx}_{start}_{end}",
                "base_median_mm": base_median,
                "new_median_mm": median,
                "improvement_mm": base_median - median,
                "new_rms_mm": float(np.sqrt(np.mean(trans * trans))),
                "new_max_mm": float(np.max(trans)),
                "suspected_reason": "possible cluster/tag movement" if base_median - median > 10.0 else "unknown",
            }
        )
    return rows


def group_cross_validation(detections, n_samples):
    groups = []
    for group_idx, (start, end) in enumerate(group_ranges(n_samples)):
        dets = dets_in_range(detections, start, end)
        groups.append((f"group_{group_idx}_{start}_{end}", start, end, dets))

    rows = []
    for train_name, _, _, train_dets in groups:
        if len(train_dets) < 3:
            continue
        tx_wb, tx_cg = calibrate_hand_eye(train_dets, "tcp_pose")
        for val_name, _, _, val_dets in groups:
            if not val_dets:
                continue
            trans, rot = evaluate_hand_eye(val_dets, tx_wb, tx_cg, "tcp_pose")
            rows.append(
                {
                    "train_group": train_name,
                    "val_group": val_name,
                    "train_n": len(train_dets),
                    "val_n": len(val_dets),
                    "trans_mean": float(trans.mean()),
                    "trans_median": float(np.median(trans)),
                    "trans_rms": float(np.sqrt(np.mean(trans * trans))),
                    "trans_max": float(trans.max()),
                    "rot_mean": float(rot.mean()),
                    "rot_median": float(np.median(rot)),
                    "rot_max": float(rot.max()),
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default=str(DEFAULT_PKL))
    parser.add_argument("--intr_json", default=str(DEFAULT_INTR))
    parser.add_argument("--aruco_yaml", default=str(DEFAULT_ARUCO))
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--hand_eye_json", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max_reproj_px", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pickle.load(open(args.pkl, "rb"))
    raw_intr = parse_fisheye_intrinsics(json.load(open(args.intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(args.aruco_yaml, "r")))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"pkl={args.pkl}")
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"aruco_yaml={args.aruco_yaml}")

    detections = load_detections(samples, "raw", raw_intr, aruco_config, args.tag_id, marker_size_m)
    if len(detections) < 3:
        raise ValueError(f"Need at least 3 tag detections, got {len(detections)}")

    if args.hand_eye_json:
        tx_world_base, tx_camera_gripper = load_hand_eye(args.hand_eye_json)
        hand_eye_source = args.hand_eye_json
    else:
        tx_world_base, tx_camera_gripper = calibrate_hand_eye(detections, "tcp_pose")
        hand_eye_source = "calibrated_raw_all_for_diagnosis"
    residual_by_idx = evaluate_per_sample(detections, tx_world_base, tx_camera_gripper)

    write_samples_csv(output_dir / "samples_diagnostics.csv", samples, detections, residual_by_idx)

    diag_rows = []
    for det in detections:
        idx = int(det["idx"])
        residual = residual_by_idx[idx]
        sample = det["sample"]
        diag_rows.append(
            {
                "idx": idx,
                "norm_mm": det["norm_mm"],
                "z_mm": det["z_mm"],
                "reproj": det["reprojection_mean_px"],
                "frame_robot_delta_ms": sample.get("frame_robot_delta_ms", sample.get("time_delta_ms", np.nan)),
                "tcp_x": float(np.asarray(sample["tcp_pose"])[0]),
                "tcp_y": float(np.asarray(sample["tcp_pose"])[1]),
                "tcp_z": float(np.asarray(sample["tcp_pose"])[2]),
                "trans_mm": residual["trans_mm"],
                "rot_deg": residual["rot_deg"],
            }
        )

    idxs = [r["idx"] for r in diag_rows]
    trans = [r["trans_mm"] for r in diag_rows]
    norms = [r["norm_mm"] for r in diag_rows]
    reproj = [r["reproj"] for r in diag_rows]
    deltas = [r["frame_robot_delta_ms"] for r in diag_rows]
    tcp_x = [r["tcp_x"] for r in diag_rows]
    tcp_y = [r["tcp_y"] for r in diag_rows]
    tcp_z = [r["tcp_z"] for r in diag_rows]

    write_simple_plot(output_dir / "residual_by_index.png", idxs, trans, "sample index", "translation residual mm", "Residual by Index")
    write_simple_plot(output_dir / "residual_by_distance.png", norms, trans, "norm_mm", "translation residual mm", "Residual by Distance")
    write_simple_plot(output_dir / "residual_by_reproj.png", reproj, trans, "reprojection mean px", "translation residual mm", "Residual by Reprojection")
    write_simple_plot(output_dir / "residual_by_time_delta.png", deltas, trans, "frame_robot_delta_ms", "translation residual mm", "Residual by Time Delta")
    write_simple_plot(output_dir / "norm_mm_by_index.png", idxs, norms, "sample index", "norm_mm", "Distance by Index")
    write_simple_plot(output_dir / "z_mm_by_index.png", idxs, [r["z_mm"] for r in diag_rows], "sample index", "z_mm", "Z by Index")

    group_rows = []
    for group_idx, (start, end) in enumerate(group_ranges(len(samples))):
        group_dets = dets_in_range(detections, start, end)
        group_rows.append(summarize_group(f"group_{group_idx}_{start}_{end}", group_dets, residual_by_idx))
    write_csv(output_dir / "group_residual_summary.csv", group_rows)

    outlier_rows = []
    variants = {
        "raw_all": (None, None),
        "raw_250_350": (250.0, 350.0),
        "raw_300_350": (300.0, 350.0),
    }
    for name, (min_mm, max_mm) in variants.items():
        used, _ = filter_detections(detections, min_mm, max_mm, args.max_reproj_px)
        outlier_rows.extend(leave_one_out(name, used, output_dir))
    write_csv(output_dir / "outlier_candidates.csv", outlier_rows)

    cross_rows = group_cross_validation(detections, len(samples))
    write_csv(output_dir / "group_cross_validation.csv", cross_rows)

    possible_boundaries = []
    valid_group_rows = [r for r in group_rows if r.get("count", 0) > 0]
    for prev, cur in zip(valid_group_rows[:-1], valid_group_rows[1:]):
        if abs(prev["trans_median"] - cur["trans_median"]) > 20.0:
            possible_boundaries.append(f"{prev['group']} -> {cur['group']}")

    top_suspicious = sorted(diag_rows, key=lambda r: r["trans_mm"], reverse=True)[:10]
    report = {
        "pkl": args.pkl,
        "hand_eye_source": hand_eye_source,
        "samples_total": len(samples),
        "samples_with_tag": len(detections),
        "translation_stats_mm": stats(np.asarray(trans, dtype=np.float64)),
        "rotation_stats_deg": stats(np.asarray([r["rot_deg"] for r in diag_rows], dtype=np.float64)),
        "correlations": {
            "residual_vs_norm_mm": safe_corr(trans, norms),
            "residual_vs_reproj": safe_corr(trans, reproj),
            "residual_vs_frame_robot_delta_ms": safe_corr(trans, deltas),
            "residual_vs_tcp_x": safe_corr(trans, tcp_x),
            "residual_vs_tcp_y": safe_corr(trans, tcp_y),
            "residual_vs_tcp_z": safe_corr(trans, tcp_z),
        },
        "group_residual_summary": group_rows,
        "possible_cluster_boundaries": possible_boundaries,
        "top_suspicious_samples": top_suspicious,
        "conclusion": "possible cluster/tag movement" if possible_boundaries else "no obvious index cluster from median group residuals",
    }
    json.dump(report, open(output_dir / "diagnosis_report.json", "w"), indent=2)

    with open(output_dir / "diagnosis_report.txt", "w") as f:
        f.write(json.dumps(report, indent=2))
        f.write("\n")

    print("DATASET_CONSISTENCY_SUMMARY")
    print(f"samples_total={len(samples)} samples_with_tag={len(detections)}")
    print(f"translation_stats_mm={report['translation_stats_mm']}")
    print(f"rotation_stats_deg={report['rotation_stats_deg']}")
    print(f"correlations={report['correlations']}")
    print("group_residual_summary:")
    for row in group_rows:
        print(f"  {row}")
    print(f"possible_cluster_boundaries={possible_boundaries}")
    print("top_suspicious_samples:")
    for row in top_suspicious[:8]:
        reason = "unknown"
        if row["reproj"] > args.max_reproj_px:
            reason = "bad PnP"
        elif np.isfinite(row["frame_robot_delta_ms"]) and row["frame_robot_delta_ms"] > 50.0:
            reason = "time sync"
        elif row["trans_mm"] > report["translation_stats_mm"]["median"] * 1.5:
            reason = "geometric outlier / possible tag movement"
        print(
            f"  sample={row['idx']:03d} trans={row['trans_mm']:.3f}mm rot={row['rot_deg']:.3f}deg "
            f"norm={row['norm_mm']:.3f}mm reproj={row['reproj']:.4f}px "
            f"delta_ms={row['frame_robot_delta_ms']} reason={reason}"
        )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
