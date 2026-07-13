#!/usr/bin/env python3
"""
Diagnose ARX5 FK joint order, FK sensitivity, and possible joint zero offsets.

This script does not modify the URDF. It parses the configured URDF chain from
base_link to eef_link and uses that chain for local FK diagnostics. Input pickle
samples must contain joint_state for sensitivity and zero-offset optimization.
"""

import argparse
import json
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import mat_to_pose
from umi.common.pose_util import pose_to_mat


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPO_ROOT / "umi-deploy/arx5-sdk/models/L5_assembly.urdf"
DEFAULT_PKL = REPO_ROOT / "umi-deploy/data_local/hand_eye_recalib_640/hand_eye_calib.pkl"
DEFAULT_INTR = REPO_ROOT / "umi-deploy/data_local/calibration/cam0_sensor_intrinsics.json"
DEFAULT_ARUCO = REPO_ROOT / "umi-deploy/data_local/hand_eye_tags/aruco_config_tag12_147mm.yaml"


def parse_xyz(value):
    if value is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray([float(x) for x in value.split()], dtype=np.float64)


def tx_from_xyz_rpy(xyz, rpy):
    tx = np.eye(4, dtype=np.float64)
    tx[:3, 3] = xyz
    tx[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return tx


def axis_angle_tx(axis, angle):
    tx = np.eye(4, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Joint axis has zero norm")
    tx[:3, :3] = R.from_rotvec(axis / norm * angle).as_matrix()
    return tx


def parse_urdf_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    child_to_joint = {}
    parent_to_joints = {}
    for joint_el in root.findall("joint"):
        name = joint_el.attrib["name"]
        joint_type = joint_el.attrib.get("type", "fixed")
        parent = joint_el.find("parent").attrib["link"]
        child = joint_el.find("child").attrib["link"]
        origin_el = joint_el.find("origin")
        xyz = parse_xyz(origin_el.attrib.get("xyz") if origin_el is not None else None)
        rpy = parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None)
        axis_el = joint_el.find("axis")
        axis = parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None)
        mimic_el = joint_el.find("mimic")
        mimic = None
        if mimic_el is not None:
            mimic = {
                "joint": mimic_el.attrib.get("joint"),
                "multiplier": float(mimic_el.attrib.get("multiplier", "1")),
                "offset": float(mimic_el.attrib.get("offset", "0")),
            }
        joint = {
            "name": name,
            "type": joint_type,
            "parent": parent,
            "child": child,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis": axis,
            "mimic": mimic,
        }
        joints.append(joint)
        child_to_joint[child] = joint
        parent_to_joints.setdefault(parent, []).append(joint)
    return joints, child_to_joint, parent_to_joints


def find_chain(parent_to_joints, base_link, eef_link):
    stack = [(base_link, [])]
    visited = set()
    while stack:
        link, chain = stack.pop()
        if link == eef_link:
            return chain
        if link in visited:
            continue
        visited.add(link)
        for joint in parent_to_joints.get(link, []):
            stack.append((joint["child"], chain + [joint]))
    raise ValueError(f"No chain from {base_link} to {eef_link}")


def fk(chain, joint_state):
    q = np.asarray(joint_state, dtype=np.float64).reshape(-1)
    active_idx = 0
    tx = np.eye(4, dtype=np.float64)
    for joint in chain:
        tx = tx @ tx_from_xyz_rpy(joint["origin_xyz"], joint["origin_rpy"])
        if joint["type"] in ("revolute", "continuous"):
            tx = tx @ axis_angle_tx(joint["axis"], q[active_idx])
            active_idx += 1
        elif joint["type"] == "prismatic":
            move = np.eye(4, dtype=np.float64)
            move[:3, 3] = joint["axis"] * q[active_idx]
            tx = tx @ move
            active_idx += 1
        elif joint["type"] == "fixed":
            pass
        else:
            raise ValueError(f"Unsupported joint type {joint['type']} for {joint['name']}")
    if active_idx != q.size:
        raise ValueError(f"FK consumed {active_idx} joints but joint_state has {q.size}")
    return tx


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(), np.median(values), values.max()


def print_error_summary(name, trans_mm, rot_deg):
    tm, tmed, tmax = summarize(trans_mm)
    rm, rmed, rmax = summarize(rot_deg)
    print(
        f"{name}: "
        f"trans mean={tm:.6g} mm median={tmed:.6g} mm max={tmax:.6g} mm; "
        f"rot mean={rm:.6g} deg median={rmed:.6g} deg max={rmax:.6g} deg"
    )


def ee_pose_to_mat(ee_pose):
    ee_pose = np.asarray(ee_pose, dtype=np.float64).reshape(6)
    tx = np.eye(4, dtype=np.float64)
    tx[:3, 3] = ee_pose[:3]
    tx[:3, :3] = R.from_euler("xyz", ee_pose[3:]).as_matrix()
    return tx


def ee_pose_to_tcp_mat(ee_pose):
    tx_ee_tcp = np.eye(4, dtype=np.float64)
    tx_ee_tcp[:3, :3] = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ],
        dtype=np.float64,
    )
    return ee_pose_to_mat(ee_pose) @ tx_ee_tcp


def pose_error(tx_a, tx_b):
    tx_err = np.linalg.inv(tx_a) @ tx_b
    trans_mm = np.linalg.norm(tx_err[:3, 3]) * 1000.0
    rot_deg = np.rad2deg(R.from_matrix(tx_err[:3, :3]).magnitude())
    return trans_mm, rot_deg


def transform_from_rvec_tvec(rvec, tvec):
    tx = np.eye(4, dtype=np.float64)
    tx[:3, :3] = R.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()
    tx[:3, 3] = np.asarray(tvec).reshape(3)
    return tx


def detect_tag_samples(samples, intr_json, aruco_yaml, tag_id):
    raw_intr = parse_fisheye_intrinsics(json.load(open(intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, "r")))
    detected = []
    for idx, sample in enumerate(samples):
        if "aruco" in sample and sample["aruco"] is not None:
            aruco = sample["aruco"]
            if int(aruco.get("tag_id", tag_id)) == tag_id:
                detected.append(
                    {
                        "idx": idx,
                        "tx_camera_world": transform_from_rvec_tvec(
                            aruco["rvec"],
                            aruco["tvec"],
                        ),
                    }
                )
                continue
        img = sample["img"]
        intr = convert_fisheye_intrinsics_resolution(raw_intr, img.shape[:2][::-1])
        tag_dict = detect_localize_aruco_tags(
            img,
            aruco_config["aruco_dict"],
            aruco_config["marker_size_map"],
            intr,
        )
        if tag_id not in tag_dict:
            continue
        tag = tag_dict[tag_id]
        detected.append(
            {
                "idx": idx,
                "tx_camera_world": transform_from_rvec_tvec(tag["rvec"], tag["tvec"]),
            }
        )
    return detected


def calibrate_and_score(chain, samples, detections, offsets, signs=None):
    offsets = np.asarray(offsets, dtype=np.float64).reshape(6)
    if signs is None:
        signs = np.ones(6, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64).reshape(6)
    r_world2cam = []
    t_world2cam = []
    r_base2gripper = []
    t_base2gripper = []
    tx_base_grippers = []
    for det in detections:
        sample = samples[det["idx"]]
        joint_state = np.asarray(sample["joint_state"], dtype=np.float64).reshape(6)
        tx_base_gripper = fk(chain, joint_state * signs + offsets)
        tx_gripper_base = np.linalg.inv(tx_base_gripper)
        tx_base_grippers.append(tx_base_gripper)
        r_world2cam.append(R.from_matrix(det["tx_camera_world"][:3, :3]).as_rotvec())
        t_world2cam.append(det["tx_camera_world"][:3, 3])
        tv_gripper_base = mat_to_pose(tx_gripper_base)
        r_base2gripper.append(tv_gripper_base[3:])
        t_base2gripper.append(tv_gripper_base[:3])

    r_b2w, t_b2w, r_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        R_world2cam=r_world2cam,
        t_world2cam=t_world2cam,
        R_base2gripper=r_base2gripper,
        t_base2gripper=t_base2gripper,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    tx_world_base = np.eye(4, dtype=np.float64)
    tx_world_base[:3, :3] = r_b2w
    tx_world_base[:3, 3] = t_b2w.squeeze()
    tx_camera_gripper = np.eye(4, dtype=np.float64)
    tx_camera_gripper[:3, :3] = r_g2c
    tx_camera_gripper[:3, 3] = t_g2c.squeeze()

    trans_mm = []
    rot_deg = []
    for det, tx_base_gripper in zip(detections, tx_base_grippers):
        tx_gripper_base = np.linalg.inv(tx_base_gripper)
        left = det["tx_camera_world"] @ tx_world_base
        right = tx_camera_gripper @ tx_gripper_base
        tx_err = np.linalg.inv(left) @ right
        trans_mm.append(np.linalg.norm(tx_err[:3, 3]) * 1000.0)
        rot_deg.append(np.rad2deg(R.from_matrix(tx_err[:3, :3]).magnitude()))
    return np.asarray(trans_mm), np.asarray(rot_deg)


def fk_self_consistency(chain, samples):
    fk_ee_trans = []
    fk_ee_rot = []
    tcp_trans = []
    tcp_rot = []
    per_sample = []
    for idx, sample in enumerate(samples):
        tx_fk = fk(chain, np.asarray(sample["joint_state"], dtype=np.float64).reshape(6))
        tx_saved_ee = ee_pose_to_mat(sample["ee_pose"])
        tx_saved_tcp = pose_to_mat(np.asarray(sample["tcp_pose"], dtype=np.float64).reshape(6))
        tx_ee2tcp = ee_pose_to_tcp_mat(sample["ee_pose"])

        e_fk_t, e_fk_r = pose_error(tx_fk, tx_saved_ee)
        e_tcp_t, e_tcp_r = pose_error(tx_ee2tcp, tx_saved_tcp)
        fk_ee_trans.append(e_fk_t)
        fk_ee_rot.append(e_fk_r)
        tcp_trans.append(e_tcp_t)
        tcp_rot.append(e_tcp_r)
        per_sample.append((idx, e_fk_t, e_fk_r, e_tcp_t, e_tcp_r))

    print("fk_self_consistency:")
    print_error_summary("  fk_vs_saved_ee", fk_ee_trans, fk_ee_rot)
    print_error_summary("  tcp_vs_ee2tcp", tcp_trans, tcp_rot)
    print("  worst_fk_vs_saved_ee:")
    for idx, trans, rot, tcp_t, tcp_r in sorted(per_sample, key=lambda x: x[1], reverse=True)[:5]:
        print(
            f"    sample={idx:03d} fk_trans={trans:.6g} mm fk_rot={rot:.6g} deg "
            f"tcp_trans={tcp_t:.6g} mm tcp_rot={tcp_r:.6g} deg"
        )
    print("  worst_tcp_vs_ee2tcp:")
    for idx, trans, rot, tcp_t, tcp_r in sorted(per_sample, key=lambda x: x[3], reverse=True)[:5]:
        print(
            f"    sample={idx:03d} tcp_trans={tcp_t:.6g} mm tcp_rot={tcp_r:.6g} deg "
            f"fk_trans={trans:.6g} mm fk_rot={rot:.6g} deg"
        )


def run_bounded_offset_optimization(chain, samples, detections, bound_deg, multistart, maxiter, seed):
    rng = np.random.default_rng(seed)
    bound_rad = np.deg2rad(bound_deg)
    bounds = [(-bound_rad, bound_rad)] * 6
    starts = [np.zeros(6, dtype=np.float64)]
    for _ in range(max(0, multistart - 1)):
        starts.append(rng.uniform(-bound_rad, bound_rad, size=6))

    def objective(offsets):
        trans_mm, _ = calibrate_and_score(chain, samples, detections, offsets)
        return float(np.sqrt(np.mean(trans_mm * trans_mm)))

    best = None
    for start_idx, start in enumerate(starts):
        opt = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        if best is None or opt.fun < best.fun:
            best = opt
        print(
            f"    start={start_idx:02d} success={opt.success} "
            f"rms_trans={opt.fun:.6g} offsets_deg={np.rad2deg(opt.x).round(6).tolist()}"
        )
    return best


def run_sign_flip_diagnostics(chain, samples, detections, max_results=10):
    results = []
    for mask in range(64):
        signs = np.ones(6, dtype=np.float64)
        for i in range(6):
            if mask & (1 << i):
                signs[i] = -1.0
        trans_mm, rot_deg = calibrate_and_score(chain, samples, detections, np.zeros(6), signs=signs)
        results.append(
            {
                "mask": mask,
                "signs": signs,
                "trans_mean": float(np.mean(trans_mm)),
                "trans_median": float(np.median(trans_mm)),
                "trans_max": float(np.max(trans_mm)),
                "rot_mean": float(np.mean(rot_deg)),
                "rot_median": float(np.median(rot_deg)),
                "rot_max": float(np.max(rot_deg)),
            }
        )
    results.sort(key=lambda r: (r["trans_median"], r["trans_mean"]))
    print("sign_flip_diagnostics_top:")
    for r in results[:max_results]:
        flipped = [i for i, s in enumerate(r["signs"]) if s < 0]
        print(
            f"  mask={r['mask']:02d} flipped={flipped} signs={r['signs'].astype(int).tolist()} "
            f"trans mean={r['trans_mean']:.6g} mm median={r['trans_median']:.6g} mm max={r['trans_max']:.6g} mm; "
            f"rot mean={r['rot_mean']:.6g} deg median={r['rot_median']:.6g} deg max={r['rot_max']:.6g} deg"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--base_link", default="ARXR5_arm_only_no_gray_base_base_link_arm")
    parser.add_argument("--eef_link", default="DAS_Controller_V3_with_flange_link_Flange")
    parser.add_argument("--pkl", default=str(DEFAULT_PKL))
    parser.add_argument("--intr_json", default=str(DEFAULT_INTR))
    parser.add_argument("--aruco_yaml", default=str(DEFAULT_ARUCO))
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--delta_deg", type=float, default=0.5)
    parser.add_argument("--optimize_offsets", action="store_true")
    parser.add_argument(
        "--bounded_offset_bound_deg",
        type=float,
        action="append",
        default=[],
        help="Run bounded zero-offset optimization with +/- this bound in degrees. Can be repeated.",
    )
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sign_flip_diagnostics", action="store_true")
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    joints, _, parent_to_joints = parse_urdf_joints(args.urdf)
    chain = find_chain(parent_to_joints, args.base_link, args.eef_link)
    active_chain = [j for j in chain if j["type"] != "fixed"]

    print(f"urdf: {args.urdf}")
    print(f"base_link: {args.base_link}")
    print(f"eef_link: {args.eef_link}")
    print("urdf_chain:")
    for i, joint in enumerate(chain):
        print(
            f"  {i:02d} {joint['name']} type={joint['type']} "
            f"parent={joint['parent']} child={joint['child']} "
            f"axis={np.round(joint['axis'], 6).tolist()}"
        )
    print("active_fk_joint_order:")
    for i, joint in enumerate(active_chain):
        print(f"  q[{i}] -> {joint['name']} axis={np.round(joint['axis'], 6).tolist()}")

    samples = pickle.load(open(args.pkl, "rb"))
    print(f"samples_total: {len(samples)}")
    print(f"sample0_keys: {list(samples[0].keys())}")
    if "joint_state" not in samples[0]:
        print("NO_JOINT_STATE: sensitivity and zero-offset optimization require joint_state in the pickle.")
        return 0

    joint_states = np.asarray([sample["joint_state"] for sample in samples], dtype=np.float64)
    if joint_states.ndim != 2 or joint_states.shape[1] != len(active_chain):
        raise ValueError(
            f"joint_state shape {joint_states.shape} does not match active FK joints {len(active_chain)}"
        )
    print("runtime_joint_state_order_assumed:")
    for i, joint in enumerate(active_chain):
        print(f"  joint_state[{i}] -> {joint['name']}")

    fk_self_consistency(chain, samples)

    delta = np.deg2rad(args.delta_deg)
    sensitivities = []
    for joint_idx in range(joint_states.shape[1]):
        mm_per_deg = []
        for q in joint_states:
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[joint_idx] += delta
            q_minus[joint_idx] -= delta
            p_plus = fk(chain, q_plus)[:3, 3]
            p_minus = fk(chain, q_minus)[:3, 3]
            mm_per_deg.append(np.linalg.norm(p_plus - p_minus) * 1000.0 / (2.0 * args.delta_deg))
        sensitivities.append(mm_per_deg)

    print("fk_position_sensitivity:")
    for i, values in enumerate(sensitivities):
        mean_v, median_v, max_v = summarize(values)
        deg_for_20mm = np.inf if mean_v == 0 else 20.0 / mean_v
        print(
            f"  joint_state[{i}] {active_chain[i]['name']}: "
            f"mean={mean_v:.4f} mm/deg median={median_v:.4f} mm/deg max={max_v:.4f} mm/deg "
            f"offset_for_20mm_mean={deg_for_20mm:.3f} deg"
        )

    if args.optimize_offsets or args.bounded_offset_bound_deg or args.sign_flip_diagnostics:
        detections = detect_tag_samples(samples, args.intr_json, args.aruco_yaml, args.tag_id)
        if len(detections) < 3:
            raise ValueError("Need at least 3 tag detections for hand-eye calibration")
        print(f"detections_used_for_hand_eye: {len(detections)}")

    if args.bounded_offset_bound_deg:
        before_trans, before_rot = calibrate_and_score(chain, samples, detections, np.zeros(6))
        print("bounded_zero_offset_optimization:")
        print_error_summary("  before", before_trans, before_rot)
        for bound_deg in args.bounded_offset_bound_deg:
            print(f"  bound_deg=+/-{bound_deg} multistart={args.multistart}")
            opt = run_bounded_offset_optimization(
                chain,
                samples,
                detections,
                bound_deg=bound_deg,
                multistart=args.multistart,
                maxiter=args.maxiter,
                seed=args.seed,
            )
            after_trans, after_rot = calibrate_and_score(chain, samples, detections, opt.x)
            print(
                f"  best_bound_{bound_deg:g}: success={opt.success} "
                f"offsets_deg={np.rad2deg(opt.x).round(6).tolist()}"
            )
            print_error_summary(f"  after_bound_{bound_deg:g}", after_trans, after_rot)

    if args.sign_flip_diagnostics:
        run_sign_flip_diagnostics(chain, samples, detections)

    if args.optimize_offsets:

        def objective(offsets):
            trans_mm, _ = calibrate_and_score(chain, samples, detections, offsets)
            return float(np.sqrt(np.mean(trans_mm * trans_mm)))

        zero = np.zeros(joint_states.shape[1], dtype=np.float64)
        before_trans, before_rot = calibrate_and_score(chain, samples, detections, zero)
        opt = minimize(
            objective,
            zero,
            method="Powell",
            options={"maxiter": args.maxiter, "xtol": 1e-5, "ftol": 1e-5},
        )
        after_trans, after_rot = calibrate_and_score(chain, samples, detections, opt.x)
        print("unbounded_zero_offset_optimization:")
        print(f"  success={opt.success} message={opt.message}")
        print(f"  offsets_deg={np.rad2deg(opt.x).round(6).tolist()}")
        for label, trans, rot in (("before", before_trans, before_rot), ("after", after_trans, after_rot)):
            print_error_summary(f"  {label}", trans, rot)


if __name__ == "__main__":
    raise SystemExit(main())
