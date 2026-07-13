#!/usr/bin/env python3
"""
Compare hand-eye calibration variants with distance filtering and train/val residuals.

This script does not modify the URDF, runtime calibration, motor zero, or hand-eye
matrix convention. Virtual joint-zero offsets are diagnostic only.
"""

import argparse
import csv
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
from umi.common.pose_util import mat_to_pose, pose_to_mat


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PKL = REPO_ROOT / "umi-deploy/data_local/hand_eye_recalib_640/hand_eye_calib_300_350.pkl"
DEFAULT_INTR = REPO_ROOT / "umi-deploy/data_local/calibration/cam0_sensor_intrinsics.json"
DEFAULT_ARUCO = REPO_ROOT / "umi-deploy/data_local/hand_eye_tags/aruco_config_tag12_147mm.yaml"
DEFAULT_URDF = REPO_ROOT / "umi-deploy/arx5-sdk/models/L5_assembly.urdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "umi-deploy/data_local/hand_eye_recalib_640/compare_variants"
DEFAULT_BASE_LINK = "ARXR5_arm_only_no_gray_base_base_link_arm"
DEFAULT_EEF_LINK = "DAS_Controller_V3_with_flange_link_Flange"


def pose6_to_mat(pose, rotation_repr="rotvec"):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    tx = np.eye(4, dtype=np.float64)
    tx[:3, 3] = pose[:3]
    if rotation_repr == "rotvec":
        tx[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    elif rotation_repr == "euler_xyz":
        tx[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
    else:
        raise ValueError(f"Unsupported rotation_repr: {rotation_repr}")
    return tx


def transform_from_rvec_tvec(rvec, tvec):
    tx = np.eye(4, dtype=np.float64)
    tx[:3, :3] = R.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    tx[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return tx


def marker_object_points(marker_size_m):
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)


def project_fisheye(rvec, tvec, marker_size_m, intr):
    projected, _ = cv2.fisheye.projectPoints(
        marker_object_points(marker_size_m),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        intr["K"],
        intr["D"],
    )
    return projected.reshape(-1, 2)


def project_pinhole(rvec, tvec, marker_size_m, K):
    projected, _ = cv2.projectPoints(
        marker_object_points(marker_size_m),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def reprojection_error(corners, rvec, tvec, marker_size_m, intr, detection_mode):
    if detection_mode in ("rectified", "processed"):
        projected = project_pinhole(rvec, tvec, marker_size_m, intr["K"])
    else:
        projected = project_fisheye(rvec, tvec, marker_size_m, intr)
    return np.linalg.norm(projected - np.asarray(corners, dtype=np.float64).reshape(4, 2), axis=1)


def corner_metrics(corners):
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    bbox = corners.max(axis=0) - corners.min(axis=0)
    return float(sides.mean()), float(bbox[0]), float(bbox[1])


def build_rectifier(raw_intr, resolution, balance=0.0, fov_scale=1.0):
    intr = convert_fisheye_intrinsics_resolution(raw_intr, resolution)
    size = (int(resolution[0]), int(resolution[1]))
    K = np.asarray(intr["K"], dtype=np.float64)
    D = np.asarray(intr["D"], dtype=np.float64)
    rectified_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        size,
        np.eye(3),
        balance=float(balance),
        new_size=size,
        fov_scale=float(fov_scale),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        rectified_K,
        size,
        cv2.CV_16SC2,
    )
    return {
        "map1": map1,
        "map2": map2,
        "intr": {
            "DIM": np.asarray(size, dtype=np.int64),
            "K": np.asarray(rectified_K, dtype=np.float64),
            "D": np.zeros((4, 1), dtype=np.float64),
        },
    }


def detect_localize_aruco_tags_pinhole(img, aruco_dict, marker_size_map, K):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
    if ids is None or len(corners) == 0:
        return {}
    tag_dict = {}
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        if this_id not in marker_size_map:
            continue
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            this_corners,
            marker_size_map[this_id],
            np.asarray(K, dtype=np.float64),
            np.zeros((5, 1), dtype=np.float64),
        )
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


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
    axis = axis / np.linalg.norm(axis)
    tx[:3, :3] = R.from_rotvec(axis * angle).as_matrix()
    return tx


def parse_urdf_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    parent_to_joints = {}
    for joint_el in root.findall("joint"):
        origin_el = joint_el.find("origin")
        axis_el = joint_el.find("axis")
        joint = {
            "name": joint_el.attrib["name"],
            "type": joint_el.attrib.get("type", "fixed"),
            "parent": joint_el.find("parent").attrib["link"],
            "child": joint_el.find("child").attrib["link"],
            "origin_xyz": parse_xyz(origin_el.attrib.get("xyz") if origin_el is not None else None),
            "origin_rpy": parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None),
            "axis": parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None),
        }
        parent_to_joints.setdefault(joint["parent"], []).append(joint)
    return parent_to_joints


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
    raise ValueError(f"No URDF chain from {base_link} to {eef_link}")


def fk(chain, joint_state):
    q = np.asarray(joint_state, dtype=np.float64).reshape(-1)
    tx = np.eye(4, dtype=np.float64)
    active_idx = 0
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
        raise ValueError(f"FK consumed {active_idx} active joints but joint_state has {q.size}")
    return tx


def load_raw_detection(sample, sample_idx, tag_id, marker_size_m, raw_intr, aruco_config):
    img = sample["img"]
    intr = convert_fisheye_intrinsics_resolution(raw_intr, img.shape[:2][::-1])
    source = "saved"
    if (
        "aruco" in sample
        and sample["aruco"] is not None
        and sample["aruco"].get("detected", True)
        and int(sample["aruco"].get("tag_id", tag_id)) == tag_id
        and all(key in sample["aruco"] for key in ("rvec", "tvec", "corners"))
    ):
        tag = sample["aruco"]
    else:
        source = "redetected"
        tag_dict = detect_localize_aruco_tags(
            img,
            aruco_config["aruco_dict"],
            aruco_config["marker_size_map"],
            intr,
        )
        if tag_id not in tag_dict:
            return None
        tag = tag_dict[tag_id]
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    reproj = reprojection_error(corners, rvec, tvec, marker_size_m, intr, "raw")
    side_px, bbox_w_px, bbox_h_px = corner_metrics(corners)
    return {
        "idx": sample_idx,
        "sample": sample,
        "detection_mode": "raw",
        "source": source,
        "tx_camera_world": transform_from_rvec_tvec(rvec, tvec),
        "rvec": rvec,
        "tvec": tvec,
        "corners": corners,
        "z_mm": float(tvec[2] * 1000.0),
        "norm_mm": float(np.linalg.norm(tvec) * 1000.0),
        "reprojection_mean_px": float(reproj.mean()),
        "reprojection_max_px": float(reproj.max()),
        "corner_side_px": side_px,
        "bbox_width_px": bbox_w_px,
        "bbox_height_px": bbox_h_px,
    }


def load_rectified_detection(sample, sample_idx, tag_id, marker_size_m, raw_intr, aruco_config, rectifier):
    img_rgb = sample["img"]
    intr = convert_fisheye_intrinsics_resolution(raw_intr, img_rgb.shape[:2][::-1])
    img_bgr = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
    rectified_bgr = cv2.remap(
        img_bgr,
        rectifier["map1"],
        rectifier["map2"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    tag_dict = detect_localize_aruco_tags_pinhole(
        rectified_rgb,
        aruco_config["aruco_dict"],
        aruco_config["marker_size_map"],
        rectifier["intr"]["K"],
    )
    if tag_id not in tag_dict:
        return None
    tag = tag_dict[tag_id]
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    reproj = reprojection_error(corners, rvec, tvec, marker_size_m, rectifier["intr"], "rectified")
    side_px, bbox_w_px, bbox_h_px = corner_metrics(corners)
    _ = intr  # keep raw-resolution validation explicit above
    return {
        "idx": sample_idx,
        "sample": sample,
        "detection_mode": "rectified",
        "source": "rectified_redetected",
        "tx_camera_world": transform_from_rvec_tvec(rvec, tvec),
        "rvec": rvec,
        "tvec": tvec,
        "corners": corners,
        "z_mm": float(tvec[2] * 1000.0),
        "norm_mm": float(np.linalg.norm(tvec) * 1000.0),
        "reprojection_mean_px": float(reproj.mean()),
        "reprojection_max_px": float(reproj.max()),
        "corner_side_px": side_px,
        "bbox_width_px": bbox_w_px,
        "bbox_height_px": bbox_h_px,
    }


def load_stored_aruco_detection(sample, sample_idx, aruco_key, image_mode, tag_id, marker_size_m, raw_intr, rectifier=None):
    tag = sample.get("aruco" if aruco_key == "raw_aruco" else aruco_key)
    if not tag or not tag.get("detected", True):
        return None
    if int(tag.get("tag_id", tag_id)) != tag_id:
        return None
    if not all(key in tag for key in ("rvec", "tvec", "corners")):
        return None
    img_rgb = np.asarray(sample["img"])
    if image_mode == "processed":
        if "processed_K" in tag:
            intr = {
                "DIM": np.asarray(img_rgb.shape[:2][::-1], dtype=np.int64),
                "K": np.asarray(tag["processed_K"], dtype=np.float64),
                "D": np.zeros((4, 1), dtype=np.float64),
            }
        elif rectifier is not None:
            intr = rectifier["intr"]
        else:
            raise ValueError(f"sample {sample_idx} has no processed_K and no rectifier")
        reproj_mode = "processed"
    else:
        intr = convert_fisheye_intrinsics_resolution(raw_intr, img_rgb.shape[:2][::-1])
        reproj_mode = "raw"
    rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
    corners = np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2)
    reproj = reprojection_error(corners, rvec, tvec, marker_size_m, intr, reproj_mode)
    side_px, bbox_w_px, bbox_h_px = corner_metrics(corners)
    return {
        "idx": sample_idx,
        "sample": sample,
        "detection_mode": image_mode,
        "image_mode": image_mode,
        "aruco_source": aruco_key,
        "source": aruco_key,
        "tx_camera_world": transform_from_rvec_tvec(rvec, tvec),
        "rvec": rvec,
        "tvec": tvec,
        "corners": corners,
        "z_mm": float(tag.get("z_mm", tvec[2] * 1000.0)),
        "norm_mm": float(tag.get("norm_mm", np.linalg.norm(tvec) * 1000.0)),
        "reprojection_mean_px": float(tag.get("reprojection_mean_px", reproj.mean())),
        "reprojection_max_px": float(tag.get("reprojection_max_px", reproj.max())),
        "corner_side_px": side_px,
        "bbox_width_px": bbox_w_px,
        "bbox_height_px": bbox_h_px,
        "intr": intr,
    }


def load_stored_aruco_detections(samples, aruco_key, image_mode, raw_intr, tag_id, marker_size_m, rectifier=None):
    detections = []
    failed = []
    for idx, sample in enumerate(samples):
        det = load_stored_aruco_detection(
            sample,
            idx,
            aruco_key,
            image_mode,
            tag_id,
            marker_size_m,
            raw_intr,
            rectifier=rectifier,
        )
        if det is None:
            failed.append(idx)
        else:
            detections.append(det)
    if failed:
        print(f"{aruco_key} stored detection unavailable sample indices: {failed}")
    return detections


def load_detections(samples, detection_mode, raw_intr, aruco_config, tag_id, marker_size_m, rectifier=None):
    detections = []
    redetected = 0
    failed = []
    for idx, sample in enumerate(samples):
        if detection_mode == "rectified":
            det = load_rectified_detection(sample, idx, tag_id, marker_size_m, raw_intr, aruco_config, rectifier)
        else:
            det = load_raw_detection(sample, idx, tag_id, marker_size_m, raw_intr, aruco_config)
            if det is not None and det["source"] == "redetected":
                redetected += 1
        if det is None:
            failed.append(idx)
        else:
            detections.append(det)
    if redetected:
        print(f"WARNING: raw detection recomputed for {redetected} samples because sample['aruco'] was absent.")
    if failed:
        print(f"{detection_mode} detection failed sample indices: {failed}")
    return detections


def draw_polyline(img, points, color, thickness=2):
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness=thickness, lineType=cv2.LINE_AA)


def write_detection_overlays(output_dir, detections, detection_mode, marker_size_m, raw_intr, rectifier=None):
    overlay_dir = output_dir / f"{detection_mode}_detection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for det in detections:
        img_rgb = det["sample"]["img"]
        if detection_mode in ("rectified", "processed", "processed_aruco"):
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            vis_bgr = cv2.remap(
                img_bgr,
                rectifier["map1"],
                rectifier["map2"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            K = det.get("intr", rectifier["intr"])["K"]
            projected = project_pinhole(det["rvec"], det["tvec"], marker_size_m, K)
        else:
            vis = img_rgb.copy()
            intr = convert_fisheye_intrinsics_resolution(raw_intr, img_rgb.shape[:2][::-1])
            projected = project_fisheye(det["rvec"], det["tvec"], marker_size_m, intr)
        draw_polyline(vis, det["corners"], (0, 255, 0), thickness=2)
        draw_polyline(vis, projected, (0, 180, 255), thickness=1)
        text = (
            f"{detection_mode} sample={det['idx']:03d} "
            f"norm={det['norm_mm']:.1f}mm reproj={det['reprojection_mean_px']:.2f}px"
        )
        cv2.putText(vis, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
        out_path = overlay_dir / f"sample_{det['idx']:03d}_{det['norm_mm']:.1f}mm.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        if det["norm_mm"] < 250.0:
            key = "lt250"
        elif det["norm_mm"] < 300.0:
            key = "250_300"
        elif det["norm_mm"] <= 350.0:
            key = "300_350"
        else:
            key = "gt350"
        grouped.setdefault(key, []).append((det["idx"], vis))

    for key, items in sorted(grouped.items()):
        thumbs = []
        for _, img in sorted(items, key=lambda item: item[0]):
            thumbs.append(cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA))
        cols = min(5, len(thumbs))
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = np.full((rows * 240, cols * 320, 3), 32, dtype=np.uint8)
        for idx, thumb in enumerate(thumbs):
            row = idx // cols
            col = idx % cols
            sheet[row * 240 : (row + 1) * 240, col * 320 : (col + 1) * 320] = thumb
        cv2.imwrite(str(overlay_dir / f"contact_sheet_{key}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"{detection_mode}_detection_overlay_dir={overlay_dir}")


def filter_detections(detections, min_distance_mm, max_distance_mm, max_reproj_px):
    used = []
    excluded = []
    for det in detections:
        reason = None
        if min_distance_mm is not None and det["norm_mm"] < min_distance_mm:
            reason = f"distance_below_{min_distance_mm:g}mm"
        elif max_distance_mm is not None and det["norm_mm"] > max_distance_mm:
            reason = f"distance_above_{max_distance_mm:g}mm"
        elif det["reprojection_mean_px"] > max_reproj_px:
            reason = f"reprojection_above_{max_reproj_px:g}px"
        if reason is None:
            used.append(det)
        else:
            excluded.append({**det, "reason": reason})
    return used, excluded


def split_stratified(detections, train_ratio, seed):
    rng = np.random.default_rng(seed)
    bins = {}
    for det in detections:
        d = det["norm_mm"]
        if d < 250.0:
            key = "lt250"
        elif d < 300.0:
            key = "250_300"
        elif d <= 350.0:
            key = "300_350"
        else:
            key = "gt350"
        bins.setdefault(key, []).append(det)

    train = []
    val = []
    for key in sorted(bins):
        group = list(bins[key])
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        n_train = int(round(len(group) * train_ratio))
        n_train = min(max(n_train, 1), len(group) - 1)
        train.extend(group[:n_train])
        val.extend(group[n_train:])
    train.sort(key=lambda d: d["idx"])
    val.sort(key=lambda d: d["idx"])
    return train, val


def pose_provider_tcp(det, offsets=None, chain=None):
    _ = offsets, chain
    return pose6_to_mat(det["sample"]["tcp_pose"], "rotvec")


def pose_provider_joint_fk(det, offsets, chain):
    joint_state = np.asarray(det["sample"]["joint_state"], dtype=np.float64).reshape(6)
    return fk(chain, joint_state + np.asarray(offsets, dtype=np.float64).reshape(6))


def calibrate_hand_eye(detections, pose_source, offsets=None, chain=None):
    if len(detections) < 3:
        raise ValueError(f"Need at least 3 detections for calibration, got {len(detections)}")
    r_world2cam = []
    t_world2cam = []
    r_base2gripper = []
    t_base2gripper = []
    for det in detections:
        tx_camera_world = det["tx_camera_world"]
        if pose_source == "joint_state_corrected_fk":
            tx_base_gripper = pose_provider_joint_fk(det, offsets, chain)
        else:
            tx_base_gripper = pose_provider_tcp(det)
        tx_gripper_base = np.linalg.inv(tx_base_gripper)
        tv_gripper_base = mat_to_pose(tx_gripper_base)
        r_world2cam.append(R.from_matrix(tx_camera_world[:3, :3]).as_rotvec())
        t_world2cam.append(tx_camera_world[:3, 3])
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
    tx_world_base[:3, 3] = np.asarray(t_b2w).reshape(3)
    tx_camera_gripper = np.eye(4, dtype=np.float64)
    tx_camera_gripper[:3, :3] = r_g2c
    tx_camera_gripper[:3, 3] = np.asarray(t_g2c).reshape(3)
    return tx_world_base, tx_camera_gripper


def evaluate_hand_eye(detections, tx_world_base, tx_camera_gripper, pose_source, offsets=None, chain=None):
    trans = []
    rot = []
    for det in detections:
        if pose_source == "joint_state_corrected_fk":
            tx_base_gripper = pose_provider_joint_fk(det, offsets, chain)
        else:
            tx_base_gripper = pose_provider_tcp(det)
        tx_gripper_base = np.linalg.inv(tx_base_gripper)
        left = det["tx_camera_world"] @ tx_world_base
        right = tx_camera_gripper @ tx_gripper_base
        tx_err = np.linalg.inv(left) @ right
        trans.append(np.linalg.norm(tx_err[:3, 3]) * 1000.0)
        rot.append(np.rad2deg(R.from_matrix(tx_err[:3, :3]).magnitude()))
    return np.asarray(trans, dtype=np.float64), np.asarray(rot, dtype=np.float64)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "median": None, "rms": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "max": float(values.max()),
    }


def optimize_offsets(train_detections, chain, bound_deg, multistart, maxiter, seed):
    rng = np.random.default_rng(seed)
    bound = np.deg2rad(bound_deg)
    starts = [np.zeros(6, dtype=np.float64)]
    for _ in range(max(0, multistart - 1)):
        starts.append(rng.uniform(-bound, bound, size=6))
    bounds = [(-bound, bound)] * 6

    def objective(offsets):
        tx_world_base, tx_camera_gripper = calibrate_hand_eye(
            train_detections,
            "joint_state_corrected_fk",
            offsets=offsets,
            chain=chain,
        )
        trans, _ = evaluate_hand_eye(
            train_detections,
            tx_world_base,
            tx_camera_gripper,
            "joint_state_corrected_fk",
            offsets=offsets,
            chain=chain,
        )
        return float(np.sqrt(np.mean(trans * trans)))

    best = None
    for start in starts:
        opt = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        if best is None or opt.fun < best.fun:
            best = opt
    return best.x, best


def save_hand_eye_json(path, variant, tx_world_base, tx_camera_gripper, offsets_deg=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tx_base2world": tx_world_base.tolist(),
        "tx_gripper2camera": tx_camera_gripper.tolist(),
        "matrix_semantics": {
            "tx_base2world_key_actual_semantics": "T_world_base (^w T_b), OpenCV base-to-world output stored under legacy key tx_base2world",
            "tx_gripper2camera_key_actual_semantics": "T_camera_gripper (^c T_g), OpenCV gripper-to-camera output stored under legacy key tx_gripper2camera",
            "closed_loop_formula": "T_camera_world @ T_world_base == T_camera_gripper @ T_gripper_base",
        },
        "pose_key": "tcp_pose",
        "pose_rotation_repr": "rotvec",
        "variant": variant,
    }
    if offsets_deg is not None:
        payload["virtual_joint_zero_offsets_deg"] = offsets_deg
        payload["runtime_warning"] = (
            "Diagnostic only. Do not deploy unless runtime current_tcp_pose is computed "
            "from joint_state with the same offsets."
        )
    json.dump(payload, open(path, "w"), indent=2)


def make_variant_defs():
    return [
        {
            "name": "raw_all",
            "detection_mode": "raw",
            "distance_filter": "all",
            "min_mm": None,
            "max_mm": None,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "raw_250_350",
            "detection_mode": "raw",
            "distance_filter": "250_350",
            "min_mm": 250.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "raw_300_350",
            "detection_mode": "raw",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "rectified_250_350",
            "detection_mode": "rectified",
            "distance_filter": "250_350",
            "min_mm": 250.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
            "comparison_only": True,
        },
        {
            "name": "rectified_300_350",
            "detection_mode": "rectified",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
            "comparison_only": True,
        },
        {
            "name": "offset_300_350_bound5",
            "detection_mode": "raw",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "joint_state_corrected_fk",
            "runtime_safe": False,
            "offset_bound_deg": 5.0,
        },
        {
            "name": "offset_250_350_bound5",
            "detection_mode": "raw",
            "distance_filter": "250_350",
            "min_mm": 250.0,
            "max_mm": 350.0,
            "pose_source": "joint_state_corrected_fk",
            "runtime_safe": False,
            "offset_bound_deg": 5.0,
        },
        {
            "name": "rectified_offset_300_350_bound5",
            "detection_mode": "rectified",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "joint_state_corrected_fk",
            "runtime_safe": False,
            "offset_bound_deg": 5.0,
            "comparison_only": True,
        },
    ]


def make_processed_aruco_variant_defs():
    return [
        {
            "name": "raw_aruco_all",
            "detection_key": "raw_aruco",
            "detection_mode": "raw",
            "image_mode": "raw",
            "aruco_source": "raw_aruco",
            "distance_filter": "all",
            "min_mm": None,
            "max_mm": None,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "raw_aruco_250_350",
            "detection_key": "raw_aruco",
            "detection_mode": "raw",
            "image_mode": "raw",
            "aruco_source": "raw_aruco",
            "distance_filter": "250_350",
            "min_mm": 250.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "raw_aruco_300_350",
            "detection_key": "raw_aruco",
            "detection_mode": "raw",
            "image_mode": "raw",
            "aruco_source": "raw_aruco",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
        },
        {
            "name": "processed_aruco_all",
            "detection_key": "processed_aruco",
            "detection_mode": "processed",
            "image_mode": "processed",
            "aruco_source": "processed_aruco",
            "distance_filter": "all",
            "min_mm": None,
            "max_mm": None,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
            "comparison_only": True,
        },
        {
            "name": "processed_aruco_250_350",
            "detection_key": "processed_aruco",
            "detection_mode": "processed",
            "image_mode": "processed",
            "aruco_source": "processed_aruco",
            "distance_filter": "250_350",
            "min_mm": 250.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
            "comparison_only": True,
        },
        {
            "name": "processed_aruco_300_350",
            "detection_key": "processed_aruco",
            "detection_mode": "processed",
            "image_mode": "processed",
            "aruco_source": "processed_aruco",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "tcp_pose",
            "runtime_safe": True,
            "comparison_only": True,
        },
        {
            "name": "processed_aruco_offset_300_350_bound5",
            "detection_key": "processed_aruco",
            "detection_mode": "processed",
            "image_mode": "processed",
            "aruco_source": "processed_aruco",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "joint_state_corrected_fk",
            "runtime_safe": False,
            "offset_bound_deg": 5.0,
            "comparison_only": True,
        },
        {
            "name": "raw_aruco_offset_300_350_bound5",
            "detection_key": "raw_aruco",
            "detection_mode": "raw",
            "image_mode": "raw",
            "aruco_source": "raw_aruco",
            "distance_filter": "300_350",
            "min_mm": 300.0,
            "max_mm": 350.0,
            "pose_source": "joint_state_corrected_fk",
            "runtime_safe": False,
            "offset_bound_deg": 5.0,
        },
    ]


def val_median(result):
    return result.get("val_translation", {}).get("median")


def mark_recommendations(results):
    for result in results:
        result["recommended_for_runtime"] = False
        if result.get("status") != "ok":
            result["not_recommended_reason"] = result.get("reason", "variant failed")
        elif not result["runtime_safe"]:
            result["not_recommended_reason"] = "diagnostic virtual joint offset variant; runtime TCP frame would not match"
        elif result["samples_used_val"] < 3:
            result["not_recommended_reason"] = "validation sample count is too small"
        elif result.get("comparison_only"):
            result["not_recommended_reason"] = "comparison variant; recommend only if validation hand-eye residual beats raw TCP variant"
        else:
            result["not_recommended_reason"] = "not selected"

    ok = [r for r in results if r.get("status") == "ok" and r["runtime_safe"] and r["samples_used_val"] >= 3]
    raw300 = next((r for r in ok if r["variant_name"] in ("raw_300_350", "raw_aruco_300_350")), None)
    raw250 = next((r for r in ok if r["variant_name"] in ("raw_250_350", "raw_aruco_250_350")), None)
    rect300 = next(
        (r for r in ok if r["variant_name"] in ("rectified_300_350", "processed_aruco_300_350")),
        None,
    )

    selected = None
    if raw300 is not None and val_median(raw300) is not None and val_median(raw300) <= 12.0:
        selected = raw300
    elif raw250 is not None and val_median(raw250) is not None and val_median(raw250) <= 12.0:
        selected = raw250
    elif (
        rect300 is not None
        and raw300 is not None
        and val_median(rect300) is not None
        and val_median(raw300) is not None
        and val_median(rect300) <= 0.8 * val_median(raw300)
        and val_median(rect300) <= 12.0
    ):
        selected = rect300

    if selected is not None:
        selected["recommended_for_runtime"] = True
        selected["not_recommended_reason"] = ""
    return results


def print_variant_table(results):
    print("\nVARIANT_SUMMARY")
    header = (
        "variant | aruco_source | image_mode | filter | pose_source | train_n | val_n | "
        "train_trans_median | val_trans_median | train_rot_median | val_rot_median | "
        "runtime_safe | recommended | reason"
    )
    print(header)
    for r in results:
        if r.get("status") != "ok":
            print(
                f"{r['variant_name']} | {r.get('aruco_source', r['detection_mode'])} | "
                f"{r.get('image_mode', r['detection_mode'])} | {r['distance_filter']} | "
                f"{r['pose_source']} | 0 | 0 | N/A | N/A | N/A | N/A | "
                f"{r['runtime_safe']} | False | {r.get('reason', 'failed')}"
            )
            continue
        print(
            f"{r['variant_name']} | {r.get('aruco_source', r['detection_mode'])} | "
            f"{r.get('image_mode', r['detection_mode'])} | {r['distance_filter']} | {r['pose_source']} | "
            f"{r['samples_used_train']} | {r['samples_used_val']} | "
            f"{r['train_translation']['median']:.3f} | "
            f"{r['val_translation']['median'] if r['val_translation']['median'] is not None else 'N/A'} | "
            f"{r['train_rotation']['median']:.3f} | "
            f"{r['val_rotation']['median'] if r['val_rotation']['median'] is not None else 'N/A'} | "
            f"{r['runtime_safe']} | {r['recommended_for_runtime']} | {r.get('not_recommended_reason', '')}"
        )


def write_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_name",
        "aruco_source",
        "image_mode",
        "detection_mode",
        "distance_filter",
        "pose_source",
        "samples_total",
        "samples_with_tag",
        "samples_used_train",
        "samples_used_val",
        "excluded_by_distance",
        "excluded_by_reprojection",
        "train_trans_mean",
        "train_trans_median",
        "train_trans_rms",
        "train_trans_max",
        "val_trans_mean",
        "val_trans_median",
        "val_trans_rms",
        "val_trans_max",
        "train_rot_mean",
        "train_rot_median",
        "train_rot_max",
        "val_rot_mean",
        "val_rot_median",
        "val_rot_max",
        "offsets_deg",
        "runtime_safe",
        "recommended_for_runtime",
        "hand_eye_json",
        "not_recommended_reason",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "variant_name": r["variant_name"],
                    "aruco_source": r.get("aruco_source", r.get("detection_mode")),
                    "image_mode": r.get("image_mode", r.get("detection_mode")),
                    "detection_mode": r["detection_mode"],
                    "distance_filter": r["distance_filter"],
                    "pose_source": r["pose_source"],
                    "samples_total": r["samples_total"],
                    "samples_with_tag": r.get("samples_with_tag"),
                    "samples_used_train": r.get("samples_used_train"),
                    "samples_used_val": r.get("samples_used_val"),
                    "excluded_by_distance": r.get("excluded_by_distance"),
                    "excluded_by_reprojection": r.get("excluded_by_reprojection"),
                    "train_trans_mean": r.get("train_translation", {}).get("mean"),
                    "train_trans_median": r.get("train_translation", {}).get("median"),
                    "train_trans_rms": r.get("train_translation", {}).get("rms"),
                    "train_trans_max": r.get("train_translation", {}).get("max"),
                    "val_trans_mean": r.get("val_translation", {}).get("mean"),
                    "val_trans_median": r.get("val_translation", {}).get("median"),
                    "val_trans_rms": r.get("val_translation", {}).get("rms"),
                    "val_trans_max": r.get("val_translation", {}).get("max"),
                    "train_rot_mean": r.get("train_rotation", {}).get("mean"),
                    "train_rot_median": r.get("train_rotation", {}).get("median"),
                    "train_rot_max": r.get("train_rotation", {}).get("max"),
                    "val_rot_mean": r.get("val_rotation", {}).get("mean"),
                    "val_rot_median": r.get("val_rotation", {}).get("median"),
                    "val_rot_max": r.get("val_rotation", {}).get("max"),
                    "offsets_deg": r.get("offsets_deg"),
                    "runtime_safe": r.get("runtime_safe"),
                    "recommended_for_runtime": r.get("recommended_for_runtime"),
                    "hand_eye_json": r.get("hand_eye_json"),
                    "not_recommended_reason": r.get("not_recommended_reason", r.get("reason")),
                }
            )


def run_variant(variant, args, train_samples, val_samples, detections_by_mode, val_detections_by_mode, chain):
    detection_key = variant.get("detection_key", variant["detection_mode"])
    all_train_dets = detections_by_mode[detection_key]
    all_val_dets = val_detections_by_mode.get(detection_key, [])
    filtered_train_all, excluded_train = filter_detections(
        all_train_dets,
        variant["min_mm"],
        variant["max_mm"],
        args.max_reproj_px,
    )
    excluded_by_distance = sum("distance" in item["reason"] for item in excluded_train)
    excluded_by_reprojection = sum("reprojection" in item["reason"] for item in excluded_train)

    if args.val_pkl:
        train_dets = filtered_train_all
        val_dets, excluded_val = filter_detections(
            all_val_dets,
            variant["min_mm"],
            variant["max_mm"],
            args.max_reproj_px,
        )
        excluded_by_distance += sum("distance" in item["reason"] for item in excluded_val)
        excluded_by_reprojection += sum("reprojection" in item["reason"] for item in excluded_val)
    else:
        train_dets, val_dets = split_stratified(filtered_train_all, args.train_ratio, args.seed)

    base_result = {
        "variant_name": variant["name"],
        "detection_mode": variant["detection_mode"],
        "image_mode": variant.get("image_mode", variant["detection_mode"]),
        "aruco_source": variant.get("aruco_source", variant["detection_mode"]),
        "distance_filter": variant["distance_filter"],
        "pose_source": variant["pose_source"],
        "samples_total": len(train_samples) + (len(val_samples) if args.val_pkl else 0),
        "samples_with_tag": len(all_train_dets) + (len(all_val_dets) if args.val_pkl else 0),
        "samples_used_train": len(train_dets),
        "samples_used_val": len(val_dets),
        "excluded_by_distance": excluded_by_distance,
        "excluded_by_reprojection": excluded_by_reprojection,
        "runtime_safe": bool(variant["runtime_safe"]),
        "comparison_only": bool(variant.get("comparison_only", False)),
    }

    if len(train_dets) < 3:
        base_result.update({"status": "skipped", "reason": "fewer than 3 training samples after filtering"})
        return base_result
    if variant["pose_source"] == "joint_state_corrected_fk":
        missing_joint = [det["idx"] for det in train_dets + val_dets if "joint_state" not in det["sample"]]
        if missing_joint:
            base_result.update({"status": "skipped", "reason": f"missing joint_state in samples {missing_joint}"})
            return base_result

    offsets = np.zeros(6, dtype=np.float64)
    offsets_deg = None
    if variant["pose_source"] == "joint_state_corrected_fk":
        offsets, opt = optimize_offsets(
            train_dets,
            chain,
            bound_deg=variant["offset_bound_deg"],
            multistart=args.offset_multistart,
            maxiter=args.offset_maxiter,
            seed=args.seed,
        )
        offsets_deg = np.rad2deg(offsets).round(6).tolist()
        base_result["offset_optimization_success"] = bool(opt.success)
        base_result["offset_optimization_objective"] = float(opt.fun)

    tx_world_base, tx_camera_gripper = calibrate_hand_eye(
        train_dets,
        variant["pose_source"],
        offsets=offsets,
        chain=chain,
    )
    train_trans, train_rot = evaluate_hand_eye(
        train_dets,
        tx_world_base,
        tx_camera_gripper,
        variant["pose_source"],
        offsets=offsets,
        chain=chain,
    )
    val_trans, val_rot = evaluate_hand_eye(
        val_dets,
        tx_world_base,
        tx_camera_gripper,
        variant["pose_source"],
        offsets=offsets,
        chain=chain,
    )
    json_path = Path(args.output_dir).resolve() / f"{variant['name']}.json"
    train_indices_path = Path(args.output_dir).resolve() / f"{variant['name']}_train_indices.txt"
    val_indices_path = Path(args.output_dir).resolve() / f"{variant['name']}_val_indices.txt"
    train_indices_path.write_text("\n".join(str(int(det["idx"])) for det in train_dets) + "\n")
    val_indices_path.write_text("\n".join(str(int(det["idx"])) for det in val_dets) + ("\n" if val_dets else ""))
    save_hand_eye_json(json_path, variant, tx_world_base, tx_camera_gripper, offsets_deg=offsets_deg)

    base_result.update(
        {
            "status": "ok",
            "train_translation": stats(train_trans),
            "val_translation": stats(val_trans),
            "train_rotation": stats(train_rot),
            "val_rotation": stats(val_rot),
            "offsets_deg": offsets_deg,
            "hand_eye_json": str(json_path),
            "train_indices_file": str(train_indices_path),
            "val_indices_file": str(val_indices_path),
            "used_train_samples": [
                {
                    "idx": int(det["idx"]),
                    "z_mm": det["z_mm"],
                    "norm_mm": det["norm_mm"],
                    "reprojection_mean_px": det["reprojection_mean_px"],
                }
                for det in train_dets
            ],
            "used_val_samples": [
                {
                    "idx": int(det["idx"]),
                    "z_mm": det["z_mm"],
                    "norm_mm": det["norm_mm"],
                    "reprojection_mean_px": det["reprojection_mean_px"],
                }
                for det in val_dets
            ],
        }
    )
    if variant["name"] == "raw_300_350" and len(train_dets) + len(val_dets) < 12:
        base_result["sample_count_warning"] = "raw_300_350 has fewer than 12 total samples; do not strongly recommend"
    if variant["pose_source"] == "joint_state_corrected_fk":
        base_result["diagnostic_warning"] = (
            "NOT RUNTIME SAFE unless runtime applies the same corrected FK. "
            "Do not write these offsets to motor zero or runtime config from this script."
        )
    if variant.get("comparison_only"):
        base_result["comparison_warning"] = (
            "Rectified is a comparison variant. Lower reprojection alone is not enough; "
            "use validation hand-eye residual."
        )
    return base_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", default=str(DEFAULT_PKL))
    parser.add_argument("--val_pkl", default=None)
    parser.add_argument("--intr_json", default=str(DEFAULT_INTR))
    parser.add_argument("--aruco_yaml", default=str(DEFAULT_ARUCO))
    parser.add_argument("--tag_id", type=int, default=12)
    parser.add_argument("--max_reproj_px", type=float, default=4.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--base_link", default=DEFAULT_BASE_LINK)
    parser.add_argument("--eef_link", default=DEFAULT_EEF_LINK)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rectify_balance", type=float, default=0.0)
    parser.add_argument("--rectify_fov_scale", type=float, default=1.0)
    parser.add_argument("--offset_multistart", type=int, default=8)
    parser.add_argument("--offset_maxiter", type=int, default=80)
    parser.add_argument(
        "--write_detection_overlays",
        action="store_true",
        help="Write raw and rectified ArUco detected-corner overlays before calibration.",
    )
    parser.add_argument(
        "--compare_processed_aruco",
        action="store_true",
        help="Compare stored sample['aruco'] against stored sample['processed_aruco'] without mixing detection/check modes.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = pickle.load(open(args.pkl, "rb"))
    val_samples = pickle.load(open(args.val_pkl, "rb")) if args.val_pkl else []
    raw_intr = parse_fisheye_intrinsics(json.load(open(args.intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(args.aruco_yaml, "r")))
    marker_size_m = float(aruco_config["marker_size_map"][args.tag_id])
    print(f"pkl={args.pkl}")
    if args.val_pkl:
        print(f"val_pkl={args.val_pkl}")
    print(f"tag_id={args.tag_id} marker_size={marker_size_m:.6f} m")
    print(f"aruco_yaml={args.aruco_yaml}")

    if not train_samples:
        raise ValueError("Training pkl contains no samples")
    resolution = train_samples[0]["img"].shape[:2][::-1]
    rectifier = build_rectifier(
        raw_intr,
        resolution,
        balance=args.rectify_balance,
        fov_scale=args.rectify_fov_scale,
    )

    if args.compare_processed_aruco:
        detections_by_mode = {
            "raw_aruco": load_stored_aruco_detections(
                train_samples,
                "raw_aruco",
                "raw",
                raw_intr,
                args.tag_id,
                marker_size_m,
                rectifier=rectifier,
            ),
            "processed_aruco": load_stored_aruco_detections(
                train_samples,
                "processed_aruco",
                "processed",
                raw_intr,
                args.tag_id,
                marker_size_m,
                rectifier=rectifier,
            ),
        }
    else:
        detections_by_mode = {
            "raw": load_detections(train_samples, "raw", raw_intr, aruco_config, args.tag_id, marker_size_m),
            "rectified": load_detections(
                train_samples,
                "rectified",
                raw_intr,
                aruco_config,
                args.tag_id,
                marker_size_m,
                rectifier=rectifier,
            ),
        }
    if args.write_detection_overlays:
        if args.compare_processed_aruco:
            write_detection_overlays(output_dir, detections_by_mode["raw_aruco"], "raw_aruco", marker_size_m, raw_intr)
            write_detection_overlays(
                output_dir,
                detections_by_mode["processed_aruco"],
                "processed_aruco",
                marker_size_m,
                raw_intr,
                rectifier=rectifier,
            )
        else:
            write_detection_overlays(output_dir, detections_by_mode["raw"], "raw", marker_size_m, raw_intr)
            write_detection_overlays(
                output_dir,
                detections_by_mode["rectified"],
                "rectified",
                marker_size_m,
                raw_intr,
                rectifier=rectifier,
            )
    val_detections_by_mode = {}
    if args.val_pkl:
        if args.compare_processed_aruco:
            val_detections_by_mode = {
                "raw_aruco": load_stored_aruco_detections(
                    val_samples,
                    "raw_aruco",
                    "raw",
                    raw_intr,
                    args.tag_id,
                    marker_size_m,
                    rectifier=rectifier,
                ),
                "processed_aruco": load_stored_aruco_detections(
                    val_samples,
                    "processed_aruco",
                    "processed",
                    raw_intr,
                    args.tag_id,
                    marker_size_m,
                    rectifier=rectifier,
                ),
            }
        else:
            val_detections_by_mode = {
                "raw": load_detections(val_samples, "raw", raw_intr, aruco_config, args.tag_id, marker_size_m),
                "rectified": load_detections(
                    val_samples,
                    "rectified",
                    raw_intr,
                    aruco_config,
                    args.tag_id,
                    marker_size_m,
                    rectifier=rectifier,
                ),
            }

    parent_to_joints = parse_urdf_joints(args.urdf)
    chain = find_chain(parent_to_joints, args.base_link, args.eef_link)
    active_joints = [joint for joint in chain if joint["type"] != "fixed"]
    print("active_fk_joint_order:")
    for i, joint in enumerate(active_joints):
        print(f"  joint_state[{i}] -> {joint['name']} axis={np.round(joint['axis'], 6).tolist()}")

    results = []
    variant_defs = make_processed_aruco_variant_defs() if args.compare_processed_aruco else make_variant_defs()
    for variant in variant_defs:
        print(f"\nRUN_VARIANT {variant['name']}")
        try:
            result = run_variant(
                variant,
                args,
                train_samples,
                val_samples,
                detections_by_mode,
                val_detections_by_mode,
                chain,
            )
        except Exception as exc:
            result = {
                "variant_name": variant["name"],
                "detection_mode": variant["detection_mode"],
                "image_mode": variant.get("image_mode", variant["detection_mode"]),
                "aruco_source": variant.get("aruco_source", variant["detection_mode"]),
                "distance_filter": variant["distance_filter"],
                "pose_source": variant["pose_source"],
                "samples_total": len(train_samples) + (len(val_samples) if args.val_pkl else 0),
                "samples_with_tag": None,
                "samples_used_train": 0,
                "samples_used_val": 0,
                "excluded_by_distance": None,
                "excluded_by_reprojection": None,
                "runtime_safe": bool(variant["runtime_safe"]),
                "comparison_only": bool(variant.get("comparison_only", False)),
                "status": "failed",
                "reason": repr(exc),
            }
        results.append(result)

    results = mark_recommendations(results)
    json.dump({"variants": results}, open(output_dir / "variant_summary.json", "w"), indent=2)
    write_csv(output_dir / "variant_summary.csv", results)
    print_variant_table(results)
    print(f"output_dir={output_dir}")
    print("NOTE: virtual joint-zero offset variants are diagnostic only and are never direct runtime recommendations.")
    print("NOTE: rectified variants are comparison variants; lower reprojection is not sufficient without lower validation hand-eye residual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
