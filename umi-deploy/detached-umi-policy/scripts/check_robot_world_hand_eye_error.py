import json
import os
import pickle
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import pose_to_mat


def pose6_to_mat(pose, rotation_repr):
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    if rotation_repr == "rotvec":
        return pose_to_mat(pose)
    if rotation_repr == "euler_xyz":
        tx = np.eye(4, dtype=np.float64)
        tx[:3, 3] = pose[:3]
        tx[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
        return tx
    raise ValueError(f"Unsupported rotation_repr: {rotation_repr}")


def transform_from_rvec_tvec(rvec, tvec):
    tx = np.eye(4)
    tx[:3, :3] = R.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()
    tx[:3, 3] = np.asarray(tvec).reshape(3)
    return tx


def rotation_error_deg(tx_err):
    return np.rad2deg(R.from_matrix(tx_err[:3, :3]).magnitude())


def summarize(name, values, unit):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        print(f"{name}: no data")
        return
    rms = np.sqrt(np.mean(values * values))
    print(
        f"{name}: mean={values.mean():.6g}{unit}, "
        f"median={np.median(values):.6g}{unit}, "
        f"rms={rms:.6g}{unit}, max={values.max():.6g}{unit}"
    )


def marker_object_points(marker_size):
    half = marker_size / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def project_marker_corners(tx_camera_marker, marker_size, intr):
    object_points = marker_object_points(marker_size).reshape(-1, 1, 3)
    rvec = np.ascontiguousarray(
        R.from_matrix(tx_camera_marker[:3, :3]).as_rotvec().reshape(3, 1),
        dtype=np.float64,
    )
    tvec = np.ascontiguousarray(tx_camera_marker[:3, 3].reshape(3, 1), dtype=np.float64)
    projected, _ = cv2.fisheye.projectPoints(
        object_points,
        rvec,
        tvec,
        intr["K"],
        intr["D"],
    )
    return projected.reshape(-1, 2)


def project_marker_corners_pinhole(tx_camera_marker, marker_size, K):
    object_points = marker_object_points(marker_size).reshape(-1, 1, 3)
    rvec = np.ascontiguousarray(
        R.from_matrix(tx_camera_marker[:3, :3]).as_rotvec().reshape(3, 1),
        dtype=np.float64,
    )
    tvec = np.ascontiguousarray(tx_camera_marker[:3, 3].reshape(3, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        np.zeros((5, 1), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def project_marker_corners_model(tx_camera_marker, marker_size, intr, detect_on_rectified):
    if detect_on_rectified:
        return project_marker_corners_pinhole(tx_camera_marker, marker_size, intr["K"])
    return project_marker_corners(tx_camera_marker, marker_size, intr)


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
            K,
            np.zeros((5, 1), dtype=np.float64),
        )
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


def draw_polyline(img, points, color, thickness=2):
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def save_overlays(overlay_dir, overlay_items, overlay_top_k):
    if overlay_dir is None:
        return
    os.makedirs(overlay_dir, exist_ok=True)
    selected = sorted(overlay_items, key=lambda x: x["trans_mm"], reverse=True)
    if overlay_top_k > 0:
        selected = selected[:overlay_top_k]

    for item in selected:
        img = item["img"].copy()
        # Script images are RGB; OpenCV drawing colors below are RGB tuples.
        draw_polyline(img, item["detected_corners"], (0, 255, 0), thickness=2)
        draw_polyline(img, item["detected_projected_corners"], (0, 128, 255), thickness=1)
        draw_polyline(img, item["predicted_projected_corners"], (255, 0, 0), thickness=2)
        text = (
            f"sample={item['sample_idx']} "
            f"trans={item['trans_mm']:.1f}mm rot={item['rot_deg']:.1f}deg "
            f"norm={item['norm_mm']:.1f}mm reproj={item['reproj_mean_px']:.2f}px"
        )
        cv2.putText(
            img,
            text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        out_path = os.path.join(
            overlay_dir,
            f"sample_{item['sample_idx']:03d}_trans_{item['trans_mm']:.1f}mm.png",
        )
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_sample_indices(path):
    if path is None:
        return None
    indices = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.add(int(line))
    return indices


@click.command()
@click.option("-i", "--input", "input_pkl", required=True)
@click.option("--hand_eye_json", required=True)
@click.option("--intr_json", required=True)
@click.option("--aruco_yaml", required=True)
@click.option("--tag_id", type=int, required=True)
@click.option("--overlay_dir", default=None, help="Optional directory for tag overlay images.")
@click.option("--overlay_top_k", type=int, default=8, help="Save worst K overlays by translation error; <=0 saves all.")
@click.option("--include_samples_file", default=None, help="Optional newline-delimited sample indices to evaluate.")
@click.option("--pose_key", default="tcp_pose", show_default=True, help="Robot pose key in the pickle, e.g. tcp_pose or ee_pose.")
@click.option("--detect_on_rectified", is_flag=True, help="Detect ArUco on rectified image and use rectified_K with zero distortion.")
@click.option("--detect_on_processed", is_flag=True, help="Alias for --detect_on_rectified.")
@click.option(
    "--aruco_source",
    type=click.Choice(["detect", "raw_aruco", "processed_aruco"]),
    default="detect",
    show_default=True,
    help="Use live detection, stored sample['aruco'], or stored sample['processed_aruco'].",
)
@click.option("--processed_overlay", is_flag=True, help="Draw overlays on processed/rectified images.")
@click.option("--rectify_balance", type=float, default=0.0, show_default=True)
@click.option("--rectify_fov_scale", type=float, default=1.0, show_default=True)
@click.option(
    "--pose_rotation_repr",
    type=click.Choice(["rotvec", "euler_xyz"]),
    default="rotvec",
    show_default=True,
    help="Rotation representation used by pose_key. ARX SDK ee_pose is euler_xyz.",
)
def main(
    input_pkl,
    hand_eye_json,
    intr_json,
    aruco_yaml,
    tag_id,
    overlay_dir,
    overlay_top_k,
    include_samples_file,
    pose_key,
    detect_on_rectified,
    detect_on_processed,
    aruco_source,
    processed_overlay,
    rectify_balance,
    rectify_fov_scale,
    pose_rotation_repr,
):
    samples = pickle.load(open(input_pkl, "rb"))
    include_indices = load_sample_indices(include_samples_file)
    raw_intr = parse_fisheye_intrinsics(json.load(open(intr_json, "r")))
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, "r")))
    result = json.load(open(hand_eye_json, "r"))

    # OpenCV calibrateRobotWorldHandEye returns:
    #   tx_base2world == ^w T_b
    #   tx_gripper2camera == ^c T_g
    # and solves A X = Z B, where:
    #   A == ^c T_w, X == ^w T_b, Z == ^c T_g, B == ^g T_b.
    tx_world_base = np.asarray(result["tx_base2world"], dtype=float)
    tx_camera_gripper = np.asarray(result["tx_gripper2camera"], dtype=float)
    marker_size = aruco_config["marker_size_map"][tag_id]
    rectifier = None
    if detect_on_processed:
        detect_on_rectified = True
    use_processed_image = detect_on_rectified or aruco_source == "processed_aruco" or processed_overlay
    if use_processed_image:
        example_img = samples[0]["img"]
        rectifier = build_rectifier(
            raw_intr,
            example_img.shape[:2][::-1],
            balance=rectify_balance,
            fov_scale=rectify_fov_scale,
        )
        print(f"detection_mode: {'stored_processed_aruco' if aruco_source == 'processed_aruco' else 'rectified'}")
        print(f"rectified_K: {rectifier['intr']['K'].tolist()}")
    else:
        print(f"detection_mode: {'stored_raw_aruco' if aruco_source == 'raw_aruco' else 'raw'}")
    print(f"aruco_source: {aruco_source}")

    closed_loop_translation_mm = []
    closed_loop_rotation_deg = []
    aruco_reprojection_px = []
    aruco_reprojection_per_sample = []
    per_sample = []
    overlay_items = []

    for i, sample in enumerate(samples):
        if include_indices is not None and i not in include_indices:
            continue
        if pose_key not in sample:
            raise KeyError(
                f"Sample {i} does not contain pose_key={pose_key!r}. "
                f"Available keys: {list(sample.keys())}"
            )
        img = sample["img"]
        use_pinhole = False
        if aruco_source in ("raw_aruco", "processed_aruco"):
            key = "aruco" if aruco_source == "raw_aruco" else "processed_aruco"
            tag = sample.get(key)
            if not tag or not tag.get("detected", True):
                continue
            if not all(k in tag for k in ("rvec", "tvec", "corners")):
                continue
            if aruco_source == "processed_aruco":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rectified_bgr = cv2.remap(
                    img_bgr,
                    rectifier["map1"],
                    rectifier["map2"],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                img_for_detection = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
                if "processed_K" in tag:
                    intr = {
                        "DIM": np.asarray(img_for_detection.shape[:2][::-1], dtype=np.int64),
                        "K": np.asarray(tag["processed_K"], dtype=np.float64),
                        "D": np.zeros((4, 1), dtype=np.float64),
                    }
                else:
                    intr = rectifier["intr"]
                use_pinhole = True
            else:
                img_for_detection = img
                intr = convert_fisheye_intrinsics_resolution(raw_intr, img.shape[:2][::-1])
            tag = {
                "rvec": np.asarray(tag["rvec"], dtype=np.float64).reshape(3),
                "tvec": np.asarray(tag["tvec"], dtype=np.float64).reshape(3),
                "corners": np.asarray(tag["corners"], dtype=np.float64).reshape(4, 2),
            }
        elif detect_on_rectified:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rectified_bgr = cv2.remap(
                img_bgr,
                rectifier["map1"],
                rectifier["map2"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            img_for_detection = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
            intr = rectifier["intr"]
            tag_dict = detect_localize_aruco_tags_pinhole(
                img_for_detection,
                aruco_config["aruco_dict"],
                aruco_config["marker_size_map"],
                intr["K"],
            )
            use_pinhole = True
            if tag_id not in tag_dict:
                continue
            tag = tag_dict[tag_id]
        else:
            img_for_detection = img
            intr = convert_fisheye_intrinsics_resolution(raw_intr, img.shape[:2][::-1])
            tag_dict = detect_localize_aruco_tags(
                img_for_detection,
                aruco_config["aruco_dict"],
                aruco_config["marker_size_map"],
                intr,
            )
            if tag_id not in tag_dict:
                continue
            tag = tag_dict[tag_id]
        tx_camera_world = transform_from_rvec_tvec(tag["rvec"], tag["tvec"])
        detected_projected_corners = project_marker_corners_model(
            tx_camera_world,
            marker_size,
            intr,
            use_pinhole,
        )
        reproj_err = np.linalg.norm(
            detected_projected_corners - np.asarray(tag["corners"], dtype=float),
            axis=1,
        )
        z_mm = float(np.asarray(tag["tvec"], dtype=float).reshape(3)[2] * 1000.0)
        norm_mm = float(np.linalg.norm(np.asarray(tag["tvec"], dtype=float).reshape(3)) * 1000.0)
        aruco_reprojection_px.extend(reproj_err.tolist())
        aruco_reprojection_per_sample.append((i, z_mm, norm_mm, float(reproj_err.mean()), float(reproj_err.max())))

        tx_base_gripper = pose6_to_mat(sample[pose_key], pose_rotation_repr)
        tx_gripper_base = np.linalg.inv(tx_base_gripper)

        left = tx_camera_world @ tx_world_base
        right = tx_camera_gripper @ tx_gripper_base
        tx_err = np.linalg.inv(left) @ right

        trans_mm = np.linalg.norm(tx_err[:3, 3]) * 1000.0
        rot_deg = rotation_error_deg(tx_err)
        closed_loop_translation_mm.append(trans_mm)
        closed_loop_rotation_deg.append(rot_deg)
        per_sample.append((i, trans_mm, rot_deg))

        if overlay_dir is not None:
            tx_base_world = np.linalg.inv(tx_world_base)
            tx_camera_world_pred = tx_camera_gripper @ tx_gripper_base @ tx_base_world
            overlay_items.append(
                {
                    "sample_idx": i,
                    "img": img_for_detection,
                    "detected_corners": tag["corners"],
                    "detected_projected_corners": project_marker_corners_model(
                        tx_camera_world,
                        marker_size,
                        intr,
                        use_pinhole,
                    ),
                    "predicted_projected_corners": project_marker_corners_model(
                        tx_camera_world_pred,
                        marker_size,
                        intr,
                        use_pinhole,
                    ),
                    "trans_mm": trans_mm,
                    "rot_deg": rot_deg,
                    "norm_mm": norm_mm,
                    "reproj_mean_px": float(reproj_err.mean()),
                }
            )

    print(f"samples_total: {len(samples)}")
    if include_indices is not None:
        print(f"include_samples_file: {include_samples_file}")
        print(f"samples_requested_by_include_file: {len(include_indices)}")
    print(f"samples_used_with_tag_{tag_id}: {len(per_sample)}")
    print(f"pose_key: {pose_key}")
    print(f"pose_rotation_repr: {pose_rotation_repr}")
    summarize("aruco_reproj_error_px", aruco_reprojection_px, " px")
    summarize("closed_loop_translation", closed_loop_translation_mm, " mm")
    summarize("closed_loop_rotation", closed_loop_rotation_deg, " deg")

    print("aruco_pnp_per_sample:")
    for i, z_mm, norm_mm, mean_px, max_px in aruco_reprojection_per_sample:
        print(
            f"  sample={i:03d}  z_mm={z_mm:.3f}  norm_mm={norm_mm:.3f} "
            f"reproj_mean={mean_px:.4f} px  reproj_max={max_px:.4f} px"
        )

    print("worst_aruco_reproj_samples_by_max:")
    for i, z_mm, norm_mm, mean_px, max_px in sorted(
        aruco_reprojection_per_sample, key=lambda x: x[4], reverse=True
    )[:5]:
        print(
            f"  sample={i:03d}  z_mm={z_mm:.3f}  norm_mm={norm_mm:.3f} "
            f"mean={mean_px:.4f} px  max={max_px:.4f} px"
        )

    print("worst_samples_by_translation:")
    for i, trans_mm, rot_deg in sorted(per_sample, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  sample={i:03d}  trans={trans_mm:.3f} mm  rot={rot_deg:.3f} deg")

    save_overlays(overlay_dir, overlay_items, overlay_top_k)
    if overlay_dir is not None:
        print(f"overlay_dir: {overlay_dir}")


if __name__ == "__main__":
    main()
