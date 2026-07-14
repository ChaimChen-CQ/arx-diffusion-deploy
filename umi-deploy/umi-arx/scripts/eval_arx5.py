"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import sys
import os
import json
from queue import Queue

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import math
import numpy as np
import scipy.spatial.transform as st
from omegaconf import OmegaConf
from utils.other_util import precise_wait
from peripherals.keystroke_counter import KeystrokeCounter, Key, KeyCode
from utils.real_inference_util import (
    get_real_obs_dict,
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
    make_runtime_pose_transform,
    convert_env_obs_to_policy_frame,
    convert_episode_start_pose_to_policy_frame,
    convert_policy_action_to_env_frame,
    get_camera_frame_umi_action,
)
from peripherals.spacemouse_shared_memory import Spacemouse
from utils.pose_util import pose_to_mat, mat_to_pose
from utils.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
from modules.arx5_controller import GripperControlPhase
from modules.arx5_env import Arx5Env
import zmq

OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_GRIPPER_FISHEYE_INTRINSICS = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "calibration",
        "cam0_sensor_intrinsics.json",
    )
)


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = (
        np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    )
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta


def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0])  # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(
                robots_config[this_robot_idx]["sphere_center"]
            )
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(
                robots_config[that_robot_idx]["sphere_center"]
            )
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = (
                robots_config[this_robot_idx]["sphere_radius"]
                + robots_config[that_robot_idx]["sphere_radius"]
            )
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print("avoid collision between two arms")
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(
                    this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local)
                )
                ee_poses[that_robot_idx][:6] = mat_to_pose(
                    np.linalg.inv(this_that_mat)
                    @ that_sphere_mat_global
                    @ np.linalg.inv(that_sphere_mat_local)
                )


def _load_structured_config(config_path):
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext == ".json":
            payload = json.load(f)
        else:
            payload = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    if payload is None:
        payload = dict()
    if "runtime_pose_transform" in payload:
        payload = payload["runtime_pose_transform"]
    return payload


def _get_optional_transform(payload, candidate_keys):
    for key in candidate_keys:
        value = payload.get(key, None)
        if value is None:
            continue
        return np.asarray(value, dtype=np.float64), key
    return None, None


def load_runtime_pose_transform(runtime_calibration):
    if runtime_calibration is None:
        return make_runtime_pose_transform(), dict()

    payload = _load_structured_config(runtime_calibration)
    tx_policy_frame_from_env_base, tx_policy_frame_from_env_base_key = (
        _get_optional_transform(
            payload,
            [
                "tx_policy_frame_from_arx_base",
                "tx_policy_frame_from_env_base",
                "tx_base2world",
            ],
        )
    )
    tx_env_base_from_policy_frame, tx_env_base_from_policy_frame_key = (
        _get_optional_transform(
            payload,
            [
                "tx_arx_base_from_policy_frame",
                "tx_env_base_from_policy_frame",
                "tx_world2base",
            ],
        )
    )
    tx_policy_tcp_from_env_tcp, tx_policy_tcp_from_env_tcp_key = (
        _get_optional_transform(
            payload,
            [
                "tx_policy_tcp_from_arx_tcp",
                "tx_policy_tcp_from_env_tcp",
            ],
        )
    )
    tx_env_tcp_from_policy_tcp, tx_env_tcp_from_policy_tcp_key = (
        _get_optional_transform(
            payload,
            [
                "tx_arx_tcp_from_policy_tcp",
                "tx_env_tcp_from_policy_tcp",
            ],
        )
    )
    tx_env_tcp_camera, tx_env_tcp_camera_key = _get_optional_transform(
        payload,
        [
            "tx_arx_tcp_camera",
            "tx_env_tcp_camera",
        ],
    )
    tx_camera_policy_tcp, tx_camera_policy_tcp_key = _get_optional_transform(
        payload,
        [
            "tx_camera_policy_tcp",
            "tx_cam_tcp",
        ],
    )
    tx_gripper2camera, tx_gripper2camera_key = _get_optional_transform(
        payload, ["tx_gripper2camera"]
    )
    tx_camera2gripper, tx_camera2gripper_key = _get_optional_transform(
        payload, ["tx_camera2gripper"]
    )
    if tx_gripper2camera is None and tx_camera2gripper is not None:
        tx_gripper2camera = np.linalg.inv(tx_camera2gripper)
        tx_gripper2camera_key = tx_camera2gripper_key
    if tx_env_tcp_camera is None and tx_gripper2camera is not None:
        tx_env_tcp_camera = np.linalg.inv(tx_gripper2camera)
        tx_env_tcp_camera_key = "inv(tx_gripper2camera)"

    runtime_pose_transform = make_runtime_pose_transform(
        tx_policy_frame_from_env_base=tx_policy_frame_from_env_base,
        tx_env_base_from_policy_frame=tx_env_base_from_policy_frame,
        tx_policy_tcp_from_env_tcp=tx_policy_tcp_from_env_tcp,
        tx_env_tcp_from_policy_tcp=tx_env_tcp_from_policy_tcp,
        tx_env_tcp_camera=tx_env_tcp_camera,
        tx_camera_policy_tcp=tx_camera_policy_tcp,
        action_reference_frame=payload.get("action_reference_frame", "policy"),
        source_path=runtime_calibration,
        reference_tx_gripper2camera=tx_gripper2camera,
    )
    loaded_keys = {
        "tx_policy_frame_from_env_base_key": tx_policy_frame_from_env_base_key,
        "tx_env_base_from_policy_frame_key": tx_env_base_from_policy_frame_key,
        "tx_policy_tcp_from_env_tcp_key": tx_policy_tcp_from_env_tcp_key,
        "tx_env_tcp_from_policy_tcp_key": tx_env_tcp_from_policy_tcp_key,
        "tx_env_tcp_camera_key": tx_env_tcp_camera_key,
        "tx_camera_policy_tcp_key": tx_camera_policy_tcp_key,
        "action_reference_frame": runtime_pose_transform.action_reference_frame,
        "tx_gripper2camera_key": tx_gripper2camera_key,
    }
    return runtime_pose_transform, loaded_keys


def load_gripper_fisheye_intrinsics(intrinsics_path):
    if intrinsics_path is None:
        raise ValueError(
            "ARX5 runtime requires explicit UMI gripper fisheye intrinsics; "
            "pass --gripper_fisheye_intrinsics."
        )
    with open(intrinsics_path, "r") as f:
        intrinsics = parse_fisheye_intrinsics(json.load(f))
    capture_resolution = tuple(int(value) for value in intrinsics["DIM"])
    if capture_resolution[0] <= 0 or capture_resolution[1] <= 0:
        raise ValueError(
            f"Invalid gripper fisheye intrinsics resolution from {intrinsics_path}: "
            f"{capture_resolution}"
        )
    return intrinsics, capture_resolution


def extract_robot_pose_from_obs(obs, robot_id=0):
    return np.concatenate(
        [
            obs[f"robot{robot_id}_eef_pos"][-1],
            obs[f"robot{robot_id}_eef_rot_axis_angle"][-1],
        ],
        axis=-1,
    )


def summarize_runtime_pose_transform(runtime_pose_transform, loaded_keys):
    print("[RUNTIME_XFORM] enabled:", runtime_pose_transform.enabled)
    if runtime_pose_transform.source_path is not None:
        print("[RUNTIME_XFORM] source:", runtime_pose_transform.source_path)
    if loaded_keys:
        print("[RUNTIME_XFORM] loaded_keys:", loaded_keys)
    print(
        "[RUNTIME_XFORM] tx_policy_frame_from_env_base:\n",
        runtime_pose_transform.tx_policy_frame_from_env_base,
    )
    print(
        "[RUNTIME_XFORM] tx_policy_tcp_from_env_tcp:\n",
        runtime_pose_transform.tx_policy_tcp_from_env_tcp,
    )
    print(
        "[RUNTIME_XFORM] tx_env_tcp_camera:\n",
        runtime_pose_transform.tx_env_tcp_camera,
    )
    print(
        "[RUNTIME_XFORM] tx_camera_policy_tcp:\n",
        runtime_pose_transform.tx_camera_policy_tcp,
    )
    print(
        "[RUNTIME_XFORM] action_reference_frame:",
        runtime_pose_transform.action_reference_frame,
    )
    if runtime_pose_transform.reference_tx_gripper2camera is not None:
        print(
            "[RUNTIME_XFORM] tx_gripper2camera loaded as OpenCV T_camera_tcp. "
            "Runtime uses inv(tx_gripper2camera) as T_arx_tcp_camera unless "
            "tx_arx_tcp_camera is explicitly provided.\n",
            runtime_pose_transform.reference_tx_gripper2camera,
        )


def log_runtime_transform_step(
    raw_arx_obs,
    policy_obs,
    raw_policy_action,
    policy_action,
    converted_arx_action,
    final_tcp_pose_cmd,
    runtime_pose_transform=None,
):
    raw_arx_pose = extract_robot_pose_from_obs(raw_arx_obs)
    converted_policy_pose = extract_robot_pose_from_obs(policy_obs)
    if runtime_pose_transform is not None:
        print(
            "[RUNTIME_XFORM] action_reference_frame:",
            runtime_pose_transform.action_reference_frame,
        )
    print(
        "[RUNTIME_XFORM] raw_arx_pose:",
        np.round(raw_arx_pose, 4).tolist(),
    )
    print(
        "[RUNTIME_XFORM] converted_policy_pose:",
        np.round(converted_policy_pose, 4).tolist(),
        )
    if (
        runtime_pose_transform is not None
        and runtime_pose_transform.uses_camera_frame_action
    ):
        tx_base_tcp = pose_to_mat(raw_arx_pose)
        tx_base_camera = tx_base_tcp @ runtime_pose_transform.tx_env_tcp_camera
        print(
            "[RUNTIME_XFORM] current_tx_base_camera:\n",
            np.round(tx_base_camera, 4),
        )
        print(
            "[FRAME_DEBUG] camera_axes_in_base x/y/z:",
            np.round(tx_base_camera[:3, 0], 4).tolist(),
            np.round(tx_base_camera[:3, 1], 4).tolist(),
            np.round(tx_base_camera[:3, 2], 4).tolist(),
        )
        if raw_policy_action is not None:
            raw_xyz = np.asarray(raw_policy_action[0, :3], dtype=np.float64)
            raw_xyz_as_base = tx_base_camera[:3, :3] @ raw_xyz
            print(
                "[FRAME_DEBUG] raw_action_xyz_as_camera_delta:",
                np.round(raw_xyz, 5).tolist(),
            )
            print(
                "[FRAME_DEBUG] raw_camera_delta_rotated_to_base:",
                np.round(raw_xyz_as_base, 5).tolist(),
            )
        if final_tcp_pose_cmd is not None:
            final_delta_base = (
                np.asarray(final_tcp_pose_cmd[0, :3], dtype=np.float64)
                - raw_arx_pose[:3]
            )
            print(
                "[FRAME_DEBUG] final_cmd_delta_base:",
                np.round(final_delta_base, 5).tolist(),
            )
            if raw_policy_action is not None:
                print(
                    "[FRAME_DEBUG] final_minus_raw_xyz_delta:",
                    np.round(final_delta_base - raw_xyz_as_base, 5).tolist(),
                )
    if raw_policy_action is not None:
        print(
            "[RUNTIME_XFORM] raw_policy_action[0]:",
            np.round(raw_policy_action[0], 4).tolist(),
        )
    if policy_action is not None:
        policy_delta = (
            np.asarray(policy_action[0, :3], dtype=np.float64)
            - converted_policy_pose[:3]
        )
        print(
            "[FRAME_DEBUG] policy_action_delta_policy_frame:",
            np.round(policy_delta, 5).tolist(),
        )
    if policy_action is not None:
        print(
            "[RUNTIME_XFORM] policy_action[0]:",
            np.round(policy_action[0], 4).tolist(),
        )
    if converted_arx_action is not None:
        print(
            "[RUNTIME_XFORM] converted_arx_action[0]:",
            np.round(converted_arx_action[0], 4).tolist(),
        )
    if final_tcp_pose_cmd is not None:
        final_delta_base = (
            np.asarray(final_tcp_pose_cmd[0, :3], dtype=np.float64)
            - raw_arx_pose[:3]
        )
        print(
            "[FRAME_DEBUG] final_cmd_delta_base:",
            np.round(final_delta_base, 5).tolist(),
        )
        print(
            "[RUNTIME_XFORM] final_tcp_pose_cmd[0]:",
            np.round(final_tcp_pose_cmd[0], 4).tolist(),
        )


def log_policy_obs_image_stats(obs_dict_np, prefix="[OBS_DEBUG]"):
    for key, value in obs_dict_np.items():
        if not key.endswith("_rgb"):
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            print(f"{prefix} {key}: empty")
            continue
        stats = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
        if arr.shape[0] >= 2:
            stats["temporal_absdiff_mean"] = float(np.mean(np.abs(arr[-1] - arr[-2])))
        print(f"{prefix} {key}:", stats)


def log_action_chunk_summary(raw_arx_obs, target_poses, prefix="[CHUNK_DEBUG]"):
    if target_poses is None or len(target_poses) == 0:
        print(f"{prefix} empty target chunk")
        return
    raw_arx_pose = extract_robot_pose_from_obs(raw_arx_obs)
    target_poses = np.asarray(target_poses, dtype=np.float64)
    deltas = target_poses[:, :3] - raw_arx_pose[:3]
    idxs = sorted(set([0, len(deltas) // 2, len(deltas) - 1]))
    sample = {
        int(i): np.round(deltas[i], 5).tolist()
        for i in idxs
    }
    print(
        f"{prefix} count={len(deltas)} "
        f"delta_xyz first/mid/last={sample} "
        f"min={np.round(np.min(deltas, axis=0), 5).tolist()} "
        f"max={np.round(np.max(deltas, axis=0), 5).tolist()}"
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def convert_offset_vector(offset, input_frame, tx_base_tcp, tx_base_camera):
    frame_rot_in_base = {
        "base": np.eye(3),
        "tcp": tx_base_tcp[:3, :3],
        "camera": tx_base_camera[:3, :3],
    }
    offset = np.asarray(offset, dtype=np.float64)
    offset_base = frame_rot_in_base[input_frame] @ offset
    return {
        "base": offset_base,
        "tcp": frame_rot_in_base["tcp"].T @ offset_base,
        "camera": frame_rot_in_base["camera"].T @ offset_base,
    }


def convert_point_to_base(point, input_frame, tx_base_tcp, tx_base_camera):
    point = np.asarray(point, dtype=np.float64)
    if input_frame == "base":
        return point
    if input_frame == "tcp":
        return tx_base_tcp[:3, :3] @ point + tx_base_tcp[:3, 3]
    if input_frame == "camera":
        return tx_base_camera[:3, :3] @ point + tx_base_camera[:3, 3]
    raise ValueError(f"Unsupported input_frame: {input_frame}")


def parse_bias_input(text):
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != 3:
        raise ValueError("Expected exactly three numbers: dx dy dz")
    return np.asarray(values, dtype=np.float64)


def apply_action_z_bias(action, action_z_bias):
    """Apply a fixed ARX-base Z offset to every robot target pose."""
    action_arr = np.asarray(action)
    if action_arr.shape[-1] % 7 != 0:
        raise ValueError(
            "Expected action last dimension to be a multiple of 7 "
            f"(pose6 + gripper1), got {action_arr.shape[-1]}"
        )

    biased_action = np.array(action_arr, copy=True)
    biased_action[..., 2::7] += action_z_bias
    return biased_action


def make_runtime_bias_snapshot(
    episode_id,
    iter_idx,
    raw_arx_obs,
    policy_obs,
    raw_policy_action,
    policy_action,
    converted_arx_action,
    final_tcp_pose_cmd,
    runtime_pose_transform,
):
    raw_arx_pose = extract_robot_pose_from_obs(raw_arx_obs)
    tx_base_tcp = pose_to_mat(raw_arx_pose)
    tx_base_camera = tx_base_tcp @ runtime_pose_transform.tx_env_tcp_camera
    return {
        "timestamp": time.time(),
        "episode_id": int(episode_id),
        "iter_idx": int(iter_idx),
        "raw_arx_pose": raw_arx_pose,
        "converted_policy_pose": extract_robot_pose_from_obs(policy_obs),
        "current_tx_base_tcp": tx_base_tcp,
        "current_tx_base_camera": tx_base_camera,
        "raw_policy_action": raw_policy_action,
        "policy_action": policy_action,
        "converted_arx_action": converted_arx_action,
        "final_tcp_pose_cmd": final_tcp_pose_cmd,
        "runtime_pose_transform": runtime_pose_transform.to_debug_dict(),
    }


def attach_manual_bias(
    snapshot,
    vector,
    input_frame,
    input_mode,
):
    tx_base_tcp = np.asarray(snapshot["current_tx_base_tcp"], dtype=np.float64)
    tx_base_camera = np.asarray(snapshot["current_tx_base_camera"], dtype=np.float64)
    final_tcp_pose_cmd = np.asarray(snapshot["final_tcp_pose_cmd"], dtype=np.float64)
    final_tcp_pos = final_tcp_pose_cmd[-1, :3]

    if input_mode == "offset":
        offsets = convert_offset_vector(vector, input_frame, tx_base_tcp, tx_base_camera)
        residual_base = offsets["base"]
    elif input_mode == "object_center":
        object_center_base = convert_point_to_base(
            vector, input_frame, tx_base_tcp, tx_base_camera
        )
        residual_base = object_center_base - final_tcp_pos
        offsets = convert_offset_vector(
            residual_base, "base", tx_base_tcp, tx_base_camera
        )
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    snapshot["manual_bias"] = {
        "input_mode": input_mode,
        "input_frame": input_frame,
        "input_vector": vector,
        "residual_base": residual_base,
        "residual_tcp": offsets["tcp"],
        "residual_camera": offsets["camera"],
    }
    return snapshot


def save_runtime_bias_snapshot(
    output,
    snapshot,
    prompt_bias_on_stop,
    bias_input_frame,
    bias_input_mode,
):
    if snapshot is None:
        return None

    record = dict(snapshot)
    if prompt_bias_on_stop:
        prompt = (
            f"[BIAS] 输入 {bias_input_frame} frame 下的 "
            f"{bias_input_mode} dx dy dz，单位米；空行跳过: "
        )
        try:
            text = input(prompt).strip()
        except EOFError:
            text = ""
        if text:
            try:
                vector = parse_bias_input(text)
                record = attach_manual_bias(
                    record,
                    vector=vector,
                    input_frame=bias_input_frame,
                    input_mode=bias_input_mode,
                )
                manual_bias = record["manual_bias"]
                print(
                    "[BIAS] residual_base:",
                    np.round(manual_bias["residual_base"], 5).tolist(),
                )
                print(
                    "[BIAS] residual_camera:",
                    np.round(manual_bias["residual_camera"], 5).tolist(),
                )
                print(
                    "[BIAS] residual_tcp:",
                    np.round(manual_bias["residual_tcp"], 5).tolist(),
                )
            except ValueError as e:
                record["manual_bias_error"] = str(e)
                print(f"[BIAS] 输入无效，保存 snapshot 但不保存 bias: {e}")

    bias_dir = os.path.join(output, "bias_records")
    os.makedirs(bias_dir, exist_ok=True)
    path = os.path.join(
        bias_dir,
        "episode_{:04d}_iter_{:06d}.json".format(
            record["episode_id"], record["iter_idx"]
        ),
    )
    with open(path, "w") as f:
        json.dump(_jsonable(record), f, indent=2)
    print(f"[BIAS] saved runtime bias snapshot: {path}")
    return path


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--config", "-c", default=None, help="Path to specific yaml config file")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option("--policy_ip", default="localhost")
@click.option("--policy_port", default=8766)
@click.option(
    "--match_dataset",
    "-m",
    default=None,
    help="Dataset used to overlay and adjust initial condition",
)
@click.option(
    "--match_episode",
    "-me",
    default=None,
    type=int,
    help="Match specific episode from the match dataset",
)
@click.option("--match_camera", "-mc", default=0, type=int)
@click.option("--camera_reorder", "-cr", default="0")
@click.option(
    "--vis_camera_idx", default=0, type=int, help="Which RealSense camera to visualize."
)
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=False,
    help="Whether to initialize robot joint configuration in the beginning.",
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=16,
    type=int,
    help="Action horizon for inference.",
)
@click.option(
    "--max_duration",
    "-md",
    default=2000000,
    help="Max duration for each epoch in seconds.",
)
@click.option(
    "--frequency", "-f", default=8, type=float, help="Control frequency in Hz."
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.25,
    type=float,
    help="Seconds to schedule robot actions into the future.",
)
@click.option(
    "--disable_dynamic_latency",
    is_flag=True,
    default=False,
    help="Disable trajectory phase matching and honor scheduled action timestamps.",
)
@click.option("-nm", "--no_mirror", is_flag=True, default=False)
@click.option("-sf", "--sim_fov", type=float, default=None)
@click.option(
    "-ci",
    "--camera_intrinsics",
    type=str,
    default=None,
    help="Deprecated alias for --gripper_fisheye_intrinsics.",
)
@click.option(
    "--gripper_fisheye_intrinsics",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULT_GRIPPER_FISHEYE_INTRINSICS,
    show_default=True,
    help="UMI gripper built-in fisheye intrinsics JSON. Runtime capture resolution is read from this file.",
)
@click.option(
    "--runtime_calibration",
    type=str,
    default=None,
    help="Path to runtime pose calibration json/yaml.",
)
@click.option(
    "--action_z_bias",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Fixed Z offset in meters applied to every target action in the ARX base "
        "frame after pose conversion. Negative values move targets downward."
    ),
)
@click.option("--mirror_swap", is_flag=True, default=False)
@click.option(
    "--log_runtime_transforms/--no_log_runtime_transforms",
    default=True,
    help="Print runtime pose/action conversion debug information when enabled.",
)
@click.option(
    "--record_bias_on_stop/--no_record_bias_on_stop",
    default=True,
    help="Save final pose/camera/action snapshot when policy control stops.",
)
@click.option(
    "--prompt_bias_on_stop",
    is_flag=True,
    default=False,
    help="Prompt for manual dx dy dz bias when policy control stops.",
)
@click.option(
    "--bias_input_frame",
    type=click.Choice(["base", "camera", "tcp"]),
    default="base",
    help="Frame for manual bias input.",
)
@click.option(
    "--bias_input_mode",
    type=click.Choice(["offset", "object_center"]),
    default="offset",
    help="Interpret manual input as an offset vector or an object center point.",
)
@click.option(
    "--disable_video_recording",
    is_flag=True,
    default=False,
    help="Disable mp4 recording and keep only live policy frames in shared memory.",
)
@click.option(
    "--dry_run_policy",
    is_flag=True,
    default=False,
    help="Run inference and print converted actions without sending them to the robot.",
)
@click.option("--no_spacemouse", is_flag=True, default=False, help="Disable SpaceMouse connection if no hardware exists.")
def main(
    input,
    config,
    output,
    policy_ip,
    policy_port,
    match_dataset,
    match_episode,
    match_camera,
    camera_reorder,
    vis_camera_idx,
    init_joints,
    steps_per_inference,
    max_duration,
    frequency,
    command_latency,
    disable_dynamic_latency,
    no_mirror,
    sim_fov,
    camera_intrinsics,
    gripper_fisheye_intrinsics,
    runtime_calibration,
    action_z_bias,
    mirror_swap,
    log_runtime_transforms,
    record_bias_on_stop,
    prompt_bias_on_stop,
    bias_input_frame,
    bias_input_mode,
    disable_video_recording,
    dry_run_policy,
    no_spacemouse,
):
    pid = os.getpid()
    # os.sched_setaffinity(pid, [7]) # FIX: Do not pin entire multiprocessing tree to one core, causes USB V4L2 timeouts
    # Gen gripper target_distance / encoder are both documented in [0.0, 0.103] m.
    # Keeping the host-side limit aligned with the device protocol avoids silently
    # clamping a valid live gripper state (for example ~0.098 m) back to 0.085/0.082.
    max_gripper_width = 0.103
    gripper_speed = 0.02
    cartesian_speed = 0.4
    orientation_speed = 0.8

    if action_z_bias != 0.0:
        print(
            f"[ACTION_BIAS] Applying ARX-base Z bias to all policy targets: "
            f"{action_z_bias:+.6f} m"
        )

    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "obs"), exist_ok=True)
    os.makedirs(os.path.join(output, "action"), exist_ok=True)
    os.makedirs(os.path.join(output, "bias_records"), exist_ok=True)

    # 双臂才有用！！！！！！！！！！！！！！！！！！！！
    # 使用自己手眼标定得到的 4x4 变换矩阵 ------------ L5
    tx_left_right = np.array(
        [
            [ 0.99782674, -0.04947446, -0.04352101, -0.00405455],
            [ 0.05866956,  0.96774069,  0.24502213, -0.0777969 ],
            [ 0.02999471, -0.24704299,  0.96854018,  0.10988608],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ]
    )
    # # 使用自己手眼标定得到的 4x4 变换矩阵 ------------ L5_assembly
    # tx_left_right = np.array(
    #     [
    #         [ 0.99474746, -0.10135301,  0.01431954, -0.01124722],
    #         [ 0.09772664,  0.98199573,  0.16165979, -0.07346948],
    #         [-0.03044643, -0.15941127,  0.98674265, -0.03313451],
    #         [ 0.0,         0.0,         0.0,         1.0       ]
    #     ]
    # )
    tx_robot1_robot0 = tx_left_right

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    # --- 新增的自定义配置加载逻辑 ---
    if config is not None:
        cfg_path = config
    else:
        cfg_path = ckpt_path.replace(".ckpt", ".yaml")
        
    print(f"Loading config from: {cfg_path}") # 打印出来确认一下
    # ---------------------------------
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.load(f)
    # import torch
    # payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    # cfg = payload['cfg']
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    print("obs_pose_rep", obs_pose_rep)
    print("action_pose_repr", action_pose_repr)
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1 / frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    if camera_intrinsics is not None:
        print(
            "[CAMERA] --camera_intrinsics is deprecated; "
            "using it as --gripper_fisheye_intrinsics."
        )
        gripper_fisheye_intrinsics = camera_intrinsics

    opencv_intr_dict, gripper_camera_resolution = load_gripper_fisheye_intrinsics(
        gripper_fisheye_intrinsics
    )
    print(
        "[CAMERA] gripper_fisheye_intrinsics:",
        gripper_fisheye_intrinsics,
        "resolution:",
        f"{gripper_camera_resolution[0]}x{gripper_camera_resolution[1]}",
    )

    fisheye_converter = None
    if sim_fov is not None:
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov,
        )
        print(
            "[RUNTIME_XFORM] fisheye rectification enabled with",
            gripper_fisheye_intrinsics,
        )

    runtime_pose_transform, runtime_pose_transform_keys = load_runtime_pose_transform(
        runtime_calibration
    )
    summarize_runtime_pose_transform(
        runtime_pose_transform, runtime_pose_transform_keys
    )

    robots_config = [
        {
            "robot_type": "arx5",
            "robot_ip": "127.0.0.1",
            "robot_port": 8765,
            "robot_obs_latency": 0.005,  # TODO: need to measure
            "robot_action_latency": 0.04,  # TODO: need to measure
            "height_threshold": -0.2,  # TODO: ncscseed to measure
            "sphere_radius": 0.1,  # TODO: need to measure
            "sphere_center": [0, -0.06, -0.185],  # TODO: need to measure
        }
    ]
    if runtime_pose_transform.enabled and len(robots_config) != 1:
        raise ValueError(
            "Runtime pose calibration currently supports single-arm ARX5 only."
        )

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{policy_ip}:{policy_port}")

    # ===== 安全终止与姿态规划常量区 =====

    # Joint-space initial/rest targets for ARX Python SDK control.
    INIT_JOINT_POS = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    REST_JOINT_POS = np.zeros(6)
    ENABLE_JOINT_HOTKEYS = True
    ENABLE_JOINT_REST_ON_EXIT = False
    JOINT_MOVE_DURATION_S = 2.5
    JOINT_MAX_STEP_RAD = 1.6

    # 真实示教获取的 15° 俯角完美绝对观察坐标
    # OBS_POSE = np.array([ 0.2822,  0.0005,  0.1973, -1.3598,  1.3505, -1.1064]) # L5
    # OBS_POSE = np.array([ 0.1197, -0.0002,  0.2411, -1.3549,  1.351 , -1.1128]) # L5-assembly
    OBS_POSE = np.array([ 0.124 , -0.0002,  0.2129,  2.524 , -2.5074,  1.8395])

    GRIPPER_INIT = 0.0819
    # JOINT_POS = np.array([-0.0029,  0.4217,  0.3058, -0.1905, -0.0074, -0.004 ])

    # 真实示教的安全休眠位，用于退出时降低重心防碰撞
    # REST_POSE = np.array([ 0.252 ,  0.0003,  0.1549, -1.2431,  1.2411, -1.1859]) # L5
    REST_POSE = np.array([ 0.1001, -0.0002,  0.1587, -1.2507,  1.2467, -1.1858]) # L5-assembly
    REST_GRIPPER = 0.0688

    def safe_teardown(env_obj, s_obj, c_obj):
        print("\n[SAFE TEARDOWN] 触发安全退出流程。")
        robot_states = None
        try:
            robot_states = env_obj.get_robot_state()
        except Exception as e:
            print(f"[SAFE TEARDOWN] 读取当前夹爪状态失败，回退到默认 REST_GRIPPER: {e}")
        for idx in range(len(env_obj.robots)):
            if not env_obj.robots[idx].is_alive():
                print(f"[SAFE TEARDOWN] 机器人 {idx} 控制进程已异常终止，跳过平滑趴伏操作！")
                continue
            gripper_target = REST_GRIPPER
            if robot_states is not None and idx < len(robot_states):
                try:
                    gripper_target = float(robot_states[idx]["gripper_position"])
                except Exception:
                    pass
            if ENABLE_JOINT_REST_ON_EXIT:
                print(f"[SAFE TEARDOWN] 机器人 {idx} 正在移动至 joint rest 全 0...")
                env_obj.robots[idx].set_joint_pos(
                    REST_JOINT_POS,
                    gripper_target,
                    duration=2.0,
                    max_joint_step_rad=np.pi,
                    timeout=4.0,
                )
            else:
                print(
                    f"[SAFE TEARDOWN] joint rest disabled; "
                    f"机器人 {idx} 保持当前目标并退出。"
                )
            
        print("[SAFE TEARDOWN] 强制等待 2 秒以确保机械臂已趴稳...")
        time.sleep(2.0)
        
        if s_obj:
            print("[SAFE TEARDOWN] 姿态已锁定，安全断开 ZMQ Socket 通信流...")
            s_obj.close()
            
        print("[SAFE TEARDOWN] 资源释放完毕，底层进程将自动进入 Damping (瘫软) 状态，安全退出。")
    # ====================================

    print("steps_per_inference:", steps_per_inference)
    max_camera_frame_staleness = max(1.0, 2.0 * steps_per_inference * dt)
    camera_warmup_timeout = 10.0

    def end_episode_safely(env_obj):
        if env_obj is None:
            return
        try:
            env_obj.end_episode()
        except Exception as e:
            print(f"[WARN] end_episode during cleanup failed: {e}")

    def fatal_exit(env_obj, message):
        print(f"[FATAL] {message}")
        end_episode_safely(env_obj)
        if env_obj is not None:
            safe_teardown(env_obj, socket, context)
        else:
            if socket is not None:
                socket.close()
        raise SystemExit(1)

    def fail_if_camera_unhealthy(
        env_obj, phase, min_ring_buffer_count=None, max_frame_staleness=None
    ):
        try:
            env_obj.assert_camera_healthy(
                min_ring_buffer_count=min_ring_buffer_count,
                max_frame_staleness=max_frame_staleness,
            )
        except RuntimeError as e:
            status_lines = list()
            for status in env_obj.get_camera_health():
                stale_for = status["stale_for"]
                stale_repr = "n/a" if stale_for is None else f"{stale_for:.3f}s"
                status_lines.append(
                    "camera{camera_idx}: alive={process_alive}, ready={ready}, "
                    "recovering={recovering}, failed={failed}, count={ring_buffer_count}, "
                    "stale={stale}, path={dev_video_path}".format(
                        camera_idx=status["camera_idx"],
                        process_alive=status["process_alive"],
                        ready=status["ready"],
                        recovering=status["recovering"],
                        failed=status["failed"],
                        ring_buffer_count=status["ring_buffer_count"],
                        stale=stale_repr,
                        dev_video_path=status["dev_video_path"],
                    )
                )
            detail = "\n".join(status_lines)
            fatal_exit(env_obj, f"{phase}: {e}\n{detail}")

    policy_camera_keys = [
        key
        for key, attr in cfg.task.shape_meta.obs.items()
        if attr.get("type", "low_dim") == "rgb"
        and not attr.get("ignore_by_policy", False)
    ]
    if len(policy_camera_keys) == 0:
        raise ValueError("No RGB observations are enabled for policy visualization.")

    def get_policy_vis_image(obs):
        vis_keys = [key for key in policy_camera_keys if key in obs]
        if len(vis_keys) == 0:
            raise KeyError(
                "None of the policy RGB observation keys were found in env obs: "
                f"{policy_camera_keys}"
            )
        vis_imgs = [obs[key][-1] for key in vis_keys]
        if len(vis_imgs) == 1:
            vis_img = vis_imgs[0]
        else:
            vis_img = np.concatenate(vis_imgs, axis=1)
        if vis_img.dtype != np.uint8:
            vis_img = np.clip(vis_img, 0.0, 1.0)
            vis_img = (vis_img * 255).astype(np.uint8)
        return vis_img, vis_keys

    def set_gripper_control_phase(env_obj, phase: GripperControlPhase):
        env_obj.set_gripper_control_phase(phase)
        print(f"[GRIPPER] Control phase -> {phase.name}")

    class DummySpacemouse:
        def __init__(self, *args, **kwargs): pass
        def set_key_counter(self, kc): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def get_motion_state_transformed(self): return np.zeros(6)
        def is_button_pressed(self, button_id): return False

    with SharedMemoryManager() as shm_manager:
        SpacemouseClass = DummySpacemouse if no_spacemouse else Spacemouse
        with SpacemouseClass(
            shm_manager=shm_manager, deadzone=0.1
        ) as sm, KeystrokeCounter() as key_counter, Arx5Env(
            output_dir=output,
            robots_config=robots_config,
            frequency=frequency,
            obs_image_resolution=obs_res,
            obs_float32=True,
            camera_reorder=[int(x) for x in camera_reorder],
            init_joints=init_joints,
            enable_multi_cam_vis=False,
            # latency
            camera_obs_latency=0.17,
            # obs
            camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            no_mirror=no_mirror,
            fisheye_converter=fisheye_converter,
            gripper_camera_resolution=gripper_camera_resolution,
            enable_video_recording=not disable_video_recording,
            mirror_swap=mirror_swap,
            # action
            max_pos_speed=2.0,
            max_rot_speed=6.0,
            shm_manager=shm_manager,
        ) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(3.0)

            print("Waiting for env ready.")
            while not env.is_ready:
                time.sleep(0.1)
            print("Env is ready")
            set_gripper_control_phase(env, GripperControlPhase.PRE_POLICY_HOLD)

            # Wait for camera ring buffer to accumulate enough frames.
            # get_obs() requests k frames; the camera ready_event fires after
            # just 1 frame, so we must wait for the buffer to fill.
            k_needed = (
                math.ceil(
                    cfg.task.shape_meta.obs.camera0_rgb.horizon
                    * 1  # camera_down_sample_steps
                    * (60 / frequency)
                )
                + 2
            )
            print(f"Waiting for camera ring buffer to have >= {k_needed} frames...")
            counts = []
            warmup_deadline = time.time() + camera_warmup_timeout
            while True:
                fail_if_camera_unhealthy(env, "camera warmup")
                counts = []
                for cam in env.camera.cameras.values():
                    counts.append(cam.ring_buffer.count)
                if all(c >= k_needed for c in counts):
                    break
                if time.time() > warmup_deadline:
                    fatal_exit(
                        env,
                        "Timed out waiting for camera ring buffer to fill. "
                        f"Need >= {k_needed} frames, got counts={counts}",
                    )
                time.sleep(0.1)
            print(f"Camera ring buffer counts: {counts}")

            video_paths = []
            if not disable_video_recording:
                print(f"Warming up video recording")
                video_dir = env.video_dir.joinpath("test")
                video_dir.mkdir(exist_ok=True, parents=True)
                n_cameras = env.camera.n_cameras
                for i in range(n_cameras):
                    video_path = str(video_dir.joinpath(f"{i}.mp4").absolute())
                    video_paths.append(video_path)
                env.camera.start_recording(video_path=video_paths, start_time=time.time())
            else:
                print("Video recording disabled; skipping recording warmup.")

            print(f"Warming up policy inference")
            fail_if_camera_unhealthy(
                env,
                "policy warmup",
                min_ring_buffer_count=k_needed,
                max_frame_staleness=max_camera_frame_staleness,
            )
            obs = env.get_obs()
            print(obs)
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate(
                    [
                        obs[f"robot{robot_id}_eef_pos"],
                        obs[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )[-1]
                episode_start_pose.append(pose)
            policy_obs = convert_env_obs_to_policy_frame(obs, runtime_pose_transform)
            policy_episode_start_pose = convert_episode_start_pose_to_policy_frame(
                episode_start_pose, runtime_pose_transform
            )
            obs_dict_np = get_real_umi_obs_dict(
                env_obs=policy_obs,
                shape_meta=cfg.task.shape_meta,
                obs_pose_repr=obs_pose_rep,
                tx_robot1_robot0=tx_robot1_robot0,
                episode_start_pose=policy_episode_start_pose,
            )
            if log_runtime_transforms:
                log_policy_obs_image_stats(obs_dict_np, prefix="[OBS_DEBUG warmup]")

            socket.send_pyobj(obs_dict_np)
            print(
                f"    obs_dict_np sent to PolicyInferenceNode at tcp://{policy_ip}:{policy_port}. Waiting for response."
            )
            start_time = time.monotonic()
            raw_action = socket.recv_pyobj()
            if type(raw_action) == str:
                fatal_exit(
                    env,
                    f"Warmup inference from PolicyInferenceNode failed: {raw_action}. Please check the model.",
                )
            print(
                f"Got response from PolicyInferenceNode. Inference time: {time.monotonic() - start_time:.3f} s"
            )

            if not disable_video_recording:
                env.camera.stop_recording()
                print(
                    f"Warming up video recording finished. Video stored to {env.video_dir.joinpath(str(0))}"
                )

            assert raw_action.shape[-1] == 10 * len(robots_config)
            if runtime_pose_transform.uses_camera_frame_action:
                policy_action = get_camera_frame_umi_action(
                    raw_action,
                    obs,
                    runtime_pose_transform,
                    action_pose_repr,
                )
                action = policy_action
            else:
                policy_action = get_real_umi_action(
                    raw_action, policy_obs, action_pose_repr
                )
                action = convert_policy_action_to_env_frame(
                    policy_action, runtime_pose_transform
                )
            action = apply_action_z_bias(action, action_z_bias)
            assert action.shape[-1] == 7 * len(robots_config)
            if log_runtime_transforms:
                log_runtime_transform_step(
                    raw_arx_obs=obs,
                    policy_obs=policy_obs,
                    raw_policy_action=raw_action,
                    policy_action=policy_action,
                    converted_arx_action=action,
                    final_tcp_pose_cmd=action,
                    runtime_pose_transform=runtime_pose_transform,
                )

            print("Ready!")
            while True:
                # ========= human control loop ==========
                set_gripper_control_phase(env, GripperControlPhase.PRE_POLICY_HOLD)
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])
                print("[INFO] 返回/进入人类控制，SpaceMouse 游标基准已重新与当前物理坐标对齐。")

                gripper_target_pos = np.asarray(
                    [rs["gripper_position"] for rs in robot_states]
                )

                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    fail_if_camera_unhealthy(
                        env,
                        "human control",
                        min_ring_buffer_count=k_needed,
                        max_frame_staleness=max_camera_frame_staleness,
                    )
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    os.makedirs(
                        os.path.join(output, "obs", f"{episode_id}"), exist_ok=True
                    )
                    os.makedirs(
                        os.path.join(output, "action", f"{episode_id}"), exist_ok=True
                    )
                    vis_img, vis_keys = get_policy_vis_image(obs)

                    text = f"Episode: {episode_id} | Policy views: {', '.join(vis_keys)}"
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0, 0, 0),
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("default", vis_img[..., ::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            # Exit program
                            print("\n[ACTION] q键触发：执行安全退出...")
                            end_episode_safely(env)
                            safe_teardown(env, socket, context)
                            exit(0)
                        elif key_stroke == KeyCode(char="c"):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char="e"):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(
                                    match_episode + 1, env.replay_buffer.n_episodes - 1
                                )
                        elif key_stroke == KeyCode(char="w"):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == Key.backspace:
                            if click.confirm("Are you sure to drop an episode?"):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char="a"):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char="1"):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char="2"):
                            control_robot_idx_list = [1]
                        elif key_stroke == KeyCode(char="r"):
                            robot_states = env.get_robot_state()
                            target_pose = np.stack(
                                [rs["ActualTCPPose"] for rs in robot_states]
                            )
                            gripper_target_pos = np.asarray(
                                [rs["gripper_position"] for rs in robot_states]
                            )
                            print("\n" + "="*60)
                            print("🎯 [RECORD] 当前实体位姿已按实测状态刻录。请复制以下代码替换顶部常量：")
                            for robot_idx in range(len(env.robots)):
                                current_pose = target_pose[robot_idx]
                                curr_gripper = gripper_target_pos[robot_idx]
                                current_joint = robot_states[robot_idx]["ActualQ"]
                                pose_str = np.array2string(
                                    current_pose, precision=4, separator=', ', suppress_small=True
                                )
                                joint_str = np.array2string(
                                    current_joint, precision=4, separator=', ', suppress_small=True
                                )
                                print(f"\n# Robot {robot_idx}")
                                print(f"OBS_POSE = np.array({pose_str})")
                                print(f"REST_POSE = np.array({pose_str})")
                                print(f"GRIPPER_POS = {curr_gripper:.4f}")
                                print(f"JOINT_POS = np.array({joint_str})")
                            print("="*60 + "\n")
                        elif key_stroke == KeyCode(char="i"):
                            if not ENABLE_JOINT_HOTKEYS:
                                print(
                                    "\n[SAFE BLOCK] i键 joint initial 已禁用："
                                    "刚才的 SET_JOINT_POS 路径会导致实机下坠。"
                                    "请先用 arx5-sdk 独立 joint 测试验证目标。"
                                )
                                robot_states = env.get_robot_state()
                                target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])
                                gripper_target_pos = np.asarray(
                                    [rs["gripper_position"] for rs in robot_states]
                                )
                                t_start = time.monotonic()
                                iter_idx = 0
                                continue
                            print("\n[ACTION] i键触发：正在移动至 joint initial [0, 1.5, 0, 0, 0, 0]...")
                            robot_states = env.get_robot_state()
                            for robot_idx in control_robot_idx_list:
                                obs_gripper = GRIPPER_INIT
                                if robot_idx < len(robot_states):
                                    try:
                                        obs_gripper = float(robot_states[robot_idx]["gripper_position"])
                                    except Exception:
                                        pass
                                env.robots[robot_idx].set_joint_pos(
                                    INIT_JOINT_POS,
                                    obs_gripper,
                                    duration=JOINT_MOVE_DURATION_S,
                                    max_joint_step_rad=JOINT_MAX_STEP_RAD,
                                    timeout=4.0,
                                )
                            
                            start_wait_t = time.monotonic()
                            wait_duration = JOINT_MOVE_DURATION_S
                            aborted = False
                            # 非阻塞事件泵循环
                            while time.monotonic() - start_wait_t < wait_duration:
                                _ = cv2.pollKey()
                                sub_events = key_counter.get_press_events()
                                for sub_k in sub_events:
                                    if sub_k == KeyCode(char="q"):
                                        print("\n[ACTION] q键触发(在此期间)：执行安全退出...")
                                        end_episode_safely(env)
                                        safe_teardown(env, socket, context)
                                        exit(0)
                                    if sub_k == KeyCode(char="s"):
                                        print("\n[ACTION] s键触发(在此期间)：切换到 damping 并中断 joint 轨迹！")
                                        for robot_idx in control_robot_idx_list:
                                            env.robots[robot_idx].set_to_damping()
                                        aborted = True
                                        break
                                if aborted:
                                    break
                                time.sleep(0.01)
                            
                            print("[INFO] joint initial 指令完毕，强制重新读取系统状态。")
                            robot_states = env.get_robot_state()
                            target_pose = np.stack([rs["ActualTCPPose"] for rs in robot_states])
                            gripper_target_pos = np.asarray(
                                [rs["gripper_position"] for rs in robot_states]
                            )
                            print("[INFO] joint initial 已到达或被中断，SpaceMouse 游标基准已重新对齐实体坐标。")
                            
                            t_start = time.monotonic()
                            iter_idx = 0
                            continue # 直接跳过本轮执行下发，防跳变


                    if start_policy:
                        break
                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (0.5 / frequency) * cartesian_speed
                    drot_xyz = sm_state[3:] * (1.5 / frequency) * orientation_speed

                    if no_spacemouse:
                        for ks in press_events:
                            if ks == Key.up: dpos[0] += 0.005
                            elif ks == Key.down: dpos[0] -= 0.005
                            elif ks == Key.left: dpos[1] += 0.005
                            elif ks == Key.right: dpos[1] -= 0.005
                            elif ks == KeyCode(char="u"): dpos[2] += 0.005
                            elif ks == KeyCode(char="j"): dpos[2] -= 0.005
                            elif ks == KeyCode(char="y"): drot_xyz[0] += 0.05
                            elif ks == KeyCode(char="h"): drot_xyz[0] -= 0.05
                            elif ks == KeyCode(char="o"): drot_xyz[1] += 0.05
                            elif ks == KeyCode(char="l"): drot_xyz[1] -= 0.05
                            elif ks == KeyCode(char="n"): drot_xyz[2] += 0.05
                            elif ks == KeyCode(char="m"): drot_xyz[2] -= 0.05

                    drot = st.Rotation.from_euler("xyz", drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (
                            drot * st.Rotation.from_rotvec(target_pose[robot_idx, 3:])
                        ).as_rotvec()
                        # target_pose[robot_idx, 2] = np.maximum(target_pose[robot_idx, 2], 0.055)

                    dpos = 0
                    if sm.is_button_pressed(0) and sm.is_button_pressed(1):
                        print("Reset robot arm to home. Please wait...")
                        for robot_idx in control_robot_idx_list:
                            env.robots[robot_idx].reset_to_home()
                            robot_states = env.get_robot_state()
                            target_pose[robot_idx] = np.stack(
                                [rs["ActualTCPPose"] for rs in robot_states]
                            )
                            gripper_target_pos[robot_idx] = np.asarray(
                                [rs["gripper_position"] for rs in robot_states]
                            )

                    elif sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    elif sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                        
                    if no_spacemouse:
                        for ks in press_events:
                            if ks == KeyCode(char="v"):
                                dpos -= gripper_speed / frequency * 3
                            elif ks == KeyCode(char="b"):
                                dpos += gripper_speed / frequency * 3

                    # Do not rewrite gripper targets when there is no user input.
                    # The previous code clipped every cycle, which meant simply
                    # entering human control could mutate a valid current width.
                    if dpos != 0:
                        for robot_idx in control_robot_idx_list:
                            gripper_target_pos[robot_idx] = np.clip(
                                gripper_target_pos[robot_idx] + dpos,
                                0,
                                max_gripper_width,
                            )

                    # # solve collision with table
                    # for robot_idx in control_robot_idx_list:
                    #     solve_table_collision(
                    #         ee_pose=target_pose[robot_idx],
                    #         gripper_width=gripper_target_pos[robot_idx],
                    #         height_threshold=robots_config[robot_idx]['height_threshold'])

                    # # solve collison between two robots
                    # solve_sphere_collision(
                    #     ee_poses=target_pose,
                    #     robots_config=robots_config
                    # )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0 : 7 * robot_idx + 6] = target_pose[
                            robot_idx
                        ]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]

                    # execute teleop command
                    env.exec_actions(
                        actions=[action],
                        timestamps=[t_command_target - time.monotonic() + time.time()],
                        compensate_latency=False,
                    )
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    # start episode
                    fail_if_camera_unhealthy(
                        env,
                        "before policy episode start",
                        min_ring_buffer_count=k_needed,
                        max_frame_staleness=max_camera_frame_staleness,
                    )
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    set_gripper_control_phase(env, GripperControlPhase.POLICY_CONTROL)
                    episode_id = env.replay_buffer.n_episodes

                    # get current pose
                    fail_if_camera_unhealthy(
                        env,
                        "before first policy observation",
                        min_ring_buffer_count=k_needed,
                        max_frame_staleness=max_camera_frame_staleness,
                    )
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate(
                            [
                                obs[f"robot{robot_id}_eef_pos"],
                                obs[f"robot{robot_id}_eef_rot_axis_angle"],
                            ],
                            axis=-1,
                        )[-1]
                        episode_start_pose.append(pose)
                    policy_episode_start_pose = (
                        convert_episode_start_pose_to_policy_frame(
                            episode_start_pose, runtime_pose_transform
                        )
                    )

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    last_bias_snapshot = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        fail_if_camera_unhealthy(
                            env,
                            "policy control",
                            min_ring_buffer_count=k_needed,
                            max_frame_staleness=max_camera_frame_staleness,
                        )
                        obs = env.get_obs()
                        obs_timestamps = obs["timestamp"]
                        obs_latency = time.time() - obs_timestamps[-1]
                        print(f"Obs latency {obs_latency}")
                        if obs_latency > max_camera_frame_staleness:
                            fatal_exit(
                                env,
                                f"Observation latency {obs_latency:.3f}s exceeded threshold "
                                f"{max_camera_frame_staleness:.3f}s during policy control.",
                            )

                        # run inference
                        s = time.time()
                        policy_obs = convert_env_obs_to_policy_frame(
                            obs, runtime_pose_transform
                        )
                        obs_dict_np = get_real_umi_obs_dict(
                            env_obs=policy_obs,
                            shape_meta=cfg.task.shape_meta,
                            obs_pose_repr=obs_pose_rep,
                            tx_robot1_robot0=tx_robot1_robot0,
                            episode_start_pose=policy_episode_start_pose,
                        )
                        if log_runtime_transforms:
                            log_policy_obs_image_stats(
                                obs_dict_np, prefix="[OBS_DEBUG policy]"
                            )
                        obs_data = {
                            "obs_dict_np": obs_dict_np,
                            "obs_pose_rep": obs_pose_rep,
                            "obs": obs,
                            "policy_obs": policy_obs,
                            "episode_start_pose": episode_start_pose,
                            "policy_episode_start_pose": policy_episode_start_pose,
                            "tx_robot1_robot0": tx_robot1_robot0,
                            "runtime_pose_transform": runtime_pose_transform.to_debug_dict(),
                        }
                        np.save(
                            os.path.join(
                                output, "obs", f"{episode_id}", f"{iter_idx}.npy"
                            ),
                            obs_data,
                            allow_pickle=True,
                        )

                        socket.send_pyobj(obs_dict_np)
                        raw_action = socket.recv_pyobj()
                        if type(raw_action) == str:
                            print(
                                f"Inference from PolicyInferenceNode failed: {raw_action}. Please check the model."
                            )
                            print("[WARN] 侦测到异常内容字符串，ZMQ Socket 已暴力重建，通信管道已洗净。")
                            socket.close()
                            socket = context.socket(zmq.REQ)
                            socket.connect(f"tcp://{policy_ip}:{policy_port}")
                            end_episode_safely(env)
                            set_gripper_control_phase(
                                env, GripperControlPhase.PRE_POLICY_HOLD
                            )
                            break
                        if runtime_pose_transform.uses_camera_frame_action:
                            policy_action = get_camera_frame_umi_action(
                                raw_action,
                                obs,
                                runtime_pose_transform,
                                action_pose_repr,
                            )
                            action = policy_action
                        else:
                            policy_action = get_real_umi_action(
                                raw_action, policy_obs, action_pose_repr
                            )
                            action = convert_policy_action_to_env_frame(
                                policy_action, runtime_pose_transform
                            )
                        action = apply_action_z_bias(action, action_z_bias)
                        
                        # # --- FIX: Convert network's Axis-Angle back to ARX5 Native Euler (XYZ) ---
                        # for r_idx in range(len(robots_config)):
                        #     rot_vecs = action[:, 7*r_idx+3 : 7*r_idx+6]
                        #     action[:, 7*r_idx+3 : 7*r_idx+6] = st.Rotation.from_rotvec(rot_vecs).as_euler('xyz')
                        # # ----------------------------------------------------------------------

                        action_data = {
                            "action": action,
                            "policy_action": policy_action,
                            "raw_action": raw_action,
                            "action_pose_repr": action_pose_repr,
                            "action_reference_frame": runtime_pose_transform.action_reference_frame,
                            "action_z_bias": action_z_bias,
                            "runtime_pose_transform": runtime_pose_transform.to_debug_dict(),
                        }
                        np.save(
                            os.path.join(
                                output, "action", f"{episode_id}", f"{iter_idx}.npy"
                            ),
                            action_data,
                            allow_pickle=True,
                        )
                        print("Inference latency:", time.time() - s)

                        # convert policy action to env actions
                        this_target_poses = action.copy()
                        # Keep policy gripper targets within the documented Gen gripper
                        # protocol range. Using a narrower ad-hoc host-side limit here
                        # can force unexpected opening/closing targets that disagree with
                        # the live encoder state.
                        for r_idx in range(len(robots_config)):
                            this_target_poses[:, 7 * r_idx + 6] = np.clip(
                                this_target_poses[:, 7 * r_idx + 6],
                                0,
                                max_gripper_width,
                            )
                        if log_runtime_transforms:
                            log_action_chunk_summary(
                                obs,
                                this_target_poses,
                                prefix="[CHUNK_DEBUG full_model_chunk]",
                            )
                        submit_horizon = min(
                            max(1, int(steps_per_inference)),
                            len(this_target_poses),
                        )
                        if submit_horizon < len(this_target_poses):
                            print(
                                f"[ACTION] Truncating action chunk: "
                                f"{len(this_target_poses)} -> {submit_horizon} "
                                f"steps_per_inference={steps_per_inference}"
                            )
                            this_target_poses = this_target_poses[:submit_horizon]
                        if log_runtime_transforms:
                            log_action_chunk_summary(
                                obs,
                                this_target_poses,
                                prefix="[CHUNK_DEBUG submitted_chunk]",
                            )
                        if runtime_pose_transform.enabled:
                            last_bias_snapshot = make_runtime_bias_snapshot(
                                episode_id=episode_id,
                                iter_idx=iter_idx,
                                raw_arx_obs=obs,
                                policy_obs=policy_obs,
                                raw_policy_action=raw_action,
                                policy_action=policy_action,
                                converted_arx_action=action,
                                final_tcp_pose_cmd=this_target_poses,
                                runtime_pose_transform=runtime_pose_transform,
                            )
                        if log_runtime_transforms:
                            log_runtime_transform_step(
                                raw_arx_obs=obs,
                                policy_obs=policy_obs,
                                raw_policy_action=raw_action,
                                policy_action=policy_action,
                                converted_arx_action=action,
                                final_tcp_pose_cmd=this_target_poses,
                                runtime_pose_transform=runtime_pose_transform,
                            )
                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        # for target_pose in this_target_poses:
                        #     for robot_idx in range(len(robots_config)):
                        #         solve_table_collision(
                        #             ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                        #             gripper_width=target_pose[robot_idx * 7 + 6],
                        #             height_threshold=robots_config[robot_idx]['height_threshold']
                        #         )

                        #     # solve collison between two robots
                        #     solve_sphere_collision(
                        #         ee_poses=target_pose.reshape([len(robots_config), -1]),
                        #         robots_config=robots_config
                        #     )

                        # deal with timing
                        # the same step actions are always the target for
                        action_start_time = time.time() + command_latency
                        action_timestamps = (
                            np.arange(len(this_target_poses), dtype=np.float64)
                        ) * dt + action_start_time
                        if log_runtime_transforms:
                            print(
                                "[TIMING_DEBUG] action_start_time_offset_from_now:",
                                round(action_start_time - time.time(), 4),
                                "submitted_steps:",
                                len(this_target_poses),
                            )
                        # action_exec_latency = 0.01
                        # curr_time = time.time()
                        # is_new = action_timestamps > (curr_time + action_exec_latency)
                        # if np.sum(is_new) == 0:
                        #     # exceeded time budget, still do something
                        #     this_target_poses = this_target_poses[[-1]]
                        #     # schedule on next available step
                        #     next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        #     action_timestamp = eval_t_start + (next_step_idx) * dt
                        #     print('Over budget', action_timestamp - curr_time)
                        #     action_timestamps = np.array([action_timestamp])
                        # else:
                        #     this_target_poses = this_target_poses[is_new]
                        #     action_timestamps = action_timestamps[is_new]

                        # execute actions
                        if dry_run_policy:
                            print(
                                "[DRY_RUN] 跳过 env.exec_actions；本轮只用于检查坐标系/动作转换。"
                            )
                            stop_episode = True
                        else:
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                # compensate_latency=True
                                dynamic_latency=not disable_dynamic_latency,
                            )
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img, vis_keys = get_policy_vis_image(obs)
                        text = "Episode: {}, Time: {:.1f}, Policy views: {}".format(
                            episode_id,
                            time.monotonic() - t_start,
                            ", ".join(vis_keys),
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, 20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255, 255, 255),
                        )
                        cv2.imshow("default", vis_img[..., ::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="s"):
                                # Stop episode
                                # Hand control back to human
                                print("[ACTION] s键触发：停止 Policy 推理阶段！")
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("[ACTION] Max Duration reached. 停止 Policy 推理阶段...")
                            stop_episode = True
                        if stop_episode:
                            if record_bias_on_stop:
                                save_runtime_bias_snapshot(
                                    output=output,
                                    snapshot=last_bias_snapshot,
                                    prompt_bias_on_stop=prompt_bias_on_stop,
                                    bias_input_frame=bias_input_frame,
                                    bias_input_mode=bias_input_mode,
                                )
                            end_episode_safely(env)
                            set_gripper_control_phase(
                                env, GripperControlPhase.PRE_POLICY_HOLD
                            )
                            print("[WARN] 侦测到 s 键退出，ZMQ Socket 已暴力重建，通信管道已洗净。")
                            socket.close()
                            socket = context.socket(zmq.REQ)
                            socket.connect(f"tcp://{policy_ip}:{policy_port}")
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("\n[ACTION] 侦测到 Ctrl+C (Interrupted!)：执行安全退出...")
                    # stop robot.
                    end_episode_safely(env)
                    safe_teardown(env, socket, context)
                    exit(0)
                except RuntimeError as e:
                    fatal_exit(env, f"Policy control loop crashed: {e}")

                print("Stopped.")


# %%
if __name__ == "__main__":
    main()
