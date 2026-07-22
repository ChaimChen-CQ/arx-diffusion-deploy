from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List, Optional
import numpy as np
import collections
from utils.other_util import get_image_transform
from utils.pose_repr_util import convert_pose_mat_rep
from utils.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat


@dataclass(frozen=True)
class RuntimePoseTransform:
    tx_policy_frame_from_env_base: np.ndarray
    tx_env_base_from_policy_frame: np.ndarray
    tx_policy_tcp_from_env_tcp: np.ndarray
    tx_env_tcp_from_policy_tcp: np.ndarray
    tx_env_tcp_camera: np.ndarray
    tx_camera_policy_tcp: np.ndarray
    action_reference_frame: str = "policy"
    source_path: Optional[str] = None
    reference_tx_gripper2camera: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        identity = np.eye(4, dtype=np.float64)
        return not (
            np.allclose(
                self.tx_policy_frame_from_env_base, identity, atol=1e-9, rtol=1e-9
            )
            and np.allclose(
                self.tx_policy_tcp_from_env_tcp, identity, atol=1e-9, rtol=1e-9
            )
            and self.action_reference_frame == "policy"
        )

    @property
    def uses_camera_frame_action(self) -> bool:
        return self.action_reference_frame == "camera"

    def to_debug_dict(self) -> Dict[str, np.ndarray]:
        out = {
            "enabled": self.enabled,
            "source_path": self.source_path,
            "tx_policy_frame_from_env_base": self.tx_policy_frame_from_env_base,
            "tx_env_base_from_policy_frame": self.tx_env_base_from_policy_frame,
            "tx_policy_tcp_from_env_tcp": self.tx_policy_tcp_from_env_tcp,
            "tx_env_tcp_from_policy_tcp": self.tx_env_tcp_from_policy_tcp,
            "tx_env_tcp_camera": self.tx_env_tcp_camera,
            "tx_camera_policy_tcp": self.tx_camera_policy_tcp,
            "action_reference_frame": self.action_reference_frame,
        }
        if self.reference_tx_gripper2camera is not None:
            out["reference_tx_gripper2camera"] = self.reference_tx_gripper2camera
        return out


def _as_transform_matrix(
    tx: Optional[np.ndarray], name: str, default_identity: bool = True
) -> np.ndarray:
    if tx is None:
        if default_identity:
            return np.eye(4, dtype=np.float64)
        raise ValueError(f"{name} is required")
    arr = np.asarray(tx, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {arr.shape}")
    return arr


def make_runtime_pose_transform(
    tx_policy_frame_from_env_base: Optional[np.ndarray] = None,
    tx_env_base_from_policy_frame: Optional[np.ndarray] = None,
    tx_policy_tcp_from_env_tcp: Optional[np.ndarray] = None,
    tx_env_tcp_from_policy_tcp: Optional[np.ndarray] = None,
    tx_env_tcp_camera: Optional[np.ndarray] = None,
    tx_camera_policy_tcp: Optional[np.ndarray] = None,
    action_reference_frame: str = "policy",
    source_path: Optional[str] = None,
    reference_tx_gripper2camera: Optional[np.ndarray] = None,
) -> RuntimePoseTransform:
    if action_reference_frame not in ("policy", "camera"):
        raise ValueError(
            "action_reference_frame must be either 'policy' or 'camera', "
            f"got {action_reference_frame!r}"
        )
    tx_policy_frame_from_env_base = _as_transform_matrix(
        tx_policy_frame_from_env_base, "tx_policy_frame_from_env_base"
    )
    if tx_env_base_from_policy_frame is None:
        tx_env_base_from_policy_frame = np.linalg.inv(tx_policy_frame_from_env_base)
    else:
        tx_env_base_from_policy_frame = _as_transform_matrix(
            tx_env_base_from_policy_frame, "tx_env_base_from_policy_frame"
        )
    if not np.allclose(
        tx_env_base_from_policy_frame,
        np.linalg.inv(tx_policy_frame_from_env_base),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "tx_policy_frame_from_env_base and tx_env_base_from_policy_frame are inconsistent"
        )

    tx_env_tcp_camera = _as_transform_matrix(
        tx_env_tcp_camera, "tx_env_tcp_camera"
    )
    tx_camera_policy_tcp = _as_transform_matrix(
        tx_camera_policy_tcp, "tx_camera_policy_tcp"
    )
    tx_env_tcp_policy_tcp_from_camera = tx_env_tcp_camera @ tx_camera_policy_tcp
    if tx_policy_tcp_from_env_tcp is None:
        tx_policy_tcp_from_env_tcp = tx_env_tcp_policy_tcp_from_camera
    else:
        tx_policy_tcp_from_env_tcp = _as_transform_matrix(
            tx_policy_tcp_from_env_tcp, "tx_policy_tcp_from_env_tcp"
        )
        if not np.allclose(tx_env_tcp_camera, np.eye(4), atol=1e-9, rtol=1e-9) and (
            not np.allclose(
                tx_policy_tcp_from_env_tcp,
                tx_env_tcp_policy_tcp_from_camera,
                atol=1e-6,
                rtol=1e-6,
            )
        ):
            raise ValueError(
                "tx_policy_tcp_from_env_tcp must equal "
                "tx_env_tcp_camera @ tx_camera_policy_tcp when both camera matrices are provided"
            )
    if tx_env_tcp_from_policy_tcp is None:
        tx_env_tcp_from_policy_tcp = np.linalg.inv(tx_policy_tcp_from_env_tcp)
    else:
        tx_env_tcp_from_policy_tcp = _as_transform_matrix(
            tx_env_tcp_from_policy_tcp, "tx_env_tcp_from_policy_tcp"
        )
    if not np.allclose(
        tx_env_tcp_from_policy_tcp,
        np.linalg.inv(tx_policy_tcp_from_env_tcp),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError(
            "tx_policy_tcp_from_env_tcp and tx_env_tcp_from_policy_tcp are inconsistent"
        )

    if reference_tx_gripper2camera is not None:
        reference_tx_gripper2camera = _as_transform_matrix(
            reference_tx_gripper2camera, "reference_tx_gripper2camera"
        )

    return RuntimePoseTransform(
        tx_policy_frame_from_env_base=tx_policy_frame_from_env_base,
        tx_env_base_from_policy_frame=tx_env_base_from_policy_frame,
        tx_policy_tcp_from_env_tcp=tx_policy_tcp_from_env_tcp,
        tx_env_tcp_from_policy_tcp=tx_env_tcp_from_policy_tcp,
        tx_env_tcp_camera=tx_env_tcp_camera,
        tx_camera_policy_tcp=tx_camera_policy_tcp,
        action_reference_frame=action_reference_frame,
        source_path=source_path,
        reference_tx_gripper2camera=reference_tx_gripper2camera,
    )


def _apply_runtime_pose_transform(
    pose: np.ndarray,
    pose_transform: Optional[RuntimePoseTransform],
    inverse: bool = False,
) -> np.ndarray:
    pose_arr = np.asarray(pose)
    if pose_transform is None or not pose_transform.enabled:
        return np.array(pose_arr, copy=True)

    pose_mat = pose_to_mat(pose_arr)
    if inverse:
        transformed_pose_mat = (
            pose_transform.tx_env_base_from_policy_frame
            @ pose_mat
            @ pose_transform.tx_env_tcp_from_policy_tcp
        )
    else:
        transformed_pose_mat = (
            pose_transform.tx_policy_frame_from_env_base
            @ pose_mat
            @ pose_transform.tx_policy_tcp_from_env_tcp
        )
    transformed_pose = mat_to_pose(transformed_pose_mat)
    return transformed_pose.astype(pose_arr.dtype, copy=False)


def _get_robot_prefixes(env_obs: Dict[str, np.ndarray]) -> List[str]:
    robot_prefixes = set()
    for key in env_obs.keys():
        if key.endswith("_eef_pos"):
            robot_prefix = key[: -len("_eef_pos")]
            rot_key = robot_prefix + "_eef_rot_axis_angle"
            if rot_key in env_obs:
                robot_prefixes.add(robot_prefix)
    return sorted(robot_prefixes)


def convert_env_obs_to_policy_frame(
    env_obs: Dict[str, np.ndarray],
    pose_transform: Optional[RuntimePoseTransform],
) -> Dict[str, np.ndarray]:
    if pose_transform is None or not pose_transform.enabled:
        return env_obs

    transformed_obs = dict(env_obs)
    for robot_prefix in _get_robot_prefixes(env_obs):
        pose = np.concatenate(
            [
                env_obs[robot_prefix + "_eef_pos"],
                env_obs[robot_prefix + "_eef_rot_axis_angle"],
            ],
            axis=-1,
        )
        transformed_pose = _apply_runtime_pose_transform(
            pose=pose, pose_transform=pose_transform, inverse=False
        )
        transformed_obs[robot_prefix + "_eef_pos"] = transformed_pose[..., :3]
        transformed_obs[robot_prefix + "_eef_rot_axis_angle"] = transformed_pose[..., 3:]
    return transformed_obs


def convert_episode_start_pose_to_policy_frame(
    episode_start_pose: Optional[List[np.ndarray]],
    pose_transform: Optional[RuntimePoseTransform],
) -> Optional[List[np.ndarray]]:
    if episode_start_pose is None or pose_transform is None or not pose_transform.enabled:
        return episode_start_pose

    return [
        _apply_runtime_pose_transform(
            pose=np.asarray(pose), pose_transform=pose_transform, inverse=False
        )
        for pose in episode_start_pose
    ]


def convert_policy_action_to_env_frame(
    action: np.ndarray,
    pose_transform: Optional[RuntimePoseTransform],
) -> np.ndarray:
    action_arr = np.asarray(action)
    if pose_transform is None or not pose_transform.enabled:
        return np.array(action_arr, copy=True)
    if action_arr.shape[-1] % 7 != 0:
        raise ValueError(
            "Expected action last dimension to be a multiple of 7 "
            f"(pose6 + gripper1), got {action_arr.shape[-1]}"
        )

    env_action = np.array(action_arr, copy=True)
    n_robots = int(action_arr.shape[-1] // 7)
    for robot_idx in range(n_robots):
        start = robot_idx * 7
        env_action[..., start : start + 6] = _apply_runtime_pose_transform(
            pose=action_arr[..., start : start + 6],
            pose_transform=pose_transform,
            inverse=True,
        )
    return env_action


def get_camera_frame_umi_action(
    action: np.ndarray,
    env_obs: Dict[str, np.ndarray],
    pose_transform: RuntimePoseTransform,
    action_pose_repr: str = "abs",
) -> np.ndarray:
    if not pose_transform.uses_camera_frame_action:
        raise ValueError("get_camera_frame_umi_action requires action_reference_frame='camera'")

    action_arr = np.asarray(action)
    n_robots = int(action_arr.shape[-1] // 10)
    env_action = list()
    tx_env_tcp_policy_tcp = (
        pose_transform.tx_env_tcp_camera @ pose_transform.tx_camera_policy_tcp
    )
    tx_policy_tcp_env_tcp = np.linalg.inv(tx_env_tcp_policy_tcp)

    for robot_idx in range(n_robots):
        tx_env_base_env_tcp = pose_to_mat(
            np.concatenate(
                [
                    env_obs[f"robot{robot_idx}_eef_pos"][-1],
                    env_obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1],
                ],
                axis=-1,
            )
        )
        tx_env_base_camera = tx_env_base_env_tcp @ pose_transform.tx_env_tcp_camera

        start = robot_idx * 10
        action_pose10d = action_arr[..., start : start + 9]
        action_grip = action_arr[..., start + 9 : start + 10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        tx_camera_policy_tcp_target = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=pose_transform.tx_camera_policy_tcp,
            pose_rep=action_pose_repr,
            backward=True,
        )
        tx_env_base_env_tcp_target = (
            tx_env_base_camera @ tx_camera_policy_tcp_target @ tx_policy_tcp_env_tcp
        )

        action_pose = mat_to_pose(tx_env_base_env_tcp_target)
        env_action.append(action_pose)
        env_action.append(action_grip)

    return np.concatenate(env_action, axis=-1)


def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            co, ho, wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
    env_obs: Dict[str, np.ndarray],
    shape_meta: dict,
) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                )
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
        elif type == "low_dim":
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_umi_obs_dict(
    env_obs: Dict[str, np.ndarray],
    shape_meta: dict,
    obs_pose_repr: str = "abs",
    tx_robot1_robot0: np.ndarray = None,
    episode_start_pose: List[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta["obs"]
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = env_obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                )
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
        elif type == "low_dim" and ("eef" not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split("_")
            if ks[0].startswith("robot"):
                robot_prefix_map[ks[0]].append(key)

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    env_obs[robot_prefix + "_eef_pos"],
                    env_obs[robot_prefix + "_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=pose_mat[-1], pose_rep=obs_pose_repr, backward=False
        )

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + "_eef_pos"] = obs_pose[..., :3]
        obs_dict_np[robot_prefix + "_eef_rot_axis_angle"] = obs_pose[..., 3:]

    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f"robot{robot_id}" in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(
            np.concatenate(
                [
                    env_obs[f"robot{robot_id}_eef_pos"],
                    env_obs[f"robot{robot_id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(
                np.concatenate(
                    [
                        env_obs[f"robot{other_robot_id}_eef_pos"],
                        env_obs[f"robot{other_robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f"robot{robot_id}_eef_pos_wrt{other_robot_id}"] = rel_obs_pose[
                :, :3
            ]
            obs_dict_np[f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"] = (
                rel_obs_pose[:, 3:]
            )

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):
            # convert pose to mat
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        env_obs[f"robot{robot_id}_eef_pos"],
                        env_obs[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep="relative",
                backward=False,
            )

            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rel_obs_pose[
                :, 3:
            ]

    return obs_dict_np


def get_real_umi_action(
    action: np.ndarray, env_obs: Dict[str, np.ndarray], action_pose_repr: str = "abs"
):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    env_obs[f"robot{robot_idx}_eef_pos"][-1],
                    env_obs[f"robot{robot_idx}_eef_rot_axis_angle"][-1],
                ],
                axis=-1,
            )
        )

        start = robot_idx * 10
        action_pose10d = action[..., start : start + 9]
        action_grip = action[..., start + 9 : start + 10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True,
        )

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
