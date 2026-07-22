import argparse
import glob
import os
import pathlib
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect saved eval obs/action npy files and export policy-visible images."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to data/experiments/<timestamp> directory.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific iter step to inspect. Default: inspect all saved steps.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory for exported PNGs. Default: <experiment>/debug_inspect",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Print aggregate stats across all saved obs/action files.",
    )
    return parser.parse_args()


def describe_array(name: str, array: np.ndarray):
    print(
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"min={array.min():.6f} max={array.max():.6f}"
    )


def channel_layout(array: np.ndarray) -> str:
    if array.ndim < 3:
        return "not-image"
    if array.shape[-1] in (1, 3):
        return "HWC/THWC"
    if array.shape[-3] in (1, 3):
        return "CHW/TCHW"
    return "unknown"


def ensure_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, got shape={frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    return frame


def export_frames(name: str, array: np.ndarray, save_dir: pathlib.Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    if array.ndim != 4:
        print(f"skip image export for {name}: expected 4D, got {array.shape}")
        return
    first = ensure_uint8_hwc(array[0])
    last = ensure_uint8_hwc(array[-1])
    cv2.imwrite(str(save_dir / f"{name}_first.png"), cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / f"{name}_last.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))


def sorted_npy_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def print_obs_stats(step: int, obs_data: Dict, save_dir: pathlib.Path):
    raw_obs = obs_data["obs"]
    obs_dict_np = obs_data["obs_dict_np"]
    print(f"\n=== step={step} obs ===")
    for key in [
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "timestamp",
    ]:
        if key in raw_obs:
            describe_array(f"raw_obs[{key}]", raw_obs[key])
    if "camera0_rgb" in raw_obs:
        print(f"raw_obs[camera0_rgb] layout={channel_layout(raw_obs['camera0_rgb'])}")
        export_frames(f"step{step:04d}_raw_camera0_rgb", raw_obs["camera0_rgb"], save_dir)
    for key in [
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_gripper_width",
    ]:
        if key in obs_dict_np:
            describe_array(f"obs_dict_np[{key}]", obs_dict_np[key])
    if "camera0_rgb" in obs_dict_np:
        print(f"obs_dict_np[camera0_rgb] layout={channel_layout(obs_dict_np['camera0_rgb'])}")
        export_frames(
            f"step{step:04d}_policy_camera0_rgb",
            obs_dict_np["camera0_rgb"],
            save_dir,
        )


def print_action_stats(step: int, action_data: Dict, obs_data: Dict):
    action = action_data["action"]
    raw_action = action_data["raw_action"]
    curr_pose = np.concatenate(
        [
            obs_data["obs"]["robot0_eef_pos"][-1],
            obs_data["obs"]["robot0_eef_rot_axis_angle"][-1],
            obs_data["obs"]["robot0_gripper_width"][-1],
        ]
    )
    pose_delta = action - curr_pose[None, :]
    print(f"\n=== step={step} action ===")
    describe_array("raw_action", raw_action)
    describe_array("decoded_action", action)
    print(f"current_pose={np.array2string(curr_pose, precision=6, suppress_small=True)}")
    print(f"decoded_action[0]={np.array2string(action[0], precision=6, suppress_small=True)}")
    print(
        "delta(decoded_action[0]-current)="
        f"{np.array2string(pose_delta[0], precision=6, suppress_small=True)}"
    )
    print(
        "decoded_xyz_delta_min="
        f"{np.array2string(pose_delta[:, :3].min(axis=0), precision=6, suppress_small=True)}"
    )
    print(
        "decoded_xyz_delta_max="
        f"{np.array2string(pose_delta[:, :3].max(axis=0), precision=6, suppress_small=True)}"
    )
    print(
        "decoded_gripper_minmax="
        f"({action[:, 6].min():.6f}, {action[:, 6].max():.6f})"
    )


def aggregate_stats(obs_files: Iterable[str], action_files: Iterable[str]):
    raw_obs_acc = {
        "robot0_eef_pos": [],
        "robot0_eef_rot_axis_angle": [],
        "robot0_gripper_width": [],
    }
    proc_obs_acc = {
        "robot0_eef_pos": [],
        "robot0_eef_rot_axis_angle": [],
        "robot0_eef_rot_axis_angle_wrt_start": [],
        "robot0_gripper_width": [],
    }
    raw_action_acc = []
    action_acc = []
    for path in obs_files:
        data = np.load(path, allow_pickle=True).item()
        for key in raw_obs_acc:
            raw_obs_acc[key].append(data["obs"][key])
        for key in proc_obs_acc:
            if key in data["obs_dict_np"]:
                proc_obs_acc[key].append(data["obs_dict_np"][key])
    for path in action_files:
        data = np.load(path, allow_pickle=True).item()
        raw_action_acc.append(data["raw_action"])
        action_acc.append(data["action"])

    print("\n=== aggregate raw obs ===")
    for key, values in raw_obs_acc.items():
        array = np.concatenate(values, axis=0)
        print(
            f"{key}: min={np.array2string(array.min(axis=0), precision=6, suppress_small=True)} "
            f"max={np.array2string(array.max(axis=0), precision=6, suppress_small=True)} "
            f"mean={np.array2string(array.mean(axis=0), precision=6, suppress_small=True)} "
            f"std={np.array2string(array.std(axis=0), precision=6, suppress_small=True)}"
        )

    print("\n=== aggregate policy obs ===")
    for key, values in proc_obs_acc.items():
        array = np.concatenate(values, axis=0)
        print(
            f"{key}: min={np.array2string(array.min(axis=0), precision=6, suppress_small=True)} "
            f"max={np.array2string(array.max(axis=0), precision=6, suppress_small=True)} "
            f"mean={np.array2string(array.mean(axis=0), precision=6, suppress_small=True)} "
            f"std={np.array2string(array.std(axis=0), precision=6, suppress_small=True)}"
        )

    print("\n=== aggregate action ===")
    for name, values in [("raw_action", raw_action_acc), ("decoded_action", action_acc)]:
        array = np.concatenate(values, axis=0)
        print(
            f"{name}: min={np.array2string(array.min(axis=0), precision=6, suppress_small=True)} "
            f"max={np.array2string(array.max(axis=0), precision=6, suppress_small=True)} "
            f"mean={np.array2string(array.mean(axis=0), precision=6, suppress_small=True)} "
            f"std={np.array2string(array.std(axis=0), precision=6, suppress_small=True)}"
        )


def main():
    args = parse_args()
    experiment = pathlib.Path(args.experiment).resolve()
    save_dir = (
        pathlib.Path(args.save_dir).resolve()
        if args.save_dir is not None
        else experiment / "debug_inspect"
    )
    obs_dir = experiment / "obs" / str(args.episode)
    action_dir = experiment / "action" / str(args.episode)
    obs_files = sorted_npy_files(str(obs_dir / "*.npy"))
    action_files = sorted_npy_files(str(action_dir / "*.npy"))
    if not obs_files:
        raise FileNotFoundError(f"No obs files found under {obs_dir}")
    if not action_files:
        raise FileNotFoundError(f"No action files found under {action_dir}")

    action_map = {
        int(pathlib.Path(path).stem): path
        for path in action_files
    }
    selected_steps = [args.step] if args.step is not None else [int(pathlib.Path(p).stem) for p in obs_files]

    for step in selected_steps:
        obs_path = obs_dir / f"{step}.npy"
        action_path = pathlib.Path(action_map[step])
        obs_data = np.load(obs_path, allow_pickle=True).item()
        action_data = np.load(action_path, allow_pickle=True).item()
        print_obs_stats(step, obs_data, save_dir)
        print_action_stats(step, action_data, obs_data)

    if args.aggregate:
        aggregate_stats(obs_files, action_files)

    print(f"\nExported PNGs to {save_dir}")


if __name__ == "__main__":
    main()
