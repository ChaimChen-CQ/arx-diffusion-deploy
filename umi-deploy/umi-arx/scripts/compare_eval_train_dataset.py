import argparse
import glob
import os
import pathlib
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved eval policy camera0_rgb against a training replay buffer."
    )
    parser.add_argument(
        "--experiment",
        default="data/experiments/20260429_131650",
        help="Path to data/experiments/<timestamp> directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to training dataset .npz, .zarr directory, or .zarr.zip replay buffer.",
    )
    parser.add_argument(
        "--allow-missing-dataset",
        action="store_true",
        help="Export eval policy image and report dataset absence instead of failing.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--eval-step",
        type=int,
        default=None,
        help="Eval iter step to visualize. Default: first saved step.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Dataset frame index to visualize. Default: random index.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory for exported comparison images. Default: <experiment>/debug_compare",
    )
    return parser.parse_args()


def sorted_npy_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def load_npz_replay(dataset_path: pathlib.Path) -> Dict[str, np.ndarray]:
    data = {}
    with np.load(dataset_path, allow_pickle=False) as archive:
        for key in archive.files:
            if key.startswith("data__"):
                data[key[len("data__") :]] = archive[key]
            elif key.startswith("meta__"):
                data[key[len("meta__") :]] = archive[key]
    return data


def load_zarr_replay(dataset_path: pathlib.Path) -> Dict[str, np.ndarray]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "zarr is required for .zarr/.zarr.zip training datasets; "
            "run this script in the training/deploy conda environment."
        ) from exc

    if dataset_path.is_dir():
        from zarr.storage import DirectoryStore

        store = DirectoryStore(str(dataset_path))
    else:
        store = zarr.ZipStore(str(dataset_path), mode="r")
    with store:
        root = zarr.group(store=store)
        return {key: np.asarray(root["data"][key]) for key in root["data"].array_keys()}


def load_replay_dataset(dataset_path: pathlib.Path) -> Dict[str, np.ndarray]:
    if dataset_path.suffix == ".npz":
        return load_npz_replay(dataset_path)
    if dataset_path.is_dir() or str(dataset_path).endswith(".zarr.zip"):
        return load_zarr_replay(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
    return frame


def save_rgb_image(path: pathlib.Path, frame: np.ndarray):
    Image.fromarray(to_hwc_uint8(frame), mode="RGB").save(path)


def save_labeled_side_by_side(path: pathlib.Path, left: np.ndarray, right: np.ndarray, left_label: str, right_label: str):
    left_img = Image.fromarray(to_hwc_uint8(left), mode="RGB")
    right_img = Image.fromarray(to_hwc_uint8(right), mode="RGB")
    if left_img.size != right_img.size:
        right_img = right_img.resize(left_img.size, Image.Resampling.BILINEAR)

    label_h = 26
    out = Image.new("RGB", (left_img.width + right_img.width, left_img.height + label_h), (255, 255, 255))
    out.paste(left_img, (0, label_h))
    out.paste(right_img, (left_img.width, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), left_label, fill=(0, 0, 0))
    draw.text((left_img.width + 6, 6), right_label, fill=(0, 0, 0))
    out.save(path)


def image_channel_stats(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3):
            arr = np.moveaxis(arr, 1, -1)
        flat = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        flat = arr.reshape(-1, arr.shape[-1])
    else:
        flat = arr.reshape(-1, 1)
    return flat


def stat_line(name: str, array: np.ndarray):
    flat = image_channel_stats(array)
    print(
        f"{name}: shape={array.shape} "
        f"min={np.array2string(flat.min(axis=0), precision=6, suppress_small=True)} "
        f"max={np.array2string(flat.max(axis=0), precision=6, suppress_small=True)} "
        f"mean={np.array2string(flat.mean(axis=0), precision=6, suppress_small=True)} "
        f"std={np.array2string(flat.std(axis=0), precision=6, suppress_small=True)}"
    )


def collect_eval_stats(experiment: pathlib.Path, episode: int) -> Dict[str, np.ndarray]:
    obs_files = sorted_npy_files(str(experiment / "obs" / str(episode) / "*.npy"))
    action_files = sorted_npy_files(str(experiment / "action" / str(episode) / "*.npy"))
    obs_acc = {
        "robot0_eef_pos": [],
        "robot0_eef_rot_axis_angle": [],
        "robot0_gripper_width": [],
    }
    obs_policy_acc = {
        "robot0_eef_pos": [],
        "robot0_eef_rot_axis_angle": [],
        "robot0_eef_rot_axis_angle_wrt_start": [],
        "robot0_gripper_width": [],
    }
    action_acc = []
    raw_action_acc = []
    for path in obs_files:
        data = np.load(path, allow_pickle=True).item()
        for key in obs_acc:
            obs_acc[key].append(data["obs"][key])
        for key in obs_policy_acc:
            if key in data["obs_dict_np"]:
                obs_policy_acc[key].append(data["obs_dict_np"][key])
    for path in action_files:
        data = np.load(path, allow_pickle=True).item()
        action_acc.append(data["action"])
        raw_action_acc.append(data["raw_action"])
    result = {}
    for key, values in obs_acc.items():
        result[f"eval_raw::{key}"] = np.concatenate(values, axis=0)
    for key, values in obs_policy_acc.items():
        result[f"eval_policy::{key}"] = np.concatenate(values, axis=0)
    result["eval_decoded::action"] = np.concatenate(action_acc, axis=0)
    result["eval_raw::action"] = np.concatenate(raw_action_acc, axis=0)
    return result


def main():
    args = parse_args()
    experiment = pathlib.Path(args.experiment).resolve()
    save_dir = (
        pathlib.Path(args.save_dir).resolve()
        if args.save_dir is not None
        else experiment / "debug_compare"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    obs_files = sorted_npy_files(str(experiment / "obs" / str(args.episode) / "*.npy"))
    if not obs_files:
        raise FileNotFoundError(f"No saved obs files under {experiment / 'obs' / str(args.episode)}")
    eval_step = args.eval_step if args.eval_step is not None else int(pathlib.Path(obs_files[0]).stem)
    eval_obs = np.load(experiment / "obs" / str(args.episode) / f"{eval_step}.npy", allow_pickle=True).item()
    eval_img = to_hwc_uint8(eval_obs["obs_dict_np"]["camera0_rgb"][-1])
    eval_img_path = save_dir / f"eval_step{eval_step:04d}_policy_camera0_rgb.png"
    save_rgb_image(eval_img_path, eval_img)

    if args.dataset is None:
        print(f"eval_step={eval_step}")
        print("\n=== image stats ===")
        stat_line("eval policy camera0_rgb", eval_obs["obs_dict_np"]["camera0_rgb"])
        print(f"\nSaved eval policy image to {eval_img_path}")
        print("No --dataset was provided, so no training side-by-side was generated.")
        return

    dataset_path = pathlib.Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        msg = f"Training dataset not found: {dataset_path}"
        if not args.allow_missing_dataset:
            raise FileNotFoundError(msg)
        print(msg)
        print(f"Saved eval policy image to {eval_img_path}")
        return

    dataset = load_replay_dataset(dataset_path)
    if "camera0_rgb" not in dataset:
        raise KeyError(f"{dataset_path} does not contain data__camera0_rgb")
    rng = np.random.default_rng(0)
    dataset_index = (
        int(args.dataset_index)
        if args.dataset_index is not None
        else int(rng.integers(0, len(dataset["camera0_rgb"])))
    )

    train_img = to_hwc_uint8(dataset["camera0_rgb"][dataset_index])
    compare_path = save_dir / f"eval_step{eval_step:04d}_vs_train_idx{dataset_index:06d}.png"
    save_labeled_side_by_side(
        compare_path,
        eval_img,
        train_img,
        f"eval policy step {eval_step}",
        f"train idx {dataset_index}",
    )

    print(f"eval_step={eval_step} dataset_index={dataset_index}")
    print("\n=== image stats ===")
    stat_line("eval policy camera0_rgb", eval_obs["obs_dict_np"]["camera0_rgb"])
    stat_line("train camera0_rgb", dataset["camera0_rgb"][dataset_index : dataset_index + 1])

    print("\n=== training dataset stats ===")
    for key in [
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "action",
    ]:
        if key in dataset:
            stat_line(f"train::{key}", dataset[key])

    print("\n=== eval stats ===")
    eval_stats = collect_eval_stats(experiment, args.episode)
    for key in [
        "eval_raw::robot0_eef_pos",
        "eval_raw::robot0_eef_rot_axis_angle",
        "eval_raw::robot0_gripper_width",
        "eval_policy::robot0_eef_pos",
        "eval_policy::robot0_eef_rot_axis_angle",
        "eval_policy::robot0_eef_rot_axis_angle_wrt_start",
        "eval_policy::robot0_gripper_width",
        "eval_raw::action",
        "eval_decoded::action",
    ]:
        if key in eval_stats:
            stat_line(key, eval_stats[key])

    print(f"\nSaved eval policy image to {eval_img_path}")
    print(f"Saved side-by-side image to {compare_path}")


if __name__ == "__main__":
    main()
