#!/usr/bin/env python3
"""Convert one or more MCAP files into a UMI-compatible Zarr dataset."""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import zarr
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from zarr.storage import ZipStore

try:
    from zarr.storage import LocalStore
except ImportError:
    from zarr.storage import DirectoryStore as LocalStore


ROOT_DIR = Path(__file__).resolve().parent.parent  # umi-diffusion-training/
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


DEFAULT_POSE_TOPIC = "/robot0/vio/relative_eef_pose"
DEFAULT_GRIPPER_TOPIC = "/robot0/sensor/magnetic_encoder"
DEFAULT_CAMERA_TOPIC = "/robot0/sensor/camera0/compressed"


def parse_image_size(text):
    width, height = (int(x) for x in text.split(","))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {text}")
    return width, height


def expand_inputs(inputs):
    files = []
    for raw in inputs:
        path = Path(os.path.expanduser(raw)).resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.mcap")))
        else:
            files.append(path)

    unique_files = []
    seen = set()
    for path in files:
        if path.suffix != ".mcap":
            continue
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    if not unique_files:
        raise ValueError("No .mcap files found.")

    missing = [str(path) for path in unique_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    return unique_files


def deduplicate_last_by_timestamp(data):
    """Keep the last sample for each duplicated timestamp."""
    if len(data) == 0:
        return data
    order = np.argsort(data[:, 0], kind="stable")
    sorted_data = data[order]
    keep_mask = np.ones(len(sorted_data), dtype=bool)
    keep_mask[:-1] = sorted_data[:-1, 0] != sorted_data[1:, 0]
    return sorted_data[keep_mask]


def deduplicate_images_by_timestamp(img_ts, images):
    if len(img_ts) == 0:
        return img_ts, images
    order = np.argsort(img_ts, kind="stable")
    img_ts = img_ts[order]
    images = images[order]
    keep_mask = np.ones(len(img_ts), dtype=bool)
    keep_mask[:-1] = img_ts[:-1] != img_ts[1:]
    return img_ts[keep_mask], images[keep_mask]


def decode_h264_packets(camera_packets):
    if not camera_packets:
        raise ValueError("No camera packets found.")

    packet_bytes = b"".join(packet for _, packet in camera_packets)
    container = av.open(io.BytesIO(packet_bytes))
    images = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    img_ts = np.array([ts for ts, _ in camera_packets], dtype=np.float64)

    if not images:
        raise ValueError("Failed to decode any camera frames from H264 packets.")

    if len(images) != len(camera_packets):
        print(
            f"Warning: decoded {len(images)} frames from {len(camera_packets)} packets. "
            "Truncating to the shorter length."
        )
    count = min(len(images), len(camera_packets))
    return img_ts[:count], np.asarray(images[:count], dtype=np.uint8)


def load_mcap_streams(mcap_path, pose_topic, gripper_topic, camera_topic):
    eef_data = []
    gripper_data = []
    camera_packets = []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded_msg in reader.iter_decoded_messages():
            topic = channel.topic
            ts = message.log_time / 1e9

            if topic == pose_topic:
                pos = decoded_msg.pose.position
                quat = decoded_msg.pose.orientation
                eef_data.append([ts, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w])
            elif topic == gripper_topic:
                gripper_data.append([ts, decoded_msg.value])
            elif topic == camera_topic:
                camera_packets.append((ts, decoded_msg.data))

    if not eef_data:
        raise ValueError(f"{mcap_path} has no pose data on topic {pose_topic}")
    if not gripper_data:
        raise ValueError(f"{mcap_path} has no gripper data on topic {gripper_topic}")
    if not camera_packets:
        raise ValueError(f"{mcap_path} has no camera data on topic {camera_topic}")

    eef_data = deduplicate_last_by_timestamp(np.asarray(eef_data, dtype=np.float64))
    gripper_data = deduplicate_last_by_timestamp(np.asarray(gripper_data, dtype=np.float64))
    camera_packets.sort(key=lambda item: item[0])
    img_ts, images = decode_h264_packets(camera_packets)
    img_ts, images = deduplicate_images_by_timestamp(img_ts, images)

    if len(eef_data) < 2:
        raise ValueError(f"{mcap_path} needs at least 2 pose samples after deduplication.")
    if len(gripper_data) < 2:
        raise ValueError(f"{mcap_path} needs at least 2 gripper samples after deduplication.")
    if len(img_ts) < 1:
        raise ValueError(f"{mcap_path} needs at least 1 decoded image.")

    return eef_data, gripper_data, img_ts, images


def estimate_frequency_hz(eef_ts):
    diffs = np.diff(eef_ts)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        raise ValueError("Cannot estimate resample frequency from pose timestamps.")
    return float(np.round(1.0 / np.median(diffs), 3))


def nearest_indices(reference_ts, query_ts):
    right = np.searchsorted(reference_ts, query_ts, side="left")
    right = np.clip(right, 0, len(reference_ts) - 1)
    left = np.clip(right - 1, 0, len(reference_ts) - 1)
    choose_left = np.abs(query_ts - reference_ts[left]) <= np.abs(reference_ts[right] - query_ts)
    return np.where(choose_left, left, right)


def resize_images(images, image_size):
    width, height = image_size
    if images.shape[2] == width and images.shape[1] == height:
        return images
    resized = [
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    return np.asarray(resized, dtype=np.uint8)


def build_resample_grid(eef_ts, gripper_ts, img_ts, target_hz):
    start_time = max(eef_ts[0], gripper_ts[0], img_ts[0])
    end_time = min(eef_ts[-1], gripper_ts[-1], img_ts[-1])
    if end_time <= start_time:
        raise ValueError("No overlapping time range across pose, gripper, and camera streams.")

    step = 1.0 / target_hz
    count = int(np.floor((end_time - start_time) / step)) + 1
    if count < 2:
        raise ValueError(
            f"Overlap window too short for {target_hz:.3f}Hz resampling: "
            f"{end_time - start_time:.6f}s"
        )
    return start_time + np.arange(count, dtype=np.float64) * step


def convert_single_mcap(mcap_path, image_size, target_hz, pose_topic, gripper_topic, camera_topic):
    eef_data, gripper_data, img_ts, images = load_mcap_streams(
        mcap_path=mcap_path,
        pose_topic=pose_topic,
        gripper_topic=gripper_topic,
        camera_topic=camera_topic,
    )

    eef_ts = eef_data[:, 0]
    gripper_ts = gripper_data[:, 0]
    if target_hz is None:
        target_hz = estimate_frequency_hz(eef_ts)

    sample_ts = build_resample_grid(eef_ts, gripper_ts, img_ts, target_hz)

    pos_interp = interp1d(eef_ts, eef_data[:, 1:4], axis=0, kind="linear", assume_sorted=True)
    pos = pos_interp(sample_ts)

    quat = eef_data[:, 4:8]
    quat_slerp = Slerp(eef_ts, Rotation.from_quat(quat))
    rot_aa = quat_slerp(sample_ts).as_rotvec()

    gripper_interp = interp1d(
        gripper_ts,
        gripper_data[:, 1],
        axis=0,
        kind="linear",
        assume_sorted=True,
    )
    gripper = gripper_interp(sample_ts)[:, None]

    image_indices = nearest_indices(img_ts, sample_ts)
    aligned_images = resize_images(images[image_indices], image_size)

    start_pose = np.concatenate([pos[0], rot_aa[0]], axis=0).astype(np.float32)
    end_pose = np.concatenate([pos[-1], rot_aa[-1]], axis=0).astype(np.float32)
    episode_length = len(sample_ts)

    episode = {
        "robot0_eef_pos": pos.astype(np.float32),
        "robot0_eef_rot_axis_angle": rot_aa.astype(np.float32),
        "robot0_gripper_width": gripper.astype(np.float32),
        "camera0_rgb": aligned_images.astype(np.uint8),
        "robot0_demo_start_pose": np.repeat(start_pose[None], episode_length, axis=0),
        "robot0_demo_end_pose": np.repeat(end_pose[None], episode_length, axis=0),
    }
    stats = {
        "path": str(mcap_path),
        "target_hz": target_hz,
        "episode_length": episode_length,
        "start_time": float(sample_ts[0]),
        "end_time": float(sample_ts[-1]),
        "image_shape": tuple(aligned_images.shape[1:]),
    }
    return episode, stats


def remove_existing_output(output_path):
    if not output_path.exists():
        return
    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def save_numpy_archive(replay_buffer, output_path):
    archive_data = {
        **{f"data__{key}": value for key, value in replay_buffer.data.items()},
        **{f"meta__{key}": value for key, value in replay_buffer.meta.items()},
    }
    np.savez_compressed(str(output_path), **archive_data)


def save_replay_buffer(replay_buffer, output_path):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_existing_output(output_path)
    if output_path.suffix == ".npz":
        save_numpy_archive(replay_buffer, output_path)
    elif output_path.suffix == ".zip":
        with ZipStore(str(output_path), mode="w") as store:
            replay_buffer.save_to_store(store)
    else:
        replay_buffer.save_to_store(LocalStore(str(output_path)))


def convert_mcap_files_to_zarr(
    mcap_paths,
    output_path,
    image_size,
    target_hz,
    pose_topic,
    gripper_topic,
    camera_topic,
):
    replay_buffer = ReplayBuffer.create_empty_numpy()
    episode_stats = []
    dataset_hz = target_hz

    for mcap_path in mcap_paths:
        episode, stats = convert_single_mcap(
            mcap_path=mcap_path,
            image_size=image_size,
            target_hz=dataset_hz,
            pose_topic=pose_topic,
            gripper_topic=gripper_topic,
            camera_topic=camera_topic,
        )
        dataset_hz = stats["target_hz"]
        replay_buffer.add_episode(episode)
        episode_stats.append(stats)
        print(
            f"Added {mcap_path.name}: T={stats['episode_length']}, "
            f"hz={stats['target_hz']:.3f}, image_shape={stats['image_shape']}"
        )

    save_replay_buffer(replay_buffer, output_path)
    total_steps = int(replay_buffer.n_steps)

    print(f"\nWrote {output_path}")
    print(f"Episodes: {replay_buffer.n_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Resample frequency: {dataset_hz:.3f}Hz")
    print(f"Recommended training command:")
    print(
        "cd /home/phi5090ii/NYX/arx-difussion-deploy/umi-diffusion-training && "
        "python train.py --config-name=train_diffusion_unet_timm_umi_workspace "
        f"task.dataset_path={output_path} task.dataset_frequeny={dataset_hz:.3f}"
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Convert one or more MCAP files into a UMI-compatible Zarr dataset."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="MCAP files or directories containing .mcap files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="umi-diffusion-training/data/mcap_dataset.npz",
        help="Output dataset path. Recommended: .npz. .zarr and .zarr.zip are also accepted.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=None,
        help="Target fixed resample frequency. Defaults to pose-rate estimation from the first MCAP.",
    )
    parser.add_argument(
        "--image-size",
        default="224,224",
        help="Output image size as width,height. Default: 224,224",
    )
    parser.add_argument("--pose-topic", default=DEFAULT_POSE_TOPIC)
    parser.add_argument("--gripper-topic", default=DEFAULT_GRIPPER_TOPIC)
    parser.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_files = expand_inputs(args.inputs)
    output_path = Path(os.path.expanduser(args.output))
    image_size = parse_image_size(args.image_size)
    if args.hz is not None and args.hz <= 0:
        raise ValueError("--hz must be positive.")

    convert_mcap_files_to_zarr(
        mcap_paths=input_files,
        output_path=output_path,
        image_size=image_size,
        target_hz=args.hz,
        pose_topic=args.pose_topic,
        gripper_topic=args.gripper_topic,
        camera_topic=args.camera_topic,
    )


if __name__ == "__main__":
    main()
