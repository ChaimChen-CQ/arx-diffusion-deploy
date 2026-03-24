"""
Offline evaluation of a checkpoint on the validation split.
Does NOT require any real robot hardware.

Usage:
    python eval_offline.py \
        -i data/outputs/2026.03.23/16.58.09_train_diffusion_unet_timm_umi/checkpoints/epoch=0110-train_loss=0.012.ckpt \
        -d data/umi-pick-cube/dataset.npz \
        [--num_samples 200] \
        [--batch_size 16] \
        [--device cuda]
"""

import argparse
import pathlib
import sys

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── make sure repo root is on the path ──────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_workspace_and_policy(ckpt_path: str, device: str):
    payload = torch.load(
        open(ckpt_path, "rb"), pickle_module=dill, map_location=device
    )
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp/eval_offline_tmp")

    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("Using EMA model for evaluation.")

    policy.eval()
    policy.to(device)
    return cfg, workspace, policy


def build_val_dataloader(cfg, dataset_path: str, batch_size: int):
    task_cfg = cfg.task
    ds_cfg = task_cfg.dataset
    OmegaConf.update(ds_cfg, "dataset_path", dataset_path, merge=True)
    OmegaConf.update(ds_cfg, "val_ratio", 0.1, merge=True)

    dataset = hydra.utils.instantiate(ds_cfg)
    val_dataset = dataset.get_validation_dataset()
    print(f"Validation samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return val_loader


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compute per-component action MSE. Assumes last dim: [pos(3) | rot(6) | width(1)]."""
    mse = torch.nn.functional.mse_loss
    metrics = {
        "action_mse":       mse(pred,           gt).item(),
        "action_mse_pos":   mse(pred[..., :3],  gt[..., :3]).item(),
        "action_mse_rot":   mse(pred[..., 3:9], gt[..., 3:9]).item(),
        "action_mse_width": mse(pred[..., 9],   gt[..., 9]).item(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--checkpoint", required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("-d", "--dataset", default=None,
                        help="Override dataset path (default: use path from checkpoint cfg)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max number of samples to evaluate (default: all val samples)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    cfg, workspace, policy = load_workspace_and_policy(args.checkpoint, args.device)

    dataset_path = args.dataset or cfg.task.dataset_path
    print(f"Dataset: {dataset_path}")

    val_loader = build_val_dataloader(cfg, dataset_path, args.batch_size)

    all_metrics = []
    n_evaluated = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
            gt_action = batch["action"].to(args.device)

            result = policy.predict_action(obs, None)
            pred_action = result["action_pred"]

            metrics = compute_metrics(pred_action, gt_action)
            all_metrics.append(metrics)

            n_evaluated += gt_action.shape[0]
            if args.num_samples is not None and n_evaluated >= args.num_samples:
                break

    # ── aggregate ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Evaluated on {n_evaluated} validation samples")
    print(f"{'='*50}")
    for key in all_metrics[0]:
        mean_val = np.mean([m[key] for m in all_metrics])
        print(f"  {key:<30s}: {mean_val:.6f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
