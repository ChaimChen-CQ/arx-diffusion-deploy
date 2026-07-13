import argparse
import pathlib
import sys

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check runtime policy behavior on a saved obs npy file."
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt")
    parser.add_argument(
        "--obs-npy",
        required=True,
        help="Path to saved eval obs/<episode>/<step>.npy file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional yaml config override, matching detached_policy_inference.py behavior.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_workspace(ckpt_path: str, config_path: str, device: str):
    payload = torch.load(open(ckpt_path, "rb"), map_location=device, pickle_module=dill)
    if config_path is not None:
        cfg = OmegaConf.load(config_path)
    else:
        cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    return payload, cfg, workspace


def main():
    args = parse_args()
    _, cfg, workspace = load_workspace(args.ckpt, args.config, args.device)
    model = getattr(workspace, "model", None)
    ema_model = getattr(workspace, "ema_model", None)

    obs_data = np.load(args.obs_npy, allow_pickle=True).item()
    obs_dict_np = obs_data["obs_dict_np"]
    obs_dict_np = {
        key: value.astype(np.float32) if value.dtype != np.uint8 else value
        for key, value in obs_dict_np.items()
    }
    obs_dict = dict_apply(
        obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(args.device)
    )

    print("config_target:", cfg._target_)
    print("policy_target:", cfg.policy._target_)
    print("has_model:", model is not None)
    print("has_ema_model:", ema_model is not None)
    if model is not None:
        print("model_type:", type(model).__name__)
        model.to(args.device)
        model.eval()
    if ema_model is not None:
        print("ema_model_type:", type(ema_model).__name__)
        ema_model.to(args.device)
        ema_model.eval()
    print("obs_keys:", sorted(obs_dict_np.keys()))
    for key, value in obs_dict_np.items():
        print(
            f"obs[{key}]: shape={value.shape} dtype={value.dtype} "
            f"min={value.min():.6f} max={value.max():.6f}"
        )

    with torch.no_grad():
        if model is not None and hasattr(model, "obs_encoder"):
            z1 = model.obs_encoder(obs_dict)
            z2 = model.obs_encoder(obs_dict)
            print(
                "encoder_repeat_diff:",
                {
                    "equal": bool(torch.allclose(z1, z2)),
                    "max_abs_diff": float((z1 - z2).abs().max().item()),
                    "mean_abs_diff": float((z1 - z2).abs().mean().item()),
                },
            )

        if model is not None:
            a1 = model.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()
            a2 = model.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()
            print(
                "model_repeat_diff:",
                {
                    "equal": bool(np.allclose(a1, a2)),
                    "max_abs_diff": float(np.abs(a1 - a2).max()),
                    "mean_abs_diff": float(np.abs(a1 - a2).mean()),
                },
            )
            print("model_first_action:", np.array2string(a1[0], precision=6, suppress_small=True))

        if model is not None and ema_model is not None:
            torch.manual_seed(0)
            a_model = (
                model.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()
            )
            torch.manual_seed(0)
            a_ema = (
                ema_model.predict_action(obs_dict)["action_pred"][0].detach().cpu().numpy()
            )
            print(
                "model_vs_ema:",
                {
                    "max_abs_diff": float(np.abs(a_model - a_ema).max()),
                    "mean_abs_diff": float(np.abs(a_model - a_ema).mean()),
                },
            )
            print(
                "ema_first_action:",
                np.array2string(a_ema[0], precision=6, suppress_small=True),
            )


if __name__ == "__main__":
    main()
