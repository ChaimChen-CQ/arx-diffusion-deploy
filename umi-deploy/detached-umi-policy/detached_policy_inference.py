import sys
import os
import time
import click
import numpy as np
import torch
import dill
import hydra
import zmq
import importlib.machinery
import types

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply
import omegaconf
import traceback

def patch_huggingface_hub_cached_download():
    """Keep older diffusers imports working with newer huggingface_hub."""
    try:
        import huggingface_hub
    except ImportError:
        return
    if hasattr(huggingface_hub, "cached_download"):
        return
    if not hasattr(huggingface_hub, "hf_hub_download"):
        return

    def cached_download(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = cached_download

def patch_optional_wandb():
    """Avoid import-time failures from broken wandb installs during inference."""
    try:
        import wandb  # noqa: F401
    except Exception:
        wandb = types.ModuleType("wandb")
        wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
        wandb.log = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return "".join(tb_lines)

class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, config_path: str, ip: str, port: int, device: str, lerobot: bool = False):
        self.lerobot = lerobot
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        
        # 加载权重
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        
        # 确定配置文件路径逻辑
        if config_path is not None:
            self.cfg = omegaconf.OmegaConf.load(config_path)
            print(f"Loading config from specified path: {config_path}")
        else:
            self.cfg = payload['cfg']
            # 原逻辑：导出或读取同名 yaml
            cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
            if not os.path.exists(cfg_path):
                with open(cfg_path, 'w') as f:
                    f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"Loading config from default path: {cfg_path}")

        if self.lerobot:
            omegaconf.OmegaConf.update(
                self.cfg,
                "policy.obs_encoder.lerobot_deploy",
                True,
                merge=False,
                force_add=True,
            )
        print(f"Deployment mode: {'LeRobot/DiT' if self.lerobot else 'legacy U-Net'}")

        self.device = torch.device(device)
        self.policy = self.load_policy(payload)
        self.policy.to(self.device)
        self.policy.eval()
        self.expected_obs_keys = None
        if self.lerobot:
            obs_shape_meta = self.cfg.policy.shape_meta.obs
            self.expected_obs_keys = tuple(
                key
                for key, meta in obs_shape_meta.items()
                if not bool(meta.get("ignore_by_policy", False))
            )
            print(f"Policy observation keys: {list(self.expected_obs_keys)}")
        self.ip = ip
        self.port = port

    def load_policy(self, payload):
        patch_huggingface_hub_cached_download()
        patch_optional_wandb()
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Match the training/evaluation workspace: use EMA weights whenever the
        # checkpoint config enables EMA, otherwise use the raw model weights.
        if self.lerobot and bool(self.cfg.training.get("use_ema", False)) and getattr(workspace, "ema_model", None) is not None:
            policy = workspace.ema_model
            print("[INFO] 已按训练配置加载 workspace.ema_model")
        elif hasattr(workspace, 'model'):
            policy = workspace.model
            print("[INFO] 已加载 workspace.model")
        elif hasattr(workspace, 'policy'):
            policy = workspace.policy
            print("[INFO] 检测到 Unet 架构，已成功引用 workspace.policy")
        else:
            raise AttributeError("Workspace 对象中既找不到 'model' 也找不到 'policy' 属性。")
        # --------------------------------------------------
        
        return policy

    def predict_action(self, obs_dict_np: dict):
        with torch.no_grad():
            if self.lerobot:
                missing_keys = [
                    key for key in self.expected_obs_keys if key not in obs_dict_np
                ]
                if missing_keys:
                    raise KeyError(
                        f"Missing policy observation keys: {missing_keys}; "
                        f"received: {sorted(obs_dict_np.keys())}"
                    )
                obs_dict_np = {
                    key: obs_dict_np[key] for key in self.expected_obs_keys
                }
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
            del result
            del obs_dict
        return action
    
    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")
        while True:
            obs_dict_np = socket.recv_pyobj()
            try:
                start_time = time.monotonic()
                action = self.predict_action(obs_dict_np)
                print(f'Inference time: {time.monotonic() - start_time:.3f} s')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                action = err_str
            # NumPy 2.x pickles arrays through ``numpy._core`` while older
            # clients may only provide ``numpy.core``. Send successful actions
            # as builtin lists so inference works across NumPy major versions.
            if isinstance(action, np.ndarray):
                action = action.tolist()
            socket.send_pyobj(action)
    
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--config', '-c', default=None, help='Path to specific yaml config file') # 新增参数
@click.option('--ip', default="0.0.0.0")
@click.option('--port', default=8766, help="Port to listen on")
@click.option('--device', default="cuda:0")
@click.option('--lerobot/--no-lerobot', default=False, help='Enable LeRobot/DiT deployment compatibility. Default keeps the legacy U-Net path.')
def main(input, config, ip, port, device, lerobot):
    node = PolicyInferenceNode(input, config, ip, port, device, lerobot=lerobot)
    node.run_node()

if __name__ == "__main__":
    main()
