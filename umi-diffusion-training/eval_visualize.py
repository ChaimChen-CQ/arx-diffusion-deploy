"""
Offline trajectory visualization with 3D side panel.

Layout (side by side):

  ┌──────────────────────┬──────────────────────────┐
  │                      │   3D trajectory view      │
  │  wrist RGB image     │   (pred rainbow + GT green│
  │  + 2D top-down       │    rotating slowly)       │
  │  overlay             ├──────────────────────────┤
  │                      │   Z height  |  Gripper    │
  │                      │   over time |  over time  │
  └──────────────────────┴──────────────────────────┘

Usage
-----
    python eval_visualize.py \\
        -i <checkpoint.ckpt> \\
        -d <dataset.npz> \\
        -o eval_vis.mp4 \\
        [--fps 10] [--num_frames 300] [--scale_m 0.35] [--rotate]
"""

import argparse
import io
import pathlib
import sys

import cv2
import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OmegaConf.register_new_resolver("eval", eval, replace=True)

# ── colours (BGR for cv2, RGB for matplotlib) ─────────────────────
C_GT      = (0,  200,  80)        # cv2 BGR
C_ORIGIN  = (0,  200, 255)        # cv2 BGR  – pred start
C_TEXT    = (255, 255, 255)       # cv2 BGR


def rainbow_bgr(t: float):
    r = int(255 * t);  g = int(255 * (1 - abs(2*t-1)));  b = int(255 * (1-t))
    return (b, g, r)

def rainbow_rgb(t: float):
    b, g, r = rainbow_bgr(t)
    return (r/255, g/255, b/255)


# ─────────────────────────────────────────────────────────────────────
# Checkpoint / dataset
# ─────────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, device):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill,
                         map_location=device)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    ws  = cls(cfg, output_dir="/tmp/eval_vis_tmp")
    ws.load_payload(payload)
    policy = ws.ema_model if getattr(ws, "ema_model", None) else ws.model
    print("Using", "EMA" if getattr(ws, "ema_model", None) else "base", "model.")
    policy.eval().to(device)
    return cfg, policy


def build_val_loader(cfg, dataset_path):
    ds_cfg = cfg.task.dataset
    OmegaConf.update(ds_cfg, "dataset_path", dataset_path, merge=True)
    OmegaConf.update(ds_cfg, "val_ratio",    0.15,         merge=True)
    ds  = hydra.utils.instantiate(ds_cfg)
    val = ds.get_validation_dataset()
    print(f"Val samples: {len(val)}")
    return DataLoader(val, batch_size=1, shuffle=False, num_workers=0)


# ─────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────
def tensor_to_bgr(t: torch.Tensor, out_hw: int = 448) -> np.ndarray:
    img  = t.cpu().float().numpy()
    mean = np.array([0.485,0.456,0.406], np.float32)[:,None,None]
    std  = np.array([0.229,0.224,0.225], np.float32)[:,None,None]
    img  = np.clip(img*std + mean, 0, 1)
    img  = (img.transpose(1,2,0)*255).astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.shape[0] != out_hw:
        img = cv2.resize(img, (out_hw, out_hw), cv2.INTER_LINEAR)
    return img


# ─────────────────────────────────────────────────────────────────────
# 2-D top-down overlay (left panel)
# ─────────────────────────────────────────────────────────────────────
def world_to_px(xy, img_hw, scale_m):
    cx = cy = img_hw // 2
    s  = (img_hw/2) / scale_m
    u  = (cx + xy[:,0]*s).astype(int)
    v  = (cy - xy[:,1]*s).astype(int)
    return np.stack([u,v], 1)


def draw_2d_overlay(canvas, pred_rel, gt_rel, scale_m, frame_idx, mse):
    H, W = canvas.shape[:2]
    Tp   = pred_rel.shape[0]

    pred_px = world_to_px(pred_rel[:,:2], H, scale_m)
    gt_px   = world_to_px(gt_rel[:,  :2], H, scale_m)

    z_all   = np.concatenate([pred_rel[:,2], gt_rel[:,2]])
    z_min, z_max = z_all.min()-0.01, z_all.max()+0.01
    z_range = max(z_max-z_min, 1e-3)
    def z_r(z): return int(5 + 10*(z-z_min)/z_range)

    # GT dashed
    for i in range(len(gt_px)-1):
        if i%2==0:
            cv2.line(canvas, tuple(gt_px[i]), tuple(gt_px[i+1]), C_GT, 1, cv2.LINE_AA)
    for i,(px,py) in enumerate(gt_px):
        cv2.circle(canvas, (px,py), z_r(gt_rel[i,2]), C_GT, 1, cv2.LINE_AA)

    # Pred rainbow
    for i in range(len(pred_px)-1):
        cv2.line(canvas, tuple(pred_px[i]), tuple(pred_px[i+1]),
                 rainbow_bgr(i/max(Tp-1,1)), 2, cv2.LINE_AA)
    for i,(px,py) in enumerate(pred_px):
        col = rainbow_bgr(i/max(Tp-1,1))
        cv2.circle(canvas, (px,py), z_r(pred_rel[i,2]), col, -1, cv2.LINE_AA)
        gw = pred_rel[i,9] if pred_rel.shape[1]>9 else 0
        cv2.circle(canvas, (px,py), z_r(pred_rel[i,2])+2,
                   (0,255,255) if gw>0.04 else (0,60,200), 1, cv2.LINE_AA)

    # Origin markers
    cx = cy = H//2
    cv2.drawMarker(canvas, (cx,cy), C_ORIGIN, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    gt0 = tuple(gt_px[0])
    pts = np.array([[gt0[0], gt0[1]-10],[gt0[0]-7, gt0[1]+6],[gt0[0]+7, gt0[1]+6]], np.int32)
    cv2.fillPoly(canvas, [pts], C_GT)

    # Scale bar
    bar_px = int(W*0.12)
    bar_m  = bar_px / ((W/2)/scale_m)
    cv2.line(canvas, (10, H-20), (10+bar_px, H-20), C_TEXT, 2)
    cv2.putText(canvas, f"{bar_m:.2f}m", (10, H-26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)

    # HUD
    for li, txt in enumerate([f"frame {frame_idx:04d}", f"MSE {mse:.5f}",
                               "blue->red: near->far",
                               "* pred t0  ^ GT t0"]):
        cv2.putText(canvas, txt, (8, 16+li*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT, 1, cv2.LINE_AA)

    # Legend (top-right)
    lx = W-170
    cv2.putText(canvas, "-- pred", (lx, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, rainbow_bgr(0.4), 1, cv2.LINE_AA)
    cv2.putText(canvas, "-- GT",   (lx, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GT, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────
# 3-D matplotlib panel (right top)
# ─────────────────────────────────────────────────────────────────────
class Panel3D:
    """Reusable matplotlib 3-D figure — update data each frame."""

    def __init__(self, px_w=460, px_h=320, dpi=110, rotate=False):
        self.fig = plt.figure(figsize=(px_w/dpi, px_h/dpi), dpi=dpi)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X (m)", fontsize=7, labelpad=2)
        self.ax.set_ylabel("Y (m)", fontsize=7, labelpad=2)
        self.ax.set_zlabel("Z (m)", fontsize=7, labelpad=2)
        self.ax.tick_params(labelsize=6)
        self.fig.tight_layout(pad=0.5)
        self._px_w    = px_w
        self._px_h    = px_h
        self._rotate  = rotate
        self._frame   = 0
        self._elev    = 25
        self._azim    = -60

    def render(self, pred_rel, gt_rel):
        """Return (H,W,3) uint8 BGR numpy array."""
        ax = self.ax
        ax.cla()

        Tp   = pred_rel.shape[0]
        t_ax = np.linspace(0, 1, Tp)

        # GT trajectory
        ax.plot(gt_rel[:,0], gt_rel[:,1], gt_rel[:,2],
                color=(0,0.78,0.31), lw=1.4, ls="--", label="GT", zorder=2)
        ax.scatter(gt_rel[:,0], gt_rel[:,1], gt_rel[:,2],
                   c=[[0,0.78,0.31]]*Tp, s=8, alpha=0.6, zorder=3)

        # Predicted trajectory (coloured segments)
        for i in range(Tp-1):
            col = rainbow_rgb(i/max(Tp-1,1))
            ax.plot(pred_rel[i:i+2,0], pred_rel[i:i+2,1], pred_rel[i:i+2,2],
                    color=col, lw=2.2, zorder=4)
        colors_pred = [rainbow_rgb(i/max(Tp-1,1)) for i in range(Tp)]
        ax.scatter(pred_rel[:,0], pred_rel[:,1], pred_rel[:,2],
                   c=colors_pred, s=14, zorder=5)

        # Origin
        ax.scatter([0],[0],[0], marker="*", color="gold", s=120, zorder=6)

        # Axis limits — keep stable across frames
        rng = 0.02
        all_pts = np.vstack([pred_rel[:,:3], gt_rel[:,:3]])
        for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            lo = all_pts[:,dim].min()-rng
            hi = all_pts[:,dim].max()+rng
            mid = (lo+hi)/2;  half = max((hi-lo)/2, 0.05)
            setter(mid-half, mid+half)

        ax.set_xlabel("X",fontsize=7,labelpad=1)
        ax.set_ylabel("Y",fontsize=7,labelpad=1)
        ax.set_zlabel("Z",fontsize=7,labelpad=1)
        ax.tick_params(labelsize=6, pad=1)
        ax.set_title("3D trajectory", fontsize=8, pad=4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.5)

        # slow azimuth rotation
        azim = self._azim + (self._frame * 0.8 if self._rotate else 0)
        ax.view_init(elev=self._elev, azim=azim)
        self._frame += 1

        self.fig.tight_layout(pad=0.3)
        self.fig.canvas.draw()
        fw, fh = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fh, fw, 3)
        if (fw, fh) != (self._px_w, self._px_h):
            buf = cv2.resize(buf, (self._px_w, self._px_h), cv2.INTER_LINEAR)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────
# Time-series panel (right bottom) — Z and gripper
# ─────────────────────────────────────────────────────────────────────
class PanelTimeSeries:
    def __init__(self, px_w=460, px_h=200, dpi=110):
        self.fig = plt.figure(figsize=(px_w/dpi, px_h/dpi), dpi=dpi)
        self._px_w = px_w;  self._px_h = px_h

    def render(self, pred_rel, gt_rel):
        self.fig.clf()
        gs  = gridspec.GridSpec(1, 2, figure=self.fig, wspace=0.4)
        Tp  = pred_rel.shape[0]
        t   = np.arange(Tp)

        # Z
        ax1 = self.fig.add_subplot(gs[0])
        ax1.plot(t, pred_rel[:,2], color="steelblue", lw=1.5, label="pred")
        ax1.plot(t, gt_rel[:,2],   color=(0,0.78,0.31), lw=1.2, ls="--", label="GT")
        ax1.set_title("Z (height)", fontsize=8)
        ax1.set_xlabel("step", fontsize=7); ax1.set_ylabel("m", fontsize=7)
        ax1.tick_params(labelsize=6); ax1.legend(fontsize=6); ax1.grid(lw=0.3)

        # Gripper
        ax2 = self.fig.add_subplot(gs[1])
        ax2.plot(t, pred_rel[:,9], color="darkorange", lw=1.5, label="pred")
        ax2.plot(t, gt_rel[:,9],   color=(0,0.78,0.31), lw=1.2, ls="--", label="GT")
        ax2.set_ylim(-0.01, 0.10)
        ax2.set_title("Gripper width", fontsize=8)
        ax2.set_xlabel("step", fontsize=7); ax2.set_ylabel("m", fontsize=7)
        ax2.tick_params(labelsize=6); ax2.legend(fontsize=6); ax2.grid(lw=0.3)

        self.fig.tight_layout(pad=0.4)
        self.fig.canvas.draw()
        fw, fh = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fh, fw, 3)
        if (fw, fh) != (self._px_w, self._px_h):
            buf = cv2.resize(buf, (self._px_w, self._px_h), cv2.INTER_LINEAR)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────
# Compose final frame
# ─────────────────────────────────────────────────────────────────────
def compose(left: np.ndarray, top_right: np.ndarray,
            bot_right: np.ndarray) -> np.ndarray:
    """
    left:      (H, W, 3)
    top_right: (H_top, W_right, 3)
    bot_right: (H_bot, W_right, 3)
    Always returns exactly (H, W + W_r, 3).
    """
    H   = left.shape[0]
    W_r = top_right.shape[1]

    H_top = int(H * 0.62)
    H_bot = H - H_top                       # sum is exactly H

    tr = cv2.resize(top_right, (W_r, H_top), cv2.INTER_LINEAR)
    br = cv2.resize(bot_right, (W_r, H_bot), cv2.INTER_LINEAR)

    right = np.vstack([tr, br])             # exactly (H, W_r, 3)
    # guard: ensure left is also H tall (should always be true)
    if left.shape[0] != H:
        left = cv2.resize(left, (left.shape[1], H), cv2.INTER_LINEAR)

    return np.hstack([left, right])         # (H, W + W_r, 3)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--checkpoint", required=True)
    parser.add_argument("-d", "--dataset",    required=True)
    parser.add_argument("-o", "--output",     default="eval_vis.mp4")
    parser.add_argument("--fps",        type=int,   default=10)
    parser.add_argument("--num_frames", type=int,   default=None)
    parser.add_argument("--img_size",   type=int,   default=448)
    parser.add_argument("--right_w",    type=int,   default=500,
                        help="Width of the right 3D panel (pixels)")
    parser.add_argument("--scale_m",    type=float, default=0.35)
    parser.add_argument("--rotate",     action="store_true",
                        help="Slowly rotate the 3D view (looks nice in video)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.dataset}")
    print(f"Output     : {args.output}")

    cfg, policy = load_policy(args.checkpoint, args.device)
    loader      = build_val_loader(cfg, args.dataset)

    p3d  = Panel3D(px_w=args.right_w, px_h=int(args.img_size*0.62),
                   rotate=args.rotate)
    pts  = PanelTimeSeries(px_w=args.right_w,
                           px_h=args.img_size - int(args.img_size*0.62))

    writer    = None
    frame_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Rendering"):
            if args.num_frames and frame_idx >= args.num_frames:
                break

            obs    = {k: v.to(args.device) for k, v in batch["obs"].items()}
            gt_act = batch["action"].to(args.device)

            pred_act = policy.predict_action(obs, None)["action_pred"]
            mse      = torch.nn.functional.mse_loss(pred_act, gt_act).item()

            pred_np  = pred_act[0].cpu().numpy()    # (Tp, 10)
            gt_np    = gt_act[0].cpu().numpy()

            # relative coords
            pred_rel = pred_np.copy(); pred_rel[:,:3] -= pred_np[0,:3]
            gt_rel   = gt_np.copy();   gt_rel[:,  :3] -= gt_np[0,  :3]

            # ── left panel: RGB + 2D overlay ──────────────────────
            rgb_key = "camera0_rgb"
            if rgb_key in obs:
                left = tensor_to_bgr(obs[rgb_key][0,-1].cpu(), args.img_size)
            else:
                left = np.zeros((args.img_size, args.img_size, 3), np.uint8)
            left = draw_2d_overlay(left, pred_rel, gt_rel,
                                   args.scale_m, frame_idx, mse)

            # ── right panels ──────────────────────────────────────
            tr = p3d.render(pred_rel, gt_rel)
            br = pts.render(pred_rel, gt_rel)

            frame = compose(left, tr, br)

            if writer is None:
                h, w   = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out    = str(pathlib.Path(args.output).resolve())
                writer = cv2.VideoWriter(out, fourcc, args.fps, (w, h))
                print(f"Video {w}×{h} @ {args.fps} fps → {out}")

            writer.write(frame)
            frame_idx += 1

    if writer:
        writer.release()
        plt.close("all")
        print(f"\nDone — {frame_idx} frames → {args.output}")


if __name__ == "__main__":
    main()
