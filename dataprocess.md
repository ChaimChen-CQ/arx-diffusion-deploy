# MCAP 转 Diffusion 训练数据

目标目录：

```bash
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy
```

训练代码目录：

```bash
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
```

## 1. 准备数据目录

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy
mkdir -p data
```

把原始 `.mcap` 放到：

```text
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/20260713
```

容器里对应路径是：

```text
/app/data/20260713
```

## 2. 启动 Matrix Studio 容器

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy

sudo bash start_studio_sdk.sh \
  --image-name imagepublic.genrobotai.com/genrobot/matrix-studio:0.2.15
```

如果镜像不存在，先拉镜像：

```bash
sudo docker pull imagepublic.genrobotai.com/genrobot/matrix-studio:0.2.15
```

## 3. 单个 MCAP 测试 VIO

在容器里执行：

```bash
mkdir -p /app/data/vio_output

bash /app/scripts/process_mcap_inner.sh \
  /app/data/20260713/你的文件名.mcap \
  --output-dir /app/data/vio_output
```

检查输出：

```bash
ls /app/data/vio_output
```

## 4. 检查 VIO topic

回到宿主机：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
conda activate umi_mcap

python utils/inspect_mcap.py \
  /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output/输出文件名_vio.mcap
```

确认有这个 topic：

```text
/robot0/vio/relative_eef_pose
```

## 5. 批量跑 VIO

单个文件没问题后，在容器里执行：

```bash
mkdir -p /app/data/vio_output

for f in /app/data/20260713/*.mcap; do
  bash /app/scripts/process_mcap_inner.sh "$f" --output-dir /app/data/vio_output
done
```

## 6. 转成训练用 dataset.npz

回到宿主机：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
conda activate umi_mcap

mkdir -p data/tron1_gripper_20260713

python utils/mcap_to_zarr.py \
  /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/data/vio_output \
  -o data/tron1_gripper_20260713/dataset_raw.npz \
  --image-size 224,224
```

## 7. 训练参数

转换完成后训练使用：

```bash
task.dataset_path=data/tron1_gripper_20260713/dataset_trim2s.npz
task.dataset_frequeny=30
```

注意：这里的 `dataset_frequeny` 是代码里的原始拼写，不要改成 `frequency`。

## 8. 裁掉每条 episode 前 2 秒

本批数据按 30Hz 处理，2 秒对应 60 帧。

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training
conda activate umi_mcap
```

先检查 episode 长度：

```bash
python - <<'PY'
import numpy as np
p = "data/tron1_gripper_20260713/dataset_raw.npz"
with np.load(p, allow_pickle=False) as d:
    ends = d["meta__episode_ends"]
starts = np.r_[0, ends[:-1]]
lens = ends - starts
print("episodes:", len(lens))
print("min/max/mean len:", lens.min(), lens.max(), lens.mean())
print("shorter than 60:", (lens <= 60).sum())
PY
```

裁切生成 `dataset_trim2s.npz`：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

in_path = Path("data/tron1_gripper_20260713/dataset_raw.npz")
out_path = Path("data/tron1_gripper_20260713/dataset_trim2s.npz")
drop_frames = 60

with np.load(in_path, allow_pickle=False) as src:
    arrays = {k: src[k] for k in src.files}

episode_ends = arrays["meta__episode_ends"]
starts = np.r_[0, episode_ends[:-1]]

keep_indices = []
new_episode_ends = []
total = 0

for start, end in zip(starts, episode_ends):
    ep_len = end - start
    if ep_len <= drop_frames:
        print(f"[SKIP episode] too short: {ep_len} frames")
        continue
    kept = np.arange(start + drop_frames, end)
    keep_indices.append(kept)
    total += len(kept)
    new_episode_ends.append(total)

if not keep_indices:
    raise RuntimeError("No episodes left after trimming.")

keep_indices = np.concatenate(keep_indices)
new_episode_ends = np.asarray(new_episode_ends, dtype=episode_ends.dtype)

out = {}
old_total = int(episode_ends[-1])

for k, v in arrays.items():
    if k == "meta__episode_ends":
        out[k] = new_episode_ends
    elif k.startswith("data__") and len(v) == old_total:
        out[k] = v[keep_indices]
    else:
        out[k] = v

if (
    "data__robot0_eef_pos" in out
    and "data__robot0_eef_rot_axis_angle" in out
    and "data__robot0_demo_start_pose" in out
    and "data__robot0_demo_end_pose" in out
):
    starts2 = np.r_[0, new_episode_ends[:-1]]
    for s, e in zip(starts2, new_episode_ends):
        start_pose = np.concatenate([
            out["data__robot0_eef_pos"][s],
            out["data__robot0_eef_rot_axis_angle"][s],
        ]).astype(np.float32)
        end_pose = np.concatenate([
            out["data__robot0_eef_pos"][e - 1],
            out["data__robot0_eef_rot_axis_angle"][e - 1],
        ]).astype(np.float32)
        out["data__robot0_demo_start_pose"][s:e] = start_pose
        out["data__robot0_demo_end_pose"][s:e] = end_pose

np.savez_compressed(out_path, **out)

print(f"Wrote: {out_path}")
print(f"Episodes: {len(episode_ends)} -> {len(new_episode_ends)}")
print(f"Frames: {old_total} -> {int(new_episode_ends[-1])}")
PY
```

确认输出：

```bash
ls -lh data/tron1_gripper_20260713/
```

## 9. 传到服务器

训练只需要传最终的 `dataset_trim2s.npz`。

服务器先创建目录：

```bash
ssh ubuntu@117.50.75.55
mkdir -p /home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260713
exit
```

本地传输：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-diffusion-training

scp data/tron1_gripper_20260713/dataset_trim2s.npz \
  ubuntu@117.50.75.55:/home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training/data/tron1_gripper_20260713/
```

不需要传原始 `.mcap`、`_vio.mcap`、`_vio.json`、`dataset_raw.npz`。

## 10. 服务器训练

```bash
ssh ubuntu@117.50.75.55
cd /home/ubuntu/projects/arx-difussion-deploy/umi-diffusion-training
conda activate umi4090
```

单卡测试：

```bash
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=data/tron1_gripper_20260713/dataset_trim2s.npz \
  task.dataset_frequeny=30 \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32
```

多卡 0、1、3：

```bash
CUDA_VISIBLE_DEVICES=0,1,3 accelerate launch --num_processes 3 train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=data/tron1_gripper_20260713/dataset_trim2s.npz \
  task.dataset_frequeny=30 \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32
```
