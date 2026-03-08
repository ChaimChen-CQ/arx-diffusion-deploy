# Diffusion Policy —— UMI 扩散策略训练框架

基于扩散模型（Diffusion Model）的机器人操作策略学习框架，支持 UMI（Universal Manipulation Interface）单臂/双臂真实机器人场景，以及 RoboMimic、Push-T 等仿真基准任务。

---

## 目录

- [项目概览](#项目概览)
- [环境安装](#环境安装)
- [数据来源与格式](#数据来源与格式)
- [数据预处理](#数据预处理)
- [训练模型](#训练模型)
- [配置文件说明](#配置文件说明)
- [模型架构](#模型架构)
- [推理与部署](#推理与部署)
- [目录结构](#目录结构)

---

## 项目概览

本框架实现了两种主要的扩散策略架构：

| 架构 | 视觉编码器 | 扩散骨干网络 | 适用场景 |
|------|-----------|-------------|---------|
| **Transformer（推荐）** | CLIP ViT-B/16（所有 patch tokens） | Transformer Decoder（7层，768维） | UMI 单臂/双臂 |
| **UNet** | TIMM（ResNet/ViT，注意力池化） | 条件 UNet 1D | 仿真基准/消融实验 |

核心特性：
- 使用 OpenAI CLIP 预训练 ViT-B/16 作为视觉骨干，冻结或低学习率微调
- 支持多相机输入，内置时间戳对齐与延迟补偿
- 姿态统一表示为 **rotation_6d**（6D 旋转），避免欧拉角/四元数奇点问题
- 支持相对姿态（relative）、绝对姿态（absolute）、增量姿态（delta）三种形式
- 支持 EMA（指数移动平均）、WandB 日志、Accelerate 多 GPU 训练

---

## 环境安装

### 1. 创建 Conda 环境

```bash
conda env create -f conda_environment.yaml
conda activate robodiff
```

> 依赖 CUDA 12.1 + PyTorch 2.1.0，请确认 GPU 驱动版本兼容。

### 2. 主要依赖说明

| 类别 | 包 | 版本 |
|------|-----|------|
| 深度学习框架 | PyTorch | 2.1.0 |
| 视觉模型库 | timm | 0.9.7 |
| 扩散调度器 | diffusers | 0.18.2 |
| 配置管理 | hydra-core | 1.2.0 |
| 分布式训练 | accelerate | 0.24 |
| 数据存储 | zarr | 2.16 |
| 实验日志 | wandb | 0.15.8 |
| 机器人控制 | ur-rtde | 1.5.6 |
| 仿真环境 | free-mujoco-py | 2.1.6 |

---

## 数据来源与格式

### 数据采集方式

数据通过 **UMI（Universal Manipulation Interface）** 手持夹爪硬件采集。操作者手持 UMI 夹爪设备执行示教（Teleoperation/Human Demonstration），系统同步记录：

- **RGB 图像**：GoPro 鱼眼相机（第一视角，机载相机），分辨率 224×224，约 15fps
- **末端执行器位姿**：由 ARTag 视觉定位或 SLAM 计算，包含位置 (x, y, z) 和旋转（轴角）
- **夹爪宽度**：模拟量，范围约 0～0.09m

### 数据格式：Zarr

所有演示数据统一存储为 **`.zarr.zip`** 格式（Zarr v2 压缩归档）：

```
dataset.zarr.zip
└── data/
    ├── camera0_rgb        # (N, H, W, 3)  uint8，所有帧连续存储
    ├── robot0_eef_pos     # (N, 3)         末端位置 [m]
    ├── robot0_eef_rot_axis_angle  # (N, 3) 末端旋转（轴角）[rad]
    ├── robot0_gripper_width       # (N, 1) 夹爪宽度 [m]
    └── ...
└── meta/
    └── episode_ends       # (E,) 每段演示的结束帧索引
```

> `episode_ends` 数组记录了每条演示轨迹的结束位置，用于切分不同的数据段。

### 双臂场景额外字段

双臂任务（`umi_bimanual`）额外包含：

```
robot1_eef_pos                      # 第二臂末端位置
robot1_eef_rot_axis_angle           # 第二臂末端旋转
robot1_gripper_width                # 第二臂夹爪
robot0_eef_pos_wrt1                 # 臂0相对臂1的位置（跨臂相对姿态）
robot1_eef_pos_wrt0                 # 臂1相对臂0的位置
```

---

## 数据预处理

### 真实采集数据转换

若使用 UMI 原始采集格式，需先转换为训练用 zarr 格式：

```bash
python scripts/real_dataset_conversion.py \
    --input /path/to/raw_demo_session \
    --output /path/to/dataset.zarr.zip
```

或者直接使用 UMI 项目提供的工具完成采集后转换（参考上级目录的采集脚本）。

### 数据集位置配置

在任务配置文件（如 `config/task/umi.yaml`）中指定数据集路径：

```yaml
dataset_path: /path/to/your/dataset.zarr.zip
val_ratio: 0.05   # 5% 数据用于验证
```

也可以通过命令行覆盖：

```bash
python train.py task.dataset_path=/path/to/dataset.zarr.zip
```

### 数据加载与在线处理（`UmiDataset`）

训练时 `dataset/umi_dataset.py` 会自动完成以下在线处理：

1. **下采样**：每隔 3 帧取 1 帧（`down_sample_steps=3`），有效帧率约 5Hz
2. **延迟补偿**：相机延迟 0.125s、机器人延迟 0.0001s、夹爪延迟 0.02s，通过插值对齐各传感器时间戳
3. **姿态转换**：轴角 → 4×4 变换矩阵 → rotation_6d（6D 旋转表示）
4. **相对姿态计算**：以每段轨迹末端时刻为参考帧，计算各时刻的相对姿态
5. **归一化**：
   - 位置（pos）：范围归一化 → `[-1, 1]`
   - 旋转（rot6d）：恒等归一化（保持不变，因为已是单位尺度）
   - 夹爪宽度（gripper）：范围归一化 → `[-1, 1]`
6. **图像增强**（训练时）：随机裁剪（0.95）、随机旋转（±5°）、颜色抖动

---

## 训练模型

### 快速开始（单 GPU）

```bash
# 进入项目根目录（包含 train.py 的那一级）
cd /path/to/umi-diffusion-training

# 使用 Transformer 架构训练（推荐）
python train.py \
    --config-name=train_diffusion_transformer_umi_workspace \
    task.dataset_path=example_demo_session/dataset.zarr.zip \
    training.device=cuda:0

# 使用 UNet 架构训练
python train.py \
    --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=example_demo_session/dataset.zarr.zip \
    training.device=cuda:0
```

### 多 GPU 训练（Accelerate）

```bash
# 先配置 accelerate
accelerate config

# 启动多 GPU 训练
accelerate launch train.py \
    --config-name=train_diffusion_transformer_umi_workspace \
    task.dataset_path=/path/to/dataset.zarr.zip \
    training.device=cuda
```

### 双臂任务训练

```bash
python train.py \
    --config-name=train_diffusion_unet_umi_bimanual_workspace \
    task.dataset_path=/path/to/bimanual_dataset.zarr.zip
```

### 常用训练参数覆盖

```bash
python train.py \
    --config-name=train_diffusion_transformer_umi_workspace \
    task.dataset_path=/path/to/dataset.zarr.zip \
    training.num_epochs=300 \              # 训练轮次（默认 200）
    training.batch_size=32 \              # batch size（默认 64）
    optimizer.lr=1e-4 \                   # 学习率（默认 3e-4）
    training.seed=43 \                    # 随机种子
    logging.project=my_umi_project \      # WandB 项目名
    checkpoint.topk.k=10                  # 保存最佳 Top-K 检查点
```

### 输出文件

训练结果保存在 `data/outputs/<日期-时间>/` 下：

```
data/outputs/2026.03.08/10.00.00_train_diffusion_transformer_timm/
├── checkpoints/
│   ├── latest.ckpt           # 最新检查点
│   └── epoch=200-val_loss=0.0123.ckpt  # TopK 最佳检查点
├── normalizer.pkl            # 归一化参数（推理时必需）
├── .hydra/                   # Hydra 配置备份
└── wandb/                    # WandB 本地日志
```

---

## 配置文件说明

配置文件位于 `diffusion_policy/config/`，使用 **Hydra** 管理。

### 主训练配置

| 配置文件 | 对应架构 | 适用场景 |
|---------|---------|---------|
| `train_diffusion_transformer_umi_workspace.yaml` | Transformer + CLIP ViT | UMI 单臂（**推荐**） |
| `train_diffusion_unet_timm_umi_workspace.yaml` | UNet + TIMM | UMI 单臂（备选） |
| `train_diffusion_unet_umi_bimanual_workspace.yaml` | UNet + TIMM | UMI 双臂 |

### 关键超参数（Transformer 配置）

```yaml
policy:
  # 扩散调度器
  noise_scheduler:
    num_train_timesteps: 50        # 训练时扩散步数
    num_inference_steps: 16        # 推理时去噪步数（DDIM 加速）
    beta_schedule: squaredcos_cap_v2
  
  # Transformer 结构
  n_layer: 7       # Transformer Decoder 层数
  n_head: 8        # 注意力头数
  n_emb: 768       # 嵌入维度
  p_drop_attn: 0.1 # Dropout 概率
  input_pertub: 0.1 # 输入扰动强度（缓解 exposure bias）

optimizer:
  lr: 3.0e-4               # 主干（Transformer）学习率
  obs_encoder_lr: 3.0e-5   # 视觉编码器学习率（更小，微调预训练模型）
  weight_decay: 1.0e-6

training:
  num_epochs: 200
  batch_size: 64
  lr_warmup_steps: 2000    # Cosine 调度器预热步数
  use_ema: true            # 启用 EMA
```

### 任务配置（`config/task/umi.yaml`）

```yaml
obs:
  camera0_rgb:
    shape: [3, 224, 224]
    horizon: 2              # 使用最近 2 帧图像作为观测
    down_sample_steps: 3    # 每 3 步采 1 帧
  robot0_eef_pos:
    shape: [3]
    horizon: 2
  robot0_eef_rot_axis_angle:
    shape: [6]              # rotation_6d（内部转换）
    horizon: 2

action:
  shape: [10]     # pos(3) + rot6d(6) + gripper(1)
  horizon: 16     # 预测未来 16 步动作序列
```

---

## 模型架构

### 整体流程

```
RGB 图像 (B, T, 3, 224, 224)
      │
      ▼ CLIP ViT-B/16（冻结/低 lr 微调）
视觉 Tokens (B, T×197, 768)   ← 197 = 1 CLS + 196 patch tokens
      │
      │ + 低维观测（pos/rot/gripper → Linear → 768）
      ▼
条件 Tokens (B, N_cond, 768)
      │
      ▼ Transformer Decoder（7层，8头，768维）
      │  ↑ 动作序列（加噪声后，含可学习位置编码）
      ▼
预测噪声 → DDIM 去噪（16步） → 动作序列 (B, 16, 10)
      │
      ▼ 反归一化
最终动作：pos(3) + rot6d(6) + gripper(1)
```

### 关键设计

- **全 patch tokens**：ViT 输出所有 197 个 tokens（不做全局池化），保留完整空间信息，让 Transformer 自行决定如何使用
- **预层归一化（norm_first=True）**：Transformer 使用 Pre-LN 结构，训练更稳定
- **DDIM 推理**：训练用 50 步 DDPM，推理用 16 步 DDIM，大幅加速推理
- **EMA**：采用指数移动平均（power=0.75，max=0.9999）提升策略稳定性

---

## 推理与部署

### 加载检查点进行推理

```python
from diffusion_policy.workspace.train_diffusion_transformer_timm_workspace import (
    TrainDiffusionTransformerTimmWorkspace
)
import hydra
from omegaconf import OmegaConf

# 加载配置和检查点
cfg = OmegaConf.load('path/to/.hydra/config.yaml')
workspace = TrainDiffusionTransformerTimmWorkspace(cfg, output_dir='.')
workspace.load_checkpoint('path/to/checkpoints/latest.ckpt')

policy = workspace.model
policy.eval()

# 推理（输入观测字典）
obs_dict = {
    'camera0_rgb': camera_frames,           # (T, 3, 224, 224) float32
    'robot0_eef_pos': eef_pos_history,       # (T, 3) float32
    'robot0_eef_rot_axis_angle': rot_history,# (T, 3) float32
    'robot0_gripper_width': gripper_history, # (T, 1) float32
}
action_dict = policy.predict_action(obs_dict)
action = action_dict['action']  # (B, 16, 10)
```

### 真实机器人部署

真实机器人推理相关工具在 `real_world/` 目录下：

- `umi_env.py`：UMI 真实环境封装，管理相机和机器人控制器
- `real_inference_util.py`：推理工具，处理观测缓冲和动作执行
- `multi_realsense.py`：多 RealSense 相机管理
- `rtde_interpolation_controller.py`：UR5 机器人 RTDE 控制器（含轨迹插值）

---

## 目录结构

```
diffusion_policy/
├── config/                    # Hydra 配置文件
│   ├── task/                  # 任务定义（obs/action space）
│   │   ├── umi.yaml           # UMI 单臂任务
│   │   ├── umi_bimanual.yaml  # UMI 双臂任务
│   │   └── umi_image.yaml     # UMI 图像任务
│   ├── train_diffusion_transformer_umi_workspace.yaml  # 主训练配置（推荐）
│   └── train_diffusion_unet_timm_umi_workspace.yaml
│
├── dataset/                   # 数据集实现
│   ├── umi_dataset.py         # UMI 核心数据集（含在线预处理）
│   └── robomimic_replay_dataset.py
│
├── model/                     # 模型组件
│   ├── diffusion/
│   │   ├── transformer_for_action_diffusion.py  # 扩散 Transformer
│   │   └── conditional_unet1d.py                # 条件 UNet 1D
│   └── vision/
│       ├── transformer_obs_encoder.py  # 主视觉编码器（CLIP ViT）
│       └── timm_obs_encoder.py         # TIMM 备选编码器
│
├── policy/                    # 策略（模型 + 调度器 组合）
│   ├── diffusion_transformer_timm_policy.py  # Transformer 策略（推荐）
│   └── diffusion_unet_timm_policy.py         # UNet 策略
│
├── workspace/                 # 训练流程管理
│   ├── train_diffusion_transformer_timm_workspace.py  # 主训练入口
│   └── train_diffusion_unet_image_workspace.py
│
├── common/                    # 通用工具
│   ├── replay_buffer.py       # Zarr 回放缓冲区
│   ├── sampler.py             # 序列采样器（含延迟补偿）
│   ├── normalize_util.py      # 归一化工具
│   └── pose_repr_util.py      # 姿态表示转换
│
├── real_world/                # 真实机器人接口
│   ├── umi_env.py             # UMI 环境
│   ├── rtde_interpolation_controller.py  # UR5 控制器
│   └── multi_realsense.py     # 多相机管理
│
├── scripts/                   # 数据转换脚本
│   └── real_dataset_conversion.py
│
└── conda_environment.yaml     # 环境依赖
```

---

## 参考资料

- [UMI 论文](https://arxiv.org/abs/2402.10329)：Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- [CLIP ViT](https://github.com/openai/CLIP)：OpenAI CLIP 预训练视觉编码器
- [timm](https://github.com/huggingface/pytorch-image-models)：PyTorch 视觉模型库
