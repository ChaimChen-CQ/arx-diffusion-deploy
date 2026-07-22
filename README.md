# arx-difussion-deploy

`arx-difussion-deploy` 是一个面向 ARX5 机械臂的 Diffusion Policy 训练与部署仓库，目标是把 UMI 风格的视觉动作策略接到真实机器人系统上，形成一条从数据、训练、离线评估到真实执行的完整链路。

这个仓库不是单一模块，而是几套能力的组合：

- `umi-diffusion-training`：负责数据处理、模型训练、检查点管理、离线评估。
- `umi-deploy/umi-arx`：负责 ARX5 上层控制环境、观测组织、策略执行脚本。
- `umi-deploy/arx5-sdk`：负责 ARX5 底层通信与控制接口。
- `umi-deploy/gen_controller_sdk_release`：负责 Gen 夹爪串口/ROS 通信与设备侧接入。
- `umi-deploy/detached-umi-policy`：负责将策略推理拆成独立服务，便于 GPU 机器与控制机器分离部署。

## 适用场景

这个仓库适合以下工作流：

- 采集真实机器人演示数据并整理成 UMI 可训练格式。
- 训练基于图像和低维状态的扩散策略。
- 对 checkpoint 做离线评估与可视化分析。
- 将策略部署到 ARX5 + Gen 夹爪硬件链路上执行。

## 仓库结构

```text
arx-difussion-deploy/
├── umi-diffusion-training/
│   ├── train.py
│   ├── eval_offline.py
│   ├── eval_real.py
│   └── diffusion_policy/
├── umi-deploy/
│   ├── umi-arx/
│   │   ├── scripts/eval_arx5.py
│   │   ├── modules/
│   │   └── README_Deploy.md
│   ├── arx5-sdk/
│   ├── gen_controller_sdk_release/
│   ├── detached-umi-policy/
│   └── readmedeploy.md
└── setup.txt
```

## 系统链路

整体上，这个仓库把策略模型和真实机器人控制系统连接起来：

1. 相机、机械臂状态、夹爪状态被组织成观测。
2. 扩散策略根据观测预测未来动作序列。
3. ARX5 控制器执行末端位姿指令。
4. Gen 夹爪驱动执行开合指令。

如果采用分离式部署，策略推理可以放在独立 GPU 机器上，通过网络向控制端返回动作。

## 主要入口

如果你的目标是训练模型，优先看：

- [umi-diffusion-training/README.md](umi-diffusion-training/README.md)
- [umi-diffusion-training/diffusion_policy/README.md](umi-diffusion-training/diffusion_policy/README.md)

如果你的目标是上真实机器人，优先看：

- [umi-deploy/readmedeploy.md](umi-deploy/readmedeploy.md)
- [umi-deploy/umi-arx/README_Deploy.md](umi-deploy/umi-arx/README_Deploy.md)
- [umi-deploy/arx5-sdk/README.md](umi-deploy/arx5-sdk/README.md)
- [umi-deploy/gen_controller_sdk_release/README_CN.md](umi-deploy/gen_controller_sdk_release/README_CN.md)

## 快速上手

### 1. 训练侧

在 `umi-diffusion-training` 中完成：

- 环境安装
- 数据集准备
- 配置检查
- 训练与离线评估

常见入口脚本：

```bash
cd umi-diffusion-training
python train.py --config-name=train_diffusion_unet_timm_umi_workspace
python eval_offline.py -i /path/to/checkpoint.ckpt -d /path/to/dataset
```

### 2. 部署侧

在 `umi-deploy` 中完成：

- ARX5 的 CAN/USB-CAN 配置
- Gen 夹爪串口与 ROS 节点配置
- 相机设备确认
- 策略推理服务启动
- 主控脚本执行

真实部署通常涉及这些模块：

- `umi-arx/scripts/eval_arx5.py`
- `detached-umi-policy`
- `arx5-sdk`
- `gen_controller_sdk_release`

## 注意事项

- 真实机器人部署前，必须先完成设备别名、CAN 接口、串口权限和相机枚举确认。
- 离线评估结果不等价于真实执行成功率，部署前仍需要实际联调。
- 建议将每次训练使用的数据集、checkpoint 和部署参数记录在实验日志中，便于复现。
- `setup.txt` 里只保留了一个最小化的 CAN 启动示例，不应替代正式部署文档。

## 一句话总结

这个仓库的核心价值，是把 UMI 风格的 Diffusion Policy 从训练端一直打通到 ARX5 真机执行端，适合作为 ARX5 视觉策略训练与部署的一体化工程基础。
