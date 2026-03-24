# ARX5 + Gen夹爪 Diffusion Policy 部署指南

## 硬件组成

- **ARX5 机械臂**：通过 USB-CAN (SLCAN) → CAN 通信
- **Gen 末端夹爪**：通过 USB 串口通信，自带中/左/右三个摄像头
- **Policy 推理机**：带 GPU 的机器（可以和控制机器相同或不同）

---

## 整体架构

```
[Gen夹爪 databus ROS节点]   ← USB串口 →  [Gen夹爪硬件]
  发布 /encoder (Float32)                  含3个摄像头
  订阅 /target_distance (Float32)

[MultiUvcCamera]  ← V4L2 直接读 →  [Gen夹爪中间摄像头 /dev/videoN]

[arx5-sdk ZMQ Server]  ← CAN →  [ARX5机械臂]
  tcp://127.0.0.1:8765

[Arx5Env + Arx5Controller]   (umi-arx/scripts/eval_arx5.py)
  ↑ 摄像头图像 (MultiUvcCamera)
  ↑ 机械臂状态 (ZMQ)
  ↑ 夹爪位置 (/encoder)
  ↓ 机械臂指令 (ZMQ)
  ↓ 夹爪指令 (/target_distance)

[detached_policy_inference.py]  ← ZMQ obs/action →  (GPU机器, port 8766)
  加载 .ckpt checkpoint
  DDIM 16步推理
```

**注意**：摄像头不经过 ROS，由 MultiUvcCamera 直接读 V4L2，只有夹爪串口控制走 ROS。

---

## 代码改动（已完成）

### 1. `umi-arx/modules/arx5_env.py` — 摄像头分辨率

Gen夹爪摄像头原生分辨率为 1600x1296，原代码硬编码 1920x1080 已修正：

```python
# 第115行
res = (1600, 1296)  # gen gripper center camera native resolution
fps = 30
```

> 如果 `v4l2-ctl -d /dev/videoN --list-formats-ext` 显示不支持 1600x1296，改成实际支持的最高分辨率。

### 2. `umi-arx/modules/arx5_controller.py` — Gen夹爪 ROS 控制

`run()` 方法里已加入：
- 启动 ROS 节点，发布 `/target_distance`，订阅 `/encoder`
- `set_tcp_pose` 不再传 gripper 命令给 arx5 SDK（传 0.0）
- 从 `/encoder` 读取实际夹爪开口距离作为状态反馈

---

## 训练配置说明（`train_diffusion_unet_timm_umi_workspace.yaml`）

| 参数 | 值 |
|------|----|
| 摄像头 | `camera0_rgb` 224x224，horizon=2 |
| 机器人状态 | `eef_pos`(3D) + `eef_rot_axis_angle`(6D rotation_6d) + `gripper_width`(1D) |
| Action shape | `[10]` = pos(3) + rot_6d(6) + gripper(1) |
| Action horizon | 16 |
| obs_pose_repr | `relative` |
| action_pose_repr | `relative` |
| 视觉编码器 | `vit_base_patch16_clip_224.openai` |
| obs_down_sample_steps | 3 |

---

## 启动前准备（只需配置一次）

### 1. USB-CAN udev 规则

```bash
udevadm info -a -n /dev/ttyACM? | egrep 'idVendor|idProduct|serial'
# 记录 idVendor, idProduct, serial，写入规则：
sudo vim /etc/udev/rules.d/arx_can.rules
# 内容：
# SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="YOUR_SERIAL", SYMLINK+="arxcan1"
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2. Gen 夹爪 USB 规则

```bash
cd gen_controller_sdk_release
# 按 README.md 配置 config/99-usb-serial.rules，然后：
sudo cp config/99-usb-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 成功后 /dev/ttyDeviceLeft 应指向夹爪串口
```

### 3. 确认摄像头 V4L2 设备编号

```bash
# 查看所有摄像头设备
v4l2-ctl --list-devices

# 确认 gen 夹爪中间摄像头支持的分辨率
v4l2-ctl -d /dev/videoN --list-formats-ext

# 查看 get_sorted_v4l_paths() 排序后的 index
cd umi-arx
python3 -c "
from utils.usb_util import get_sorted_v4l_paths
for i, p in enumerate(get_sorted_v4l_paths()):
    print(i, p)
"
# 记下 gen 夹爪中间摄像头对应的 index，填到 --camera_reorder
```

### 4. 编译 gen_controller_sdk

```bash
cd gen_controller_sdk_release
catkin_make
source devel/setup.bash
```

---

## 完整启动流程（每次部署执行）

### Terminal 1 — ARX5 机械臂 ZMQ Server

```bash
sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1   # 确认 UP + ERROR-ACTIVE

cd arx5-sdk
python python/communication/zmq_server.py L5_umi can1
# 模型名根据实际机型：X5 / L5 / X5_umi / L5_umi
# 默认端口 8765
```

### Terminal 2 — Gen 夹爪 ROS 节点

```bash
roscore &

cd gen_controller_sdk_release
source devel/setup.bash

# 只启动串口节点，不启动 camera_view_single（避免摄像头设备冲突）
rosrun robot_driver databus_single.py \
    _serial_port:=/dev/ttyDeviceLeft \
    _topic_encoder:=encoder \
    _topic_target_distance:=target_distance
```

验证：
```bash
rostopic echo /encoder           # 应有 0.0 左右的值持续输出
rostopic pub /target_distance std_msgs/Float32 "data: 0.05"  # 测试夹爪运动
```

### Terminal 3 — Policy Inference Server（GPU 机器）

```bash
cd detached-umi-policy

# 首次运行会自动导出 .yaml 配置文件到 checkpoint 同目录
python detached_policy_inference.py \
    -i /path/to/your/checkpoint.ckpt \
    --ip 0.0.0.0 \
    --port 8766 \
    --device cuda
```

### Terminal 4 — 主控脚本

```bash
cd umi-arx

python scripts/eval_arx5.py \
    -i /path/to/your/checkpoint.ckpt \
    -o /path/to/output_dir \
    --policy_ip <GPU机器IP> \
    --policy_port 8766 \
    --frequency 8 \
    --steps_per_inference 12 \
    --camera_reorder N     # N = gen夹爪中间摄像头的 index
```

---

## 操作说明（运行中）

| 按键 | 功能 |
|------|------|
| `C` | 将控制权交给 policy（开始推理执行） |
| `S` | 停止 policy，恢复人工控制 |
| `Q` | 退出程序 |
| SpaceMouse 移动 | 控制末端 XY 平移 |
| SpaceMouse 右键 | 解锁 Z 轴 |
| SpaceMouse 左键 | 解锁旋转轴 |
| SpaceMouse 左右键同时 | 机械臂回 home 位 |
| SpaceMouse 单按左/右 | 夹爪关/开 |

> **安全提示**：按下 `C` 前确保手边有急停按钮，policy 控制期间机械臂会自主运动。

---

## 数据流（Policy 控制阶段）

```
Gen夹爪中间摄像头 (1600x1296) -> resize 224x224 -> camera0_rgb
ARX5 TCP Pose (6D) + gripper_width (1D)  x  obs_horizon=2
          ↓  get_real_umi_obs_dict()  (pose_repr=relative)
obs_dict_np ──ZMQ──> detached_policy_inference.py (port 8766)
                              ↓ DDIM 16步推理
          raw_action: [steps=12, 10]  <──ZMQ──
          ↓  get_real_umi_action()  (action_repr=relative)
action: [steps=12, 7]  =  [x,y,z, rx,ry,rz, gripper]
          ↓  Arx5Env.exec_actions()  dynamic_latency=True
Arx5Controller.add_waypoint() -> 轨迹插值 @ 200Hz
          ├─ arx5-sdk ZMQ -> CAN -> ARX5机械臂
          └─ ROS /target_distance -> USB串口 -> Gen夹爪
```

---

## ROS Topic 说明

| Topic | 类型 | 方向 | 范围 | 说明 |
|-------|------|------|------|------|
| `/target_distance` | `std_msgs/Float32` | 写入 | [0.0, 0.103] m | 夹爪目标开口距离 |
| `/encoder` | `std_msgs/Float32` | 读取 | [0.0, 0.103] m | 夹爪实际开口距离反馈 |
| `/tactile/left` | `Int8MultiArray` | 读取 | — | 左侧触觉（部署不使用） |
| `/tactile/right` | `Int8MultiArray` | 读取 | — | 右侧触觉（部署不使用） |

---

## 常见问题

### CAN 接口不通
```bash
sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
# 若 /dev/arxcan1 不存在，先检查 udev 规则或直接用 /dev/ttyACMx
```

### 摄像头打开失败 / 设备冲突
```bash
sudo chmod 666 /dev/videoN
# 如果 camera_view_single.py 在运行，先 kill 掉
rosnode kill /camera
```

### Policy 报错 shape mismatch
- 确认 `--camera_reorder` 选到的是 gen 夹爪中间摄像头
- 确认 checkpoint 对应的 `.yaml` 已生成（首次运行 `detached_policy_inference.py` 自动生成）
- 检查 `arx5_env.py` 里摄像头分辨率是否与实际一致

### 夹爪无响应
```bash
rostopic echo /encoder         # 确认有数据输出
ls -la /dev/ttyDeviceLeft      # 确认设备映射正确
sudo chmod 666 /dev/ttyDeviceLeft
```

### arx5-sdk 模型名报错
支持的模型名：`X5`、`L5`、`X5_umi`、`L5_umi`，根据实际机型填写，**不要填错否则可能产生危险动作**。

---

## 文件结构速查

```
arx-difussion-deploy/
├── arx5-sdk/
│   └── python/communication/zmq_server.py        # ARX5 ZMQ服务端，需先启动
├── umi-arx/
│   ├── scripts/eval_arx5.py                      # 主控脚本，部署入口
│   ├── modules/arx5_env.py                       # 环境封装（摄像头+机械臂）
│   ├── modules/arx5_controller.py                # ARX5控制器 + Gen夹爪ROS接口
│   └── modules/arx5_zmq_client.py                # ZMQ客户端
├── detached-umi-policy/
│   └── detached_policy_inference.py              # Policy推理服务端
├── gen_controller_sdk_release/
│   ├── src/robot_driver/scripts/databus_single.py   # 夹爪串口ROS节点
│   └── src/robot_driver/launch/single_gripper_start.launch
├── umi-diffusion-training/
│   └── diffusion_policy/config/
│       ├── train_diffusion_unet_timm_umi_workspace.yaml  # 训练配置
│       └── task/umi.yaml                         # shape_meta 定义
└── readmedeploy.md                               # 本文件
```
