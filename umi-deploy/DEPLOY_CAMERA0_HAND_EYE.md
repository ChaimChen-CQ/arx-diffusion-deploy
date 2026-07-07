# ARX + GenRobot camera0 光心策略部署命令（Python 夹爪通信版）

这份说明对应当前训练方式：

```text
GenRobot MCAP -> UMI dataset
robot0_eef_pos 实际表示 camera0 光学中心 pose
diffusion policy 输出 camera0 光心目标轨迹
ARX 底层仍执行 TCP/EEF 目标
```

已在 `umi-arx/scripts/eval_arx5.py` 中加入手眼转换：

```text
当前 ARX TCP pose  ->  当前 camera0 pose  ->  policy
policy 输出 camera0 target  ->  ARX TCP target  ->  robot
```

夹爪通信也已改成纯 Python：

```text
policy 输出 gripper_width
  -> umi-arx/modules/arx5_controller.py
  -> gen_con_sdk_python_release/scripts/databus.py
  -> GenRobot/DAS 夹爪串口
```

所以现在不需要 `roscore`，也不需要 `rostopic /target_distance`、`/encoder`。

第一次使用前，在你部署用的 Python/conda 环境里装一下 Gen SDK 依赖：

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/gen_con_sdk_python_release
pip install -r requirements.txt
```

默认读取：

```bash
/home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/hand_eye_result.json
```

并默认按 `eef_to_camera` 理解，也就是：

```text
T_tcp_camera0
```

如果发现执行方向明显反了，先不要硬试，把主控脚本参数改成：

```bash
--hand_eye_direction camera_to_eef
```

---

## Terminal 1：启动 ARX5 ZMQ Server

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy

sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1

cd arx5-sdk
python python/communication/zmq_server.py L5_umi can1
```

确认点：

```text
ZMQ server 默认端口：8765
模型名按实际机械臂选择：L5 / L5_umi / X5 / X5_umi
```

---

## Terminal 2：可选，检查 GenRobot 夹爪 Python 串口

这一步不是部署必需项，只是先单独确认夹爪串口能打开、编码器能读、目标宽度能写。

不要在真正部署时一直占着这个脚本；测试完退出。主控部署脚本会自己打开同一个串口。

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/gen_con_sdk_python_release

python - <<'PY'
import struct
import time
from scripts.databus import DataBus

def encoder_cb(record_data: bytes):
    value = struct.unpack(">f", record_data)[0]
    print(f"encoder: {value:.4f} m")

bus = DataBus(
    tty_port="/dev/ttyDeviceLeft",
    baudrate=921600,
    encoder_freq=30,
    encoder_callback=encoder_cb,
)

try:
    print("open to 0.050 m")
    bus.set_target_distance(0.050)
    time.sleep(3)
    print("close to 0.020 m")
    bus.set_target_distance(0.020)
    time.sleep(3)
finally:
    bus.stop()
PY
```

---

## Terminal 3：启动 policy inference server

把 `CKPT` 换成你训练得到的 checkpoint。

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/detached-umi-policy

python detached_policy_inference.py \
  -i CKPT \
  --ip 0.0.0.0 \
  --port 8766 \
  --device cuda
```

如果 policy server 和主控脚本在同一台机器，主控里 `--policy_ip localhost` 即可。

---

## Terminal 4：启动 ARX 主控部署脚本

把 `CKPT`、`OUTPUT_DIR`、`CAMERA_INDEX` 换成实际值。

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/umi-arx

python scripts/eval_arx5.py \
  -i CKPT \
  -o OUTPUT_DIR \
  --policy_ip localhost \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 12 \
  --camera_reorder CAMERA_INDEX \
  --hand_eye_path /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/hand_eye_result.json \
  --hand_eye_direction eef_to_camera \
  --gen_gripper_sdk_path /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/gen_con_sdk_python_release \
  --gripper_serial_port /dev/ttyDeviceLeft \
  --gripper_encoder_frequency 30
```

如果 policy server 在另一台 GPU 机器：

```bash
--policy_ip GPU机器IP
```

---

## 怎么查 CAMERA_INDEX

```bash
cd /home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/umi-arx

python3 -c "
from utils.usb_util import get_sorted_v4l_paths
for i, p in enumerate(get_sorted_v4l_paths()):
    print(i, p)
"
```

把 GenRobot 中间相机 camera0 对应的 index 填给：

```bash
--camera_reorder CAMERA_INDEX
```

---

## 运行中按键

```text
C：把控制权交给 policy
S：停止 policy，回到人工控制
Q：退出
SpaceMouse：人工移动 TCP
SpaceMouse 左右键同时按：回 home
SpaceMouse 左键：打开夹爪
SpaceMouse 右键：关闭夹爪
```

安全提醒：

```text
第一次按 C 前，速度调低，手放急停。
如果运动方向明显不对，立刻停，尝试把 --hand_eye_direction 换成 camera_to_eef。
```

---

## 这次代码转换的位置

手眼 / camera0 光心转换在：

```bash
/home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/umi-arx/scripts/eval_arx5.py
```

新增逻辑：

```text
load_hand_eye_transform()
convert_obs_tcp_to_camera()
convert_camera_action_to_tcp()
```

作用：

```text
obs 阶段：
  T_base_tcp_current @ T_tcp_camera0 = T_base_camera0_current

action 阶段：
  T_base_camera0_target @ inverse(T_tcp_camera0) = T_base_tcp_target
```

夹爪 ROS -> Python 串口转换在：

```bash
/home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/umi-arx/modules/arx5_controller.py
/home/phi5090ii/CZY/arx-difussion-deploy/umi-deploy/umi-arx/modules/arx5_env.py
```

作用：

```text
原来：
  Arx5Controller -> rospy publish /target_distance
  rospy subscribe /encoder

现在：
  Arx5Controller -> DataBus.set_target_distance()
  DataBus encoder_callback -> gripper_position
```
