# Orin 与 RTX 5090 通信及实机运行

Orin `192.168.31.142` 负责相机采集、机械臂服务和控制主程序；RTX 5090
`192.168.31.109:8766` 负责 Policy 推理。

## 固定设备路径

不要直接依赖可能随拔插变化的 `/dev/ttyACM0`、`/dev/ttyACM1` 或
`/dev/ttyUSB0`，优先使用以下稳定路径：

```bash
# 机械臂 CANable2
/dev/serial/by-id/usb-Openlight_Labs_CANable2_b158aa7_github.com_normaldotcom_canable2.git_206932AD5052-if00

# Gen 夹爪 CH341 串口
/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

在 Orin 上确认设备：

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/* 2>/dev/null
```

如果 CH341 短暂出现后立刻消失，并且 `sudo dmesg` 中出现 `brltty sets
config`，在不使用盲文设备的前提下卸载 `brltty`，然后重新拔插夹爪 USB：

```bash
sudo apt purge brltty
```

## 终端 1：RTX 5090 推理服务

如果本机 Conda 的 OpenSSL 污染了系统 SSH，使用 `env -u LD_LIBRARY_PATH`：

```bash
env -u LD_LIBRARY_PATH /usr/bin/ssh phi5090ii@192.168.31.109

unset PYTHONPATH
conda activate umi-cu128-py311
cd /home/phi5090ii/CQ/arx-diffusion-deploy/umi-deploy/detached-umi-policy

python detached_policy_inference.py \
  -i /home/phi5090ii/CQ/arx-diffusion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/phi5090ii/CQ/arx-diffusion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  --ip 0.0.0.0 \
  --port 8766 \
  --device cuda:0
```

正常输出：

```text
PolicyInferenceNode is listening on 0.0.0.0:8766
```

## 终端 2：Orin 机械臂服务

```bash
env -u LD_LIBRARY_PATH /usr/bin/ssh zyd@192.168.31.142

sudo modprobe slcan
sudo pkill slcand 2>/dev/null || true
sudo ip link set can1 down 2>/dev/null || true
sudo slcand -o -c -f -s8 \
  /dev/serial/by-id/usb-Openlight_Labs_CANable2_b158aa7_github.com_normaldotcom_canable2.git_206932AD5052-if00 \
  can1
sudo ip link set can1 up
ip -details link show can1

conda activate arx-py310
cd /home/zyd/CQ/arx-difussion-deploy/umi-deploy/arx5-sdk
export LD_LIBRARY_PATH="$PWD/lib/aarch64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python python/communication/zmq_server.py L5_umi can1
```

## 终端 3：Orin 控制主程序

```bash
env -u LD_LIBRARY_PATH /usr/bin/ssh -Y -C zyd@192.168.31.142

conda activate umi-arx
unset PYTHONPATH
export PYNPUT_BACKEND=xorg
cd /home/zyd/CQ/arx-difussion-deploy/umi-deploy/umi-arx

nc -vz -w 3 192.168.31.109 8766
```

### 低延迟 Dry-run

只检查相机到 5090 推理再返回 Orin 的链路，不发送 Policy 动作：

```bash
GEN_GRIPPER_SERIAL_PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
python scripts/eval_arx5.py \
  -i /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  -o data_local/eval_remote_5090_dryrun \
  --policy_ip 192.168.31.109 \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 2 \
  --command_latency 0.1 \
  --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml \
  --camera_reorder 0 \
  --disable_video_recording \
  --no_log_runtime_transforms \
  --no_save_policy_io_debug \
  --dry_run_policy \
  --no_spacemouse
```

延迟日志：

```text
[LATENCY] obs_prep=... policy_rtt=... postprocess=... debug_save=0.000s total=...
```

### 首次实机安全测试

保留原有动态轨迹匹配，使用 8 步动作块和 0.2 秒调度余量，并在约 2 秒后
自动结束 Policy episode：

```bash
GEN_GRIPPER_SERIAL_PORT=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
python scripts/eval_arx5.py \
  -i /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  -o data_local/eval_remote_5090_safe_test \
  --policy_ip 192.168.31.109 \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 8 \
  --command_latency 0.2 \
  --max_duration 2 \
  --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml \
  --camera_reorder 0 \
  --disable_video_recording \
  --log_runtime_transforms \
  --no_save_policy_io_debug \
  --no_spacemouse
```

程序显示 `Human in control!` 后，按 `c` 开始 Policy。此命令没有
`--dry_run_policy`，机械臂会真实运动。达到 `--max_duration 2` 后会自动停止，
即使没有按 `s`，日志也会显示 `Max Duration reached`。

启动时必须确认夹爪连接成功：

```text
[Arx5Controller] Gen gripper Python SDK connected: /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
```

如果出现 `Gen gripper Python SDK disabled`，不要继续实机测试。

## 安全说明

- 第一次运行时让急停保持在手边，并清空机械臂工作空间。
- `--dry_run_policy` 只阻止 Policy 动作，不一定跳过机械臂初始化。
- 不要把 CANable 的 `/dev/ttyACM*` 路径当作夹爪串口。
- Ctrl+C 退出时控制器可能进入 damping，机械臂可能下坠；提前托住机械臂。
- 未验证前不要同时降低动作块、调度余量并关闭动态轨迹匹配。
