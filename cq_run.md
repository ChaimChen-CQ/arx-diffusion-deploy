# CQ 运行命令

项目路径：`/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy`

## 终端 1：CAN 与机械臂服务

```bash
ls /dev/ttyACM* /dev/ttyUSB* /dev/arxcan1 2>/dev/null

sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1

conda activate arx-py310
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/arx5-sdk
export LD_LIBRARY_PATH="$PWD/lib/x86_64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python python/communication/zmq_server.py L5_umi can1
```

> 如果没有 `/dev/arxcan1`，将命令中的设备名换成实际的 `/dev/ttyACMx`。

## 终端 2：Policy 推理服务

```bash
unset PYTHONPATH
conda activate umi-cu128-py311
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy

python detached_policy_inference.py \
  -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  --ip 0.0.0.0 \
  --port 8766 \
  --device cuda
```

## 终端 3：运行机器人

```bash
unset PYTHONPATH
conda activate umi-arx
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/umi-arx

GEN_GRIPPER_SERIAL_PORT=/dev/ttyUSB0 python scripts/eval_arx5.py \
  -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  -o data_local/eval_unet_20260714 \
  --policy_ip 127.0.0.1 \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 8 \
  --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml \
  --camera_reorder 1 \
  --disable_video_recording \
  --command_latency 0.2 \
  --log_runtime_transforms \
  --action_z_bias -0.015
```

