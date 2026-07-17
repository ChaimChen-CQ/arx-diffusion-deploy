# 工控机与 5090 通信

工控机 `192.168.31.142` 运行机械臂服务和主程序；5090 `192.168.31.109:8766` 运行推理。

## 终端 1：5090 推理

```bash
ssh phi5090ii@192.168.31.109
unset PYTHONPATH
conda activate umi-cu128-py311
cd /home/phi5090ii/CQ/arx-diffusion-deploy/umi-deploy/detached-umi-policy

python detached_policy_inference.py \
  -i /home/phi5090ii/CQ/arx-diffusion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/phi5090ii/CQ/arx-diffusion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  --ip 0.0.0.0 --port 8766 --device cuda:0
```

## 终端 2：工控机机械臂服务

```bash
ssh zyd@192.168.31.142
sudo modprobe slcan
sudo pkill slcand 2>/dev/null || true
sudo slcand -o -c -f -s8 /dev/ttyACM0 can1
sudo ip link set can1 up

conda activate arx-py310
cd /home/zyd/CQ/arx-difussion-deploy/umi-deploy/arx5-sdk
export LD_LIBRARY_PATH="$PWD/lib/aarch64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python python/communication/zmq_server.py L5_umi can1
```

## 终端 3：工控机主程序

```bash
ssh -Y -C zyd@192.168.31.142
conda activate umi-arx
unset PYTHONPATH
export PYNPUT_BACKEND=xorg
cd /home/zyd/CQ/arx-difussion-deploy/umi-deploy/umi-arx

nc -vz -w 3 192.168.31.109 8766

GEN_GRIPPER_SERIAL_PORT=/dev/ttyUSB0 python scripts/eval_arx5.py \
  -i /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.ckpt \
  -c /home/zyd/CQ/arx-difussion-deploy/trained_ckpt/tron1_gripper_20260714/latest.yaml \
  -o data_local/eval_remote_5090_dryrun \
  --policy_ip 192.168.31.109 --policy_port 8766 \
  --frequency 8 --steps_per_inference 8 \
  --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml \
  --camera_reorder 1 --disable_video_recording \
  --command_latency 0.2 --log_runtime_transforms \
  --dry_run_policy --no_spacemouse
```

按 `c` 开始推理。正式控制时删除 `--dry_run_policy`。
