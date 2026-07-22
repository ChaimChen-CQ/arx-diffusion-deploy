<!-- can connection -->
# 1) Plug in the USB-CAN adapter, then confirm the actual device name.
ls /dev/ttyACM* /dev/ttyUSB* /dev/arxcan1 2>/dev/null

# 2) If /dev/arxcan1 exists, use the stable alias. Otherwise replace
#    /dev/arxcan1 below with the real /dev/ttyACMx device from step 1.
sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1


conda activate arx-py311
cd ~/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/arx5-sdk

export LD_LIBRARY_PATH=$PWD/lib/x86_64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python python/communication/zmq_server.py L5_umi can1


cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy

python detached_policy_inference.py \
  -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/data/latest.ckpt \
  -c /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/data/latest.yaml \
  --ip 0.0.0.0 \
  --port 8766 \
  --device cuda

(umi-arx) chaim@chaim:~/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/umi-arx$ python scripts/eval_arx5.py   -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy/data/models/epoch=0190-train_loss=0.014.ckpt   -o data_local/eval_latest_debug_gripperfix   --policy_ip 127.0.0.1   --policy_port 8766   --frequency 8   --steps_per_inference 8   --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml   --camera_reorder 1  --disable_video_recording  --command_latency 0.25

GEN_GRIPPER_SERIAL_PORT=/dev/ttyUSB0 python scripts/eval_arx5.py \
  -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy/data/models/epoch=0190-train_loss=0.014.ckpt \
  -o data_local/eval_latest_debug_gripperfix \
  --policy_ip 127.0.0.1 \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 8 \
  --runtime_calibration config/arx5_runtime_calibration_policy_identity_base.yaml \
  --camera_reorder 1 \
  --disable_video_recording \
  --command_latency 0.25

  GEN_GRIPPER_SERIAL_PORT=/dev/ttyUSB0 python scripts/eval_arx5.py \
  -i /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/data/latest.ckpt \
  -c /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/data/latest.yaml \
  -o data_local/eval_latest_debug_gripperfix \
  --policy_ip 127.0.0.1 \
  --policy_port 8766 \
  --frequency 8 \
  --steps_per_inference 8 \
  --runtime_calibration config/arx5_runtime_calibration_policy_action.yaml \
  --camera_reorder 1 \
  --disable_video_recording \
  --command_latency 0.2 \
  --log_runtime_transforms
