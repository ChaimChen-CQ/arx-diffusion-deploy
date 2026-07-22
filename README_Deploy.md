# ARX5 Diffusion Policy Deploy

## 一、机械臂通信以及环境配置 (https://github.com/LYtingN/arx-difussion-deploy/blob/main/umi-deploy/umi-arx/README_Deploy.md)(https://github.com/LYtingN/arx-difussion-deploy/blob/main/umi-deploy/arx5-sdk/README.md)

**环境依赖**

我们为所有 cmake 依赖设置了一个 conda 环境，因此不需要系统包。如果你想运行并修改 C++ 源文件后，请确保你处于创建的 conda 环境（等等）。`cmakemakearx-py310`

我们推荐[Mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)来创建conda环境，这大约只需1分钟。你也可以用 ，但时间要长得多（~10分钟）。`conda`

```bash
mamba env create -f conda_environments/py310_environment.yaml
# if you do not have mamba, you can also use conda, which takes significantly longer
# Currently available python versions: 3.8, 3.9, 3.10, 3.11
conda activate arx-py310 # 但是好像是arx-nyx-py310
mkdir build && cd build
cmake ..
make -j
# At this point, you should be able to run test scripts below.
```

```bash
# To install the C++ package your system, run:
make install
```

**EtherCAT-CAN 设置**

用USB线给EtherCAT-CAN适配器供电，用以太网线连接到电脑。在终端运行后，你应该能找到接口名称，通常是（现有的以太网口）或（额外的 USB-Ethernet 适配器）。`ip aeth.en..........`

然后你应该启用Python解释器的以太网访问（通常在你的文件夹里）。注意，这通常会给你一个符号链接（比如），但在这种情况下不起作用。你需要查到实际的档案（通常是）。`binwhich python~/miniforge3/envs/arx-py310/bin/pythonpython3.x`

```bash
mamba activate arx-py310 # 但是好像是arx-nyx-py310
ls -l $(which python)
sudo setcap "cap_net_admin,cap_net_raw=eip" your/path/to/conda/envs/arx-py310/bin/python3.10
```

要运行 C++，编译后每次都需要启用该可执行文件。你还需要用正确的接口更新C++脚本。

`sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_cartesian_controller
sudo setcap "cap_net_admin,cap_net_raw=eip" build/test_joint_controller`

安装基础工具：

```bash
sudo apt install -y can-utils net-tools
```

如需使用 SpaceMouse 遥操作，再安装：

```bash
sudo apt install -y libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
```

---

**识别 USB-CAN 设备**

插入 USB-CAN 适配器后，检查系统是否识别到新的串口设备：

```bash
ls /dev/ttyACM*
ls /dev/ttyUSB*
```

示例输出：

```bash
/dev/ttyACM9
```

如果出现新的 `/dev/ttyACMx` 设备，通常说明该适配器走的是 **SLCAN 路线**，应通过 `slcand` 创建 CAN 接口，而不是直接使用 `ip link set canX up ...` 的原生 CAN 方式。

---

**配置固定设备别名（udev）**

为了避免设备每次重新插拔后 `/dev/ttyACMx` 编号变化，建议为 USB-CAN 配置固定别名。

**1. 查看设备属性**

```bash
udevadm info -a -n /dev/ttyACM9 | egrep 'idVendor|idProduct|serial'
```

记录输出中的：

- `idVendor`
- `idProduct`
- `serial`

**2. 编写 udev 规则**

新建规则文件：

```bash
sudo vim /etc/udev/rules.d/arx_can.rules
```

示例内容如下：

```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="YOUR_SERIAL", SYMLINK+="arxcan1"
```

说明：

- `YOUR_SERIAL` 替换为设备实际序列号
- 规则生效后，系统会为该设备创建固定别名 `/dev/arxcan1`

**3. 重载规则**

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**4. 验证别名是否生效**

```bash
readlink -f /dev/arxcan1
```

若输出类似：

```bash
/dev/ttyACM9
```

说明别名配置成功。

---

**创建 CAN 接口**

使用 `slcand` 将串口型设备桥接为 Linux CAN 接口：

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
```

参数说明：

- `/dev/arxcan1`：udev 生成的固定串口别名
- `can1`：创建出的 Linux CAN 接口名
- `s8`：对应 **1 Mbps** 波特率

> 建议保持设备别名和接口命名一致，例如 `/dev/arxcan1 -> can1`，便于排查与维护。
> 

---

**验证接口状态**

执行：

```bash
ip -details link show can1
```

正常情况下应看到类似状态：

```
can1: <NOARP,UP,LOWER_UP> ...
can state ERROR-ACTIVE
```

关键字段说明：

- `UP`：接口已启用
- `LOWER_UP`：底层链路已连通
- `ERROR-ACTIVE`：CAN 控制器处于正常工作状态

> 对于 `slcand/slcan` 类型设备，`ip -details` 中出现 `bitrate 0` 并不一定表示失败。 这通常是因为内核无法从 `slcan` 设备回读实际 bit rate 信息，而不是配置未生效。
> 

---

**已验证配置**

本次实际验证通过的配置如下：

```bash
/dev/arxcan1 -> /dev/ttyACM9
CAN interface: can1
```

对应命令：

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1
```

接口状态为：

- `UP`
- `LOWER_UP`
- `ERROR-ACTIVE`

说明底层链路已经打通，可以继续进行 `arx5-sdk` 层验证。

---

**在 arx5-sdk 中使用**

完成 CAN 接口配置后，后续所有 `arx5-sdk` 命令都应使用 `can1`。

示例：

**1. 基础关节控制测试**

`python examples/test_joint_control.py L5 can1`

## 二、夹爪通信以及环境配置 (https://github.com/LYtingN/arx-difussion-deploy/blob/main/umi-deploy/readmedeploy.md)(https://github.com/LYtingN/arx-difussion-deploy/blob/main/umi-deploy/gen_controller_sdk_release/README_CN.md)

#### 配置docker环境

```bash
cd umi-deploy/gen_controller_sdk_release

docker build -t tron1-rl-deploy:noetic .
```

**两条规则**

1. 所有 `docker run` / `docker exec` 命令都必须在宿主机执行，不是在容器里执行。
2. 所有 ROS 终端都必须进入同一个容器 `tron_gripper`，不要重复 `docker run` 新建临时容器。

**1. 宿主机启动唯一容器**

```bash
docker rm -f tron_gripper 2>/dev/null

docker run -dit \
  --name tron_gripper \
  --privileged \
  --network host \
  -v /dev:/dev \
  -v /home/yd/program/nyx/arx-difussion-deploy:/workspace/arx-difussion-deploy \
  tron1-rl-deploy:noetic /bin/bash
```

检查容器：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

预期至少看到：

```
tron_gripper   Up ...
```

**2. 宿主机初始化容器内依赖**

只做一次：

```bash
docker exec -it tron_gripper bash -lc '
cd /workspace/arx-difussion-deploy/umi-deploy/gen_controller_sdk_release &&
source /opt/ros/noetic/setup.bash &&
python3 -m pip install pyserial &&
pip3 install -r requirements.txt &&
catkin_make &&
source devel/setup.bash &&
python3 -c "import serial; print(serial.__file__)" &&
ls /dev/ttyUSB*
'
```

预期结果：

- `import serial` 成功
- 能看到 `/dev/ttyUSB0`

如果不是 `/dev/ttyUSB0`，后面所有命令里的串口都替换成真实设备名。

### 验证

**Terminal 1：启动 roscore**

```bash
docker exec -it tron_gripper bash -lc '
source /opt/ros/noetic/setup.bash &&
roscore
'
```

**Terminal 2：启动夹爪串口节点**

```bash
docker exec -it tron_gripper bash -lc '
cd /workspace/arx-difussion-deploy/umi-deploy/gen_controller_sdk_release &&
source /opt/ros/noetic/setup.bash &&
source devel/setup.bash &&
rosrun robot_driver databus_single.py _serial_port:=/dev/ttyUSB0 _topic_encoder:=/encoder _topic_target_distance:=/target_distance
'
```

预期结果：

- 进程持续运行，不应立刻退出
- 不应再出现 `ModuleNotFoundError: No module named serial`

**Terminal 3：查看编码器反馈**

```bash
docker exec -it tron_gripper bash -lc '
cd /workspace/arx-difussion-deploy/umi-deploy/gen_controller_sdk_release &&
source /opt/ros/noetic/setup.bash &&
source devel/setup.bash &&
rostopic echo /encoder
'
```

预期结果：

- 连续输出浮点数
- 当前机器已验证类似：

```
data: 0.0503
```

**Terminal 4：发送一次开合指令**

```bash
docker exec -it tron_gripper bash -lc '
cd /workspace/arx-difussion-deploy/umi-deploy/gen_controller_sdk_release &&
source /opt/ros/noetic/setup.bash &&
source devel/setup.bash &&
rostopic pub -1 /target_distance std_msgs/Float32 "data: 0.05"
'
```

预期结果：

- 终端显示 `publishing and latching message for 3.0 seconds`
- 夹爪实际动作
- Terminal 3 中 `/encoder` 数值变化

## 三、部署流程

T1：打开roscore

```bash
docker start tron_gripper
docker exec -it tron_gripper bash -lc '
source /opt/ros/noetic/setup.bash &&
roscore
'
```

T2：夹爪通信

```bash
  docker exec -it tron_gripper bash -lc '
  cd /workspace/arx-difussion-deploy/umi-deploy/gen_controller_sdk_release &&
  source /opt/ros/noetic/setup.bash &&
  source devel/setup.bash &&
  rosrun robot_driver databus_single.py _serial_port:=/dev/ttyDeviceLeft _topic_encoder:=/encoder _topic_target_distance:=/target_distance
  '
```

T3：机械臂通信

```bash
  conda activate arx-nyx-py310

  sudo pkill slcand
  sudo slcand -o -f -s8 /dev/arxcan1 can1
  sudo ip link set can1 up
  ip -details link show can1

  cd /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/arx5-sdk
  python python/communication/zmq_server.py L5 can1
```

T4：启动推理节点

环境配置：`umi` 里有这类推理通常会依赖的包：

- `torch 2.1.0`
- `torchvision 0.16.0`
- `diffusers 0.18.2`
- `accelerate 0.24.1`
- `hydra-core 1.2.0`
- `omegaconf 2.2.3`
- `einops 0.6.1`
- `opencv-python 4.7.0`
- `av 10.0.0`
- `zarr 2.16.1`

```bash
unset PYTHONPATH
  conda activate umi
  cd /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy

  python detached_policy_inference.py \
      -i /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy/data/models/epoch=0190-train_loss=0.014.ckpt \
      -c /home/yd/program/nyx/arx-difussion-deploy/umi-diffusion-training/diffusion_policy/config/train_diffusion_transformer_umi_workspace.yaml
```

T5：启动主控

环境配置：`umi-arx` 是“机器人运行环境”

- ROS 相关包：`rclpy`、`rospy`、`rosgraph`、`cv-bridge`、`catkin`、`catkin-pkg`
- 设备交互相关：`pyzmq`、`evdev`、`spnav`
- 视觉与数值：`opencv-python 4.13.0`、`numpy 1.24.4`
- 推理：`torch 2.11.0`
- 消息/工具链：大量 `std-msgs`、`sensor-msgs`、`tf2-*` 包

```bash
unset PYTHONPATH
  source /home/yd/anaconda3/etc/profile.d/conda.sh
  conda activate umi-arx
  pkill -f eval_arx5
  cd /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/umi-arx

  python scripts/eval_arx5.py \
    -i /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/detached-umi-policy/data/models/epoch=0190-train_loss=0.014.ckpt \
    -c /home/yd/program/nyx/arx-difussion-deploy/umi-diffusion-training/diffusion_policy/config/train_diffusion_transformer_umi_workspace.yaml \
    -o data/experiments/$(date +%Y%m%d_%H%M%S) \
    --runtime_calibration config/arx5_runtime_calibration.yaml \
    --camera_reorder 0 \
    --no_mirror \
    --no_spacemouse
```

等出来相机画面后

-按“i”：机械臂到达initial position （15°）

-按“c”：开始跑policy

-按“s”：暂停推理

-按“q”：推出主控，机械臂回原位



### 6. USB 故障监控（推荐）
在复现相机/串口掉线问题前，先在宿主机启动 USB 监控脚本：

```bash
cd /home/yd/program/nyx/arx-difussion-deploy/umi-deploy/umi-arx
sudo python3 scripts/monitor_usb_host.py \
    --output-dir data_local/usb_monitor/$(date +%Y%m%d_%H%M%S)
```
