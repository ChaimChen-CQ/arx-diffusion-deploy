````markdown
# Gen Controller SDK 单夹爪部署 README（Ubuntu 22.04 + Docker）

## 1. 说明

原始 SDK 文档要求：

- Ubuntu 20.04
- ROS1
- USB 3.0

当前实际使用环境为：

- 宿主机：Ubuntu 22.04
- 容器：Ubuntu 20.04 + ROS Noetic
- 运行方式：Docker + 宿主机 udev 映射 + X11 图形转发

本 README 记录了单夹爪在当前机器上的完整部署流程、最终配置结果和已知问题。

---

## 2. 项目路径

真实项目路径为：

```bash
/home/phi5090ii/NYX/gen_controller_sdk_release
````

注意不要误用空目录：

```bash
/home/phi5090ii/genrobot_controller_sdk
```

之前如果把空目录挂进 Docker，会导致容器内目录为空。

---

## 3. Docker 安装

宿主机安装 Docker 后，先确认 Docker 正常运行。

### 3.1 验证 Docker

```bash
docker --version
docker compose version
docker run hello-world
```

如果 `hello-world` 能正常输出，说明 Docker 已安装成功。

---

## 5. 单夹爪 udev 规则配置

### 5.1 串口参数 1 获取方法

执行：

```bash
ls /dev/ttyUSB*
udevadm info -a -n /dev/ttyUSB0 | grep -E "KERNELS|DRIVERS"
```

根据原文档规则，取输出中的**第二个 `KERNELS` 值**作为参数 1。

本机最终值为：

```bash
1-3.4:1.0
```

---

### 5.2 相机参数 2 获取方法

执行：

```bash
v4l2-ctl --list-devices
```

本机相机设备为：

```text
TSTC USB20 WEB CAMERA: TSTC USB (usb-0000:0c:00.0-3.1):
    /dev/video0
    /dev/video1
    /dev/media0

TSTC USB20 WEB CAMERA: TSTC USB (usb-0000:0c:00.0-3.2):
    /dev/video2
    /dev/video3
    /dev/media1

TSTC USB20 WEB CAMERA: TSTC USB (usb-0000:0c:00.0-3.3):
    /dev/video4
    /dev/video5
    /dev/media2
```

因此三路相机对应为：

* 第一路相机：`1-3.1:1.0`
* 第二路相机：`1-3.2:1.0`
* 第三路相机：`1-3.3:1.0`

串口控制器对应为：

* 串口：`1-3.4:1.0`

---

### 5.3 最终单夹爪 `config/99-usb-serial.rules`

将 `config/99-usb-serial.rules` 修改为：

```udev
SUBSYSTEM=="tty", KERNELS=="1-3.4:1.0", SYMLINK+="ttyDeviceLeft", MODE="0666"
#SUBSYSTEM=="tty", KERNELS=="1-1.4:1.0", SYMLINK+="ttyDeviceRight", MODE="0666"

SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.1:1.0", ATTR{index}=="0", SYMLINK+="left_video_0_main", MODE="0666"
SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.1:1.0", ATTR{index}=="1", SYMLINK+="left_video_0_sec", MODE="0666"

SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.2:1.0", ATTR{index}=="0", SYMLINK+="left_video_1_main", MODE="0666"
SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.2:1.0", ATTR{index}=="1", SYMLINK+="left_video_1_sec", MODE="0666"

SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.3:1.0", ATTR{index}=="0", SYMLINK+="left_video_2_main", MODE="0666"
SUBSYSTEM=="video4linux", KERNEL=="video[0-9]*", KERNELS=="1-3.3:1.0", ATTR{index}=="1", SYMLINK+="left_video_2_sec", MODE="0666"
```

---

### 5.4 安装 udev 规则

在宿主机执行：

```bash
cd /home/phi5090ii/NYX/gen_controller_sdk_release
sudo cp config/99-usb-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

然后重新插拔夹爪 USB 设备。

---

### 5.5 检查 udev 映射结果

执行：

```bash
ls -l /dev/ttyDeviceLeft
ls -l /dev/left_video_*
```

本机最终结果为：

```bash
/dev/ttyDeviceLeft -> ttyUSB0
/dev/left_video_0_main -> video0
/dev/left_video_0_sec  -> video1
/dev/left_video_1_main -> video2
/dev/left_video_1_sec  -> video3
/dev/left_video_2_main -> video4
/dev/left_video_2_sec  -> video5
```

说明单夹爪串口和三路相机别名都已稳定生成。

---

## 6. Docker 镜像构建

项目使用的镜像名为：

```bash
genrobot-ros1:noetic-focal
```

如果需要重建镜像，在项目根目录执行：

```bash
cd /home/phi5090ii/NYX/gen_controller_sdk_release
docker build -t genrobot-ros1:noetic-focal .
```

---

## 7. 启动支持图形界面的 Docker 容器

### 7.1 宿主机先开放 X11

在宿主机执行：

```bash
echo $DISPLAY
xhost +local:root
```

本机 `DISPLAY` 为：

```bash
:1
```

---

### 7.2 启动容器

```bash
docker run -it --rm \
  --name genrobot_sdk \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -v /home/phi5090ii/NYX/gen_controller_sdk_release:/workspace/gen_controller_sdk_release \
  genrobot-ros1:noetic-focal
```

说明：

* `--net=host`：方便 ROS1 通信
* `--privileged`：简化硬件访问权限问题
* `-e DISPLAY=$DISPLAY`：允许容器访问宿主机图形显示
* `-v /tmp/.X11-unix:/tmp/.X11-unix:rw`：映射 X11 socket
* `-v /dev:/dev`：映射宿主机设备
* `-v /run/udev:/run/udev:ro`：映射 udev 信息
* `-v /home/phi5090ii/NYX/gen_controller_sdk_release:/workspace/gen_controller_sdk_release`：挂载真实项目目录

---

## 8. 容器内依赖安装

进入容器后，在项目目录执行：

```bash
cd /workspace/gen_controller_sdk_release
pip3 install -r requirements.txt
python3 -m pip install pyserial
```

`pyserial` 必须安装，否则会报：

```text
ModuleNotFoundError: No module named 'serial'
```

如果有 GTK 提示，可安装：

```bash
apt-get update
apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
```

这类 `Gtk-Message: Failed to load module "canberra-gtk-module"` 一般不是主故障，但装上可以减少警告。

---

## 9. ROS 运行方式

推荐使用 3 个终端，全部进入**同一个正在运行的容器**。
不要重复 `docker run` 新建同名容器；多终端进入同一个容器应使用：

```bash
docker exec -it genrobot_sdk bash
```

---

### 9.1 终端 1：启动 roscore

```bash
source /opt/ros/noetic/setup.bash
roscore
```

---

### 9.2 终端 2：启动主驱动

```bash
docker exec -it genrobot_sdk bash
cd /workspace/gen_controller_sdk_release
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch robot_driver single_gripper_start.launch
```

---

### 9.3 终端 3：启动控制脚本

当前实际跑通的是左夹爪命名空间脚本：

```bash
docker exec -it genrobot_sdk bash
cd /workspace/gen_controller_sdk_release/src/robot_driver/scripts
source /opt/ros/noetic/setup.bash
source /workspace/gen_controller_sdk_release/devel/setup.bash
python3 left_das_controller_infer.py
```

启动后会看到类似输出：

```text
[INFO] Gripper Data Converter Node Started
[INFO] Publish rate: 100Hz
```

这表示该节点已经常驻运行，而不是执行完退出。

---

## 10. 当前 ROS 话题状态

当前已经看到：

```bash
/gripper/left/current_distance
/left_gripper/encoder
/left_gripper/target_distance
/rosout
/rosout_agg
/target_gripper/left_gripper
```

说明：

* ROS master 已正常运行
* 控制转换节点已正常启动
* 当前控制链路走的是**左夹爪命名空间**
* 当前更适合配合 `left_das_controller_infer.py` 使用

---

## 11. 当前已完成内容

目前已经确认完成：

1. Docker 安装与正常运行
2. Ubuntu 22.04 宿主机上通过 Docker 跑 Ubuntu 20.04 + ROS Noetic
3. CH340 串口识别
4. 单夹爪串口 udev 固定映射
5. 三路相机 udev 固定映射
6. 容器图形显示基本打通
7. `pyserial` 缺失问题已解决
8. ROS topic 已能正常出现
9. 左夹爪控制脚本可启动并保持运行

---

## 12. 当前未完全解决的问题

### 12.1 databus / 协议解析问题

当前仍存在串口回传数据解析错误，例如：

```text
struct.error: unpack requires a buffer of 4 bytes
```

具体位置在：

```python
encoder_value = struct.unpack(">f", record_data)[0]
```

这说明：

* 串口已经成功打开
* 夹爪控制链路已经建立
* 但 `encoder_callback()` 收到的 `record_data` 长度不正确
* 上游数据包切分/解包存在错帧或协议不匹配问题

这不是 USB 映射问题，而是更上层的 databus / pack 协议解析问题。

---

### 12.2 camera 节点与 GUI

图像窗口相关问题已经基本从“无法显示”推进到“可尝试显示”，当前更主要的问题已转移到数据解析线程。

---

## 13. 常用命令

### 查看串口别名

```bash
ls -l /dev/ttyDeviceLeft
ls -l /dev/ttyUSB*
```

### 查看相机别名

```bash
ls -l /dev/left_video_*
```

### 查看相机设备

```bash
v4l2-ctl --list-devices
```

### 查看串口属性

```bash
udevadm info -a -n /dev/ttyUSB0 | grep -E "KERNELS|DRIVERS"
```

### 进入已有容器

```bash
docker exec -it genrobot_sdk bash
```

### 查看 ROS topic

```bash
rostopic list
rostopic echo /left_gripper/target_distance
rostopic echo /left_gripper/encoder
```

---

## 14. 当前建议

建议后续继续分两条线排查：

### 14.1 保持设备链路稳定

* 保持当前 udev 规则不再随意改动
* 保持 Docker 启动参数固定
* 先继续沿用已跑通的 `/left_gripper/...` 命名空间

### 14.2 单独修复 databus 解析

重点检查以下文件：

* `src/robot_driver/scripts/pack.py`
* `src/robot_driver/scripts/databus_single.py`

特别关注：

* record length
* 字节序（大端 / 小端）
* `encoder_callback()` 的输入长度保护
* 是否存在协议版本不一致

---

## 15. 当前结论

截至目前，**单夹爪设备映射、容器启动、图形转发、ROS 基本运行和左夹爪话题通信已经打通**。
真正剩余的核心问题不是部署，而是：

**设备状态回传数据在 SDK 中的协议解析仍不稳定，需要继续修复 `databus_single.py` / `pack.py`。**

```
```
