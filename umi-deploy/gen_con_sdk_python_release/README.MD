# Gen Gripper Controller Python SDK

> Pure Python SDK for Gen Controller gripper — camera streaming, tactile sensing, encoder feedback, and distance control. No ROS dependency.

[中文](README(ZH).MD)

[GitHub repository](https://github.com/genrobot-ai/gen_con_sdk_python_release)

License: [MIT License](LICENSE)

## 1 Features

- Pure Python SDK with no ROS dependency
- Three-camera image streaming with optional live preview
- Tactile sensor data via callbacks (left / right)
- Encoder feedback for gripper opening distance
- Gripper distance control via `DataBus.set_target_distance()`
- Single-gripper and dual-gripper support via `left` / `right` CLI argument
- Utility scripts for calibration, device ID, and tactile debugging

## 2 Requirements


| Item     | Requirement                   |
| -------- | ----------------------------- |
| OS       | Ubuntu 20.04 (recommended)    |
| Python   | 3.8+                          |
| USB      | USB 3.0 port                  |
| Hardware | Gen Controller gripper device |




## 3 Quick Start

> First-time users must complete [USB configuration](docs/usb-setup.md) before launching the driver.

```shell
git clone https://github.com/genrobot-ai/gen_con_sdk_python_release.git
cd gen_con_sdk_python_release
pip3 install -r requirements.txt
python3 start_gripper.py left
```

Three preview windows appear by default. Encoder distance and tactile data are printed to the terminal. Set a fixed opening distance:

```shell
python3 start_gripper.py left --distance 0.05
```



## 4 Program Interface



### 4.1 Sensor Callbacks

Pass these callbacks to `GripperSystem` (see [start_gripper.py](start_gripper.py) for examples):


| Callback                  | Description                          | Data format                                      |
| ------------------------- | ------------------------------------ | ------------------------------------------------ |
| `capture_frames_callback` | Camera frame capture loop            | Receives `CameraCapture` instance                |
| `tactile_callback`        | Tactile sensor data (left / right)   | `bytes` — 448-byte raw tactile record            |
| `encoder_callback`        | Gripper opening distance feedback    | `bytes` — big-endian float (`struct.unpack(">f")`) |


Preview windows map to three cameras:

| Window    | Camera        |
| --------- | ------------- |
| `camera_0` | Center camera |
| `camera_1` | Left camera   |
| `camera_2` | Right camera  |




### 4.2 Gripper Control

Send target opening distance through `DataBus`:

```python
if self.system.databus:
    self.system.databus.set_target_distance(value)  # range [0.0, 0.103] meters
```

Valid range: `[0.0, 0.103]` m (max ~10.3 cm).

For dual-gripper setups, run separate processes with `left` or `right` — each connects to its own serial port (`/dev/ttyDeviceLeft` or `/dev/ttyDeviceRight`).

## 5 Installation



### 5.1 Install system and Python dependencies

```shell
sudo apt update
sudo apt install python3-pip v4l-utils
pip3 install -r requirements.txt
```

`v4l-utils` provides `v4l2-ctl`, required for USB configuration.

### 5.2 Clone the repository

```shell
git clone https://github.com/genrobot-ai/gen_con_sdk_python_release.git
cd gen_con_sdk_python_release
pip3 install -r requirements.txt
```



## 6 USB Configuration

Configure udev rules once per USB port before first use. The template is at [config/99-usb-serial.rules](./config/99-usb-serial.rules).

Summary:

1. Query serial and camera `KERNELS` values with `udevadm` and `v4l2-ctl`
2. Edit `config/99-usb-serial.rules`
3. Copy to `/etc/udev/rules.d/` and reload rules

For step-by-step instructions with screenshots, see:

- [USB 配置指南 (ZH)](docs/usb-setup_CN.md)
- [USB Configuration Guide (EN)](docs/usb-setup.md)

Default serial symlinks after dual-gripper setup: `/dev/ttyDeviceLeft`, `/dev/ttyDeviceRight`.

## 7 Usage



### 7.1 Single Gripper

```shell
cd gen_con_sdk_python_release

python3 start_gripper.py left                              # Fixed open at 5 cm (default)
python3 start_gripper.py left --distance 0.08              # Fixed open at 8 cm
python3 start_gripper.py left --sine-wave                    # Open/close continuously for 10 s
python3 start_gripper.py left --camera-fps 60                # V4 Controller: set 60 fps for 30 fps output
python3 start_gripper.py left --no-preview                   # Disable preview windows
python3 start_gripper.py left --print-tactile-info           # Print tactile grid to terminal
```



### 7.2 Dual Gripper

```shell
cd gen_con_sdk_python_release

# Terminal 1 — left gripper
python3 start_gripper.py left

# Terminal 2 — right gripper
python3 start_gripper.py right

# Optional: print tactile grid for right gripper
python3 start_gripper.py right --print-tactile-info

# V4 Controller frame rate
python3 start_gripper.py left --camera-fps 60
```

Six preview windows appear (three per gripper). Do **not** run multiple device utility commands simultaneously.

### 7.3 Device Utilities

Use [scripts/camera_cmd.py](scripts/camera_cmd.py) for calibration and device ID queries. No ROS or separate daemon required.

**Single device:**

```shell
python3 scripts/camera_cmd.py camerarc   # Center camera calibration
python3 scripts/camera_cmd.py camerarl   # Left camera calibration
python3 scripts/camera_cmd.py camerarr   # Right camera calibration
python3 scripts/camera_cmd.py MCUID      # Device ID
```

**Dual device (left / right):**

```shell
python3 scripts/camera_cmd.py left camerarc
python3 scripts/camera_cmd.py left camerarl
python3 scripts/camera_cmd.py left camerarr
python3 scripts/camera_cmd.py left MCUID

python3 scripts/camera_cmd.py right camerarc
python3 scripts/camera_cmd.py right camerarl
python3 scripts/camera_cmd.py right camerarr
python3 scripts/camera_cmd.py right MCUID
```

Calibration YAML files are saved under `scripts/calib_result/`.



## 8 Troubleshooting


| Problem                         | Solution                                                                 |
| ------------------------------- | ------------------------------------------------------------------------ |
| Serial port not found           | Run `sudo apt remove brltty`, then replug the device                     |
| Camera or serial has wrong path | Re-check udev rules; see [docs/usb-setup.md](docs/usb-setup.md)          |
| V4 Controller frame rate halved | Pass `--camera-fps 60` to `start_gripper.py`                             |
| Import or module errors         | Run `pip3 install -r requirements.txt` in the repository root            |
| No sensor data                  | Confirm udev rules are loaded and `/dev/ttyDevice*` symlinks exist       |
| Permission denied on serial     | Reload udev rules or run `sudo chmod 666 /dev/ttyDeviceLeft` (or Right) |




## 9 Documentation


| Description           | Link                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| User Manual           | [DAS Controller User Manual](https://docs.genrobot.ai/en/products/das-controller)        |
| URDF                  | [URDF Model](https://zcnma1sv5kma.feishu.cn/wiki/CKpbwye45iOlrckukIPc5hKCndh)            |
| Troubleshooting Guide | [Problem Solving Guide](https://zcnma1sv5kma.feishu.cn/docx/E6VJdNqIfoGrSDxKIHKck7tLn3b) |
| USB 配置 (ZH)           | [docs/usb-setup_CN.md](docs/usb-setup_CN.md)                                             |
| USB setup (EN)        | [docs/usb-setup.md](docs/usb-setup.md)                                                   |
| udev rules template   | [config/99-usb-serial.rules](config/99-usb-serial.rules)                                 |
| Main entry script     | [start_gripper.py](start_gripper.py)                                                     |
| Device utility script | [scripts/camera_cmd.py](scripts/camera_cmd.py)                                           |
| SDK modules           | [scripts/](scripts/)                                                                     |

