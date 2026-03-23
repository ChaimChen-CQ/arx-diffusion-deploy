
# UMI-ARX 中 ARX5 USB-CAN（SLCAN）配置说明

本文档介绍如何在 Linux 下通过 **USB-CAN 适配器 + SLCAN** 方式为 ARX5 机械臂建立 CAN 通信链路，并给出已验证可用的配置与常见问题排查方法。

---

## 目录

- [概述](#概述)
- [环境依赖](#环境依赖)
- [识别 USB-CAN 设备](#识别-usb-can-设备)
- [配置固定设备别名（udev）](#配置固定设备别名udev)
- [创建 CAN 接口](#创建-can-接口)
- [验证接口状态](#验证接口状态)
- [已验证配置](#已验证配置)
- [在 arx5-sdk 中使用](#在-arx5-sdk-中使用)
- [推荐启动流程](#推荐启动流程)
- [常见问题排查](#常见问题排查)
- [总结](#总结)

---

## 概述

本配置流程对应的通信链路如下：

```text
USB-CAN adapter -> /dev/ttyACMx -> slcand -> canX -> arx5-sdk
````

其中：

* `/dev/ttyACMx`：系统识别出的串口型 USB-CAN 设备
* `slcand`：将串口型 SLCAN 设备桥接为 Linux CAN 网络接口
* `canX`：Linux 中实际使用的 CAN 接口，例如 `can0`、`can1`

本次实际验证通过的链路为：

```text
/dev/arxcan1 -> /dev/ttyACM9 -> can1
```

因此，后续所有 `arx5-sdk` 命令均应使用 `can1` 作为 CAN 接口名。

---

## 环境依赖

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

## 识别 USB-CAN 设备

插入 USB-CAN 适配器后，检查系统是否识别到新的串口设备：

```bash
ls /dev/ttyACM*
ls /dev/ttyUSB*
```

示例输出：

```bash
/dev/ttyACM9
```

如果出现新的 `/dev/ttyACMx` 设备，通常说明该适配器走的是 **SLCAN 路线**，应通过 `slcand` 创建 CAN 接口，而不是直接使用 `ip link set canX up ...` 的原生 CAN 方式。

---

## 配置固定设备别名（udev）

为了避免设备每次重新插拔后 `/dev/ttyACMx` 编号变化，建议为 USB-CAN 配置固定别名。

### 1. 查看设备属性

```bash
udevadm info -a -n /dev/ttyACM9 | egrep 'idVendor|idProduct|serial'
```

记录输出中的：

* `idVendor`
* `idProduct`
* `serial`

### 2. 编写 udev 规则

新建规则文件：

```bash
sudo vim /etc/udev/rules.d/arx_can.rules
```

示例内容如下：

```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="YOUR_SERIAL", SYMLINK+="arxcan1"
```

说明：

* `YOUR_SERIAL` 替换为设备实际序列号
* 规则生效后，系统会为该设备创建固定别名 `/dev/arxcan1`

### 3. 重载规则

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 4. 验证别名是否生效

```bash
readlink -f /dev/arxcan1
```

若输出类似：

```bash
/dev/ttyACM9
```

说明别名配置成功。

---

## 创建 CAN 接口

使用 `slcand` 将串口型设备桥接为 Linux CAN 接口：

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
```

参数说明：

* `/dev/arxcan1`：udev 生成的固定串口别名
* `can1`：创建出的 Linux CAN 接口名
* `-s8`：对应 **1 Mbps** 波特率

> 建议保持设备别名和接口命名一致，例如 `/dev/arxcan1 -> can1`，便于排查与维护。

---

## 验证接口状态

执行：

```bash
ip -details link show can1
```

正常情况下应看到类似状态：

```text
can1: <NOARP,UP,LOWER_UP> ...
can state ERROR-ACTIVE
```

关键字段说明：

* `UP`：接口已启用
* `LOWER_UP`：底层链路已连通
* `ERROR-ACTIVE`：CAN 控制器处于正常工作状态

> 对于 `slcand/slcan` 类型设备，`ip -details` 中出现 `bitrate 0` 并不一定表示失败。
> 这通常是因为内核无法从 `slcan` 设备回读实际 bit rate 信息，而不是配置未生效。

---

## 已验证配置

本次实际验证通过的配置如下：

```text
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

* `UP`
* `LOWER_UP`
* `ERROR-ACTIVE`

说明底层链路已经打通，可以继续进行 `arx5-sdk` 层验证。

---

## 在 arx5-sdk 中使用

完成 CAN 接口配置后，后续所有 `arx5-sdk` 命令都应使用 `can1`。

示例：

### 1. 基础关节控制测试

```bash
python examples/test_joint_control.py L5 can1
```

### 2. 键盘遥操作

```bash
python examples/keyboard_teleop.py L5 can1
```

### 3. SpaceMouse 遥操作

```bash
python examples/spacemouse_teleop.py L5_umi can1
```

### 4. 启动 ZMQ 服务

```bash
python python/communication/zmq_server.py L5_umi can1
```

> 模型参数请根据仓库实际支持情况填写，例如 `X5`、`L5`、`X5_umi`、`L5_umi`。
> 请勿填错模型名，否则可能导致危险动作。

---

## 推荐启动流程

每次重新插拔 USB-CAN 适配器后，建议按如下顺序执行：

### 1. 确认设备别名

```bash
readlink -f /dev/arxcan1
```

### 2. 清理旧的 `slcand` 进程

```bash
sudo pkill slcand
```

### 3. 重新创建并启用 CAN 接口

```bash
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
```

### 4. 检查接口状态

```bash
ip -details link show can1
```

### 5. 运行 SDK 测试

```bash
python examples/test_joint_control.py X5 can1
```

确认基础控制正常后，再继续运行遥操作脚本或上层服务。

---

推荐的标准命令为：

```bash
sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1
```

