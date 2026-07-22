# R5 + UMI 新 URDF 适配与实机验收指南

## 1. 目的与当前配置

本文用于在**不修改 `arx5-sdk` 控制代码**的前提下，验证新导入的 R5 + DAS 法兰 + UMI URDF 是否可以安全用于现有部署。

项目根目录：

```text
/home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy
```

原始导入模型：

```text
ARX_Model/R5_umi/urdf/R5_umi.urdf
```

`arx5-sdk` 实际加载模型：

```text
umi-deploy/arx5-sdk/models/L5_umi.urdf
```

当前启动参数仍叫 `L5_umi`，但该文件的机械结构已经替换为新导入的 R5 + UMI 模型。两个 URDF 的机械参数相同，SDK 版本只修改了机器人名称和 mesh 路径。

> 重要：机器人此前能够正常运动，说明 CAN、电机 ID、基本关节顺序和控制链路已经基本兼容。本指南重点验证新 URDF 的关节方向、TCP、FK/IK 和重力补偿。

## 2. 安全规则

每次实机测试都必须满足：

- 急停开关在手边，并确认有效。
- 机器人周围没有人员、线缆和障碍物。
- 第一次测试不要安装易碎或锋利物品。
- 先在 Home 附近、手臂未完全伸直的姿态测试。
- 一次只改变一个变量、一个关节或一个笛卡尔方向。
- 首次关节步长使用 `0.2°`，首次笛卡尔步长使用 `0.5 mm`。
- 出现方向错误、突跳、剧烈抖动、异响、持续大电流或发热时立即按 `Esc` 进入阻尼并停止测试。
- 不要在奇异位形、机械限位附近或手臂完全伸直时测试 IK。

启动 ZMQ server 可能初始化控制器或触发回零。因此，**启动服务本身也应按可能运动来准备**，不能当成绝对只读操作。

## 3. 第一步：固定当前可用基线

进入项目目录：

```bash
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy
```

查看当前修改：

```bash
git status --short
git -C umi-deploy/arx5-sdk status --short
```

保存一份带时间标记的 URDF 备份：

```bash
cp umi-deploy/arx5-sdk/models/L5_umi.urdf \
  umi-deploy/arx5-sdk/models/L5_umi.urdf.verified_candidate_20260722.bak
```

记录校验值：

```bash
sha256sum \
  ARX_Model/R5_umi/urdf/R5_umi.urdf \
  umi-deploy/arx5-sdk/models/L5_umi.urdf
```

通过标准：

- [ ] 备份文件存在。
- [ ] 当前能够正常运动的 URDF 已被单独保存。
- [ ] 没有覆盖或还原其他本地修改。

## 4. 第二步：离线检查 XML 和模型结构

### 4.1 检查 XML 是否能解析

```bash
python -c "import xml.etree.ElementTree as ET; ET.parse('umi-deploy/arx5-sdk/models/L5_umi.urdf'); print('URDF XML: OK')"
```

预期输出：

```text
URDF XML: OK
```

### 4.2 检查活动关节数量和顺序

```bash
python -c "import xml.etree.ElementTree as ET; r=ET.parse('umi-deploy/arx5-sdk/models/L5_umi.urdf').getroot(); print([(j.attrib['name'],j.attrib['type']) for j in r.findall('joint') if j.attrib.get('type')!='fixed'])"
```

预期只有六个活动关节：

```text
joint1, joint2, joint3, joint4, joint5, joint6
```

检查基座和末端：

```bash
rg -n '<link name="base_link"|<link name="eef_link"|<joint name="gripper_fixed_joint"' \
  umi-deploy/arx5-sdk/models/L5_umi.urdf
```

通过标准：

- [ ] XML 解析成功。
- [ ] 恰好有六个 revolute 活动关节。
- [ ] 顺序为 `joint1` 到 `joint6`。
- [ ] 同时存在 `base_link` 和 `eef_link`。
- [ ] UMI/DAS 通过固定结构安装在第六轴之后。

### 4.3 检查 mesh 文件

```bash
find umi-deploy/arx5-sdk/models/meshes/L5_umi -type f | sort
ls -l umi-deploy/arx5-sdk/models/meshes/DAS_Controller_v3_r5_end_adapter_flange.STL
```

注意 Linux 区分大小写，例如 `link1.STL` 与 `Link1.STL` 不是同一个文件。

mesh 主要用于显示和碰撞几何；FK/IK 主要使用 joint/link 关系，但缺失 mesh 仍可能让部分解析器或可视化工具失败。

## 5. 第三步：确认当前 TCP 定义

查看末端固定关节：

```bash
sed -n '959,996p' umi-deploy/arx5-sdk/models/L5_umi.urdf
```

当前 `eef_link` 相对 `link6` 的名义变换约为：

```text
xyz = [0.1039414, 0.0000388, 0.0767217] m
rpy = [0, 0.2618, 0] rad
```

即约为：

```text
X = 103.94 mm
Y = 0.04 mm
Z = 76.72 mm
绕 Y = 15°
```

在继续之前，明确策略使用的 TCP 是哪一个物理点：

- [ ] UMI 两指闭合中心。
- [ ] UMI 抓取工作点。
- [ ] 相机光心。
- [ ] UMI 安装基座中心。
- [ ] 其他：____________。

如果训练数据使用的是抓取中心，而 URDF 的 `eef_link` 是安装基座，那么机器人可以“运动正常”，但抓取会产生固定偏差。

此阶段先记录，不急着修改数值。

## 6. 第四步：启动 CAN 与机械臂服务

### 6.1 配置 CAN

终端 1：

```bash
ls /dev/ttyACM* /dev/ttyUSB* /dev/arxcan1 2>/dev/null

sudo pkill slcand
sudo slcand -o -f -s8 /dev/arxcan1 can1
sudo ip link set can1 up
ip -details link show can1
```

预期状态包含：

```text
UP
LOWER_UP
ERROR-ACTIVE
```

如果没有 `/dev/arxcan1`，应先查明真实设备路径，不能凭猜测替换。

### 6.2 启动 ZMQ server

确保机器人处于安全姿态、急停可触达，然后执行：

```bash
conda activate arx-py310
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/arx5-sdk
export LD_LIBRARY_PATH="$PWD/lib/x86_64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python python/communication/zmq_server.py L5_umi can1
```

检查日志：

- [ ] `Found root directory` 指向当前 `arx5-sdk`。
- [ ] URDF models directory 指向当前 `models` 目录。
- [ ] 初始化型号为 `L5_umi`。
- [ ] 接口为 `can1`。
- [ ] 没有电机离线、URDF 解析、IK solver 或维度错误。
- [ ] 启动过程中没有突然运动、剧烈抖动或异响。

如果日志加载了其他目录下的 `.so` 或模型，先修正 `PYTHONPATH/LD_LIBRARY_PATH`，不要继续测试。

## 7. 第五步：读取并记录当前状态

保持 server 运行，打开终端 2：

```bash
unset PYTHONPATH
conda activate umi-arx
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/umi-arx
python ping_arx5.py
```

记录输出：

```text
Joint Pos: ______________________________
EE Pose:   ______________________________
Gripper:   ______________________________
```

连续执行三次：

```bash
python ping_arx5.py
python ping_arx5.py
python ping_arx5.py
```

通过标准：

- [ ] 六个关节值都有限且合理，不包含 `nan/inf`。
- [ ] 静止时三次关节角没有明显跳变。
- [ ] 末端位置处于合理工作空间，不是数米或异常大数值。
- [ ] 静止时末端位姿没有明显跳变。

注意：`ping_arx5.py` 会连接已经运行的 server；不要在 server 未启动时反复尝试实机命令。

## 8. 第六步：小幅关节方向测试

项目已经提供带安全门的键盘点动工具。先运行默认 dry-run：

```bash
unset PYTHONPATH
conda activate umi-arx
cd /home/chaim/Desktop/umi-on-tron/nyx/arx-difussion-deploy/umi-deploy/umi-arx
python scripts/keyboard_jog_arx5.py \
  --mode joint \
  --joint_step_deg 0.2 \
  --max_joint_error_deg 0.5
```

dry-run 不应发送运动命令。熟悉按键：

```text
q/a: J1 +/-    w/s: J2 +/-    e/d: J3 +/-
r/f: J4 +/-    t/g: J5 +/-    y/h: J6 +/-
Esc: 阻尼并退出
```

确认 dry-run 正常后，才添加 `--run`：

```bash
python scripts/keyboard_jog_arx5.py \
  --mode joint \
  --run \
  --joint_step_deg 0.2 \
  --max_joint_error_deg 0.5 \
  --duration_sec 0.3 \
  --settle_sec 0.5
```

每个关节只执行一次正向和一次反向，记录：

| 关节 | 正向按键 | 实机方向正确 | SDK 数值同号变化 | 反向能回到附近 |
|---|---|---:|---:|---:|
| J1 | q / a | [ ] | [ ] | [ ] |
| J2 | w / s | [ ] | [ ] | [ ] |
| J3 | e / d | [ ] | [ ] | [ ] |
| J4 | r / f | [ ] | [ ] | [ ] |
| J5 | t / g | [ ] | [ ] | [ ] |
| J6 | y / h | [ ] | [ ] | [ ] |

停止条件：

- 任一关节方向与预期相反。
- 按一个关节但另一个关节明显运动。
- 运动量远大于 `0.2°`。
- 发生突跳、异响或碰撞趋势。

通过后可将步长提高到 `0.5°` 再重复一次，但此阶段不建议更大。

## 9. 第七步：小幅笛卡尔方向测试

先 dry-run：

```bash
python scripts/keyboard_jog_arx5.py \
  --mode cartesian \
  --cartesian_pose tcp \
  --cartesian_step_mm 0.5
```

按键：

```text
i/k: X +/-
j/l: Y +/-
u/o: Z +/-
Esc: 阻尼并退出
```

确认命令内容合理后执行：

```bash
python scripts/keyboard_jog_arx5.py \
  --mode cartesian \
  --run \
  --cartesian_pose tcp \
  --cartesian_step_mm 0.5 \
  --settle_sec 0.5
```

每个方向先正向一次，再反向一次。记录：

| 方向 | 实机方向正确 | 实际增量接近 0.5 mm | 返回后接近原位 |
|---|---:|---:|---:|
| X | [ ] | [ ] | [ ] |
| Y | [ ] | [ ] | [ ] |
| Z | [ ] | [ ] | [ ] |

通过标准：

- [ ] X/Y/Z 方向与机器人基坐标系定义一致。
- [ ] 没有明显耦合或反向运动。
- [ ] 连续小步运动没有 IK 跳解。
- [ ] 回到反方向后末端接近原位。

如果关节测试正确但笛卡尔方向错误，优先检查 URDF 的 joint axis、固定关节旋转和 `eef_link`，不要通过上层把方向强行取反。

## 10. 第八步：TCP 实测与修正判断

准备一个容易重复识别的物理参考点，例如桌面标记或标定板角点。不要让 UMI 与参考物发生挤压。

选择至少五个安全姿态：

1. Home 附近。
2. 工作区左侧。
3. 工作区右侧。
4. 较高位置。
5. 常用抓取高度。

每个姿态记录：

| 姿态 | SDK TCP xyz/rvec | 实测 TCP | 位置误差 | 角度误差 |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

判断方法：

- 所有姿态下误差近似固定：优先修正 `gripper_fixed_joint` 的 `xyz/rpy`。
- 误差随某个关节角规律变化：检查该关节的轴、origin 或零偏。
- 位置正确但方向固定偏差：检查 TCP 的 `rpy`。
- Home 正确但远离 Home 后误差增大：检查连杆尺寸、关节轴或零偏。

修改 TCP 时，每次只改一个小量，保存前后数值并重新启动 server。不要同时修改关节几何和 TCP，否则无法判断是哪项生效。

## 11. 第九步：重力补偿与动力学检查

当前 ZMQ server 会设置：

```python
controller_config.gravity_compensation = True
```

新 URDF 的 `link6` 当前包含约 `0.775 kg` 的聚合末端质量。必须确认它是否准确代表 DAS 法兰、UMI、相机、线材等实际负载。

在以下三种姿态各保持 20～30 秒：

1. Home 附近。
2. 手臂半伸展。
3. 常用抓取姿态。

记录：

| 姿态 | 下沉/上抬 | 抖动 | 异响 | 电机温升/电流表现 |
|---|---|---|---|---|
| Home |  |  |  |  |
| 半伸展 |  |  |  |  |
| 抓取姿态 |  |  |  |  |

停止条件：

- 明显持续下沉或自行上抬。
- 高频抖动或周期摆动。
- 肩部/肘部电流持续异常。
- 电机快速发热。

判断方法：

- 水平伸展时下沉：质量或力臂可能低估。
- 水平伸展时上抬：质量或力臂可能高估。
- 不同末端方向下表现差异明显：质心方向可能不准。
- 所有姿态都抖动：还需要检查控制增益和摩擦，不能只改惯量。

不要根据一次主观观察大幅修改质量。应小幅修改、重复同一姿态并保存对比记录。

## 12. 第十步：确认 SDK 外部安全限制

URDF 的 `<limit>` 不是当前控制器唯一的限制来源。`arx5-sdk` 的 `L5_umi` 配置仍在下面的文件中定义关节位置、速度和力矩限制：

```text
umi-deploy/arx5-sdk/include/app/config.h
```

本轮计划是不修改 SDK，因此至少需要做到：

- [ ] 从 R5 官方资料确认六轴机械限位。
- [ ] 当前测试姿态全部远离机械限位。
- [ ] 策略工作空间不会触及 R5 的机械限位。
- [ ] 未确认之前保持低速、低加速度运行。

如果 R5 限位比 L5 更窄，应先在上层限制工作空间；后续再单独评估是否修改 `config.h`。修改 `config.h` 属于 SDK 配置变更，需要重新编译，不属于仅修改 URDF。

## 13. 第十一步：完整策略前的最终验收

只有以下项目全部通过，才恢复策略推理：

- [ ] 新 URDF XML 和结构检查通过。
- [ ] server 确认加载当前 `L5_umi.urdf`。
- [ ] 六个关节方向、顺序和小步跟踪正确。
- [ ] X/Y/Z 小幅笛卡尔运动正确。
- [ ] TCP 与训练数据定义一致。
- [ ] 常用工作区没有 IK 跳解。
- [ ] Home、半伸展和抓取姿态的重力补偿稳定。
- [ ] 没有持续异常电流、发热、抖动或异响。
- [ ] GEN 夹爪独立工作正常。
- [ ] 当前 URDF、环境和启动命令已有备份记录。

第一次恢复策略时采用保守设置：

- 操作人员始终握住急停。
- 先不放置物体。
- 缩小策略工作空间。
- 降低运行频率或动作幅度。
- 只运行很短的一段轨迹。
- 检查日志后再逐步恢复正常参数。

## 14. 修改 URDF 后如何使其生效

如果只修改以下文件：

```text
models/L5_umi.urdf
models/meshes/...
```

不需要重新编译，执行以下操作即可：

1. 安全停止机器人控制。
2. 停止 ZMQ server。
3. 保存新 URDF 和修改记录。
4. 重新启动 ZMQ server。
5. 从日志确认模型路径。
6. 从小幅关节和笛卡尔测试重新验收。

如果修改以下文件，则需要重新编译：

```text
include/app/config.h
include/app/*.h
src/app/*.cpp
python/arx5_pybind.cpp
```

## 15. 建议保存的验收记录

建议为每一版 URDF 保存：

```text
日期：
URDF SHA256：
机器人编号：
CAN 接口：
UMI/DAS/相机实际安装配置：
末端总质量：
TCP xyz/rpy：
关节方向测试：通过 / 未通过
笛卡尔测试：通过 / 未通过
重力补偿测试：通过 / 未通过
策略短轨迹测试：通过 / 未通过
发现的问题：
对应日志或数据目录：
```

完成并保存以上记录后，这一版 URDF 才应被标记为“实机验证版本”。
