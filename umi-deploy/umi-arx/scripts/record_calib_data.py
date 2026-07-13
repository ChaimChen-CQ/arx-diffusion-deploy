import sys
import os
import json
import cv2
import numpy as np
import time
import scipy.spatial.transform as st

# --- 强行加入系统路径 ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

SDK_PYTHON_DIR = os.path.join(os.path.dirname(ROOT_DIR), "arx5-sdk", "python")
sys.path.append(SDK_PYTHON_DIR)
os.chdir(ROOT_DIR)
# ------------------------

from communication.zmq_client import Arx5Client
from peripherals.keystroke_counter import KeystrokeCounter, KeyCode

# 屏蔽烦人的 OpenCV Qt 字体警告
os.environ["QT_QPA_PLATFORM"] = "xcb" 

GRIPPER_FISHEYE_INTRINSICS = os.path.abspath(
    os.path.join(
        ROOT_DIR,
        "..",
        "data_local",
        "calibration",
        "cam0_sensor_intrinsics.json",
    )
)


def get_gripper_camera_resolution():
    with open(GRIPPER_FISHEYE_INTRINSICS, "r", encoding="utf-8") as file:
        intrinsics = json.load(file)
    if intrinsics.get("intrinsic_type") != "FISHEYE":
        raise ValueError(
            f"Expected FISHEYE intrinsics in {GRIPPER_FISHEYE_INTRINSICS}, "
            f"got {intrinsics.get('intrinsic_type')}"
        )
    return int(intrinsics["image_width"]), int(intrinsics["image_height"])

def main():
    output_dir = "calib_workspace"
    images_dir = os.path.join(output_dir, "images")
    poses_file = os.path.join(output_dir, "poses.txt")
    os.makedirs(images_dir, exist_ok=True)

    print("1. 正在初始化摄像头...")
    # 恢复默认 V4L2 读取，不强制加底层限制，防止死锁
    cap = cv2.VideoCapture(0)
    
    # 仅保留 MJPG 压缩和分辨率设定
    expected_width, expected_height = get_gripper_camera_resolution()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_height)

    # 预热摄像头
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    if not cap.isOpened():
        print("❌ 摄像头打开失败！")
        return
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise RuntimeError(
            f"固定内参要求 {expected_width}x{expected_height}，"
            f"但相机实际输出 {actual_width}x{actual_height}"
        )
    print(f"✅ 摄像头就绪！resolution={actual_width}x{actual_height}")

    print("2. 正在连接机械臂并解锁...")
    try:
        arm_client = Arx5Client(zmq_ip="127.0.0.1", zmq_port=8765)
        arm_client.set_to_damping()
        gain = arm_client.get_gain()
        gain['kp'] = gain['kp'] * 0.0
        gain['kd'] = gain['kd'] * 0.1
        arm_client.set_gain(gain)
        print("✅ 机械臂已解锁！可以自由拖拽。")
    except Exception as e:
        print(f"❌ 机械臂连接失败: {e}")
        return

    with open(poses_file, "w") as f:
        f.write("# x, y, z, rx, ry, rz (m, rad)\n")
    pose_f = open(poses_file, "a")
    count = 1

    print("\n=== RealManRobot 手眼标定数据采集程序 ===")
    print("操作指南：")
    print("  - 's' 键: 保存当前图像与末端位姿")
    print("  - 'q' 键: 结束采集并退出")
    print("⚠️ 忽略终端可能弹出的 QFontDatabase 警告，不影响使用！")

    with KeystrokeCounter() as key_counter:
        while True:
            # 【核心修改】：软件层面清空缓冲区。连续抓取几帧丢弃，只取最新的一帧。
            # 这样既能防止画面延迟，又能避免底层 USB 带宽堆积断流。
            for _ in range(4):
                cap.grab()
            ret, frame = cap.retrieve()
            
            if not ret:
                print("⚠️ 画面获取失败，重试中...")
                time.sleep(0.05)
                continue

            disp_frame = cv2.resize(frame, (800, 648))
            cv2.imshow("Calibration View (S: Save, Q: Quit)", disp_frame)
            cv2.waitKey(1)

            press_events = key_counter.get_press_events()
            exit_flag = False

            for ks in press_events:
                if ks == KeyCode(char="q") or ks == KeyCode(char="Q"):
                    print(f"\n采集结束，共保存 {count - 1} 组数据。")
                    exit_flag = True
                    break
                    
                elif ks == KeyCode(char="s") or ks == KeyCode(char="S"):
                    arm_client.get_state()
                    current_tcp_pose = arm_client.tcp_pose 
                    
                    pos = current_tcp_pose[:3]
                    rotvec = current_tcp_pose[3:6]
                    euler = st.Rotation.from_rotvec(rotvec).as_euler('xyz', degrees=False)
                    rm_pose = np.concatenate((pos, euler))
                    
                    pose_line = f"{rm_pose[0]:.6f}, {rm_pose[1]:.6f}, {rm_pose[2]:.6f}, {rm_pose[3]:.6f}, {rm_pose[4]:.6f}, {rm_pose[5]:.6f}\n"
                    
                    pose_f.write(pose_line)
                    pose_f.flush()
                    
                    img_path = os.path.join(images_dir, f"{count}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    print(f"✅ Saved frame {count}.jpg, pose: [{pose_line.strip()}]")
                    count += 1

            if exit_flag:
                break

    pose_f.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
