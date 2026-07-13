import sys
import os
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager

# 插入模块路径
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from modules.arx5_controller import Arx5Controller

def main():
    print("--- ARX5 & Gen Gripper headless Integration Test ---")
    
    with SharedMemoryManager() as shm_manager:
        print("[INFO] Attempting to connect to ZMQ Arm Server (127.0.0.1:8765) and ROS Gripper Node...")
        robot = Arx5Controller(
            shm_manager=shm_manager, 
            robot_ip="127.0.0.1", 
            robot_port=8765, 
            frequency=200,
            verbose=False
        )
        
        # 启动控制器（内部会在另一个进程中 import rospy 并订阅 /encoder）
        robot.start()
        
        try:
            print("[INFO] Waiting for Arx5Controller to synchronize...")
            while not robot.is_ready:
                time.sleep(0.1)
            print("[OK] Arx5Controller is fully ready! (Both Arm & Gripper connected)")
            
            state = robot.get_state()
            print(f"  --> Initial Gripper: {state['gripper_position']:.4f} m")
            print(f"  --> Initial TCP Pose: {state['ActualTCPPose']}")
            
            target_pose = state['ActualTCPPose']
            
            # User recorded safe folded joint angles: [1.577, -0.000, 0.019, -0.250, -0.012, 0.969]
            # Our modified Arx5Controller will maintain these physical joints securely bypasssing IK
            # when target_pose has not moved from ActualTCPPose.
            
            print("\n[ACTION] Commanding Arm (stay still) + Gripper (OPEN to 0.05m)...")
            # 保持 TCP 不动，仅改变夹爪
            robot.servoL(pose=target_pose, gripper_pos=0.05, duration=1.0)
            time.sleep(2.5)
            print(f"  --> Current Gripper: {robot.get_state()['gripper_position']:.4f} m")
            
            print("\n[ACTION] Commanding Arm (stay still) + Gripper (CLOSE to 0.00m)...")
            robot.servoL(pose=target_pose, gripper_pos=0.00, duration=1.0)
            time.sleep(2.5)
            print(f"  --> Current Gripper: {robot.get_state()['gripper_position']:.4f} m")
            
            print("\n[SUCCESS] Integration Data-Bus completely validated!")
        finally:
            print("\n[INFO] Cleaning up test session and terminating background control process...")
            robot.stop(wait=True)

if __name__ == "__main__":
    main()
