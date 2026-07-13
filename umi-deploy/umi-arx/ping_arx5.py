import sys
import os
import time

# 添加当前目录到 sys.path，以便导入 modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.arx5_zmq_client import Arx5Client
except ImportError as e:
    print(f"Import error: {e}. Please ensure you run this script from the 'umi-arx' directory.")
    sys.exit(1)

def main():
    print("Initiating ARX5 Client connection to ZMQ Server (127.0.0.1:8765)...")
    try:
        # readmedeploy.md mentions port 8765
        client = Arx5Client(zmq_ip="127.0.0.1", zmq_port=8765)
        print("Connected successfully!")
        
        print("\n--- Current ARX5 Status ---")
        print(f"Timestamp:   {client.timestamp}")
        print(f"Joint Pos:   {client.joint_pos}")
        print(f"EE Pose:     {client.ee_pose}")
        print(f"Gripper Pos: {client.gripper_pos}")
        print("---------------------------\n")
        print("Ping test PASSED: Data received from arm successfully.")
        
    except Exception as e:
        print(f"Ping failed: {e}")

if __name__ == "__main__":
    main()
