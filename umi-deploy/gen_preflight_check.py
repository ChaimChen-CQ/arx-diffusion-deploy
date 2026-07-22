import os
import sys

def check_ros1_env():
    # Check if roscore or catkin_make are available
    has_catkin = os.system("which catkin_make > /dev/null 2>&1") == 0
    has_roscore = os.system("which roscore > /dev/null 2>&1") == 0
    
    if not has_catkin or not has_roscore:
        print("[FAIL] ROS 1 environment is missing. 'catkin_make' or 'roscore' not found in PATH.")
        print("       (Gen Gripper SDK officially requires ROS 1, but we detected you only have ROS 2 Humble in standard paths).")
        print("       Action needed: We either need a ROS 1 Docker/Conda env, or use a gen-gripper ROS 2 port.")
        raise Exception("ROS 1 tools missing.")
    print("[OK] ROS 1 environment ('catkin_make', 'roscore') is available.")

def check_gripper_usb():
    tty_usb_found = os.popen(r"ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null").read().strip().split()
    # Filter out ttyACM0 since we know it's the arm
    tty_usb_found = [dev for dev in tty_usb_found if dev != '/dev/ttyACM0']
    
    if not tty_usb_found:
        print("[WARN] No additional /dev/ttyUSB* or /dev/ttyACM* devices found for the gripper. Please ensure Gen Gripper USB is plugged in.")
        raise Exception("Gripper USB device not detected.")
    else:
        print(f"[OK] Found candidate Gripper USB serial devices: {', '.join(tty_usb_found)}")

def main():
    print("--- Running Gen Gripper SDK Preflight Checks ---")
    try:
        check_ros1_env()
        check_gripper_usb()
        print("\nAll Checks Passed. Environment is ready for Gripper compilation.")
    except Exception as e:
        print(f"\n[FAIL] Preflight check failed:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
