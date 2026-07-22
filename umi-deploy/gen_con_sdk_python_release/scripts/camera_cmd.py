#!/usr/bin/env python3
"""
Camera Command Tool - Pure Python implementation
Replaces camera_cmd.sh for sending camera calibration commands
"""

import sys
import time
import os

# Supports running as script: python camera_cmd.py or python -m ...scripts.camera_cmd
if __name__ == "__main__" or not __package__:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_script_dir)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from scripts.databus import DataBus, find_serial_port
    from scripts.pack import CmdPack
else:
    from .databus import DataBus, find_serial_port
    from .pack import CmdPack


def camera_calib_callback(camera_pack):
    """Callback when camera calibration payload is received."""
    print("Camera calibration data received")


def main():
    """CLI entry point."""
    SIDE_CONFIG = {
        'single': {'serial_port': ''},   # Single-device: no default port; use env or auto-find
        'left': {'serial_port': "/dev/ttyDeviceLeft"},
        'right': {'serial_port': "/dev/ttyDeviceRight"},
    }

    usage = """
Usage:
  Single-device mode (default when left/right omitted):
    python -m gen_controller_python_sdk.camera_cmd {1234|camerarc|camerarl|camerarr|MCUID|DMZEROSET}
  Dual-device mode (left or right):
    python -m gen_controller_python_sdk.camera_cmd {left|right} {1234|camerarc|camerarl|camerarr|MCUID|DMZEROSET}

  Optional env: SERIAL_PORT=/dev/ttyUSB0 (overrides left/right default port)

  Arguments:
    left/right - Optional gripper side (omit for single-device mode)
    1234       - Confirm calibration complete
    camerarc   - Calibrate center camera (writes cam0_sensor_{single|left|right}.yaml)
    camerarl   - Calibrate left camera (writes cam1_sensor_{single|left|right}.yaml)
    camerarr   - Calibrate right camera (writes cam2_sensor_{single|left|right}.yaml)
    MCUID      - Query device MCUID
    DMZEROSET  - Set DM zero offset
    """
    
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)
    
    if len(sys.argv) == 2:
        side = 'single'
        record_value = sys.argv[1]
    else:
        side = sys.argv[1].lower()
        if side not in SIDE_CONFIG:
            print(f"Error: first argument must be left or right, got '{sys.argv[1]}'")
            print(usage)
            sys.exit(1)
        record_value = sys.argv[2]
    
    valid_commands = ['1234', 'camerarc', 'camerarl', 'camerarr', 'MCUID', 'DMZEROSET']
    if record_value not in valid_commands:
        print(f"Error: argument must be one of {valid_commands}")
        print(usage)
        sys.exit(1)
    
    yaml_filename = ""
    if record_value == "camerarc":
        yaml_filename = f"cam0_sensor_{side}.yaml"
    elif record_value == "camerarl":
        yaml_filename = f"cam1_sensor_{side}.yaml"
    elif record_value == "camerarr":
        yaml_filename = f"cam2_sensor_{side}.yaml"
    
    if yaml_filename:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, "calib_result")
        
        os.environ['CALIB_YAML_FILENAME'] = yaml_filename
        print(f"Will write YAML: {yaml_filename}")
        print(f"Save path: {os.path.join(result_dir, yaml_filename)}")
    else:
        if 'CALIB_YAML_FILENAME' in os.environ:
            del os.environ['CALIB_YAML_FILENAME']
    
    serial_port = os.environ.get('SERIAL_PORT', '')
    if not serial_port:
        serial_port = SIDE_CONFIG[side]['serial_port']
    if not serial_port:
        serial_port = find_serial_port("ttyUSB")
    
    if not serial_port:
        print(" No configured serial port found")
        sys.exit(1)
    
    print(f"Side: {side}")
    print(f"Serial port: {serial_port}")
    print(f"Sending camera calibration command: {record_value}")
    
    try:
        bus = DataBus(
            tty_port=serial_port,
            baudrate=921600,
            is_calib_cmd=True,
            calib_cmd_name=record_value,
            camera_calib_callback=camera_calib_callback,
        )
        time.sleep(1.0)
        bus.add_cmd(CmdPack.pack_calib(record=record_value.encode("utf-8")))

        if record_value in ("MCUID", "DMZEROSET"):
            bus.wait_for_calib_response(timeout=3.0)
        elif record_value.startswith("camera"):
            bus.wait_for_calib_response(timeout=2.0)
        else:
            time.sleep(0.5)

        bus.stop()

        if record_value == "1234":
            print("Calibration OK !")
        elif record_value == "MCUID":
            print("MCUID query executed")
        elif record_value == "DMZEROSET":
            print("DMZEROSET command executed")
        else:
            print(f"Finished sending command: {record_value}")
            if yaml_filename:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                result_dir = os.path.join(script_dir, "calib_result")
                yaml_path = os.path.join(result_dir, yaml_filename)
                if os.path.exists(yaml_path):
                    print(f"YAML file created: {yaml_path}")
                else:
                    print(f" YAML file not found; check device response")
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
