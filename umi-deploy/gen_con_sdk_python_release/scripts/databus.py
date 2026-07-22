#!/usr/bin/env python3
"""
DataBus - Pure Python implementation for gripper communication
Removed ROS dependencies, using callbacks instead of ROS topics
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
import queue
import traceback
import struct
import os
import subprocess
from typing import Callable, Optional
from .pack import CmdPack, MessagePack, Opcode, RecordType
from .das_protocol import DASProtocol


# Default callbacks live in user scripts (gripper_controller.py, start_gripper.py, camera_cmd.py)
# so they stay editable even if databus.py is shipped obfuscated.


class DataBus:
    def __init__(
        self,
        tty_port="/dev/ttyUSB0",
        baudrate=921600,
        timeout=0.5,
        is_calib_cmd=False,
        calib_cmd_name: str = None,
        encoder_freq: float = None,
        tactile_freq: float = None,
        tactile_callback: Optional[Callable] = None,
        encoder_callback: Optional[Callable] = None,
        camera_calib_callback: Optional[Callable] = None,
    ):
        """
        Initialize DataBus.

        Args:
            tty_port: Serial device path.
            baudrate: Baud rate.
            timeout: Read timeout (seconds).
            is_calib_cmd: Calibration command mode.
            calib_cmd_name: Calibration command name (e.g. MCUID, DMZEROSET).
            encoder_freq: Encoder poll rate (Hz).
            tactile_freq: Tactile poll rate (Hz).
            tactile_callback: Tactile record handler.
            encoder_callback: Encoder record handler.
            camera_calib_callback: Camera calibration handler.
        """
        self.tty_port = tty_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.is_running = False

        self._open_serial_success = False
        self.protocol: DASProtocol = DASProtocol()
        self.data_buffer: bytes = b""
        self.data_buffer_lock = threading.Lock()
        self.serial_lock = threading.Lock()

        self.cmd_queue = queue.Queue(1000)

        self.read_thread: threading.Thread = None
        self.parse_thread: threading.Thread = None
        self.send_thread: threading.Thread = None

        self.encoder_freq = encoder_freq
        self.tactile_freq = tactile_freq
        self.encoder_thread: threading.Thread = None
        self.tactile_thread: threading.Thread = None
        
        self.gripper_dis = 0.0
        self.angle_lock = threading.Lock()
        self.is_calib_cmd = is_calib_cmd
        self.calib_cmd_name = calib_cmd_name
        if calib_cmd_name:
            os.environ["CALIB_CMD_NAME"] = calib_cmd_name
        
        self.tactile_callback = tactile_callback
        self.encoder_callback = encoder_callback
        self.camera_calib_callback = camera_calib_callback

        self._open_serial()
        if not self._open_serial_success:
            raise RuntimeError(f"Failed to open serial port: {tty_port}")
        
        self.is_running = True
        self._start_reading()
        self._start_parsing()
        self._start_sending()
        
        if self.encoder_freq:
            self._start_encoder_loop()
        if self.tactile_freq:
            self._start_tactile_loop()

    def set_target_distance(self, distance: float):
        """
        Set target gripper opening (encoder setpoint).

        Args:
            distance: Meters in [0.0, 0.103] (~10 cm max).
        """
        if distance < 0.0 or distance > 0.103:
            raise ValueError(f"Distance must be in [0.0, 0.103], got: {distance}")
        
        with self.angle_lock:
            self.gripper_dis = distance

    def get_target_distance(self) -> float:
        """Current target distance."""
        with self.angle_lock:
            return self.gripper_dis

    def drive_motor(self, angle_dgree: float):
        """Send drive command."""
        self.add_cmd(
            CmdPack.pack(
                opcode=Opcode.WriteDrive,
                record_type=RecordType.Drive,
                record=struct.pack(">f", angle_dgree),
            )
        )

    def disable_motor(self):
        """Disable motor drive."""
        self.add_cmd(
            CmdPack.pack(
                opcode=Opcode.DisableDrive,
                record_type=RecordType.Drive,
            )
        )
    
    def calib_encoder(self):
        """Request encoder calibration."""
        self.add_cmd(
            CmdPack.pack(
                opcode=Opcode.CalibEncoder,
                record_type=RecordType.Drive,
            )
        )

    def send_camera_calib_cmd(self, camera_cmd: str):
        """Enqueue camera calibration command string."""
        try:
            self.is_calib_cmd = True
            cmd = CmdPack.pack_calib(
                record=camera_cmd.encode('utf-8')
            )
            success = self.add_cmd(cmd)
            if success:
                print(f"Sent camera calibration command: {camera_cmd}")
            else:
                print(f"Failed to queue camera calibration: {camera_cmd}")
            return success
        except Exception as e:
            print(f"Error sending camera calibration command: {e}")
            return False

    def add_cmd(self, cmd: CmdPack) -> bool:
        """Push command to send queue."""
        try:
            self.cmd_queue.put(cmd, block=True, timeout=1)
            return True
        except queue.Full:
            print("Command queue full; drop")
            return False

    def is_opened(self):
        """True if serial opened successfully."""
        return self._open_serial_success

    def register_tactile_callback(self, callback: Callable):
        self.tactile_callback = callback

    def register_encoder_callback(self, callback: Callable):
        self.encoder_callback = callback

    def register_camera_calib_callback(self, callback: Callable):
        self.camera_calib_callback = callback

    def _open_serial(self):
        try:
            self.ser = serial.Serial(
                port=self.tty_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )

            if self.ser.is_open:
                print(f"Serial opened: {self.tty_port}, baudrate: {self.baudrate}")
                self._open_serial_success = True
            else:
                print(f"Serial open failed: {self.tty_port}")
                self._open_serial_success = False
        except Exception as e:
            print(f"Serial open error: {e}")
            self._open_serial_success = False

    def _start_reading(self):
        self.read_thread = threading.Thread(target=self._reading_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Read thread started")
        return True

    def _start_parsing(self):
        self.parse_thread = threading.Thread(target=self._parsing_loop)
        self.parse_thread.daemon = True
        self.parse_thread.start()
        print("Parse thread started")
        return True

    def _start_encoder_loop(self):
        self.encoder_thread = threading.Thread(target=self._send_encoder_loop)
        self.encoder_thread.daemon = True
        self.encoder_thread.start()
        print("Encoder loop thread started")
        return True

    def _start_tactile_loop(self):
        self.tactile_thread = threading.Thread(target=self._send_tactile_loop)
        self.tactile_thread.daemon = True
        self.tactile_thread.start()
        print("Tactile loop thread started")
        return True

    def _start_sending(self):
        self.send_thread = threading.Thread(target=self._sending_loop)
        self.send_thread.daemon = True
        self.send_thread.start()
        print("Send thread started")
        return True

    def _sending_loop(self):
        while self.is_running:
            try:
                cmd: CmdPack = self.cmd_queue.get(block=True, timeout=0.1)
                with self.serial_lock:
                    if self.ser and self.ser.is_open:
                        self.ser.write(cmd.data)
                        self.ser.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Send error: {e}")
                time.sleep(0.01)

    def _reading_loop(self):
        while self.is_running:
            try:
                with self.serial_lock:
                    if self.ser and self.ser.is_open:
                        n = self.ser.inWaiting()
                        if n:
                            read_size = min(n, 16384)
                            data = self.ser.read(read_size)
                            if data:
                                with self.data_buffer_lock:
                                    self.data_buffer = self.data_buffer + data
                                if n > read_size:
                                    continue

            except Exception as e:
                print(f"Read loop error: {e}")
                time.sleep(0.1)

            time.sleep(0.001)

    def _parsing_loop(self):
        while self.is_running:
            packets_to_process = []
            
            with self.data_buffer_lock:
                if len(self.data_buffer) > 0:
                    packets, remain = DASProtocol.find_packet(self.data_buffer)
                    self.data_buffer = remain
                    packets_to_process = packets.copy()
                    
            for packet in packets_to_process:
                try:
                    if self.is_calib_cmd:
                        magic = DASProtocol.MAGIC
                        is_camera_cmd = (
                            self.calib_cmd_name
                            and self.calib_cmd_name.startswith("camera")
                        )

                        if is_camera_cmd:
                            camera_pack = MessagePack.unpack_camera_calib(packet)
                            if camera_pack:
                                if self.camera_calib_callback:
                                    self.camera_calib_callback(camera_pack)
                                self.is_calib_cmd = False
                                continue

                        if (
                            len(packet) > 2 * len(magic)
                            and packet.startswith(magic)
                            and packet.endswith(magic)
                        ):
                            middle = packet[len(magic):-len(magic)]
                            try:
                                text = middle.decode("ascii")
                            except Exception:
                                text = middle.hex()
                            if self.calib_cmd_name == "MCUID":
                                print("MCUID:", text)
                            else:
                                print(f"Device response ({self.calib_cmd_name}): {text}")
                            self.is_calib_cmd = False
                            continue

                        if not is_camera_cmd:
                            camera_pack = MessagePack.unpack_camera_calib(packet)
                            if camera_pack:
                                if self.camera_calib_callback:
                                    self.camera_calib_callback(camera_pack)
                                self.is_calib_cmd = False
                                continue

                        pack = MessagePack.unpack(packet)
                        if pack:
                            for record in pack.records_:
                                if record.record_type == RecordType.Echo:
                                    try:
                                        text = record.record_data.decode("utf-8")
                                    except Exception:
                                        text = record.record_data.hex()
                                    print(f"Device response ({self.calib_cmd_name}): {text}")
                                    self.is_calib_cmd = False
                                    break
                    else:
                        pack = MessagePack.unpack(packet)
                        if not pack:
                            continue

                        for record in pack.records_:
                            try:
                                if record.record_type == RecordType.Tactile:
                                    if self.tactile_callback:
                                        self.tactile_callback(record.record_data)
                                elif record.record_type == RecordType.Encoder:
                                    if self.encoder_callback:
                                        self.encoder_callback(record.record_data)
                                elif record.record_type == RecordType.Echo:
                                    pass
                                else:
                                    logging.error(
                                        "record type:{} invalid !".format(record.record_type)
                                    )
                            except Exception as e:
                                logging.error(f"Callback error: {e}")
                                
                except Exception as e:
                    logging.error(f"Packet handling error: {e}")

            if packets_to_process:
                time.sleep(0.001)
            else:
                time.sleep(0.005)

    def _send_encoder_loop(self):
        if not self.encoder_freq:
            return
            
        interval = 1.0 / self.encoder_freq
        print(f"Encoder loop running at {self.encoder_freq} Hz, interval {interval:.3f}s")
        
        while self.is_running:
            start_time = time.time()
            
            with self.angle_lock:
                dis_target = self.gripper_dis
            
            self.add_cmd(
                CmdPack.pack(
                    opcode=Opcode.ReadBatch, 
                    record_type=RecordType.Encoder, 
                    record=struct.pack(">f", dis_target)
                ),
            )
            
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Encoder loop thread exiting")

    def _send_tactile_loop(self):
        if not self.tactile_freq:
            return
            
        interval = 1.0 / self.tactile_freq
        print(f"Tactile loop running at {self.tactile_freq} Hz, interval {interval:.3f}s")
        
        while self.is_running:
            start_time = time.time()
            self.add_cmd(
                CmdPack.pack(opcode=Opcode.ReadSingle, record_type=RecordType.Tactile, record=struct.pack(">f", 0.0))
            )
            
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Tactile loop thread exiting")

    def wait_for_calib_response(self, timeout=3.0, poll_interval=0.05):
        """Wait until calib response is received or timeout expires."""
        if not self.is_calib_cmd:
            return True
        print("Waiting for device response...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_calib_cmd:
                return True
            time.sleep(poll_interval)
        return not self.is_calib_cmd

    def stop(self):
        """Stop worker threads and close serial."""
        print("Stopping all threads...")
        self.is_running = False
        
        threads_to_join = []
        if self.read_thread and self.read_thread.is_alive():
            threads_to_join.append(self.read_thread)
        if self.send_thread and self.send_thread.is_alive():
            threads_to_join.append(self.send_thread)
        if self.parse_thread and self.parse_thread.is_alive():
            threads_to_join.append(self.parse_thread)
        if self.encoder_thread and self.encoder_thread.is_alive():
            threads_to_join.append(self.encoder_thread)
        if self.tactile_thread and self.tactile_thread.is_alive():
            threads_to_join.append(self.tactile_thread)
        
        for thread in threads_to_join:
            thread.join(timeout=2)
        
        if self.ser and self.ser.is_open:
            self.ser.close()

    def get_serial_info(self):
        if self.ser and self.ser.is_open:
            info = {
                "tty_port": self.tty_port,
                "baudrate": self.ser.baudrate,
                "bytesize": self.ser.bytesize,
                "parity": self.ser.parity,
                "stopbits": self.ser.stopbits,
                "timeout": self.ser.timeout,
                "in_waiting": self.ser.in_waiting,
            }
            return info
        return None


def check_and_fix_permission(port):
    """Ensure current user can read/write the serial node."""
    if not os.path.exists(port):
        return False
    
    if os.access(port, os.R_OK | os.W_OK):
        return True
    
    print(f"Trying to fix permissions on {port}...")
    try:
        subprocess.run(['sudo', 'chmod', '666', port], check=True)
        print(f"Permissions fixed: {port}")
        return True
    except subprocess.CalledProcessError:
        print(f"Permission fix failed; run manually: sudo chmod 666 {port}")
        return False


def find_configured_serial_port():
    """
    Find configured USB serial symlinks under /dev/ttyDevice*.

    Returns:
        First accessible port path, or None.
    """
    import glob
    configured_ports = glob.glob('/dev/ttyDevice*')
    
    if not configured_ports:
        return None
    
    for port in sorted(configured_ports):
        if os.path.exists(port) and check_and_fix_permission(port):
            return port
    
    return sorted(configured_ports)[0] if configured_ports else None


def find_serial_port(pattern="ttyUSB", max_retries=3, retry_interval=2):
    """
    Prefer /dev/ttyDevice* symlinks from udev rules (raw ttyUSB is not used).

    Args:
        pattern: Deprecated, kept for API compatibility.
        max_retries: Deprecated.
        retry_interval: Deprecated.

    Returns:
        Port path or None (prints setup hints).
    """
    configured_port = find_configured_serial_port()
    
    if configured_port:
        print(f"Using configured serial device: {configured_port}")
        return configured_port
    
    print("\n" + "=" * 60)
    print(" No configured USB serial symlink found")
    print("=" * 60)
    print("\nSetup:")
    print("1. See repository docs (e.g. README_En.md) for udev examples")
    print("2. Create a rules file (e.g. 99-usb-serial.rules)")
    print("3. Copy to /etc/udev/rules.d/")
    print("4. Reload:")
    print("   sudo udevadm control --reload-rules")
    print("   sudo udevadm trigger")
    print("\nYou should then see /dev/ttyDevice* symlinks")
    print("e.g. /dev/ttyDeviceLeft or your custom names")
    print("=" * 60 + "\n")
    return None
