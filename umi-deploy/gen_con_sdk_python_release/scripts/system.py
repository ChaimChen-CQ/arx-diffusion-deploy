#!/usr/bin/env python3
"""
GripperSystem — main orchestration class for the gripper stack.
Starts and manages serial communication and cameras.
"""

import time
import signal
import threading
from typing import Optional, List, Callable
from .databus import DataBus, find_serial_port
from .camera import CameraCapture


class GripperSystem:
    """Main gripper system controller."""
    def __init__(
        self,
        serial_port: Optional[str] = None,
        camera_resolutions: str = "1600x1296",
        show_preview: bool = True,
        video_devices: Optional[List[str]] = None,
        tactile_callback: Optional[Callable] = None,
        encoder_callback: Optional[Callable] = None,
        capture_frames_callback: Optional[Callable] = None,
        camera_fps: int = 30,
    ):
        """
        Initialize the gripper system.

        Args:
            serial_port: Serial device path; if None, auto-detect.
            camera_resolutions: Resolution string like "widthxheight".
            show_preview: Whether to show OpenCV preview windows.
            video_devices: Optional list e.g. ["/dev/video0", ...].
            tactile_callback: Optional tactile data callback.
            encoder_callback: Optional encoder data callback.
            capture_frames_callback: Optional frame capture callback; if omitted,
                uses camera.capture_frames_callback().
            camera_fps: Target camera display frame rate (default 30).
        """
        self.running = True
        self.serial_port = serial_port
        self.camera_resolutions = camera_resolutions
        self.show_preview = show_preview
        self.video_devices = video_devices
        self.tactile_callback = tactile_callback
        self.encoder_callback = encoder_callback
        self.capture_frames_callback = capture_frames_callback
        self.camera_fps = camera_fps
        
        # Parse resolution strings
        self.resolutions = []
        for res_str in camera_resolutions.split(','):
            try:
                width, height = map(int, res_str.strip().split('x'))
                self.resolutions.append((width, height))
            except:
                pass
        
        if not self.resolutions:
            self.resolutions = [(1600, 1296)]
        
        self.databus = None
        self.camera = None
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM."""
        if not self.running:
            import sys
            sys.exit(0)
        print(f"\nReceived signal ({signum}), shutting down...")
        self.running = False
    
    def start(self):
        """Start cameras and serial bus."""
        print("=" * 60)
        print("Starting gripper system...")
        print("=" * 60)
        
        # 1) Resolve serial port (for camera init; DataBus comes after cameras)
        if not self.serial_port:
            self.serial_port = find_serial_port("ttyUSB")
            if not self.serial_port:
                print(" No usable serial port found")
                return False
        
        print(f"Using serial port: {self.serial_port}")
        
        # 2) Cameras first
        print("\n[1/2] Initializing cameras...")
        try:
            self.camera = CameraCapture(
                serial_port=self.serial_port,
                camera_count=3,
                resolutions=self.resolutions,
                show_preview=self.show_preview,
                video_devices=self.video_devices,
                target_fps=self.camera_fps,
            )
            print("Cameras initialized")
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            print(f" Camera init failed: {e}")
            self.stop()
            return False
        
        time.sleep(1.0)
        
        # 3) DataBus (serial protocol)
        print("\n[2/2] Initializing serial communication...")
        try:
            self.databus = DataBus(
                tty_port=self.serial_port,
                baudrate=921600,
                encoder_freq=30,  # 30 Hz encoder polling
                tactile_callback=self.tactile_callback,
                encoder_callback=self.encoder_callback,
            )
            print("Serial communication ready")
        except Exception as e:
            print(f" Serial communication failed: {e}")
            self.stop()
            return False
        
        print("\n" + "=" * 60)
        print("System started")
        print("=" * 60)
        print("\nUsage:")
        print("  - Three camera preview windows (if enabled)")
        print("  - Control gripper with GripperController (or your own code)")
        print("  - Press ESC in preview or Ctrl+C to stop")
        print("=" * 60)
        
        try:
            if self.capture_frames_callback:
                camera_thread = threading.Thread(target=self.capture_frames_callback, args=(self.camera,))
            else:
                camera_thread = threading.Thread(target=self.camera.capture_frames_callback)
            camera_thread.daemon = True
            camera_thread.start()
            
            while self.running:
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    self.running = False
                    break
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nSystem error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop cameras and serial."""
        print("\nStopping system...")
        
        if self.camera:
            self.camera.stop()
        
        if self.databus:
            self.databus.stop()
        
        print("System stopped")
    
    def set_gripper_distance(self, distance: float):
        """
        Set target gripper opening distance.

        Args:
            distance: Target distance in [0.0, 0.103] m (~10 cm max).
        """
        if self.databus:
            self.databus.set_target_distance(distance)
        else:
            print(" DataBus not initialized")
