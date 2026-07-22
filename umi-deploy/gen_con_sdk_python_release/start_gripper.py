#!/usr/bin/env python3
"""Startup script — launches gripper hardware and cameras."""

import sys
import os
import argparse
import struct
import time
import cv2
import math
import threading
from typing import Optional

_sdk_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)
from scripts import GripperSystem
from tactile_processing import (
    convert_tactile_448_to_1000,
    set_tactile_grid_print_enabled,
    set_tactile_grid_print_max_hz,
    submit_tactile_1000_grid_print,
)

def capture_frames_callback(camera):
    """Callback for camera frame capture — uses background grab thread."""
    if camera.show_preview:
        for cam in camera.cameras:
            cv2.namedWindow(cam['window_name'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cam['window_name'], 640, 480)

    camera._start_grab_threads()

    frame_interval = 1.0 / camera.target_fps

    try:
        while camera.running:
            start_time = time.monotonic()
            frames_data = []

            for cam in camera.cameras:
                frame, ts_ns = camera._get_latest(cam)
                if frame is not None:
                    now = time.monotonic()
                    cam['disp_fps_ts'].append(now)
                    if len(cam['disp_fps_ts']) > 30:
                        cam['disp_fps_ts'] = cam['disp_fps_ts'][-30:]
                    if len(cam['disp_fps_ts']) >= 2:
                        dt = cam['disp_fps_ts'][-1] - cam['disp_fps_ts'][0]
                        if dt > 0:
                            cam['disp_fps_val'] = (len(cam['disp_fps_ts']) - 1) / dt

                frames_data.append((cam, frame if frame is not None else None))

            if camera.show_preview:
                _display_frames(camera, frames_data)

            elapsed = time.monotonic() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"Capture error: {e}")
    finally:
        camera._release_resources()


def _display_frames(camera, frames_data):
    """Show camera preview windows with FPS overlay."""
    for cam, frame in frames_data:
        if frame is not None:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            info_text = f"Camera_{cam['id']} | {timestamp} | Frames: {cam['frame_count']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            fps_text = f"Cap: {cam['cap_fps_val']:.1f}  Disp: {cam['disp_fps_val']:.1f}"
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.imshow(cam['window_name'], frame)
    if cv2.waitKey(1) == 27:
        camera.running = False


def tactile_callback(record_data: bytes):
    """Tactile callback: convert and enqueue grid print (non-blocking for serial parse thread)."""
    try:
        left_tactile_500, right_tactile_500 = convert_tactile_448_to_1000(record_data)
        submit_tactile_1000_grid_print(left_tactile_500 + right_tactile_500)
    except Exception as e:
        print(f"Tactile data handler error: {e}")


def encoder_callback(record_data: bytes):
    """Encoder data callback."""
    try:
        encoder_value = struct.unpack(">f", record_data)[0]
        print(f"gripper distance: {encoder_value:.3f} m")
    except Exception as e:
        print(f"Encoder data handler error: {e}")


class SineWaveController:
    """Sinusoidal gripper position control."""
    
    def __init__(self, system: GripperSystem, amplitude: float = 0.05, 
                 center: float = 0.05, frequency: float = 0.5, duration: float = 1000):
        self.system = system
        self.amplitude = amplitude
        self.center = center
        self.frequency = frequency
        self.duration = duration
        self.running = False
        self.control_thread = None
        self.start_time = 0
        self.control_interval = 1.0 / 30.0
        
    def start(self):
        """Start sinusoidal control."""
        if self.running:
            return
        if self.amplitude <= 0 or self.center - self.amplitude < 0 or self.center + self.amplitude > 0.103:
            print(" Sine wave parameters out of valid range")
            return
        
        self.running = True
        self.start_time = time.time()
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print(f"🚀 Sine wave started: center={self.center:.3f}m, amplitude=±{self.amplitude:.3f}m, freq={self.frequency:.2f}Hz")
    
    def stop(self):
        """Stop sinusoidal control."""
        if not self.running:
            return
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
    
    def _control_loop(self):
        """Control loop thread body."""
        try:
            while self.running:
                cycle_start = time.time()
                current_time = time.time() - self.start_time
                
                if self.duration > 0 and current_time >= self.duration:
                    self.running = False
                    break
                
                value = self.center + self.amplitude * math.sin(2 * math.pi * self.frequency * current_time)
                value = max(0.0, min(0.103, value))
                
                if self.system.databus:
                    self.system.databus.set_target_distance(value)
                
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.control_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f" Sine wave control error: {e}")
            self.running = False


class GripperController:
    """High-level gripper control (fixed distance vs sine wave)."""
    
    def __init__(self, system: GripperSystem):
        self.system = system
        self.sine_wave_controller: Optional[SineWaveController] = None
        
    def set_fixed_distance(self, distance: float):
        """Set a fixed gripper opening distance."""
        if distance < 0.0 or distance > 0.103:
            print(f" Warning: distance {distance} out of range [0.0, 0.103], ignored")
            return
        
        if self.sine_wave_controller and self.sine_wave_controller.running:
            self.sine_wave_controller.stop()
        
        try:
            self.system.set_gripper_distance(distance)
            print(f"Fixed gripper distance set: {distance} m ({distance*100:.1f} cm)")
        except Exception as e:
            print(f" Failed to set gripper distance: {e}")
    
    def start_sine_wave(self, amplitude: float = 0.05, center: float = 0.05, 
                        frequency: float = 0.5, duration: float = 60.0):
        """Start sinusoidal control."""
        if self.sine_wave_controller and self.sine_wave_controller.running:
            self.sine_wave_controller.stop()
        
        self.sine_wave_controller = SineWaveController(
            system=self.system, amplitude=amplitude, center=center,
            frequency=frequency, duration=duration
        )
        self.sine_wave_controller.start()
    
    def stop_sine_wave(self):
        """Stop sinusoidal control."""
        if self.sine_wave_controller:
            self.sine_wave_controller.stop()
    
    def is_sine_wave_running(self) -> bool:
        """Return True if sine wave control is active."""
        return self.sine_wave_controller.running if self.sine_wave_controller else False


def main():
    """CLI entry point."""
    SIDE_CONFIG = {
        'left': {
            'serial_port': "/dev/ttyDeviceLeft",
            'video_devices': [
                "/dev/left_video_0_main", "/dev/left_video_0_sec",
                "/dev/left_video_1_main", "/dev/left_video_1_sec",
                "/dev/left_video_2_main", "/dev/left_video_2_sec"
            ]
        },
        'right': {
            'serial_port': "/dev/ttyDeviceRight",
            'video_devices': [
                "/dev/right_video_0_main", "/dev/right_video_0_sec",
                "/dev/right_video_1_main", "/dev/right_video_1_sec",
                "/dev/right_video_2_main", "/dev/right_video_2_sec"
            ]
        }
    }
    
    parser = argparse.ArgumentParser(description="Start gripper system (optional sine wave mode)")
    parser.add_argument("side", type=str, choices=['left', 'right'],
                       help="Gripper side: left or right")
    parser.add_argument("--camera-resolutions", type=str, default="1600x1296",
                       help="Camera resolution as 'widthxheight'")
    parser.add_argument("--no-preview", action="store_true",
                       help="Do not show camera preview windows")
    parser.add_argument("--camera-fps", type=int, default=30,
                       help="Target camera display frame rate (default 30)")
    
    control_group = parser.add_mutually_exclusive_group()
    control_group.add_argument("--distance", type=float, default=None,
                              help="Fixed gripper distance in meters, range [0.0, 0.103]")
    control_group.add_argument("--sine-wave", action="store_true",
                              help="Enable sine wave control mode")
    
    parser.add_argument("--amplitude", type=float, default=0.025,
                       help="Sine amplitude in meters (default 0.025)")
    parser.add_argument("--center", type=float, default=0.05,
                       help="Sine center position in meters (default 0.05)")
    parser.add_argument("--frequency", type=float, default=0.5,
                       help="Sine frequency in Hz (default 0.5)")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Sine duration in seconds; 0 = run forever (default 10.0)")
    parser.add_argument(
        "--print-tactile-info",
        action="store_true",
        help="Print tactile grid to terminal (50 lines: L10 + gap + R10 per line); default is off",
    )
    parser.add_argument(
        "--tactile-print-hz",
        type=float,
        default=0.0,
        help="Cap tactile grid print rate (Hz); 0 = no cap. Reduces terminal load while showing latest frame per print.",
    )
    
    args = parser.parse_args()
    set_tactile_grid_print_enabled(args.print_tactile_info)
    set_tactile_grid_print_max_hz(args.tactile_print_hz)
    config = SIDE_CONFIG[args.side]
    
    system = GripperSystem(
        serial_port=config['serial_port'],
        camera_resolutions=args.camera_resolutions,
        show_preview=not args.no_preview,
        video_devices=config['video_devices'],
        tactile_callback=tactile_callback,
        encoder_callback=encoder_callback,
        capture_frames_callback=capture_frames_callback,
        camera_fps=args.camera_fps,
    )
    
    controller = GripperController(system)
    
    def setup_control_mode():
        """Apply control mode after DataBus is ready."""
        max_wait_time = 10.0
        wait_interval = 0.1
        elapsed_time = 0.0
        
        while elapsed_time < max_wait_time:
            if system.databus is not None:
                time.sleep(0.5)
                if args.sine_wave:
                    controller.start_sine_wave(
                        amplitude=args.amplitude, center=args.center,
                        frequency=args.frequency, duration=args.duration
                    )
                elif args.distance is not None:
                    controller.set_fixed_distance(args.distance)
                else:
                    controller.set_fixed_distance(0.05)
                return
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        print(" Warning: system init timed out; control mode not applied")
    
    threading.Thread(target=setup_control_mode, daemon=True).start()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if controller.is_sine_wave_running():
            controller.stop_sine_wave()
        system.stop()


if __name__ == "__main__":
    main()
