#!/usr/bin/env python3
"""
Camera - Pure Python implementation for camera capture
Removed ROS dependencies, using callbacks instead of ROS topics
"""

import cv2
import os
import time
import glob
import subprocess
import signal
import sys
import numpy as np
import threading
import re
from typing import List, Callable, Optional, Tuple


class CameraCapture:
    def __init__(
        self,
        serial_port: str = "",
        camera_count: int = 3,
        resolutions: List[Tuple[int, int]] = None,
        show_preview: bool = True,
        video_devices: List[str] = None,
        frame_callback: Optional[Callable] = None,
        target_fps: int = 30,
    ):
        """
        Initialize multi-camera capture.

        Args:
            serial_port: USB serial path (used when filtering V4L devices).
            camera_count: Number of cameras.
            resolutions: List of (width, height) tuples.
            show_preview: Show OpenCV preview windows.
            video_devices: Explicit device paths e.g. ["/dev/video0", ...].
            frame_callback: callback(camera_id, frame, timestamp_ns).
            target_fps: Target display frame rate (default 30).
        """
        self.serial_port = serial_port
        self.camera_count = camera_count
        self.resolutions = resolutions or [(1600, 1296)]
        self.show_preview = show_preview
        self.video_devices = video_devices or []
        self.frame_callback = frame_callback
        self.target_fps = target_fps
        
        self.cameras = []
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._init_cameras()

    def _signal_handler(self, signum, frame):
        """Handle stop signals."""
        print(f"\nReceived signal ({signum}), stopping capture...")
        self.running = False

    def _get_physical_devices(self):
        """Discover physical V4L2 video nodes."""
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                 capture_output=True, text=True)
            devices = []
            current_dev = ""
            device_names = {}
            
            for line in result.stdout.split('\n'):
                if not line.strip():
                    continue
                if ':' in line and not line.startswith('/dev/'):
                    current_dev = line.split(':')[0].strip()
                elif line.startswith('/dev/video'):
                    dev_path = line.strip()
                    if os.path.exists(dev_path):
                        devices.append(dev_path)
                        device_names[dev_path] = current_dev
            
            print(f"Detected video devices: {devices}")
            
            if self.video_devices:
                filtered_devices = []
                for dev in self.video_devices:
                    if os.path.exists(dev):
                        filtered_devices.append(dev)
                if filtered_devices:
                    devices = filtered_devices
                    print(f"Using explicit video devices: {devices}")
            
            return sorted(list(set(devices))) if devices else sorted(glob.glob('/dev/video*'))
        except Exception as e:
            print(f"Error listing video devices: {e}")
            return sorted(glob.glob('/dev/video*'))

    def _try_reset_device(self, dev_path):
        """Try USB reset via sysfs."""
        try:
            udev_info = subprocess.run(
                ['udevadm', 'info', '-q', 'path', '-n', dev_path],
                capture_output=True, text=True
            ).stdout.strip()
            
            if udev_info:
                usb_path = f"/sys{udev_info}/../reset"
                if os.path.exists(usb_path):
                    with open(usb_path, 'w') as f:
                        f.write('1')
                    time.sleep(2)
                    return True
        except:
            pass
        return False

    def _init_camera(self, dev_path, cam_id):
        """Open and configure one camera node."""
        for attempt in range(3):
            try:
                if not os.path.exists(dev_path):
                    print(f"Device {dev_path} does not exist")
                    continue

                if attempt > 0:
                    self._try_reset_device(dev_path)
                    os.system(f'sudo chmod 666 {dev_path}')
                    os.system(f'sudo fuser -k {dev_path} 2>/dev/null')

                cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
                if not cap.isOpened():
                    print('OpenCV failed to open device')
                    return False

                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                success = False
                actual_width = 0
                actual_height = 0
                
                for res in self.resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if actual_width == res[0] and actual_height == res[1]:
                        print(f"Camera {cam_id} set to {actual_width}x{actual_height}")
                        success = True
                        break
                    else:
                        print(f"Camera {cam_id}: requested {res[0]}x{res[1]}, got {actual_width}x{actual_height}")
                
                if not success:
                    print(f"Camera {cam_id}: no requested resolution matched; using {actual_width}x{actual_height}")
                
                for _ in range(5):
                    cap.grab()
                    time.sleep(0.01)

                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                window_name = f'Camera_{cam_id}_{actual_width}x{actual_height}'
                
                self.cameras.append({
                    'id': cam_id,
                    'cap': cap,
                    'dev': dev_path,
                    'frame_count': 0,
                    'width': actual_width,
                    'height': actual_height,
                    'window_name': window_name,
                    'lock': threading.Lock(),
                    'latest_frame': None,
                    'latest_ts_ns': 0,
                    'cap_fps_ts': [],
                    'cap_fps_val': 0.0,
                    'disp_fps_ts': [],
                    'disp_fps_val': 0.0,
                })
                return True

            except Exception as e:
                print(f"Attempt #{attempt+1} failed for {dev_path}: {str(e)}")
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                time.sleep(1)
        return False

    def _init_main_or_second_camera(self, dev_main, dev_sec, cam_id):
        """Try main device, then secondary fallback."""
        if dev_main and os.path.exists(dev_main):
            if self._init_camera(dev_main, cam_id):
                print(f"Initialized {dev_main} as camera_{cam_id}")
                return True
        elif dev_main:
            print(f" Device {dev_main} missing, trying secondary")
        
        if dev_sec and os.path.exists(dev_sec):
            if self._init_camera(dev_sec, cam_id):
                print(f"Initialized {dev_sec} as camera_{cam_id}")
                return True
        
        print(f" Failed to open camera {cam_id} (main: {dev_main}, sec: {dev_sec})")
        return False

    def _find_configured_camera_devices(self):
        """
        Find udev symlinks like /dev/*_video_*_main and *_sec.

        Returns:
            Dict mapping camera index (0,1,2) to (main_dev, sec_dev).
        """
        import glob
        
        main_devices = glob.glob('/dev/*_video_*_main')
        sec_devices = glob.glob('/dev/*_video_*_sec')
        
        camera_configs = {}
        
        for dev in main_devices:
            match = re.search(r'video_(\d+)_main', dev)
            if match:
                cam_num = int(match.group(1))
                if cam_num not in camera_configs:
                    camera_configs[cam_num] = (dev, None)
                else:
                    camera_configs[cam_num] = (dev, camera_configs[cam_num][1])
        
        for dev in sec_devices:
            match = re.search(r'video_(\d+)_sec', dev)
            if match:
                cam_num = int(match.group(1))
                if cam_num not in camera_configs:
                    camera_configs[cam_num] = (None, dev)
                else:
                    camera_configs[cam_num] = (camera_configs[cam_num][0], dev)
        
        return camera_configs

    def _init_cameras(self):
        """Open all cameras according to video_devices or udev symlinks."""
        if self.video_devices:
            if len(self.video_devices) >= 6:
                video_configs = [
                    (self.video_devices[0], self.video_devices[1]),
                    (self.video_devices[2], self.video_devices[3]),
                    (self.video_devices[4], self.video_devices[5]),
                ]
            elif len(self.video_devices) == 3:
                video_configs = [
                    (self.video_devices[0], ''),
                    (self.video_devices[1], ''),
                    (self.video_devices[2], ''),
                ]
            else:
                print(f" video_devices length {len(self.video_devices)} invalid; expect 3 or 6")
                video_configs = None
            
            if video_configs:
                # cam1/cam2 swap per launch file convention
                video_mapping = [
                    (video_configs[0][0], video_configs[0][1]),
                    (video_configs[2][0], video_configs[2][1]),
                    (video_configs[1][0], video_configs[1][1]),
                ]
                
                for cam_id, (dev_main, dev_sec) in enumerate(video_mapping):
                    self._init_main_or_second_camera(dev_main, dev_sec, cam_id)
        else:
            camera_configs = self._find_configured_camera_devices()
            
            if not camera_configs:
                print("\n" + "=" * 60)
                print(" No configured camera symlinks found")
                print("=" * 60)
                print("\nSetup steps:")
                print("1. See repository docs (e.g. README_En.md) for udev camera symlinks")
                print("2. Create a udev rules file (e.g. 99-usb-serial.rules)")
                print("3. Add SYMLINK rules such as:")
                print("   SYMLINK+=\"left_video_0_main\"  (center camera main)")
                print("   SYMLINK+=\"left_video_0_sec\"   (center camera secondary)")
                print("   SYMLINK+=\"left_video_1_main\"  (left camera main)")
                print("   SYMLINK+=\"left_video_1_sec\"   (left camera secondary)")
                print("   SYMLINK+=\"left_video_2_main\"  (right camera main)")
                print("   SYMLINK+=\"left_video_2_sec\"   (right camera secondary)")
                print("4. Copy rules to /etc/udev/rules.d/")
                print("5. Reload:")
                print("   sudo udevadm control --reload-rules")
                print("   sudo udevadm trigger")
                print("\nAfter setup you should see *_video_*_main / *_sec under /dev/")
                print("e.g. /dev/left_video_0_main, /dev/left_video_0_sec")
                print("=" * 60 + "\n")
                sys.exit(1)
            
            required_cams = [0, 1, 2]
            missing_cams = []
            for cam_num in required_cams:
                if cam_num not in camera_configs:
                    missing_cams.append(cam_num)
                elif camera_configs[cam_num][0] is None:
                    missing_cams.append(cam_num)
            
            if missing_cams:
                print(f"\n Missing configured cameras")
                print(f"Missing indices: {missing_cams}")
                print("Ensure video_0, video_1, video_2 symlinks exist (see repository udev docs)")
                sys.exit(1)
            
            video_mapping = [
                (camera_configs[0][0], camera_configs[0][1]),
                (camera_configs[2][0], camera_configs[2][1]),
                (camera_configs[1][0], camera_configs[1][1]),
            ]
            
            print(f"Configured camera devices:")
            for cam_id, (dev_main, dev_sec) in enumerate(video_mapping):
                print(f"  Camera_{cam_id}: main={dev_main}, sec={dev_sec}")
            
            for cam_id, (dev_main, dev_sec) in enumerate(video_mapping):
                self._init_main_or_second_camera(dev_main, dev_sec, cam_id)

        if not self.cameras:
            print("\n No cameras available")
            print("Diagnostics:")
            print("1. ls /dev/video*")
            print("2. v4l2-ctl --list-devices")
            print("3. ls -l /dev/*_video_*")
            print("4. sudo chmod 666 /dev/*_video_*")
            print("\nTip: configure camera symlinks per repository udev documentation")
            sys.exit(1)
        
        print(f"\nOpened {len(self.cameras)} camera(s)")

    def _sync_grab_loop(self):
        """Background thread: grab all cameras synchronously, then retrieve and cache."""
        while self.running:
            # Grab all cameras as close together as possible
            grab_results = {}
            for cam in self.cameras:
                grab_results[cam['id']] = cam['cap'].grab()

            now = time.monotonic()
            ts_ns = time.time_ns()

            for cam in self.cameras:
                if not grab_results[cam['id']]:
                    continue
                ret, frame = cam['cap'].retrieve()
                if not ret or frame is None:
                    continue

                # Fire frame callback
                if self.frame_callback:
                    try:
                        self.frame_callback(cam['id'], frame, ts_ns)
                    except Exception as e:
                        print(f"Frame callback error: {e}")
                cam['frame_count'] += 1

                with cam['lock']:
                    cam['latest_frame'] = frame
                    cam['latest_ts_ns'] = ts_ns

                # Capture FPS (sliding window of last 30 frames)
                cam['cap_fps_ts'].append(now)
                if len(cam['cap_fps_ts']) > 30:
                    cam['cap_fps_ts'] = cam['cap_fps_ts'][-30:]
                if len(cam['cap_fps_ts']) >= 2:
                    dt = cam['cap_fps_ts'][-1] - cam['cap_fps_ts'][0]
                    if dt > 0:
                        cam['cap_fps_val'] = (len(cam['cap_fps_ts']) - 1) / dt

    def _start_grab_threads(self):
        """Start the background sync-grab thread."""
        t = threading.Thread(target=self._sync_grab_loop, daemon=True)
        self._grab_thread = t
        t.start()

    def _stop_grab_threads(self):
        """Stop the background grab thread."""
        self.running = False
        t = getattr(self, '_grab_thread', None)
        if t and t.is_alive():
            t.join(timeout=3)

    def _get_latest(self, cam):
        """Thread-safe: return the latest cached frame and timestamp, then clear."""
        with cam['lock']:
            frame = cam['latest_frame']
            ts_ns = cam['latest_ts_ns']
            cam['latest_frame'] = None
        return frame, ts_ns

    def _display_frames(self, frames_data):
        """Draw overlay and show frames."""
        for cam, frame in frames_data:
            if frame is not None:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                info_text = f"Camera_{cam['id']} | {timestamp} | Frames: {cam['frame_count']}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)
                fps_text = f"Cap: {cam['cap_fps_val']:.1f}  Disp: {cam['disp_fps_val']:.1f}"
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 255), 2)

                cv2.imshow(cam['window_name'], frame)

        if cv2.waitKey(1) == 27:
            self.running = False

    def capture_frames_callback(self):
        """
        Capture loop: background thread grabs frames, main loop displays.
        """
        print(f"\nCapturing from {len(self.cameras)} camera(s)...")
        print("Press ESC or Ctrl+C to stop")

        if self.show_preview:
            for cam in self.cameras:
                RESIZE_WIDTH = 640
                RESIZE_HEIGHT = 480
                cv2.namedWindow(cam['window_name'], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cam['window_name'], RESIZE_WIDTH, RESIZE_HEIGHT)

        self._start_grab_threads()

        frame_interval = 1.0 / self.target_fps

        try:
            while self.running:
                start_time = time.monotonic()
                frames_data = []

                for cam in self.cameras:
                    frame, ts_ns = self._get_latest(cam)
                    if frame is not None:
                        now = time.monotonic()
                        cam['disp_fps_ts'].append(now)
                        if len(cam['disp_fps_ts']) > 30:
                            cam['disp_fps_ts'] = cam['disp_fps_ts'][-30:]
                        if len(cam['disp_fps_ts']) >= 2:
                            dt = cam['disp_fps_ts'][-1] - cam['disp_fps_ts'][0]
                            if dt > 0:
                                cam['disp_fps_val'] = (len(cam['disp_fps_ts']) - 1) / dt

                    frames_data.append((cam, frame))

                if self.show_preview:
                    self._display_frames(frames_data)

                elapsed = time.monotonic() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Capture error: {e}")
        finally:
            self._release_resources()
    
    def capture_frames(self):
        """Backward-compatible alias for capture_frames_callback."""
        self.capture_frames_callback()

    def _release_resources(self):
        """Stop grab threads, release OpenCV captures and windows."""
        self._stop_grab_threads()
        for cam in self.cameras:
            try:
                cam['cap'].release()
            except:
                pass

        if self.show_preview:
            for cam in self.cameras:
                try:
                    cv2.destroyWindow(cam['window_name'])
                except:
                    pass

    def stop(self):
        """Stop capture thread logic."""
        self.running = False
        self._stop_grab_threads()

