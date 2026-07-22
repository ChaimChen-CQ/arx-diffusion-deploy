import copy
from typing import Optional, Callable, Dict
import enum
import time
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from modules.timestamp_accumulator import get_accumulate_timestamp_idxs
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from peripherals.video_recorder import VideoRecorder
from utils.usb_util import reset_usb_device
import os


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class UvcCamera(mp.Process):
    """
    Call umi.common.usb_util.reset_all_elgato_devices
    if you are using Elgato capture cards.
    Required to workaround firmware bugs.
    """

    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes
    DEFAULT_REOPEN_ATTEMPTS = 40
    DEFAULT_REOPEN_INTERVAL_SEC = 0.5

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        # v4l2 device file path
        # e.g. /dev/video0
        # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
        dev_video_path,
        resolution=(1280, 720),
        capture_fps=60,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        cap_buffer_size=1,
        capture_fourcc=None,
        cpu_affinity=None,
        num_threads=2,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        enable_video_recording=True,
        verbose=False,
        reopen_attempts=DEFAULT_REOPEN_ATTEMPTS,
        reopen_interval=DEFAULT_REOPEN_INTERVAL_SEC,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {"color": np.empty(shape=shape + (3,), dtype=np.uint8)}
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0
        print(f"{examples['color'].shape=}")

        vis_examples = copy.deepcopy(examples)
        ### WTF why this doesn't work?
        # tf_example = {'color': np.empty(shape=shape+(3,), dtype=np.uint8)}
        # vis_examples = examples.copy()
        # vis_shape = vis_transform(tf_example)["color"].shape
        # print(f"{vis_shape=}")
        if vis_transform is not None:
            vis_shape = (720, 960, 3)
        else:
            vis_shape = shape + (3,)
        vis_examples["color"] = np.empty(shape=vis_shape, dtype=np.uint8)
        # print(f"{vis_examples['color'].shape=}, {examples['color'].shape=}")
        # print(f"{vis_examples['color'].flags=}, {examples['color'].flags=}")

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=vis_examples,
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.RESTART_PUT.value,
            "put_start_time": 0.0,
            "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            "recording_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        self.enable_video_recording = bool(enable_video_recording)
        # create video recorder
        if self.enable_video_recording and video_recorder is None:
            # default to nvenc GPU encoder
            video_recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps,
                input_pix_fmt="bgr24",
                bit_rate=6000 * 1000,
            )
        if self.enable_video_recording:
            assert video_recorder is not None
            assert video_recorder.fps == capture_fps
        else:
            video_recorder = None

        # copied variables
        self.shm_manager = shm_manager
        self.dev_video_path = dev_video_path
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.capture_fourcc = capture_fourcc
        self.cpu_affinity = cpu_affinity
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads
        self.reopen_attempts = int(reopen_attempts)
        self.reopen_interval = float(reopen_interval)

        if self.reopen_attempts <= 0:
            raise ValueError("reopen_attempts must be > 0")
        if self.reopen_interval <= 0:
            raise ValueError("reopen_interval must be > 0")

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.recovering_event = mp.Event()
        self.failed_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.last_frame_time = mp.Value("d", 0.0)
        self.last_failure_time = mp.Value("d", 0.0)
        self.failure_count = mp.Value("i", 0)

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        self.ready_event.clear()
        self.recovering_event.clear()
        self.failed_event.clear()
        with self.last_frame_time.get_lock():
            self.last_frame_time.value = 0.0
        with self.last_failure_time.get_lock():
            self.last_failure_time.value = 0.0
        with self.failure_count.get_lock():
            self.failure_count.value = 0
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        if self.video_recorder is not None:
            self.video_recorder.start(
                shm_manager=self.shm_manager, data_example=data_example
            )
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self.video_recorder is not None:
            self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        # Avoid dead-lock: if the camera subprocess crashes before signaling ready,
        # the parent would otherwise wait forever.
        while not self.ready_event.wait(timeout=0.5):
            if not self.is_alive():
                raise RuntimeError(
                    f"UvcCamera subprocess exited before ready (dev_video_path={self.dev_video_path}). "
                    f"Check its stderr/traceback above for the root cause."
                )
        if self.video_recorder is not None:
            self.video_recorder.start_wait()

    def end_wait(self):
        self.join()
        if self.video_recorder is not None:
            self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    @property
    def is_recovering(self):
        return self.recovering_event.is_set()

    @property
    def has_failed(self):
        return self.failed_event.is_set()

    def get_health_status(self):
        with self.last_frame_time.get_lock():
            last_frame_time = self.last_frame_time.value
        with self.last_failure_time.get_lock():
            last_failure_time = self.last_failure_time.value
        with self.failure_count.get_lock():
            failure_count = self.failure_count.value
        return {
            "dev_video_path": self.dev_video_path,
            "process_alive": self.is_alive(),
            "ready": self.is_ready,
            "recovering": self.is_recovering,
            "failed": self.has_failed,
            "ring_buffer_count": self.ring_buffer.count,
            "last_frame_time": last_frame_time,
            "last_failure_time": last_failure_time,
            "failure_count": failure_count,
        }

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    def start_recording(self, video_path: str, start_time: float = -1):
        if self.video_recorder is None:
            print(f"[UvcCamera {self.dev_video_path}] video recording disabled; ignoring start_recording.")
            return
        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        if self.video_recorder is None:
            return
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    def _describe_dev_path(self):
        dev_path = self.dev_video_path
        info = {"configured_path": dev_path}
        if isinstance(dev_path, str):
            info["path_exists"] = os.path.exists(dev_path)
            info["path_lexists"] = os.path.lexists(dev_path)
            try:
                real_path = os.path.realpath(dev_path)
            except Exception as e:
                real_path = f"<realpath failed: {e}>"
            info["real_path"] = real_path
            info["real_path_exists"] = (
                os.path.exists(real_path) if isinstance(real_path, str) else False
            )
        return info

    def _open_capture(self):
        """
        Resolve the current video node from the configured path/symlink and open it.
        This is intentionally re-runnable so the process can recover from USB
        re-enumeration without requiring a full env restart.
        """
        dev_path = self.dev_video_path
        real_path = None
        video_index = None
        if isinstance(dev_path, int):
            video_index = dev_path
        elif isinstance(dev_path, str):
            if dev_path.isdigit():
                video_index = int(dev_path)
            else:
                try:
                    real_path = os.path.realpath(dev_path)
                except Exception:
                    real_path = None
                if real_path is not None and real_path.startswith("/dev/video"):
                    suffix = real_path[len("/dev/video") :]
                    if suffix.isdigit():
                        video_index = int(suffix)

        if video_index is not None:
            cap = cv2.VideoCapture(video_index, cv2.CAP_V4L2)
            opened_as = (
                f"index={video_index}"
                if real_path is None
                else f"index={video_index} (from {dev_path} -> {real_path})"
            )
        else:
            cap = cv2.VideoCapture(dev_path)
            opened_as = f"path={dev_path}"

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            path_info = self._describe_dev_path()
            raise RuntimeError(
                "Failed to open UVC camera. "
                f"Tried opening as {opened_as}. "
                f"Path state: {path_info}. "
                "If you see 'can't be used to capture by name', prefer passing /dev/videoN "
                "or a /dev/v4l/by-id/... symlink that resolves to /dev/videoN."
            )

        w, h = self.resolution
        fps = self.capture_fps
        if self.capture_fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.capture_fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w > 0 and actual_h > 0 and (actual_w, actual_h) != (w, h):
            cap.release()
            raise RuntimeError(
                f"UVC camera reported resolution {actual_w}x{actual_h} after "
                f"requesting {w}x{h}. Use intrinsics calibrated for the actual "
                "capture resolution or change the requested resolution."
            )
        print(
            f"UvcCamera {self.dev_video_path} set resolution {w}x{h} and fps {fps}, buffer size {self.cap_buffer_size}"
        )
        return cap, opened_as

    # ========= interval API ===========
    def run(self):

        if self.cpu_affinity is not None:
            pid = os.getpid()
            try:
                os.sched_setaffinity(pid, list(self.cpu_affinity))
            except Exception as e:
                print(f"[UvcCamera {self.dev_video_path}] failed to set CPU affinity {self.cpu_affinity}: {e}")
        # limit threads
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        cap = None
        opened_as = None
        try:
            cap, opened_as = self._open_capture()
            frame_buffer = np.empty(shape=self.resolution[::-1] + (3,), dtype=np.uint8)
            dropped_recording_frames = 0

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                try:
                    ts = time.time()
                    ret = cap.grab()
                    if not ret:
                        raise RuntimeError(
                            f"OpenCV cap.grab() failed for camera {opened_as}. "
                            "This usually means the device cannot stream (busy, permission issue, "
                            "unsupported resolution/fps, or capture backend failure)."
                        )

                    ret, frame = cap.retrieve(frame_buffer)
                    t_recv = time.time()
                    if not ret:
                        raise RuntimeError(
                            f"OpenCV cap.retrieve() failed for camera {opened_as}. "
                            "This usually indicates a streaming/capture failure."
                        )
                except Exception as e:
                    recovery_start = time.monotonic()
                    print(
                        f"[UvcCamera {self.dev_video_path}] capture failure: {e}. "
                        f"path_state={self._describe_dev_path()}"
                    )
                    self.recovering_event.set()
                    with self.last_failure_time.get_lock():
                        self.last_failure_time.value = time.time()
                    with self.failure_count.get_lock():
                        self.failure_count.value += 1
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None

                    reopened = False
                    attempts_made = 0
                    for attempt in range(self.reopen_attempts):
                        if self.stop_event.is_set():
                            break
                        time.sleep(self.reopen_interval)
                        attempts_made = attempt + 1
                        try:
                            cap, opened_as = self._open_capture()
                            self.recovering_event.clear()
                            recovery_elapsed = time.monotonic() - recovery_start
                            print(
                                f"[UvcCamera {self.dev_video_path}] recovered on reopen attempt "
                                f"{attempt + 1} after {recovery_elapsed:.3f}s: {opened_as}"
                            )
                            reopened = True
                            break
                        except Exception as reopen_error:
                            recovery_elapsed = time.monotonic() - recovery_start
                            print(
                                f"[UvcCamera {self.dev_video_path}] reopen attempt {attempt + 1} "
                                f"failed after {recovery_elapsed:.3f}s: {reopen_error}"
                            )

                    if not reopened:
                        self.failed_event.set()
                        recovery_elapsed = time.monotonic() - recovery_start
                        stop_reason = (
                            "stop requested during recovery"
                            if self.stop_event.is_set()
                            else "reopen budget exhausted"
                        )
                        print(
                            f"[UvcCamera {self.dev_video_path}] giving up after "
                            f"{attempts_made} reopen attempts over "
                            f"{recovery_elapsed:.3f}s ({stop_reason})"
                        )
                        raise
                    continue
                mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t_cap = mt_cap - time.monotonic() + time.time()
                t_cal = t_recv - self.receive_latency  # calibrated latency
                with self.last_frame_time.get_lock():
                    self.last_frame_time.value = t_recv

                # record frame
                if self.video_recorder is not None and self.video_recorder.is_ready():
                    try:
                        recording_frame = frame
                        if self.recording_transform is not None:
                            recording_frame = self.recording_transform(
                                {"color": frame.copy()}
                            )["color"]
                        self.video_recorder.write_frame(recording_frame, frame_time=t_cal)
                    except Full:
                        dropped_recording_frames += 1
                        log_every = int(max(1, self.capture_fps))
                        if dropped_recording_frames == 1 or dropped_recording_frames % log_every == 0:
                            qsize = (
                                self.video_recorder.img_queue.qsize()
                                if self.video_recorder.img_queue is not None
                                else "n/a"
                            )
                            print(
                                f"[UvcCamera {self.dev_video_path}] video recorder queue full; "
                                f"dropped recording frame count={dropped_recording_frames}, qsize={qsize}. "
                                "Capture continues."
                            )

                data = dict()
                data["camera_receive_timestamp"] = t_cap
                data["camera_capture_timestamp"] = t_recv
                data["color"] = frame

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[t_cal],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True,
                    )

                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        put_data["timestamp"] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[UvcCamera {self.dev_video_path}] FPS {frequency}")

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command["video_path"])
                        start_time = command["recording_start_time"]
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(
                            video_path, start_time=start_time
                        )
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()

                iter_idx += 1
        except Exception:
            self.failed_event.set()
            self.recovering_event.clear()
            raise
        finally:
            self.recovering_event.clear()
            if self.video_recorder is not None:
                self.video_recorder.stop()
            # When everything done, release the capture
            if cap is not None:
                cap.release()
