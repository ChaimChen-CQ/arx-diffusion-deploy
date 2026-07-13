import multiprocessing as mp
import enum
import os
import struct
import sys
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, cast
import numpy as np
from utils.other_util import precise_wait
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from modules.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import time
from modules.arx5_zmq_client import Arx5Client
import numpy.typing as npt

from multiprocessing import Value
import ctypes
import threading


MAX_GRIPPER_WIDTH = 0.103


@dataclass(frozen=True)
class GripperControlConfig:
    demo_open_width: float = 0.095
    demo_close_width: float = 0.03
    demo_period: float = 1.0
    # The 0706 pick-place training set starts episodes with the gripper open
    # at about 0.095 m. Holding it closed before policy makes gripper obs OOD.
    pre_policy_hold_width: float = 0.095
    policy_gripper_enabled: bool = True


DEFAULT_GRIPPER_CONTROL_CONFIG = GripperControlConfig()


class GripperControlPhase(enum.Enum):
    CONNECTED_DEMO = 0
    PRE_POLICY_HOLD = 1
    POLICY_CONTROL = 2


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_TO_HOME = 3
    ADD_WAYPOINT = 4
    UPDATE_TRAJECTORY = 5
    SET_GRIPPER_PHASE = 6
    SET_JOINT_POS = 7


class Arx5Controller(mp.Process):
    pass

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        robot_port: int,
        launch_timeout: float = 10,
        frequency: float = 100,
        get_max_k: Optional[int] = None,
        verbose: bool = False,
        receive_latency: float = 0.0,
        skip_home: bool = False,
    ):
        super().__init__(name="Arx5Controller")
        self.robot_ip = robot_ip
        self.robot_port = robot_port

        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),
            "target_joint_pos": np.zeros((6,), dtype=np.float64),
            "gripper_pos": 0.0,
            "duration": 0.0,
            "target_time": 0.0,
            "gripper_phase": GripperControlPhase.CONNECTED_DEMO.value,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=example, buffer_size=256
        )
        self.verbose = verbose

        # build ring buffer
        receive_keys = [
            ("ActualTCPPose", "tcp_pose"),
            ("ActualQ", "joint_pos"),
            ("ActualQd", "joint_vel"),
            ("gripper_position", "gripper_pos"),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if "joint" in func_name:
                example[key] = np.zeros(6)
            elif "tcp_pose" in func_name:
                example[key] = np.zeros(6)
        example["gripper_position"] = 0.0

        example["robot_receive_timestamp"] = time.time()
        example["robot_timestamp"] = time.time()

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )
        self.launch_timeout = launch_timeout
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        self.frequency = frequency
        self.receive_latency = receive_latency

        # Will be initialized in the subprocess
        self.robot_client: Arx5Client
        self.reset_success = Value(ctypes.c_bool, False)
        self.skip_home = skip_home

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
        if self.verbose:
            print(f"[Arx5Controller] Controller process terminated at {self.pid}")

    def start_wait(self):
        print(f"[Arx5Controller] Waiting for controller process to be ready")
        print(f"{self.launch_timeout=}")
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= command methods ============
    def servoL(self, pose: npt.NDArray[np.float64], gripper_pos: float, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "duration": duration,
        }
        self.input_queue.put(message)

    def schedule_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def add_waypoint(
        self, pose: npt.NDArray[np.float64], gripper_pos: float, target_time: float
    ):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.ADD_WAYPOINT.value,
            "target_pose": pose,
            "gripper_pos": gripper_pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def update_trajectory(self):
        message = {
            "cmd": Command.UPDATE_TRAJECTORY.value,
        }
        self.input_queue.put(message)

    def set_gripper_control_phase(self, phase: GripperControlPhase):
        message = {
            "cmd": Command.SET_GRIPPER_PHASE.value,
            "gripper_phase": phase.value,
        }
        self.input_queue.put(message)

    def set_joint_pos(
        self,
        joint_pos: npt.NDArray[np.float64],
        gripper_pos: float,
        duration: float = 2.0,
        max_joint_step_rad: float = np.pi,
        timeout: float = 5.0,
    ):
        assert self.is_alive()
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        assert joint_pos.shape == (6,)
        self.reset_success.value = False
        message = {
            "cmd": Command.SET_JOINT_POS.value,
            "target_joint_pos": joint_pos,
            "gripper_pos": gripper_pos,
            "duration": float(duration),
            "target_time": float(max_joint_step_rad),
        }
        self.input_queue.put(message)
        start_time = time.monotonic()
        while not self.reset_success.value:
            if time.monotonic() - start_time > timeout:
                print(
                    f"\n[WARN] set_joint_pos 等待超时 ({timeout}s)，继续用当前实机状态。"
                )
                break
            time.sleep(0.05)

    def reset_to_home(self, timeout=5.0):
        self.reset_success.value = False
        message = {"cmd": Command.RESET_TO_HOME.value}
        self.input_queue.put(message)
        start_time = time.monotonic()
        while not self.reset_success.value:
            if time.monotonic() - start_time > timeout:
                print(f"\n[SAFE TEARDOWN] reset_to_home 等待超时 ({timeout}s)，触发短路保护以防止死锁。")
                break
            time.sleep(0.1)

    # ========= main loop in process ============
    def run(self):
        _encoder_val = [0.0]
        _encoder_lock = threading.Lock()

        def _encoder_cb(record_data: bytes):
            try:
                encoder_value = struct.unpack(">f", record_data)[0]
            except Exception as e:
                print(f"[Arx5Controller] Encoder data handler error: {e}")
                return
            with _encoder_lock:
                _encoder_val[0] = float(encoder_value)

        gripper_bus = None
        sdk_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "gen_con_sdk_python_release")
        )
        if sdk_root not in sys.path:
            sys.path.insert(0, sdk_root)
        try:
            from scripts.databus import DataBus

            serial_port = os.environ.get("GEN_GRIPPER_SERIAL_PORT", "/dev/ttyDeviceLeft")
            gripper_bus = DataBus(
                tty_port=serial_port,
                encoder_freq=30,
                encoder_callback=_encoder_cb,
            )
            print(f"[Arx5Controller] Gen gripper Python SDK connected: {serial_port}")
        except Exception as e:
            print(f"[Arx5Controller] Gen gripper Python SDK disabled: {e}")

        self.robot_client = Arx5Client(self.robot_ip, self.robot_port)
        if self.skip_home:
            print("[Arx5Controller] skip_home=True: holding current pose (bumpless)")
            self.robot_client.hold_current_pose()
        else:
            self.robot_client.reset_to_home()
        time.sleep(1)
        gain = self.robot_client.get_gain()
        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
        self.robot_client.set_gain(gain)
        np.set_printoptions(precision=3, suppress=True)

        self.waypoint_buffer = []
        gripper_control_config = DEFAULT_GRIPPER_CONTROL_CONFIG

        try:

            dt = 1 / self.frequency
            self.robot_client.get_state()
            curr_pose = self.robot_client.tcp_pose
            with _encoder_lock:
                curr_gripper_pos = _encoder_val[0]
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]), poses=np.array([curr_pose])
            )
            gripper_pos_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
            )
            gripper_control_phase = GripperControlPhase.CONNECTED_DEMO
            gripper_phase_start_time = curr_t

            def _clip_gripper_width(width: float) -> float:
                return float(np.clip(width, 0.0, MAX_GRIPPER_WIDTH))

            def _make_gripper_interp(target_width: float, at_time: float):
                return PoseTrajectoryInterpolator(
                    times=np.array([at_time]),
                    poses=np.array(
                        [[_clip_gripper_width(target_width), 0, 0, 0, 0, 0]]
                    ),
                )

            def _reset_gripper_interp(at_time: float):
                with _encoder_lock:
                    current_width = _encoder_val[0]
                return _make_gripper_interp(current_width, at_time)

            def _set_gripper_phase(phase: GripperControlPhase, at_time: float):
                nonlocal gripper_control_phase, gripper_phase_start_time, gripper_pos_interp
                if phase == gripper_control_phase:
                    return
                gripper_control_phase = phase
                gripper_phase_start_time = at_time
                gripper_pos_interp = _reset_gripper_interp(at_time)
                if self.verbose:
                    print(
                        f"[Arx5Controller] Gripper control phase -> {gripper_control_phase.name}"
                    )

            def _resolve_gripper_command(target_width: float, now: float) -> float:
                if gripper_control_phase == GripperControlPhase.CONNECTED_DEMO:
                    if gripper_control_config.demo_period <= 0:
                        return _clip_gripper_width(
                            gripper_control_config.demo_open_width
                        )
                    cycle_progress = (
                        (now - gripper_phase_start_time)
                        % gripper_control_config.demo_period
                    ) / gripper_control_config.demo_period
                    demo_width = (
                        gripper_control_config.demo_open_width
                        if cycle_progress < 0.5
                        else gripper_control_config.demo_close_width
                    )
                    return _clip_gripper_width(demo_width)
                if gripper_control_phase == GripperControlPhase.PRE_POLICY_HOLD:
                    return _clip_gripper_width(
                        gripper_control_config.pre_policy_hold_width
                    )
                if gripper_control_config.policy_gripper_enabled:
                    return _clip_gripper_width(target_width)
                return _clip_gripper_width(gripper_control_config.pre_policy_hold_width)

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                pose_cmd = pose_interp(t_now)
                scheduled_gripper_cmd = float(gripper_pos_interp(t_now)[0])
                gripper_cmd = _resolve_gripper_command(scheduled_gripper_cmd, t_now)

                try:
                    self.robot_client.set_tcp_pose(pose_cmd, 0.0)
                except ValueError as e:
                    print(f"\n[WARN] 忽略跳变动作 (ZMQ Reject): {e}")
                    self.robot_client.get_state()
                    curr_pose = self.robot_client.tcp_pose
                    pose_interp = PoseTrajectoryInterpolator(
                        times=np.array([t_now]), poses=np.array([curr_pose])
                    )
                    gripper_pos_interp = _reset_gripper_interp(t_now)
                    self.waypoint_buffer.clear()
                    last_waypoint_time = t_now
                if gripper_bus is not None:
                    try:
                        gripper_bus.set_target_distance(float(gripper_cmd))
                    except Exception as e:
                        print(f"[Arx5Controller] Gripper command failed: {e}")
                state = dict()
                for key, func_name in self.receive_keys:
                    if func_name == "gripper_pos":
                        with _encoder_lock:
                            state[key] = _encoder_val[0]
                    else:
                        state[key] = getattr(self.robot_client, func_name)
                t_recv = time.time()
                state["robot_receive_timestamp"] = t_recv
                state["robot_timestamp"] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # if self.verbose:
                #     print(f"Current: {state['ActualTCPPose']} target: {pose_cmd}, gripper: {state['gripper_position']:.3f}/{gripper_cmd:.3f}")

                try:
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands["cmd"])
                except Empty:
                    commands = {}
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.drive_to_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                f"[Arx5Controller] New pose target: {target_pose} duration {duration:.3f}s"
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        target_time = time.monotonic() - time.time() + target_time
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        gripper_pos_interp = gripper_pos_interp.schedule_waypoint(
                            pose=[command["gripper_pos"], 0, 0, 0, 0, 0],
                            time=target_time,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time

                    elif cmd == Command.RESET_TO_HOME.value:
                        self.robot_client.reset_to_home()
                        self.robot_client.get_state()

                        self.ring_buffer.clear()
                        state = dict()
                        for key, func_name in self.receive_keys:
                            if func_name == "gripper_pos":
                                with _encoder_lock:
                                    state[key] = _encoder_val[0]
                            else:
                                state[key] = getattr(self.robot_client, func_name)
                        t_recv = time.time()
                        state["robot_receive_timestamp"] = t_recv
                        state["robot_timestamp"] = t_recv - self.receive_latency
                        self.ring_buffer.put(state)
                        if self.verbose:
                            print(f"[Arx5Controller] Reset to home")

                        curr_pose = self.robot_client.tcp_pose
                        with _encoder_lock:
                            curr_gripper_pos = _encoder_val[0]
                        curr_t = time.monotonic()
                        last_waypoint_time = curr_t
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]), poses=np.array([curr_pose])
                        )
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]),
                            poses=np.array([[curr_gripper_pos, 0, 0, 0, 0, 0]]),
                        )
                        self.reset_success.value = True
                        gain = self.robot_client.get_gain()
                        gain["kp"] = np.array([300, 300, 400, 80, 50, 30])
                        self.robot_client.set_gain(gain)

                    elif cmd == Command.SET_JOINT_POS.value:
                        target_joint_pos = np.asarray(command["target_joint_pos"], dtype=np.float64)
                        duration = float(command["duration"])
                        max_joint_step_rad = float(command["target_time"])
                        with _encoder_lock:
                            current_gripper_pos = _encoder_val[0]
                        gripper_target = command["gripper_pos"]
                        if np.isfinite(gripper_target):
                            current_gripper_pos = float(gripper_target)
                        self.robot_client.set_joint_pos(
                            target_joint_pos,
                            gripper_pos=current_gripper_pos,
                            duration=duration,
                            max_joint_step_rad=max_joint_step_rad,
                        )
                        self.robot_client.get_state()

                        self.ring_buffer.clear()
                        state = dict()
                        for key, func_name in self.receive_keys:
                            if func_name == "gripper_pos":
                                with _encoder_lock:
                                    state[key] = _encoder_val[0]
                            else:
                                state[key] = getattr(self.robot_client, func_name)
                        t_recv = time.time()
                        state["robot_receive_timestamp"] = t_recv
                        state["robot_timestamp"] = t_recv - self.receive_latency
                        self.ring_buffer.put(state)

                        curr_pose = self.robot_client.tcp_pose
                        curr_t = time.monotonic()
                        last_waypoint_time = curr_t
                        pose_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]), poses=np.array([curr_pose])
                        )
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=np.array([curr_t]),
                            poses=np.array([[state["gripper_position"], 0, 0, 0, 0, 0]]),
                        )
                        self.waypoint_buffer = []
                        self.reset_success.value = True
                        if self.verbose:
                            print(f"[Arx5Controller] Set joint pos: {target_joint_pos}")

                    elif cmd == Command.ADD_WAYPOINT.value:
                        if len(self.waypoint_buffer) > 0:
                            last_waypoint_time = self.waypoint_buffer[-1]["target_time"]
                            if command["target_time"] <= last_waypoint_time:
                                print(
                                    f"[Arx5Controller] Waypoint time {command['target_time']:.3f} is not in the future, skipping"
                                )
                                continue
                        self.waypoint_buffer.append(command)
                    elif cmd == Command.UPDATE_TRAJECTORY.value:
                        if len(self.waypoint_buffer) == 0:
                            print(
                                f"[Arx5Controller] No new waypoints to update trajectory"
                            )
                            continue
                        start_time = time.monotonic()
                        # Dynamic latency matching
                        matching_dt = 0.01
                        pose_samples = np.zeros((3, 6))
                        pose_samples[0, :] = pose_interp(t_now - matching_dt)
                        pose_samples[1, :] = pose_interp(t_now)
                        pose_samples[2, :] = pose_interp(t_now + matching_dt)

                        input_poses = np.array(
                            [cmd["target_pose"] for cmd in self.waypoint_buffer]
                        )  # (N, 6)
                        input_times = np.array(
                            [cmd["target_time"] for cmd in self.waypoint_buffer]
                        )
                        input_gripper_pos = np.array(
                            [cmd["gripper_pos"] for cmd in self.waypoint_buffer]
                        )
                        input_pose_interp = PoseTrajectoryInterpolator(
                            times=input_times - input_times[0], poses=input_poses
                        )

                        latency_precision = 0.02
                        max_latency = 1.2
                        smoothing_time = 0.4
                        errors = []
                        error_weights = np.array(
                            [1, 1, 1, 0.1, 0.1, 0.1]
                        )  # x, y, z, rx, ry, rz
                        for latency in np.arange(
                            matching_dt, max_latency, latency_precision
                        ):
                            input_pose_samples = np.zeros((3, 6))
                            input_pose_samples[0, :] = input_pose_interp(
                                latency - matching_dt
                            )
                            input_pose_samples[1, :] = input_pose_interp(latency)
                            input_pose_samples[2, :] = input_pose_interp(
                                latency + matching_dt
                            )
                            error = np.sum(
                                np.abs(input_pose_samples - pose_samples)
                                * error_weights
                            )
                            errors.append(error)
                        errors = np.array(errors)
                        best_latency = np.arange(
                            matching_dt, max_latency, latency_precision
                        )[np.argmin(errors)]
                        # best_latency = 0.0

                        smoothened_input_poses = input_poses
                        new_times = input_times - input_times[0] + t_now - best_latency
                        for i in range((smoothened_input_poses.shape[0])):
                            if new_times[i] < t_now:
                                smoothened_input_poses[i] = pose_interp(new_times[i])
                            elif t_now <= new_times[i] < t_now + smoothing_time:
                                alpha = (new_times[i] - t_now) / smoothing_time
                                smoothened_input_poses[i] = (1 - alpha) * pose_interp(
                                    new_times[i]
                                ) + alpha * input_poses[i]
                            else:
                                smoothened_input_poses[i] = input_poses[i]

                        pose_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=smoothened_input_poses
                        )
                        extended_gripper_pos = np.zeros_like(input_poses)
                        extended_gripper_pos[:, 0] = input_gripper_pos
                        gripper_pos_interp = PoseTrajectoryInterpolator(
                            times=new_times, poses=extended_gripper_pos
                        )
                        if self.verbose:
                            print(
                                f"[Arx5Controller] latency: {best_latency:.3f}s, error: {errors.min():.3f}, time: {time.monotonic() - start_time:.3f}s"
                            )
                        # clear buffer
                        self.waypoint_buffer = []
                    elif cmd == Command.SET_GRIPPER_PHASE.value:
                        phase = GripperControlPhase(int(command["gripper_phase"]))
                        _set_gripper_phase(phase, t_now)
                    else:
                        keep_running = False
                        print(f"[Arx5Controller] Unknown command {cmd}")
                        break
                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                    if self.verbose:
                        print("[Arx5Controller] Controller process is ready")
                iter_idx += 1
                # if self.verbose:
                #     print(f"[Arx5Controller] Actual frequency {1/(time.monotonic() - t_now)} Hz")

        finally:
            print("[Arx5Controller] Setting robot to damping")
            self.robot_client.set_to_damping()
            if gripper_bus is not None:
                gripper_bus.stop()
            del self.robot_client
            self.ready_event.set()
            if self.verbose:
                print("[Arx5Controller] Controller process terminated")
