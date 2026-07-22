from __future__ import annotations

from typing import Any, Optional, Union, cast
import zmq
from enum import IntEnum, auto
import numpy.typing as npt
import numpy as np
import sys
import traceback


def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)


CTRL_DT = 0.005
GRIPPER_WIDTH = 0.08


def normalize_pose6d(pose: npt.ArrayLike, name: str = "pose") -> npt.NDArray[np.float64]:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.size != 6:
        raise ValueError(f"{name} expected 6 elements, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr.reshape(6))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf: {arr}")
    return arr


def normalize_joint6(joint_pos: npt.ArrayLike, name: str = "joint_pos") -> npt.NDArray[np.float64]:
    arr = np.asarray(joint_pos, dtype=np.float64)
    if arr.size != 6:
        raise ValueError(f"{name} expected 6 elements, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr.reshape(6))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf: {arr}")
    return arr


def rotm2rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to rotation vector
    """
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        k = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        k = k / (2 * np.sin(theta))
        return theta * k


def rotvec2rotm(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation vector to rotation matrix
    """
    theta = np.linalg.norm(rotvec)
    if np.isclose(theta, 0):
        return np.eye(3)
    else:
        k = rotvec / theta
        K = np.array(
            [
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return R


def rpy2rotm(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert roll-pitch-yaw angles to rotation matrix
    """
    roll, pitch, yaw = rpy
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotm2rpy(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert rotation matrix to roll-pitch-yaw angles
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def ee2tcp(ee_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert SDK eef pose to the runtime pose convention.

    The L5_umi URDF now defines ``eef_link`` at ``das_base_link``. Runtime code
    still uses the historical names ``tcp_pose`` / ``ActualTCPPose``, but those
    poses are now das_base_link poses with rotation stored as a rotvec. The SDK
    reports eef orientation as RPY, so this function only changes orientation
    representation and does not apply the old ARX EE->TCP fixed rotation.
    """
    ee_pose = normalize_pose6d(ee_pose, "ee_pose")
    ee_cartesian = ee_pose[:3]
    ee_rpy = ee_pose[3:]
    ee_rot_mat = rpy2rotm(ee_rpy)
    tcp_rotvec = rotm2rotvec(ee_rot_mat)
    return np.concatenate([ee_cartesian, tcp_rotvec])


def tcp2ee(tcp_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert runtime das_base_link rotvec pose to SDK eef RPY pose."""
    tcp_pose = normalize_pose6d(tcp_pose, "tcp_pose")
    tcp_cartesian = tcp_pose[:3]
    tcp_rotvec = tcp_pose[3:]
    tcp_rot_mat = rotvec2rotm(tcp_rotvec)
    ee_rpy = rotm2rpy(tcp_rot_mat)
    return np.concatenate([tcp_cartesian, ee_rpy])


class Arx5Client:
    def __init__(
        self,
        zmq_ip: str,
        zmq_port: int,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        self.latest_state: dict[str, Union[npt.NDArray[np.float64], float]]
        print(
            f"Arx5Client is connected to {self.zmq_ip}:{self.zmq_port}. Fetching state..."
        )
        self.get_state()
        print(f"Initial state fetched")

    def send_recv(self, msg: dict[str, Any]):
        try:
            self.socket.send_pyobj(msg)
            reply_msg = self.socket.recv_pyobj()
            return reply_msg
        except KeyboardInterrupt:
            print("Arx5Client: KeyboardInterrupt. connection is re-established.")
            return {"cmd": msg["cmd"], "data": "KeyboardInterrupt"}
        except zmq.error.ZMQError:
            # Usually happens when the process is interrupted before receiving reply
            print("Arx5Client: ZMQError. connection is re-established.")
            print(echo_exception())
            self.socket.close()
            del self.socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.zmq_ip}:{self.zmq_port}")
            return {"cmd": msg["cmd"], "data": "ZMQError"}

    def get_state(self):
        reply_msg = self.send_recv({"cmd": "GET_STATE", "data": None})
        assert reply_msg["cmd"] == "GET_STATE"
        assert isinstance(reply_msg["data"], dict)
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def set_ee_pose(
        self, pose_6d: npt.NDArray[np.float64], gripper_pos: Union[float, None] = None
    ):
        pose_6d = normalize_pose6d(pose_6d, "ee_pose")
        reply_msg = self.send_recv(
            {
                "cmd": "SET_EE_POSE",
                "data": {"ee_pose": pose_6d, "gripper_pos": gripper_pos},
            }
        )
        assert reply_msg["cmd"] == "SET_EE_POSE"
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def set_tcp_pose(
        self,
        tcp_pose_6d: npt.NDArray[np.float64],
        gripper_pos: Union[float, None] = None,
    ):
        tcp_pose_6d = normalize_pose6d(tcp_pose_6d, "tcp_pose")
        ee_pose = tcp2ee(tcp_pose_6d)
        return self.set_ee_pose(ee_pose, gripper_pos)

    def set_joint_pos(
        self,
        joint_pos: npt.ArrayLike,
        gripper_pos: Union[float, None] = None,
        duration: float = 0.15,
        max_joint_step_rad: float = 0.05,
    ):
        joint_pos = normalize_joint6(joint_pos)
        reply_msg = self.send_recv(
            {
                "cmd": "SET_JOINT_POS",
                "data": {
                    "joint_pos": joint_pos,
                    "gripper_pos": gripper_pos,
                    "duration": float(duration),
                    "max_joint_step_rad": float(max_joint_step_rad),
                },
            }
        )
        assert reply_msg["cmd"] == "SET_JOINT_POS"
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def reset_to_home(self):
        reply_msg = self.send_recv({"cmd": "RESET_TO_HOME", "data": None})
        assert reply_msg["cmd"] == "RESET_TO_HOME"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")
        self.get_state()

    # [NEW] Bumpless transfer: hold at current physical pose, no motion
    def hold_current_pose(self):
        reply_msg = self.send_recv({"cmd": "HOLD_CURRENT_POSE", "data": None})
        assert reply_msg["cmd"] == "HOLD_CURRENT_POSE"
        if not isinstance(reply_msg["data"], dict):
            raise ValueError(f"Error: {reply_msg['data']}")
        # Sync local state cache
        self.get_state()

    def set_to_damping(self):
        reply_msg = self.send_recv({"cmd": "SET_TO_DAMPING", "data": None})
        assert reply_msg["cmd"] == "SET_TO_DAMPING"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")
        self.get_state()

    def get_gain(self):
        reply_msg = self.send_recv({"cmd": "GET_GAIN", "data": None})
        assert reply_msg["cmd"] == "GET_GAIN"
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        return cast(dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"])

    def set_gain(self, gain: dict[str, Union[npt.NDArray[np.float64], float]]):
        reply_msg = self.send_recv({"cmd": "SET_GAIN", "data": gain})
        assert reply_msg["cmd"] == "SET_GAIN"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")

    @property
    def timestamp(self):
        timestamp = self.latest_state["timestamp"]
        return cast(float, timestamp)

    @property
    def ee_pose(self):
        ee_pose = self.latest_state["ee_pose"]
        return cast(npt.NDArray[np.float64], ee_pose)

    @property
    def joint_pos(self):
        joint_pos = self.latest_state["joint_pos"]
        return cast(npt.NDArray[np.float64], joint_pos)

    @property
    def joint_cmd_pos(self):
        joint_cmd_pos = self.latest_state.get("joint_cmd_pos", self.latest_state["joint_pos"])
        return cast(npt.NDArray[np.float64], joint_cmd_pos)

    @property
    def joint_vel(self):
        joint_vel = self.latest_state["joint_vel"]
        return cast(npt.NDArray[np.float64], joint_vel)

    @property
    def joint_torque(self):
        joint_torque = self.latest_state["joint_torque"]
        return cast(npt.NDArray[np.float64], joint_torque)

    @property
    def gripper_pos(self):
        gripper_pos = self.latest_state["gripper_pos"]
        return cast(float, gripper_pos)

    @property
    def gripper_vel(self):
        gripper_vel = self.latest_state["gripper_vel"]
        return cast(float, gripper_vel)

    @property
    def gripper_torque(self):
        gripper_torque = self.latest_state["gripper_torque"]
        return cast(float, gripper_torque)

    @property
    def tcp_pose(self):
        tcp_pose = np.zeros(6, dtype=np.float64)
        tcp_pose[:] = ee2tcp(self.ee_pose)
        return tcp_pose

    def __del__(self):
        self.socket.close()
        self.context.term()
        print("Arx5Client is closed")
