import time

import click
import zmq


def request(socket: zmq.Socket, msg: dict):
    socket.send_pyobj(msg)
    return socket.recv_pyobj()


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True)
@click.option("--dx", default=0.01, show_default=True, help="X offset in meters.")
@click.option("--reset/--no-reset", default=True, show_default=True, help="Reset to home before sending the step.")
def main(host: str, port: int, dx: float, reset: bool):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    socket.connect(f"tcp://{host}:{port}")

    print(request(socket, {"cmd": "PING"}))
    if reset:
        print(request(socket, {"cmd": "RESET_TO_HOME"}))

    state_reply = request(socket, {"cmd": "GET_STATE"})
    print(state_reply)
    state = state_reply["data"]
    pose = state["ee_pose"].copy()
    target = pose.copy()
    target[0] += dx

    print("current:", pose)
    print("target:", target)

    move_reply = request(
        socket,
        {
            "cmd": "SET_EE_POSE",
            "data": {
                "ee_pose": target,
                "gripper_pos": state["gripper_pos"],
            },
        },
    )
    print(move_reply)

    time.sleep(0.5)
    print(request(socket, {"cmd": "GET_STATE"}))


if __name__ == "__main__":
    main()
