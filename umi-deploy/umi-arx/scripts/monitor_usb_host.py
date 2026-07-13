#!/usr/bin/env python3
from __future__ import annotations

"""
Host-side USB monitor for ARX5 + Gen gripper deployments.

The monitor records three signal sources into a timestamped directory:
1. Kernel log follow output (dmesg / journalctl fallback)
2. /dev node snapshots and diffs
3. USB topology snapshots from `lsusb -t`

Use this on the host machine before reproducing a USB disconnect issue.
"""

import argparse
import difflib
import glob
import json
import os
import pathlib
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "data_local" / "usb_monitor"
DEFAULT_DEVICE_GLOBS = [
    "/dev/ttyDevice*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
    "/dev/video*",
    "/dev/v4l/by-id/*",
    "/dev/serial/by-id/*",
]
DMESG_COMMAND_CANDIDATES = [
    ["dmesg", "--follow-new", "--human"],
    ["journalctl", "-kf", "-n", "0", "-o", "short-iso"],
    ["dmesg", "-wT"],
]
DMESG_KEYWORDS = (
    "usb ",
    "usbcore",
    "uvc",
    "video",
    "ttyusb",
    "ttyacm",
    "ttydevice",
    "ch341",
    "cdc_acm",
    "xhci",
    "hub ",
    "slcan",
    "canable",
    "can0",
    "can1",
    "v4l",
)


def now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def shell_join(cmd: List[str]) -> str:
    return shlex.join(cmd)


class EventLogger:
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.events_path = output_dir / "events.log"
        self.dmesg_path = output_dir / "dmesg.log"
        self.lock = threading.Lock()
        self._events_fp = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._dmesg_fp = self.dmesg_path.open("a", encoding="utf-8", buffering=1)

    def close(self):
        with self.lock:
            self._events_fp.close()
            self._dmesg_fp.close()

    def event(self, message: str, echo: bool = True):
        line = f"[{now_iso()}] {message}"
        with self.lock:
            self._events_fp.write(line + "\n")
        if echo:
            print(line, flush=True)

    def event_block(self, title: str, body: str, echo: bool = False):
        header = f"[{now_iso()}] {title}"
        block = header + "\n" + body.rstrip() + "\n"
        with self.lock:
            self._events_fp.write(block)
            if not block.endswith("\n"):
                self._events_fp.write("\n")
        if echo:
            print(block, end="", flush=True)
        else:
            print(header, flush=True)

    def dmesg(self, line: str):
        with self.lock:
            self._dmesg_fp.write(line.rstrip("\n") + "\n")


def run_command(
    cmd: List[str], timeout: float = 5.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def capture_command_text(cmd: List[str]) -> str:
    try:
        result = run_command(cmd)
    except FileNotFoundError:
        return f"$ {shell_join(cmd)}\ncommand not found\n"
    except subprocess.TimeoutExpired:
        return f"$ {shell_join(cmd)}\ncommand timed out\n"

    text = f"$ {shell_join(cmd)}\n"
    if result.stdout:
        text += result.stdout
        if not text.endswith("\n"):
            text += "\n"
    if result.stderr:
        text += "[stderr]\n" + result.stderr
        if not text.endswith("\n"):
            text += "\n"
    if result.returncode != 0 and not result.stderr:
        text += f"[exit_code] {result.returncode}\n"
    return text


def file_type_from_mode(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISCHR(mode):
        return "char"
    if stat.S_ISBLK(mode):
        return "block"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def snapshot_entry(path_str: str) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "path": path_str,
        "lexists": os.path.lexists(path_str),
        "exists": os.path.exists(path_str),
    }
    if not entry["lexists"]:
        return entry

    lstat_result = os.lstat(path_str)
    entry["type"] = file_type_from_mode(lstat_result.st_mode)
    entry["mode"] = stat.filemode(lstat_result.st_mode)

    if entry["type"] == "symlink":
        try:
            entry["target"] = os.readlink(path_str)
        except OSError as e:
            entry["target"] = f"<readlink failed: {e}>"

    try:
        real_path = os.path.realpath(path_str)
    except OSError as e:
        real_path = f"<realpath failed: {e}>"
    entry["real_path"] = real_path
    entry["real_path_exists"] = os.path.exists(real_path) if isinstance(real_path, str) else False

    if entry["exists"]:
        stat_result = os.stat(path_str)
        entry["inode"] = stat_result.st_ino
        if stat.S_ISCHR(stat_result.st_mode) or stat.S_ISBLK(stat_result.st_mode):
            entry["major"] = os.major(stat_result.st_rdev)
            entry["minor"] = os.minor(stat_result.st_rdev)

    return entry


def collect_device_snapshot(patterns: List[str]) -> Dict[str, object]:
    matches_by_pattern: Dict[str, List[str]] = {}
    matched_paths: List[str] = []

    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and not glob.has_magic(pattern):
            matches = [pattern]
        matches_by_pattern[pattern] = matches
        matched_paths.extend(matches)

    unique_paths = sorted(set(matched_paths))
    entries = {path: snapshot_entry(path) for path in unique_paths}
    return {
        "patterns": patterns,
        "matches_by_pattern": matches_by_pattern,
        "entries": entries,
    }


def snapshot_json(snapshot: Dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)


def snapshot_text(snapshot: Dict[str, object]) -> str:
    lines = []
    entries: Dict[str, Dict[str, object]] = snapshot["entries"]  # type: ignore[assignment]
    matches_by_pattern: Dict[str, List[str]] = snapshot["matches_by_pattern"]  # type: ignore[assignment]

    lines.append("Patterns:")
    for pattern, matches in matches_by_pattern.items():
        lines.append(f"  {pattern}")
        if matches:
            for match in matches:
                lines.append(f"    - {match}")
        else:
            lines.append("    - <no matches>")

    lines.append("")
    lines.append("Entries:")
    if not entries:
        lines.append("  <no matched entries>")
    else:
        for path, entry in entries.items():
            lines.append(f"  {path}")
            for key in sorted(entry.keys()):
                if key == "path":
                    continue
                lines.append(f"    {key}: {entry[key]}")
    return "\n".join(lines) + "\n"


def write_text(path: pathlib.Path, text: str):
    path.write_text(text, encoding="utf-8")


def is_usb_related_dmesg(line: str) -> bool:
    normalized = line.lower()
    return any(keyword in normalized for keyword in DMESG_KEYWORDS)


class DmesgFollower:
    def __init__(self, logger: EventLogger, output_dir: pathlib.Path):
        self.logger = logger
        self.output_dir = output_dir
        self.proc: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.command: Optional[List[str]] = None

    def start(self):
        for cmd in DMESG_COMMAND_CANDIDATES:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                self.logger.event(f"Kernel log command not found: {shell_join(cmd)}")
                continue

            time.sleep(0.2)
            if proc.poll() is not None:
                output, _ = proc.communicate(timeout=1.0)
                message = output.strip() or f"exit code {proc.returncode}"
                self.logger.event(
                    f"Kernel log command failed: {shell_join(cmd)} | {message}"
                )
                continue

            self.proc = proc
            self.command = cmd
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            self.logger.event(f"Started kernel log monitor: {shell_join(cmd)}")
            return

        self.logger.event(
            "Kernel log monitor is unavailable. Run this script with sudo if you need dmesg/journalctl access."
        )

    def stop(self):
        self.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _reader_loop(self):
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.stop_event.is_set():
                break
            self.logger.dmesg(line)
            if is_usb_related_dmesg(line):
                self.logger.event(f"DMESG {line.rstrip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record /dev, dmesg, and lsusb changes while reproducing USB issues."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_ROOT / now_compact(),
        help="Directory for logs and snapshots.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between /dev snapshots.",
    )
    parser.add_argument(
        "--device-glob",
        action="append",
        default=None,
        help="Additional glob to monitor. Can be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    patterns = list(DEFAULT_DEVICE_GLOBS)
    if args.device_glob:
        patterns.extend(args.device_glob)

    logger = EventLogger(output_dir)
    stop_event = threading.Event()
    dmesg_follower = DmesgFollower(logger, output_dir)

    def handle_signal(signum, _frame):
        logger.event(f"Received signal {signum}, stopping monitor.")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        write_text(
            output_dir / "config.json",
            json.dumps(
                {
                    "started_at": now_iso(),
                    "cwd": os.getcwd(),
                    "argv": sys.argv,
                    "patterns": patterns,
                    "poll_interval": args.poll_interval,
                    "uid": os.getuid(),
                    "euid": os.geteuid(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

        logger.event(f"Writing monitor output to {output_dir}")
        logger.event(
            "Monitoring device globs: " + ", ".join(patterns)
        )

        initial_snapshot = collect_device_snapshot(patterns)
        initial_json = snapshot_json(initial_snapshot)
        write_text(snapshot_dir / "dev_000_initial.json", initial_json + "\n")
        write_text(snapshot_dir / "dev_000_initial.txt", snapshot_text(initial_snapshot))
        write_text(snapshot_dir / "lsusb_t_000_initial.txt", capture_command_text(["lsusb", "-t"]))
        logger.event("Captured initial /dev and lsusb snapshots.")

        dmesg_follower.start()

        previous_snapshot = initial_snapshot
        previous_json = initial_json
        change_index = 0

        while not stop_event.is_set():
            time.sleep(args.poll_interval)
            current_snapshot = collect_device_snapshot(patterns)
            current_json = snapshot_json(current_snapshot)
            if current_json == previous_json:
                continue

            change_index += 1
            tag = f"{change_index:03d}_{now_compact()}"
            dev_json_path = snapshot_dir / f"dev_{tag}.json"
            dev_txt_path = snapshot_dir / f"dev_{tag}.txt"
            lsusb_path = snapshot_dir / f"lsusb_t_{tag}.txt"
            diff_path = snapshot_dir / f"dev_diff_{tag}.txt"

            write_text(dev_json_path, current_json + "\n")
            write_text(dev_txt_path, snapshot_text(current_snapshot))
            write_text(lsusb_path, capture_command_text(["lsusb", "-t"]))

            diff_lines = list(
                difflib.unified_diff(
                    previous_json.splitlines(),
                    current_json.splitlines(),
                    fromfile="previous",
                    tofile="current",
                    lineterm="",
                )
            )
            diff_text = "\n".join(diff_lines) + ("\n" if diff_lines else "")
            write_text(diff_path, diff_text)

            logger.event(
                f"Detected /dev change #{change_index}. "
                f"snapshot={dev_json_path.name}, diff={diff_path.name}, lsusb={lsusb_path.name}"
            )
            if diff_text:
                logger.event_block("Device diff", diff_text, echo=False)

            previous_snapshot = current_snapshot
            previous_json = current_json

        logger.event("USB monitor stopped.")
    finally:
        dmesg_follower.stop()
        logger.close()


if __name__ == "__main__":
    main()
