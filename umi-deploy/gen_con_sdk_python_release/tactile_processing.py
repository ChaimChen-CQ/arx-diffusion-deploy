"""Helpers for reshaping 448-byte tactile packets into 500-value arrays."""

import queue
import sys
import threading
import time
from typing import List, Optional, Tuple

# Whether to print tactile grid to stdout: 1000 values (500 left + 500 right), 50 rows, each row left 10 + 10-space gap + right 10
_tactile_grid_print_enabled: bool = False
# After a frame was printed, emit a blank line before the next frame
_tactile_grid_blank_before_next_frame: bool = False
# Minimum interval (s) between prints in the print thread; 0 = unlimited (still one print per full block)
_tactile_print_min_interval_sec: float = 0.0

# Background print: receive thread only enqueues so print does not block serial/parse thread
_print_queue: queue.Queue = queue.Queue(maxsize=1)
_tactile_print_thread: Optional[threading.Thread] = None
_tactile_print_thread_lock = threading.Lock()


def get_tactile_grid_print_enabled() -> bool:
    """Return whether tactile grid printing to stdout is enabled."""
    return _tactile_grid_print_enabled


def set_tactile_grid_print_enabled(enabled: bool) -> None:
    """Enable or disable tactile grid printing; disabling resets frame spacing state."""
    global _tactile_grid_print_enabled, _tactile_grid_blank_before_next_frame
    _tactile_grid_print_enabled = enabled
    if not enabled:
        _tactile_grid_blank_before_next_frame = False
        try:
            while True:
                _print_queue.get_nowait()
        except queue.Empty:
            pass


def set_tactile_grid_print_max_hz(max_hz: float) -> None:
    """
    Limit how often the background printer flushes a grid (reduces stdout load).

    max_hz <= 0: no limit (still non-blocking for the tactile callback).
    """
    global _tactile_print_min_interval_sec
    if max_hz is None or max_hz <= 0:
        _tactile_print_min_interval_sec = 0.0
    else:
        _tactile_print_min_interval_sec = 1.0 / float(max_hz)


def _build_tactile_1000_grid_text(all_tactile: List[int]) -> str:
    """
    Format 1000 values as 50 lines: row i shows left pad row i (10 cells), 10 blank chars,
    then right pad row i (10 cells). all_tactile[:500] = left, all_tactile[500:] = right.
    """
    global _tactile_grid_blank_before_next_frame
    if len(all_tactile) != 1000:
        raise ValueError(f"Expected 1000 tactile values, got {len(all_tactile)}")
    parts: List[str] = []
    if _tactile_grid_blank_before_next_frame:
        parts.append("")
    gap = " " * 10
    for row in range(50):
        lo = row * 10
        left_line = all_tactile[lo : lo + 10]
        right_line = all_tactile[500 + lo : 500 + lo + 10]
        left_s = " ".join(f"{x:3d}" for x in left_line)
        right_s = " ".join(f"{x:3d}" for x in right_line)
        parts.append(f"{left_s}{gap}{right_s}")
    _tactile_grid_blank_before_next_frame = True
    return "\n".join(parts)


def _tactile_print_worker() -> None:
    next_deadline = 0.0
    while True:
        data = _print_queue.get()
        while True:
            try:
                data = _print_queue.get_nowait()
            except queue.Empty:
                break
        if _tactile_print_min_interval_sec > 0:
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
                while True:
                    try:
                        data = _print_queue.get_nowait()
                    except queue.Empty:
                        break
        try:
            if _tactile_grid_print_enabled:
                block = _build_tactile_1000_grid_text(data)
                print(block, flush=True)
        except Exception as e:
            print(f"[tactile_grid_print] {e}", file=sys.stderr, flush=True)
        if _tactile_print_min_interval_sec > 0:
            next_deadline = time.monotonic() + _tactile_print_min_interval_sec


def _ensure_tactile_print_thread() -> None:
    global _tactile_print_thread
    with _tactile_print_thread_lock:
        if _tactile_print_thread is not None and _tactile_print_thread.is_alive():
            return
        _tactile_print_thread = threading.Thread(
            target=_tactile_print_worker,
            name="tactile_grid_print",
            daemon=True,
        )
        _tactile_print_thread.start()


def submit_tactile_1000_grid_print(all_tactile: List[int]) -> None:
    """
    Hand one frame of 1000 tactile values to a background thread for printing (50 lines: each line left 10 + gap 10 + right 10); returns immediately on the caller thread.

    No-op if printing is disabled. Queue holds only the latest frame: a new frame replaces the pending one if the background has not finished yet.
    """
    if not _tactile_grid_print_enabled:
        return
    if len(all_tactile) != 1000:
        return
    _ensure_tactile_print_thread()
    payload = list(all_tactile)
    try:
        _print_queue.put_nowait(payload)
    except queue.Full:
        try:
            _print_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _print_queue.put_nowait(payload)
        except queue.Full:
            pass


def print_tactile_1000_grid(all_tactile: List[int]) -> None:
    """
    Print 1000 tactile values as 50 lines: each line is left row (10) + gap (10 spaces) + right row (10).

    No-op if printing is disabled via set_tactile_grid_print_enabled(False).
    Uses a single print() to reduce lock/syscall overhead versus line-by-line printing.
    """
    if not _tactile_grid_print_enabled:
        return
    print(_build_tactile_1000_grid_text(all_tactile), flush=True)


LEFT_NEG_COORDS = [
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0),
    (0, 7), (0, 8), (0, 9), (1, 8), (1, 9), (2, 9),
    (49, 0), (49, 1), (49, 2), (49, 3),
    (48, 0), (48, 1), (48, 2), (48, 3),
    (47, 0), (47, 1), (47, 2),
    (46, 0), (46, 1), (46, 2),
    (45, 0), (45, 1), (45, 2),
    (44, 0), (44, 1),
    (43, 0),
    (49, 6), (49, 7), (49, 8), (49, 9),
    (48, 6), (48, 7), (48, 8), (48, 9),
    (47, 7), (47, 8), (47, 9),
    (46, 7), (46, 8), (46, 9),
    (45, 7), (45, 8), (45, 9),
    (44, 8), (44, 9),
    (43, 9),
]

RIGHT_NEG_COORDS = [
    (50, 0), (50, 1), (50, 2), (51, 0), (51, 1), (52, 0),
    (50, 7), (50, 8), (50, 9), (51, 8), (51, 9), (52, 9),
    (99, 0), (99, 1), (99, 2), (99, 3),
    (98, 0), (98, 1), (98, 2), (98, 3),
    (97, 0), (97, 1), (97, 2),
    (96, 0), (96, 1), (96, 2),
    (95, 0), (95, 1), (95, 2),
    (94, 0), (94, 1),
    (93, 0),
    (99, 6), (99, 7), (99, 8), (99, 9),
    (98, 6), (98, 7), (98, 8), (98, 9),
    (97, 7), (97, 8), (97, 9),
    (96, 7), (96, 8), (96, 9),
    (95, 7), (95, 8), (95, 9),
    (94, 8), (94, 9),
    (93, 9),
]


def _duplicate_samples(values: List[int]) -> List[int]:
    expanded = []
    for value in values:
        expanded.append(value)
        expanded.append(value)
    return expanded


def _to_signed_int8(values: List[int]) -> List[int]:
    return [value if value == -1 else (value if value < 128 else value - 256) for value in values]


def convert_tactile_448_to_1000(record_data: bytes) -> Tuple[List[int], List[int]]:
    """Convert one 448-byte tactile packet into left/right 500-value arrays."""
    if len(record_data) != 448:
        raise ValueError(f"Expected 448 tactile bytes, got {len(record_data)}")

    raw_left_224 = list(record_data[:224])
    raw_right_224 = list(record_data[224:])

    left_expanded_448 = _duplicate_samples(raw_left_224)
    right_expanded_448 = _duplicate_samples(raw_right_224)

    total_grid = [[0 for _ in range(10)] for _ in range(100)]

    for row, col in LEFT_NEG_COORDS:
        total_grid[row][col] = -1
    for row, col in RIGHT_NEG_COORDS:
        total_grid[row][col] = -1

    left_idx = 0
    for row in range(50):
        for col in range(10):
            if total_grid[row][col] != -1 and left_idx < len(left_expanded_448):
                total_grid[row][col] = left_expanded_448[left_idx]
                left_idx += 1

    right_idx = 0
    for row in range(50, 100):
        for col in range(10):
            if total_grid[row][col] != -1 and right_idx < len(right_expanded_448):
                total_grid[row][col] = right_expanded_448[right_idx]
                right_idx += 1

    left_flat = []
    for row in range(50):
        left_flat.extend(total_grid[row])

    right_flat = []
    for row in range(50, 100):
        right_flat.extend(total_grid[row])

    return _to_signed_int8(left_flat), _to_signed_int8(right_flat)
