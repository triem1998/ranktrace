"""GPU memory of one step from PyTorch's allocation history: PyTorch's viewer as an HTML file,
plus every allocation and free with the line of user code behind it (for our peak table)."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.cuda import _memory_viz

MAX_ENTRIES = 1_000_000  # allocator events kept; the oldest are dropped beyond


def start() -> int:
    """Start recording; returns the memory already allocated (no line of code for it)."""
    torch.cuda.memory._record_memory_history(stacks="python", max_entries=MAX_ENTRIES)
    return torch.cuda.memory_allocated()


def save(stem: Path, base: int, wall0: float) -> None:
    """Stop recording; write `stem.html` (PyTorch's viewer) and `stem.csv` (t_s since `wall0`,
    action, addr, size, site)."""
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".html").write_text(_memory_viz.trace_plot(snap))
    events = snap["device_traces"][torch.cuda.current_device()]
    t_start = events[0]["time_us"] / 1e6 - wall0 if events else 0.0
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "action", "addr", "size", "site"])
        w.writerow([round(t_start, 6), "start", 0, base, "before recording"])
        for e in events:
            if e["action"] in ("alloc", "free_completed"):
                w.writerow([round(e["time_us"] / 1e6 - wall0, 6), e["action"], e["addr"], e["size"],
                            _site(e["frames"])])


def _site(frames: list[dict]) -> str:
    """Innermost frame of user code (frames are innermost first)."""
    for fr in frames:
        fn = fr["filename"]
        if not ("/torch/" in fn or "torchinductor" in fn or fn.startswith("<")):
            return f"{Path(fn).name}:{fr['line']} {fr['name']}"
    return "torch internals"
