"""Profiler: no profiling, plus what every mode shares (schedule window, step stats)."""
from __future__ import annotations

import json
import resource
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class Profiler:
    """No profiling (mode None). Same schedule as torch.profiler: skip `wait` steps, warm up
    `warmup` steps (profiled, discarded), then record `active` steps.

    stats: write `trace_dir/stats/rank{r}.json` on exit (step times, peak GPU / host memory,
    trace size), also when the run crashes.
    """

    mode: str | None = None
    trace_file: Path | None = None  # set by modes that write a trace themselves

    def __init__(self, trace_dir: str = "traces", wait: int = 5, warmup: int = 1, active: int = 3,
                 stats: bool = False):
        self.trace_dir = Path(trace_dir)
        self.wait, self.warmup, self.active = wait, warmup, active
        self.stats = stats
        self._n = 0
        self._t: list[float] = []
        self._dev_peak = 0
        self._alloc_peak = 0

    @property
    def _first(self) -> int:
        return self.wait + self.warmup + 1

    @property
    def _last(self) -> int:
        return self.wait + self.warmup + self.active

    def __enter__(self) -> Profiler:
        return self

    def __exit__(self, *exc) -> None:
        if self.stats:
            self._write_stats()

    def step(self) -> None:
        """Call once per batch."""
        self._n += 1
        if self.stats:
            self._t.append(time.perf_counter())
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                self._dev_peak = max(self._dev_peak, total - free)
                # read per step: callers may reset the peak counter between steps
                self._alloc_peak = max(self._alloc_peak, torch.cuda.max_memory_allocated())

    @contextmanager
    def region(self, name: str):
        """Optional section label; no-op here."""
        yield

    def _write_stats(self) -> None:
        dt = [b - a for a, b in zip(self._t, self._t[1:])]
        trace = self.trace_file
        s = {
            "rank": _rank(), "mode": self.mode, "steps": self._n,
            "active_steps": [self._first, self._last],  # 1-based, inclusive
            "s_per_step_median": statistics.median(dt) if dt else None,
            "step_s": [round(d, 4) for d in dt],
            "gpu_alloc_peak_gb": self._alloc_peak / 1e9,
            "gpu_device_peak_gb": self._dev_peak / 1e9,
            "host_rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
            "trace_gb": trace.stat().st_size / 1e9 if trace and trace.exists() else 0.0,
        }
        # not next to the traces: HTA loads every .json in trace_dir as a trace
        (self.trace_dir / "stats").mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "stats" / f"rank{_rank()}.json").write_text(json.dumps(s, indent=1))
