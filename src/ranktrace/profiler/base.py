"""Profiler: no profiling, plus what every mode shares (schedule window, stats, GPU memory
recording, analysis at the end)."""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ..utils import run_logged
from . import memory
from .sampler import Sampler


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class Profiler:
    """No profiling (mode None). Same schedule as torch.profiler: skip `wait` steps, warm up
    `warmup` steps (profiled, discarded), then record `active` steps.

    stats: write `trace_dir/stats/rank{r}.json` on exit (step times, peak GPU / host memory,
    trace size, CPU / RAM / data read / GPU memory every second), also when the run crashes.
    memory: record the GPU allocations of the last recorded step (`stats/memory_rank{r}.html`
    and `.csv`); slows that step a little.
    analyze: at the end of the run, rank 0 analyzes all ranks' outputs (see `_tools`).
    """

    mode: str | None = None
    trace_file: Path | None = None  # set by modes that write a trace themselves

    def __init__(self, trace_dir: str = "traces", wait: int = 5, warmup: int = 1, active: int = 3,
                 stats: bool = False, memory: bool = False, analyze: bool = True):
        self.trace_dir = Path(trace_dir)
        self.wait, self.warmup, self.active = wait, warmup, active
        self.stats, self.memory, self.analyze = stats, memory, analyze
        self._n = 0
        self._t: list[float] = []
        self._dev_peak = 0
        self._alloc_peak = 0
        self._t0 = self._wall0 = 0.0
        self._sampler: Sampler | None = None
        self._mem_base: int | None = None  # set while the GPU memory is recorded

    @property
    def _first(self) -> int:
        return self.wait + self.warmup + 1

    @property
    def _last(self) -> int:
        return self.wait + self.warmup + self.active

    def _tools(self) -> list[str]:
        """`ranktrace.analyze` tools run at the end of the run."""
        return ["stats"] if self.stats or self.memory else []

    def __enter__(self) -> Profiler:
        self._t0, self._wall0 = time.perf_counter(), time.time()
        if self.stats:
            self._sampler = Sampler(self._t0)
            self._sampler.start()
        self._record_memory()
        return self

    def __exit__(self, *exc) -> None:
        if self._sampler:
            self._sampler.stop()
        if self._mem_base is not None:  # ended mid-window
            self._save_memory()
        if self.stats:
            self._write_stats()
        if self.analyze and exc[0] is None and self._tools():  # not after a crash: other ranks may be gone
            self._analyze()

    def step(self) -> None:
        """Call once per batch."""
        self._n += 1
        if self.stats:
            self._t.append(time.perf_counter())
            if torch.cuda.is_available():
                self._sampler.device = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info()
                self._dev_peak = max(self._dev_peak, total - free)
                # read per step: callers may reset the peak counter between steps
                self._alloc_peak = max(self._alloc_peak, torch.cuda.max_memory_allocated())
        self._record_memory()

    @contextmanager
    def region(self, name: str):
        """Optional section label; no-op here."""
        yield

    def _record_memory(self) -> None:
        """GPU allocations of the last recorded batch: between step() calls `_last - 1` and `_last`."""
        if not (self.memory and torch.cuda.is_available()):
            return
        if self._n == self._last - 1:
            self._mem_base = memory.start()
        elif self._n == self._last and self._mem_base is not None:
            self._save_memory()

    def _save_memory(self) -> None:
        memory.save(self.trace_dir / "stats" / f"memory_rank{_rank()}", self._mem_base, self._wall0)
        self._mem_base = None

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
            "cpus": len(os.sched_getaffinity(0)),  # cores this rank may use
            "step_t": [round(t - self._t0, 3) for t in self._t],  # end of each batch, same clock as `system`
            "system": self._sampler.data,
        }
        # not next to the traces: HTA loads every .json in trace_dir as a trace
        (self.trace_dir / "stats").mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "stats" / f"rank{_rank()}.json").write_text(json.dumps(s))

    def _analyze(self) -> None:
        # CPU (gloo) barriers with a long timeout: the analysis takes minutes, longer than NCCL's
        # timeout, which would abort the other ranks waiting for rank 0 in their next collective
        group = None
        if dist.is_available() and dist.is_initialized():
            group = dist.new_group(backend="gloo", timeout=timedelta(hours=6))
            dist.barrier(group=group)  # all ranks' outputs are written
        if _rank() == 0:
            out = self.trace_dir / "analysis"
            out.mkdir(parents=True, exist_ok=True)
            for tool in self._tools():
                t0 = time.perf_counter()
                run_logged([sys.executable, "-m", "ranktrace.analyze", str(self.trace_dir), "--tool", tool],
                           out / f"{tool}.log")
                print(f"[ranktrace] {tool}: {time.perf_counter() - t0:.1f} s -> {out / tool}", flush=True)
        if group is not None:
            dist.barrier(group=group)  # the other ranks wait here for rank 0
            dist.destroy_process_group(group)
