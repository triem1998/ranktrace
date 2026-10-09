"""TorchProfiler: torch.profiler trace per rank, for HTA and TraceLens."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from ..utils import run_logged
from .base import Profiler, _rank


class TorchProfiler(Profiler):
    """Writes `trace_dir/rank{r}.pt.trace.json`. torch already labels the step
    (`ProfilerStep#N`), the DataLoader and the optimizer.

    analyze: at the end of the run, rank 0 runs HTA and TraceLens on all ranks' traces
    (separate processes; RAM ~5x the total trace size, GPUs held meanwhile).
    """

    mode = "torch"

    def __init__(self, record_shapes: bool = False, analyze: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.analyze = analyze
        acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        # ROCm: the warmup phase loses all GPU kernels (MI300A, ROCm 10 dev), so skip those steps instead;
        # the recorded steps stay the same
        wait, warmup = (self.wait + self.warmup, 0) if torch.version.hip else (self.wait, self.warmup)
        self._prof = torch.profiler.profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=self.active, repeat=1),
            on_trace_ready=self._save,
            record_shapes=record_shapes,
        )

    @property
    def trace_file(self) -> Path:  # rank read late: the profiler may exist before dist init
        return self.trace_dir / f"rank{_rank()}.pt.trace.json"

    def __enter__(self) -> TorchProfiler:
        self._prof.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._prof.__exit__(*exc)
        super().__exit__(*exc)
        if self.analyze and exc[0] is None:  # not after a crash: other ranks may be gone
            self._analyze()

    def _analyze(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # all ranks' traces are written
        if _rank() != 0:
            return
        out = self.trace_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        for tool in ("hta", "tracelens"):
            t0 = time.perf_counter()
            run_logged([sys.executable, "-m", "ranktrace.analyze", str(self.trace_dir), "--tool", tool],
                       out / f"{tool}.log")
            print(f"[ranktrace] {tool}: {time.perf_counter() - t0:.1f} s -> {out / tool}", flush=True)

    def step(self) -> None:
        super().step()
        self._prof.step()

    @contextmanager
    def region(self, name: str):
        """`record_function` label. Autograd runs on the calling thread inside it, else
        backward kernels fall outside the section."""
        with torch.autograd.set_multithreading_enabled(False), torch.profiler.record_function(name):
            yield

    def _save(self, prof) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(self.trace_file))
