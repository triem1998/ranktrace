"""NvtxProfiler: NVTX ranges and a capture window, for nsys."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .base import Profiler


class NvtxProfiler(Profiler):
    """cudaProfilerStart/Stop around the recorded steps and one `step` NVTX range per
    recorded batch (like torch's `ProfilerStep#N`). Run under
    `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop`."""

    mode = "nvtx"

    def __exit__(self, *exc) -> None:
        if self._first - 1 <= self._n < self._last:  # ended mid-window
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
        super().__exit__(*exc)

    def step(self) -> None:
        super().step()
        if self._first <= self._n <= self._last:
            torch.cuda.nvtx.range_pop()
        if self._n == self._first - 1:
            torch.cuda.profiler.start()
        elif self._n == self._last:
            torch.cuda.profiler.stop()
        if self._first - 1 <= self._n < self._last:
            torch.cuda.nvtx.range_push("step")

    @contextmanager
    def region(self, name: str):
        """NVTX range. Autograd runs on the calling thread inside it, else backward
        kernels fall outside the section."""
        with torch.autograd.set_multithreading_enabled(False), torch.cuda.nvtx.range(name):
            yield
