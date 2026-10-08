"""Trace capture: one profiler class per mode, chosen by `make_profiler`."""
from .base import Profiler
from .nvtx_profiler import NvtxProfiler
from .torch_profiler import TorchProfiler

PROFILERS = {None: Profiler, "torch": TorchProfiler, "nvtx": NvtxProfiler}


def make_profiler(mode: str | None = None, **kwargs) -> Profiler:
    """The profiler for `mode`: None (off), "torch" or "nvtx"."""
    if mode not in PROFILERS:
        raise ValueError(f"mode must be one of {tuple(PROFILERS)}, got {mode!r}")
    return PROFILERS[mode](**kwargs)


# public names (what `from ranktrace.profiler import *` gives)
__all__ = ["PROFILERS", "NvtxProfiler", "Profiler", "TorchProfiler", "make_profiler"]
