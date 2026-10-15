"""Sampler: CPU, RAM and data read of the training process and its children (DataLoader and
compile workers), plus its GPU memory, every second."""
from __future__ import annotations

import threading
import time

import psutil
import torch

SERIES = ("t", "cpu_main", "cpu_children", "ram_main_gb", "ram_children_gb", "read_mb_s", "gpu_alloc_gb")


class Sampler(threading.Thread):
    """Background thread. `data` holds one list per series; `t` in s since `t0` (perf_counter),
    CPU in % of one core, read = bytes of `read()` calls (memory-mapped reads are not counted)."""

    def __init__(self, t0: float, interval: float = 1.0):
        super().__init__(daemon=True)
        self.t0, self.interval = t0, interval
        self.device: int | None = None  # set by the training thread (this thread's device is GPU 0)
        self.data: dict[str, list] = {k: [] for k in SERIES}
        self._done = threading.Event()
        self._me = psutil.Process()
        self._prev: dict[int, tuple[float, int]] = {}  # pid -> (CPU s, bytes read)
        self._prev_t: float | None = None

    def run(self) -> None:
        while not self._done.is_set():
            self._sample()
            self._done.wait(self.interval)
        self._sample()

    def stop(self) -> None:
        self._done.set()
        self.join()

    def _sample(self) -> None:
        now = time.perf_counter()
        cpu, ram, read, cur = [0.0, 0.0], [0, 0], 0, {}
        for p in [self._me, *self._me.children(recursive=True)]:
            try:
                with p.oneshot():
                    c, rss, r = p.cpu_times(), p.memory_info().rss, p.io_counters().read_chars
            except psutil.Error:  # exited meanwhile
                continue
            child = p.pid != self._me.pid
            cur[p.pid] = (c.user + c.system, r)
            prev = self._prev.get(p.pid, cur[p.pid])  # new process: no difference yet
            cpu[child] += cur[p.pid][0] - prev[0]
            read += cur[p.pid][1] - prev[1]
            ram[child] += rss
        dt = now - self._prev_t if self._prev_t else None
        self._prev, self._prev_t = cur, now
        gpu = torch.cuda.memory_allocated(self.device) / 1e9 if self.device is not None else None
        for k, v in zip(SERIES, (now - self.t0, *(100 * c / dt if dt else None for c in cpu),
                                 ram[0] / 1e9, ram[1] / 1e9, read / dt / 1e6 if dt else None, gpu)):
            self.data[k].append(None if v is None else round(v, 3))
