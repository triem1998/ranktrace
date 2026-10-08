import json

import pytest
import torch

from ranktrace import make_profiler


def _run(p, n=6):
    with p:
        for _ in range(n):
            with p.region("my_region"):
                torch.randn(64, 64) @ torch.randn(64, 64)
            p.step()


def test_torch_writes_one_trace(tmp_path):
    _run(make_profiler("torch", trace_dir=tmp_path, wait=1, active=2, stats=True, analyze=False))
    files = list(tmp_path.glob("*.json"))  # HTA reads every .json here
    assert [f.name for f in files] == ["rank0.pt.trace.json"]
    assert "my_region" in files[0].read_text()


def test_off_writes_nothing(tmp_path):
    _run(make_profiler(None, trace_dir=tmp_path))
    assert not any(tmp_path.iterdir())


def test_stats(tmp_path):
    _run(make_profiler("torch", trace_dir=tmp_path, wait=1, active=2, stats=True, analyze=False))
    s = json.loads((tmp_path / "stats" / "rank0.json").read_text())
    assert s["steps"] == 6 and len(s["step_s"]) == 5 and s["active_steps"] == [3, 4]
    assert s["trace_gb"] > 0 and s["host_rss_peak_gb"] > 0


def test_bad_mode():
    with pytest.raises(ValueError):
        make_profiler("custom")


def test_analyze_never_raises(tmp_path):
    _run(make_profiler("torch", trace_dir=tmp_path, wait=1, active=2))  # tools may fail on a CPU trace
    assert {f.name for f in (tmp_path / "analysis").glob("*.log")} == {"hta.log", "tracelens.log"}
