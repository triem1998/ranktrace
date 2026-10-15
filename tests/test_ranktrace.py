import json

import pytest
import torch

from ranktrace import make_profiler


def _run(p, n=6, device="cpu"):
    with p:
        for _ in range(n):
            with p.region("my_region"):
                torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
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
    assert len(s["step_t"]) == 6 and len(s["system"]["t"]) >= 2  # sampled at start and end


def test_stats_analyzed_in_every_mode(tmp_path):
    _run(make_profiler(None, trace_dir=tmp_path, wait=1, active=2, stats=True))
    out = tmp_path / "analysis" / "stats"
    assert (out / "10_system.html").exists() and (out / "summary.csv").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_memory(tmp_path):
    _run(make_profiler(None, trace_dir=tmp_path, wait=1, active=2, stats=True, memory=True), device="cuda")
    assert {f.name for f in (tmp_path / "stats").glob("memory_rank0.*")} == {"memory_rank0.csv", "memory_rank0.html"}
    out = tmp_path / "analysis" / "stats"
    assert (out / "11_memory.html").exists() and "test_ranktrace.py" in (out / "11_memory_peak.csv").read_text()


def test_bad_mode():
    with pytest.raises(ValueError):
        make_profiler("custom")


def test_analyze_never_raises(tmp_path):
    _run(make_profiler("torch", trace_dir=tmp_path, wait=1, active=2))  # tools may fail on a CPU trace
    assert {f.name for f in (tmp_path / "analysis").glob("*.log")} == {"hta.log", "tracelens.log"}
