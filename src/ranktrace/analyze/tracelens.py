"""TraceLens on torch traces: its tables (it makes no figures), plus our figures of them."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.express as px

from ..utils import run_logged


def run_tracelens(trace_dir: Path, out: Path, jobs: int) -> None:
    traces = sorted(trace_dir.glob("rank*.pt.trace.json"))
    out.mkdir(parents=True, exist_ok=True)
    report = lambda t: run_logged(  # one process per rank, ~13x the trace size in RAM
        [sys.executable, "-m", "TraceLens.Reporting.generate_perf_report_pytorch", "--profile_json_path", str(t),
         "--output_csvs_dir", str(out / t.name.split(".")[0])], out / f"{t.name.split('.')[0]}.log")
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(report, traces))
    if len(traces) > 1:  # collectives across ranks: needs all ranks
        run_logged([sys.executable, "-m", "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch",
                    "--trace_pattern", str(trace_dir / "rank*.pt.trace.json"), "--world_size", str(len(traces)),
                    "--output_csvs_dir", str(out / "collectives")], out / "collectives.log")
    plot_tracelens(out)
    print(f"TraceLens results in {out}")


def plot_tracelens(out: Path) -> None:
    """Our figures, drawn from TraceLens tables as they are (TraceLens makes none), numbered as
    the HTA outputs answering the same question."""
    (out / "figures").mkdir(exist_ok=True)

    def read(table):  # one TraceLens table, all ranks stacked
        fs = sorted(out.glob(f"rank*/{table}.csv"), key=lambda f: int(f.parent.name[4:]))
        return pd.concat([pd.read_csv(f).assign(rank=f.parent.name) for f in fs]) if fs else None

    def save(fig, name, table):
        fig.update_layout(title=f"{name} (drawn by ranktrace from TraceLens {table})")
        fig.write_html(out / "figures" / f"{name}.html", include_plotlyjs="directory")

    if (tl := read("gpu_timeline")) is not None:
        tl = tl[tl.type.isin(["computation_time", "exposed_comm_time", "exposed_memcpy_time", "idle_time"])]
        save(px.bar(tl, x="rank", y="time ms", color="type"), "02_gpu_timeline", "gpu_timeline.csv")
    if (cat := read("ops_summary_by_category")) is not None:
        save(px.bar(cat, x="op category", y="total_direct_kernel_time_ms", color="rank", barmode="group"),
             "04_ops_by_category", "ops_summary_by_category.csv")
    if (f := out / "collectives" / "nccl_summary_implicit_sync.csv").exists():
        c = pd.read_csv(f)  # comm_latency = transfer, skew in start time = wait for the latest rank (us)
        save(px.bar(c, x="Collective name", y=["comm_latency_mean", "skew in start time_mean"], barmode="group",
                    labels={"value": "us"}), "09_collectives", f.name)
    if (f := out / "collectives" / "straggler_summary.csv").exists():
        s = pd.read_csv(f).astype({"rank": str})  # slow rank: least wait, highest pct_arrived_last
        save(px.bar(s, x="rank", y="total_wait_time_us", text="pct_arrived_last"), "06_stragglers", f.name)
