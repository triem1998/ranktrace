"""Per-rank stats written by `stats=True` / `memory=True`: the summary table, plus our figures
10 (CPU / RAM / data read over time) and 11 (GPU memory over time, what holds it at the peak)."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

SYSTEM = {"cpu_main": "CPU % (100 = 1 core)", "cpu_children": "CPU % (100 = 1 core)",
          "ram_main_gb": "RAM GB", "ram_children_gb": "RAM GB", "read_mb_s": "data read MB/s"}
PROCESS = {"main": "training loop", "children": "DataLoader workers"}


def run_stats(trace_dir: Path, out: Path, ranks: list[int] | None = None) -> None:
    """Summary of all ranks; figures for `ranks` only, by default 10 for rank 0 and 11 for ranks
    0 and 1 (rank 0 often does extra work, the others are alike)."""
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(f.read_text()) for f in sorted((trace_dir / "stats").glob("rank*.json"))]
    if rows:
        _summary(rows, out)
        series = pd.concat([pd.DataFrame(r["system"]).assign(rank=r["rank"]) for r in rows])
        pick = lambda rs: series[series["rank"].isin(rs)].astype({"rank": str})
        _system(pick(ranks or [0]), rows[0], out)
        _gpu_memory(pick(ranks or [0, 1]), rows[0], out)
    _memory_peak(sorted((trace_dir / "stats").glob("memory_rank*.csv")), out)
    print(f"stats results in {out}")


def _save(fig, out: Path, name: str, title: str) -> None:
    fig.update_layout(title=f"{title} (drawn by ranktrace from its stats)")
    fig.write_html(out / f"{name}.html", include_plotlyjs="directory")


def _steps(fig, r: dict) -> None:
    """Dotted line at the end of each batch, recorded batches shaded."""
    t = r["step_t"]
    if len(t) <= 200:  # beyond, the lines hide the curves
        for x in t:
            fig.add_vline(x=x, line_dash="dot", line_width=1, line_color="grey")
    first, last = r["active_steps"]
    if len(t) >= last:
        fig.add_vrect(x0=t[first - 2] if first >= 2 else 0, x1=t[last - 1], fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="recorded")


def _summary(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows).drop(columns=["step_s", "step_t", "system"]).set_index("rank")
    df.to_csv(out / "summary.csv")
    print(df.to_string())
    print(f"max GPU device {df.gpu_device_peak_gb.max():.2f} GB | max host RSS {df.host_rss_peak_gb.max():.2f} GB"
          f" | total trace {df.trace_gb.sum():.3f} GB over {len(df)} ranks")


def _system(series: pd.DataFrame, r0: dict, out: Path) -> None:
    """10: CPU, RAM and data read over time, per process (the training loop, its DataLoader
    workers); with several ranks, colour = rank and dash = process."""
    d = series.melt(id_vars=["t", "rank"], value_vars=list(SYSTEM), var_name="series")
    d["metric"] = d.series.map(SYSTEM)
    d["process"] = d.series.map(lambda s: PROCESS.get(s.split("_")[1], "all processes"))
    one = d["rank"].nunique() == 1
    fig = px.line(d, x="t", y="value", color="process" if one else "rank", line_dash=None if one else "process",
                  facet_row="metric", height=900, labels={"t": "s since start", "value": ""})
    fig.update_yaxes(matches=None)
    _steps(fig, r0)
    ranks = ", ".join(sorted(d["rank"].unique()))
    _save(fig, out, "10_system", f"CPU, RAM and data read over time, rank {ranks} ({r0['cpus']} cores per rank)")


def _gpu_memory(series: pd.DataFrame, r0: dict, out: Path) -> None:
    """11: GPU memory allocated by PyTorch over time."""
    if series.gpu_alloc_gb.isna().all():  # no GPU
        return
    fig = px.line(series, x="t", y="gpu_alloc_gb", color="rank", labels={"t": "s since start", "gpu_alloc_gb": "GB"})
    _steps(fig, r0)
    _save(fig, out, "11_memory", f"GPU memory allocated over time, rank {', '.join(sorted(series['rank'].unique()))}")


def _memory_peak(files: list[Path], out: Path) -> None:
    """11: at the peak of the recorded step, GPU memory held per line of user code (top 10 per rank)."""
    tables = []
    for f in files:
        d = pd.read_csv(f)
        total = d["size"].where(d.action != "free_completed", -d["size"]).cumsum()
        live, before = {}, 0  # addr -> (size, site); memory allocated before the recording
        for e in d.iloc[: int(total.idxmax()) + 1].itertuples():  # replay up to the peak
            if e.action == "start":
                before = e.size
            elif e.action == "alloc":
                live[e.addr] = (e.size, e.site)
            elif live.pop(e.addr, None) is None:  # freed, allocated before the recording
                before -= e.size
        held = pd.DataFrame(list(live.values()), columns=["size", "site"]).groupby("site")["size"].sum()
        held["before recording"] = before
        top = held.div(1e9).rename("GB").sort_values(ascending=False).head(10).rename_axis("site").reset_index()
        tables.append(top.assign(rank=f.stem.removeprefix("memory_rank"), peak_gb=total.max() / 1e9))
    if not tables:
        return
    df = pd.concat(tables)
    df.to_csv(out / "11_memory_peak.csv", index=False)
    fig = px.bar(df, x="GB", y="site", color="rank", barmode="group", orientation="h", height=600)
    fig.update_yaxes(autorange="reversed")
    _save(fig, out, "11_memory_peak", f"GPU memory held at the peak ({df.peak_gb.max():.2f} GB), by line of code")
