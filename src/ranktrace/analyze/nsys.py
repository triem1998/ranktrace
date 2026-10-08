"""nsys recipes on `.nsys-rep` files: each recipe's table as CSV, plus our figure of it."""
from pathlib import Path

import pandas as pd
import plotly.express as px

from ..utils import run_logged

# (number as the HTA output answering the same question, recipe, args, its table, our figure);
# default thresholds hide everything on a busy GPU
NSYS_RECIPES = [
    ("01", "nvtx_gpu_proj_sum", [], "rank_stats",  # GPU time ("Sum") and CPU time per region, one panel per rank
     lambda d: px.bar(d.rename(columns={"Sum": "GPU time", "Sum of NVTX Ranges on CPU": "CPU time"}),
                      x="Text", y=["GPU time", "CPU time"], barmode="group", facet_col="Rank",
                      facet_col_wrap=4, labels={"value": "ns"})),
    ("02", "gpu_time_util", ["--threshold", "100"], "analysis", lambda d: px.bar(d, x="Rank", y="In-Use")),
    ("03", "gpu_gaps", ["--gap", "1"], "analysis", lambda d: px.scatter(d, x="Start", y="Duration", color="Rank")),
    ("04", "cuda_gpu_kern_sum", [], "rank_stats",  # the 10 kernels with the most total time
     lambda d: px.bar(d[d.Name.isin(d.groupby("Name").Sum.sum().nlargest(10).index)], x="Name", y="Sum",
                      color="Rank", barmode="group", labels={"Sum": "ns"})),
    ("05", "nccl_gpu_overlap_trace", [], "rank_trace", lambda d: px.box(d, x="Rank", y="Compute Overlap")),
    ("07", "cuda_api_sync", [], "analysis", None),  # each CPU wait for the GPU: table only
    ("08", "cuda_gpu_mem_time_sum", [], "rank_stats",
     lambda d: px.bar(d, x="Name", y="Sum", color="Rank", barmode="group", labels={"Sum": "ns"})),
    ("09", "nccl_sum", [], "rank_stats", lambda d: px.bar(d, x=d.columns[0], y="Sum", color="Rank", barmode="group")),
]


def run_nsys(trace_dir: Path, out: Path, nsys: str) -> None:
    """nsys recipes on all rank*.nsys-rep; each table saved as CSV with our figure of it."""
    out.mkdir(parents=True, exist_ok=True)
    for num, recipe, args, table, plot in NSYS_RECIPES:
        run_logged([nsys, "recipe", recipe, "--input", str(trace_dir), "--output", str(out / recipe),
                    "--force-overwrite", *args], out / f"{recipe}.log")
        for nb in [*(out / recipe).glob("*.ipynb"), *(out / recipe).glob("*.py")]:
            nb.unlink()  # NVIDIA's notebooks and their helpers: not used
        f = out / recipe / f"{table}.parquet"
        if not f.exists():  # e.g. no NCCL kernels on 1 GPU
            continue
        d = pd.read_parquet(f).reset_index()
        d["Rank"] = d["Rank"].astype(str)
        d.to_csv(out / f"{num}_{recipe}.csv", index=False)
        if plot is None:
            continue
        fig = plot(d).update_layout(title=f"{recipe} (drawn by ranktrace from nsys recipe {table})")
        fig.write_html(out / f"{num}_{recipe}.html", include_plotlyjs="directory")
    print(f"nsys results in {out}")
