"""Per-rank stats written by `stats=True`: step times, peak memory, trace size."""
import json
from pathlib import Path

import pandas as pd


def run_stats(trace_dir: Path) -> None:
    rows = [json.loads(f.read_text()) for f in sorted((trace_dir / "stats").glob("rank*.json"))]
    df = pd.DataFrame(rows).drop(columns=["step_s"]).set_index("rank")
    print(df.to_string())
    print(f"max GPU device {df.gpu_device_peak_gb.max():.2f} GB | max host RSS {df.host_rss_peak_gb.max():.2f} GB"
          f" | total trace {df.trace_gb.sum():.3f} GB over {len(df)} ranks")
