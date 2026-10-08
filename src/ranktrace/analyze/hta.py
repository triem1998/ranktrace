"""HTA on torch traces: HTA's own tables and figures, plus our `01_sections` figure."""
from pathlib import Path

import pandas as pd
import plotly.basedatatypes
import plotly.express as px
from hta.trace_analysis import TraceAnalysis


def _sections(a, out: Path) -> pd.DataFrame:
    """HTA's CPU time per annotation (each step, DataLoader, optimizer, user regions), with our
    figure of it (HTA's own is a pie of nested, overlapping annotations). CPU, not GPU: the step is
    synced every batch, the DataLoader has no GPU work, and HTA splits GPU annotations per stream."""
    d = a.get_gpu_user_annotation_breakdown(use_gpu_annotation=False, visualize=False)
    fig = px.bar(d.astype({"rank": str}), x="name", y="mean (us)", color="rank", barmode="group",
                 title="time per batch on CPU (drawn by ranktrace from HTA get_gpu_user_annotation_breakdown)")
    fig.write_html(out / "01_sections.html", include_plotlyjs="directory")
    return d


# (output name, HTA call returning the table, HTA figures kept: 1 = the summary one, None = all)
HTA_ANALYSES = [
    ("01_sections", lambda a, r, o: _sections(a, o), 0),
    ("02_temporal", lambda a, r, o: a.get_temporal_breakdown(visualize=True), 1),
    ("03_idle", lambda a, r, o: a.get_idle_time_breakdown(visualize=True)[0], 1),
    ("04_kernel_types", lambda a, r, o: a.get_gpu_kernel_breakdown(visualize=True)[0], 1),
    ("05_comm_overlap", lambda a, r, o: a.get_comm_comp_overlap(visualize=True), 1),
    ("06_stragglers", lambda a, r, o: a.get_potential_stragglers(num_candidates=1, visualize=True), None),
    ("07_queue_length", lambda a, r, o: a.get_queue_length_summary(ranks=r), 0),
    ("07b_full_queue", lambda a, r, o: a.get_time_spent_blocked_on_full_queue(a.get_queue_length_time_series(ranks=r)), 0),
    ("08_memory_bw", lambda a, r, o: a.get_memory_bw_summary(), 0),
]


def run_hta(trace_dir: Path, out: Path, ranks: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    # HTA calls fig.show() and does not return the figure: collect it instead
    plotly.basedatatypes.BaseFigure.show = lambda fig, *a, **k: figs.append(fig)
    a = TraceAnalysis(trace_dir=str(trace_dir))
    for name, call, keep in HTA_ANALYSES:
        figs.clear()
        try:
            r = call(a, ranks, out)
        except Exception as e:
            print(f"{name}: failed: {type(e).__name__}: {e}")
            continue
        # HTA returns a DataFrame, a {rank: DataFrame} dict, a list or a Series
        df = pd.concat(r, names=["rank"]) if isinstance(r, dict) else pd.DataFrame(r)
        df.to_csv(out / f"{name}.csv")
        kept = figs[:keep] if keep is not None else figs
        for i, fig in enumerate(kept):
            fig.write_html(out / f"{name}{'' if len(kept) == 1 else f'_{i}'}.html", include_plotlyjs="directory")
    print(f"HTA results in {out}")
