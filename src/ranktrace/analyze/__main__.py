"""python -m ranktrace.analyze <trace_dir> --tool hta|tracelens|nsys|stats"""
import argparse
from pathlib import Path

from . import run_hta, run_nsys, run_stats, run_tracelens

p = argparse.ArgumentParser(prog="python -m ranktrace.analyze", description="Analyze a directory of per-rank traces.")
p.add_argument("trace_dir", type=Path)
p.add_argument("--tool", choices=["hta", "tracelens", "nsys", "stats"], default="hta")
p.add_argument("--out", type=Path, default=None, help="output dir (default: <trace_dir>/analysis/<tool>)")
p.add_argument("--jobs", type=int, default=4, help="TraceLens ranks analysed in parallel")
p.add_argument("--nsys", default="nsys", help="nsys executable")
p.add_argument("--ranks", type=int, nargs="+", default=None,
               help="ranks for per-rank outputs (default: HTA 07 rank 0, stats 10 rank 0, 11 ranks 0 and 1)")
args = p.parse_args()
out = args.out or args.trace_dir / "analysis" / args.tool
if args.tool == "hta":
    run_hta(args.trace_dir, out, args.ranks or [0])
elif args.tool == "tracelens":
    run_tracelens(args.trace_dir, out, args.jobs)
elif args.tool == "nsys":
    run_nsys(args.trace_dir, out, args.nsys)
else:
    run_stats(args.trace_dir, out, args.ranks)
