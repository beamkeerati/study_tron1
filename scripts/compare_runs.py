#!/usr/bin/env python3
"""Compare tensorboard scalars across several training runs.

Usage
-----
    # list the scalar tags a run recorded
    python scripts/compare_runs.py --list ../tron1-rl-isaaclab/logs/rsl_rl/sf_tron_1a_flat/2026-09-04_15-00-00

    # plot one or more tags for several runs
    python scripts/compare_runs.py \
        ../tron1-rl-isaaclab/logs/rsl_rl/sf_tron_1a_flat/* \
        --tags "Train/mean_reward" "Train/mean_episode_length" \
        --out runs/exp_001_vs_002.png

Runs inside the `tron1` conda env; tensorboard is already installed there.
Labels default to the run directory name; pass --labels to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

DEFAULT_TAGS = ["Train/mean_reward", "Train/mean_episode_length"]


def load(run_dir: Path) -> EventAccumulator:
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def smooth(values: list[float], weight: float) -> list[float]:
    """TensorBoard-style exponential moving average."""
    if weight <= 0:
        return values
    out, last = [], values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run directories containing tfevents files")
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS, help="scalar tags to plot")
    p.add_argument("--labels", nargs="+", default=None, help="legend labels, one per run")
    p.add_argument("--smooth", type=float, default=0.6, help="EMA weight in [0,1); 0 disables")
    p.add_argument("--out", type=Path, default=Path("compare.png"))
    p.add_argument("--list", action="store_true", help="print available tags for the first run and exit")
    args = p.parse_args()

    runs = [r for r in args.runs if r.is_dir()]
    if not runs:
        print("no run directories found", file=sys.stderr)
        return 1

    if args.list:
        acc = load(runs[0])
        print(f"scalar tags in {runs[0]}:")
        for t in sorted(acc.Tags()["scalars"]):
            print(f"  {t}")
        return 0

    labels = args.labels or [r.name for r in runs]
    if len(labels) != len(runs):
        print("--labels must match the number of runs", file=sys.stderr)
        return 1

    accs = [load(r) for r in runs]

    n = len(args.tags)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False, sharex=True)

    for ax, tag in zip(axes[:, 0], args.tags):
        plotted = False
        for acc, label in zip(accs, labels):
            if tag not in acc.Tags()["scalars"]:
                continue
            events = acc.Scalars(tag)
            steps = [e.step for e in events]
            vals = smooth([e.value for e in events], args.smooth)
            ax.plot(steps, vals, label=label, linewidth=1.4)
            plotted = True
        ax.set_ylabel(tag.split("/")[-1])
        ax.set_title(tag, fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f"tag not found: {tag}", ha="center", transform=ax.transAxes)

    axes[-1, 0].set_xlabel("iteration")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
