#!/usr/bin/env python3
"""Supplementary Fig. S3: daily contributions and active accounts by community and authorship.

Inputs: ``design_daily_posts_by_type.csv`` (contributions) and
``design_daily_active_authors.csv`` (active accounts). The x-axis keeps
calendar-day indices, with gaps for the excluded dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from repro.paths import FIGURES_DIR, RESULTS_DIR, require

DEFAULT_STEM = "figS3_daily_activity"


def make_figure(tables_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR,
                *, stem_name: str = DEFAULT_STEM) -> list[Path]:
    """Draw selected calendar-date counts, preserving excluded dates as gaps."""
    daily = pd.read_csv(tables_dir / "design_daily_active_authors.csv")
    posts = pd.read_csv(tables_dir / "design_daily_posts_by_type.csv")
    dates = pd.date_range("2026-05-11", "2026-06-08")
    style = {"font.size": 11.5, "axes.labelsize": 11.5, "xtick.labelsize": 11.5,
             "ytick.labelsize": 11.5, "legend.fontsize": 11.5}
    colors = {"human": "#3c3c3c", "ai": "#0072B2"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey="row")
        for col, group in enumerate(["G1", "G2"]):
            d = (daily.loc[daily.group.eq(group)]
                 .assign(date=lambda x: pd.to_datetime(x.label_date))
                 .set_index("date").reindex(dates))
            for kind, source in [("human", "human_non_admin"), ("ai", "ai")]:
                p = posts.loc[posts.group.eq(group) & posts.author_class.eq(source)].groupby("label_date")["items"].sum()
                p.index = pd.to_datetime(p.index)
                values = p.reindex(dates)
                x = (dates - pd.Timestamp("2026-05-09")).days
                active = d["active_human_non_admin_authors" if kind == "human" else "active_ai_authors"]
                for row, y in [(0, values), (1, active)]:
                    axes[row, col].plot(x, y, color=colors[kind], label="Human" if kind == "human" else "AI",
                                        marker="o", markersize=2.7, linewidth=1.4)
            for row in range(2):
                ax = axes[row, col]
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_ylim(bottom=0)
                ax.set_xticks([2, 6, 9, 18, 22, 26, 30])
                ax.text(.01, .96, "abcd"[row * 2 + col], transform=ax.transAxes, weight="bold", va="top")
            axes[1, col].set_xlabel("Calendar-day index")
        axes[0, 0].set_ylabel("Contributions")
        axes[1, 0].set_ylabel("Active accounts")
        axes[0, 0].legend(frameon=False, loc="upper center", fontsize=9)
        fig.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        out = figures_dir / stem_name
        paths = [out.with_suffix(".pdf"), out.with_suffix(".png")]
        fig.savefig(paths[0], bbox_inches="tight")
        fig.savefig(paths[1], dpi=160, bbox_inches="tight")
        plt.close(fig)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=RESULTS_DIR,
                        help=f"directory of result tables (default: {RESULTS_DIR})")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR,
                        help=f"output directory (default: {FIGURES_DIR})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in make_figure(require(args.tables), args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
