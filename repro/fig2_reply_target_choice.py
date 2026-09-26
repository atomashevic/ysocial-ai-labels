#!/usr/bin/env python3
"""Fig. 2: AI-target selection and standardized differences between communities (RQ1).

Panel a shows observed shares of human replies directed at AI-authored posts or
comments (95% Wilson intervals, from ``rq3_group_summary.csv`` and
``rq1_choice_set_attrition.csv``) and standardized posterior probabilities
(``rq1_primary_estimand.csv``). Panel b shows hidden-minus-visible contrasts for
the primary estimate and three sensitivity projections (``rq1_model_estimands.csv``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from repro.paths import FIGURES_DIR, RESULTS_DIR, require

DEFAULT_STEM = "fig2_reply_target_choice"

GROUP_COLORS = {"G1": "#0072B2", "G2": "#D55E00"}
GROUP_LABELS = {"G1": "Visible labels", "G2": "Hidden labels"}
NEUTRAL = "#4D4D4D"
LIGHT = "#BDBDBD"

MM = 1.0 / 25.4
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[1], dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return paths


def wilson_interval(successes: float, total: float, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the 95% Wilson score interval for a binomial proportion."""

    if total <= 0:
        return float("nan"), float("nan")
    p_hat = successes / total
    denominator = 1.0 + z**2 / total
    centre = (p_hat + z**2 / (2.0 * total)) / denominator
    half_width = z * ((p_hat * (1.0 - p_hat) / total + z**2 / (4.0 * total**2)) ** 0.5) / denominator
    return centre - half_width, centre + half_width


def load_rq1_raw_shares(tables_dir: Path) -> pd.DataFrame:
    """Return the raw share of analysed replies whose target was AI-authored, per community."""

    summary = pd.read_csv(tables_dir / "rq3_group_summary.csv")
    targets = summary[
        (summary["subset"] == "rq1_analysis_set")
        & (summary["reaction_placement"] == "round_start")
        & (summary["metric"] == "target_received_human_downvote_24h")
    ]
    attrition = pd.read_csv(tables_dir / "rq1_choice_set_attrition.csv")
    retained = attrition[attrition["outcome"] == "retained"].set_index("group")["human_reply_events"]
    rows = []
    for group in ("G1", "G2"):
        ai_targets = int(targets[(targets["group"] == group) & (targets["dyad"] == "H->AI")]["denominator_n"].iloc[0])
        human_targets = int(targets[(targets["group"] == group) & (targets["dyad"] == "H->H")]["denominator_n"].iloc[0])
        replies = int(retained[group])
        if ai_targets + human_targets != replies:
            raise ValueError(f"{group}: {ai_targets} + {human_targets} targets != {replies} retained replies")
        low, high = wilson_interval(ai_targets, replies)
        rows.append({"group": group, "ai_targets": ai_targets, "replies": replies, "share": ai_targets / replies, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def load_rq1_standardized(tables_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(tables_dir / "rq1_primary_estimand.csv")
    return frame.set_index("group")


def load_rq1_contrasts(tables_dir: Path) -> pd.DataFrame:
    """Return hidden-minus-visible contrasts for the primary estimand and its sensitivities."""

    estimands = pd.read_csv(tables_dir / "rq1_model_estimands.csv")
    contrasts = estimands[(estimands["model"] == "M1") & (estimands["group"] == "G2-G1")].set_index("estimand")
    spec = (
        ("G2_minus_G1_common_opportunity_ai_choice_probability", "Primary", True),
        ("mixed_risk_set_common_opportunity_ai_choice_probability", "Mixed candidates", False),
        ("equal_participant_mixed_risk_set_ai_choice_probability", "Equal participant weight", False),
        ("direct_G2xAI_common_opportunity_probability", "Authorship term only", False),
    )
    rows = []
    for key, label, primary in spec:
        row = contrasts.loc[key]
        rows.append({"label": label, "mean": row["posterior_mean"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "primary": primary})
    return pd.DataFrame(rows)


def draw_rq1_levels(ax: plt.Axes, raw: pd.DataFrame, standardized: pd.DataFrame) -> None:
    """Show observed and standardized AI-target selection by community."""
    raw = raw.set_index("group")
    for row_y, frame, value in ((1, raw, "share"), (0, standardized, "posterior_mean")):
        for group, offset, marker in (("G1", 0.12, "o"), ("G2", -0.12, "s")):
            row = frame.loc[group]
            color = GROUP_COLORS[group]
            mean = 100 * row[value]
            ax.errorbar(mean, row_y + offset,
                        xerr=[[mean - 100 * row["ci_low"]], [100 * row["ci_high"] - mean]],
                        fmt=marker, markersize=5.5, markerfacecolor=color,
                        markeredgecolor=color, markeredgewidth=1, ecolor=color,
                        elinewidth=1, capsize=2.5)
    ax.set(xlim=(0, 45), ylim=(-.55, 1.55), xticks=[0, 10, 20, 30, 40],
           yticks=[1, 0], yticklabels=["Observed", "Standardized"],
           xlabel="AI-target selection (%)")
    handles = [Line2D([], [], marker=marker, markerfacecolor=GROUP_COLORS[group],
                      markeredgecolor=GROUP_COLORS[group], linestyle="none", markersize=5,
                      label=GROUP_LABELS[group]) for group, marker in (("G1", "o"), ("G2", "s"))]
    ax.legend(handles=handles, loc="lower left", frameon=False, handletextpad=.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3)


def draw_rq1_contrasts(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    """Neutral colour for between-community contrasts, with no duplicated numbers."""
    ys = list(range(len(contrasts)))[::-1]
    ax.axvline(0, color=LIGHT, linewidth=.8, linestyle=(0, (3, 3)), zorder=0)
    for y, (_, row) in zip(ys, contrasts.iterrows()):
        ax.plot([100 * row["ci_low"], 100 * row["ci_high"]], [y, y], color=NEUTRAL,
                linewidth=1.3 if row["primary"] else 1, solid_capstyle="butt")
        ax.plot(100 * row["mean"], y, marker="D", markersize=5.5,
                color=NEUTRAL, linestyle="none")
    ax.set(yticks=ys, yticklabels=contrasts["label"].tolist(), ylim=(-.6, 3.6),
           xlim=(-1, 10), xticks=[0, 2, 4, 6, 8, 10],
           xlabel="Hidden - visible (percentage points)")
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3)


def make_figure(tables_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR,
                *, stem_name: str = DEFAULT_STEM) -> list[Path]:
    """Render Fig. 2 as PDF and PNG."""
    raw = load_rq1_raw_shares(tables_dir)
    standardized = load_rq1_standardized(tables_dir)
    contrasts = load_rq1_contrasts(tables_dir)
    style = dict(RC, **{"font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
                       "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                       "ytick.labelsize": 8, "legend.fontsize": 8})
    with plt.rc_context(style):
        fig = plt.figure(figsize=(180 * MM, 78 * MM))
        ax_a = fig.add_axes([.12, .19, .30, .70])
        ax_b = fig.add_axes([.665, .19, .32, .70])
        draw_rq1_levels(ax_a, raw, standardized)
        draw_rq1_contrasts(ax_b, contrasts)
        fig.text(.02, .94, "a", fontsize=10, fontweight="bold")
        fig.text(.435, .94, "b", fontsize=10, fontweight="bold")
        return _save(fig, figures_dir / stem_name)


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
