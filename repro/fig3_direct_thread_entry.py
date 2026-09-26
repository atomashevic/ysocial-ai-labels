#!/usr/bin/env python3
"""Fig. 3: direct thread entry after recorded feed-card renders (additional RQ1 analysis).

Panel a shows equal-participant standardized entry probabilities by post authorship
and community; panel b shows hidden-minus-visible contrasts and the difference in
the AI-minus-participant gap. Input: ``rq1b_primary_estimands.csv``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


from repro.paths import FIGURES_DIR, RESULTS_DIR, require

DEFAULT_TABLE = RESULTS_DIR / "rq1b_primary_estimands.csv"
DEFAULT_OUTPUT_DIR = FIGURES_DIR
DEFAULT_STEM = "fig3_direct_thread_entry"

FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 78.0
MM = 1.0 / 25.4
PNG_DPI = 300

PRIMARY_SCOPE = "equal_participant_observed_opportunities"
GROUP_COLORS = {"G1": "#0072B2", "G2": "#D55E00"}
GROUP_MARKERS = {"G1": "o", "G2": "s"}
GROUP_LABELS = {"G1": "Visible labels", "G2": "Hidden labels"}
AUTHOR_ORDER = ("AI", "H")
AUTHOR_LABELS = {"AI": "AI-authored posts", "H": "Participant-authored\nposts"}
COMPONENT_COLOR = "#4D4D4D"
PRIMARY_COLOR = COMPONENT_COLOR
ZERO_COLOR = "#9E9E9E"
GRID_COLOR = "#E5E5E5"

LEVEL_ESTIMAND = "direct_thread_entry_probability"
WITHIN_GROUP_ESTIMAND = "AI_minus_H_direct_thread_entry_probability"
CONDITION_ESTIMAND = "G2_minus_G1_direct_thread_entry_probability"
PRIMARY_ESTIMAND = "G2_minus_G1_change_in_AI_minus_H_gap"

PRIMARY_CELL_ORDER = (
    (LEVEL_ESTIMAND, "G1", "AI"),
    (LEVEL_ESTIMAND, "G2", "AI"),
    (LEVEL_ESTIMAND, "G1", "H"),
    (LEVEL_ESTIMAND, "G2", "H"),
    (WITHIN_GROUP_ESTIMAND, "G1", "AI-H"),
    (WITHIN_GROUP_ESTIMAND, "G2", "AI-H"),
    (CONDITION_ESTIMAND, "G2-G1", "AI"),
    (CONDITION_ESTIMAND, "G2-G1", "H"),
    (PRIMARY_ESTIMAND, "G2-G1", "AI-H"),
)

REQUIRED_COLUMNS = {
    "estimand",
    "group",
    "author_type",
    "posterior_mean",
    "ci_low",
    "ci_high",
    "standardization_scope",
}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.2,
    "axes.titlesize": 9.2,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 9.2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def load_estimands(table_path: Path = DEFAULT_TABLE) -> pd.DataFrame:
    """Load and validate the nine equal-participant RQ1b estimands."""

    if not table_path.is_file():
        raise FileNotFoundError(
            f"RQ1b table not found: {table_path}"
        )
    frame = pd.read_csv(table_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"RQ1b table is missing required columns: {missing_columns}")

    frame = frame.copy()
    for column in ("estimand", "group", "author_type", "standardization_scope"):
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[frame["standardization_scope"].eq(PRIMARY_SCOPE)].copy()

    key_columns = ["estimand", "group", "author_type"]
    duplicates = frame.duplicated(key_columns, keep=False)
    if duplicates.any():
        cells = sorted(
            set(map(tuple, frame.loc[duplicates, key_columns].itertuples(index=False, name=None)))
        )
        raise ValueError(f"RQ1b table has duplicate primary cells: {cells}")

    expected = set(PRIMARY_CELL_ORDER)
    observed = set(map(tuple, frame[key_columns].itertuples(index=False, name=None)))
    if observed != expected or len(frame) != len(PRIMARY_CELL_ORDER):
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            "RQ1b table must contain exactly the nine equal-participant primary cells; "
            f"missing={missing}, unexpected={unexpected}"
        )

    numeric_columns = ("posterior_mean", "ci_low", "ci_high")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("RQ1b estimates and credible-interval bounds must be finite")
    invalid_interval = (
        frame["ci_low"].gt(frame["posterior_mean"])
        | frame["posterior_mean"].gt(frame["ci_high"])
    )
    if invalid_interval.any():
        cells = sorted(
            map(
                tuple,
                frame.loc[invalid_interval, key_columns].itertuples(index=False, name=None),
            )
        )
        raise ValueError(f"RQ1b credible intervals do not contain their means: {cells}")

    order = {cell: index for index, cell in enumerate(PRIMARY_CELL_ORDER)}
    frame["_order"] = [order[cell] for cell in frame[key_columns].itertuples(index=False, name=None)]
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _select_row(
    frame: pd.DataFrame,
    estimand: str,
    group: str,
    author_type: str,
) -> pd.Series:
    return frame[
        frame["estimand"].eq(estimand)
        & frame["group"].eq(group)
        & frame["author_type"].eq(author_type)
    ].iloc[0]


def _draw_interval(
    ax: plt.Axes,
    row: pd.Series,
    y: float,
    *,
    color: str,
    marker: str,
    markersize: float,
) -> None:
    mean = 100.0 * float(row["posterior_mean"])
    low = 100.0 * float(row["ci_low"])
    high = 100.0 * float(row["ci_high"])
    ax.plot(
        [low, high],
        [y, y],
        color=color,
        linewidth=1.15,
        solid_capstyle="butt",
        zorder=2,
    )
    ax.plot(
        mean,
        y,
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        color=color,
        linestyle="none",
        zorder=3,
    )


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.0,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
    )


def _probability_limit(frame: pd.DataFrame) -> float:
    levels = frame[frame["estimand"].eq(LEVEL_ESTIMAND)]
    maximum = 100.0 * float(levels["ci_high"].max())
    return max(10.0, 5.0 * math.ceil((maximum + 0.5) / 5.0))


def _contrast_scale(frame: pd.DataFrame) -> tuple[float, float]:
    contrasts = frame[
        frame["estimand"].isin((CONDITION_ESTIMAND, PRIMARY_ESTIMAND))
    ]
    maximum = 100.0 * float(
        np.abs(contrasts[["ci_low", "ci_high"]].to_numpy(dtype=float)).max()
    )
    step = 2.0 if maximum <= 10.0 else 5.0
    limit = max(6.0, step * math.ceil((maximum + 0.5) / step))
    return limit, step


def _format_probability_axis(ax: plt.Axes, limit: float) -> None:
    ax.set_xlim(0.0, limit)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.set_xlabel("Probability of direct thread entry (%)")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)


def _format_contrast_axis(ax: plt.Axes, limit: float, step: float) -> None:
    ax.axvline(0.0, color=ZERO_COLOR, linewidth=0.75, linestyle="--", zorder=1)
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_locator(MultipleLocator(max(step, 5.0)))
    ax.set_xlabel("Hidden minus visible\n(percentage points)")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)


def build_figure(frame: pd.DataFrame) -> Figure:
    """Build the two-panel RQ1b figure without writing files."""

    probability_limit = _probability_limit(frame)
    contrast_limit, contrast_step = _contrast_scale(frame)
    with plt.rc_context(RC):
        fig, (probability_ax, contrast_ax) = plt.subplots(
            1,
            2,
            figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM),
            gridspec_kw={"width_ratios": (1.0, 1.12), "wspace": 0.58},
        )

        bases = {"AI": 1.0, "H": 0.0}
        offsets = {"G1": 0.13, "G2": -0.13}
        for author_type in AUTHOR_ORDER:
            for group in ("G1", "G2"):
                _draw_interval(
                    probability_ax,
                    _select_row(frame, LEVEL_ESTIMAND, group, author_type),
                    bases[author_type] + offsets[group],
                    color=GROUP_COLORS[group],
                    marker=GROUP_MARKERS[group],
                    markersize=5.0,
                )
        probability_ax.set_yticks(
            [bases[author_type] for author_type in AUTHOR_ORDER],
            [AUTHOR_LABELS[author_type] for author_type in AUTHOR_ORDER],
        )
        probability_ax.set_ylim(-0.48, 1.48)
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker=GROUP_MARKERS[group],
                    color=GROUP_COLORS[group],
                    markerfacecolor=GROUP_COLORS[group],
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    linestyle="none",
                    markersize=5.0,
                    label=GROUP_LABELS[group],
                )
                for group in ("G1", "G2")
            ],
            loc="upper center",
            bbox_to_anchor=(0.30, 1.0),
            frameon=False,
            ncol=2,
            handletextpad=0.35,
            columnspacing=1.0,
        )
        _format_probability_axis(probability_ax, probability_limit)
        _panel_letter(probability_ax, "a")

        contrast_rows = (
            (
                "AI-authored posts",
                _select_row(frame, CONDITION_ESTIMAND, "G2-G1", "AI"),
                COMPONENT_COLOR,
                "D",
            ),
            (
                "Participant-authored posts",
                _select_row(frame, CONDITION_ESTIMAND, "G2-G1", "H"),
                COMPONENT_COLOR,
                "D",
            ),
            (
                "AI-minus-participant\ncontrast (primary)",
                _select_row(frame, PRIMARY_ESTIMAND, "G2-G1", "AI-H"),
                PRIMARY_COLOR,
                "D",
            ),
        )
        contrast_y = (2.0, 1.0, -0.3)
        for y, (_, row, color, marker) in zip(contrast_y, contrast_rows):
            _draw_interval(
                contrast_ax,
                row,
                y,
                color=color,
                marker=marker,
                markersize=6.0 if marker == "D" else 4.6,
            )
        contrast_ax.set_yticks(contrast_y, [row[0] for row in contrast_rows], linespacing=1.05)
        contrast_ax.get_yticklabels()[-1].set_fontweight("bold")
        contrast_ax.set_ylim(-0.8, 2.55)
        _format_contrast_axis(contrast_ax, contrast_limit, contrast_step)
        _panel_letter(contrast_ax, "b")

        fig.subplots_adjust(left=0.22, right=0.985, top=0.80, bottom=0.20)
        return fig


def save_figure(
    fig: Figure,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    stem_name: str = DEFAULT_STEM,
) -> list[Path]:
    """Write a journal-width PDF and a 300 dpi PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem_name}.pdf"
    png_path = output_dir / f"{stem_name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=PNG_DPI)
    plt.close(fig)
    return [pdf_path, png_path]


def make_figure(
    table_path: Path = DEFAULT_TABLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    stem_name: str = DEFAULT_STEM,
) -> list[Path]:
    """Render Fig. 3 as PDF and PNG."""

    frame = load_estimands(table_path)
    return save_figure(build_figure(frame), output_dir, stem_name=stem_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=RESULTS_DIR,
                        help=f"directory of result tables (default: {RESULTS_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in make_figure(require(args.tables / DEFAULT_TABLE.name), args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
