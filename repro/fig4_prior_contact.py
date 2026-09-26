#!/usr/bin/env python3
"""Fig. 4, Supplementary Fig. S1 and Supplementary Fig. S2: prior contact (RQ2).

Fig. 4 shows the change in selecting the same AI account after one prior reply
24 valid hours earlier (``rq2_primary_estimands.csv``). Fig. S1 shows prior-contact
frequencies (``rq2_history_support.csv``). Fig. S2 adds the doubled-prior refit
(``rq2_doubled_prior_estimands.csv``) and one-to-two-reply projections
(``rq2_one_to_two_estimands.csv``).
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
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


from repro.paths import FIGURES_DIR, RESULTS_DIR, require

DEFAULT_TABLE = RESULTS_DIR / "rq2_primary_estimands.csv"
DEFAULT_OUTPUT_DIR = FIGURES_DIR
DEFAULT_STEM = "fig4_prior_contact"

FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 64.0
MM = 1.0 / 25.4
PNG_DPI = 300

GROUP_COLORS = {"G1": "#0072B2", "G2": "#D55E00"}
GROUP_MARKERS = {"G1": "o", "G2": "s"}
GROUP_LABELS = {"G1": "Visible labels", "G2": "Hidden labels"}
CONTRAST_COLOR = "#4D4D4D"
ZERO_COLOR = "#9E9E9E"
GRID_COLOR = "#E5E5E5"

DIRECTION_ORDER = ("ai_to_human", "human_to_ai")
DIRECTION_LABELS = {
    "human_to_ai": "Participant → AI",
    "ai_to_human": "AI → participant",
}
PRIMARY_DIRECTION_LABELS = {
    "human_to_ai": "Participant replied\nto AI account",
    "ai_to_human": "AI account replied\nto participant",
}
GROUP_ORDER = ("G1", "G2", "G2-G1")

REQUIRED_COLUMNS = {
    "direction",
    "group",
    "posterior_median",
    "ci_low",
    "ci_high",
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
    """Load and validate the six primary RQ2 estimates used in Fig. 4."""

    if not table_path.is_file():
        raise FileNotFoundError(
            f"RQ2 table not found: {table_path}"
        )
    frame = pd.read_csv(table_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"RQ2 table is missing required columns: {missing_columns}")

    frame = frame.copy()
    frame["direction"] = frame["direction"].astype(str).str.strip()
    frame["group"] = frame["group"].astype(str).str.strip()
    expected = {
        (direction, group)
        for direction in DIRECTION_ORDER
        for group in GROUP_ORDER
    }
    observed = set(zip(frame["direction"], frame["group"]))
    duplicates = frame.duplicated(["direction", "group"], keep=False)
    if duplicates.any():
        cells = sorted(
            set(zip(frame.loc[duplicates, "direction"], frame.loc[duplicates, "group"]))
        )
        raise ValueError(f"RQ2 table has duplicate direction-group cells: {cells}")
    if observed != expected or len(frame) != len(expected):
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            "RQ2 table must contain exactly the six primary direction-group cells; "
            f"missing={missing}, unexpected={unexpected}"
        )

    numeric_columns = ("posterior_median", "ci_low", "ci_high")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("RQ2 estimates and credible-interval bounds must be finite")
    invalid_interval = (
        frame["ci_low"].gt(frame["posterior_median"])
        | frame["posterior_median"].gt(frame["ci_high"])
    )
    if invalid_interval.any():
        cells = sorted(
            zip(
                frame.loc[invalid_interval, "direction"],
                frame.loc[invalid_interval, "group"],
            )
        )
        raise ValueError(f"RQ2 credible intervals do not contain their medians: {cells}")

    direction_rank = {value: index for index, value in enumerate(DIRECTION_ORDER)}
    group_rank = {value: index for index, value in enumerate(GROUP_ORDER)}
    frame["_direction_order"] = frame["direction"].map(direction_rank)
    frame["_group_order"] = frame["group"].map(group_rank)
    return frame.sort_values(["_direction_order", "_group_order"]).drop(
        columns=["_direction_order", "_group_order"]
    ).reset_index(drop=True)


def _symmetric_limit(frame: pd.DataFrame) -> tuple[float, float]:
    maximum = 100.0 * float(
        np.abs(frame[["ci_low", "ci_high"]].to_numpy(dtype=float)).max()
    )
    step = 2.0 if maximum <= 8.0 else 5.0
    limit = max(step, step * math.ceil((maximum + 0.5) / step))
    return limit, step


def _draw_interval(
    ax: plt.Axes,
    row: pd.Series,
    y: float,
    *,
    color: str,
    marker: str,
    markersize: float,
    filled: bool = True,
) -> None:
    median = 100.0 * float(row["posterior_median"])
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
        median,
        y,
        marker=marker,
        markersize=markersize,
        markerfacecolor=color if filled else "white",
        markeredgecolor="white" if filled else color,
        markeredgewidth=0.55,
        color=color,
        linestyle="none",
        zorder=3,
    )


def _format_axis(ax: plt.Axes, *, limit: float, step: float) -> None:
    ax.axvline(0.0, color=ZERO_COLOR, linewidth=0.75, linestyle="--", zorder=1)
    ax.set_xlim(-limit, limit)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)


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


def build_figure(frame: pd.DataFrame) -> Figure:
    """Build the two-panel RQ2 figure without writing files."""

    for group, limit in (("G1", 15), ("G2", 15), ("G2-G1", 5)):
        if (100 * frame.loc[frame.group.eq(group), ["ci_low", "ci_high"]].abs() > limit).any().any():
            raise ValueError(f"{group} interval exceeds the planned axis limits")
    with plt.rc_context(RC):
        fig, (within_ax, contrast_ax) = plt.subplots(
            1,
            2,
            figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM),
            gridspec_kw={"width_ratios": (1.4, 1), "wspace": 0.18},
            sharey=True,
        )

        bases = {"ai_to_human": 1.0, "human_to_ai": 0.0}
        offsets = {"G1": 0.13, "G2": -0.13}
        for direction in DIRECTION_ORDER:
            for group in ("G1", "G2"):
                row = frame[
                    frame["direction"].eq(direction) & frame["group"].eq(group)
                ].iloc[0]
                _draw_interval(
                    within_ax,
                    row,
                    bases[direction] + offsets[group],
                    color=GROUP_COLORS[group],
                    marker=GROUP_MARKERS[group],
                    markersize=5.0,
                )
        within_ax.set_yticks(
            [bases[direction] for direction in DIRECTION_ORDER],
            [PRIMARY_DIRECTION_LABELS[direction] for direction in DIRECTION_ORDER],
            linespacing=1.05,
        )
        within_ax.set_ylim(-0.48, 1.48)
        _panel_letter(within_ax, "a")
        within_ax.text(0.5, 1.04, "One prior reply vs none",
                       transform=within_ax.transAxes, ha="center", va="bottom", fontsize=8)
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
            bbox_to_anchor=(0.60, 1.0),
            frameon=False,
            ncol=2,
            handletextpad=0.35,
            columnspacing=1.0,
        )

        for direction in DIRECTION_ORDER:
            row = frame[
                frame["direction"].eq(direction) & frame["group"].eq("G2-G1")
            ].iloc[0]
            _draw_interval(
                contrast_ax,
                row,
                bases[direction],
                color=CONTRAST_COLOR,
                marker="D",
                markersize=4.6,
            )
        contrast_ax.tick_params(axis="y", labelleft=False)
        contrast_ax.set_ylim(-0.48, 1.48)
        _panel_letter(contrast_ax, "b")

        _format_axis(within_ax, limit=15, step=5)
        _format_axis(contrast_ax, limit=5, step=2)
        within_ax.set_xlabel("Change in later selection (pp)")
        contrast_ax.set_xlabel("Hidden − visible (pp)")
        fig.subplots_adjust(left=0.20, right=0.985, top=0.82, bottom=0.20)
        return fig


def load_support(table_path: Path) -> pd.DataFrame:
    """Select the opportunity denominator used by the primary projections."""
    frame = pd.read_csv(table_path)
    frame = frame.loc[frame["scope"].eq("estimand_competing_ai_receiver")].copy()
    levels = ("none", "one_prior_reply", "two_or_more_prior_replies")
    keys = ["direction", "group", "history_class"]
    expected = {(d, g, h) for d in DIRECTION_ORDER for g in ("G1", "G2") for h in levels}
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].values)) != expected:
        raise ValueError("Contact support must contain the twelve competing-opportunity cells")
    counts = pd.to_numeric(frame["receiver_opportunities"], errors="raise")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("Contact support counts must be finite nonnegative integers")
    frame["receiver_opportunities"] = counts
    totals = frame.groupby(["direction", "group"])["receiver_opportunities"].transform("sum")
    if totals.le(0).any():
        raise ValueError("Each contact-support distribution requires a positive denominator")
    frame["share"] = counts / totals
    if not np.allclose(frame["share"], frame["share_within_group_direction_scope"], rtol=1e-12):
        raise ValueError("Contact-support shares disagree with their denominators")
    by_direction = frame.groupby(["direction", "group"])["receiver_opportunities"].sum().unstack(0)
    if not by_direction.iloc[:, 0].equals(by_direction.iloc[:, 1]):
        raise ValueError("Both contact directions must use the same opportunities")
    return frame


def build_support_figure(frame: pd.DataFrame) -> Figure:
    """Show prior-contact frequency on the primary opportunity denominator."""
    levels = ("none", "one_prior_reply", "two_or_more_prior_replies")
    colors = ("#E8DFF1", "#AA84C1", "#5F337F")
    titles = {
        "ai_to_human": "AI account replied\nto participant",
        "human_to_ai": "Participant replied\nto AI account",
    }
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, sharey=True,
                                 figsize=(FIGURE_WIDTH_MM * MM, 66 * MM),
                                 gridspec_kw={"wspace": 0.18})
        for ax, direction, letter in zip(axes, DIRECTION_ORDER, "ab"):
            part = frame.loc[frame.direction.eq(direction)].set_index(["group", "history_class"])
            left = np.zeros(2)
            for level, color in zip(levels, colors):
                widths = 100 * np.array([part.loc[(g, level), "share"] for g in ("G1", "G2")])
                ax.barh([1, 0], widths, left=left, height=0.40,
                        color=color, edgecolor="white", linewidth=0.7)
                left += widths
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_ylim(-0.55, 1.55)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(titles[direction], fontsize=9.2, pad=7)
            ax.text(0, 1.10, letter, transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=10.2, fontweight="bold")
        axes[0].set_yticks([1, 0], [GROUP_LABELS[g] for g in ("G1", "G2")])
        for label, group in zip(axes[0].get_yticklabels(), ("G1", "G2")):
            label.set_color(GROUP_COLORS[group])
        fig.legend([Patch(facecolor=c) for c in colors], ["0", "1", "2+"],
                   title="Prior replies", loc="upper center", bbox_to_anchor=(0.60, 1.0),
                   ncol=3, frameon=False, title_fontsize=9.2)
        fig.supxlabel("Opportunities (%)", x=0.58, y=0.04, fontsize=9.2)
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.20, top=0.65)
        return fig


def build_sensitivity_figure(primary: pd.DataFrame, doubled: pd.DataFrame,
                             repeat: pd.DataFrame) -> Figure:
    """Keep all six primary, doubled-prior and one-to-two-contact estimates."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, sharey=True,
                                 figsize=(FIGURE_WIDTH_MM * MM, 91 * MM),
                                 gridspec_kw={"width_ratios": (1.4, 1), "wspace": 0.18})
        rows = [(d, g) for d in DIRECTION_ORDER for g in GROUP_ORDER]
        positions = [6, 5, 4, 2, 1, 0]
        for (direction, group), y in zip(rows, positions):
            color = GROUP_COLORS.get(group, CONTRAST_COLOR)
            marker = GROUP_MARKERS.get(group, "D")
            for frame, offset, filled in ((primary, 0.13, True), (doubled, -0.13, False)):
                row = frame.loc[frame.direction.eq(direction) & frame.group.eq(group)].iloc[0]
                _draw_interval(axes[0], row, y + offset, color=color, marker=marker,
                               markersize=4.1, filled=filled)
            row = repeat.loc[repeat.direction.eq(direction) & repeat.group.eq(group)].iloc[0]
            _draw_interval(axes[1], row, y, color=color, marker=marker, markersize=4.1)
        axes[0].set_yticks(positions, ["Visible", "Hidden", "Difference"] * 2)
        for label, (_, group) in zip(axes[0].get_yticklabels(), rows):
            label.set_color(GROUP_COLORS.get(group, CONTRAST_COLOR))
        for y, direction in ((5, "ai_to_human"), (1, "human_to_ai")):
            axes[0].text(-0.37, y, DIRECTION_LABELS[direction],
                         transform=axes[0].get_yaxis_transform(), rotation=90,
                         ha="center", va="center", fontsize=9.2)
        for ax, letter, frame in (
            (axes[0], "a", pd.concat([primary, doubled])),
            (axes[1], "b", repeat),
        ):
            limit, step = _symmetric_limit(frame)
            _format_axis(ax, limit=limit, step=step)
            ax.set_ylim(-0.6, 6.6)
            _panel_letter(ax, letter)
        fig.legend([Line2D([], [], marker="o", linestyle="none", color=CONTRAST_COLOR,
                           markerfacecolor=face, markersize=4.5)
                    for face in (CONTRAST_COLOR, "white")],
                   ["Primary", "Doubled priors"], loc="upper center",
                   bbox_to_anchor=(0.60, 1.0), ncol=2, frameon=False)
        fig.supxlabel("Selection change (pp)", x=0.61, y=0.04, fontsize=9.2)
        fig.subplots_adjust(left=0.23, right=0.985, bottom=0.15, top=0.87)
        return fig


def make_supplementary_figures(table_path: Path, output_dir: Path) -> list[Path]:
    """Render Figs. S1 and S2 from tables next to the primary RQ2 table."""
    directory = table_path.parent
    primary = load_estimands(table_path)
    support = load_support(directory / "rq2_history_support.csv")
    doubled = load_estimands(directory / "rq2_doubled_prior_estimands.csv")
    repeat = load_estimands(directory / "rq2_one_to_two_estimands.csv")
    opportunities = support.loc[support.direction.eq(DIRECTION_ORDER[0]), "receiver_opportunities"].sum()
    for frame in (primary, doubled, repeat):
        if not frame["ai_receiver_opportunities"].eq(opportunities).all():
            raise ValueError("Support and projection denominators disagree")
    paths = save_figure(build_support_figure(support), output_dir,
                        stem_name="figS1_prior_contact_support")
    paths += save_figure(build_sensitivity_figure(primary, doubled, repeat), output_dir,
                         stem_name="figS2_prior_contact_sensitivity")
    return paths


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
    """Render Fig. 4 as PDF and PNG."""

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
    table = require(args.tables / DEFAULT_TABLE.name)
    paths = make_figure(table, args.out) + make_supplementary_figures(table, args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
