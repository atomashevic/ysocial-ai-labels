"""Manuscript figures rendered from admitted tables in ``results/tables``.

These figures are display items for the manuscript, not analysis outputs. They
read only admitted CSV tables and write PDF/PNG pairs under
``artifacts/manuscript/figures``. Time axes use consecutive analysis days
(1--21); the calendar gap and warm-up are described in the manuscript text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES_DIR = ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = ROOT / "artifacts" / "manuscript" / "figures"

GROUP_COLORS = {"G1": "#0072B2", "G2": "#D55E00"}
GROUP_LABELS = {"G1": "Visible labels", "G2": "Hidden labels"}
GROUP_COUNTS = {"G1": (20, 40), "G2": (16, 32)}
NEUTRAL = "#4D4D4D"
LIGHT = "#BDBDBD"
EXPECTED_ANALYSIS_DAYS = 21

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


def _panel_letter(ax: plt.Axes, letter: str, *, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def load_daily_contributions(tables_dir: Path) -> pd.DataFrame:
    """Return daily contributions per community and author class on a 1..21 day axis."""

    frame = pd.read_csv(tables_dir / "design_daily_posts_by_type.csv")
    frame["label_date"] = pd.to_datetime(frame["label_date"])
    daily = (
        frame.groupby(["group", "label_date", "author_class"], as_index=False)["items"]
        .sum()
        .sort_values(["group", "label_date"])
    )
    day_index = (
        daily[["group", "label_date"]]
        .drop_duplicates()
        .sort_values(["group", "label_date"])
        .assign(analysis_day=lambda d: d.groupby("group").cumcount() + 1)
    )
    daily = daily.merge(day_index, on=["group", "label_date"], how="left")
    for group, count in daily.groupby("group")["analysis_day"].max().items():
        if int(count) != EXPECTED_ANALYSIS_DAYS:
            raise ValueError(f"{group} has {count} analysis days; expected {EXPECTED_ANALYSIS_DAYS}")
    return daily


def _draw_account_row(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    *,
    name: str,
    is_ai: bool,
    labelled: bool,
    color: str,
) -> None:
    """Draw one mock feed row: avatar, username, optional AI chip, text bar."""

    ax.add_patch(
        Rectangle((x, y - 0.032), width, 0.064, facecolor="white", edgecolor=LIGHT, linewidth=0.5)
    )
    marker = "s" if is_ai else "o"
    ax.plot(x + 0.03, y, marker=marker, markersize=4, color=NEUTRAL, markeredgewidth=0)
    ax.text(x + 0.06, y, name, fontsize=5.5, va="center", ha="left", color=NEUTRAL)
    chip_x = x + 0.06 + 0.0135 * len(name) + 0.012
    if is_ai and labelled:
        ax.add_patch(
            FancyBboxPatch(
                (chip_x, y - 0.016),
                0.05,
                0.032,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(chip_x + 0.025, y, "AI", fontsize=4.8, color="white", ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((x + width - 0.11, y - 0.007), 0.08, 0.014, facecolor=LIGHT, edgecolor="none"))


def draw_design_schematic(ax: plt.Axes) -> None:
    """Panel a: shared inputs feeding two communities that differ only in the AI label."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.add_patch(
        FancyBboxPatch(
            (0.10, 0.82),
            0.80,
            0.16,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="#F2F2F2",
            edgecolor=NEUTRAL,
            linewidth=0.6,
        )
    )
    ax.text(0.50, 0.935, "Identical YSocial configuration", ha="center", va="center", fontweight="bold")
    ax.text(
        0.50,
        0.87,
        "Same personas, prompts and feed material;\nAI accounts with relationship memory",
        ha="center",
        va="center",
        fontsize=6,
        color=NEUTRAL,
        linespacing=1.1,
    )

    box_w, box_h, box_y = 0.46, 0.58, 0.10
    for group, x0 in (("G1", 0.02), ("G2", 0.52)):
        color = GROUP_COLORS[group]
        participants, accounts = GROUP_COUNTS[group]
        ax.add_patch(
            FancyArrowPatch(
                (x0 + box_w / 2, 0.815),
                (x0 + box_w / 2, box_y + box_h + 0.012),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.7,
                color=NEUTRAL,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x0, box_y),
                box_w,
                box_h,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(x0 + box_w / 2, box_y + box_h - 0.055, GROUP_LABELS[group], ha="center", va="center", fontweight="bold", color=color)
        ax.text(
            x0 + box_w / 2,
            box_y + box_h - 0.14,
            f"{participants} participants\n{accounts} AI accounts",
            ha="center",
            va="center",
            fontsize=6,
            color=NEUTRAL,
            linespacing=1.1,
        )
        labelled = group == "G1"
        rows = (("ana_reads", False), ("velvet_vinyl", True), ("noir_notes", True))
        row_y = box_y + box_h - 0.26
        for name, is_ai in rows:
            _draw_account_row(ax, x0 + 0.03, row_y, box_w - 0.06, name=name, is_ai=is_ai, labelled=labelled, color=color)
            row_y -= 0.085

    handles = [
        Line2D([], [], marker="o", color=NEUTRAL, linestyle="none", markersize=4, label="Participant"),
        Line2D([], [], marker="s", color=NEUTRAL, linestyle="none", markersize=4, label="AI account"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=1.2,
    )


def draw_choice_set(ax: plt.Axes) -> None:
    """Panel c: one participant reply as a choice among earlier same-thread contributions."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # (x, y, is_ai, status, label); y runs downward with time.
    nodes = {
        "root": (0.26, 0.92, False, "eligible", "Root post"),
        "c1": (0.42, 0.78, True, "eligible", ""),
        "c2": (0.66, 0.78, False, "eligible", ""),
        "c3": (0.42, 0.62, False, "self", "Replier's own\nearlier comment"),
        "c4": (0.66, 0.62, True, "selected", "Selected\ntarget"),
        "c5": (0.88, 0.70, True, "eligible", ""),
        "reply": (0.66, 0.42, False, "reply", "Observed reply"),
        "c6": (0.26, 0.28, True, "later", "Posted after the reply"),
    }
    edges = (("root", "c1"), ("root", "c2"), ("c1", "c3"), ("c2", "c4"), ("c2", "c5"), ("c4", "reply"), ("root", "c6"))
    for parent, child in edges:
        px, py = nodes[parent][:2]
        cx, cy = nodes[child][:2]
        later = nodes[child][3] == "later"
        ax.plot([px, cx], [py, cy], color=LIGHT if later else NEUTRAL, linewidth=0.7, linestyle=":" if later else "-", zorder=1)

    reply_y = nodes["reply"][1]
    ax.plot([0.14, 0.98], [reply_y, reply_y], color=NEUTRAL, linewidth=0.6, linestyle="--", zorder=0)
    ax.text(0.16, reply_y - 0.025, "time of reply", ha="left", va="top", fontsize=6, color=NEUTRAL, style="italic")
    ax.annotate(
        "",
        xy=(0.08, 0.20),
        xytext=(0.08, 0.96),
        arrowprops={"arrowstyle": "-|>", "color": NEUTRAL, "linewidth": 0.6, "mutation_scale": 7},
    )
    ax.text(0.06, 0.58, "time", rotation=90, ha="right", va="center", fontsize=6, color=NEUTRAL)

    style = {
        "eligible": {"facecolor": "white", "edgecolor": NEUTRAL, "linewidth": 1.1},
        "selected": {"facecolor": NEUTRAL, "edgecolor": NEUTRAL, "linewidth": 1.1},
        "self": {"facecolor": "#E6E6E6", "edgecolor": LIGHT, "linewidth": 0.8},
        "later": {"facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.8},
        "reply": {"facecolor": "white", "edgecolor": NEUTRAL, "linewidth": 1.6},
    }
    for key, (x, y, is_ai, status, label) in nodes.items():
        kw = style[status]
        marker = "s" if is_ai else "o"
        ax.plot(
            x,
            y,
            marker=marker,
            markersize=8.5,
            markerfacecolor=kw["facecolor"],
            markeredgecolor=kw["edgecolor"],
            markeredgewidth=kw["linewidth"],
            linestyle="none",
            zorder=3,
        )
        if key == "reply":
            ax.text(
                x + 0.055,
                y,
                label,
                ha="left",
                va="center",
                fontsize=6,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
                zorder=4,
            )
        elif key == "c4":
            ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=6, linespacing=1.0)
        elif key == "c3":
            ax.text(x - 0.055, y, label, ha="right", va="center", fontsize=6, color=NEUTRAL, linespacing=1.0)
        elif key == "root":
            ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=6)
        elif key == "c6":
            ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=6, color="#9E9E9E")

    ax.text(
        0.55,
        0.12,
        "Choice set: the 5 earlier same-thread contributions\nnot written by the replier (2 AI-authored)",
        ha="center",
        va="center",
        fontsize=6,
        color=NEUTRAL,
        linespacing=1.1,
    )
    handles = [
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=NEUTRAL, linestyle="none", markersize=5, label="Eligible candidate"),
        Line2D([], [], marker="o", markerfacecolor=NEUTRAL, markeredgecolor=NEUTRAL, linestyle="none", markersize=5, label="Selected target"),
        Line2D([], [], marker="o", markerfacecolor="#E6E6E6", markeredgecolor=LIGHT, linestyle="none", markersize=5, label="Excluded"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def draw_daily_activity(axes: tuple[plt.Axes, plt.Axes], daily: pd.DataFrame) -> None:
    """Panel b: daily contributions by author class, one facet per community."""

    ymax = daily.groupby(["group", "analysis_day", "author_class"])["items"].sum().max()
    for ax, group in zip(axes, ("G1", "G2")):
        color = GROUP_COLORS[group]
        sub = daily[daily["group"] == group]
        for author_class, linestyle, marker, label in (
            ("ai", "-", "s", "AI accounts"),
            ("human_non_admin", "--", "o", "Participants"),
        ):
            series = sub[sub["author_class"] == author_class].sort_values("analysis_day")
            ax.plot(
                series["analysis_day"],
                series["items"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3,
                markerfacecolor=color if author_class == "ai" else "white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                label=label,
            )
        ax.set_xlim(0.5, EXPECTED_ANALYSIS_DAYS + 0.5)
        ax.set_xticks([1, 7, 14, 21])
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlabel("Analysis day")
        ax.text(0.02, 0.97, GROUP_LABELS[group], transform=ax.transAxes, ha="left", va="top", color=color, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5)
    axes[0].set_ylabel("Contributions per day\n(root posts and comments)")
    axes[1].tick_params(labelleft=False)
    axes[1].legend(loc="upper right", frameon=False, handlelength=2.2)


def make_figure_design(
    tables_dir: Path = DEFAULT_TABLES_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    *,
    stem_name: str = "figure_1_design",
) -> list[Path]:
    """Render Figure 1: design schematic, choice-set illustration, and daily activity."""

    daily = load_daily_contributions(tables_dir)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(180 * MM, 150 * MM))
        outer = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.0, 1.0], hspace=0.28, wspace=0.12)
        ax_a = fig.add_subplot(outer[0, 0])
        ax_c = fig.add_subplot(outer[0, 1])
        inner = outer[1, :].subgridspec(1, 2, wspace=0.08)
        ax_b1 = fig.add_subplot(inner[0, 0])
        ax_b2 = fig.add_subplot(inner[0, 1], sharey=ax_b1)

        draw_design_schematic(ax_a)
        draw_choice_set(ax_c)
        draw_daily_activity((ax_b1, ax_b2), daily)

        _panel_letter(ax_a, "a", x=0.0, y=0.99)
        _panel_letter(ax_c, "c", x=0.0, y=0.99)
        _panel_letter(ax_b1, "b", x=-0.14, y=1.0)
        return _save(fig, figures_dir / stem_name)


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
    """Return hidden-minus-visible contrasts for the primary estimand and admitted sensitivities."""

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


TOPIC_LABELS = {
    "recommendation_seeking": "Recommendation seeking",
    "fandom_identity": "Fandom and identity",
    "gossip_speculation": "Gossip and speculation",
    "debate_heavy": "Debate",
}


def load_rq1_topic_families(tables_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(tables_dir / "rq1_topic_family_reply_shares.csv")
    rates = frame[(frame["row_type"] == "affordance_rate") & frame["topic_affordance"].isin(TOPIC_LABELS)]
    wide = rates.pivot(index="topic_affordance", columns="group", values=["estimate", "ci_low", "ci_high", "numerator", "denominator_n"])
    wide.columns = [f"{value}_{group}" for value, group in wide.columns]
    wide["difference"] = wide["estimate_G2"] - wide["estimate_G1"]
    return wide.sort_values("difference")


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


def draw_rq1_topics(ax: plt.Axes, topics: pd.DataFrame) -> None:
    """Panel c: AI-target share of participant replies by topic family and community."""

    n = len(topics)
    ys = list(range(n))
    labels = []
    for y, (family, row) in zip(ys, topics.iterrows()):
        g1, g2 = 100 * row["estimate_G1"], 100 * row["estimate_G2"]
        shift = 0.14
        ax.plot([g1, g2], [y + shift, y - shift], color=LIGHT, linewidth=1.0, zorder=1, solid_capstyle="butt")
        for group, value, dy in (("G1", g1, shift), ("G2", g2, -shift)):
            color = GROUP_COLORS[group]
            ax.plot(
                [100 * row[f"ci_low_{group}"], 100 * row[f"ci_high_{group}"]],
                [y + dy, y + dy],
                color=color,
                linewidth=0.8,
                alpha=0.9,
                zorder=2,
            )
            ax.plot(value, y + dy, marker="o", markersize=4.5, color=color, linestyle="none", zorder=3)
        ax.text(
            1.02,
            y,
            f"+{100 * row['difference']:.1f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6,
            color=NEUTRAL,
        )
        n1, n2 = int(row["denominator_n_G1"]), int(row["denominator_n_G2"])
        labels.append(f"{TOPIC_LABELS[family]}\n$n$ = {n1} / {n2}")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, linespacing=1.1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Replies directed at an AI-authored contribution (%)")
    ax.text(1.02, 1.0, "Difference (pp)", transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=NEUTRAL)
    handles = [
        Line2D([], [], marker="o", color=GROUP_COLORS["G1"], linestyle="none", markersize=4.5, label=GROUP_LABELS["G1"]),
        Line2D([], [], marker="o", color=GROUP_COLORS["G2"], linestyle="none", markersize=4.5, label=GROUP_LABELS["G2"]),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5)


def make_figure_rq1(
    tables_dir: Path = DEFAULT_TABLES_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    *,
    stem_name: str = "figure_2_rq1_uptake",
) -> list[Path]:
    """Render the main RQ1 levels and four precisely defined contrasts."""
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


def make_all(tables_dir: Path = DEFAULT_TABLES_DIR, figures_dir: Path = DEFAULT_FIGURES_DIR) -> list[Path]:
    return make_figure_design(tables_dir, figures_dir) + make_figure_rq1(tables_dir, figures_dir)
