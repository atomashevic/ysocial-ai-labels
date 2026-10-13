"""Publication-facing figures and compact result report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import read_table
from .supplementary import write_supplementary_outputs


COLORS = {"G1": "#0072B2", "G2": "#D55E00", "H": "#4D4D4D", "AI": "#CC79A7"}


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_withheld(stem: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 2.4))
    ax.text(0.5, 0.5, label, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    _save_figure(fig, stem)


def _poisson_rate_interval(
    count: float,
    exposure_seconds: float,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return an exact Poisson interval on the per-1,000-second scale."""

    from scipy.stats import chi2

    if not np.isfinite(count) or not np.isfinite(exposure_seconds) or exposure_seconds <= 0:
        return np.nan, np.nan
    events = max(0.0, float(count))
    low_count = 0.0 if events == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2.0 * events)
    high_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (events + 1.0))
    scale = 1_000.0 / float(exposure_seconds)
    return float(low_count * scale), float(high_count * scale)


def _draw_opportunity_flow(ax: plt.Axes, replies: pd.DataFrame) -> None:
    """Draw the aggregated viewer-to-author opportunity and reply network."""

    source_y = {"G1": 0.75, "G2": 0.25}
    target_y = {
        ("G1", "H"): 0.88,
        ("G1", "AI"): 0.62,
        ("G2", "H"): 0.38,
        ("G2", "AI"): 0.12,
    }
    exposure_max = max(float(replies["verified_visible_seconds"].max()), 1.0)
    action_max = max(float(replies["linked_actions_n"].max()), 1.0)
    for group in ("G1", "G2"):
        arm = replies[replies["group"].eq(group)]
        if arm.empty:
            continue
        y_source = source_y[group]
        ax.scatter([0.0, 2.0], [y_source, y_source], s=[520, 420], color=COLORS[group], zorder=4)
        ax.text(0.0, y_source, group, color="white", ha="center", va="center", weight="bold", zorder=5)
        total_actions = int(pd.to_numeric(arm["linked_actions_n"], errors="coerce").fillna(0).sum())
        ax.text(
            2.0,
            y_source,
            str(total_actions),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            zorder=5,
        )
        for author_type in ("H", "AI"):
            selected = arm[arm["author_type"].eq(author_type)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            y_target = target_y[(group, author_type)]
            exposure = float(row["verified_visible_seconds"])
            actions = float(row["linked_actions_n"])
            ax.scatter([1.0], [y_target], s=440, color=COLORS[author_type], zorder=4)
            ax.text(1.0, y_target, author_type, color="white", ha="center", va="center", weight="bold", zorder=5)
            exposure_width = 0.8 + 7.0 * exposure / exposure_max
            action_width = 0.8 + 7.0 * actions / action_max
            ax.add_patch(
                FancyArrowPatch(
                    (0.08, y_source),
                    (0.92, y_target),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.06",
                    linewidth=exposure_width,
                    color=COLORS[author_type],
                    alpha=0.38,
                    zorder=1,
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    (1.08, y_target),
                    (1.92, y_source),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.06",
                    linewidth=action_width,
                    color=COLORS[group],
                    alpha=0.46,
                    zorder=2,
                )
            )
            ax.text(
                0.50,
                0.5 * (y_source + y_target) + 0.025,
                f"{exposure / 3_600:.1f} h",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
            ax.text(
                1.50,
                0.5 * (y_source + y_target) + 0.025,
                f"{int(actions)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
    ax.text(0.0, 1.01, "Human viewers", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.01, "Rendered author", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(1.0, 1.01, "Verified linked replies", ha="center", va="bottom", transform=ax.transAxes)
    ax.set_xlim(-0.22, 2.22)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "A. Observed browser-telemetry flow",
        loc="left",
        pad=34,
        fontsize=12,
        weight="bold",
    )
    ax.set_axis_off()


def plot_browser_conversion(summary: pd.DataFrame, stem: Path) -> None:
    required = {"group", "action_type", "author_type", "rate_per_1000_verified_seconds"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Browser conversion summary is missing columns: {missing}")
    plot = summary.copy()
    detailed = {
        "verified_visible_seconds",
        "linked_actions_n",
    }.issubset(plot.columns)
    if detailed:
        observed = plot[
            plot["action_type"].astype(str).isin(["comment", "reply"])
            & pd.to_numeric(plot["verified_visible_seconds"], errors="coerce").gt(0)
            & pd.to_numeric(plot["linked_actions_n"], errors="coerce").notna()
        ].copy()
        observed = observed.drop_duplicates(["group", "author_type"])
        if set(observed["group"]) == {"G1", "G2"} and not observed.empty:
            observed["rate"] = pd.to_numeric(
                observed["rate_per_1000_verified_seconds"], errors="coerce"
            )
            intervals = [
                _poisson_rate_interval(count, exposure)
                for count, exposure in zip(
                    observed["linked_actions_n"], observed["verified_visible_seconds"]
                )
            ]
            observed["rate_ci_low"] = [value[0] for value in intervals]
            observed["rate_ci_high"] = [value[1] for value in intervals]
            fig, (flow_ax, rate_ax) = plt.subplots(
                1,
                2,
                figsize=(11.2, 4.9),
                gridspec_kw={"width_ratios": [1.35, 1.0]},
            )
            _draw_opportunity_flow(flow_ax, observed)
            labels = ["H-authored", "AI-authored"]
            x = np.arange(2, dtype=float)
            width = 0.34
            for offset, group in zip((-width / 2, width / 2), ("G1", "G2")):
                arm = observed[observed["group"].eq(group)].set_index("author_type")
                values = np.asarray(
                    [float(arm.loc[kind, "rate"]) if kind in arm.index else np.nan for kind in ("H", "AI")]
                )
                lows = np.asarray(
                    [float(arm.loc[kind, "rate_ci_low"]) if kind in arm.index else np.nan for kind in ("H", "AI")]
                )
                highs = np.asarray(
                    [float(arm.loc[kind, "rate_ci_high"]) if kind in arm.index else np.nan for kind in ("H", "AI")]
                )
                rate_ax.bar(x + offset, values, width, color=COLORS[group], label=group)
                rate_ax.errorbar(
                    x + offset,
                    values,
                    yerr=np.vstack([values - lows, highs - values]),
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    linewidth=1.0,
                )
            rate_ax.set_xticks(x, labels)
            rate_ax.set_ylabel("Replies per 1,000 verified visible-seconds")
            rate_ax.set_title(
                "B. Verified linked-reply rate among observed telemetry",
                loc="left",
                fontsize=12,
                weight="bold",
            )
            rate_ax.legend(frameon=False, title="Arm", loc="upper right")
            rate_ax.spines[["top", "right"]].set_visible(False)
            rate_ax.text(
                0.0,
                -0.23,
                (
                    "Exact Poisson 95% intervals cover linked counts only; linkage-selection "
                    "uncertainty and cross-arm inference are excluded. Tracking inference "
                    "gates failed unless the accompanying QC table states otherwise."
                ),
                transform=rate_ax.transAxes,
                fontsize=8,
            )
            fig.tight_layout()
            _save_figure(fig, stem)
            return

    plot["label"] = plot["action_type"].astype(str) + " → " + plot["author_type"].astype(str)
    labels = list(dict.fromkeys(plot["label"]))
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    has_exposure = "verified_visible_seconds" in plot.columns
    if has_exposure:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(10.0, len(labels) * 1.25), 4.8),
            gridspec_kw={"width_ratios": [0.85, 2.15]},
        )
        exposure_ax, ax = axes
        exposure = plot.drop_duplicates(["group", "author_type"])
        pivot = exposure.pivot(index="group", columns="author_type", values="verified_visible_seconds")
        pivot = pivot.reindex(index=["G1", "G2"], columns=["H", "AI"]).fillna(0.0)
        shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
        bottom = np.zeros(len(shares))
        for author_type in ["H", "AI"]:
            values = shares[author_type].to_numpy(dtype=float)
            exposure_ax.bar(
                shares.index,
                values,
                bottom=bottom,
                color=COLORS[author_type],
                label=author_type,
            )
            bottom += np.nan_to_num(values)
        exposure_ax.set_ylabel("Share of verified visible time")
        exposure_ax.set_xlabel("Arm")
        exposure_ax.set_ylim(0, 1)
        exposure_ax.legend(frameon=False, title="Author")
        exposure_ax.spines[["top", "right"]].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.2), 4.8))
    for offset, group in zip((-width / 2, width / 2), ("G1", "G2")):
        arm = plot[plot["group"].eq(group)].set_index("label")
        values = [float(arm.loc[label, "rate_per_1000_verified_seconds"]) if label in arm.index else np.nan for label in labels]
        ax.bar(x + offset, values, width, label=group, color=COLORS[group])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Actions per 1,000 verified visible-seconds")
    ax.set_xlabel("")
    ax.legend(frameon=False, title="Arm")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, stem)


def plot_primary_estimands(estimands: pd.DataFrame, stem: Path) -> None:
    required = {"estimand", "posterior_median", "ci_low", "ci_high"}
    missing = sorted(required.difference(estimands.columns))
    if missing:
        raise ValueError(f"Primary estimands are missing columns: {missing}")
    plot = estimands.dropna(subset=["posterior_median", "ci_low", "ci_high"]).copy()
    focal_names = {
        "common_opportunity_ai_choice_probability",
        "G2_minus_G1_common_opportunity_ai_choice_probability",
        "AI_account_return_probability_difference_one_prior_reply_24h",
        "AI_account_response_probability_difference_one_reverse_reply_24h",
    }
    focal = plot[plot["estimand"].astype(str).isin(focal_names)]
    if not focal.empty:
        plot = focal.copy()
    display_names = {
        "common_opportunity_ai_choice_probability": "AI target probability",
        "G2_minus_G1_common_opportunity_ai_choice_probability": (
            "G2−G1 AI target probability"
        ),
        "AI_account_return_probability_difference_one_prior_reply_24h": (
            "Controlled repetition-history projection (24 h)"
        ),
        "AI_account_response_probability_difference_one_reverse_reply_24h": (
            "Controlled reverse-history projection (24 h)"
        ),
    }
    plot["estimand_label"] = plot["estimand"].map(display_names).fillna(
        plot["estimand"].astype(str)
    )
    if "group" in plot.columns:
        plot["display_estimand"] = (
            plot["group"].astype(str) + " · " + plot["estimand_label"]
        )
    else:
        plot["display_estimand"] = plot["estimand_label"]
    plot = plot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    center = plot["posterior_median"].to_numpy(dtype=float)
    low = center - plot["ci_low"].to_numpy(dtype=float)
    high = plot["ci_high"].to_numpy(dtype=float) - center
    colors = [COLORS.get(str(value), "#333333") for value in plot.get("group", pd.Series("", index=plot.index))]
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.52 * len(plot))))
    ax.errorbar(center, y, xerr=np.vstack([low, high]), fmt="none", ecolor="#555555", capsize=3, linewidth=1.2)
    ax.scatter(center, y, c=colors, s=38, zorder=3)
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, plot["display_estimand"])
    ax.set_xlabel("Posterior estimate with 95% credible interval")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save_figure(fig, stem)


def plot_posterior_predictive(checks: pd.DataFrame, stem: Path) -> None:
    required = {"group", "metric", "observed", "posterior_median", "ci_low", "ci_high"}
    missing = sorted(required.difference(checks.columns))
    if missing:
        raise ValueError(f"Posterior predictive checks are missing columns: {missing}")
    plot = checks.copy()
    focal_metrics = {
        "h_to_ai_share",
        "distinct_ai_alters",
        "repeat_contact_share",
        "top_alter_concentration",
        "reciprocal_context_share",
        "two_path_closure_mean",
    }
    focal = plot[plot["metric"].astype(str).isin(focal_metrics)]
    if not focal.empty:
        plot = focal.copy()
    metric_labels = {
        "h_to_ai_share": "H→AI share",
        "distinct_ai_alters": "Distinct AI alters",
        "repeat_contact_share": "Repeated-contact share",
        "top_alter_concentration": "Top-alter concentration",
        "reciprocal_context_share": "Reciprocal-context share",
        "two_path_closure_mean": "Two-path closure",
    }
    display_metric = plot["metric"].map(metric_labels).fillna(plot["metric"].astype(str))
    plot["label"] = (
        plot["group"].astype(str)
        + " · "
        + display_metric
        + " (observed "
        + plot["observed"].map(lambda value: f"{float(value):.3g}")
        + ")"
    )
    plot = plot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    observed = plot["observed"].to_numpy(dtype=float)
    if np.any(~np.isfinite(observed) | np.isclose(observed, 0.0)):
        raise ValueError("Posterior predictive ratio plot requires finite nonzero observations")
    center = plot["posterior_median"].to_numpy(dtype=float) / observed
    low = center - plot["ci_low"].to_numpy(dtype=float) / observed
    high = plot["ci_high"].to_numpy(dtype=float) / observed - center
    fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.48 * len(plot))))
    ax.errorbar(center, y, xerr=np.vstack([low, high]), fmt="o", color="#666666", capsize=3, label="Posterior prediction")
    ax.scatter(np.ones(len(plot)), y, marker="|", s=130, linewidth=2, color="#000000", label="Observed")
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Posterior prediction divided by the observed statistic")
    ax.legend(frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save_figure(fig, stem)


def plot_topic_heterogeneity(summary: pd.DataFrame, stem: Path) -> None:
    """Plot conditional AI-versus-human candidate odds by topic and arm."""

    required = {
        "topic",
        "group",
        "conditional_odds_ratio_median",
        "conditional_odds_ratio_ci_low",
        "conditional_odds_ratio_ci_high",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Topic heterogeneity summary is missing columns: {missing}")
    plot = summary.dropna(subset=list(required)).copy()
    if plot.empty:
        raise ValueError("Topic heterogeneity summary has no released estimates")
    plot["label"] = plot["topic"].astype(str) + " · " + plot["group"].astype(str)
    plot = plot.sort_values(
        ["conditional_odds_ratio_median", "topic", "group"],
        kind="stable",
    ).reset_index(drop=True)
    y = np.arange(len(plot))
    center = plot["conditional_odds_ratio_median"].to_numpy(dtype=float)
    low = center - plot["conditional_odds_ratio_ci_low"].to_numpy(dtype=float)
    high = plot["conditional_odds_ratio_ci_high"].to_numpy(dtype=float) - center
    colors = [COLORS.get(str(group), "#555555") for group in plot["group"]]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.34 * len(plot))))
    ax.errorbar(
        center,
        y,
        xerr=np.vstack([low, high]),
        fmt="none",
        ecolor="#666666",
        capsize=2.5,
        linewidth=1.0,
    )
    ax.scatter(center, y, c=colors, s=30, zorder=3)
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, plot["label"])
    ax.set_xlabel("Conditional odds ratio for selecting an AI-authored candidate")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save_figure(fig, stem)


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return "No result was produced."
    return frame[available].to_markdown(index=False)


def _readiness_gate_passed(readiness: pd.DataFrame, gate: str) -> bool:
    selected = readiness[
        readiness.get("gate", pd.Series(dtype=object)).eq(gate)
    ]
    return bool(
        len(selected) == 1
        and selected.get("passed", pd.Series(False, index=selected.index))
        .fillna(False)
        .astype(bool)
        .all()
    )


def _primary_release_passed(readiness: pd.DataFrame) -> bool:
    return _readiness_gate_passed(readiness, "primary_estimates_release")


def build_report(
    config: AnalysisConfig,
    estimands: pd.DataFrame,
    browser_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    predictive_checks: pd.DataFrame,
    integrity: pd.DataFrame | None = None,
    source_availability: pd.DataFrame | None = None,
    tracking_qc: pd.DataFrame | None = None,
    hazard_terms: pd.DataFrame | None = None,
    browser_sensitivity: pd.DataFrame | None = None,
    readiness: pd.DataFrame | None = None,
    attrition: pd.DataFrame | None = None,
    risk_set_composition: pd.DataFrame | None = None,
    prior_sensitivity: pd.DataFrame | None = None,
    forward_comparisons: pd.DataFrame | None = None,
    participant_psis: pd.DataFrame | None = None,
    participant_refits: pd.DataFrame | None = None,
    participant_reloo_summary: pd.DataFrame | None = None,
    latent_status: pd.DataFrame | None = None,
    event_timing_checks: pd.DataFrame | None = None,
    topic_heterogeneity: pd.DataFrame | None = None,
    supplementary_inventory: pd.DataFrame | None = None,
) -> str:
    integrity = pd.DataFrame() if integrity is None else integrity
    source_availability = pd.DataFrame() if source_availability is None else source_availability
    tracking_qc = pd.DataFrame() if tracking_qc is None else tracking_qc
    hazard_terms = pd.DataFrame() if hazard_terms is None else hazard_terms
    browser_sensitivity = (
        pd.DataFrame() if browser_sensitivity is None else browser_sensitivity
    )
    readiness_table = pd.DataFrame() if readiness is None else readiness
    attrition = pd.DataFrame() if attrition is None else attrition
    risk_set_composition = (
        pd.DataFrame() if risk_set_composition is None else risk_set_composition
    )
    prior_sensitivity = pd.DataFrame() if prior_sensitivity is None else prior_sensitivity
    forward_comparisons = (
        pd.DataFrame() if forward_comparisons is None else forward_comparisons
    )
    participant_psis = pd.DataFrame() if participant_psis is None else participant_psis
    participant_refits = pd.DataFrame() if participant_refits is None else participant_refits
    participant_reloo_summary = (
        pd.DataFrame()
        if participant_reloo_summary is None
        else participant_reloo_summary
    )
    latent_status = pd.DataFrame() if latent_status is None else latent_status
    event_timing_checks = (
        pd.DataFrame() if event_timing_checks is None else event_timing_checks
    )
    topic_heterogeneity = (
        pd.DataFrame() if topic_heterogeneity is None else topic_heterogeneity
    )
    supplementary_inventory = (
        pd.DataFrame() if supplementary_inventory is None else supplementary_inventory
    )
    model_ready = _primary_release_passed(readiness_table)
    network_claims_ready = _readiness_gate_passed(
        readiness_table,
        "network_history_predictive_gain_over_M1",
    )
    readiness_message = (
        "Every required release gate passed. The primary posterior tables are eligible for manuscript interpretation."
        if model_ready
        else (
            "At least one required release gate failed or remains incomplete. "
            "Primary posterior estimates and model-based figures are withheld from this report."
        )
    )
    displayed_estimands = estimands
    if model_ready and not network_claims_ready and "estimand" in estimands.columns:
        displayed_estimands = estimands[
            ~estimands["estimand"].astype(str).str.startswith("AI_account_")
        ]
    estimand_table = (
        _markdown_table(
            displayed_estimands,
            [
                "model",
                "group",
                "estimand",
                "weighting",
                "posterior_median",
                "ci_low",
                "ci_high",
                "posterior_probability_gt_zero",
            ],
        )
        if model_ready
        else "Withheld pending the required release gates."
    )
    network_claim_message = (
        "Network-history projections remain withheld with the other model-based results."
        if not model_ready
        else (
            "Repetition and reverse-history projections passed their forward-prediction claim gate."
            if network_claims_ready
            else (
            "Repetition and reverse-history projections are withheld because M3 did not "
            "improve future-event prediction over M1 by at least two standard errors."
            )
        )
    )
    predictive_table = (
        _markdown_table(
            predictive_checks,
            ["group", "metric", "observed", "posterior_median", "ci_low", "ci_high", "covered"],
        )
        if model_ready
        else "Withheld pending the required release gates."
    )
    withheld = "Withheld pending the required release gates."
    browser_sensitivity_table = (
        _markdown_table(
            browser_sensitivity,
            [
                "estimand",
                "group",
                "posterior_median_full_thread",
                "posterior_median_strict_browser",
                "posterior_median_difference_strict_minus_full",
                "target_choice_tracking_gate_passed",
            ],
        )
        if model_ready
        else withheld
    )
    diagnostics_table = (
        _markdown_table(
            diagnostics,
            ["model", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "passed"],
        )
        if model_ready
        else withheld
    )
    prior_table = (
        _markdown_table(
            prior_sensitivity,
            [
                "estimand",
                "group",
                "absolute_median_shift",
                "max_allowed_absolute_median_shift",
                "credible_intervals_overlap",
                "evidence_class_stable",
                "passed",
            ],
        )
        if model_ready
        else withheld
    )
    forward_table = (
        _markdown_table(
            forward_comparisons,
            [
                "fold",
                "reference_model",
                "comparator_model",
                "test_choice_sets",
                "elpd_difference",
                "elpd_difference_se",
                "gain_at_least_two_se",
                "mean_rank_improvement",
                "mean_brier_improvement",
            ],
        )
        if model_ready
        else withheld
    )
    participant_table = (
        _markdown_table(
            participant_psis,
            ["group", "actor_id", "choice_sets", "pareto_k", "needs_refit"],
        )
        if model_ready
        else withheld
    )
    reloo_table = (
        _markdown_table(
            participant_reloo_summary,
            [
                "participants",
                "flagged_participants",
                "exact_refits",
                "grouped_reloo_elpd",
                "participant_cluster_se",
                "all_problematic_contributions_replaced",
                "history_policy",
            ],
        )
        if model_ready
        else withheld
    )
    latent_table = (
        _markdown_table(
            latent_status,
            [
                "k",
                "smallest_block_n",
                "mean_membership_stability",
                "confident_membership_share",
                "mean_normalized_membership_entropy",
                "opportunity_activity_stress_test_pass",
                "collapsed_or_tiny_block",
                "manuscript_prominence_pass",
                "gate_status",
            ],
        )
        if model_ready
        else withheld
    )
    authenticated_refit_count = (
        int(
            participant_refits.get(
                "authenticated_refit_contribution", pd.Series(dtype=bool)
            )
            .fillna(False)
            .sum()
        )
        if model_ready
        else 0
    )
    return f"""# Opportunity-to-Network Publication Analysis

## Readiness status

{readiness_message}

{_markdown_table(readiness_table, ['gate', 'passed', 'required_for_primary_release', 'observed'])}

## Empirical judgment

The primary analysis treats each human reply as a choice among earlier valid content items in the same thread. Browser episodes provide a stricter observed-opportunity layer. Repetition, reciprocity, receiver popularity, and shared-thread position are calculated from the network state before each reply.

## Primary estimands

{estimand_table}

{network_claim_message}

### Reply-to-risk-set flow

{_markdown_table(attrition, ['group', 'outcome', 'human_reply_events', 'all_human_replies', 'share'])}

{_markdown_table(risk_set_composition, ['group', 'composition', 'choice_sets', 'candidate_rows'])}

## Browser opportunity and conversion

{_markdown_table(browser_summary, ['group', 'action_type', 'author_type', 'verified_visible_seconds', 'linked_actions_n', 'rate_per_1000_verified_seconds', 'action_rate_ci_low', 'action_rate_ci_high', 'standardized_rate_per_1000_verified_seconds', 'standardized_rate_ci_low', 'standardized_rate_ci_high', 'standardization_status', 'inference_status'])}

The rendered-target gate evaluates whether linked actions have valid pre-action impressions. The interval-hazard gate additionally requires exact or audit-approved episode closure. Arm coefficients remain ecological descriptions because participant clustering cannot supply independent communities within a condition. Poisson intervals cover linked action counts and omit uncertainty from selective telemetry linkage.

### Tracking decision gates

{_markdown_table(tracking_qc, ['metric', 'group', 'numerator', 'denominator', 'value', 'threshold', 'gate_pass', 'observed_status'])}

### Verified-interval hazard model

{_markdown_table(hazard_terms, ['action_type', 'term', 'hazard_ratio', 'ci_low', 'ci_high', 'p_value', 'inference_status'])}

### Full-thread versus rendered-risk-set sensitivity

{browser_sensitivity_table}

## Model diagnostics

{diagnostics_table}

## Recursive network-process checks

{predictive_table}

These simulations retain observed event times and thread opportunity sets. Each simulated human target updates repetition, reciprocity, receiver popularity, and two-path closure before the next reply. AI events and human replies outside the retained choice sample update the state at their observed targets. The simulation does not generate new event times or new thread opportunities.

### Companion event-timing calibration

{_markdown_table(event_timing_checks, ['group', 'metric', 'observed', 'posterior_predictive_median', 'ci_low', 'ci_high', 'covered', 'posterior_draws'])}

The timing companion uses a group-specific Bayesian transition histogram on inter-reply gaps measured by the valid-analysis clock. It conditions on the observed number of replies and checks gap distributions and short-range sequence dependence. It does not change the target-choice likelihood into a generative event-rate model.

## Robustness and predictive validation

### Doubled prior scales

{prior_table}

### Rolling-origin comparisons

{forward_table}

### Participant-grouped influence

{participant_table}

{reloo_table}

Authenticated participant refits: {authenticated_refit_count}. Each grouped contribution integrates one shared new-participant AI-selection slope across that participant's held-out replies. The deletion score conditions on the realized earlier network history, including earlier actions by the deleted participant.

### Latent receiver blocks

{latent_table}

## Supplementary evidence

### Hierarchical topic heterogeneity

{_markdown_table(topic_heterogeneity, ['topic', 'group', 'conditional_odds_ratio_median', 'conditional_odds_ratio_ci_low', 'conditional_odds_ratio_ci_high', 'posterior_draws', 'evidence_status'])}

Topic rows compare otherwise equal AI- and human-authored candidates within an observed same-thread choice set, with actor and receiver effects at their population means. They are conditional coefficient-scale comparisons, not marginal topic probabilities or causal arm effects.

### Retained and unavailable analyses

{_markdown_table(supplementary_inventory, ['supplement', 'analysis_role', 'canonical_status', 'source_window_status', 'recommended_output', 'claim_boundary'])}

The legacy rows are frozen, hash-indexed appendix sources. Their estimates are not pooled with the opportunity-conditioned model. Untimestamped reaction marks and unvalidated semantic reply valence remain explicit evidence gaps.

## Input and risk-set integrity

{_markdown_table(integrity, ['check', 'passed', 'observed', 'required'])}

### Dynamic-source availability

{_markdown_table(source_availability, ['group', 'source', 'available_rows', 'total_rows', 'coverage', 'status'])}

## Claim boundary

The two arms are separate communities with different participant cohorts. G2–G1 estimates are ecological, opportunity-standardized contrasts. Repetition and reciprocity terms describe path-dependent target-selection associations conditional on the observed reply and opportunity history. Arm-level causal label effects are outside this design's identified estimands.

## Reproducibility

- Analysis set: `{config.analysis_set}`
- Human-entry cutoff: `{config.start_time}`
- Run ID: `{config.run_id}`
- Source and software hashes: `manifest.json`
- Network object audit: `reports/network-audit.md`
"""


def _optional_table(stem: Path) -> pd.DataFrame:
    try:
        return read_table(stem)
    except FileNotFoundError:
        return pd.DataFrame()


def run_report(config: AnalysisConfig) -> Path:
    estimands = _optional_table(config.tables_dir / "primary_estimands")
    browser_summary = _optional_table(config.tables_dir / "browser_conversion_summary")
    diagnostics = _optional_table(config.tables_dir / "model_diagnostics")
    predictive = _optional_table(config.tables_dir / "posterior_predictive_checks")
    integrity = _optional_table(config.tables_dir / "data_integrity_checks")
    availability = _optional_table(config.tables_dir / "source_availability")
    tracking_qc = _optional_table(config.tables_dir / "tracking_linkage_qc")
    hazard_terms = _optional_table(config.tables_dir / "browser_hazard_terms")
    browser_sensitivity = _optional_table(
        config.tables_dir / "browser_full_vs_strict_estimand_comparison"
    )
    readiness = _optional_table(config.tables_dir / "publication_readiness")
    attrition = _optional_table(config.tables_dir / "choice_set_attrition")
    risk_composition = _optional_table(config.tables_dir / "risk_set_composition")
    prior_sensitivity = _optional_table(config.tables_dir / "prior_scale_sensitivity")
    forward_comparisons = _optional_table(config.tables_dir / "forward_validation_comparisons")
    participant_psis = _optional_table(
        config.tables_dir / "participant_grouped_psis_validated"
    )
    participant_refits = _optional_table(
        config.tables_dir / "participant_loo_refit_authentication"
    )
    participant_reloo_summary = _optional_table(
        config.tables_dir / "participant_grouped_reloo_summary"
    )
    latent_status = _optional_table(config.tables_dir / "latent_block_gate_status")
    event_timing_checks = _optional_table(
        config.tables_dir / "event_timing_posterior_checks"
    )
    primary_release = _primary_release_passed(readiness)
    network_claims_release = _readiness_gate_passed(
        readiness,
        "network_history_predictive_gain_over_M1",
    )
    supplementary = write_supplementary_outputs(
        config,
        release_eligible=primary_release,
    )
    topic_heterogeneity = supplementary["supplementary_topic_heterogeneity"]
    supplementary_inventory = supplementary["supplementary_source_inventory"]
    if not browser_summary.empty:
        plot_browser_conversion(browser_summary, config.figures_dir / "figure_1_browser_conversion")
    if primary_release and not estimands.empty:
        figure_estimands = estimands
        if not network_claims_release and "estimand" in figure_estimands.columns:
            figure_estimands = figure_estimands[
                ~figure_estimands["estimand"].astype(str).str.startswith("AI_account_")
            ]
        plot_primary_estimands(
            figure_estimands,
            config.figures_dir / "figure_2_network_process_estimands",
        )
    elif not primary_release:
        _plot_withheld(
            config.figures_dir / "figure_2_network_process_estimands",
            "Model-based estimands withheld pending all required release gates.",
        )
    if primary_release and not predictive.empty:
        plot_posterior_predictive(predictive, config.figures_dir / "figure_3_posterior_network_checks")
    elif not primary_release:
        _plot_withheld(
            config.figures_dir / "figure_3_posterior_network_checks",
            "Recursive network-process checks withheld pending all required release gates.",
        )
    topic_available = (
        not topic_heterogeneity.empty
        and topic_heterogeneity["evidence_status"]
        .eq("available_signed_publication_length_M3")
        .all()
    )
    if primary_release and topic_available:
        plot_topic_heterogeneity(
            topic_heterogeneity,
            config.figures_dir / "supplement_figure_topic_heterogeneity",
        )
    else:
        _plot_withheld(
            config.figures_dir / "supplement_figure_topic_heterogeneity",
            "Topic heterogeneity withheld pending a signed M3 posterior and all primary release gates.",
        )
    report = build_report(
        config,
        estimands,
        browser_summary,
        diagnostics,
        predictive,
        integrity,
        availability,
        tracking_qc,
        hazard_terms,
        browser_sensitivity,
        readiness,
        attrition,
        risk_composition,
        prior_sensitivity,
        forward_comparisons,
        participant_psis,
        participant_refits,
        participant_reloo_summary,
        latent_status,
        event_timing_checks,
        topic_heterogeneity,
        supplementary_inventory,
    )
    path = config.reports_dir / "publication_analysis.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
