"""Companion event-timing calibration for the conditional target-choice model."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _timing_metrics(gaps: np.ndarray) -> dict[str, float]:
    values = np.asarray(gaps, dtype=float)
    log_values = np.log1p(values)
    lag_correlation = float(np.corrcoef(log_values[:-1], log_values[1:])[0, 1])
    return {
        "median_gap_seconds": float(np.median(values)),
        "q90_gap_seconds": float(np.quantile(values, 0.90)),
        "burst_gap_share_le_300_seconds": float(np.mean(values <= 300.0)),
        "long_gap_share_ge_3600_seconds": float(np.mean(values >= 3_600.0)),
        "consecutive_burst_pair_share": float(
            np.mean((values[:-1] <= 300.0) & (values[1:] <= 300.0))
        ),
        "lag1_log_gap_correlation": lag_correlation,
    }


def _simulate_markov_histogram(
    gaps: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw gap sequences from a Dirichlet-Markov empirical histogram."""

    boundaries = np.asarray(
        [
            0.0,
            30.0,
            60.0,
            120.0,
            300.0,
            600.0,
            1_800.0,
            3_600.0,
            7_200.0,
            14_400.0,
            43_200.0,
            np.inf,
        ]
    )
    raw_state = np.digitize(gaps, boundaries[1:-1], right=True)
    active_states = np.unique(raw_state)
    state_lookup = {int(value): index for index, value in enumerate(active_states)}
    states = np.asarray([state_lookup[int(value)] for value in raw_state], dtype=np.int64)
    state_count = len(active_states)
    marginal_counts = np.bincount(states, minlength=state_count)
    transition_counts = np.zeros((state_count, state_count), dtype=float)
    np.add.at(transition_counts, (states[:-1], states[1:]), 1.0)
    prior = 0.5
    initial_probabilities = rng.dirichlet(marginal_counts + prior, size=draws)
    transition_gamma = rng.gamma(
        shape=transition_counts[None, :, :] + prior,
        scale=1.0,
        size=(draws, state_count, state_count),
    )
    transition_probabilities = transition_gamma / transition_gamma.sum(axis=2, keepdims=True)
    simulated_states = np.empty((draws, len(gaps)), dtype=np.int16)
    uniforms = rng.random(draws)
    simulated_states[:, 0] = (
        uniforms[:, None] > np.cumsum(initial_probabilities, axis=1)
    ).sum(axis=1)
    draw_index = np.arange(draws)
    for position in range(1, len(gaps)):
        current = simulated_states[:, position - 1]
        probabilities = transition_probabilities[draw_index, current, :]
        uniforms = rng.random(draws)
        simulated_states[:, position] = (
            uniforms[:, None] > np.cumsum(probabilities, axis=1)
        ).sum(axis=1)

    simulated = np.empty((draws, len(gaps)), dtype=float)
    for state in range(state_count):
        observed_values = gaps[states == state]
        selected = simulated_states == state
        simulated[selected] = rng.choice(
            observed_values,
            size=int(selected.sum()),
            replace=True,
        )
    return simulated


def _rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def event_timing_posterior_checks(
    event_history: pd.DataFrame,
    *,
    draws: int = 2_000,
    random_seed: int = 20260715,
) -> pd.DataFrame:
    """Check human reply gaps with a group-specific Bayesian transition model.

    This companion check conditions on the observed number of human replies. It
    assesses whether a simple arm-specific timing process reproduces the gap
    distribution and short-range sequence dependence on the compressed
    valid-analysis clock. It does not turn the conditional target-choice
    likelihood into a full generative event process.
    """

    required = {
        "group",
        "event_type",
        "human_originated",
        "valid_clock_seconds",
    }
    missing = sorted(required.difference(event_history.columns))
    if missing:
        raise ValueError(f"Event timing requires columns: {missing}")
    if draws < 100:
        raise ValueError("Event timing requires at least 100 posterior draws")
    rng = np.random.default_rng(random_seed)
    human = event_history[
        event_history["human_originated"].astype(bool)
        & event_history["event_type"].eq("comments_replies")
    ].copy()
    rows: list[dict[str, Any]] = []
    for group, arm in human.groupby("group", sort=True):
        clock = np.sort(
            pd.to_numeric(arm["valid_clock_seconds"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        gaps = np.diff(clock)
        gaps = gaps[np.isfinite(gaps) & (gaps >= 0.0)]
        simulated_gaps = _simulate_markov_histogram(
            gaps,
            draws=draws,
            rng=rng,
        )
        observed = _timing_metrics(gaps)
        metric_functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "median_gap_seconds": lambda values: np.median(values, axis=1),
            "q90_gap_seconds": lambda values: np.quantile(values, 0.90, axis=1),
            "burst_gap_share_le_300_seconds": lambda values: np.mean(
                values <= 300.0, axis=1
            ),
            "long_gap_share_ge_3600_seconds": lambda values: np.mean(
                values >= 3_600.0, axis=1
            ),
            "consecutive_burst_pair_share": lambda values: np.mean(
                (values[:, :-1] <= 300.0) & (values[:, 1:] <= 300.0),
                axis=1,
            ),
            "lag1_log_gap_correlation": lambda values: _rowwise_correlation(
                np.log1p(values[:, :-1]), np.log1p(values[:, 1:])
            ),
        }
        for metric, function in metric_functions.items():
            simulated = np.asarray(function(simulated_gaps), dtype=float)
            low = float(np.quantile(simulated, 0.025))
            high = float(np.quantile(simulated, 0.975))
            observed_value = observed[metric]
            rows.append(
                {
                    "group": str(group),
                    "metric": metric,
                    "human_reply_events": int(len(clock)),
                    "inter_event_gaps": int(len(gaps)),
                    "observed": observed_value,
                    "posterior_predictive_median": float(np.median(simulated)),
                    "ci_low": low,
                    "ci_high": high,
                    "covered": bool(low <= observed_value <= high),
                    "posterior_draws": draws,
                    "simulation_scope": (
                        "group_specific_dirichlet_markov_gap_histogram_conditional_"
                        "on_event_count_and_valid_analysis_clock"
                    ),
                    "claim_boundary": (
                        "timing calibration only; target choice remains conditional "
                        "on observed event times"
                    ),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["event_timing_posterior_checks"]
