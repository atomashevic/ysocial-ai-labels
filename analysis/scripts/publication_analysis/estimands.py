"""Publication estimands derived from signed conditional-choice posteriors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .model import (
    _common_arm_frames,
    _posterior_summary,
    _posterior_utilities,
    standardized_estimands as _base_standardized_estimands,
)


def _beta_draws(idata: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    values = idata.posterior["beta"]
    terms = tuple(str(value) for value in values.coords["term"].values)
    draws = values.stack(sample=("chain", "draw")).transpose("sample", "term").values
    return draws, terms


def _account_specific_probability_change(
    idata: Any,
    data: Any,
    arm_x: np.ndarray,
    term_values: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, int]:
    """Toggle one eligible AI receiver at a time and average its selection change."""

    beta, posterior_terms = _beta_draws(idata)
    selected_terms = [term for term in term_values if term in posterior_terms]
    if not selected_terms:
        raise ValueError("Account-specific projection has no fitted history terms")
    beta_indexes = [posterior_terms.index(term) for term in selected_terms]
    data_indexes = [data.terms.index(term) for term in selected_terms]
    low_values = np.asarray([term_values[term][0] for term in selected_terms], dtype=float)
    high_values = np.asarray([term_values[term][1] for term in selected_terms], dtype=float)
    utility = _posterior_utilities(idata, data, x_override=arm_x)
    total = np.zeros(utility.shape[0], dtype=float)
    opportunities = 0
    for choice_index in range(data.n_choices):
        positions = np.flatnonzero(data.mask[choice_index])
        receivers = data.receiver_index[choice_index, positions]
        ai_positions = positions[data.candidate_is_ai[choice_index, positions].astype(bool)]
        for receiver in np.unique(data.receiver_index[choice_index, ai_positions]):
            focal = positions[receivers == receiver]
            other = positions[receivers != receiver]
            if not len(other):
                continue
            observed = arm_x[choice_index, focal][:, data_indexes]
            low_delta = (low_values[None, :] - observed) @ beta[:, beta_indexes].T
            high_delta = (high_values[None, :] - observed) @ beta[:, beta_indexes].T
            focal_utility = utility[:, choice_index, focal]
            log_other = logsumexp(utility[:, choice_index, other], axis=1)
            low_log_mass = logsumexp(focal_utility + low_delta.T, axis=1)
            high_log_mass = logsumexp(focal_utility + high_delta.T, axis=1)
            total += expit(high_log_mass - log_other) - expit(low_log_mass - log_other)
            opportunities += 1
    if not opportunities:
        raise ValueError("No AI receiver competed with another receiver in the risk sets")
    return total / opportunities, opportunities


def _account_projection_rows(
    idata: Any,
    data: Any,
    g1_x: np.ndarray,
    g2_x: np.ndarray,
    counterfactual_values: Mapping[str, tuple[float, float]],
    terms: Sequence[str],
    estimand: str,
) -> list[dict[str, Any]]:
    values = {term: counterfactual_values[term] for term in terms}
    g1, g1_opportunities = _account_specific_probability_change(
        idata, data, g1_x, values
    )
    g2, g2_opportunities = _account_specific_probability_change(
        idata, data, g2_x, values
    )
    rows: list[dict[str, Any]] = []
    for group, draws in (("G1", g1), ("G2", g2), ("G2-G1", g2 - g1)):
        row = _posterior_summary(draws, estimand, group)
        row.update(
            {
                "source_term": ",".join(terms),
                "standardization_scope": "pooled_choice_by_AI_receiver_opportunities",
                "counterfactual": (
                    "Toggle one eligible AI receiver from no prior contact to one "
                    "account-level contact 24 hours earlier; move all eligible content "
                    "from that receiver together; hold other receiver histories observed"
                ),
                "estimand_interpretation": "model_projection_association_not_causal_intervention",
                "g1_ai_receiver_opportunities": g1_opportunities,
                "g2_ai_receiver_opportunities": g2_opportunities,
            }
        )
        rows.append(row)
    return rows


def standardized_estimands(
    idata: Any,
    data: Any,
    *,
    counterfactual_values: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Return arm projections plus candidate-specific return and response effects."""

    base = _base_standardized_estimands(
        idata,
        data,
        counterfactual_values=counterfactual_values,
    )
    obsolete = {
        "AI_candidate_prior_contact_zero_to_one_probability_difference",
        "AI_candidate_recent_reverse_zero_to_one_probability_difference",
    }
    base = base[~base["estimand"].isin(obsolete)].copy()
    g1_x, g2_x = _common_arm_frames(data)
    _, posterior_terms = _beta_draws(idata)
    fitted_terms = set(posterior_terms)
    rows: list[dict[str, Any]] = []
    repetition_terms = (
        "z_log_prior_contact_count",
        "z_recent_prior_contact",
    )
    if fitted_terms.intersection(repetition_terms):
        rows.extend(
            _account_projection_rows(
                idata,
                data,
                g1_x,
                g2_x,
                counterfactual_values,
                repetition_terms,
                "AI_account_return_probability_difference_one_prior_reply_24h",
            )
        )
    reverse_terms = (
        "z_log_reverse_contact_count",
        "z_recent_reverse_contact",
    )
    if fitted_terms.intersection(reverse_terms):
        rows.extend(
            _account_projection_rows(
                idata,
                data,
                g1_x,
                g2_x,
                counterfactual_values,
                reverse_terms,
                "AI_account_response_probability_difference_one_reverse_reply_24h",
            )
        )
    if not rows:
        return base.reset_index(drop=True)
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


__all__ = ["standardized_estimands"]
