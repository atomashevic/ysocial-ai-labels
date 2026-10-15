"""RQ2 support summaries, probability projections, validation, and reporting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .config import AnalysisConfig
from .model import _common_arm_frames, _posterior_summary, _posterior_utilities


REFERENCE_CONTACT_HOURS = 24.0
ESTIMAND_PRIOR_HUMAN_TO_AI = (
    "AI_account_selection_probability_difference_prior_human_to_AI_reply_24h"
)
ESTIMAND_PRIOR_AI_TO_HUMAN = (
    "AI_account_selection_probability_difference_prior_AI_to_human_reply_24h"
)
RQ2_DIRECTIONS: Mapping[str, tuple[str, str, str]] = {
    "human_to_ai": (
        "prior_human_to_author_count",
        "prior_human_to_author_time_since_hours",
        ESTIMAND_PRIOR_HUMAN_TO_AI,
    ),
    "ai_to_human": (
        "prior_author_to_human_count",
        "prior_author_to_human_time_since_hours",
        ESTIMAND_PRIOR_AI_TO_HUMAN,
    ),
}
RQ2_BASE_TERMS: Mapping[str, tuple[str, str]] = {
    "human_to_ai": (
        "z_log_prior_contact_count",
        "z_recent_prior_contact",
    ),
    "ai_to_human": (
        "z_log_reverse_contact_count",
        "z_recent_reverse_contact",
    ),
}
RQ2_AI_TERMS: Mapping[str, str] = {
    term: f"candidate_ai_x_{term}"
    for terms in RQ2_BASE_TERMS.values()
    for term in terms
}
RQ2_G2_AI_TERMS: Mapping[str, str] = {
    term: f"g2_x_candidate_ai_x_{term}"
    for terms in RQ2_BASE_TERMS.values()
    for term in terms
}
RQ2_REQUIRED_TERMS = frozenset(
    term
    for terms in RQ2_BASE_TERMS.values()
    for base in terms
    for term in (base, RQ2_AI_TERMS[base], RQ2_G2_AI_TERMS[base])
)


def build_history_support(network_history: pd.DataFrame) -> pd.DataFrame:
    """Summarize directional history for choice-by-AI-account opportunities."""

    required = {
        "group",
        "choice_set_id",
        "actor_id",
        "candidate_author_id",
        "candidate_author_type",
        "chosen",
        *(value for direction in RQ2_DIRECTIONS.values() for value in direction[:2]),
    }
    missing = sorted(required.difference(network_history.columns))
    if missing:
        raise ValueError(f"RQ2 history support is missing columns: {missing}")
    ai = network_history[
        network_history["candidate_author_type"].astype(str).str.upper().eq("AI")
    ].copy()
    if ai.empty:
        raise ValueError("RQ2 history support has no eligible AI-account opportunities")

    keys = ["group", "choice_set_id", "actor_id", "candidate_author_id"]
    history_columns = [
        value
        for direction in RQ2_DIRECTIONS.values()
        for value in direction[:2]
    ]
    grouped = ai.groupby(keys, sort=False, dropna=False)
    for column in history_columns:
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(
                f"RQ2 history {column!r} differs across content from one AI account "
                "within a choice set"
            )
    opportunities = grouped[history_columns].first().reset_index()
    selected = grouped["chosen"].max().astype(bool).rename("selected_ai_receiver")
    opportunities = opportunities.merge(
        selected.reset_index(), on=keys, how="left", validate="one_to_one"
    )
    competing_receivers = (
        network_history.groupby(
            ["group", "choice_set_id"], sort=False, dropna=False
        )["candidate_author_id"]
        .nunique(dropna=False)
        .gt(1)
        .rename("has_competing_receiver")
        .reset_index()
    )
    opportunities = opportunities.merge(
        competing_receivers,
        on=["group", "choice_set_id"],
        how="left",
        validate="many_to_one",
    )

    history_levels = ("none", "one_prior_reply", "two_or_more_prior_replies")
    rows: list[dict[str, Any]] = []
    for direction, (count_column, time_column, _) in RQ2_DIRECTIONS.items():
        directional = opportunities.copy()
        count = pd.to_numeric(directional[count_column], errors="raise")
        if (count < 0).any() or not np.isclose(count, np.round(count)).all():
            raise ValueError(f"RQ2 history count {count_column!r} must be nonnegative integers")
        directional["history_class"] = np.select(
            [count.eq(0), count.eq(1)],
            [history_levels[0], history_levels[1]],
            default=history_levels[2],
        )
        for scope, scoped in (
            ("eligible_ai_receiver", directional),
            (
                "estimand_competing_ai_receiver",
                directional[directional["has_competing_receiver"]],
            ),
            (
                "selected_ai_receiver",
                directional[directional["selected_ai_receiver"]],
            ),
        ):
            for group in sorted(opportunities["group"].astype(str).unique()):
                group_rows = scoped[scoped["group"].astype(str).eq(group)]
                denominator = len(group_rows)
                for history_class in history_levels:
                    part = group_rows[group_rows["history_class"].eq(history_class)]
                    counts = pd.to_numeric(part[count_column], errors="coerce")
                    positive_hours = pd.to_numeric(
                        part.loc[counts.gt(0), time_column], errors="coerce"
                    ).dropna()
                    rows.append(
                        {
                            "group": group,
                            "condition": {
                                "G1": "labels_visible",
                                "G2": "labels_hidden",
                            }.get(group, "unknown"),
                            "direction": direction,
                            "scope": scope,
                            "history_class": history_class,
                            "receiver_opportunities": int(len(part)),
                            "share_within_group_direction_scope": (
                                float(len(part) / denominator) if denominator else np.nan
                            ),
                            "choice_sets": int(part["choice_set_id"].nunique()),
                            "participants": int(part["actor_id"].nunique()),
                            "ai_accounts": int(part["candidate_author_id"].nunique()),
                            "human_ai_dyads": int(
                                part[["actor_id", "candidate_author_id"]]
                                .drop_duplicates()
                                .shape[0]
                            ),
                            "minimum_prior_replies": (
                                float(counts.min()) if len(counts) else np.nan
                            ),
                            "maximum_prior_replies": (
                                float(counts.max()) if len(counts) else np.nan
                            ),
                            "positive_recency_q25_hours": (
                                float(positive_hours.quantile(0.25))
                                if len(positive_hours)
                                else np.nan
                            ),
                            "positive_recency_median_hours": (
                                float(positive_hours.median())
                                if len(positive_hours)
                                else np.nan
                            ),
                            "positive_recency_q75_hours": (
                                float(positive_hours.quantile(0.75))
                                if len(positive_hours)
                                else np.nan
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    result["history_class"] = pd.Categorical(
        result["history_class"], categories=history_levels, ordered=True
    )
    result = result.sort_values(
        ["group", "direction", "scope", "history_class"], kind="stable"
    ).reset_index(drop=True)
    result["history_class"] = result["history_class"].astype(str)
    return result


def _standardized_counterfactual_pair(
    manifest: pd.DataFrame,
    term: str,
    *,
    raw_high: float,
) -> tuple[float, float]:
    rows = manifest[manifest["term"].astype(str).eq(term)]
    if len(rows) != 1:
        raise ValueError(f"RQ2 term manifest requires one row for {term!r}")
    row = rows.iloc[0]
    if str(row["status"]) != "included_standardized":
        raise ValueError(f"RQ2 term {term!r} must be included and standardized")
    mean = float(row["raw_mean"])
    standard_deviation = float(row["raw_sd"])
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError(f"RQ2 term {term!r} has an invalid training scaler")
    return (-mean / standard_deviation, (raw_high - mean) / standard_deviation)


def counterfactual_scenarios(
    manifest: pd.DataFrame,
    *,
    reference_hours: float = REFERENCE_CONTACT_HOURS,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Return arm-specific zero-to-one-contact values on training scales."""

    if not np.isfinite(reference_hours) or reference_hours <= 0:
        raise ValueError("RQ2 reference contact time must be positive and finite")
    recent_high = float(np.exp(-reference_hours / 24.0))
    raw_high = {
        "z_log_prior_contact_count": float(np.log1p(1.0)),
        "z_recent_prior_contact": recent_high,
        "z_log_reverse_contact_count": float(np.log1p(1.0)),
        "z_recent_reverse_contact": recent_high,
    }
    statuses = manifest.set_index("term")["status"].astype(str)
    result: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for direction, base_terms in RQ2_BASE_TERMS.items():
        arm_values: dict[str, dict[str, tuple[float, float]]] = {}
        pairs = {
            base: _standardized_counterfactual_pair(
                manifest, base, raw_high=raw_high[base]
            )
            for base in base_terms
        }
        for base in base_terms:
            for interaction in (RQ2_AI_TERMS[base], RQ2_G2_AI_TERMS[base]):
                if statuses.get(interaction) != "included_unscaled":
                    raise ValueError(f"RQ2 interaction {interaction!r} must be included unscaled")
        for group in ("G1", "G2"):
            values: dict[str, tuple[float, float]] = {}
            for base, pair in pairs.items():
                values[base] = pair
                values[RQ2_AI_TERMS[base]] = pair
                values[RQ2_G2_AI_TERMS[base]] = pair if group == "G2" else (0.0, 0.0)
            arm_values[group] = values
        result[direction] = arm_values
    return result


def _beta_draws(idata: Any) -> tuple[np.ndarray, tuple[str, ...]]:
    values = idata.posterior["beta"]
    terms = tuple(str(value) for value in values.coords["term"].values)
    draws = values.stack(sample=("chain", "draw")).transpose("sample", "term").values
    return draws, terms


def _projection_from_utility(
    beta: np.ndarray,
    posterior_terms: tuple[str, ...],
    utility: np.ndarray,
    data: Any,
    arm_x: np.ndarray,
    term_values: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, int]:
    selected_terms = tuple(term_values)
    missing = sorted(set(selected_terms).difference(posterior_terms))
    if missing:
        raise ValueError(f"RQ2 posterior is missing projection terms: {missing}")
    beta_indexes = [posterior_terms.index(term) for term in selected_terms]
    data_indexes = [data.terms.index(term) for term in selected_terms]
    low_values = np.asarray([term_values[term][0] for term in selected_terms])
    high_values = np.asarray([term_values[term][1] for term in selected_terms])
    total = np.zeros(beta.shape[0], dtype=float)
    opportunities = 0
    for choice_index in range(data.n_choices):
        positions = np.flatnonzero(data.mask[choice_index])
        receivers = data.receiver_index[choice_index, positions]
        ai_positions = positions[
            data.candidate_is_ai[choice_index, positions].astype(bool)
        ]
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
        raise ValueError("RQ2 projection has no AI account competing with another account")
    return total / opportunities, opportunities


def standardized_estimands(
    idata: Any,
    data: Any,
    term_manifest: pd.DataFrame,
    *,
    reference_hours: float = REFERENCE_CONTACT_HOURS,
    draw_chunk_size: int = 64,
) -> pd.DataFrame:
    """Estimate RQ2 account-level contrasts with bounded posterior memory."""

    if draw_chunk_size < 1:
        raise ValueError("RQ2 estimand draw chunk size must be positive")
    scenarios = counterfactual_scenarios(
        term_manifest, reference_hours=reference_hours
    )
    g1_x, g2_x = _common_arm_frames(data)
    accumulated: dict[tuple[str, str], list[np.ndarray]] = {
        (direction, group): []
        for direction in RQ2_DIRECTIONS
        for group in ("G1", "G2")
    }
    opportunity_counts: dict[tuple[str, str], int] = {}
    chains = int(idata.posterior.sizes["chain"])
    draws = int(idata.posterior.sizes["draw"])
    for chain in range(chains):
        for start in range(0, draws, draw_chunk_size):
            stop = min(draws, start + draw_chunk_size)
            chunk = idata.isel(
                chain=slice(chain, chain + 1), draw=slice(start, stop)
            )
            beta, posterior_terms = _beta_draws(chunk)
            for group, arm_x in (("G1", g1_x), ("G2", g2_x)):
                utility = _posterior_utilities(chunk, data, x_override=arm_x)
                for direction in RQ2_DIRECTIONS:
                    values, opportunities = _projection_from_utility(
                        beta,
                        posterior_terms,
                        utility,
                        data,
                        arm_x,
                        scenarios[direction][group],
                    )
                    accumulated[(direction, group)].append(values)
                    key = (direction, group)
                    previous = opportunity_counts.setdefault(key, opportunities)
                    if previous != opportunities:
                        raise RuntimeError("RQ2 opportunity count changed across draw chunks")
                del utility

    rows: list[dict[str, Any]] = []
    for direction, (_, _, estimand) in RQ2_DIRECTIONS.items():
        g1 = np.concatenate(accumulated[(direction, "G1")])
        g2 = np.concatenate(accumulated[(direction, "G2")])
        if g1.shape != g2.shape:
            raise RuntimeError("RQ2 arm projections have different posterior lengths")
        opportunities = opportunity_counts[(direction, "G1")]
        if opportunities != opportunity_counts[(direction, "G2")]:
            raise RuntimeError("RQ2 arm projections have different opportunity counts")
        for group, values in (("G1", g1), ("G2", g2), ("G2-G1", g2 - g1)):
            row = _posterior_summary(values, estimand, group)
            row.update(
                {
                    "direction": direction,
                    "condition": {
                        "G1": "labels_visible",
                        "G2": "labels_hidden",
                        "G2-G1": "labels_hidden_minus_visible",
                    }[group],
                    "estimate": float(np.median(values)),
                    "effect_scale": "absolute_selection_probability_difference",
                    "reference_contact_hours": float(reference_hours),
                    "ai_receiver_opportunities": int(opportunities),
                    "standardization_scope": (
                        "pooled_choice_by_AI_receiver_opportunities"
                    ),
                    "counterfactual": (
                        "one eligible AI account at a time: no prior reply versus "
                        f"one prior reply {reference_hours:g} valid-clock hours earlier; "
                        "all eligible "
                        "content from that account moves jointly; opposite-direction "
                        "history and other covariates remain observed"
                    ),
                    "estimand_interpretation": (
                        "model_projected_association_not_causal_intervention"
                    ),
                    "inference_boundary": (
                        "one_community_per_condition_ecological_between_arm_contrast"
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_estimand_denominators(
    estimands: pd.DataFrame,
    data: Any,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Attach the canonical RQ2 analysis window and opportunity denominator."""

    result = estimands.copy()
    result["analysis_window"] = (
        f"{config.start_time} to {config.end_time or 'snapshot extraction'}"
    )
    result["denominator"] = result["ai_receiver_opportunities"].map(
        lambda count: (
            f"{int(count)} competing choice-by-AI-account opportunities across "
            f"{data.n_choices} observed human replies"
        )
    )
    result["opportunity_boundary"] = (
        "earlier_valid_nonself_same_thread_candidates_not_verified_feed_exposure"
    )
    return result


def build_core_readiness(
    *,
    integrity: pd.DataFrame,
    model_data_reproduced: bool,
    analysis_signature: pd.DataFrame,
    signatures: pd.DataFrame,
    bundle_authentication: pd.DataFrame,
    term_manifest: pd.DataFrame,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    estimands: pd.DataFrame,
    influence_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply fail-closed checks for the full-length RQ2 core fit."""

    rows: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, observed: Any) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "required_for_rq2_core_fit": True,
                "observed": str(observed),
            }
        )

    integrity_pass = bool(
        not integrity.empty
        and integrity["passed"].fillna(False).astype(bool).all()
    )
    add(
        "rq2_canonical_data_and_strict_history",
        integrity_pass and model_data_reproduced,
        (
            f"integrity={int(integrity['passed'].fillna(False).sum())}/{len(integrity)}; "
            f"model_data_reproduced={model_data_reproduced}"
        ),
    )

    included = set(
        term_manifest.loc[
            term_manifest["status"].astype(str).str.startswith("included"), "term"
        ].astype(str)
    )
    forbidden = {"z_log_receiver_popularity", "z_log_two_path_closure"}
    specification_pass = RQ2_REQUIRED_TERMS.issubset(included) and not (
        forbidden & included
    )
    add(
        "rq2_ai_specific_condition_interacted_specification",
        specification_pass,
        (
            f"required_terms={len(RQ2_REQUIRED_TERMS & included)}/{len(RQ2_REQUIRED_TERMS)}; "
            f"forbidden_terms={sorted(forbidden & included)}"
        ),
    )

    eligible = support[
        support["scope"].astype(str).eq("estimand_competing_ai_receiver")
    ]
    expected_support = {
        (group, direction, history_class)
        for group in ("G1", "G2")
        for direction in RQ2_DIRECTIONS
        for history_class in ("none", "one_prior_reply", "two_or_more_prior_replies")
    }
    observed_support = set(
        zip(
            eligible["group"].astype(str),
            eligible["direction"].astype(str),
            eligible["history_class"].astype(str),
        )
    )
    positive_support = bool(
        expected_support == observed_support
        and eligible["receiver_opportunities"].gt(0).all()
    )
    add(
        "rq2_directional_first_and_repeat_contact_support",
        positive_support,
        (
            f"positive_cells={int(eligible['receiver_opportunities'].gt(0).sum())}/"
            f"{len(expected_support)} competing-opportunity cells"
        ),
    )

    bundle_pass = bool(
        len(bundle_authentication) == 1
        and bundle_authentication["authenticated"].fillna(False).astype(bool).all()
        and bundle_authentication["required_tables_present"]
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "rq2_fit_result_bundle_authentication",
        bundle_pass,
        (
            str(bundle_authentication["authentication_status"].iloc[0])
            if len(bundle_authentication) == 1
            else "missing authentication row"
        ),
    )

    data_hash = (
        str(analysis_signature["model_data_sha256"].iloc[0])
        if len(analysis_signature) == 1
        else None
    )
    implementation_hash = (
        str(analysis_signature["implementation_sha256"].iloc[0])
        if len(analysis_signature) == 1
        else None
    )
    main = signatures[
        signatures.get("model", pd.Series(dtype=object)).astype(str).eq("RQ2")
        & signatures.get("role", pd.Series(dtype=object)).astype(str).eq("rq2_main")
    ]
    signature_pass = bool(
        len(main) == 1
        and main["live_posterior_authenticated"].fillna(False).astype(bool).all()
        and main["publication_length_request"].fillna(False).astype(bool).all()
        and main["dimension_match"].fillna(False).astype(bool).all()
        and main["data_sha256"].astype(str).eq(data_hash).all()
        and main["implementation_sha256"].astype(str).eq(implementation_hash).all()
    )
    add(
        "rq2_signed_publication_length_core_posterior",
        signature_pass,
        f"authenticated full-length main rows={int(signature_pass)}",
    )

    mcmc_pass = bool(
        len(diagnostics) == 1
        and diagnostics["passed"].fillna(False).astype(bool).all()
    )
    add(
        "rq2_core_mcmc",
        mcmc_pass,
        (
            diagnostics[["max_rhat", "min_ess_bulk", "min_ess_tail", "divergences"]]
            .to_dict("records")[0]
            if len(diagnostics) == 1
            else "missing diagnostic row"
        ),
    )

    expected_estimands = {
        (estimand, group)
        for estimand in (ESTIMAND_PRIOR_HUMAN_TO_AI, ESTIMAND_PRIOR_AI_TO_HUMAN)
        for group in ("G1", "G2", "G2-G1")
    }
    observed_estimands = set(
        zip(estimands.get("estimand", pd.Series(dtype=object)), estimands.get("group", pd.Series(dtype=object)))
    )
    numeric_columns = ["estimate", "ci_low", "ci_high", "posterior_probability_gt_zero"]
    estimand_pass = bool(
        observed_estimands == expected_estimands
        and len(estimands) == len(expected_estimands)
        and np.isfinite(estimands[numeric_columns].to_numpy(dtype=float)).all()
        and estimands["reference_contact_hours"].eq(REFERENCE_CONTACT_HOURS).all()
    )
    add(
        "rq2_joint_probability_estimands",
        estimand_pass,
        f"complete finite rows={len(observed_estimands & expected_estimands)}/{len(expected_estimands)}",
    )

    influence_complete = True
    influence_observed: list[str] = []
    for unit in ("participant", "account", "thread"):
        frame = influence_tables.get(unit, pd.DataFrame())
        complete = bool(
            not frame.empty
            and {"pareto_k", "needs_refit", "choice_sets"}.issubset(frame.columns)
            and np.isfinite(pd.to_numeric(frame["pareto_k"], errors="coerce")).all()
        )
        influence_complete &= complete
        flagged = int(frame.get("needs_refit", pd.Series(dtype=bool)).fillna(True).sum())
        influence_observed.append(f"{unit}={flagged}/{len(frame)} flagged")
    add(
        "rq2_grouped_influence_diagnostics_computed",
        influence_complete,
        "; ".join(influence_observed),
    )
    return pd.DataFrame(rows)


def write_report(
    config: AnalysisConfig,
    *,
    estimands: pd.DataFrame,
    support: pd.DataFrame,
    diagnostics: pd.DataFrame,
    influence_tables: Mapping[str, pd.DataFrame],
    readiness: pd.DataFrame,
    signatures: pd.DataFrame,
) -> Path:
    """Write a compact report for the validated RQ2 core fit."""

    path = config.reports_dir / "rq2_result_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    core_passed = bool(
        not readiness.empty and readiness["passed"].fillna(False).astype(bool).all()
    )
    columns = [
        "direction",
        "group",
        "estimate",
        "ci_low",
        "ci_high",
        "posterior_probability_gt_zero",
        "ai_receiver_opportunities",
    ]
    estimand_table = (
        estimands[columns].to_markdown(index=False)
        if not estimands.empty
        else "No authenticated estimands."
    )
    support_focus = support[
        support["scope"].astype(str).eq("estimand_competing_ai_receiver")
    ]
    support_table = (
        support_focus[
            ["group", "direction", "history_class", "receiver_opportunities"]
        ].to_markdown(index=False)
        if not support_focus.empty
        else "No support table."
    )
    diagnostics_table = (
        diagnostics[
            ["max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "passed"]
        ].to_markdown(index=False)
        if not diagnostics.empty
        else "No diagnostics."
    )
    influence_rows = []
    for unit in ("participant", "account", "thread"):
        frame = influence_tables.get(unit, pd.DataFrame())
        influence_rows.append(
            {
                "unit": unit,
                "groups": int(len(frame)),
                "needs_exact_refit": int(
                    frame.get("needs_refit", pd.Series(dtype=bool)).fillna(True).sum()
                ),
                "maximum_pareto_k": (
                    float(pd.to_numeric(frame["pareto_k"], errors="coerce").max())
                    if not frame.empty
                    else np.nan
                ),
            }
        )
    influence_table = pd.DataFrame(influence_rows).to_markdown(index=False)
    gates_table = readiness[["gate", "passed", "observed"]].to_markdown(index=False)
    main = signatures[
        signatures.get("role", pd.Series(dtype=object)).astype(str).eq("rq2_main")
    ]
    posterior_hash = (
        str(main["posterior_sha256"].iloc[0]) if len(main) == 1 else "unavailable"
    )
    text = f"""# RQ2 directional prior-contact core-fit report

- Run ID: `{config.run_id}`
- Core-fit validation: `{'passed' if core_passed else 'failed'}`
- Publication admission: `pending sensitivity, prediction, and required exact influence refits`
- Main posterior SHA-256: `{posterior_hash}`
- Analysis window: `{config.start_time}` to `{config.end_time or 'snapshot extraction'}`
- Inference boundary: associations are conditional on observed human replies and reconstructed same-thread opportunities; the between-condition contrast is ecological

## Model specification

The model retains the M1 opportunity, content, AI-authorship, receiver, participant, and topic terms. It adds four general directional-history terms, four AI-specific deviations, and four G2-specific AI-history deviations. Popularity and closure terms are excluded.

## Primary probability estimands

Each estimate changes one eligible AI account with at least one competing receiver from no prior directional reply to one prior reply 24 valid-clock hours earlier. All eligible content from that account changes jointly. The opposite contact direction and all other covariates remain observed.

{estimand_table}

## Empirical support

{support_table}

## MCMC diagnostics

{diagnostics_table}

## Grouped influence diagnostics

{influence_table}

PSIS flags identify deletion refits that must be run before publication admission. They do not establish that a unit changes the substantive conclusion.

## Core-fit validation gates

{gates_table}
"""
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "ESTIMAND_PRIOR_AI_TO_HUMAN",
    "ESTIMAND_PRIOR_HUMAN_TO_AI",
    "REFERENCE_CONTACT_HOURS",
    "RQ2_REQUIRED_TERMS",
    "add_estimand_denominators",
    "build_core_readiness",
    "build_history_support",
    "counterfactual_scenarios",
    "standardized_estimands",
    "write_report",
]
