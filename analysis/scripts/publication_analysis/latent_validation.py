"""Fail-closed validation for optional latent receiver-role extensions.

The observed human/AI relational-event model remains primary.  This module
evaluates latent K=2/3 receiver partitions only when explicitly requested and
keeps every decision component in a separate, inspectable table.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .terms import apply_model_term_manifest, build_model_frame, included_terms_for_model
from .validation import rolling_origin_assignments, summarize_mcmc


LATENT_NETWORK_TERMS = (
    "z_log_prior_contact_count",
    "z_log_two_path_closure",
)
MINIMUM_BLOCK_SIZE = 5
MINIMUM_STABLE_MEMBERSHIP_SHARE = 0.90
MINIMUM_PAIRWISE_ARI = 0.80
MINIMUM_RECEIVER_MEMBERSHIP_PROBABILITY = 0.80
MINIMUM_CONFIDENT_MEMBERSHIP_SHARE = 0.90
MAXIMUM_MEAN_NORMALIZED_MEMBERSHIP_ENTROPY = 0.25
MINIMUM_RECOVERY_RATE = 0.80
MINIMUM_MEDIAN_ARI = 0.80
DEGREE_BALANCED_ACCURACY_THRESHOLD = 0.75
DEGREE_ACCURACY_GAIN_THRESHOLD = 0.20
TYPE_NMI_THRESHOLD = 0.50
MINIMUM_NETWORK_SLOPE_DIFFERENCE = 0.10


@dataclass(frozen=True)
class LatentRecoveryDesign:
    """Bounded recovery design, separate from the publication-size data."""

    replicates: int = 3
    n_actors: int = 12
    n_receivers: int = 24
    n_choice_sets: int = 240
    candidates_per_set: int = 5
    draws: int = 500
    tune: int = 500
    chains: int = 2

    def validate(self) -> None:
        if self.replicates < 3:
            raise ValueError("Latent recovery requires at least three independent seeds")
        if min(self.n_actors, self.n_receivers, self.n_choice_sets) < 1:
            raise ValueError("Latent recovery dimensions must be positive")
        if self.candidates_per_set < 2 or self.candidates_per_set > self.n_receivers:
            raise ValueError("Recovery candidates_per_set is outside the receiver range")
        if min(self.draws, self.tune, self.chains) < 1:
            raise ValueError("Latent recovery sampling settings must be positive")

    @property
    def scale_label(self) -> str:
        return (
            f"bounded_fixture_{self.n_actors}_actors_{self.n_receivers}_receivers_"
            f"{self.n_choice_sets}_choice_sets_{self.candidates_per_set}_candidates"
        )


def not_run_status(block_sizes: Sequence[int]) -> pd.DataFrame:
    """Return a complete fail-closed gate table for a skipped extension."""

    return pd.DataFrame(
        [
            {
                "k": int(k),
                "predictive_pass": False,
                "forward_convergence_pass": False,
                "main_mcmc_pass": False,
                "stability_pass": False,
                "block_size_pass": False,
                "recovery_pass": False,
                "ysocial_sized_recovery_pass": False,
                "degree_confounding_pass": False,
                "interpretability_pass": False,
                "opportunity_activity_stress_test_pass": False,
                "manuscript_prominence_pass": False,
                "gate_status": "not_run_optional_extension",
            }
            for k in block_sizes
        ]
    )


def recovery_scale_reference(raw_model_frame: pd.DataFrame) -> Mapping[str, Any]:
    """Derive the minimum recovery scale from the smaller experimental arm."""

    from .study import GROUPS

    required = {"choice_set_id", "group", "actor_id", "candidate_author_id"}
    missing = sorted(required.difference(raw_model_frame.columns))
    if missing:
        raise ValueError(f"Recovery scale reference is missing columns: {missing}")
    event_rows = raw_model_frame[list(required)].copy()
    choice_counts = event_rows.groupby("group")["choice_set_id"].nunique()
    if choice_counts.empty:
        raise ValueError("Latent recovery reference requires realized choice sets")
    reference_group = str(choice_counts.idxmin())
    candidate_counts = (
        event_rows[event_rows["group"].astype(str).eq(reference_group)]
        .groupby("choice_set_id")
        .size()
    )
    return {
        "reference_group": reference_group,
        "minimum_reference_actors": min(group.human_count for group in GROUPS.values()),
        "minimum_reference_receivers": min(
            group.human_count + group.ai_count for group in GROUPS.values()
        ),
        "minimum_reference_choice_sets": int(choice_counts.loc[reference_group]),
        "minimum_reference_candidates_per_set": int(np.ceil(candidate_counts.median())),
        "candidate_reference_statistic": (
            "ceiling_of_median_candidates_per_choice_set_in_smaller_arm"
        ),
    }


def recovery_design_is_ysocial_sized(
    design: LatentRecoveryDesign,
    reference: Mapping[str, Any],
) -> bool:
    """Return whether a requested recovery fixture reaches the declared scale."""

    return bool(
        design.n_actors >= int(reference["minimum_reference_actors"])
        and design.n_receivers >= int(reference["minimum_reference_receivers"])
        and design.n_choice_sets >= int(reference["minimum_reference_choice_sets"])
        and design.candidates_per_set
        >= int(reference["minimum_reference_candidates_per_set"])
    )


def _modal_partition(memberships: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(
        [np.bincount(memberships[:, index], minlength=k).argmax() for index in range(memberships.shape[1])],
        dtype=np.int64,
    )


def _best_label_alignment(partition: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    best = partition.copy()
    best_agreement = -1
    for permutation in permutations(range(k)):
        mapped = np.asarray(permutation, dtype=np.int64)[partition]
        agreement = int(np.sum(mapped == reference))
        if agreement > best_agreement:
            best = mapped
            best_agreement = agreement
    return best


def _posterior_membership_uncertainty(
    memberships: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Summarize draw-level assignments after resolving global label switching."""

    aligned = np.stack(
        [
            _best_label_alignment(draw, reference, k)
            for chain in memberships
            for draw in chain
        ]
    )
    probabilities = np.stack(
        [np.mean(aligned == block, axis=0) for block in range(k)],
        axis=1,
    )
    modal_partition = probabilities.argmax(axis=1).astype(np.int64)
    assignment_probability = probabilities.max(axis=1)
    safe_probabilities = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    normalized_entropy = -np.sum(
        probabilities * np.log(safe_probabilities), axis=1
    ) / np.log(k)
    confident = assignment_probability >= MINIMUM_RECEIVER_MEMBERSHIP_PROBABILITY
    return modal_partition, {
        "mean_posterior_membership_probability": float(assignment_probability.mean()),
        "median_posterior_membership_probability": float(np.median(assignment_probability)),
        "minimum_posterior_membership_probability": float(assignment_probability.min()),
        "confident_membership_share": float(confident.mean()),
        "mean_normalized_membership_entropy": float(normalized_entropy.mean()),
        "maximum_normalized_membership_entropy": float(normalized_entropy.max()),
    }


def membership_stability(idata: Any, k: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Measure chain agreement and draw-level posterior assignment certainty."""

    from sklearn.metrics import adjusted_rand_score

    if "receiver_block" not in idata.posterior:
        raise ValueError("Latent posterior is missing receiver_block")
    values = (
        idata.posterior["receiver_block"]
        .transpose("chain", "draw", "receiver")
        .values.astype(np.int64)
    )
    if values.shape[0] < 2:
        raise ValueError("Membership stability requires at least two chains")
    reference = values[0, 0].copy()
    aligned_draws = np.stack(
        [
            np.stack(
                [_best_label_alignment(draw, reference, k) for draw in chain]
            )
            for chain in values
        ]
    )
    chain_partitions = [_modal_partition(chain, k) for chain in aligned_draws]
    consensus = _modal_partition(
        aligned_draws.reshape(-1, aligned_draws.shape[-1]), k
    )
    aligned = np.stack(
        [_best_label_alignment(partition, consensus, k) for partition in chain_partitions]
    )
    pairwise_ari = [
        float(adjusted_rand_score(left, right))
        for left, right in combinations(chain_partitions, 2)
    ]
    per_receiver_share = np.mean(aligned == consensus[None, :], axis=0)
    chain_minimum = min(
        int(np.bincount(partition, minlength=k).min()) for partition in chain_partitions
    )
    posterior_partition, posterior_uncertainty = _posterior_membership_uncertainty(
        values, consensus, k
    )
    consensus_counts = np.bincount(posterior_partition, minlength=k)
    posterior_assignment_pass = bool(
        posterior_uncertainty["confident_membership_share"]
        >= MINIMUM_CONFIDENT_MEMBERSHIP_SHARE
        and posterior_uncertainty["mean_normalized_membership_entropy"]
        <= MAXIMUM_MEAN_NORMALIZED_MEMBERSHIP_ENTROPY
    )
    summary = pd.DataFrame(
        [
            {
                "k": k,
                "chains": int(values.shape[0]),
                "receivers": int(values.shape[-1]),
                "mean_pairwise_ari": float(np.mean(pairwise_ari)),
                "minimum_pairwise_ari": float(np.min(pairwise_ari)),
                "stable_membership_share": float(np.mean(per_receiver_share >= 0.90)),
                "mean_membership_stability": float(np.mean(per_receiver_share)),
                **posterior_uncertainty,
                "minimum_receiver_membership_probability": (
                    MINIMUM_RECEIVER_MEMBERSHIP_PROBABILITY
                ),
                "minimum_confident_membership_share": (
                    MINIMUM_CONFIDENT_MEMBERSHIP_SHARE
                ),
                "maximum_mean_normalized_membership_entropy": (
                    MAXIMUM_MEAN_NORMALIZED_MEMBERSHIP_ENTROPY
                ),
                "posterior_assignment_pass": posterior_assignment_pass,
                "smallest_block_n": int(consensus_counts.min()),
                "smallest_chain_block_n": chain_minimum,
                "collapsed_or_tiny_block": bool(
                    min(int(consensus_counts.min()), chain_minimum) < MINIMUM_BLOCK_SIZE
                ),
                "stability_pass": bool(
                    values.shape[0] >= 4
                    and np.min(pairwise_ari) >= MINIMUM_PAIRWISE_ARI
                    and np.mean(per_receiver_share >= 0.90)
                    >= MINIMUM_STABLE_MEMBERSHIP_SHARE
                    and posterior_assignment_pass
                ),
            }
        ]
    )
    return summary, posterior_partition


def opportunity_activity_stress_test_not_run(
    block_sizes: Sequence[int],
) -> pd.DataFrame:
    """Declare the unimplemented opportunity/activity perturbation gate.

    Recovery from a single bounded fixture and prediction of observed modal
    blocks from degree do not establish robustness to opportunity imbalance.
    A future implementation must perturb receiver opportunity and activity
    independently of role, refit each candidate K, and evaluate recovery and
    spurious-block rates before this gate can pass.
    """

    return pd.DataFrame(
        [
            {
                "k": int(k),
                "opportunity_activity_stress_test_pass": False,
                "stress_test_evidence_available": False,
                "stress_test_status": (
                    "not_run_joint_opportunity_activity_perturbation"
                ),
                "required_stress_test": (
                    "independently perturb receiver opportunity and activity, refit, "
                    "and verify role recovery plus false-positive control"
                ),
            }
            for k in block_sizes
        ]
    )


def receiver_activity_table(data: Any, partition: np.ndarray, k: int) -> pd.DataFrame:
    """Summarize opportunity and selected-target activity for each receiver."""

    n_receivers = len(data.receiver_labels)
    opportunity = np.bincount(data.receiver_index[data.mask], minlength=n_receivers)
    choice_index = np.arange(data.n_choices)
    selected_receiver = data.receiver_index[choice_index, data.chosen_position]
    selected = np.bincount(selected_receiver, minlength=n_receivers)
    distinct_senders = np.zeros(n_receivers, dtype=np.int64)
    for receiver in range(n_receivers):
        distinct_senders[receiver] = np.unique(
            data.actor_index[selected_receiver == receiver]
        ).size
    if len(partition) != n_receivers or set(np.unique(partition)).difference(range(k)):
        raise ValueError("Receiver partition is incompatible with prepared model data")
    return pd.DataFrame(
        {
            "receiver_label": data.receiver_labels,
            "group": data.receiver_groups,
            "author_type": np.where(data.receiver_is_ai.astype(bool), "AI", "H"),
            "modal_block": partition,
            "candidate_opportunities": opportunity,
            "selected_count": selected,
            "distinct_human_senders": distinct_senders,
            "selection_rate": np.divide(
                selected,
                opportunity,
                out=np.zeros(n_receivers, dtype=float),
                where=opportunity > 0,
            ),
        }
    )


def degree_confounding_diagnostic(activity: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Test whether residual degree/activity alone reconstructs latent roles."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    required = {
        "group",
        "author_type",
        "modal_block",
        "candidate_opportunities",
        "selected_count",
        "distinct_human_senders",
        "selection_rate",
    }
    missing = sorted(required.difference(activity.columns))
    if missing:
        raise ValueError(f"Degree-confounding input is missing columns: {missing}")
    y = activity["modal_block"].to_numpy(dtype=int)
    counts = np.bincount(y, minlength=k)
    if counts.min() < MINIMUM_BLOCK_SIZE:
        return pd.DataFrame(
            [
                {
                    "k": k,
                    "cv_balanced_accuracy": np.nan,
                    "chance_balanced_accuracy": 1.0 / k,
                    "accuracy_gain_over_chance": np.nan,
                    "degree_reducible": True,
                    "diagnostic_status": "unavailable_tiny_or_collapsed_block",
                }
            ]
        )
    features = pd.DataFrame(
        {
            "log_candidate_opportunities": np.log1p(activity["candidate_opportunities"]),
            "log_selected_count": np.log1p(activity["selected_count"]),
            "log_distinct_senders": np.log1p(activity["distinct_human_senders"]),
            "selection_rate": activity["selection_rate"].astype(float),
        }
    )
    strata = activity["group"].astype(str) + "::" + activity["author_type"].astype(str)
    residual = features - features.groupby(strata).transform("mean")
    residual = residual.fillna(0.0)
    scaled = StandardScaler().fit_transform(residual)
    folds = min(5, int(counts.min()))
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        class_weight="balanced",
        random_state=seed,
    )
    predicted = cross_val_predict(
        classifier,
        scaled,
        y,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        method="predict",
    )
    accuracy = float(balanced_accuracy_score(y, predicted))
    chance = 1.0 / k
    gain = accuracy - chance
    reducible = bool(
        accuracy >= DEGREE_BALANCED_ACCURACY_THRESHOLD
        and gain >= DEGREE_ACCURACY_GAIN_THRESHOLD
    )
    return pd.DataFrame(
        [
            {
                "k": k,
                "receivers": int(len(activity)),
                "cv_folds": folds,
                "cv_balanced_accuracy": accuracy,
                "chance_balanced_accuracy": chance,
                "accuracy_gain_over_chance": gain,
                "accuracy_threshold": DEGREE_BALANCED_ACCURACY_THRESHOLD,
                "gain_threshold": DEGREE_ACCURACY_GAIN_THRESHOLD,
                "residualization": "within_group_and_observed_author_type",
                "degree_reducible": reducible,
                "diagnostic_status": "completed",
            }
        ]
    )


def block_interpretability(
    idata: Any,
    data: Any,
    activity: pd.DataFrame,
    k: int,
    network_terms: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate residual type separation and covariate-adjusted block slopes."""

    from sklearn.metrics import normalized_mutual_info_score

    required_variables = {
        "receiver_effect",
        "actor_ai_effect",
        "topic_ai_effect",
        "block_network_slope",
    }
    covariate_adjusted = required_variables.issubset(idata.posterior.data_vars)
    required_families = {
        "opportunity_content",
        "focal",
        "repetition",
        "reverse",
        "popularity",
        "closure",
    }
    covariate_adjusted = bool(
        covariate_adjusted
        and required_families.issubset(set(data.term_families))
        and set(network_terms).issubset(data.terms)
    )
    type_nmi = float(
        normalized_mutual_info_score(
            activity["author_type"].astype(str), activity["modal_block"].astype(int)
        )
    )
    type_residual_pass = type_nmi < TYPE_NMI_THRESHOLD
    slope = (
        idata.posterior["block_network_slope"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "block", "network_term")
    )
    available_terms = [str(value) for value in slope.coords["network_term"].values]
    contrast_rows: list[dict[str, Any]] = []
    for term_index, term in enumerate(available_terms):
        for left, right in combinations(range(k), 2):
            difference = slope.values[:, right, term_index] - slope.values[:, left, term_index]
            low, high = np.quantile(difference, [0.025, 0.975])
            separated = bool(
                low > MINIMUM_NETWORK_SLOPE_DIFFERENCE
                or high < -MINIMUM_NETWORK_SLOPE_DIFFERENCE
            )
            contrast_rows.append(
                {
                    "k": k,
                    "network_term": term,
                    "block_contrast": f"{right}-{left}",
                    "posterior_median_difference": float(np.median(difference)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "minimum_meaningful_difference": MINIMUM_NETWORK_SLOPE_DIFFERENCE,
                    "meaningfully_separated": separated,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    slope_pass = bool(
        not contrasts.empty and contrasts["meaningfully_separated"].astype(bool).any()
    )
    summary = pd.DataFrame(
        [
            {
                "k": k,
                "covariate_adjusted_model_pass": covariate_adjusted,
                "observed_type_block_nmi": type_nmi,
                "maximum_type_nmi": TYPE_NMI_THRESHOLD,
                "author_type_residual_role_pass": type_residual_pass,
                "network_slope_separation_pass": slope_pass,
                "interpretability_pass": bool(
                    covariate_adjusted and type_residual_pass and slope_pass
                ),
                "interpretation_scope": (
                    "receiver roles conditional on observed author type, receiver frailty, "
                    "participant, topic, opportunity, activity, repetition, popularity, and closure"
                ),
            }
        ]
    )
    return summary, contrasts


def paired_forward_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare latent K models with observed M3 on identical held-out events."""

    required = {
        "fold",
        "k",
        "model",
        "choice_set_id",
        "group",
        "actor_label",
        "log_score",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Latent forward predictions are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for k in sorted(predictions["k"].unique()):
        subset = predictions[predictions["k"].eq(k)]
        for fold in (*sorted(subset["fold"].unique()), "ALL"):
            part = subset if fold == "ALL" else subset[subset["fold"].eq(fold)]
            observed = part[part["model"].eq("M3_observed")]
            latent = part[part["model"].eq(f"latent_K{int(k)}")]
            paired = latent.merge(
                observed,
                on=["fold", "choice_set_id", "group", "actor_label"],
                suffixes=("_latent", "_observed"),
                validate="one_to_one",
            )
            expected = max(len(observed), len(latent))
            identical = len(paired) == expected and len(observed) == len(latent)
            delta = paired["log_score_latent"] - paired["log_score_observed"]
            difference = float(delta.sum()) if identical else np.nan
            from .pipeline import _participant_clustered_total_se

            standard_error, participant_clusters = (
                _participant_clustered_total_se(delta, paired["actor_label"])
                if identical
                else (np.nan, 0)
            )
            rows.append(
                {
                    "k": int(k),
                    "fold": fold,
                    "reference_model": f"latent_K{int(k)}",
                    "comparator_model": "M3_observed",
                    "test_choice_sets": int(len(paired)),
                    "identical_future_choice_sets": identical,
                    "unseen_receiver_candidate_rows": int(
                        paired.get(
                            "unseen_receiver_candidate_rows_latent",
                            pd.Series(0, index=paired.index),
                        ).sum()
                    ),
                    "elpd_difference": difference,
                    "elpd_difference_se": standard_error,
                    "uncertainty_clusters": participant_clusters,
                    "uncertainty_unit": "participant_within_arm",
                    "gain_at_least_two_se": bool(
                        identical
                        and np.isfinite(standard_error)
                        and difference >= 2.0 * standard_error
                    ),
                    "comparison_scope": (
                        "one_step_ahead_conditional_on_realized_earlier_heldout_history"
                    ),
                }
            )
    return pd.DataFrame(rows)


def recovery_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed conditional-choice recovery by candidate K."""

    required = {
        "k",
        "replicate",
        "adjusted_rand_index",
        "passed",
        "mcmc_passed",
        "ysocial_sized_simulation",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Recovery results are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for k, part in results.groupby("k", sort=True):
        successful = part["passed"].astype(bool) & part["mcmc_passed"].astype(bool)
        ysocial_sized = bool(part["ysocial_sized_simulation"].astype(bool).all())
        rows.append(
            {
                "k": int(k),
                "recovery_replicates": int(len(part)),
                "recovery_rate": float(successful.mean()),
                "median_ari": float(part["adjusted_rand_index"].median()),
                "all_recovery_mcmc_pass": bool(part["mcmc_passed"].astype(bool).all()),
                "recovery_pass": bool(
                    len(part) >= 3
                    and successful.mean() >= MINIMUM_RECOVERY_RATE
                    and part["adjusted_rand_index"].median() >= MINIMUM_MEDIAN_ARI
                ),
                "simulation_scale": str(part["simulation_scale"].iloc[0]),
                "ysocial_sized_simulation": ysocial_sized,
                "ysocial_sized_recovery_pass": ysocial_sized,
            }
        )
    return pd.DataFrame(rows)


def assemble_gate_status(
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    degree: pd.DataFrame,
    interpretability: pd.DataFrame,
    recovery: pd.DataFrame,
    diagnostics: pd.DataFrame,
    opportunity_activity_stress_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine every prespecified component; missing evidence fails closed."""

    k_values = sorted(
        set(stability.get("k", pd.Series(dtype=int)).dropna().astype(int))
        | set(comparisons.get("k", pd.Series(dtype=int)).dropna().astype(int))
    )

    def rows_for_k(frame: pd.DataFrame, k: int) -> pd.DataFrame:
        if "k" not in frame:
            return pd.DataFrame()
        return frame[frame["k"].eq(k)]

    rows: list[dict[str, Any]] = []
    for k in k_values:
        comparison_rows = rows_for_k(comparisons, k)
        pooled = (
            comparison_rows[comparison_rows["fold"].astype(str).eq("ALL")]
            if "fold" in comparison_rows
            else pd.DataFrame()
        )
        stable = rows_for_k(stability, k)
        degree_row = rows_for_k(degree, k)
        meaning = rows_for_k(interpretability, k)
        recovered = rows_for_k(recovery, k)
        diagnostic = rows_for_k(diagnostics, k)
        stress_test = rows_for_k(
            opportunity_activity_stress_test
            if opportunity_activity_stress_test is not None
            else pd.DataFrame(),
            k,
        )
        predictive_pass = bool(
            len(pooled) == 1
            and pooled["gain_at_least_two_se"].fillna(False).astype(bool).all()
            and pooled["identical_future_choice_sets"].fillna(False).astype(bool).all()
            and pooled["unseen_receiver_candidate_rows"].fillna(1).eq(0).all()
        )
        forward_rows = (
            diagnostic[diagnostic["role"].eq("forward")]
            if "role" in diagnostic
            else pd.DataFrame()
        )
        observed_forward_rows = (
            diagnostic[diagnostic["role"].eq("observed_forward")]
            if "role" in diagnostic
            else pd.DataFrame()
        )
        main_rows = (
            diagnostic[diagnostic["role"].eq("main")]
            if "role" in diagnostic
            else pd.DataFrame()
        )
        forward_pass = bool(
            len(forward_rows) == 3
            and forward_rows["passed"].fillna(False).astype(bool).all()
            and forward_rows["requested_chains"].ge(4).all()
            and forward_rows["requested_draws"].ge(1_000).all()
            and forward_rows["requested_tune"].ge(1_000).all()
            and len(observed_forward_rows) == 3
            and observed_forward_rows["passed"].fillna(False).astype(bool).all()
            and observed_forward_rows["requested_chains"].ge(4).all()
            and observed_forward_rows["requested_draws"].ge(1_000).all()
            and observed_forward_rows["requested_tune"].ge(1_000).all()
        )
        main_pass = bool(
            len(main_rows) == 1
            and main_rows["passed"].fillna(False).astype(bool).all()
            and main_rows["requested_chains"].ge(4).all()
            and main_rows["requested_draws"].ge(1_000).all()
            and main_rows["requested_tune"].ge(1_000).all()
        )
        stability_pass = bool(
            len(stable) == 1 and stable["stability_pass"].fillna(False).astype(bool).all()
        )
        block_size_pass = bool(
            len(stable) == 1
            and ~stable["collapsed_or_tiny_block"].fillna(True).astype(bool).any()
        )
        recovery_pass = bool(
            len(recovered) == 1
            and recovered["recovery_pass"].fillna(False).astype(bool).all()
        )
        recovery_scale_pass = bool(
            len(recovered) == 1
            and recovered["ysocial_sized_recovery_pass"]
            .fillna(False)
            .astype(bool)
            .all()
        )
        degree_pass = bool(
            len(degree_row) == 1
            and ~degree_row["degree_reducible"].fillna(True).astype(bool).any()
        )
        interpretability_pass = bool(
            len(meaning) == 1
            and meaning["interpretability_pass"].fillna(False).astype(bool).all()
        )
        stress_test_pass = bool(
            len(stress_test) == 1
            and stress_test["opportunity_activity_stress_test_pass"]
            .fillna(False)
            .astype(bool)
            .all()
        )
        gates = {
            "predictive": predictive_pass,
            "forward_convergence": forward_pass,
            "main_mcmc": main_pass,
            "stability": stability_pass,
            "block_size": block_size_pass,
            "recovery": recovery_pass,
            "ysocial_sized_recovery": recovery_scale_pass,
            "degree_confounding": degree_pass,
            "interpretability": interpretability_pass,
            "opportunity_activity_stress_test": stress_test_pass,
        }
        values: dict[str, Any] = {
            "k": k,
            **{f"{name}_pass": value for name, value in gates.items()},
            "manuscript_prominence_pass": all(gates.values()),
            "gate_status": (
                "passed_all_latent_prominence_gates"
                if all(gates.values())
                else "failed:" + ",".join(name for name, value in gates.items() if not value)
            ),
        }
        for frame in (pooled, stable, recovered, degree_row, meaning, stress_test):
            if len(frame) == 1:
                for column, value in frame.iloc[0].items():
                    values.setdefault(column, value)
        rows.append(values)
    return pd.DataFrame(rows)


def _posterior_array(idata: Any, variable: str, *dimensions: str) -> np.ndarray:
    return (
        idata.posterior[variable]
        .stack(sample=("chain", "draw"))
        .transpose("sample", *dimensions)
        .values
    )


def _latent_heldout_predictions(idata: Any, training_data: Any, test_data: Any) -> pd.DataFrame:
    """Score future choices with learned memberships; unseen receivers fail the gate."""

    from .pipeline import _heldout_choice_predictions, _posterior_effect

    beta_da = idata.posterior["beta"]
    terms = tuple(str(value) for value in beta_da.coords["term"].values)
    term_indices = [test_data.terms.index(term) for term in terms]
    beta = _posterior_array(idata, "beta", "term")
    utility = np.einsum("nkp,sp->snk", test_data.x[:, :, term_indices], beta)
    for variable, dimension, train_labels, test_labels, row_index, ai_slope in [
        (
            "receiver_effect",
            "receiver",
            training_data.receiver_labels,
            test_data.receiver_labels,
            test_data.receiver_index,
            False,
        ),
        (
            "actor_ai_effect",
            "actor",
            training_data.actor_labels,
            test_data.actor_labels,
            test_data.actor_index,
            True,
        ),
        (
            "topic_ai_effect",
            "topic",
            training_data.topic_labels,
            test_data.topic_labels,
            test_data.topic_index,
            True,
        ),
    ]:
        effect = _posterior_effect(idata, variable, dimension)
        if effect is None:
            continue
        lookup = {label: index for index, label in enumerate(train_labels)}
        mapped = np.zeros((effect.shape[0], len(test_labels)), dtype=float)
        for test_index, label in enumerate(test_labels):
            train_index = lookup.get(label)
            if train_index is not None:
                mapped[:, test_index] = effect[:, train_index]
        contribution = mapped[:, row_index]
        if ai_slope:
            contribution = contribution[:, :, None] * test_data.candidate_is_ai[None, :, :]
        utility += contribution

    train_lookup = {label: index for index, label in enumerate(training_data.receiver_labels)}
    test_to_train = np.asarray(
        [train_lookup.get(label, -1) for label in test_data.receiver_labels], dtype=int
    )
    candidate_training_index = test_to_train[test_data.receiver_index]
    known = candidate_training_index >= 0
    membership = _posterior_array(idata, "receiver_block", "receiver").astype(int)
    block_intercept = _posterior_array(idata, "block_intercept", "block")
    block_weights = _posterior_array(idata, "block_weights", "block")
    safe_index = np.maximum(candidate_training_index, 0)
    candidate_block = membership[:, safe_index]
    sample = np.arange(membership.shape[0])[:, None, None]
    contribution = block_intercept[sample, candidate_block]
    expected_intercept = np.sum(block_weights * block_intercept, axis=1)
    contribution = np.where(known[None, :, :], contribution, expected_intercept[:, None, None])
    if "block_network_slope" in idata.posterior:
        slope_da = idata.posterior["block_network_slope"]
        network_terms = [str(value) for value in slope_da.coords["network_term"].values]
        slopes = _posterior_array(idata, "block_network_slope", "block", "network_term")
        for term_index, term in enumerate(network_terms):
            x = test_data.x[:, :, test_data.terms.index(term)]
            selected_slope = slopes[sample, candidate_block, term_index]
            expected_slope = np.sum(block_weights * slopes[:, :, term_index], axis=1)
            selected_slope = np.where(
                known[None, :, :], selected_slope, expected_slope[:, None, None]
            )
            contribution += selected_slope * x[None, :, :]
    utility += contribution
    utility = np.where(test_data.mask[None, :, :], utility, -np.inf)
    maximum = np.max(utility, axis=2, keepdims=True)
    weights = np.exp(utility - maximum) * test_data.mask[None, :, :]
    probability = weights / weights.sum(axis=2, keepdims=True)
    mean_probability = probability.mean(axis=0)
    choice_index = np.arange(test_data.n_choices)
    selected_probability = mean_probability[choice_index, test_data.chosen_position]
    chosen_ai = test_data.candidate_is_ai[choice_index, test_data.chosen_position]
    predicted_ai = np.sum(mean_probability * test_data.candidate_is_ai, axis=1)
    rank = 1 + np.asarray(
        [
            int(np.sum(row[test_data.mask[index]] > row[test_data.chosen_position[index]]))
            for index, row in enumerate(mean_probability)
        ]
    )
    result = pd.DataFrame(
        {
            "choice_set_id": test_data.choice_ids,
            "group": test_data.choice_groups,
            "actor_label": np.asarray(test_data.actor_labels, dtype=object)[
                test_data.actor_index
            ],
            "log_score": np.log(np.clip(selected_probability, 1.0e-12, 1.0)),
            "selected_target_rank": rank,
            "predicted_ai_probability": predicted_ai,
            "chosen_is_ai": chosen_ai,
        }
    )
    result["unseen_receiver_candidate_rows"] = np.sum(
        (~known) & test_data.mask, axis=1
    ).astype(int)
    observed = _heldout_choice_predictions(idata, training_data, test_data)
    if not result["choice_set_id"].equals(observed["choice_set_id"]):
        raise ValueError("Latent and observed held-out prediction order diverged")
    return result


def _fit_signed_latent(
    config: AnalysisConfig,
    data: Any,
    *,
    k: int,
    path: Path,
    role: str,
    network_terms: Sequence[str],
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
    resume: bool,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Fit or resume one signed latent model, with one bounded extension."""

    from .model import fit_latent_receiver_blocks
    from .pipeline import (
        _load_compatible_posterior,
        _posterior_request_signature,
        _write_posterior_atomic,
    )

    request = _posterior_request_signature(
        config,
        data,
        model=f"latent_K{k}",
        role=role,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )
    request["latent_network_terms"] = list(network_terms)
    request["latent_gate_protocol"] = "v1"
    idata = (
        _load_compatible_posterior(
            path,
            draws=draws,
            chains=chains,
            request_signature=request,
        )
        if resume
        else None
    )
    if idata is None:
        idata = fit_latent_receiver_blocks(
            data,
            config,
            K=k,
            network_terms=network_terms,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
        )
        signature = _write_posterior_atomic(idata, path, request)
    else:
        import json

        signature = json.loads(
            path.with_suffix(path.suffix + ".fit.json").read_text(encoding="utf-8")
        )
    diagnostic = summarize_mcmc(idata)
    saved_request = signature["request"]
    already_extended = int(saved_request["draws"]) > draws or int(saved_request["tune"]) > tune
    if diagnostic["passed"] or already_extended:
        diagnostic["sampling_extended"] = already_extended
        diagnostic["requested_draws"] = int(saved_request["draws"])
        diagnostic["requested_tune"] = int(saved_request["tune"])
        diagnostic["requested_chains"] = int(saved_request["chains"])
        return idata, signature, diagnostic
    close = getattr(idata, "close", None)
    if callable(close):
        close()
    extended_request = dict(request)
    extended_request["draws"] = 2 * draws
    extended_request["tune"] = 2 * tune
    idata = fit_latent_receiver_blocks(
        data,
        config,
        K=k,
        network_terms=network_terms,
        draws=2 * draws,
        tune=2 * tune,
        chains=chains,
        random_seed=random_seed,
        progressbar=False,
    )
    signature = _write_posterior_atomic(idata, path, extended_request)
    diagnostic = summarize_mcmc(idata)
    diagnostic["sampling_extended"] = True
    diagnostic["requested_draws"] = 2 * draws
    diagnostic["requested_tune"] = 2 * tune
    diagnostic["requested_chains"] = chains
    return idata, signature, diagnostic


def run_latent_validation(
    config: AnalysisConfig,
    raw_model_frame: pd.DataFrame,
    term_manifest: pd.DataFrame,
    full_data: Any,
    *,
    draws: int,
    tune: int,
    chains: int,
    resume: bool,
    recovery_design: LatentRecoveryDesign,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], list[Path]]:
    """Fit and validate the optional latent extension across every decision gate."""

    from .model import (
        latent_block_recovery_gate,
        prepare_model_data,
        simulate_latent_block_recovery,
    )
    from .pipeline import (
        _fit_with_diagnostic_extension,
        _heldout_choice_predictions,
        _signature_row,
    )

    recovery_design.validate()
    scale_reference = recovery_scale_reference(raw_model_frame)
    ysocial_sized_recovery = recovery_design_is_ysocial_sized(
        recovery_design, scale_reference
    )
    network_terms = [term for term in LATENT_NETWORK_TERMS if term in full_data.terms]
    if list(network_terms) != list(LATENT_NETWORK_TERMS):
        raise ValueError(
            "Latent validation requires both prespecified network terms: "
            + ", ".join(LATENT_NETWORK_TERMS)
        )
    signature_rows: list[dict[str, Any]] = []
    model_paths: list[Path] = []
    diagnostics: list[dict[str, Any]] = []
    stability_tables: list[pd.DataFrame] = []
    activity_tables: list[pd.DataFrame] = []
    degree_tables: list[pd.DataFrame] = []
    meaning_tables: list[pd.DataFrame] = []
    contrast_tables: list[pd.DataFrame] = []
    for k in config.latent_block_sizes:
        path = config.models_dir / f"latent_K{k}.nc"
        idata, signature, diagnostic = _fit_signed_latent(
            config,
            full_data,
            k=k,
            path=path,
            role="optional_latent_receiver_blocks_main",
            network_terms=network_terms,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=config.seed + 200 + k,
            resume=resume,
        )
        signature_rows.append(_signature_row(path, signature, full_data))
        model_paths.append(path)
        diagnostics.append({"k": k, "role": "main", "fold": np.nan, **diagnostic})
        stability, partition = membership_stability(idata, k)
        activity = receiver_activity_table(full_data, partition, k)
        activity.insert(0, "k", k)
        degree = degree_confounding_diagnostic(activity, k, config.seed + k)
        meaning, contrasts = block_interpretability(
            idata, full_data, activity, k, network_terms
        )
        stability_tables.append(stability)
        activity_tables.append(activity)
        degree_tables.append(degree)
        meaning_tables.append(meaning)
        contrast_tables.append(contrasts)

    assignments = rolling_origin_assignments(raw_model_frame)
    prediction_tables: list[pd.DataFrame] = []
    for fold in sorted(assignments["fold"].unique()):
        split = assignments[assignments["fold"].eq(fold)]
        train_ids = set(split.loc[split["role"].eq("train"), "choice_set_id"])
        test_ids = set(split.loc[split["role"].eq("test"), "choice_set_id"])
        train_raw = raw_model_frame[raw_model_frame["choice_set_id"].isin(train_ids)]
        _, fold_manifest = build_model_frame(train_raw)
        transformed = apply_model_term_manifest(raw_model_frame, fold_manifest)
        train_frame = transformed[transformed["choice_set_id"].isin(train_ids)]
        test_frame = transformed[transformed["choice_set_id"].isin(test_ids)]
        available = included_terms_for_model("M3", fold_manifest)
        numeric_terms = tuple(term for term in available if term != "candidate_target_is_ai")
        train_data = prepare_model_data(
            train_frame,
            numeric_terms,
            ai_term="candidate_target_is_ai",
            closure_term="z_log_two_path_closure",
        )
        test_data = prepare_model_data(
            test_frame,
            numeric_terms,
            ai_term="candidate_target_is_ai",
            closure_term="z_log_two_path_closure",
        )
        observed_path = config.models_dir / f"M3_forward_fold{fold}.nc"
        observed_idata, observed_signature, observed_diagnostic = (
            _fit_with_diagnostic_extension(
                config,
                train_data,
                "M3",
                observed_path,
                role=f"forward_fold_{int(fold)}",
                draws=draws,
                tune=tune,
                chains=chains,
                seed_offset=300 + 10 * int(fold) + 3,
                resume=resume,
            )
        )
        signature_rows.append(_signature_row(observed_path, observed_signature, train_data))
        model_paths.append(observed_path)
        observed_prediction = _heldout_choice_predictions(
            observed_idata, train_data, test_data
        )
        for k in config.latent_block_sizes:
            diagnostics.append(
                {
                    "k": k,
                    "role": "observed_forward",
                    "fold": int(fold),
                    "requested_draws": int(observed_signature["request"]["draws"]),
                    "requested_tune": int(observed_signature["request"]["tune"]),
                    "requested_chains": int(observed_signature["request"]["chains"]),
                    **observed_diagnostic,
                }
            )
            latent_path = config.models_dir / f"latent_K{k}_forward_fold{fold}.nc"
            latent_idata, latent_signature, latent_diagnostic = _fit_signed_latent(
                config,
                train_data,
                k=k,
                path=latent_path,
                role=f"optional_latent_forward_fold_{int(fold)}",
                network_terms=network_terms,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=config.seed + 700 + 10 * int(fold) + k,
                resume=resume,
            )
            signature_rows.append(_signature_row(latent_path, latent_signature, train_data))
            model_paths.append(latent_path)
            diagnostics.append(
                {"k": k, "role": "forward", "fold": int(fold), **latent_diagnostic}
            )
            latent_prediction = _latent_heldout_predictions(
                latent_idata, train_data, test_data
            )
            observed_for_k = observed_prediction.copy()
            observed_for_k.insert(0, "model", "M3_observed")
            observed_for_k.insert(0, "k", k)
            observed_for_k.insert(0, "fold", int(fold))
            observed_for_k["unseen_receiver_candidate_rows"] = 0
            latent_prediction.insert(0, "model", f"latent_K{k}")
            latent_prediction.insert(0, "k", k)
            latent_prediction.insert(0, "fold", int(fold))
            prediction_tables.extend([observed_for_k, latent_prediction])

    recovery_rows: list[dict[str, Any]] = []
    recovery_design_rows: list[dict[str, Any]] = []
    for k in config.latent_block_sizes:
        for replicate in range(recovery_design.replicates):
            fixture_seed = config.seed + 1_000 * k + replicate
            fixture = simulate_latent_block_recovery(
                K=k,
                n_actors=recovery_design.n_actors,
                n_receivers=recovery_design.n_receivers,
                n_choice_sets=recovery_design.n_choice_sets,
                candidates_per_set=recovery_design.candidates_per_set,
                random_seed=fixture_seed,
            )
            recovery_data = prepare_model_data(fixture.choice_rows, ["prior_contact"])
            recovery_path = config.models_dir / f"latent_K{k}_recovery_seed{fixture_seed}.nc"
            recovery_idata, recovery_signature, recovery_diagnostic = _fit_signed_latent(
                config,
                recovery_data,
                k=k,
                path=recovery_path,
                role=f"bounded_recovery_replicate_{replicate}",
                network_terms=["prior_contact"],
                draws=recovery_design.draws,
                tune=recovery_design.tune,
                chains=recovery_design.chains,
                random_seed=fixture_seed + 50_000,
                resume=resume,
            )
            signature_rows.append(
                _signature_row(recovery_path, recovery_signature, recovery_data)
            )
            model_paths.append(recovery_path)
            result = latent_block_recovery_gate(recovery_idata, recovery_data, fixture)
            recovery_rows.append(
                {
                    "k": k,
                    "replicate": replicate,
                    "fixture_seed": fixture_seed,
                    "simulation_scale": recovery_design.scale_label,
                    "ysocial_sized_simulation": ysocial_sized_recovery,
                    "mcmc_passed": bool(recovery_diagnostic["passed"]),
                    **result,
                }
            )
            recovery_design_rows.append(
                {
                    "k": k,
                    "replicate": replicate,
                    "simulation_scale": recovery_design.scale_label,
                    "n_actors": recovery_design.n_actors,
                    "n_receivers": recovery_design.n_receivers,
                    "n_choice_sets": recovery_design.n_choice_sets,
                    "candidates_per_set": recovery_design.candidates_per_set,
                    "draws": recovery_design.draws,
                    "tune": recovery_design.tune,
                    "chains": recovery_design.chains,
                    "ysocial_sized_simulation": ysocial_sized_recovery,
                    **scale_reference,
                    "scope": "bounded_conditional_choice_recovery_not_recursive_event_history",
                }
            )
            diagnostics.append(
                {
                    "k": k,
                    "role": "recovery",
                    "fold": np.nan,
                    "replicate": replicate,
                    **recovery_diagnostic,
                }
            )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    comparisons = paired_forward_comparisons(predictions)
    stability = pd.concat(stability_tables, ignore_index=True)
    activity = pd.concat(activity_tables, ignore_index=True)
    degree = pd.concat(degree_tables, ignore_index=True)
    meaning = pd.concat(meaning_tables, ignore_index=True)
    contrasts = pd.concat(contrast_tables, ignore_index=True)
    recovery_results = pd.DataFrame(recovery_rows)
    recovered = recovery_summary(recovery_results)
    diagnostic_table = pd.DataFrame(diagnostics)
    opportunity_activity_stress_test = opportunity_activity_stress_test_not_run(
        config.latent_block_sizes
    )
    status = assemble_gate_status(
        comparisons,
        stability,
        degree,
        meaning,
        recovered,
        diagnostic_table,
        opportunity_activity_stress_test,
    )
    return (
        {
            "latent_block_gate_status": status,
            "latent_forward_predictions": predictions,
            "latent_forward_comparisons": comparisons,
            "latent_fit_diagnostics": diagnostic_table,
            "latent_membership_stability": stability,
            "latent_receiver_profiles": activity,
            "latent_degree_confounding": degree,
            "latent_interpretability": meaning,
            "latent_network_slope_contrasts": contrasts,
            "latent_recovery_results": recovery_results,
            "latent_recovery_summary": recovered,
            "latent_recovery_design": pd.DataFrame(recovery_design_rows),
            "latent_opportunity_activity_stress_test": (
                opportunity_activity_stress_test
            ),
        },
        signature_rows,
        model_paths,
    )


__all__ = [
    "LatentRecoveryDesign",
    "assemble_gate_status",
    "block_interpretability",
    "degree_confounding_diagnostic",
    "membership_stability",
    "not_run_status",
    "opportunity_activity_stress_test_not_run",
    "paired_forward_comparisons",
    "receiver_activity_table",
    "recovery_design_is_ysocial_sized",
    "recovery_scale_reference",
    "recovery_summary",
    "run_latent_validation",
]
