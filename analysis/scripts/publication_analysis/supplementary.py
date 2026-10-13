"""Claim-bounded supplementary tables backed by current or frozen artifacts.

The streamlined pipeline does not rerun every historical analysis. This module
turns the smaller set of retained supplementary results into an auditable
inventory and extracts topic heterogeneity only from a signed publication-
length M3 posterior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import read_table, sha256_file, write_table


TOPIC_COLUMNS = [
    "topic",
    "group",
    "ai_log_utility_contrast_median",
    "ai_log_utility_contrast_ci_low",
    "ai_log_utility_contrast_ci_high",
    "conditional_odds_ratio_median",
    "conditional_odds_ratio_ci_low",
    "conditional_odds_ratio_ci_high",
    "topic_deviation_median",
    "topic_deviation_ci_low",
    "topic_deviation_ci_high",
    "posterior_draws",
    "evidence_status",
    "source_path",
    "source_sha256",
    "estimand_scope",
    "claim_boundary",
]

TOPIC_CLAIM_BOUNDARY = (
    "Conditional log-utility comparison of otherwise equal AI- and human-authored "
    "candidates within an observed same-thread choice set, with actor and receiver "
    "effects at their population means; not a marginal topic probability or a "
    "causal arm effect."
)


LEGACY_SUPPLEMENTS: tuple[Mapping[str, str], ...] = (
    {
        "supplement": "topic_heterogeneity_legacy_benchmark",
        "filename": "61_topic_affordance_hai_tests.csv",
        "analysis_role": "reproduction_benchmark",
        "recommended_output": "inventory_only_after_current_M3_is_available",
        "claim_boundary": (
            "Aggregated topic comparisons do not condition on the reply risk set. "
            "Use them only to reconcile the current hierarchical M3 topic result."
        ),
    },
    {
        "supplement": "reply_toxicity_toxigen",
        "filename": "08_toxigen_toxicity.csv",
        "analysis_role": "legacy_supplementary_valence",
        "recommended_output": "compact_supplementary_table",
        "claim_boundary": (
            "Toxic language is one dimension of reply valence. It does not identify "
            "approval, correction, testing, or relational uptake."
        ),
    },
    {
        "supplement": "reply_toxicity_detoxify",
        "filename": "08b_detoxify_toxicity.csv",
        "analysis_role": "legacy_supplementary_robustness",
        "recommended_output": "classifier_robustness_table",
        "claim_boundary": (
            "This classifier is a robustness measure for harmful language and is not "
            "a general measure of positive or negative social response."
        ),
    },
    {
        "supplement": "openai_moderation",
        "filename": "09_openai_moderation_flags_BLOCKED.md",
        "analysis_role": "unavailable_optional_measure",
        "recommended_output": "status_only",
        "claim_boundary": (
            "No moderation scores were produced. The blocked marker cannot support a "
            "content-safety comparison."
        ),
    },
    {
        "supplement": "fixed_observed_type_sbm",
        "filename": "07_two_block_SBM.csv",
        "analysis_role": "legacy_static_appendix_diagnostic",
        "recommended_output": "compact_supplementary_table",
        "claim_boundary": (
            "This table summarizes aggregated directed edge density by observed H/AI "
            "type. It does not condition on opportunities or model event timing."
        ),
    },
    {
        "supplement": "degree_corrected_residual_sbm",
        "filename": "21_residual_sbm_model_summary.csv",
        "analysis_role": "legacy_static_appendix_diagnostic",
        "recommended_output": "compact_supplementary_table",
        "claim_boundary": (
            "Blocks describe a static residual graph and may retain activity or degree "
            "heterogeneity. They do not establish dynamic receiver roles."
        ),
    },
    {
        "supplement": "residual_sbm_null_checks",
        "filename": "60_residual_sbm_null_tests.csv",
        "analysis_role": "legacy_static_appendix_robustness",
        "recommended_output": "compact_supplementary_table",
        "claim_boundary": (
            "Null checks qualify residual type alignment in the aggregated graph. They "
            "do not replace opportunity-conditioned forward validation."
        ),
    },
    {
        "supplement": "bayesian_residual_architecture",
        "filename": "44_bayesian_posterior_residual_architecture_block_summary.csv",
        "analysis_role": "legacy_static_appendix_diagnostic",
        "recommended_output": "inventory_only",
        "claim_boundary": (
            "This posterior residual architecture comes from the aggregated dyadic "
            "model and remains separate from the primary relational-event posterior."
        ),
    },
)


def _quantile_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Posterior topic summaries require finite draws")
    return (
        float(np.median(finite)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def _topic_contrast_row(
    topic: Any,
    group: str,
    focal: np.ndarray,
    arm_interaction: np.ndarray | float,
    deviation: np.ndarray,
) -> dict[str, Any]:
    total = focal + deviation + arm_interaction
    total_summary = _quantile_summary(total)
    odds_summary = _quantile_summary(np.exp(np.clip(total, -700.0, 700.0)))
    deviation_summary = _quantile_summary(deviation)
    return {
        "topic": str(topic),
        "group": group,
        "ai_log_utility_contrast_median": total_summary[0],
        "ai_log_utility_contrast_ci_low": total_summary[1],
        "ai_log_utility_contrast_ci_high": total_summary[2],
        "conditional_odds_ratio_median": odds_summary[0],
        "conditional_odds_ratio_ci_low": odds_summary[1],
        "conditional_odds_ratio_ci_high": odds_summary[2],
        "topic_deviation_median": deviation_summary[0],
        "topic_deviation_ci_low": deviation_summary[1],
        "topic_deviation_ci_high": deviation_summary[2],
        "posterior_draws": int(len(total)),
        "evidence_status": "available_signed_publication_length_M3",
        "source_path": None,
        "source_sha256": None,
        "estimand_scope": "population_mean_actor_receiver_conditional_comparison",
        "claim_boundary": TOPIC_CLAIM_BOUNDARY,
    }


def summarize_topic_heterogeneity(posterior: Any) -> pd.DataFrame:
    """Summarize topic-specific conditional AI contrasts from an M3 posterior.

    The contrast is on the conditional-logit utility scale. It compares otherwise
    equal AI- and human-authored candidates in the same choice set; it is not a
    topic-standardized selection probability.
    """

    required = {"beta", "topic_ai_effect"}
    missing = sorted(required.difference(posterior.data_vars))
    if missing:
        raise ValueError(f"M3 posterior is missing topic variables: {missing}")
    if "term" not in posterior["beta"].coords or "topic" not in posterior["topic_ai_effect"].coords:
        raise ValueError("M3 posterior is missing term or topic coordinates")

    terms = {str(value) for value in posterior["beta"].coords["term"].values}
    focal_terms = {"candidate_target_is_ai", "g2_x_candidate_ai"}
    missing_terms = sorted(focal_terms.difference(terms))
    if missing_terms:
        raise ValueError(f"M3 posterior is missing focal terms: {missing_terms}")

    focal = np.asarray(
        posterior["beta"].sel(term="candidate_target_is_ai").values,
        dtype=float,
    ).reshape(-1)
    arm_interaction = np.asarray(
        posterior["beta"].sel(term="g2_x_candidate_ai").values,
        dtype=float,
    ).reshape(-1)
    topic_effect = posterior["topic_ai_effect"]
    rows: list[dict[str, Any]] = []
    for topic in topic_effect.coords["topic"].values:
        deviation = np.asarray(topic_effect.sel(topic=topic).values, dtype=float).reshape(-1)
        if len(deviation) != len(focal):
            raise ValueError("Topic and fixed-effect posterior draw counts differ")
        for group, interaction in (("G1", 0.0), ("G2", arm_interaction)):
            rows.append(
                _topic_contrast_row(topic, group, focal, interaction, deviation)
            )
    return pd.DataFrame(rows, columns=TOPIC_COLUMNS)


def _unavailable_topic_table(status: str, detail: str) -> pd.DataFrame:
    row = {column: None for column in TOPIC_COLUMNS}
    row.update(
        {
            "posterior_draws": 0,
            "evidence_status": status,
            "estimand_scope": "population_mean_actor_receiver_conditional_comparison",
            "claim_boundary": detail,
        }
    )
    return pd.DataFrame([row], columns=TOPIC_COLUMNS)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _signed_m3_record(config: AnalysisConfig) -> tuple[pd.Series | None, str]:
    try:
        signatures = read_table(config.tables_dir / "fit_run_signature")
    except FileNotFoundError:
        return None, "fit_run_signature_missing"
    required = {
        "model",
        "role",
        "posterior_path",
        "posterior_sha256",
        "dimension_match",
        "publication_length_request",
    }
    if not required.issubset(signatures.columns):
        return None, "fit_run_signature_incomplete"
    selected = signatures[
        signatures["model"].eq("M3")
        & signatures["role"].eq("main_nested_sequence")
    ]
    selected = selected[
        selected["dimension_match"].map(_as_bool)
        & selected["publication_length_request"].map(_as_bool)
    ]
    if len(selected) != 1:
        return None, "signed_publication_length_M3_not_unique_or_missing"
    return selected.iloc[0], "available"


def _authenticated_m3_path(
    config: AnalysisConfig,
) -> tuple[Path | None, str, str | None]:
    record, status = _signed_m3_record(config)
    if record is None:
        return None, status, None
    posterior_path = Path(str(record["posterior_path"])).resolve()
    expected_path = (config.models_dir / "M3.nc").resolve()
    if posterior_path != expected_path or not posterior_path.exists():
        return None, "signed_M3_path_invalid", None
    actual_sha = sha256_file(posterior_path)
    if actual_sha != str(record["posterior_sha256"]):
        return None, "signed_M3_hash_mismatch", actual_sha
    sidecar_path = posterior_path.with_suffix(posterior_path.suffix + ".fit.json")
    if not sidecar_path.exists():
        return None, "signed_M3_sidecar_missing", actual_sha
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("actual", {}).get("posterior_sha256") != actual_sha:
        return None, "signed_M3_sidecar_hash_mismatch", actual_sha
    return posterior_path, "available", actual_sha


def _topic_authentication_detail(status: str) -> str:
    details = {
        "fit_run_signature_missing": "The signed fit record is missing.",
        "fit_run_signature_incomplete": "The signed fit record is incomplete.",
        "signed_publication_length_M3_not_unique_or_missing": (
            "A unique signed publication-length main M3 posterior is required."
        ),
        "signed_M3_path_invalid": (
            "The signed M3 record does not resolve to the canonical run posterior."
        ),
        "signed_M3_hash_mismatch": (
            "The M3 posterior hash differs from the signed fit record."
        ),
        "signed_M3_sidecar_missing": (
            "The M3 posterior provenance sidecar is required."
        ),
        "signed_M3_sidecar_hash_mismatch": (
            "The M3 posterior sidecar does not authenticate the current file."
        ),
    }
    return details.get(status, "The M3 posterior failed supplementary authentication.")


def build_topic_heterogeneity(
    config: AnalysisConfig,
    *,
    release_eligible: bool,
) -> pd.DataFrame:
    """Build a release-gated topic table from the signed main M3 posterior."""

    if not release_eligible:
        return _unavailable_topic_table(
            "withheld_primary_release_gates_incomplete",
            "Topic posterior estimates remain withheld until all required primary "
            "model release gates pass.",
        )
    posterior_path, status, actual_sha = _authenticated_m3_path(config)
    if posterior_path is None:
        return _unavailable_topic_table(
            status,
            _topic_authentication_detail(status),
        )

    import arviz as az

    idata = az.from_netcdf(posterior_path)
    try:
        table = summarize_topic_heterogeneity(idata.posterior)
    finally:
        idata.close()
    table["source_path"] = str(posterior_path)
    table["source_sha256"] = str(actual_sha)
    return table


def _legacy_csv_window(path: Path, config: AnalysisConfig) -> tuple[int | None, str]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None, "unreadable_legacy_artifact"
    if frame.empty:
        return 0, "empty_legacy_artifact"
    required = {"analysis_set", "human_start_cutoff", "strict_thread_validity"}
    if not required.issubset(frame.columns):
        return int(len(frame)), "legacy_window_metadata_incomplete"
    analysis_sets = set(frame["analysis_set"].dropna().astype(str))
    strict_values = set(frame["strict_thread_validity"].dropna().map(_as_bool))
    cutoffs = pd.to_datetime(frame["human_start_cutoff"], errors="coerce", utc=True).dropna()
    expected_cutoff = pd.Timestamp(config.start_time).tz_convert("UTC")
    compatible = (
        analysis_sets == {config.analysis_set}
        and strict_values == {True}
        and len(cutoffs) == len(frame)
        and bool((cutoffs == expected_cutoff).all())
    )
    return int(len(frame)), (
        "compatible_primary_valid_window" if compatible else "legacy_window_mismatch"
    )


def _source_record(
    path: Path | None,
    config: AnalysisConfig,
    *,
    blocked_marker: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {
            "source_path": None,
            "source_exists": False,
            "source_sha256": None,
            "source_rows": None,
            "source_window_status": "no_source_artifact",
        }
    exists = path.exists() and path.is_file()
    if not exists:
        return {
            "source_path": str(path),
            "source_exists": False,
            "source_sha256": None,
            "source_rows": None,
            "source_window_status": "missing_legacy_artifact",
        }
    if blocked_marker:
        rows, window_status = None, "blocked_marker_without_results"
    elif path.suffix.lower() == ".csv":
        rows, window_status = _legacy_csv_window(path, config)
    elif path.suffix.lower() == ".parquet":
        try:
            rows = int(len(pd.read_parquet(path)))
            window_status = "current_run_canonical_table"
        except (OSError, ValueError):
            rows, window_status = None, "unreadable_current_run_artifact"
    else:
        rows, window_status = None, "non_tabular_status_artifact"
    return {
        "source_path": str(path.resolve()),
        "source_exists": True,
        "source_sha256": sha256_file(path),
        "source_rows": rows,
        "source_window_status": window_status,
    }


def _canonical_status(source: Mapping[str, Any], *, blocked_marker: bool = False) -> str:
    if not source["source_exists"]:
        return "unavailable_missing_artifact"
    if blocked_marker:
        return "unavailable_blocked_no_result"
    if source["source_window_status"] == "compatible_primary_valid_window":
        return "available_frozen_legacy_appendix_only"
    return "unavailable_window_or_provenance_mismatch"


def _legacy_inventory_rows(config: AnalysisConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specification in LEGACY_SUPPLEMENTS:
        filename = specification["filename"]
        path = (
            None
            if config.legacy_tables_dir is None
            else config.legacy_tables_dir / filename
        )
        blocked = filename.endswith("_BLOCKED.md")
        source = _source_record(path, config, blocked_marker=blocked)
        rows.append(
            {
                "supplement": specification["supplement"],
                "analysis_role": specification["analysis_role"],
                "canonical_status": _canonical_status(source, blocked_marker=blocked),
                "source_kind": "frozen_legacy_artifact",
                **source,
                "recommended_output": specification["recommended_output"],
                "claim_boundary": specification["claim_boundary"],
            }
        )
    return rows


def _reaction_source_status(
    config: AnalysisConfig,
) -> tuple[dict[str, Any], str]:
    source_path = config.tables_dir / "source_availability.parquet"
    source = _source_record(source_path, config)
    try:
        availability = read_table(config.tables_dir / "source_availability")
        reaction_rows = availability[
            availability["source"]
            .astype(str)
            .str.contains("reaction|upvote|downvote", regex=True)
        ]
        if reaction_rows.empty:
            return source, "unavailable_reaction_availability_unreported"
        observed = sorted(set(reaction_rows["status"].dropna().astype(str)))
        return source, ";".join(observed)
    except (FileNotFoundError, KeyError):
        return source, "unavailable_source_availability_missing"


def _evidence_gap_rows(config: AnalysisConfig) -> list[dict[str, Any]]:
    availability_source, reaction_status = _reaction_source_status(config)
    return [
        {
            "supplement": "timestamped_reaction_valence",
            "analysis_role": "unavailable_marked_behavior",
            "canonical_status": reaction_status,
            "source_kind": "current_source_availability_audit",
            **availability_source,
            "recommended_output": "status_only_until_event_timestamps_exist",
            "claim_boundary": (
                "Untimestamped backend marks cannot enter the valid clock, pre-event "
                "history, or arm-specific action hazard. Upvotes and downvotes remain "
                "separate outcomes when timestamps become available."
            ),
        },
        {
            "supplement": "semantic_reply_valence",
            "analysis_role": "unavailable_unvalidated_measure",
            "canonical_status": "unavailable_no_validated_measure",
            "source_kind": "declared_measurement_gap",
            **_source_record(None, config),
            "recommended_output": "status_only_until_annotation_or_validated_classifier",
            "claim_boundary": (
                "Toxicity cannot substitute for approval, correction, hostility, "
                "testing, or sanction. No validated multi-class reply-valence measure "
                "is present."
            ),
        },
    ]


def build_supplementary_inventory(
    config: AnalysisConfig,
    topic_table: pd.DataFrame,
    topic_path: Path,
) -> pd.DataFrame:
    """Inventory current, legacy, and unavailable supplementary evidence."""

    rows: list[dict[str, Any]] = []
    topic_source = _source_record(topic_path, config)
    topic_status = str(topic_table["evidence_status"].iloc[0])
    rows.append(
        {
            "supplement": "hierarchical_topic_ai_selection",
            "analysis_role": "current_model_supplement",
            "canonical_status": topic_status,
            "source_kind": "current_signed_posterior_summary",
            **topic_source,
            "recommended_output": (
                "supplementary_table_and_figure"
                if topic_status == "available_signed_publication_length_M3"
                else "withhold_estimates_status_only"
            ),
            "claim_boundary": str(topic_table["claim_boundary"].iloc[0]),
        }
    )

    rows.extend(_legacy_inventory_rows(config))
    rows.extend(_evidence_gap_rows(config))
    return pd.DataFrame(rows)


def write_supplementary_outputs(
    config: AnalysisConfig,
    *,
    release_eligible: bool,
) -> dict[str, pd.DataFrame]:
    """Write the two canonical supplementary tables and return their frames."""

    topic = build_topic_heterogeneity(config, release_eligible=release_eligible)
    topic_paths = write_table(topic, config.tables_dir / "supplementary_topic_heterogeneity")
    topic_path = Path(topic_paths["parquet"])
    inventory = build_supplementary_inventory(config, topic, topic_path)
    write_table(inventory, config.tables_dir / "supplementary_source_inventory")
    return {
        "supplementary_topic_heterogeneity": topic,
        "supplementary_source_inventory": inventory,
    }
