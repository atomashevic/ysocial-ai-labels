"""Resumable orchestration for the publication analysis surface."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .browser import prepare_browser
from .browser_hazard import (
    build_piecewise_exposure,
    fit_piecewise_hazards,
    mark_unobserved_action_channels,
    standardized_piecewise_rates,
    strict_browser_choice_sets,
)
from .config import AnalysisConfig
from .data import build_network_history, prepare_choice_data, prepare_data
from .io import ensure_directories, read_table, sha256_file, write_json, write_table
from .manifest import source_paths, write_artifact_manifests, write_manifest
from .provenance import (
    BUNDLE_FILENAME,
    assert_source_snapshot_unchanged,
    authenticate_fit_result_bundle,
    capture_source_snapshot,
    write_fit_result_bundle,
)
from .report import run_report
from .study import analysis_set_manifest
from .terms import apply_model_term_manifest, build_model_frame, included_terms_for_model
from .validation import (
    authenticate_fit_signatures,
    build_publication_readiness,
    build_rq1_readiness,
    data_integrity_checks,
    grouped_choice_psis,
    participant_grouped_psis,
    rolling_origin_assignments,
    run_validation,
    summarize_mcmc,
    write_network_audit,
)


PROCESS_CODE_SNAPSHOT = tuple(
    capture_source_snapshot(Path(__file__).resolve().parent.glob("*.py"))
)


TABLE_DENOMINATORS: Mapping[str, str] = {
    "event_history": "valid analysis-set platform events with strict root and target validity",
    "thread_choice_sets": "eligible prior same-thread content items for retained human replies",
    "network_history": "thread-choice candidates with network state strictly before the reply",
    "browser_opportunity_episodes": "ordered DOM content-visibility episodes",
    "browser_linked_actions": "canonical human backend actions eligible for browser linkage",
    "browser_manual_audit_sample": "deterministic stratified linked-action audit template",
    "browser_interval_audit_sample": "deterministic stratified nonexact-episode closure audit template",
    "browser_choice_sets": "thread-choice candidates annotated with nullable browser observation",
    "browser_strict_choice_sets": "pre-action rendered candidates in retained unambiguous browser sets",
    "tracking_linkage_qc": "tracking, linkage, source-integrity, and authenticated audit gates",
    "browser_conversion_summary": "verified visible content-time by arm, action, and author type",
    "browser_standardized_conversion": "piecewise-model conversion rates standardized over pooled verified time bins",
    "browser_hazard_intervals": "episode-time with explicit action-channel observation status",
    "browser_hazard_terms": "action-specific piecewise exposure-time model terms",
    "source_availability": "planned dynamic sources and their usable timestamp coverage",
    "choice_set_attrition": "all valid human reply events classified by risk-set eligibility",
    "risk_set_composition": "retained human reply choice sets by H/AI candidate variation",
    "actor_label_qc": "cross-source unique accounts by arm and authoritative H/AI field",
    "account_universe": "all authoritative arm-scoped H/AI accounts, including zero-degree accounts",
    "group_database_provenance": "one immutable source-manifest identity record per arm",
    "model_frame": "network-history candidate rows with prespecified finite transformations",
    "model_term_manifest": "prespecified term definitions and source-availability decisions",
    "data_integrity_checks": "publication-blocking canonical data invariants",
    "fit_run_signature": "one row per signed posterior artifact",
    "fit_signature_authentication": "validation-time posterior byte and request authentication",
    "analysis_data_signature": "one hash of the current prepared primary model data and implementation",
    "publication_readiness": "declared computational and claim-specific decision gates",
    "model_parameter_summary": "posterior draws from the signed nested model sequence",
    "model_estimands": "opportunity-standardized projections from each signed nested model",
    "primary_estimands": "opportunity-standardized projections from the final observed-type model",
    "model_diagnostics": "one convergence summary per signed posterior fit",
    "posterior_predictive_checks": "recursive network-process summaries by arm and statistic",
    "participant_grouped_psis": "human participants as influence-diagnostic units",
    "participant_grouped_psis_all_models": "participant-grouped PSIS diagnostics for every nested model",
    "participant_loo_refits": "one exact deletion refit per problematic human participant",
    "participant_loo_refit_authentication": "validation-time rebuilt grouped deletion contributions",
    "participant_grouped_reloo": "participant-level grouped PSIS contributions with exact conditional-history replacements",
    "participant_grouped_reloo_summary": "reloo-adjusted grouped predictive score and participant-cluster uncertainty",
    "model_loo": "choice-set pointwise log-likelihood contributions",
    "prior_scale_sensitivity": "primary estimands under original and doubled prior scales",
    "browser_strict_model_estimands": "verified pre-action browser choice sets",
    "browser_full_vs_strict_estimand_comparison": "matched full-thread and strict rendered-risk-set projections",
    "browser_strict_model_diagnostics": "signed strict-browser sensitivity posterior",
    "forward_validation_assignments": "ordered choice sets assigned to rolling train and future-test windows",
    "forward_validation_scores": "future choice sets scored by fold, arm, and nested model",
    "forward_validation_comparisons": "paired future choice sets comparing M3 with M0-M2",
    "forward_validation_predictions": "future choice-set predictions from each nested model",
    "forward_validation_diagnostics": "one convergence summary per rolling-origin model fit",
    "latent_block_gate_status": "optional latent receiver-block candidates by prespecified gate",
    "latent_forward_predictions": "paired future choice-set predictions from observed M3 and latent receiver roles",
    "latent_forward_comparisons": "paired rolling-origin ELPD contrasts between latent roles and observed M3",
    "latent_fit_diagnostics": "main and rolling-origin MCMC diagnostics for optional latent role models",
    "latent_membership_stability": "receiver-role agreement across independently initialized chains",
    "latent_receiver_profiles": "receiver role assignments with observed opportunity and selected-target activity",
    "latent_degree_confounding": "cross-validated reconstruction of roles from within-type residual activity",
    "latent_interpretability": "covariate-adjusted role separation and observed-type reduction diagnostics",
    "latent_network_slope_contrasts": "posterior block contrasts for prespecified network-history slopes",
    "latent_recovery_results": "seed-specific bounded conditional-choice partition recovery",
    "latent_recovery_summary": "multi-seed recovery gate by candidate latent block count",
    "latent_recovery_design": "exact bounded simulation and sampling dimensions for recovery checks",
    "latent_opportunity_activity_stress_test": "fail-closed evidence status for degree/opportunity role stress testing",
    "observed_network_metrics": "selected human reply targets by arm",
    "event_timing_posterior_checks": "nonnegative inter-human-reply gaps on the valid-analysis clock",
    "fit_result_bundle_authentication": "one fit-result bundle bound to source, table, and posterior bytes",
    "participant_grouped_psis_validated": "validation-time participant-grouped influence diagnostics recomputed from signed M3",
    "forward_signature_expectations": "freshly rebuilt rolling-origin training-data hashes by fold and model",
    "rq1_model_parameter_summary": "posterior draws from the signed RQ1 M1 model",
    "rq1_model_estimands": "opportunity-standardized projections from the signed RQ1 M1 model",
    "rq1_primary_estimand": "G1, G2, and G2-G1 pooled-opportunity standardized RQ1 probabilities",
    "rq1_model_diagnostics": "MCMC diagnostics for the main and prior-sensitivity RQ1 M1 fits",
    "rq1_prior_scale_sensitivity": "RQ1 primary estimand under original and doubled prior scales",
    "rq1_influence_participant": "participant-grouped PSIS influence diagnostics for RQ1 M1",
    "rq1_influence_account": "selected-account-grouped PSIS influence diagnostics for RQ1 M1",
    "rq1_influence_thread": "thread-grouped PSIS influence diagnostics for RQ1 M1",
    "rq1_influence_exact_refits": "signed exact M1 deletion refits for problematic grouped PSIS units",
    "rq1_influence_exact_estimands": "primary-estimand shifts from exact grouped deletion refits",
    "rq1_influence_exact_refits_validated": "validation-time authentication and recomputation of exact grouped deletion refits",
    "rq1_forward_validation_scores": "three-origin future-choice scores for M0 and M1",
    "rq1_forward_validation_comparisons": "paired three-origin M1 versus M0 future-choice scores",
    "rq1_forward_validation_predictions": "future choice-set predictions for RQ1 rolling origins",
    "rq1_forward_validation_diagnostics": "MCMC diagnostics for RQ1 rolling-origin fits",
    "rq1_publication_readiness": "fail-closed release gates for the final RQ1 M1 analysis",
    "rq1_result_report": "compact validated RQ1 result and provenance report",
    "rq2_event_history": "valid analysis-set platform events used for strict directional histories",
    "rq2_thread_choice_sets": "eligible earlier same-thread candidates for retained human replies",
    "rq2_network_history": "RQ2 choice candidates with directional state strictly before each reply",
    "rq2_history_support": "choice-by-AI-account support by arm, direction, and contact count class",
    "rq2_model_frame": "RQ2 candidates with general, AI-specific, and condition-specific history terms",
    "rq2_model_term_manifest": "RQ2 term definitions, training scalers, and inclusion decisions",
    "rq2_analysis_data_signature": "hash of the prepared RQ2 model arrays and implementation",
    "rq2_data_integrity_checks": "canonical RQ2 data and strict-history invariants",
    "rq2_model_parameter_summary": "posterior summary for the signed RQ2 core model",
    "rq2_primary_estimands": "directional zero-to-one-contact AI-account probability projections",
    "rq2_model_diagnostics": "MCMC diagnostics for the signed RQ2 core model",
    "rq2_influence_participant": "participant-grouped PSIS for the RQ2 core model",
    "rq2_influence_account": "selected-account-grouped PSIS for the RQ2 core model",
    "rq2_influence_thread": "thread-grouped PSIS for the RQ2 core model",
    "rq2_core_readiness": "fail-closed checks for the full-length RQ2 core fit",
    "rq2_result_report": "validated RQ2 core-fit estimates, support, and limitations",
}


PREPARE_TABLE_NAMES = frozenset(
    {
        "event_history",
        "thread_choice_sets",
        "network_history",
        "source_availability",
        "choice_set_attrition",
        "risk_set_composition",
        "actor_label_qc",
        "account_universe",
        "group_database_provenance",
        "browser_opportunity_episodes",
        "browser_linked_actions",
        "browser_manual_audit_sample",
        "browser_interval_audit_sample",
        "browser_choice_sets",
        "browser_strict_choice_sets",
        "tracking_linkage_qc",
        "browser_hazard_intervals",
        "browser_hazard_terms",
        "browser_standardized_conversion",
        "browser_conversion_summary",
        "model_frame",
        "model_term_manifest",
        "analysis_data_signature",
        "data_integrity_checks",
    }
)

VALIDATION_TABLE_NAMES = frozenset(
    {
        "publication_readiness",
        "fit_signature_authentication",
        "fit_result_bundle_authentication",
        "participant_loo_refit_authentication",
        "participant_grouped_psis_validated",
        "participant_grouped_reloo",
        "participant_grouped_reloo_summary",
        "forward_signature_expectations",
        "forward_validation_assignments",
        "observed_network_metrics",
        "event_timing_posterior_checks",
        "supplementary_topic_heterogeneity",
        "supplementary_source_inventory",
    }
)

RQ1_PREPARE_TABLE_NAMES = frozenset(
    {
        "event_history",
        "thread_choice_sets",
        "source_availability",
        "choice_set_attrition",
        "risk_set_composition",
        "actor_label_qc",
        "account_universe",
        "group_database_provenance",
        "model_frame",
        "model_term_manifest",
        "analysis_data_signature",
        "data_integrity_checks",
    }
)

RQ1_VALIDATION_TABLE_NAMES = frozenset(
    {
        "data_integrity_checks",
        "analysis_data_signature",
        "fit_result_bundle_authentication",
        "fit_signature_authentication",
        "rq1_influence_participant_validated",
        "rq1_influence_account_validated",
        "rq1_influence_thread_validated",
        "rq1_influence_exact_refits_validated",
        "rq1_publication_readiness",
    }
)

RQ1_REQUIRED_FIT_TABLES = frozenset(
    {
        "fit_run_signature",
        "rq1_model_parameter_summary",
        "rq1_model_estimands",
        "rq1_primary_estimand",
        "rq1_model_diagnostics",
        "rq1_prior_scale_sensitivity",
        "rq1_influence_participant",
        "rq1_influence_account",
        "rq1_influence_thread",
        "rq1_influence_exact_refits",
        "rq1_influence_exact_estimands",
        "rq1_forward_validation_scores",
        "rq1_forward_validation_comparisons",
        "rq1_forward_validation_predictions",
        "rq1_forward_validation_diagnostics",
    }
)

RQ2_PREPARE_TABLE_NAMES = frozenset(
    {
        "rq2_event_history",
        "rq2_thread_choice_sets",
        "rq2_network_history",
        "rq2_source_availability",
        "rq2_choice_set_attrition",
        "rq2_risk_set_composition",
        "rq2_actor_label_qc",
        "rq2_account_universe",
        "rq2_group_database_provenance",
        "rq2_history_support",
        "rq2_model_frame",
        "rq2_model_term_manifest",
        "rq2_analysis_data_signature",
        "rq2_data_integrity_checks",
    }
)

RQ2_REQUIRED_FIT_TABLES = frozenset(
    {
        "fit_run_signature",
        "rq2_model_parameter_summary",
        "rq2_primary_estimands",
        "rq2_model_diagnostics",
        "rq2_influence_participant",
        "rq2_influence_account",
        "rq2_influence_thread",
    }
)

RQ2_VALIDATION_TABLE_NAMES = frozenset(
    {
        "rq2_data_integrity_checks",
        "rq2_analysis_data_signature",
        "rq2_fit_result_bundle_authentication",
        "rq2_fit_signature_authentication",
        "rq2_influence_participant_validated",
        "rq2_influence_account_validated",
        "rq2_influence_thread_validated",
        "rq2_core_readiness",
    }
)

REQUIRED_RELEASE_FIT_TABLES = frozenset(
    {
        "fit_run_signature",
        "model_parameter_summary",
        "model_estimands",
        "primary_estimands",
        "model_diagnostics",
        "posterior_predictive_checks",
        "participant_grouped_psis",
        "participant_grouped_psis_all_models",
        "participant_loo_refits",
        "model_loo",
        "prior_scale_sensitivity",
        "forward_validation_scores",
        "forward_validation_comparisons",
        "forward_validation_predictions",
        "forward_validation_diagnostics",
        "latent_block_gate_status",
    }
)


def _directories(config: AnalysisConfig) -> None:
    ensure_directories(
        [
            config.run_dir,
            config.tables_dir,
            config.figures_dir,
            config.reports_dir,
            config.models_dir,
        ]
    )


def _artifact_metadata(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "table": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "denominator": TABLE_DENOMINATORS.get(name, "table-specific analysis units"),
    }
    if "group" in frame.columns:
        metadata["rows_by_group"] = {
            str(group): int(count)
            for group, count in frame.groupby("group", dropna=False).size().items()
        }
    if "choice_set_id" in frame.columns:
        metadata["choice_sets"] = int(frame["choice_set_id"].nunique())
    return metadata


def _write_tables(
    config: AnalysisConfig,
    tables: Mapping[str, pd.DataFrame],
) -> tuple[list[Path], dict[Path, Mapping[str, Any]]]:
    artifacts: list[Path] = []
    metadata: dict[Path, Mapping[str, Any]] = {}
    for name, frame in tables.items():
        paths = write_table(frame, config.tables_dir / name)
        for value in paths.values():
            path = Path(value)
            artifacts.append(path)
            metadata[path] = _artifact_metadata(name, frame)
    return artifacts, metadata


def _canonical_artifacts(config: AnalysisConfig) -> list[Path]:
    artifacts = {
        path
        for folder in [config.tables_dir, config.models_dir, config.figures_dir, config.reports_dir]
        if folder.exists()
        for path in folder.rglob("*")
        if path.is_file() and not path.name.endswith(".manifest.json")
    }
    gates = config.outputs_dir / "validation-gates.json"
    if gates.exists():
        artifacts.add(gates)
    return sorted(artifacts)


def _artifact_table_name(path: Path) -> str:
    return path.name.split(".", maxsplit=1)[0]


def _remove_directory_files(path: Path) -> None:
    if not path.exists():
        return
    for artifact in path.rglob("*"):
        if artifact.is_file():
            artifact.unlink(missing_ok=True)


def _invalidate_downstream_artifacts(config: AnalysisConfig, *, stage: str) -> None:
    """Remove every stale release surface before an upstream stage starts."""

    if stage not in {"prepare", "fit", "validate"}:
        raise ValueError(f"Unknown invalidation stage {stage!r}")
    _remove_directory_files(config.reports_dir)
    _remove_directory_files(config.figures_dir)
    if config.tables_dir.exists():
        for artifact in config.tables_dir.iterdir():
            if not artifact.is_file():
                continue
            table_name = _artifact_table_name(artifact)
            remove = (
                stage == "prepare"
                or (stage == "fit" and table_name not in PREPARE_TABLE_NAMES)
                or (stage == "validate" and table_name in VALIDATION_TABLE_NAMES)
            )
            if remove:
                artifact.unlink(missing_ok=True)
    (config.run_dir / "manifest.json").unlink(missing_ok=True)
    (config.outputs_dir / "validation-gates.json").unlink(missing_ok=True)
    if stage in {"prepare", "fit"}:
        (config.outputs_dir / BUNDLE_FILENAME).unlink(missing_ok=True)
        (config.run_dir / "status.json").unlink(missing_ok=True)


def _invalidate_rq1_artifacts(config: AnalysisConfig, *, stage: str) -> None:
    """Remove stale RQ1 release surfaces while preserving signed upstream stages."""

    if stage not in {"prepare", "fit", "validate"}:
        raise ValueError(f"Unknown RQ1 invalidation stage {stage!r}")
    _remove_directory_files(config.reports_dir)
    _remove_directory_files(config.figures_dir)
    if config.tables_dir.exists():
        for artifact in config.tables_dir.iterdir():
            if not artifact.is_file():
                continue
            table_name = _artifact_table_name(artifact)
            remove = (
                stage == "prepare"
                or (stage == "fit" and table_name not in RQ1_PREPARE_TABLE_NAMES)
                or (
                    stage == "validate"
                    and table_name in RQ1_VALIDATION_TABLE_NAMES
                )
            )
            if remove:
                artifact.unlink(missing_ok=True)
    (config.run_dir / "manifest.json").unlink(missing_ok=True)
    (config.outputs_dir / "validation-gates.json").unlink(missing_ok=True)
    if stage in {"prepare", "fit"}:
        (config.outputs_dir / BUNDLE_FILENAME).unlink(missing_ok=True)
        (config.run_dir / "status.json").unlink(missing_ok=True)


def _default_artifact_metadata(path: Path) -> Mapping[str, Any]:
    table_name = path.stem
    if path.suffix in {".csv", ".parquet"}:
        return {
            "table": table_name,
            "denominator": TABLE_DENOMINATORS.get(
                table_name, "table-specific analysis units"
            ),
        }
    if path.suffix == ".nc":
        return {
            "artifact": "Bayesian relational-event posterior",
            "denominator": "posterior-specific signed choice sets; see fit sidecar",
        }
    if path.suffix in {".png", ".pdf"}:
        return {
            "artifact": "publication figure",
            "denominator": "figure-specific canonical analysis units",
        }
    if path.suffix == ".md":
        return {
            "artifact": "publication report or audit",
            "denominator": "reported table-specific units",
        }
    return {
        "artifact": "publication validation artifact",
        "denominator": "artifact-specific analysis units",
    }


def _refresh_manifest(
    config: AnalysisConfig,
    config_path: Path,
    metadata: Mapping[Path, Mapping[str, Any]] | None = None,
) -> Path:
    artifacts = _canonical_artifacts(config)
    manifest = write_manifest(config, config_path, artifacts)
    if metadata:
        write_artifact_manifests(config, manifest, metadata)
    return manifest


def run_prepare_choice(
    config: AnalysisConfig,
    config_path: Path,
) -> dict[str, pd.DataFrame]:
    """Build the canonical RQ1 same-thread choice surface without browser stages."""

    if config.analysis_stage != "rq1_m1":
        raise ValueError("prepare-choice requires analysis_stage: rq1_m1")
    _directories(config)
    _invalidate_rq1_artifacts(config, stage="prepare")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    prepare_source_snapshot = capture_source_snapshot(source_paths(config, config_path))
    data_tables = prepare_choice_data(config)
    model_frame, term_manifest = build_model_frame(
        data_tables["thread_choice_sets"], model_name="M1"
    )
    from .model import prepare_model_data

    available_terms = included_terms_for_model("M1", term_manifest)
    numeric_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
    signature_data = prepare_model_data(
        model_frame,
        numeric_terms,
        ai_term="candidate_target_is_ai",
    )
    analysis_signature = pd.DataFrame(
        [
            {
                "analysis_stage": "rq1_m1",
                "model": "M1",
                "model_data_sha256": _prepared_data_hash(signature_data),
                "implementation_sha256": _model_implementation_hash(config),
                "choice_sets": signature_data.n_choices,
                "candidate_rows": int(signature_data.mask.sum()),
                "participants": len(signature_data.actor_labels),
                "terms": ",".join(signature_data.terms),
            }
        ]
    )
    expected_days = analysis_set_manifest(config.analysis_set)[
        "analysis_set_valid_days"
    ]
    integrity = data_integrity_checks(
        data_tables["event_history"],
        data_tables["thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=data_tables["actor_label_qc"],
        group_database_provenance=data_tables["group_database_provenance"],
    )
    tables = {
        **data_tables,
        "model_frame": model_frame,
        "model_term_manifest": term_manifest,
        "analysis_data_signature": analysis_signature,
        "data_integrity_checks": integrity,
    }
    assert_source_snapshot_unchanged(prepare_source_snapshot)
    artifacts, metadata = _write_tables(config, tables)
    assert_source_snapshot_unchanged(prepare_source_snapshot)
    manifest = write_manifest(config, config_path, artifacts)
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {
            "stage": "prepare-choice",
            "status": "completed",
            "run_id": config.run_id,
        },
        config.run_dir / "status.json",
    )
    return tables


def run_prepare(config: AnalysisConfig, config_path: Path) -> dict[str, pd.DataFrame]:
    """Build and persist all deterministic opportunity/history artifacts."""

    _directories(config)
    _invalidate_downstream_artifacts(config, stage="prepare")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    prepare_source_snapshot = capture_source_snapshot(source_paths(config, config_path))
    data_tables = prepare_data(config)
    browser_tables = prepare_browser(
        config,
        data_tables["event_history"],
        data_tables["thread_choice_sets"],
    )
    browser_tables["browser_strict_choice_sets"] = strict_browser_choice_sets(
        browser_tables["browser_choice_sets"]
    )
    intervals = build_piecewise_exposure(
        browser_tables["browser_opportunity_episodes"],
        browser_tables["browser_linked_actions"],
        source_availability=data_tables["source_availability"],
    )
    browser_tables["browser_hazard_intervals"] = intervals
    browser_tables["browser_hazard_terms"] = fit_piecewise_hazards(
        intervals,
        browser_tables["tracking_linkage_qc"],
    )
    standardized = standardized_piecewise_rates(
        intervals,
        browser_tables["tracking_linkage_qc"],
        random_seed=config.seed,
    )
    browser_tables["browser_standardized_conversion"] = standardized
    browser_tables["browser_conversion_summary"] = (
        browser_tables["browser_conversion_summary"]
        .merge(
            standardized,
            left_on=["group", "action_type", "target_author_type"],
            right_on=["group", "action_type", "author_type"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns="author_type")
    )
    browser_tables["browser_conversion_summary"] = mark_unobserved_action_channels(
        browser_tables["browser_conversion_summary"],
        data_tables["source_availability"],
    )
    model_frame, term_manifest = build_model_frame(data_tables["network_history"])
    from .model import prepare_model_data

    available_terms = included_terms_for_model("M3", term_manifest)
    signature_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
    signature_data = prepare_model_data(
        model_frame,
        signature_terms,
        ai_term="candidate_target_is_ai",
        closure_term="z_log_two_path_closure",
    )
    analysis_signature = pd.DataFrame(
        [
            {
                "model_data_sha256": _prepared_data_hash(signature_data),
                "implementation_sha256": _model_implementation_hash(config),
                "choice_sets": signature_data.n_choices,
                "candidate_rows": int(signature_data.mask.sum()),
                "participants": len(signature_data.actor_labels),
                "terms": ",".join(signature_data.terms),
            }
        ]
    )
    expected_days = analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"]
    integrity = data_integrity_checks(
        data_tables["event_history"],
        data_tables["thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=data_tables["actor_label_qc"],
        group_database_provenance=data_tables["group_database_provenance"],
    )
    tables = {
        **data_tables,
        **browser_tables,
        "model_frame": model_frame,
        "model_term_manifest": term_manifest,
        "analysis_data_signature": analysis_signature,
        "data_integrity_checks": integrity,
    }
    assert_source_snapshot_unchanged(prepare_source_snapshot)
    artifacts, metadata = _write_tables(config, tables)
    assert_source_snapshot_unchanged(prepare_source_snapshot)
    manifest = write_manifest(config, config_path, artifacts)
    write_artifact_manifests(config, manifest, metadata)
    return tables


def _read_required_tables(config: AnalysisConfig, names: Iterable[str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        tables[name] = read_table(config.tables_dir / name)
    return tables


def _read_optional_result(config: AnalysisConfig, name: str) -> pd.DataFrame:
    try:
        return read_table(config.tables_dir / name)
    except FileNotFoundError:
        return pd.DataFrame()


def _forward_signature_expectations(
    raw_model_frame: pd.DataFrame,
    *,
    model_names: Sequence[str] = ("M0", "M1", "M2", "M3"),
    reference_model: str = "M3",
) -> pd.DataFrame:
    """Rebuild declared rolling-origin training hashes from current choice rows."""

    from .model import prepare_model_data

    model_names = tuple(model_names)
    assignments = rolling_origin_assignments(raw_model_frame)
    rows: list[dict[str, Any]] = []
    for fold in sorted(assignments["fold"].unique()):
        split = assignments[assignments["fold"].eq(fold)]
        train_ids = set(split.loc[split["role"].eq("train"), "choice_set_id"])
        train_raw = raw_model_frame[raw_model_frame["choice_set_id"].isin(train_ids)]
        _, fold_manifest = build_model_frame(
            train_raw, model_name=reference_model
        )
        transformed = apply_model_term_manifest(raw_model_frame, fold_manifest)
        train_frame = transformed[transformed["choice_set_id"].isin(train_ids)]
        for model_name in model_names:
            available_terms = included_terms_for_model(model_name, fold_manifest)
            numeric_terms = tuple(
                term for term in available_terms if term != "candidate_target_is_ai"
            )
            training_data = prepare_model_data(
                train_frame,
                numeric_terms,
                ai_term="candidate_target_is_ai",
                closure_term=(
                    "z_log_two_path_closure"
                    if "z_log_two_path_closure" in numeric_terms
                    else None
                ),
            )
            rows.append(
                {
                    "fold": int(fold),
                    "role": f"forward_fold_{int(fold)}",
                    "model": model_name,
                    "expected_training_data_sha256": _prepared_data_hash(training_data),
                    "expected_choice_sets": training_data.n_choices,
                    "expected_candidate_rows": int(training_data.mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def _annotate_forward_signature_authentication(
    signatures: pd.DataFrame,
    expectations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach fresh fold-specific training-data checks to posterior signatures."""

    result = signatures.copy()
    result["expected_training_data_sha256"] = pd.NA
    result["forward_training_data_match"] = pd.NA
    if result.empty:
        return result
    lookup = expectations.set_index(["role", "model"])[
        "expected_training_data_sha256"
    ].to_dict()
    for index, record in result.iterrows():
        role = str(record.get("role", ""))
        if not role.startswith("forward_fold_"):
            continue
        expected = lookup.get((role, str(record.get("model", ""))))
        result.loc[index, "expected_training_data_sha256"] = expected
        result.loc[index, "forward_training_data_match"] = bool(
            expected is not None and str(record.get("data_sha256")) == str(expected)
        )
    return result


def _recompute_participant_psis(
    model_frame: pd.DataFrame,
    signatures: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute grouped influence diagnostics from the authenticated live M3."""

    selected = signatures[
        signatures.get("model", pd.Series(dtype=object)).eq("M3")
        & signatures.get("role", pd.Series(dtype=object)).eq("main_nested_sequence")
        & signatures.get(
            "live_posterior_authenticated", pd.Series(False, index=signatures.index)
        )
        .fillna(False)
        .astype(bool)
    ]
    if len(selected) != 1:
        return pd.DataFrame()
    import arviz as az

    idata = az.from_netcdf(Path(str(selected["posterior_path"].iloc[0])))
    try:
        result = participant_grouped_psis(idata, model_frame)
    finally:
        idata.close()
    result["model"] = "M3"
    result["posterior_sha256"] = str(selected["posterior_sha256"].iloc[0])
    result["recomputed_from_authenticated_posterior"] = True
    return result


def _write_rq1_result_report(
    config: AnalysisConfig,
    primary: pd.DataFrame,
    readiness: pd.DataFrame,
    signatures: pd.DataFrame,
) -> Path:
    report_path = config.reports_dir / "rq1_result_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    release_passed = bool(
        not readiness.empty
        and readiness["passed"].fillna(False).astype(bool).all()
    )
    columns = [
        "group",
        "condition",
        "estimate",
        "ci_low",
        "ci_high",
        "posterior_probability_gt_zero",
        "denominator",
    ]
    table = primary[columns].to_markdown(index=False) if not primary.empty else "No authenticated estimand."
    failed = readiness[~readiness["passed"].fillna(False).astype(bool)]
    failed_text = (
        "None."
        if failed.empty
        else "\n".join(
            f"- `{row.gate}`: {row.observed}"
            for row in failed.itertuples(index=False)
        )
    )
    main_signature = signatures[
        signatures.get("role", pd.Series(dtype=object)).eq("rq1_main")
    ]
    posterior_hash = (
        str(main_signature["posterior_sha256"].iloc[0])
        if len(main_signature) == 1
        else "unavailable"
    )
    text = f"""# RQ1 M1 validation report

- Run ID: `{config.run_id}`
- Release status: `{'passed' if release_passed else 'failed'}`
- Main posterior SHA-256: `{posterior_hash}`
- Analysis window: `{config.start_time}` to `{config.end_time or 'snapshot extraction'}`
- Opportunity set: earlier valid non-self-authored contributions in the same root thread
- Inference boundary: one community per condition; G2-G1 is an ecological, opportunity-standardized contrast

## Primary estimand

{table}

## Failed release gates

{failed_text}
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def run_validate_choice(
    config: AnalysisConfig,
    config_path: Path,
) -> dict[str, pd.DataFrame]:
    """Authenticate and apply fail-closed release gates to the final RQ1 run."""

    if config.analysis_stage != "rq1_m1":
        raise ValueError("validate-choice requires analysis_stage: rq1_m1")
    _directories(config)
    _invalidate_rq1_artifacts(config, stage="validate")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    validation_source_snapshot = capture_source_snapshot(
        source_paths(config, config_path)
    )
    tables = _read_required_tables(
        config,
        [
            "event_history",
            "thread_choice_sets",
            "model_frame",
            "model_term_manifest",
            "actor_label_qc",
            "group_database_provenance",
        ],
    )
    expected_days = analysis_set_manifest(config.analysis_set)[
        "analysis_set_valid_days"
    ]
    integrity = data_integrity_checks(
        tables["event_history"],
        tables["thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=tables["actor_label_qc"],
        group_database_provenance=tables["group_database_provenance"],
    )
    rebuilt_frame, rebuilt_manifest = build_model_frame(
        tables["thread_choice_sets"], model_name="M1"
    )
    from .model import prepare_model_data

    rebuilt_terms = included_terms_for_model("M1", rebuilt_manifest)
    rebuilt_data = prepare_model_data(
        rebuilt_frame,
        tuple(
            term
            for term in rebuilt_terms
            if term != "candidate_target_is_ai"
        ),
        ai_term="candidate_target_is_ai",
    )
    stored_terms = included_terms_for_model("M1", tables["model_term_manifest"])
    stored_data = prepare_model_data(
        tables["model_frame"],
        tuple(
            term
            for term in stored_terms
            if term != "candidate_target_is_ai"
        ),
        ai_term="candidate_target_is_ai",
    )
    rebuilt_hash = _prepared_data_hash(rebuilt_data)
    stored_hash = _prepared_data_hash(stored_data)
    integrity = pd.concat(
        [
            integrity,
            pd.DataFrame(
                [
                    {
                        "check": "rq1_model_frame_reproduces_same_thread_choices",
                        "passed": rebuilt_hash == stored_hash,
                        "observed": f"stored={stored_hash}; rebuilt={rebuilt_hash}",
                        "required": "identical prepared M1 data hashes",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    implementation_hash = _model_implementation_hash(config)
    analysis_signature = pd.DataFrame(
        [
            {
                "analysis_stage": "rq1_m1",
                "model": "M1",
                "model_data_sha256": rebuilt_hash,
                "implementation_sha256": implementation_hash,
                "choice_sets": rebuilt_data.n_choices,
                "candidate_rows": int(rebuilt_data.mask.sum()),
                "participants": len(rebuilt_data.actor_labels),
                "terms": ",".join(rebuilt_data.terms),
            }
        ]
    )
    bundle_authentication = authenticate_fit_result_bundle(
        config,
        current_model_data_sha256=rebuilt_hash,
        current_implementation_sha256=implementation_hash,
        required_tables=RQ1_REQUIRED_FIT_TABLES,
    )
    bundle_authenticated = bool(
        len(bundle_authentication) == 1
        and bundle_authentication["authenticated"].fillna(False).astype(bool).all()
    )

    def fitted_result(name: str) -> pd.DataFrame:
        return _read_optional_result(config, name) if bundle_authenticated else pd.DataFrame()

    signatures = authenticate_fit_signatures(
        _read_optional_result(config, "fit_run_signature")
    )
    forward_expectations = _forward_signature_expectations(
        tables["model_frame"],
        model_names=("M0", "M1"),
        reference_model="M1",
    )
    signatures = _annotate_forward_signature_authentication(
        signatures, forward_expectations
    )
    main_signature = signatures[
        signatures.get("model", pd.Series(dtype=object)).eq("M1")
        & signatures.get("role", pd.Series(dtype=object)).eq("rq1_main")
        & signatures.get(
            "live_posterior_authenticated", pd.Series(False, index=signatures.index)
        )
        .fillna(False)
        .astype(bool)
    ]
    validated_influence: dict[str, pd.DataFrame] = {}
    if bundle_authenticated and len(main_signature) == 1:
        import arviz as az

        idata = az.from_netcdf(Path(str(main_signature["posterior_path"].iloc[0])))
        try:
            validated_influence = {
                unit: grouped_choice_psis(
                    idata, tables["model_frame"], unit=unit
                )
                for unit in ("participant", "account", "thread")
            }
        finally:
            idata.close()
    else:
        validated_influence = {
            unit: pd.DataFrame() for unit in ("participant", "account", "thread")
        }
    if bundle_authenticated:
        validated_exact_refits = _validate_rq1_exact_influence_refits(
            config,
            tables["model_frame"],
            tables["model_term_manifest"],
            fitted_result("rq1_influence_exact_refits"),
            fitted_result("rq1_influence_exact_estimands"),
            signatures,
        )
    else:
        validated_exact_refits = pd.DataFrame()
    exact_refits_by_unit = {
        unit: validated_exact_refits[
            validated_exact_refits.get(
                "influence_unit",
                pd.Series(dtype=object, index=validated_exact_refits.index),
            )
            .astype(str)
            .eq(unit)
        ]
        for unit in ("participant", "account", "thread")
    }

    readiness = build_rq1_readiness(
        integrity=integrity,
        analysis_signature=analysis_signature,
        diagnostics=fitted_result("rq1_model_diagnostics"),
        signatures=signatures,
        primary_estimand=fitted_result("rq1_primary_estimand"),
        prior_sensitivity=fitted_result("rq1_prior_scale_sensitivity"),
        forward_scores=fitted_result("rq1_forward_validation_scores"),
        forward_diagnostics=fitted_result(
            "rq1_forward_validation_diagnostics"
        ),
        forward_comparisons=fitted_result(
            "rq1_forward_validation_comparisons"
        ),
        influence_tables=validated_influence,
        exact_influence_refits=exact_refits_by_unit,
        fit_bundle_authentication=bundle_authentication,
    )
    validation_tables = {
        "data_integrity_checks": integrity,
        "analysis_data_signature": analysis_signature,
        "fit_result_bundle_authentication": bundle_authentication,
        "fit_signature_authentication": signatures,
        "forward_signature_expectations": forward_expectations,
        "rq1_influence_participant_validated": validated_influence[
            "participant"
        ],
        "rq1_influence_account_validated": validated_influence["account"],
        "rq1_influence_thread_validated": validated_influence["thread"],
        "rq1_influence_exact_refits_validated": validated_exact_refits,
        "rq1_publication_readiness": readiness,
    }
    assert_source_snapshot_unchanged(validation_source_snapshot)
    artifacts, metadata = _write_tables(config, validation_tables)
    report_path = _write_rq1_result_report(
        config,
        fitted_result("rq1_primary_estimand"),
        readiness,
        signatures,
    )
    metadata[report_path] = {
        "artifact": "validated RQ1 result report",
        "denominator": (
            fitted_result("rq1_primary_estimand")["denominator"].iloc[0]
            if not fitted_result("rq1_primary_estimand").empty
            else "no authenticated estimand"
        ),
    }
    release_passed = bool(
        not readiness.empty
        and readiness["passed"].fillna(False).astype(bool).all()
    )
    write_json(
        {
            "analysis_stage": "rq1_m1",
            "release_passed": release_passed,
            "gates": readiness.to_dict("records"),
        },
        config.outputs_dir / "validation-gates.json",
    )
    write_json(
        {
            "stage": "validate-choice",
            "status": "completed" if release_passed else "failed_release_gates",
            "run_id": config.run_id,
        },
        config.run_dir / "status.json",
    )
    assert_source_snapshot_unchanged(validation_source_snapshot)
    manifest = _refresh_manifest(config, config_path, metadata)
    write_artifact_manifests(config, manifest, metadata)
    return validation_tables


def run_validate(config: AnalysisConfig, config_path: Path) -> dict[str, pd.DataFrame]:
    """Run deterministic, browser, network, and available fitted-model gates."""

    _directories(config)
    _invalidate_downstream_artifacts(config, stage="validate")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    validation_source_snapshot = capture_source_snapshot(
        source_paths(config, config_path)
    )
    tables = _read_required_tables(
        config,
        [
            "event_history",
            "thread_choice_sets",
            "network_history",
            "model_frame",
            "model_term_manifest",
            "actor_label_qc",
            "group_database_provenance",
        ],
    )
    expected_days = analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"]
    rebuilt_network = build_network_history(
        tables["thread_choice_sets"], tables["event_history"]
    )
    integrity = data_integrity_checks(
        tables["event_history"],
        tables["thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=tables["actor_label_qc"],
        group_database_provenance=tables["group_database_provenance"],
        network_history=tables["network_history"],
        recomputed_network_history=rebuilt_network,
    )
    rebuilt_frame, rebuilt_manifest = build_model_frame(rebuilt_network)
    rebuilt_terms = included_terms_for_model("M3", rebuilt_manifest)
    from .model import prepare_model_data

    rebuilt_data = prepare_model_data(
        rebuilt_frame,
        tuple(term for term in rebuilt_terms if term != "candidate_target_is_ai"),
        ai_term="candidate_target_is_ai",
        closure_term="z_log_two_path_closure",
    )
    stored_terms = included_terms_for_model("M3", tables["model_term_manifest"])
    stored_data = prepare_model_data(
        tables["model_frame"],
        tuple(term for term in stored_terms if term != "candidate_target_is_ai"),
        ai_term="candidate_target_is_ai",
        closure_term="z_log_two_path_closure",
    )
    rebuilt_hash = _prepared_data_hash(rebuilt_data)
    stored_hash = _prepared_data_hash(stored_data)
    integrity = pd.concat(
        [
            integrity,
            pd.DataFrame(
                [
                    {
                        "check": "model_frame_reproduces_pre_event_network_history",
                        "passed": rebuilt_hash == stored_hash,
                        "observed": f"stored={stored_hash}; rebuilt={rebuilt_hash}",
                        "required": "identical prepared model-data hashes",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    analysis_signature = pd.DataFrame(
        [
            {
                "model_data_sha256": rebuilt_hash,
                "implementation_sha256": _model_implementation_hash(config),
                "choice_sets": rebuilt_data.n_choices,
                "candidate_rows": int(rebuilt_data.mask.sum()),
                "participants": len(rebuilt_data.actor_labels),
                "terms": ",".join(rebuilt_data.terms),
            }
        ]
    )
    bundle_authentication = authenticate_fit_result_bundle(
        config,
        current_model_data_sha256=rebuilt_hash,
        current_implementation_sha256=_model_implementation_hash(config),
        required_tables=REQUIRED_RELEASE_FIT_TABLES,
    )
    bundle_authenticated = bool(
        len(bundle_authentication) == 1
        and bundle_authentication["authenticated"].fillna(False).astype(bool).all()
    )

    def fitted_result(name: str) -> pd.DataFrame:
        return _read_optional_result(config, name) if bundle_authenticated else pd.DataFrame()

    authenticated_signatures = authenticate_fit_signatures(
        _read_optional_result(config, "fit_run_signature")
    )
    forward_expectations = _forward_signature_expectations(tables["model_frame"])
    authenticated_signatures = _annotate_forward_signature_authentication(
        authenticated_signatures,
        forward_expectations,
    )
    participant_psis = _recompute_participant_psis(
        tables["model_frame"],
        authenticated_signatures,
    )
    participant_refits = fitted_result("participant_loo_refits")
    authenticated_refits = _validate_participant_refit_contributions(
        config,
        tables["model_frame"],
        participant_refits,
        authenticated_signatures,
    )
    grouped_reloo, grouped_reloo_summary = _participant_reloo_tables(
        participant_psis,
        authenticated_refits,
    )
    written, metadata = _write_tables(
        config,
        {
            "data_integrity_checks": integrity,
            "analysis_data_signature": analysis_signature,
            "fit_result_bundle_authentication": bundle_authentication,
            "fit_signature_authentication": authenticated_signatures,
            "forward_signature_expectations": forward_expectations,
            "participant_grouped_psis_validated": participant_psis,
            "participant_loo_refit_authentication": authenticated_refits,
            "participant_grouped_reloo": grouped_reloo,
            "participant_grouped_reloo_summary": grouped_reloo_summary,
        },
    )
    validation = run_validation(
        config,
        tables["thread_choice_sets"],
        tables["event_history"],
    )
    readiness = build_publication_readiness(
        integrity=integrity,
        analysis_signature=analysis_signature,
        diagnostics=fitted_result("model_diagnostics"),
        signatures=authenticated_signatures,
        prior_sensitivity=fitted_result("prior_scale_sensitivity"),
        forward_scores=fitted_result("forward_validation_scores"),
        forward_diagnostics=fitted_result("forward_validation_diagnostics"),
        forward_comparisons=fitted_result("forward_validation_comparisons"),
        participant_psis=participant_psis,
        participant_refits=authenticated_refits,
        participant_reloo_summary=grouped_reloo_summary,
        predictive_checks=fitted_result("posterior_predictive_checks"),
        event_timing_checks=validation["event_timing_posterior_checks"],
        tracking_qc=_read_optional_result(config, "tracking_linkage_qc"),
        latent_status=fitted_result("latent_block_gate_status"),
        fit_bundle_authentication=bundle_authentication,
    )
    # Keep the reusable research audit at the repository-level location required
    # by the network-audit workflow as well as inside the run directory.
    write_network_audit(
        config,
        tables["thread_choice_sets"],
        config.project_root / "research" / "reports" / "network-audit-2026-07-15.md",
    )
    for name, frame in validation.items():
        for suffix in (".csv", ".parquet"):
            artifact = config.tables_dir / f"{name}{suffix}"
            metadata[artifact] = _artifact_metadata(name, frame)
    gates = config.outputs_dir / "validation-gates.json"
    metadata[gates] = {
        "artifact": "prespecified validation thresholds",
        "denominator": "declared decision gates",
    }
    audit = config.reports_dir / "network-audit.md"
    metadata[audit] = {
        "artifact": "network object and opportunity audit",
        "denominator": "retained human reply choice sets",
    }
    assert_source_snapshot_unchanged(validation_source_snapshot)
    _refresh_manifest(config, config_path, metadata)
    try:
        readiness_paths, readiness_metadata = _write_tables(
            config, {"publication_readiness": readiness}
        )
        written.extend(readiness_paths)
        metadata.update(readiness_metadata)
        assert_source_snapshot_unchanged(validation_source_snapshot)
        _refresh_manifest(config, config_path, metadata)
    except Exception:
        for suffix in (".csv", ".parquet"):
            artifact = (config.tables_dir / "publication_readiness").with_suffix(suffix)
            artifact.unlink(missing_ok=True)
            artifact.with_suffix(artifact.suffix + ".manifest.json").unlink(
                missing_ok=True
            )
        (config.run_dir / "manifest.json").unlink(missing_ok=True)
        raise
    return {
        "data_integrity_checks": integrity,
        "publication_readiness": readiness,
        "fit_result_bundle_authentication": bundle_authentication,
        "fit_signature_authentication": authenticated_signatures,
        "participant_grouped_psis_validated": participant_psis,
        **validation,
    }


def run_report_stage(config: AnalysisConfig, config_path: Path) -> Path:
    """Render publication figures and the compact claim-bounded report."""

    _directories(config)
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    run_validate(config, config_path)
    try:
        path = run_report(config)
        report_metadata: dict[Path, Mapping[str, Any]] = {
            path: {
                "artifact": "publication report",
                "denominator": "reported table-specific units",
            }
        }
        for figure in config.figures_dir.glob("*"):
            if figure.is_file() and not figure.name.endswith(".manifest.json"):
                report_metadata[figure] = {
                    "artifact": "publication figure",
                    "denominator": "figure-specific analysis units",
                }
        for stem in (
            "supplementary_topic_heterogeneity",
            "supplementary_source_inventory",
        ):
            for suffix in (".csv", ".parquet"):
                artifact = config.tables_dir / f"{stem}{suffix}"
                if artifact.exists():
                    report_metadata[artifact] = _default_artifact_metadata(artifact)
        _refresh_manifest(
            config,
            config_path,
            report_metadata,
        )
        return path
    except Exception:
        _remove_directory_files(config.reports_dir)
        _remove_directory_files(config.figures_dir)
        for stem in (
            "supplementary_topic_heterogeneity",
            "supplementary_source_inventory",
        ):
            for suffix in (".csv", ".parquet"):
                artifact = config.tables_dir / f"{stem}{suffix}"
                artifact.unlink(missing_ok=True)
                artifact.with_suffix(artifact.suffix + ".manifest.json").unlink(
                    missing_ok=True
                )
        raise


def _gini_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if not len(counts) or counts.sum() == 0:
        return np.nan
    return float(np.abs(counts[:, None] - counts[None, :]).sum() / (2 * len(counts) * counts.sum()))


def _network_statistics(
    selected: pd.DataFrame,
    *,
    receiver_universe: Sequence[str] | None = None,
) -> dict[str, float]:
    if selected.empty:
        return {}
    dyads = selected.groupby(["actor_label", "receiver_label"]).size()
    repeat_events = float((dyads - 1).clip(lower=0).sum())
    actor_totals = selected.groupby("actor_label").size()
    actor_top = selected.groupby(["actor_label", "receiver_label"]).size().groupby(level=0).max()
    degree = selected["receiver_label"].value_counts()
    if receiver_universe is not None:
        degree = degree.reindex(list(receiver_universe), fill_value=0)
    degree_values = degree.to_numpy(dtype=float)
    reverse_column = (
        "simulated_prior_author_to_human_count"
        if "simulated_prior_author_to_human_count" in selected.columns
        else "prior_author_to_human_count"
    )
    closure_column = (
        "simulated_prior_two_path_count"
        if "simulated_prior_two_path_count" in selected.columns
        else "prior_two_path_count"
    )
    return {
        "h_to_ai_share": float(selected["candidate_target_is_ai"].astype(bool).mean()),
        "distinct_ai_alters": float(
            selected.loc[selected["candidate_target_is_ai"].astype(bool), "receiver_label"].nunique()
        ),
        "repeat_contact_share": repeat_events / len(selected),
        "top_alter_concentration": float((actor_top / actor_totals).mean()),
        "reciprocal_context_share": float(
            (pd.to_numeric(selected[reverse_column], errors="coerce") > 0).mean()
        ),
        "two_path_closure_mean": float(
            pd.to_numeric(selected[closure_column], errors="coerce").mean()
        ),
        "receiver_in_degree_gini": _gini_counts(degree_values),
        "receiver_in_degree_median": float(np.median(degree_values)),
        "receiver_in_degree_q90": float(np.quantile(degree_values, 0.90)),
        "receiver_in_degree_max": float(np.max(degree_values)),
        "selected_content_age_median_hours": float(
            pd.to_numeric(selected["candidate_age_hours"], errors="coerce").median()
        ),
    }


def _posterior_predictive_summary(
    model_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    account_universe: pd.DataFrame,
) -> pd.DataFrame:
    candidates = model_frame.copy()
    candidates["chosen_position"] = candidates.groupby("choice_set_id", sort=False).cumcount()
    candidates["actor_label"] = candidates["group"].astype(str) + "::" + candidates["actor_id"].astype(str)
    candidates["receiver_label"] = (
        candidates["group"].astype(str) + "::" + candidates["candidate_author_id"].astype(str)
    )
    columns = [
        "choice_set_id",
        "chosen_position",
        "group",
        "actor_label",
        "receiver_label",
        "candidate_target_is_ai",
        "prior_author_to_human_count",
        "prior_two_path_count",
        "candidate_age_hours",
        "chosen",
    ]
    observed_rows: list[dict[str, Any]] = []
    for group, part in candidates[candidates["chosen"].astype(bool)].groupby("group", sort=True):
        receiver_universe = tuple(
            str(group) + "::" + account_universe.loc[
                account_universe["group"].astype(str).eq(str(group)), "account_id"
            ].astype(str)
        )
        for metric, value in _network_statistics(
            part, receiver_universe=receiver_universe
        ).items():
            observed_rows.append({"group": group, "metric": metric, "observed": value})
    observed = pd.DataFrame(observed_rows)

    merged = predictions.merge(
        candidates[columns[:-1]],
        on=["choice_set_id", "chosen_position", "group"],
        how="left",
        validate="many_to_one",
    )
    draw_rows: list[dict[str, Any]] = []
    for (draw, group), part in merged.groupby(["posterior_draw_index", "group"], sort=False):
        receiver_universe = tuple(
            str(group) + "::" + account_universe.loc[
                account_universe["group"].astype(str).eq(str(group)), "account_id"
            ].astype(str)
        )
        for metric, value in _network_statistics(
            part, receiver_universe=receiver_universe
        ).items():
            draw_rows.append(
                {"posterior_draw_index": draw, "group": group, "metric": metric, "value": value}
            )
    draws = pd.DataFrame(draw_rows)
    posterior_draw_count = int(predictions["posterior_draw_index"].nunique())
    summaries: list[dict[str, Any]] = []
    scopes = predictions.get("simulation_scope", pd.Series(dtype=object)).dropna().unique()
    if len(scopes) != 1:
        raise ValueError(
            "Posterior network simulations must declare one unambiguous simulation scope"
        )
    simulation_scope = str(scopes[0])
    for (group, metric), part in draws.groupby(["group", "metric"], sort=True):
        values = pd.to_numeric(part["value"], errors="coerce").dropna().to_numpy(dtype=float)
        observed_value = float(
            observed.loc[observed["group"].eq(group) & observed["metric"].eq(metric), "observed"].iloc[0]
        )
        low = float(np.quantile(values, 0.025))
        high = float(np.quantile(values, 0.975))
        summaries.append(
            {
                "group": group,
                "metric": metric,
                "observed": observed_value,
                "posterior_median": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
                "covered": bool(low <= observed_value <= high),
                "posterior_draws": posterior_draw_count,
                "simulation_scope": simulation_scope,
            }
        )
    return pd.DataFrame(summaries)


def _fit_one_model(
    config: AnalysisConfig,
    data: Any,
    model_name: str,
    *,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    seed_offset: int = 0,
) -> Any:
    from .model import build_primary_pymc_model, fit_model

    model = build_primary_pymc_model(data, config, model_spec=model_name)
    return fit_model(
        model,
        config,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=config.seed + seed_offset,
    )


def _load_compatible_posterior(
    path: Path,
    *,
    draws: int,
    chains: int,
    request_signature: Mapping[str, Any],
) -> Any | None:
    """Load a signed posterior after verifying provenance, dimensions, and bytes.

    A posterior produced by the automatic diagnostic extension may contain more
    warm-up or retained draws than the current request.  Reusing that posterior
    is safe when every non-length request field matches and the stored file hash
    still matches the bytes on disk.
    """

    import arviz as az

    signature_path = path.with_suffix(path.suffix + ".fit.json")
    if not path.exists() or not signature_path.exists():
        return None
    try:
        saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(saved_signature, Mapping):
        return None
    try:
        saved_request = dict(saved_signature.get("request", {}))
        saved_actual = dict(saved_signature.get("actual", {}))
    except (TypeError, ValueError):
        return None
    requested = dict(request_signature)
    non_length_fields = set(saved_request) | set(requested)
    non_length_fields.difference_update({"draws", "tune"})
    request_compatible = all(
        saved_request.get(field) == requested.get(field)
        for field in non_length_fields
    )
    length_compatible = (
        int(saved_request.get("draws", -1)) >= int(requested.get("draws", draws))
        and int(saved_request.get("tune", -1)) >= int(requested.get("tune", 0))
        and int(saved_request.get("chains", -1)) == chains
    )
    recorded_hash = str(saved_actual.get("posterior_sha256", ""))
    if not request_compatible or not length_compatible or recorded_hash != sha256_file(path):
        return None
    candidate = az.from_netcdf(path)
    compatible = (
        candidate.posterior.sizes.get("draw") == int(saved_actual.get("draws", -1))
        and candidate.posterior.sizes.get("chain") == int(saved_actual.get("chains", -1))
        and candidate.posterior.sizes.get("draw") >= draws
        and candidate.posterior.sizes.get("chain") == chains
    )
    if compatible:
        return candidate
    candidate.close()
    return None


def _fit_with_diagnostic_extension(
    config: AnalysisConfig,
    data: Any,
    model_name: str,
    path: Path,
    *,
    role: str,
    draws: int,
    tune: int,
    chains: int,
    seed_offset: int,
    resume: bool,
) -> tuple[Any, dict[str, Any], dict[str, float | int | bool]]:
    """Fit or resume one model and make one longer run when gates fail.

    The extension doubles warm-up and retained draws while preserving the model,
    data, priors, chain count, and random seed.  A failed extended run is kept
    and reported as failed; repeated invocations do not grow sampling without a
    bound because an already-extended signed posterior is not extended again.
    """

    request = _posterior_request_signature(
        config,
        data,
        model=model_name,
        role=role,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=config.seed + seed_offset,
    )
    idata = None
    if resume:
        idata = _load_compatible_posterior(
            path,
            draws=draws,
            chains=chains,
            request_signature=request,
        )
    if idata is None:
        idata = _fit_one_model(
            config,
            data,
            model_name,
            draws=draws,
            tune=tune,
            chains=chains,
            seed_offset=seed_offset,
        )
        signature = _write_posterior_atomic(idata, path, request)
    else:
        signature = json.loads(
            path.with_suffix(path.suffix + ".fit.json").read_text(encoding="utf-8")
        )
    diagnostic = summarize_mcmc(idata)
    saved_request = dict(signature["request"])
    already_extended = (
        int(saved_request.get("draws", 0)) > draws
        or int(saved_request.get("tune", 0)) > tune
    )
    if diagnostic["passed"] or already_extended:
        diagnostic["sampling_extended"] = already_extended
        return idata, signature, diagnostic

    close_idata = getattr(idata, "close", None)
    if callable(close_idata):
        close_idata()
    extended_draws = 2 * draws
    extended_tune = 2 * tune
    extended_request = _posterior_request_signature(
        config,
        data,
        model=model_name,
        role=role,
        draws=extended_draws,
        tune=extended_tune,
        chains=chains,
        random_seed=config.seed + seed_offset,
    )
    idata = _fit_one_model(
        config,
        data,
        model_name,
        draws=extended_draws,
        tune=extended_tune,
        chains=chains,
        seed_offset=seed_offset,
    )
    signature = _write_posterior_atomic(idata, path, extended_request)
    signature["actual"]["sampling_extension_triggered"] = True
    signature["actual"]["base_draws"] = draws
    signature["actual"]["base_tune"] = tune
    write_json(signature, path.with_suffix(path.suffix + ".fit.json"))
    diagnostic = summarize_mcmc(idata)
    diagnostic["sampling_extended"] = True
    return idata, signature, diagnostic


def _write_posterior_atomic(
    idata: Any,
    path: Path,
    request_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a posterior beside its destination and replace it atomically."""

    temporary = path.with_name(f".{path.stem}.tmp.nc")
    temporary.unlink(missing_ok=True)
    try:
        idata.to_netcdf(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    signature = {
        "request": dict(request_signature),
        "actual": {
            "draws": int(idata.posterior.sizes.get("draw", 0)),
            "chains": int(idata.posterior.sizes.get("chain", 0)),
            "posterior_sha256": sha256_file(path),
            "posterior_size_bytes": path.stat().st_size,
        },
    }
    path.with_suffix(path.suffix + ".manifest.json").unlink(missing_ok=True)
    write_json(signature, path.with_suffix(path.suffix + ".fit.json"))
    return signature


def _model_implementation_hash(config: AnalysisConfig) -> str:
    """Hash every module that constructs, estimates, authenticates, or gates RQ1."""

    digest = hashlib.sha256()
    for name in (
        "data.py",
        "estimands.py",
        "model.py",
        "pipeline.py",
        "provenance.py",
        "rq2.py",
        "terms.py",
        "validation.py",
    ):
        path = config.project_root / "scripts" / "publication_analysis" / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prepared_data_hash(data: Any) -> str:
    digest = hashlib.sha256()
    arrays = (
        data.x,
        data.candidate_is_ai,
        data.mask,
        data.chosen_position,
        data.receiver_index,
        data.receiver_is_ai,
        data.actor_index,
        data.topic_index,
    )
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    labels = {
        "terms": list(data.terms),
        "term_families": list(data.term_families),
        "choice_ids": list(data.choice_ids),
        "receiver_labels": list(data.receiver_labels),
        "receiver_groups": list(data.receiver_groups),
        "actor_labels": list(data.actor_labels),
        "actor_groups": list(data.actor_groups),
        "topic_labels": list(data.topic_labels),
        "choice_groups": list(data.choice_groups),
    }
    digest.update(json.dumps(labels, sort_keys=True).encode())
    return digest.hexdigest()


def _posterior_request_signature(
    config: AnalysisConfig,
    data: Any,
    *,
    model: str,
    role: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "role": role,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": config.sampling.target_accept,
        "priors": asdict(config.sampling),
        "random_seed": random_seed,
        "data_sha256": _prepared_data_hash(data),
        "implementation_sha256": _model_implementation_hash(config),
    }


def _signature_row(
    path: Path,
    signature: Mapping[str, Any],
    data: Any | None = None,
) -> dict[str, Any]:
    request = dict(signature["request"])
    actual = dict(signature["actual"])
    row = {
        **request,
        "posterior_path": str(path),
        "actual_draws": actual["draws"],
        "actual_chains": actual["chains"],
        "posterior_sha256": actual["posterior_sha256"],
        "posterior_size_bytes": actual["posterior_size_bytes"],
        "dimension_match": (
            actual["draws"] == request["draws"]
            and actual["chains"] == request["chains"]
        ),
        "publication_length_request": (
            request["draws"] >= 1_000
            and request["tune"] >= 1_000
            and request["chains"] >= 4
        ),
    }
    if data is not None:
        row["denominator_choice_sets"] = int(data.n_choices)
        row["denominator_candidate_rows"] = int(data.mask.sum())
    return row


def _network_counterfactual_values(
    term_manifest: pd.DataFrame,
) -> Mapping[str, tuple[float, float]]:
    settings = term_manifest.set_index("term")
    raw_values = {
        "z_log_prior_contact_count": (0.0, float(np.log1p(1.0))),
        "z_recent_prior_contact": (0.0, float(np.exp(-1.0))),
        "z_log_reverse_contact_count": (0.0, float(np.log1p(1.0))),
        "z_recent_reverse_contact": (0.0, float(np.exp(-1.0))),
    }
    result: dict[str, tuple[float, float]] = {}
    for term, (raw_low, raw_high) in raw_values.items():
        if term not in settings.index or settings.loc[term, "status"] != "included_standardized":
            continue
        mean = float(settings.loc[term, "raw_mean"])
        standard_deviation = float(settings.loc[term, "raw_sd"])
        result[term] = (
            (raw_low - mean) / standard_deviation,
            (raw_high - mean) / standard_deviation,
        )
    return result


def _posterior_effect(idata: Any, variable: str, dimension: str) -> np.ndarray | None:
    if variable not in idata.posterior:
        return None
    return (
        idata.posterior[variable]
        .stack(sample=("chain", "draw"))
        .transpose("sample", dimension)
        .values
    )


def _posterior_scale(idata: Any, variable: str) -> np.ndarray | None:
    """Return one scalar hierarchical scale per posterior sample."""

    if variable not in idata.posterior:
        return None
    values = idata.posterior[variable].stack(sample=("chain", "draw"))
    remaining = [dimension for dimension in values.dims if dimension != "sample"]
    if remaining:
        raise ValueError(f"{variable} must be scalar within each posterior sample")
    result = values.transpose("sample").values
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError(f"{variable} must contain finite nonnegative scales")
    return result


def _receiver_scale(idata: Any) -> np.ndarray | None:
    """Return receiver scales ordered as human then AI for every sample."""

    variable = "sigma_receiver_effect"
    if variable not in idata.posterior:
        return None
    values = idata.posterior[variable].stack(sample=("chain", "draw"))
    if "author_type" not in values.dims or values.sizes["author_type"] != 2:
        raise ValueError("sigma_receiver_effect requires two author_type levels")
    author_types = [str(value).upper() for value in values.coords["author_type"].values]
    if set(author_types) == {"H", "AI"}:
        indexes = [author_types.index("H"), author_types.index("AI")]
    else:
        indexes = [0, 1]
    result = values.transpose("sample", "author_type").values[:, indexes]
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise ValueError("sigma_receiver_effect must contain finite nonnegative scales")
    return result


def _unknown_levels(
    training_labels: Sequence[str],
    test_labels: Sequence[str],
) -> np.ndarray:
    known = set(training_labels)
    return np.asarray([label not in known for label in test_labels], dtype=bool)


def _posterior_predictive_unseen_effects(
    idata: Any,
    training_data: Any,
    test_data: Any,
    *,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Draw one persistent effect for each new level and posterior sample."""

    sample_count = int(
        idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    )
    receiver_unknown = _unknown_levels(
        training_data.receiver_labels, test_data.receiver_labels
    )
    actor_unknown = _unknown_levels(training_data.actor_labels, test_data.actor_labels)
    topic_unknown = _unknown_levels(training_data.topic_labels, test_data.topic_labels)
    receiver_effect = np.zeros((sample_count, len(test_data.receiver_labels)))
    actor_effect = np.zeros((sample_count, len(test_data.actor_labels)))
    topic_effect = np.zeros((sample_count, len(test_data.topic_labels)))
    receiver_rng, actor_rng, topic_rng = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(random_seed).spawn(3)
    ]
    integrated_families: list[str] = []

    receiver_scale = _receiver_scale(idata)
    receiver_indexes = np.flatnonzero(receiver_unknown)
    if receiver_scale is not None and receiver_indexes.size:
        receiver_types = test_data.receiver_is_ai[receiver_indexes].astype(int)
        receiver_effect[:, receiver_indexes] = receiver_rng.normal(
            size=(sample_count, receiver_indexes.size)
        ) * receiver_scale[:, receiver_types]
        integrated_families.append("receiver")

    actor_scale = _posterior_scale(idata, "sigma_actor_ai_effect")
    actor_indexes = np.flatnonzero(actor_unknown)
    if actor_scale is not None and actor_indexes.size:
        actor_effect[:, actor_indexes] = actor_rng.normal(
            size=(sample_count, actor_indexes.size)
        ) * actor_scale[:, None]
        integrated_families.append("actor_ai_slope")

    topic_scale = _posterior_scale(idata, "sigma_topic_ai_effect")
    topic_indexes = np.flatnonzero(topic_unknown)
    if topic_scale is not None and topic_indexes.size:
        topic_effect[:, topic_indexes] = topic_rng.normal(
            size=(sample_count, topic_indexes.size)
        ) * topic_scale[:, None]
        integrated_families.append("topic_ai_slope")

    effects = {
        "receiver": receiver_effect,
        "actor": actor_effect,
        "topic": topic_effect,
    }
    metadata = {
        "unseen_hierarchical_effect_policy": (
            "one_reproducible_normal_draw_per_unseen_level_and_posterior_sample"
        ),
        "unseen_hierarchical_effect_random_seed": int(random_seed),
        "unseen_hierarchical_effect_families_integrated": (
            ",".join(integrated_families) if integrated_families else "none"
        ),
        "unseen_receiver_levels": int(receiver_unknown.sum()),
        "unseen_actor_levels": int(actor_unknown.sum()),
        "unseen_topic_levels": int(topic_unknown.sum()),
    }
    return effects, metadata


def _heldout_choice_utilities(
    idata: Any,
    training_data: Any,
    test_data: Any,
    *,
    unseen_receiver_effect: np.ndarray | None = None,
    unseen_actor_ai_effect: np.ndarray | None = None,
    unseen_topic_ai_effect: np.ndarray | None = None,
) -> np.ndarray:
    beta_da = idata.posterior["beta"]
    terms = tuple(str(value) for value in beta_da.coords["term"].values)
    term_indices = [test_data.terms.index(term) for term in terms]
    beta = beta_da.stack(sample=("chain", "draw")).transpose("sample", "term").values
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

    if unseen_receiver_effect is not None:
        new_receiver = np.asarray(unseen_receiver_effect, dtype=float)
        expected = (utility.shape[0], len(test_data.receiver_labels))
        if new_receiver.shape != expected:
            raise ValueError(f"Unseen-receiver effects must have shape {expected}")
        utility += new_receiver[:, test_data.receiver_index]

    if unseen_actor_ai_effect is not None:
        new_effect = np.asarray(unseen_actor_ai_effect, dtype=float)
        if new_effect.shape == (utility.shape[0],):
            unknown_actor = _unknown_levels(
                training_data.actor_labels, test_data.actor_labels
            )[test_data.actor_index]
            actor_contribution = new_effect[:, None] * unknown_actor[None, :]
        elif new_effect.shape == (utility.shape[0], len(test_data.actor_labels)):
            actor_contribution = new_effect[:, test_data.actor_index]
        else:
            raise ValueError(
                "Unseen-actor AI effects must provide one value per posterior sample "
                "or per posterior sample and test actor"
            )
        utility += (
            actor_contribution[:, :, None]
            * test_data.candidate_is_ai[None, :, :]
        )

    if unseen_topic_ai_effect is not None:
        new_topic = np.asarray(unseen_topic_ai_effect, dtype=float)
        expected = (utility.shape[0], len(test_data.topic_labels))
        if new_topic.shape != expected:
            raise ValueError(f"Unseen-topic AI effects must have shape {expected}")
        utility += (
            new_topic[:, test_data.topic_index, None]
            * test_data.candidate_is_ai[None, :, :]
        )

    return np.where(test_data.mask[None, :, :], utility, -np.inf)


def _heldout_prediction_values(
    utility: np.ndarray,
    test_data: Any,
) -> dict[str, np.ndarray]:
    row_max = np.max(utility, axis=2, keepdims=True)
    weights = np.exp(utility - row_max) * test_data.mask[None, :, :]
    probability = weights / weights.sum(axis=2, keepdims=True)
    mean_probability = probability.mean(axis=0)
    choice_index = np.arange(test_data.n_choices)
    chosen_probability = mean_probability[choice_index, test_data.chosen_position]
    chosen_ai = test_data.candidate_is_ai[choice_index, test_data.chosen_position]
    predicted_ai = np.sum(mean_probability * test_data.candidate_is_ai, axis=1)
    chosen_rank = 1 + np.asarray(
        [
            int(np.sum(row[test_data.mask[index]] > row[test_data.chosen_position[index]]))
            for index, row in enumerate(mean_probability)
        ]
    )
    return {
        "log_score": np.log(np.clip(chosen_probability, 1.0e-12, 1.0)),
        "selected_target_rank": chosen_rank,
        "predicted_ai_probability": predicted_ai,
        "chosen_is_ai": chosen_ai,
    }


def _heldout_choice_predictions(
    idata: Any,
    training_data: Any,
    test_data: Any,
    *,
    integrate_unseen_hierarchical: bool = False,
    random_seed: int | None = None,
    include_population_mean_sensitivity: bool = False,
) -> pd.DataFrame:
    population_mean_utility = _heldout_choice_utilities(
        idata, training_data, test_data
    )
    metadata: dict[str, Any] = {
        "unseen_hierarchical_effect_policy": "population_mean_zero",
        "unseen_hierarchical_effect_random_seed": pd.NA,
        "unseen_hierarchical_effect_families_integrated": "none",
        "unseen_receiver_levels": int(
            _unknown_levels(training_data.receiver_labels, test_data.receiver_labels).sum()
        ),
        "unseen_actor_levels": int(
            _unknown_levels(training_data.actor_labels, test_data.actor_labels).sum()
        ),
        "unseen_topic_levels": int(
            _unknown_levels(training_data.topic_labels, test_data.topic_labels).sum()
        ),
    }
    utility = population_mean_utility
    if integrate_unseen_hierarchical:
        if random_seed is None:
            raise ValueError(
                "random_seed is required for unseen hierarchical-effect integration"
            )
        effects, metadata = _posterior_predictive_unseen_effects(
            idata,
            training_data,
            test_data,
            random_seed=random_seed,
        )
        utility = _heldout_choice_utilities(
            idata,
            training_data,
            test_data,
            unseen_receiver_effect=effects["receiver"],
            unseen_actor_ai_effect=effects["actor"],
            unseen_topic_ai_effect=effects["topic"],
        )

    values = _heldout_prediction_values(utility, test_data)
    result = pd.DataFrame(
        {
            "choice_set_id": test_data.choice_ids,
            "group": test_data.choice_groups,
            "actor_label": np.asarray(test_data.actor_labels, dtype=object)[
                test_data.actor_index
            ],
            **values,
            **metadata,
            "prediction_scope": (
                "one_step_ahead_conditional_on_realized_earlier_heldout_history"
            ),
        }
    )
    if include_population_mean_sensitivity:
        sensitivity = _heldout_prediction_values(population_mean_utility, test_data)
        for name in (
            "log_score",
            "selected_target_rank",
            "predicted_ai_probability",
        ):
            result[f"{name}_population_mean_sensitivity"] = sensitivity[name]
        result["population_mean_sensitivity_policy"] = (
            "unseen_receiver_actor_and_topic_effects_fixed_at_zero"
        )
    return result


PARTICIPANT_REFIT_PRIMARY_QUADRATURE_NODES = 32
PARTICIPANT_REFIT_VALIDATION_QUADRATURE_NODES = 64
PARTICIPANT_REFIT_QUADRATURE_TOLERANCE = 1.0e-4


def _joint_participant_log_predictive_density(
    idata: Any,
    training_data: Any,
    test_data: Any,
    *,
    primary_quadrature_nodes: int = PARTICIPANT_REFIT_PRIMARY_QUADRATURE_NODES,
    validation_quadrature_nodes: int = (
        PARTICIPANT_REFIT_VALIDATION_QUADRATURE_NODES
    ),
    quadrature_tolerance: float = PARTICIPANT_REFIT_QUADRATURE_TOLERANCE,
) -> tuple[float, dict[str, Any]]:
    """Integrate one shared new-participant AI slope over all held-out choices.

    Gauss-Hermite quadrature integrates the participant slope separately within
    every posterior draw.  The returned ELPD uses the larger rule, while the
    smaller rule provides a deterministic node-doubling accuracy check.
    """

    from scipy.special import logsumexp

    if primary_quadrature_nodes < 2:
        raise ValueError(
            "Primary Gauss-Hermite quadrature requires at least two nodes"
        )
    if validation_quadrature_nodes <= primary_quadrature_nodes:
        raise ValueError(
            "Validation quadrature must use more nodes than the primary rule"
        )
    if not np.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0.0:
        raise ValueError("Quadrature tolerance must be a positive finite value")
    sample_count = int(
        idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    )
    if "sigma_actor_ai_effect" not in idata.posterior:
        scale = np.zeros(sample_count, dtype=float)
    else:
        scale = (
            idata.posterior["sigma_actor_ai_effect"]
            .stack(sample=("chain", "draw"))
            .transpose("sample")
            .values
        )
    if scale.shape != (sample_count,) or not np.isfinite(scale).all():
        raise ValueError(
            "Participant-scale posterior must be finite with one value per draw"
        )

    utility = _heldout_choice_utilities(idata, training_data, test_data)
    unknown_actor_levels = _unknown_levels(
        training_data.actor_labels,
        test_data.actor_labels,
    )
    if int(unknown_actor_levels.sum()) != 1:
        raise ValueError(
            "Participant deletion integration requires exactly one unseen actor"
        )
    unknown_actor = unknown_actor_levels[test_data.actor_index]
    if not unknown_actor.all():
        raise ValueError(
            "Participant deletion held-out choices must all belong to the unseen actor"
        )
    candidate_ai = test_data.candidate_is_ai.astype(bool)
    human_mask = test_data.mask & ~candidate_ai
    ai_mask = test_data.mask & candidate_ai
    human_log_mass = logsumexp(
        np.where(human_mask[None, :, :], utility, -np.inf),
        axis=2,
    )
    ai_log_mass = logsumexp(
        np.where(ai_mask[None, :, :], utility, -np.inf),
        axis=2,
    )
    choice_index = np.arange(test_data.n_choices)
    selected_utility = utility[:, choice_index, test_data.chosen_position]
    selected_ai = candidate_ai[
        choice_index, test_data.chosen_position
    ].astype(float)

    def integrated_elpd(node_count: int) -> float:
        nodes, weights = np.polynomial.hermite.hermgauss(node_count)
        invalid_quadrature = bool(
            not np.isfinite(nodes).all()
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        )
        if invalid_quadrature:
            raise ValueError(
                "Gauss-Hermite quadrature returned invalid nodes or weights"
            )
        node_log_probability = np.empty((node_count, sample_count), dtype=float)
        for node_index, node in enumerate(nodes):
            participant_effect = np.sqrt(2.0) * float(node) * scale
            choice_effect = participant_effect[:, None] * unknown_actor[None, :]
            log_denominator = np.logaddexp(
                human_log_mass,
                ai_log_mass + choice_effect,
            )
            selected = selected_utility + choice_effect * selected_ai[None, :]
            node_log_probability[node_index] = np.sum(
                selected - log_denominator,
                axis=1,
            )
        log_weight = np.log(weights) - 0.5 * np.log(np.pi)
        integrated_by_draw = logsumexp(
            node_log_probability + log_weight[:, None],
            axis=0,
        )
        return float(logsumexp(integrated_by_draw) - np.log(sample_count))

    primary_elpd = integrated_elpd(primary_quadrature_nodes)
    validation_elpd = integrated_elpd(validation_quadrature_nodes)
    absolute_difference = abs(validation_elpd - primary_elpd)
    diagnostics: dict[str, Any] = {
        "integration_method": "posterior_averaged_gauss_hermite_joint_probability",
        "posterior_draws": sample_count,
        "primary_quadrature_nodes": int(primary_quadrature_nodes),
        "validation_quadrature_nodes": int(validation_quadrature_nodes),
        "primary_quadrature_elpd": primary_elpd,
        "validation_quadrature_elpd": validation_elpd,
        "quadrature_absolute_difference": absolute_difference,
        "quadrature_tolerance": float(quadrature_tolerance),
        "quadrature_stability_pass": bool(
            absolute_difference <= quadrature_tolerance
        ),
    }
    return validation_elpd, diagnostics


def _participant_clustered_total_se(
    values: pd.Series,
    participants: pd.Series,
) -> tuple[float, int]:
    cluster_totals = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "participant": participants}
    ).groupby("participant", sort=False)["value"].sum()
    if len(cluster_totals) < 2:
        return np.nan, int(len(cluster_totals))
    return float(np.sqrt(len(cluster_totals) * cluster_totals.var(ddof=1))), int(
        len(cluster_totals)
    )


def run_forward_validation_models(
    config: AnalysisConfig,
    raw_model_frame: pd.DataFrame,
    term_manifest: pd.DataFrame,
    *,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    resume: bool,
    model_names: Sequence[str] = ("M0", "M1", "M2", "M3"),
    reference_model: str = "M3",
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Compare declared models on identical future choice sets at three origins."""

    from .model import prepare_model_data

    model_names = tuple(model_names)
    if reference_model not in model_names:
        raise ValueError("Forward-validation reference model must be fitted")
    required_manifest_terms = set(
        included_terms_for_model(reference_model, term_manifest)
    )
    if not required_manifest_terms:
        raise ValueError(
            f"Forward validation requires included {reference_model} terms"
        )
    assignments = rolling_origin_assignments(raw_model_frame)
    score_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    signature_rows: list[dict[str, Any]] = []
    for fold in sorted(assignments["fold"].unique()):
        split = assignments[assignments["fold"].eq(fold)]
        train_ids = set(split.loc[split["role"].eq("train"), "choice_set_id"])
        test_ids = set(split.loc[split["role"].eq("test"), "choice_set_id"])
        train_raw = raw_model_frame[raw_model_frame["choice_set_id"].isin(train_ids)]
        _, fold_manifest = build_model_frame(
            train_raw, model_name=reference_model
        )
        transformed = apply_model_term_manifest(raw_model_frame, fold_manifest)
        train_frame = transformed[transformed["choice_set_id"].isin(train_ids)]
        test_frame = transformed[transformed["choice_set_id"].isin(test_ids)]
        train_counts = (
            split.loc[split["role"].eq("train")].groupby("group")["choice_set_id"].nunique()
        )
        for model_index, model_name in enumerate(model_names):
            available_terms = included_terms_for_model(model_name, fold_manifest)
            numeric_terms = tuple(
                term for term in available_terms if term != "candidate_target_is_ai"
            )
            train_data = prepare_model_data(
                train_frame,
                numeric_terms,
                ai_term="candidate_target_is_ai",
                closure_term=(
                    "z_log_two_path_closure"
                    if "z_log_two_path_closure" in numeric_terms
                    else None
                ),
            )
            test_data = prepare_model_data(
                test_frame,
                numeric_terms,
                ai_term="candidate_target_is_ai",
                closure_term=(
                    "z_log_two_path_closure"
                    if "z_log_two_path_closure" in numeric_terms
                    else None
                ),
            )
            seed_offset = 300 + 10 * int(fold) + model_index
            path = config.models_dir / f"{model_name}_forward_fold{fold}.nc"
            idata, signature, diagnostic = _fit_with_diagnostic_extension(
                config,
                train_data,
                model_name,
                path,
                role=f"forward_fold_{int(fold)}",
                draws=draws or config.sampling.draws,
                tune=tune or config.sampling.tune,
                chains=chains or config.sampling.chains,
                seed_offset=seed_offset,
                resume=resume,
            )
            signature_rows.append(_signature_row(path, signature, train_data))
            diagnostic_rows.append(
                {"fold": int(fold), "model": model_name, **diagnostic}
            )
            prediction_seed = config.seed + 30_000 + 100 * int(fold) + model_index
            predictions = _heldout_choice_predictions(
                idata,
                train_data,
                test_data,
                integrate_unseen_hierarchical=True,
                random_seed=prediction_seed,
                include_population_mean_sensitivity=True,
            )
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "fold", int(fold))
            prediction_tables.append(predictions)
            grouped = list(predictions.groupby("group", sort=True)) + [("ALL", predictions)]
            for group, part in grouped:
                train_count = (
                    int(train_counts.get(group, 0)) if group != "ALL" else int(len(train_ids))
                )
                error = (
                    part["predicted_ai_probability"].astype(float)
                    - part["chosen_is_ai"].astype(float)
                )
                population_mean_error = (
                    part[
                        "predicted_ai_probability_population_mean_sensitivity"
                    ].astype(float)
                    - part["chosen_is_ai"].astype(float)
                )
                score_rows.append(
                    {
                        "fold": int(fold),
                        "model": model_name,
                        "group": group,
                        "train_choice_sets": train_count,
                        "test_choice_sets": int(len(part)),
                        "mean_future_log_score": float(part["log_score"].mean()),
                        "median_selected_target_rank": float(
                            part["selected_target_rank"].median()
                        ),
                        "mean_selected_target_rank": float(
                            part["selected_target_rank"].mean()
                        ),
                        "ai_target_brier_score": float(np.mean(error**2)),
                        "ai_calibration_difference": float(-error.mean()),
                        "mean_future_log_score_population_mean_sensitivity": float(
                            part["log_score_population_mean_sensitivity"].mean()
                        ),
                        "ai_target_brier_score_population_mean_sensitivity": float(
                            np.mean(population_mean_error**2)
                        ),
                        "mean_log_score_integration_minus_population_mean": float(
                            (
                                part["log_score"]
                                - part["log_score_population_mean_sensitivity"]
                            ).mean()
                        ),
                        "unseen_hierarchical_effect_policy": str(
                            part["unseen_hierarchical_effect_policy"].iloc[0]
                        ),
                        "unseen_hierarchical_effect_random_seed": int(
                            part["unseen_hierarchical_effect_random_seed"].iloc[0]
                        ),
                        "unseen_hierarchical_effect_families_integrated": str(
                            part[
                                "unseen_hierarchical_effect_families_integrated"
                            ].iloc[0]
                        ),
                        "population_mean_sensitivity_policy": str(
                            part["population_mean_sensitivity_policy"].iloc[0]
                        ),
                        "scaling_scope": "training_events_only",
                        "prediction_scope": (
                            "one_step_ahead_conditional_on_realized_earlier_"
                            "heldout_history"
                        ),
                    }
                )

    event_predictions = pd.concat(prediction_tables, ignore_index=True)
    comparison_rows: list[dict[str, Any]] = []
    for fold_value in (*sorted(event_predictions["fold"].unique()), "ALL"):
        fold_predictions = (
            event_predictions
            if fold_value == "ALL"
            else event_predictions[event_predictions["fold"].eq(fold_value)]
        )
        reference = fold_predictions[
            fold_predictions["model"].eq(reference_model)
        ]
        for comparator in (
            model for model in model_names if model != reference_model
        ):
            baseline = fold_predictions[fold_predictions["model"].eq(comparator)]
            paired = reference.merge(
                baseline,
                on=[
                    "fold",
                    "choice_set_id",
                    "group",
                    "actor_label",
                    "chosen_is_ai",
                ],
                suffixes=("_reference", "_comparator"),
                validate="one_to_one",
            )
            delta = (
                paired["log_score_reference"] - paired["log_score_comparator"]
            )
            difference = float(delta.sum())
            standard_error, participant_clusters = _participant_clustered_total_se(
                delta, paired["actor_label"]
            )
            population_mean_delta = (
                paired["log_score_population_mean_sensitivity_reference"]
                - paired["log_score_population_mean_sensitivity_comparator"]
            )
            population_mean_difference = float(population_mean_delta.sum())
            population_mean_se, _ = _participant_clustered_total_se(
                population_mean_delta, paired["actor_label"]
            )
            reference_brier = (
                paired["predicted_ai_probability_reference"]
                - paired["chosen_is_ai"].astype(float)
            ) ** 2
            comparator_brier = (
                paired["predicted_ai_probability_comparator"]
                - paired["chosen_is_ai"].astype(float)
            ) ** 2
            comparison_rows.append(
                {
                    "fold": (
                        "ALL"
                        if fold_value == "ALL"
                        else str(int(fold_value))
                    ),
                    "reference_model": reference_model,
                    "comparator_model": comparator,
                    "test_choice_sets": int(len(paired)),
                    "elpd_difference": difference,
                    "elpd_difference_se": standard_error,
                    "elpd_difference_population_mean_sensitivity": (
                        population_mean_difference
                    ),
                    "elpd_difference_population_mean_sensitivity_se": (
                        population_mean_se
                    ),
                    "uncertainty_clusters": participant_clusters,
                    "uncertainty_unit": "participant_within_arm",
                    "gain_at_least_two_se": bool(difference >= 2.0 * standard_error),
                    "mean_rank_improvement": float(
                        (
                            paired["selected_target_rank_comparator"]
                            - paired["selected_target_rank_reference"]
                        ).mean()
                    ),
                    "mean_brier_improvement": float(
                        (comparator_brier - reference_brier).mean()
                    ),
                    "comparison_scope": (
                        "one_step_ahead_conditional_on_realized_earlier_heldout_history"
                    ),
                    "unseen_hierarchical_effect_policy": (
                        "one_reproducible_normal_draw_per_unseen_level_and_"
                        "posterior_sample"
                    ),
                    "population_mean_sensitivity_included": True,
                }
            )
    return (
        {
            "forward_validation_scores": pd.DataFrame(score_rows),
            "forward_validation_comparisons": pd.DataFrame(comparison_rows),
            "forward_validation_predictions": event_predictions,
            "forward_validation_diagnostics": pd.DataFrame(diagnostic_rows),
        },
        signature_rows,
    )


def _participant_deletion_data(
    raw_model_frame: pd.DataFrame,
    participant_label: str,
) -> tuple[Any, Any]:
    """Rebuild training-scaled data for one participant deletion."""

    from .model import prepare_model_data

    labels = (
        raw_model_frame["group"].astype(str)
        + "::"
        + raw_model_frame["actor_id"].astype(str)
    )
    test_raw = raw_model_frame[labels.eq(participant_label)]
    train_raw = raw_model_frame[~labels.eq(participant_label)]
    if test_raw.empty:
        raise ValueError(f"No choice rows match flagged participant {participant_label}")
    _, fold_manifest = build_model_frame(train_raw)
    transformed = apply_model_term_manifest(raw_model_frame, fold_manifest)
    test_frame = transformed.loc[test_raw.index]
    train_frame = transformed.loc[train_raw.index]
    available_terms = included_terms_for_model("M3", fold_manifest)
    numeric_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
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
    return train_data, test_data


def run_participant_loo_refits(
    config: AnalysisConfig,
    raw_model_frame: pd.DataFrame,
    flagged: pd.DataFrame,
    *,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    resume: bool,
) -> pd.DataFrame:
    """Refit M3 after removing every participant with problematic PSIS diagnostics."""

    import arviz as az

    problematic = flagged[flagged["needs_refit"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for index, participant in problematic.reset_index(drop=True).iterrows():
        participant_label = f"{participant['group']}::{participant['actor_id']}"
        train_data, test_data = _participant_deletion_data(
            raw_model_frame,
            participant_label,
        )
        safe_label = participant_label.replace("::", "_")
        path = config.models_dir / f"M3_participant_refit_{safe_label}.nc"
        idata, signature, diagnostic = _fit_with_diagnostic_extension(
            config,
            train_data,
            "M3",
            path,
            role=f"participant_deletion_{participant_label}",
            draws=draws or config.sampling.draws,
            tune=tune or config.sampling.tune,
            chains=chains or config.sampling.chains,
            seed_offset=500 + index,
            resume=resume,
        )
        predictions = _heldout_choice_predictions(idata, train_data, test_data)
        exact_refit_elpd, integration = _joint_participant_log_predictive_density(
            idata,
            train_data,
            test_data,
        )
        numerically_stable = bool(integration["quadrature_stability_pass"])
        rows.append(
            {
                "group": participant["group"],
                "actor_id": participant["actor_id"],
                "choice_sets": int(len(predictions)),
                "psis_pareto_k": float(participant["pareto_k"]),
                "exact_refit_elpd": exact_refit_elpd,
                "exact_refit_method": integration["integration_method"],
                "predictive_draws": integration["posterior_draws"],
                "primary_quadrature_nodes": integration[
                    "primary_quadrature_nodes"
                ],
                "validation_quadrature_nodes": integration[
                    "validation_quadrature_nodes"
                ],
                "primary_quadrature_elpd": integration["primary_quadrature_elpd"],
                "validation_quadrature_elpd": integration[
                    "validation_quadrature_elpd"
                ],
                "quadrature_absolute_difference": integration[
                    "quadrature_absolute_difference"
                ],
                "quadrature_tolerance": integration["quadrature_tolerance"],
                "quadrature_stability_pass": numerically_stable,
                "mean_log_score": float(predictions["log_score"].mean()),
                "median_selected_target_rank": float(
                    predictions["selected_target_rank"].median()
                ),
                "ai_target_brier_score": float(
                    np.mean(
                        (
                            predictions["predicted_ai_probability"].astype(float)
                            - predictions["chosen_is_ai"].astype(float)
                        )
                        ** 2
                    )
                ),
                "refit_status": (
                    "completed_exact_participant_deletion"
                    if numerically_stable
                    else "failed_quadrature_stability"
                ),
                "unseen_actor_policy": (
                    "one_shared_new_actor_AI_slope_integrated_by_gauss_hermite"
                ),
                "history_policy": (
                    "conditional_on_realized_pre_event_network_history_including_"
                    "earlier_actions_by_deleted_participant"
                ),
                "mcmc_passed": bool(diagnostic["passed"]),
                "sampling_extended": bool(diagnostic["sampling_extended"]),
                "posterior_path": str(path.resolve()),
                "posterior_sha256": str(signature["actual"]["posterior_sha256"]),
                "training_data_sha256": str(signature["request"]["data_sha256"]),
                "implementation_sha256": str(
                    signature["request"]["implementation_sha256"]
                ),
                "signature_role": str(signature["request"]["role"]),
            }
        )
    return pd.DataFrame(rows)


def _participant_reloo_tables(
    participant_psis: pd.DataFrame,
    participant_refits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace problematic grouped PSIS contributions with exact conditional refits."""

    if participant_psis.empty:
        return (
            pd.DataFrame(
                columns=[
                    "group",
                    "actor_id",
                    "grouped_elpd",
                    "elpd_source",
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "participants": 0,
                        "flagged_participants": 0,
                        "exact_refits": 0,
                        "grouped_reloo_elpd": np.nan,
                        "participant_cluster_se": np.nan,
                        "all_problematic_contributions_replaced": False,
                        "history_policy": (
                            "conditional_on_realized_pre_event_network_history; "
                            "influence_diagnostic_not_new_network_generalization"
                        ),
                    }
                ]
            ),
        )
    contributions = participant_psis.copy()
    contributions["_key"] = (
        contributions["group"].astype(str) + "::" + contributions["actor_id"].astype(str)
    )
    refit_lookup: dict[str, float] = {}
    if not participant_refits.empty and {
        "group",
        "actor_id",
        "exact_refit_elpd",
    }.issubset(participant_refits.columns):
        refit_candidates = participant_refits.copy()
        refit_candidates["_key"] = (
            refit_candidates["group"].astype(str)
            + "::"
            + refit_candidates["actor_id"].astype(str)
        )
        refit_candidates["_finite"] = np.isfinite(
            pd.to_numeric(refit_candidates["exact_refit_elpd"], errors="coerce")
        )
        refit_candidates["_unique"] = ~refit_candidates["_key"].duplicated(
            keep=False
        )
        if "authenticated_refit_contribution" in refit_candidates.columns:
            refit_candidates["_authenticated"] = (
                refit_candidates["authenticated_refit_contribution"]
                .fillna(False)
                .astype(bool)
            )
        else:
            refit_candidates["_authenticated"] = True
        if "quadrature_stability_pass" in refit_candidates.columns:
            refit_candidates["_numerically_stable"] = (
                refit_candidates["quadrature_stability_pass"]
                .fillna(False)
                .astype(bool)
            )
        else:
            refit_candidates["_numerically_stable"] = False
        refit_candidates = refit_candidates[
            refit_candidates["_finite"]
            & refit_candidates["_unique"]
            & refit_candidates["_authenticated"]
            & refit_candidates["_numerically_stable"]
        ]
        refit_lookup = {
            str(key): float(value)
            for key, value in zip(
                refit_candidates["_key"], refit_candidates["exact_refit_elpd"]
            )
        }
    contributions["grouped_elpd"] = contributions["elpd_loo"].astype(float)
    contributions["elpd_source"] = "PSIS"
    flagged = contributions["needs_refit"].fillna(False).astype(bool)
    for index in contributions.index[flagged]:
        key = contributions.at[index, "_key"]
        if key in refit_lookup:
            contributions.at[index, "grouped_elpd"] = refit_lookup[key]
            contributions.at[index, "elpd_source"] = "exact_conditional_history_refit"
        else:
            contributions.at[index, "elpd_source"] = "missing_required_exact_refit"
    all_replaced = bool(
        contributions.loc[flagged, "elpd_source"]
        .eq("exact_conditional_history_refit")
        .all()
        and np.isfinite(
            pd.to_numeric(
                contributions.loc[flagged, "grouped_elpd"], errors="coerce"
            )
        ).all()
    )
    values = contributions["grouped_elpd"].to_numpy(dtype=float)
    standard_error = (
        float(np.sqrt(len(values) * np.var(values, ddof=1)))
        if len(values) > 1
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "participants": int(len(contributions)),
                "flagged_participants": int(flagged.sum()),
                "exact_refits": int(
                    contributions["elpd_source"]
                    .eq("exact_conditional_history_refit")
                    .sum()
                ),
                "grouped_reloo_elpd": float(values.sum()),
                "participant_cluster_se": standard_error,
                "all_problematic_contributions_replaced": all_replaced,
                "history_policy": (
                    "conditional_on_realized_pre_event_network_history; influence_"
                    "diagnostic_not_new-network_generalization"
                ),
            }
        ]
    )
    return contributions.drop(columns="_key"), summary


def _validate_participant_refit_contributions(
    config: AnalysisConfig,
    raw_model_frame: pd.DataFrame,
    participant_refits: pd.DataFrame,
    authenticated_signatures: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild and authenticate exact grouped contributions from signed posteriors."""

    columns = [
        *participant_refits.columns,
        "recomputed_training_data_sha256",
        "recomputed_exact_refit_elpd",
        "recomputed_primary_quadrature_elpd",
        "recomputed_validation_quadrature_elpd",
        "recomputed_quadrature_absolute_difference",
        "recomputed_quadrature_stability_pass",
        "recomputed_mcmc_max_rhat",
        "recomputed_mcmc_min_ess_bulk",
        "recomputed_mcmc_min_ess_tail",
        "recomputed_mcmc_divergences",
        "recomputed_mcmc_passed",
        "refit_contribution_validation_status",
        "authenticated_refit_contribution",
    ]
    if participant_refits.empty:
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))

    import arviz as az

    refits = participant_refits.copy()
    keys = refits["group"].astype(str) + "::" + refits["actor_id"].astype(str)
    duplicated = keys.duplicated(keep=False)
    implementation_hash = _model_implementation_hash(config)
    rows: list[dict[str, Any]] = []
    for position, (index, record) in enumerate(refits.iterrows()):
        key = str(keys.loc[index])
        status: list[str] = []
        recomputed_hash: str | None = None
        recomputed_elpd = np.nan
        recomputed_integration: dict[str, Any] = {}
        recomputed_mcmc: dict[str, float | int | bool] = {}
        if bool(duplicated.loc[index]):
            status.append("duplicate_participant_refit_row")
        if record.get("exact_refit_method") != (
            "posterior_averaged_gauss_hermite_joint_probability"
        ):
            status.append("unsupported_exact_refit_method")
        if record.get("unseen_actor_policy") != (
            "one_shared_new_actor_AI_slope_integrated_by_gauss_hermite"
        ):
            status.append("unsupported_unseen_actor_policy")
        quadrature_values = pd.to_numeric(
            pd.Series(
                {
                    "primary_nodes": record.get("primary_quadrature_nodes"),
                    "validation_nodes": record.get("validation_quadrature_nodes"),
                    "tolerance": record.get("quadrature_tolerance"),
                    "predictive_draws": record.get("predictive_draws"),
                }
            ),
            errors="coerce",
        )
        quadrature_protocol_pass = bool(
            np.isfinite(quadrature_values).all()
            and quadrature_values["primary_nodes"]
            == PARTICIPANT_REFIT_PRIMARY_QUADRATURE_NODES
            and quadrature_values["validation_nodes"]
            == PARTICIPANT_REFIT_VALIDATION_QUADRATURE_NODES
            and np.isclose(
                quadrature_values["tolerance"],
                PARTICIPANT_REFIT_QUADRATURE_TOLERANCE,
                rtol=0.0,
                atol=np.finfo(float).eps,
            )
        )
        if not quadrature_protocol_pass:
            status.append("unsupported_quadrature_protocol")
        saved_stability = str(record.get("quadrature_stability_pass")).strip().lower()
        if saved_stability not in {"true", "1"}:
            status.append("saved_quadrature_stability_failed")
        role = f"participant_deletion_{key}"
        signed = authenticated_signatures[
            authenticated_signatures.get("role", pd.Series(dtype=object))
            .astype(str)
            .eq(role)
        ]
        if len(signed) != 1:
            status.append("missing_or_duplicate_signed_posterior")
        elif not bool(signed["live_posterior_authenticated"].iloc[0]):
            status.append("posterior_authentication_failed")
        try:
            train_data, test_data = _participant_deletion_data(
                raw_model_frame,
                key,
            )
            recomputed_hash = _prepared_data_hash(train_data)
        except (KeyError, TypeError, ValueError) as error:
            status.append(f"deletion_data_rebuild_failed:{type(error).__name__}")
            train_data = None
            test_data = None
        if len(signed) == 1 and recomputed_hash is not None:
            signature = signed.iloc[0]
            if str(record.get("signature_role")) != role:
                status.append("refit_table_signature_role_mismatch")
            if str(signature.get("data_sha256")) != recomputed_hash:
                status.append("signed_training_data_hash_mismatch")
            if str(signature.get("implementation_sha256")) != implementation_hash:
                status.append("signed_implementation_hash_mismatch")
            if str(record.get("training_data_sha256")) != recomputed_hash:
                status.append("refit_table_training_data_hash_mismatch")
            if str(record.get("posterior_path")) != str(signature.get("posterior_path")):
                status.append("refit_table_posterior_path_mismatch")
            if str(record.get("posterior_sha256")) != str(
                signature.get("posterior_sha256")
            ):
                status.append("refit_table_posterior_hash_mismatch")
        saved_elpd = pd.to_numeric(
            pd.Series([record.get("exact_refit_elpd")]), errors="coerce"
        ).iloc[0]
        if not np.isfinite(saved_elpd):
            status.append("nonfinite_saved_exact_contribution")
        if not status and train_data is not None and test_data is not None:
            posterior_path = Path(str(signed["posterior_path"].iloc[0]))
            idata = az.from_netcdf(posterior_path)
            try:
                recomputed_mcmc = summarize_mcmc(idata)
                recomputed_elpd, recomputed_integration = (
                    _joint_participant_log_predictive_density(
                        idata,
                        train_data,
                        test_data,
                        primary_quadrature_nodes=int(
                            quadrature_values["primary_nodes"]
                        ),
                        validation_quadrature_nodes=int(
                            quadrature_values["validation_nodes"]
                        ),
                        quadrature_tolerance=float(
                            quadrature_values["tolerance"]
                        ),
                    )
                )
            finally:
                idata.close()
            if not bool(recomputed_mcmc["passed"]):
                status.append("recomputed_mcmc_diagnostics_failed")
            if not np.isclose(
                float(saved_elpd),
                recomputed_elpd,
                rtol=1.0e-10,
                atol=1.0e-8,
            ):
                status.append("exact_contribution_recomputation_mismatch")
            saved_numeric = pd.to_numeric(
                pd.Series(
                    {
                        "primary_quadrature_elpd": record.get(
                            "primary_quadrature_elpd"
                        ),
                        "validation_quadrature_elpd": record.get(
                            "validation_quadrature_elpd"
                        ),
                        "quadrature_absolute_difference": record.get(
                            "quadrature_absolute_difference"
                        ),
                    }
                ),
                errors="coerce",
            )
            for field in saved_numeric.index:
                if not np.isfinite(saved_numeric[field]) or not np.isclose(
                    float(saved_numeric[field]),
                    float(recomputed_integration[field]),
                    rtol=1.0e-10,
                    atol=1.0e-8,
                ):
                    status.append(f"{field}_recomputation_mismatch")
            if int(quadrature_values["predictive_draws"]) != int(
                recomputed_integration["posterior_draws"]
            ):
                status.append("predictive_draw_count_mismatch")
            if not bool(recomputed_integration["quadrature_stability_pass"]):
                status.append("recomputed_quadrature_stability_failed")
        rows.append(
            {
                **record.to_dict(),
                "recomputed_training_data_sha256": recomputed_hash,
                "recomputed_exact_refit_elpd": recomputed_elpd,
                "recomputed_primary_quadrature_elpd": recomputed_integration.get(
                    "primary_quadrature_elpd", np.nan
                ),
                "recomputed_validation_quadrature_elpd": recomputed_integration.get(
                    "validation_quadrature_elpd", np.nan
                ),
                "recomputed_quadrature_absolute_difference": recomputed_integration.get(
                    "quadrature_absolute_difference", np.nan
                ),
                "recomputed_quadrature_stability_pass": recomputed_integration.get(
                    "quadrature_stability_pass", False
                ),
                "recomputed_mcmc_max_rhat": recomputed_mcmc.get(
                    "max_rhat", np.nan
                ),
                "recomputed_mcmc_min_ess_bulk": recomputed_mcmc.get(
                    "min_ess_bulk", np.nan
                ),
                "recomputed_mcmc_min_ess_tail": recomputed_mcmc.get(
                    "min_ess_tail", np.nan
                ),
                "recomputed_mcmc_divergences": recomputed_mcmc.get(
                    "divergences", np.nan
                ),
                "recomputed_mcmc_passed": recomputed_mcmc.get("passed", False),
                "refit_contribution_validation_status": (
                    "authenticated_and_recomputed" if not status else ";".join(status)
                ),
                "authenticated_refit_contribution": not status,
                "validation_order": position,
            }
        )
    return pd.DataFrame(rows)


def _rq1_primary_estimand(
    estimands: pd.DataFrame,
    data: Any,
    config: AnalysisConfig,
) -> pd.DataFrame:
    arm_rows = estimands[
        estimands["estimand"].eq("common_opportunity_ai_choice_probability")
        & estimands["group"].isin(["G1", "G2"])
    ].copy()
    contrast = estimands[
        estimands["estimand"].eq(
            "G2_minus_G1_common_opportunity_ai_choice_probability"
        )
        & estimands["group"].eq("G2-G1")
    ].copy()
    result = pd.concat([arm_rows, contrast], ignore_index=True)
    if len(result) != 3 or set(result["group"]) != {"G1", "G2", "G2-G1"}:
        raise ValueError("RQ1 primary estimand requires G1, G2, and G2-G1 rows")
    result["estimand"] = (
        "rq1_opportunity_standardized_ai_target_selection_probability"
    )
    result["condition"] = result["group"].map(
        {
            "G1": "labels_visible",
            "G2": "labels_hidden",
            "G2-G1": "labels_hidden_minus_visible",
        }
    )
    result["estimate"] = result["posterior_median"]
    result["analysis_window"] = (
        f"{config.start_time} to {config.end_time or 'snapshot extraction'}"
    )
    result["pre_human_artifacts_excluded"] = True
    result["denominator"] = (
        f"{data.n_choices} observed human replies; "
        f"{int(data.mask.sum())} earlier valid non-self same-thread candidates"
    )
    result["event_type"] = "comments_replies"
    result["p_value"] = np.nan
    result["inference_boundary"] = (
        "one_community_per_condition_ecological_contrast"
    )
    result["opportunity_boundary"] = (
        "same_thread_candidates_not_verified_feed_or_read_exposure"
    )
    return result.sort_values(
        "group", key=lambda values: values.map({"G1": 0, "G2": 1, "G2-G1": 2})
    ).reset_index(drop=True)


def _rq1_prior_sensitivity(
    primary: pd.DataFrame,
    doubled: pd.DataFrame,
) -> pd.DataFrame:
    comparison = primary.merge(
        doubled,
        on=["estimand", "group"],
        suffixes=("_primary", "_doubled_prior"),
        validate="one_to_one",
    )
    comparison["absolute_median_shift"] = (
        comparison["posterior_median_doubled_prior"]
        - comparison["posterior_median_primary"]
    ).abs()
    comparison["credible_intervals_overlap"] = (
        comparison["ci_low_primary"] <= comparison["ci_high_doubled_prior"]
    ) & (
        comparison["ci_low_doubled_prior"] <= comparison["ci_high_primary"]
    )
    comparison["direction_stable"] = np.sign(
        comparison["posterior_median_primary"]
    ).eq(np.sign(comparison["posterior_median_doubled_prior"]))
    comparison["max_allowed_absolute_median_shift"] = 0.02
    comparison["passed"] = (
        comparison["credible_intervals_overlap"]
        & comparison["direction_stable"]
        & (
            comparison["absolute_median_shift"]
            <= comparison["max_allowed_absolute_median_shift"]
        )
    )
    return comparison


def _normalized_unit_ids(values: pd.Series) -> pd.Series:
    """Normalize integer-like influence identifiers for exact matching."""

    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype("string")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    normalized.loc[integer_like] = (
        numeric.loc[integer_like].round().astype("Int64").astype("string")
    )
    return normalized


def _rq1_influence_deletion_choice_ids(
    model_frame: pd.DataFrame,
    *,
    unit: str,
    group: str,
    unit_id: str,
) -> set[str]:
    """Return the choice events removed by one declared influence deletion."""

    required = {"choice_set_id", "group", "chosen"}
    if unit == "participant":
        unit_column = "actor_id"
        source = model_frame
    elif unit == "account":
        unit_column = "candidate_author_id"
        source = model_frame[model_frame["chosen"].astype(bool)]
    elif unit == "thread":
        unit_column = "root_id"
        source = model_frame
    else:
        raise ValueError("Influence unit must be participant, account, or thread")
    required.add(unit_column)
    missing = sorted(required.difference(model_frame.columns))
    if missing:
        raise ValueError(f"Model frame is missing {unit} deletion columns: {missing}")
    selected = source[
        source["group"].astype(str).eq(str(group))
        & _normalized_unit_ids(source[unit_column]).eq(str(unit_id))
    ]
    choice_ids = set(selected["choice_set_id"].astype(str))
    if not choice_ids:
        raise ValueError(f"No choices match {unit} deletion {group}::{unit_id}")
    return choice_ids


def _rq1_refit_model_data(
    model_frame: pd.DataFrame,
    term_manifest: pd.DataFrame,
    choice_ids: set[str],
) -> tuple[pd.DataFrame, Any]:
    """Build M1 data after deleting a declared set of choice events."""

    from .model import prepare_model_data

    retained = model_frame[
        ~model_frame["choice_set_id"].astype(str).isin(choice_ids)
    ].copy()
    if retained.empty:
        raise ValueError("Influence deletion removed every model choice")
    available_terms = included_terms_for_model("M1", term_manifest)
    numeric_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
    data = prepare_model_data(
        retained,
        numeric_terms,
        ai_term="candidate_target_is_ai",
    )
    return retained, data


def _fit_rq1_exact_influence_refits(
    config: AnalysisConfig,
    model_frame: pd.DataFrame,
    term_manifest: pd.DataFrame,
    primary: pd.DataFrame,
    influence_tables: Mapping[str, pd.DataFrame],
    *,
    draws: int,
    tune: int,
    chains: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[Path]]:
    """Fit exact M1 deletions for every grouped PSIS unit that needs refitting."""

    from .estimands import standardized_estimands

    flagged_rows: list[dict[str, Any]] = []
    for unit in ("participant", "account", "thread"):
        influence = influence_tables.get(unit, pd.DataFrame())
        if influence.empty:
            continue
        for record in influence[
            influence["needs_refit"].fillna(True).astype(bool)
        ].to_dict("records"):
            flagged_rows.append({**record, "influence_unit": unit})
    flagged_rows.sort(
        key=lambda row: (
            str(row["influence_unit"]),
            str(row["group"]),
            str(row["unit_id"]),
        )
    )
    refit_rows: list[dict[str, Any]] = []
    estimand_frames: list[pd.DataFrame] = []
    signature_rows: list[dict[str, Any]] = []
    posterior_paths: list[Path] = []
    main_estimates = primary.set_index("group")["estimate"].astype(float)
    for index, record in enumerate(flagged_rows):
        unit = str(record["influence_unit"])
        group = str(record["group"])
        unit_id = str(record["unit_id"])
        deleted_choices = _rq1_influence_deletion_choice_ids(
            model_frame,
            unit=unit,
            group=group,
            unit_id=unit_id,
        )
        expected = int(record["choice_sets"])
        if len(deleted_choices) != expected:
            raise ValueError(
                f"{unit} deletion {group}::{unit_id} matched "
                f"{len(deleted_choices)} choices; expected {expected}"
            )
        _, refit_data = _rq1_refit_model_data(
            model_frame,
            term_manifest,
            deleted_choices,
        )
        key = f"{unit}::{group}::{unit_id}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
        path = config.models_dir / f"M1_exact_delete_{unit}_{group}_{key_hash}.nc"
        role = f"rq1_exact_delete_{unit}_{group}_{key_hash}"
        idata, signature, diagnostic = _fit_with_diagnostic_extension(
            config,
            refit_data,
            "M1",
            path,
            role=role,
            draws=draws,
            tune=tune,
            chains=chains,
            seed_offset=10_000 + index,
            resume=resume,
        )
        refit_estimands = _rq1_primary_estimand(
            standardized_estimands(idata, refit_data, counterfactual_values={}),
            refit_data,
            config,
        )
        refit_estimands.insert(0, "deleted_unit_id", unit_id)
        refit_estimands.insert(0, "deleted_unit_group", group)
        refit_estimands.insert(0, "influence_unit", unit)
        refit_estimands["deleted_choice_sets"] = len(deleted_choices)
        refit_estimands["psis_pareto_k"] = float(record["pareto_k"])
        refit_estimands["main_estimate"] = refit_estimands["group"].map(
            main_estimates
        )
        refit_estimands["estimate_shift"] = (
            refit_estimands["estimate"] - refit_estimands["main_estimate"]
        )
        refit_estimands["absolute_estimate_shift"] = refit_estimands[
            "estimate_shift"
        ].abs()
        finite = bool(
            np.isfinite(
                refit_estimands[
                    ["estimate", "ci_low", "ci_high", "estimate_shift"]
                ].to_numpy(dtype=float)
            ).all()
        )
        estimand_frames.append(refit_estimands)
        refit_rows.append(
            {
                "influence_unit": unit,
                "group": group,
                "unit_id": unit_id,
                "choice_sets": len(deleted_choices),
                "psis_pareto_k": float(record["pareto_k"]),
                "refit_choice_sets": int(refit_data.n_choices),
                "refit_candidate_rows": int(refit_data.mask.sum()),
                "refit_data_sha256": _prepared_data_hash(refit_data),
                "estimand_rows": int(len(refit_estimands)),
                "estimands_finite": finite,
                "mcmc_passed": bool(diagnostic["passed"]),
                "sampling_extended": bool(diagnostic["sampling_extended"]),
                "max_rhat": diagnostic["max_rhat"],
                "min_ess_bulk": diagnostic["min_ess_bulk"],
                "min_ess_tail": diagnostic["min_ess_tail"],
                "divergences": diagnostic["divergences"],
                "posterior_path": str(path.resolve()),
                "posterior_sha256": str(signature["actual"]["posterior_sha256"]),
                "signature_role": role,
                "exact_refit_completed": bool(diagnostic["passed"] and finite),
            }
        )
        signature_row = _signature_row(path, signature, refit_data)
        signature_row.update(
            {
                "influence_unit": unit,
                "deleted_unit_group": group,
                "deleted_unit_id": unit_id,
            }
        )
        signature_rows.append(signature_row)
        posterior_paths.append(path)
        close_idata = getattr(idata, "close", None)
        if callable(close_idata):
            close_idata()
        del idata
        import gc

        gc.collect()
    refits = pd.DataFrame(refit_rows)
    estimands = (
        pd.concat(estimand_frames, ignore_index=True)
        if estimand_frames
        else pd.DataFrame()
    )
    return refits, estimands, signature_rows, posterior_paths


def _validate_rq1_exact_influence_refits(
    config: AnalysisConfig,
    model_frame: pd.DataFrame,
    term_manifest: pd.DataFrame,
    refits: pd.DataFrame,
    saved_estimands: pd.DataFrame,
    authenticated_signatures: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild deletion data and authenticate every exact RQ1 influence refit."""

    if refits.empty:
        return pd.DataFrame(
            columns=[
                *refits.columns,
                "recomputed_refit_data_sha256",
                "recomputed_mcmc_passed",
                "recomputed_estimands_match",
                "authenticated_exact_refit",
                "exact_refit_authentication_status",
            ]
        )
    import arviz as az
    from .estimands import standardized_estimands

    implementation_hash = _model_implementation_hash(config)
    refit_keys = (
        refits["influence_unit"].astype(str)
        + "::"
        + refits["group"].astype(str)
        + "::"
        + _normalized_unit_ids(refits["unit_id"])
    )
    duplicated = refit_keys.duplicated(keep=False)
    rows: list[dict[str, Any]] = []
    for index, record in refits.iterrows():
        unit = str(record["influence_unit"])
        group = str(record["group"])
        unit_id = str(_normalized_unit_ids(pd.Series([record["unit_id"]])).iloc[0])
        status: list[str] = []
        if bool(duplicated.loc[index]):
            status.append("duplicate_exact_refit_key")
        deleted_choices = _rq1_influence_deletion_choice_ids(
            model_frame,
            unit=unit,
            group=group,
            unit_id=unit_id,
        )
        _, refit_data = _rq1_refit_model_data(
            model_frame,
            term_manifest,
            deleted_choices,
        )
        recomputed_hash = _prepared_data_hash(refit_data)
        if recomputed_hash != str(record["refit_data_sha256"]):
            status.append("refit_data_hash_mismatch")
        if len(deleted_choices) != int(record["choice_sets"]):
            status.append("deleted_choice_count_mismatch")
        signature = authenticated_signatures[
            authenticated_signatures.get(
                "posterior_path", pd.Series(dtype=object)
            ).astype(str).eq(str(record["posterior_path"]))
            & authenticated_signatures.get("role", pd.Series(dtype=object))
            .astype(str)
            .eq(str(record["signature_role"]))
        ]
        signature_ok = bool(
            len(signature) == 1
            and signature.get(
                "live_posterior_authenticated",
                pd.Series(False, index=signature.index),
            )
            .fillna(False)
            .astype(bool)
            .all()
            and signature["data_sha256"].astype(str).eq(recomputed_hash).all()
            and signature["implementation_sha256"]
            .astype(str)
            .eq(implementation_hash)
            .all()
            and signature.get(
                "publication_length_request",
                pd.Series(False, index=signature.index),
            )
            .fillna(False)
            .astype(bool)
            .all()
            and signature.get(
                "dimension_match", pd.Series(False, index=signature.index)
            )
            .fillna(False)
            .astype(bool)
            .all()
            and signature["posterior_sha256"]
            .astype(str)
            .eq(str(record["posterior_sha256"]))
            .all()
        )
        if not signature_ok:
            status.append("posterior_signature_not_authenticated")
        recomputed_mcmc: dict[str, float | int | bool] = {}
        estimands_match = False
        if signature_ok:
            idata = az.from_netcdf(Path(str(record["posterior_path"])))
            try:
                recomputed_mcmc = summarize_mcmc(idata)
                recomputed = _rq1_primary_estimand(
                    standardized_estimands(
                        idata, refit_data, counterfactual_values={}
                    ),
                    refit_data,
                    config,
                )
                stored = saved_estimands[
                    saved_estimands["influence_unit"].astype(str).eq(unit)
                    & saved_estimands["deleted_unit_group"].astype(str).eq(group)
                    & _normalized_unit_ids(saved_estimands["deleted_unit_id"]).eq(
                        unit_id
                    )
                ].sort_values("group")
                recomputed = recomputed.sort_values("group")
                columns = [
                    "posterior_mean",
                    "posterior_median",
                    "ci_low",
                    "ci_high",
                    "posterior_probability_gt_zero",
                ]
                estimands_match = bool(
                    len(stored) == 3
                    and len(recomputed) == 3
                    and stored["group"].astype(str).tolist()
                    == recomputed["group"].astype(str).tolist()
                    and np.allclose(
                        stored[columns].to_numpy(dtype=float),
                        recomputed[columns].to_numpy(dtype=float),
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )
                )
            finally:
                idata.close()
        if not bool(recomputed_mcmc.get("passed", False)):
            status.append("recomputed_mcmc_failed")
        if not estimands_match:
            status.append("recomputed_estimands_mismatch")
        rows.append(
            {
                **record.to_dict(),
                "unit_id": unit_id,
                "recomputed_refit_data_sha256": recomputed_hash,
                "recomputed_mcmc_passed": bool(
                    recomputed_mcmc.get("passed", False)
                ),
                "recomputed_estimands_match": estimands_match,
                "authenticated_exact_refit": not status,
                "exact_refit_authentication_status": (
                    "authenticated_and_recomputed" if not status else ";".join(status)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_fit_choice(
    config: AnalysisConfig,
    config_path: Path,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    resume: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fit and authenticate the final M1 model and its RQ1 checks."""

    if config.analysis_stage != "rq1_m1":
        raise ValueError("fit-choice requires analysis_stage: rq1_m1")
    _directories(config)
    _invalidate_rq1_artifacts(config, stage="fit")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    fit_source_snapshot = capture_source_snapshot(source_paths(config, config_path))

    from .estimands import standardized_estimands
    from .model import prepare_model_data, summarize_fit

    model_frame = read_table(config.tables_dir / "model_frame")
    term_manifest = read_table(config.tables_dir / "model_term_manifest")
    available_terms = included_terms_for_model("M1", term_manifest)
    numeric_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
    data = prepare_model_data(
        model_frame,
        numeric_terms,
        ai_term="candidate_target_is_ai",
    )
    expected_draws = draws or config.sampling.draws
    expected_tune = tune or config.sampling.tune
    expected_chains = chains or config.sampling.chains

    main_path = config.models_dir / "M1.nc"
    main_idata, main_signature, main_diagnostic = _fit_with_diagnostic_extension(
        config,
        data,
        "M1",
        main_path,
        role="rq1_main",
        draws=expected_draws,
        tune=expected_tune,
        chains=expected_chains,
        seed_offset=0,
        resume=resume,
    )
    main_parameters = summarize_fit(main_idata)
    main_parameters.insert(0, "model", "M1")
    all_estimands = standardized_estimands(
        main_idata, data, counterfactual_values={}
    )
    all_estimands.insert(0, "model", "M1")
    primary = _rq1_primary_estimand(all_estimands, data, config)

    influence_tables = {
        unit: grouped_choice_psis(main_idata, model_frame, unit=unit)
        for unit in ("participant", "account", "thread")
    }

    doubled_sampling = replace(
        config.sampling,
        prior_focal_sigma=2.0 * config.sampling.prior_focal_sigma,
        prior_adjustment_sigma=2.0 * config.sampling.prior_adjustment_sigma,
        prior_scale_sigma=2.0 * config.sampling.prior_scale_sigma,
    )
    doubled_config = replace(config, sampling=doubled_sampling)
    prior_path = config.models_dir / "M1_doubled_prior_scales.nc"
    prior_idata, prior_signature, prior_diagnostic = _fit_with_diagnostic_extension(
        doubled_config,
        data,
        "M1",
        prior_path,
        role="rq1_doubled_prior_scales",
        draws=expected_draws,
        tune=expected_tune,
        chains=expected_chains,
        seed_offset=50,
        resume=resume,
    )
    prior_estimands = standardized_estimands(
        prior_idata, data, counterfactual_values={}
    )
    prior_primary = _rq1_primary_estimand(prior_estimands, data, config)
    prior_sensitivity = _rq1_prior_sensitivity(primary, prior_primary)

    forward_tables, forward_signatures = run_forward_validation_models(
        config,
        model_frame,
        term_manifest,
        draws=draws,
        tune=tune,
        chains=chains,
        resume=resume,
        model_names=("M0", "M1"),
        reference_model="M1",
    )
    (
        exact_refits,
        exact_refit_estimands,
        exact_refit_signatures,
        exact_refit_paths,
    ) = _fit_rq1_exact_influence_refits(
        config,
        model_frame,
        term_manifest,
        primary,
        influence_tables,
        draws=expected_draws,
        tune=expected_tune,
        chains=expected_chains,
        resume=resume,
    )
    diagnostics = pd.concat(
        [
            pd.DataFrame(
                [
                    {"model": "M1", "role": "rq1_main", **main_diagnostic},
                    {
                        "model": "M1",
                        "role": "rq1_doubled_prior_scales",
                        **prior_diagnostic,
                    },
                ]
            ),
            forward_tables["forward_validation_diagnostics"].assign(
                role=lambda frame: frame["fold"].map(
                    lambda fold: f"forward_fold_{int(fold)}"
                )
            ),
        ],
        ignore_index=True,
    )
    signature_rows = [
        _signature_row(main_path, main_signature, data),
        _signature_row(prior_path, prior_signature, data),
        *forward_signatures,
        *exact_refit_signatures,
    ]
    output_tables = {
        "rq1_model_parameter_summary": main_parameters,
        "rq1_model_estimands": all_estimands,
        "rq1_primary_estimand": primary,
        "rq1_model_diagnostics": diagnostics,
        "rq1_prior_scale_sensitivity": prior_sensitivity,
        "rq1_influence_participant": influence_tables["participant"],
        "rq1_influence_account": influence_tables["account"],
        "rq1_influence_thread": influence_tables["thread"],
        "rq1_influence_exact_refits": exact_refits,
        "rq1_influence_exact_estimands": exact_refit_estimands,
        "rq1_forward_validation_scores": forward_tables[
            "forward_validation_scores"
        ],
        "rq1_forward_validation_comparisons": forward_tables[
            "forward_validation_comparisons"
        ],
        "rq1_forward_validation_predictions": forward_tables[
            "forward_validation_predictions"
        ],
        "rq1_forward_validation_diagnostics": forward_tables[
            "forward_validation_diagnostics"
        ],
        "fit_run_signature": pd.DataFrame(signature_rows),
    }
    assert_source_snapshot_unchanged(fit_source_snapshot)
    artifacts, metadata = _write_tables(config, output_tables)
    model_paths = [
        main_path,
        prior_path,
        *sorted(config.models_dir.glob("M[01]_forward_fold*.nc")),
        *exact_refit_paths,
    ]
    bundle = write_fit_result_bundle(
        config,
        source_snapshot=fit_source_snapshot,
        model_data_sha256=_prepared_data_hash(data),
        implementation_sha256=_model_implementation_hash(config),
        table_artifacts=artifacts,
        posterior_paths=model_paths,
        fit_request={
            "analysis_stage": "rq1_m1",
            "models": ["M1"],
            "forward_models": ["M0", "M1"],
            "draws": expected_draws,
            "tune": expected_tune,
            "chains": expected_chains,
            "prior_scale_sensitivity": "doubled",
            "influence_units": ["participant", "account", "thread"],
            "influence_fallback": "exact_M1_deletion_refit_for_grouped_PSIS_k_gt_0.70",
        },
    )
    metadata[bundle] = {
        "artifact": "authenticated RQ1 fit-result bundle",
        "denominator": primary["denominator"].iloc[0],
    }
    assert_source_snapshot_unchanged(fit_source_snapshot)
    manifest = write_manifest(
        config, config_path, [*artifacts, *model_paths, bundle]
    )
    write_artifact_manifests(config, manifest, metadata)
    diagnostics_pass = bool(
        not diagnostics.empty
        and diagnostics["passed"].fillna(False).astype(bool).all()
    )
    write_json(
        {
            "stage": "fit-choice",
            "status": (
                "completed" if diagnostics_pass else "completed_with_failed_diagnostics"
            ),
            "run_id": config.run_id,
        },
        config.run_dir / "status.json",
    )
    return output_tables


def run_fit(
    config: AnalysisConfig,
    config_path: Path,
    *,
    models: Sequence[str] = ("M0", "M1", "M2", "M3"),
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    include_latent: bool = False,
    run_forward_validation: bool = False,
    fit_browser_sensitivity: bool = True,
    run_prior_sensitivity: bool = True,
    resume: bool = True,
    run_problematic_participant_refits: bool = False,
    latent_recovery_replicates: int = 3,
    latent_recovery_actors: int = 12,
    latent_recovery_receivers: int = 24,
    latent_recovery_choice_sets: int = 240,
    latent_recovery_candidates: int = 5,
    latent_recovery_draws: int = 500,
    latent_recovery_tune: int = 500,
    latent_recovery_chains: int = 2,
) -> dict[str, pd.DataFrame]:
    """Fit the nested relational-event sequence and conditional network checks."""

    _directories(config)
    _invalidate_downstream_artifacts(config, stage="fit")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    fit_source_snapshot = capture_source_snapshot(source_paths(config, config_path))

    import arviz as az

    from .model import (
        conditional_posterior_predictive_choices,
        prepare_model_data,
        summarize_fit,
    )
    from .estimands import standardized_estimands
    from .recursive_ppc import recursive_posterior_predictive_choices
    from .latent_validation import (
        LatentRecoveryDesign,
        not_run_status,
        run_latent_validation,
    )

    model_frame = read_table(config.tables_dir / "model_frame")
    term_manifest = read_table(config.tables_dir / "model_term_manifest")
    event_history = read_table(config.tables_dir / "event_history")
    account_universe = read_table(config.tables_dir / "account_universe")
    available_terms = included_terms_for_model("M3", term_manifest)
    numeric_terms = tuple(term for term in available_terms if term != "candidate_target_is_ai")
    data = prepare_model_data(
        model_frame,
        numeric_terms,
        ai_term="candidate_target_is_ai",
        closure_term="z_log_two_path_closure",
    )
    counterfactual_values = _network_counterfactual_values(term_manifest)
    unknown = [name for name in models if name not in {"M0", "M1", "M2", "M3"}]
    if unknown:
        raise ValueError(f"Unknown publication models: {unknown}")

    parameter_tables: list[pd.DataFrame] = []
    estimand_tables: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    model_paths: list[Path] = []
    signature_rows: list[dict[str, Any]] = []
    completed_models: list[str] = []
    for index, model_name in enumerate(models):
        write_json(
            {
                "stage": "fit",
                "active_model": model_name,
                "completed_models": completed_models,
                "run_id": config.run_id,
            },
            config.run_dir / "status.json",
        )
        path = config.models_dir / f"{model_name}.nc"
        expected_draws = draws or config.sampling.draws
        expected_tune = tune or config.sampling.tune
        expected_chains = chains or config.sampling.chains
        idata, signature, diagnostic = _fit_with_diagnostic_extension(
            config,
            data,
            model_name,
            path,
            role="main_nested_sequence",
            draws=expected_draws,
            tune=expected_tune,
            chains=expected_chains,
            seed_offset=index,
            resume=resume,
        )
        signature_rows.append(_signature_row(path, signature, data))
        model_paths.append(path)
        fitted[model_name] = idata
        completed_models.append(model_name)
        parameters = summarize_fit(idata)
        parameters.insert(0, "model", model_name)
        parameter_tables.append(parameters)
        estimands = standardized_estimands(
            idata,
            data,
            counterfactual_values=counterfactual_values,
        )
        estimands.insert(0, "model", model_name)
        estimand_tables.append(estimands)
        diagnostic_rows.append({"model": model_name, **diagnostic})

    if not fitted:
        raise ValueError("At least one model must be fitted")
    final_model = "M3" if "M3" in fitted else list(fitted)[-1]
    final_idata = fitted[final_model]
    all_parameters = pd.concat(parameter_tables, ignore_index=True)
    all_estimands = pd.concat(estimand_tables, ignore_index=True)
    primary_estimands = all_estimands[all_estimands["model"].eq(final_model)].copy()
    diagnostics = pd.DataFrame(diagnostic_rows)

    predictive_draws = min(
        1_000,
        (draws or config.sampling.draws) * (chains or config.sampling.chains),
    )
    if final_model == "M3":
        predictions = recursive_posterior_predictive_choices(
            final_idata,
            data,
            model_frame,
            event_history,
            term_manifest,
            draws=predictive_draws,
            random_seed=config.seed,
        )
    else:
        predictions = conditional_posterior_predictive_choices(
            final_idata,
            data,
            draws=predictive_draws,
            random_seed=config.seed,
        )
    predictive_checks = _posterior_predictive_summary(
        model_frame, predictions, account_universe
    )
    grouped_psis_tables: list[pd.DataFrame] = []
    for model_name, idata in fitted.items():
        grouped = participant_grouped_psis(idata, model_frame)
        grouped.insert(0, "model", model_name)
        grouped_psis_tables.append(grouped)
    participant_psis_all_models = pd.concat(grouped_psis_tables, ignore_index=True)
    participant_psis = participant_psis_all_models[
        participant_psis_all_models["model"].eq(final_model)
    ].drop(columns="model")
    loo_rows: list[dict[str, Any]] = []
    for model_name, idata in fitted.items():
        loo = az.loo(idata, var_name="choice", pointwise=True)
        loo_rows.append(
            {
                "model": model_name,
                "elpd_loo": float(loo.elpd_loo),
                "se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "pareto_k_max": float(loo.pareto_k.max()),
                "warning": bool(loo.warning),
                "unit": "choice_set",
                "analysis_role": (
                    "auxiliary_choice_level_diagnostic_not_used_for_model_comparison"
                ),
            }
        )

    output_tables: dict[str, pd.DataFrame] = {
        "model_parameter_summary": all_parameters,
        "model_estimands": all_estimands,
        "primary_estimands": primary_estimands,
        "model_diagnostics": diagnostics,
        "posterior_predictive_checks": predictive_checks,
        "participant_grouped_psis": participant_psis,
        "participant_grouped_psis_all_models": participant_psis_all_models,
        "model_loo": pd.DataFrame(loo_rows),
    }
    if run_problematic_participant_refits:
        participant_refits = (
            run_participant_loo_refits(
                config,
                model_frame,
                participant_psis,
                draws=draws,
                tune=tune,
                chains=chains,
                resume=resume,
            )
            if participant_psis["needs_refit"].any()
            else pd.DataFrame(
                columns=[
                    "group",
                    "actor_id",
                    "exact_refit_elpd",
                    "refit_status",
                    "mcmc_passed",
                ]
            )
        )
        output_tables["participant_loo_refits"] = participant_refits
        participant_paths = [
            config.models_dir
            / f"M3_participant_refit_{str(row.group)}_{str(row.actor_id)}.nc"
            for row in participant_refits.itertuples(index=False)
        ]
        model_paths.extend(participant_paths)
        for path in participant_paths:
            sidecar = path.with_suffix(path.suffix + ".fit.json")
            if sidecar.exists():
                signature_rows.append(
                    _signature_row(
                        path,
                        json.loads(sidecar.read_text(encoding="utf-8")),
                    )
                )

    if run_prior_sensitivity and final_model == "M3":
        doubled_sampling = replace(
            config.sampling,
            prior_focal_sigma=2.0 * config.sampling.prior_focal_sigma,
            prior_adjustment_sigma=2.0 * config.sampling.prior_adjustment_sigma,
            prior_scale_sigma=2.0 * config.sampling.prior_scale_sigma,
        )
        sensitivity_config = replace(config, sampling=doubled_sampling)
        sensitivity_path = config.models_dir / "M3_doubled_prior_scales.nc"
        sensitivity_idata, signature, sensitivity_diagnostic = (
            _fit_with_diagnostic_extension(
                sensitivity_config,
                data,
                "M3",
                sensitivity_path,
                role="doubled_prior_scales",
                draws=draws or config.sampling.draws,
                tune=tune or config.sampling.tune,
                chains=chains or config.sampling.chains,
                seed_offset=50,
                resume=resume,
            )
        )
        signature_rows.append(_signature_row(sensitivity_path, signature, data))
        model_paths.append(sensitivity_path)
        sensitivity_estimands = standardized_estimands(
            sensitivity_idata,
            data,
            counterfactual_values=counterfactual_values,
        )
        comparison = primary_estimands.merge(
            sensitivity_estimands,
            on=["estimand", "group"],
            suffixes=("_primary", "_doubled_prior"),
            validate="one_to_one",
        )
        comparison["absolute_median_shift"] = (
            comparison["posterior_median_doubled_prior"]
            - comparison["posterior_median_primary"]
        ).abs()
        comparison["credible_intervals_overlap"] = (
            comparison["ci_low_primary"] <= comparison["ci_high_doubled_prior"]
        ) & (comparison["ci_low_doubled_prior"] <= comparison["ci_high_primary"])
        comparison["direction_stable"] = np.sign(
            comparison["posterior_median_primary"]
        ).eq(np.sign(comparison["posterior_median_doubled_prior"]))
        rope_half_width = 0.01
        for suffix in ("primary", "doubled_prior"):
            low = comparison[f"ci_low_{suffix}"]
            high = comparison[f"ci_high_{suffix}"]
            comparison[f"evidence_class_{suffix}"] = np.select(
                [
                    low > rope_half_width,
                    high < -rope_half_width,
                    (low >= -rope_half_width) & (high <= rope_half_width),
                ],
                ["positive", "negative", "inside_ROPE"],
                default="uncertain",
            )
        comparison["evidence_class_stable"] = comparison[
            "evidence_class_primary"
        ].eq(comparison["evidence_class_doubled_prior"])
        comparison["rope_half_width"] = rope_half_width
        comparison["max_allowed_absolute_median_shift"] = 0.02
        comparison["substantive_shift_pass"] = (
            comparison["absolute_median_shift"]
            <= comparison["max_allowed_absolute_median_shift"]
        )
        comparison["passed"] = (
            comparison["credible_intervals_overlap"]
            & comparison["substantive_shift_pass"]
            & comparison["evidence_class_stable"]
        )
        output_tables["prior_scale_sensitivity"] = comparison
        diagnostic_rows.append(
            {"model": "M3_doubled_prior_scales", **sensitivity_diagnostic}
        )
        output_tables["model_diagnostics"] = pd.DataFrame(diagnostic_rows)

    if fit_browser_sensitivity:
        strict = read_table(config.tables_dir / "browser_strict_choice_sets")
        if not strict.empty:
            keys = strict[["choice_set_id", "candidate_content_id"]].drop_duplicates()
            strict_frame = model_frame.merge(
                keys,
                on=["choice_set_id", "candidate_content_id"],
                how="inner",
                validate="one_to_one",
            )
            valid_ids = [
                choice_id
                for choice_id, part in strict_frame.groupby("choice_set_id", sort=False)
                if len(part) >= 2 and int(part["chosen"].sum()) == 1
            ]
            strict_frame = strict_frame[strict_frame["choice_set_id"].isin(valid_ids)]
            if strict_frame["choice_set_id"].nunique() >= 2:
                strict_data = prepare_model_data(
                    strict_frame,
                    numeric_terms,
                    ai_term="candidate_target_is_ai",
                    closure_term="z_log_two_path_closure",
                )
                strict_path = config.models_dir / "M3_browser_strict_sensitivity.nc"
                strict_idata, signature, strict_diagnostic = (
                    _fit_with_diagnostic_extension(
                        config,
                        strict_data,
                        "M3",
                        strict_path,
                        role="browser_strict_sensitivity",
                        draws=draws or config.sampling.draws,
                        tune=tune or config.sampling.tune,
                        chains=chains or config.sampling.chains,
                        seed_offset=100,
                        resume=resume,
                    )
                )
                signature_rows.append(_signature_row(strict_path, signature, strict_data))
                model_paths.append(strict_path)
                strict_estimands = standardized_estimands(
                    strict_idata,
                    strict_data,
                    counterfactual_values=counterfactual_values,
                )
                strict_estimands.insert(0, "model", "M3_browser_strict_sensitivity")
                tracking_qc = read_table(config.tables_dir / "tracking_linkage_qc")
                target_gate = tracking_qc[
                    tracking_qc["metric"].eq("target_choice_tracking_gates_pass")
                    & tracking_qc["group"].eq("all")
                ]
                target_gate_passed = bool(
                    len(target_gate) == 1
                    and target_gate["gate_pass"].fillna(False).astype(bool).all()
                )
                strict_estimands["inference_status"] = (
                    "gate_passed_strict_rendered_risk_set"
                    if target_gate_passed
                    else "sensitivity_pending_manual_link_audit"
                )
                output_tables["browser_strict_model_estimands"] = strict_estimands
                comparison = primary_estimands.merge(
                    strict_estimands,
                    on=["estimand", "group"],
                    suffixes=("_full_thread", "_strict_browser"),
                    validate="one_to_one",
                )
                comparison["posterior_median_difference_strict_minus_full"] = (
                    comparison["posterior_median_strict_browser"]
                    - comparison["posterior_median_full_thread"]
                )
                comparison["target_choice_tracking_gate_passed"] = target_gate_passed
                output_tables["browser_full_vs_strict_estimand_comparison"] = comparison
                output_tables["browser_strict_model_diagnostics"] = pd.DataFrame(
                    [{"model": "M3_browser_strict_sensitivity", **strict_diagnostic}]
                )

    if include_latent:
        latent_tables, latent_signatures, latent_paths = run_latent_validation(
            config,
            model_frame,
            term_manifest,
            data,
            draws=draws or config.sampling.draws,
            tune=tune or config.sampling.tune,
            chains=chains or config.sampling.chains,
            resume=resume,
            recovery_design=LatentRecoveryDesign(
                replicates=latent_recovery_replicates,
                n_actors=latent_recovery_actors,
                n_receivers=latent_recovery_receivers,
                n_choice_sets=latent_recovery_choice_sets,
                candidates_per_set=latent_recovery_candidates,
                draws=latent_recovery_draws,
                tune=latent_recovery_tune,
                chains=latent_recovery_chains,
            ),
        )
        output_tables.update(latent_tables)
        signature_rows.extend(latent_signatures)
        model_paths.extend(latent_paths)
    else:
        output_tables["latent_block_gate_status"] = not_run_status(
            config.latent_block_sizes
        )

    if run_forward_validation:
        forward_tables, forward_signatures = run_forward_validation_models(
            config,
            model_frame,
            term_manifest,
            draws=draws,
            tune=tune,
            chains=chains,
            resume=resume,
        )
        output_tables.update(forward_tables)
        signature_rows.extend(forward_signatures)
        model_paths.extend(
            sorted(config.models_dir.glob("M[0-3]_forward_fold*.nc"))
        )

    signature_rows = list(
        {str(row["posterior_path"]): row for row in signature_rows}.values()
    )
    model_paths = list(dict.fromkeys(model_paths))
    output_tables["fit_run_signature"] = pd.DataFrame(signature_rows)
    assert_source_snapshot_unchanged(fit_source_snapshot)
    artifacts, metadata = _write_tables(config, output_tables)
    signatures_by_path = {
        str(row["posterior_path"]): row for row in signature_rows
    }
    for path in model_paths:
        fit_signature = signatures_by_path.get(str(path), {})
        denominator = (
            f"{int(fit_signature['denominator_choice_sets'])} human reply choice sets; "
            f"{int(fit_signature['denominator_candidate_rows'])} candidate rows"
            if "denominator_choice_sets" in fit_signature
            else "posterior-specific signed training choice sets; see fit sidecar data hash"
        )
        provenance_sidecar = path.with_suffix(path.suffix + ".manifest.json")
        if not provenance_sidecar.exists():
            metadata[path] = {
                "artifact": "Bayesian relational-event posterior",
                "denominator": denominator,
                "fit_signature": fit_signature,
            }
    assert_source_snapshot_unchanged(fit_source_snapshot)
    bundle = write_fit_result_bundle(
        config,
        source_snapshot=fit_source_snapshot,
        model_data_sha256=_prepared_data_hash(data),
        implementation_sha256=_model_implementation_hash(config),
        table_artifacts=artifacts,
        posterior_paths=model_paths,
        fit_request={
            "models": list(models),
            "draws": draws or config.sampling.draws,
            "tune": tune or config.sampling.tune,
            "chains": chains or config.sampling.chains,
            "include_latent": include_latent,
            "run_forward_validation": run_forward_validation,
            "fit_browser_sensitivity": fit_browser_sensitivity,
            "run_prior_sensitivity": run_prior_sensitivity,
            "run_problematic_participant_refits": run_problematic_participant_refits,
        },
    )
    metadata[bundle] = {
        "artifact": "authenticated fit-result bundle",
        "denominator": "all fitted result tables and posterior artifacts",
    }
    assert_source_snapshot_unchanged(fit_source_snapshot)
    manifest = write_manifest(
        config,
        config_path,
        [*artifacts, *model_paths, bundle],
    )
    write_artifact_manifests(config, manifest, metadata)
    status_diagnostics = [pd.DataFrame(diagnostic_rows)]
    for table_name in (
        "forward_validation_diagnostics",
        "browser_strict_model_diagnostics",
        "latent_fit_diagnostics",
    ):
        if table_name in output_tables:
            status_diagnostics.append(output_tables[table_name])
    if "participant_loo_refits" in output_tables:
        participant_diagnostics = output_tables["participant_loo_refits"].rename(
            columns={"mcmc_passed": "passed"}
        )
        status_diagnostics.append(participant_diagnostics)
    all_diagnostics = pd.concat(status_diagnostics, ignore_index=True)
    all_diagnostics_pass = bool(
        not all_diagnostics.empty
        and all_diagnostics.get(
            "passed", pd.Series(False, index=all_diagnostics.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    write_json(
        {
            "stage": "fit",
            "status": (
                "completed"
                if all_diagnostics_pass
                else "completed_with_failed_diagnostics"
            ),
            "completed_models": completed_models,
            "run_id": config.run_id,
        },
        config.run_dir / "status.json",
    )
    return output_tables


__all__ = [
    "run_fit",
    "run_fit_choice",
    "run_prepare",
    "run_prepare_choice",
    "run_prepare_rq2",
    "run_prepare_rq3",
    "run_fit_rq2",
    "run_estimate_rq3",
    "run_report_rq2",
    "run_report_stage",
    "run_report_rq3",
    "run_validate",
    "run_validate_choice",
    "run_validate_rq2",
    "run_validate_rq3",
]


# ---------------------------------------------------------------------------
# RQ3: human downvoting of selected reply targets within fixed windows
# ---------------------------------------------------------------------------

RQ3_PREPARE_TABLE_NAMES = frozenset(
    {
        "rq3_round_timestamps",
        "rq3_reaction_coverage",
        "rq3_event_history",
        "rq3_sanction_panel",
        "rq3_panel_signature",
        "rq3_data_integrity_checks",
        "rq3_actor_label_qc",
        "rq3_group_database_provenance",
    }
)

RQ3_ESTIMATE_TABLE_NAMES = frozenset(
    {
        "rq3_vote_panel",
        "rq3_vote_source",
        "rq3_downvote_timing",
        "rq3_unconditional_downvotes",
        "rq3_group_summary",
        "rq3_bootstrap_contrasts",
        "rq3_resolution_sensitivity",
        "rq3_primary_estimand",
    }
)

RQ3_VALIDATION_TABLE_NAMES = frozenset(
    {
        "rq3_panel_reproduction",
        "rq3_publication_readiness",
    }
)

RQ3_TABLE_DENOMINATORS: Mapping[str, str] = {
    "rq3_round_timestamps": "platform rounds mapped to clock hours from timestamped comments",
    "rq3_reaction_coverage": "reaction rows by arm, type, and actor type before and after stamping",
    "rq3_event_history": "valid analysis-set events with round-stamped reactions",
    "rq3_sanction_panel": "human replies with a selected non-self same-thread target",
    "rq3_panel_signature": "one hash of the sanction panel and its implementation",
    "rq3_data_integrity_checks": "publication-blocking canonical data invariants",
    "rq3_actor_label_qc": "cross-source unique accounts by arm and authoritative H/AI field",
    "rq3_group_database_provenance": "one immutable source-manifest identity record per arm",
    "rq3_vote_panel": "sanction-panel rows with windowed human-vote outcomes per reaction placement",
    "rq3_vote_source": "human reactions entering the windows by arm and type",
    "rq3_downvote_timing": "human downvotes on selected targets by position relative to the reply round",
    "rq3_unconditional_downvotes": "valid contributions by arm, author type, and reply status; any human downvote, no window",
    "rq3_group_summary": "selected targets by subset, placement, arm, dyad, and metric",
    "rq3_bootstrap_contrasts": "G2 minus G1 differences in participant means with participant bootstrap",
    "rq3_resolution_sensitivity": "rows reclassified between round-start and alternative placements",
    "rq3_primary_estimand": "selected AI-authored targets in the RQ1 analysis set; 24 h human downvote",
    "rq3_panel_reproduction": "validation-time panel hash comparison",
    "rq3_publication_readiness": "declared RQ3 release gates",
}
TABLE_DENOMINATORS = {**TABLE_DENOMINATORS, **RQ3_TABLE_DENOMINATORS}


def _rq3_stage_guard(config: AnalysisConfig, stage: str) -> None:
    if config.analysis_stage not in {"rq1_m1", "rq3"}:
        raise ValueError(f"{stage} requires analysis_stage: rq1_m1 or rq3")


def _invalidate_rq3_artifacts(config: AnalysisConfig, *, stage: str) -> None:
    """Remove stale RQ3 surfaces while preserving upstream RQ3 stages."""

    if stage not in {"prepare", "estimate", "validate"}:
        raise ValueError(f"Unknown RQ3 invalidation stage {stage!r}")
    _remove_directory_files(config.reports_dir)
    _remove_directory_files(config.figures_dir)
    if config.tables_dir.exists():
        for artifact in config.tables_dir.iterdir():
            if not artifact.is_file():
                continue
            table_name = _artifact_table_name(artifact)
            if not table_name.startswith("rq3_"):
                continue
            remove = (
                stage == "prepare"
                or (stage == "estimate" and table_name not in RQ3_PREPARE_TABLE_NAMES)
                or (stage == "validate" and table_name in RQ3_VALIDATION_TABLE_NAMES)
            )
            if remove:
                artifact.unlink(missing_ok=True)
    (config.run_dir / "manifest.json").unlink(missing_ok=True)
    (config.outputs_dir / "validation-gates.json").unlink(missing_ok=True)
    if stage in {"prepare", "estimate"}:
        (config.run_dir / "status.json").unlink(missing_ok=True)


def _rq3_implementation_hash(config: AnalysisConfig) -> str:
    path = Path(__file__).resolve().parent / "rq3.py"
    return sha256_file(path)


def _rq3_prepare_frames(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Build the RQ3 prepare tables from immutable sources."""

    from . import rq3
    from .data import (
        build_actor_label_qc,
        build_event_history,
        build_group_database_provenance,
        build_thread_choice_sets,
    )
    from .study import filter_main_events, load_events

    raw = load_events(config.processed_dir)
    round_map = rq3.derive_round_timestamps(
        raw,
        round_start_time=config.rq3.round_start_time,
        first_experiment_round=config.rq3.first_experiment_round,
        verify_from=config.start_time,
    )
    stamped = rq3.timestamp_reactions(raw, round_map, placement=rq3.PRIMARY_PLACEMENT)
    coverage = rq3.reaction_coverage(raw, stamped)
    main = filter_main_events(
        stamped,
        start_time=config.start_time,
        end_time=config.end_time,
        analysis_set=config.analysis_set,
    )
    main = main.drop(columns=[column for column in ("text",) if column in main.columns])
    main["group"] = main["group"].astype(str)
    event_history = build_event_history(config)
    choice_sets = build_thread_choice_sets(config, event_history=event_history)
    choice_sets.attrs.pop("attrition_events", None)
    panel = rq3.build_sanction_panel(choice_sets, event_history)
    sets_by_group = (
        choice_sets.groupby("group")["choice_set_id"].nunique() if not choice_sets.empty else pd.Series(dtype=int)
    )
    signature = pd.DataFrame(
        [
            {
                "analysis_stage": "rq3",
                "panel_sha256": rq3.panel_signature(panel),
                "implementation_sha256": _rq3_implementation_hash(config),
                "panel_rows": int(len(panel)),
                "panel_rq1_rows": int(panel["in_rq1_analysis_set"].sum()) if not panel.empty else 0,
                "rq1_choice_sets_g1": int(sets_by_group.get("G1", 0)),
                "rq1_choice_sets_g2": int(sets_by_group.get("G2", 0)),
                "round_start_time": config.rq3.round_start_time,
                "first_experiment_round": int(config.rq3.first_experiment_round),
            }
        ]
    )
    expected_days = analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"]
    actor_qc = build_actor_label_qc(config)
    provenance = build_group_database_provenance(config)
    integrity = data_integrity_checks(
        event_history,
        choice_sets,
        expected_experiment_days=expected_days,
        actor_label_qc=actor_qc,
        group_database_provenance=provenance,
    )
    return {
        "rq3_round_timestamps": round_map,
        "rq3_reaction_coverage": coverage,
        "rq3_event_history": main.reset_index(drop=True),
        "rq3_sanction_panel": panel,
        "rq3_panel_signature": signature,
        "rq3_data_integrity_checks": integrity,
        "rq3_actor_label_qc": actor_qc,
        "rq3_group_database_provenance": provenance,
    }


def run_prepare_rq3(config: AnalysisConfig, config_path: Path) -> dict[str, pd.DataFrame]:
    """Timestamp reactions from rounds and build the RQ3 sanction panel."""

    _rq3_stage_guard(config, "prepare-rq3")
    _directories(config)
    _invalidate_rq3_artifacts(config, stage="prepare")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    snapshot = capture_source_snapshot(source_paths(config, config_path))
    tables = _rq3_prepare_frames(config)
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, tables)
    assert_source_snapshot_unchanged(snapshot)
    manifest = write_manifest(config, config_path, artifacts)
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {"stage": "prepare-rq3", "status": "completed", "run_id": config.run_id},
        config.run_dir / "status.json",
    )
    return tables


def _rq3_human_reactions(event_history: pd.DataFrame) -> pd.DataFrame:
    from . import rq3

    reactions = event_history[
        event_history["event_type"].isin(rq3.REACTION_EVENTS)
        & event_history["actor_type"].astype(str).str.upper().eq("H")
    ].copy()
    reactions["event_timestamp"] = pd.to_datetime(reactions["event_timestamp"], utc=True).dt.tz_convert(
        "Europe/Belgrade"
    )
    reactions["group"] = reactions["group"].astype(str)
    return reactions


def _rq3_estimate_frames(
    config: AnalysisConfig,
    prepared: Mapping[str, pd.DataFrame],
    *,
    bootstrap_reps: int,
) -> dict[str, pd.DataFrame]:
    from . import rq3

    panel = prepared["rq3_sanction_panel"].copy()
    panel["event_timestamp"] = pd.to_datetime(panel["event_timestamp"], utc=True).dt.tz_convert("Europe/Belgrade")
    panel["group"] = panel["group"].astype(str)
    history = prepared["rq3_event_history"].copy()
    history["event_timestamp"] = pd.to_datetime(history["event_timestamp"], utc=True).dt.tz_convert(
        "Europe/Belgrade"
    )
    reactions = _rq3_human_reactions(history)
    observation_end = rq3.observation_end_by_group(history)
    valid_days = rq3.valid_days_for(config.analysis_set)
    horizons = tuple(int(value) for value in config.rq3.horizons_hours)
    metrics = rq3.default_metrics(horizons)

    panels: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    contrasts: list[pd.DataFrame] = []
    for placement in rq3.PLACEMENTS:
        placed = reactions.copy()
        if placement == "round_end":
            placed["event_timestamp"] = placed["event_timestamp"] + pd.Timedelta(hours=rq3.ROUND_HOURS)
        windowed = rq3.add_vote_windows(
            panel,
            placed,
            horizons=horizons,
            observation_end=observation_end,
            valid_days=valid_days,
            placement=placement,
        )
        panels.append(windowed)
        summaries.append(rq3.summarize_groups(windowed, metrics=metrics))
        contrasts.append(
            rq3.bootstrap_contrasts(windowed, metrics=metrics, reps=bootstrap_reps, seed=config.seed)
        )
    vote_panel = pd.concat(panels, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    contrast = pd.concat(contrasts, ignore_index=True)
    sensitivity = pd.concat(
        [rq3.resolution_sensitivity(panels[0], alternative, metrics=metrics) for alternative in panels[1:]],
        ignore_index=True,
    )
    timing = rq3.downvote_timing(panel, reactions)
    from .data import _valid_contents

    contents = _valid_contents(config)
    unconditional = rq3.unconditional_downvote_rates(contents, reactions, panel)
    primary = rq3.build_primary_estimand(
        summary,
        contrast,
        start_time=config.start_time,
        end_time=config.end_time,
        horizon_hours=horizons[0],
    )
    vote_source = (
        reactions.groupby(["group", "event_type", "actor_type"]).size().rename("rows").reset_index()
    )
    return {
        "rq3_vote_panel": vote_panel,
        "rq3_vote_source": vote_source,
        "rq3_downvote_timing": timing,
        "rq3_unconditional_downvotes": unconditional,
        "rq3_group_summary": summary,
        "rq3_bootstrap_contrasts": contrast,
        "rq3_resolution_sensitivity": sensitivity,
        "rq3_primary_estimand": primary,
    }


def run_estimate_rq3(
    config: AnalysisConfig,
    config_path: Path,
    *,
    bootstrap_reps: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute windowed human-downvote outcomes, contrasts, and sensitivities."""

    _rq3_stage_guard(config, "estimate-rq3")
    _directories(config)
    _invalidate_rq3_artifacts(config, stage="estimate")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    snapshot = capture_source_snapshot(source_paths(config, config_path))
    prepared = _read_required_tables(config, sorted(RQ3_PREPARE_TABLE_NAMES))
    reps = int(bootstrap_reps or config.rq3.bootstrap_reps)
    tables = _rq3_estimate_frames(config, prepared, bootstrap_reps=reps)
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, tables)
    assert_source_snapshot_unchanged(snapshot)
    prepared_paths = [
        path
        for name in RQ3_PREPARE_TABLE_NAMES
        for path in (config.tables_dir / f"{name}.parquet", config.tables_dir / f"{name}.csv")
    ]
    manifest = write_manifest(config, config_path, [*prepared_paths, *artifacts])
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {"stage": "estimate-rq3", "status": "completed", "run_id": config.run_id, "bootstrap_reps": reps},
        config.run_dir / "status.json",
    )
    return tables


def _write_rq3_result_report(
    config: AnalysisConfig,
    *,
    primary: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    readiness: pd.DataFrame,
    timing: pd.DataFrame | None = None,
    unconditional: pd.DataFrame | None = None,
) -> Path:
    report_path = config.reports_dir / "rq3_result_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    release_passed = bool(not readiness.empty and readiness["passed"].fillna(False).astype(bool).all())
    primary_columns = ["group", "condition", "estimate", "ci_low", "ci_high", "numerator", "denominator_n", "p_value"]
    primary_table = primary[primary_columns].to_markdown(index=False) if not primary.empty else "No estimand."
    focus = summary[
        summary["subset"].eq("rq1_analysis_set")
        & summary["reaction_placement"].eq("round_start")
        & summary["metric"].str.startswith("target_received_human_downvote")
    ]
    focus_table = (
        focus[["group", "dyad", "metric", "estimate", "numerator", "denominator_n", "ci_low", "ci_high"]]
        .to_markdown(index=False)
        if not focus.empty
        else "None."
    )
    contrast_focus = contrasts[
        contrasts["subset"].isin(["rq1_analysis_set", "fully_observed_window", "all_valid_targets"])
        & contrasts["reaction_placement"].eq("round_start")
        & contrasts["metric"].str.startswith("target_received_human_downvote")
    ]
    contrast_table = (
        contrast_focus[["subset", "dyad", "metric", "estimate", "ci_low", "ci_high", "p_value", "p_value_test"]]
        .to_markdown(index=False)
        if not contrast_focus.empty
        else "None."
    )
    sens_focus = sensitivity[sensitivity["metric"].str.startswith("target_received_human_downvote")]
    sens_columns = ["comparison", "subset", "dyad", "metric", "rows", "rows_reclassified", "contrast_round_start", "contrast_round_end"]
    sens_table = (
        sens_focus[[column for column in sens_columns if column in sens_focus.columns]].to_markdown(index=False)
        if not sens_focus.empty
        else "None."
    )
    bracket = summary[
        summary["subset"].eq("rq1_analysis_set")
        & summary["metric"].eq("target_received_human_downvote_24h")
    ]
    bracket_table = (
        bracket[["reaction_placement", "group", "dyad", "estimate", "numerator", "denominator_n", "ci_low", "ci_high"]]
        .sort_values(["dyad", "reaction_placement", "group"])
        .to_markdown(index=False)
        if not bracket.empty
        else "None."
    )
    timing_table = timing.to_markdown(index=False) if timing is not None and not timing.empty else "None."
    unconditional_table = (
        unconditional[["split", "group", "author_type", "contents_n", "downvoted_n", "rate", "ci_low", "ci_high"]].to_markdown(index=False)
        if unconditional is not None and not unconditional.empty
        else "None."
    )
    failed = readiness[~readiness["passed"].fillna(False).astype(bool)] if not readiness.empty else readiness
    failed_text = (
        "None."
        if failed.empty
        else "\n".join(f"- `{row.gate}`: {row.observed}" for row in failed.itertuples(index=False))
    )
    gates_table = readiness[["gate", "passed", "observed"]].to_markdown(index=False) if not readiness.empty else "None."
    text = f"""# RQ3 windowed human-downvote report

- Run ID: `{config.run_id}`
- Release status: `{'passed' if release_passed else 'failed'}`
- Analysis window: `{config.start_time}` to `{config.end_time or 'snapshot extraction'}`
- Unit: selected same-thread target of a human reply; primary subset = RQ1 analysis set
- Reaction time resolution: one platform round = one hour; primary placement = round start
- Inference boundary: one community per condition; G2-G1 is an ecological contrast

## Primary estimand

{primary_table}

## Windowed downvote proportions (RQ1 analysis set, round start)

{focus_table}

## Contrasts (G2 minus G1, participant-clustered bootstrap)

{contrast_table}

## Placement bracket: 24 h downvote of the selected target by reaction placement (RQ1 analysis set)

{bracket_table}

## Resolution sensitivity (round start versus alternatives)

{sens_table}

## Timing of human downvotes on selected targets relative to the reply round

{timing_table}

## Unconditional: valid contributions with at least one human downvote (no window)

{unconditional_table}

## Release gates

{gates_table}

## Failed release gates

{failed_text}
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def run_validate_rq3(config: AnalysisConfig, config_path: Path) -> dict[str, pd.DataFrame]:
    """Reproduce the RQ3 panel from sources and apply fail-closed release gates."""

    from .validation import build_rq3_readiness

    _rq3_stage_guard(config, "validate-rq3")
    _directories(config)
    _invalidate_rq3_artifacts(config, stage="validate")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    snapshot = capture_source_snapshot(source_paths(config, config_path))
    stored = _read_required_tables(config, sorted(RQ3_PREPARE_TABLE_NAMES | RQ3_ESTIMATE_TABLE_NAMES))
    rebuilt = _rq3_prepare_frames(config)
    rebuilt_sha = str(rebuilt["rq3_panel_signature"]["panel_sha256"].iloc[0])
    stored_sha = str(stored["rq3_panel_signature"]["panel_sha256"].iloc[0])
    reproduction = pd.DataFrame(
        [
            {
                "check": "rq3_panel_reproduces_from_sources",
                "passed": rebuilt_sha == stored_sha,
                "observed": f"stored={stored_sha}; rebuilt={rebuilt_sha}",
                "required": "identical panel hashes",
            }
        ]
    )
    expected_reps = int(config.rq3.bootstrap_reps)
    readiness = build_rq3_readiness(
        integrity=rebuilt["rq3_data_integrity_checks"],
        round_map=rebuilt["rq3_round_timestamps"],
        reaction_coverage=rebuilt["rq3_reaction_coverage"],
        vote_source=stored["rq3_vote_source"],
        panel=stored["rq3_sanction_panel"],
        panel_signature=stored["rq3_panel_signature"],
        rebuilt_panel_sha256=rebuilt_sha,
        primary_estimand=stored["rq3_primary_estimand"],
        contrasts=stored["rq3_bootstrap_contrasts"],
        resolution_sensitivity=stored["rq3_resolution_sensitivity"],
        expected_bootstrap_reps=expected_reps,
    )
    tables = {"rq3_panel_reproduction": reproduction, "rq3_publication_readiness": readiness}
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, tables)
    release_passed = bool(not readiness.empty and readiness["passed"].fillna(False).astype(bool).all())
    write_json(
        {"analysis_stage": "rq3", "release_passed": release_passed, "gates": readiness.to_dict("records")},
        config.outputs_dir / "validation-gates.json",
    )
    report = _write_rq3_result_report(
        config,
        primary=stored["rq3_primary_estimand"],
        summary=stored["rq3_group_summary"],
        contrasts=stored["rq3_bootstrap_contrasts"],
        sensitivity=stored["rq3_resolution_sensitivity"],
        readiness=readiness,
        timing=stored["rq3_downvote_timing"],
        unconditional=stored["rq3_unconditional_downvotes"],
    )
    assert_source_snapshot_unchanged(snapshot)
    existing = [
        path
        for name in (RQ3_PREPARE_TABLE_NAMES | RQ3_ESTIMATE_TABLE_NAMES)
        for path in (config.tables_dir / f"{name}.parquet", config.tables_dir / f"{name}.csv")
    ]
    manifest = write_manifest(config, config_path, [*existing, *artifacts, report])
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {
            "stage": "validate-rq3",
            "status": "completed" if release_passed else "failed_release_gates",
            "run_id": config.run_id,
            "release_passed": release_passed,
        },
        config.run_dir / "status.json",
    )
    return tables


def run_report_rq3(config: AnalysisConfig, config_path: Path) -> Path:
    """Re-render the RQ3 report from stored tables."""

    _rq3_stage_guard(config, "report-rq3")
    stored = _read_required_tables(
        config,
        ["rq3_primary_estimand", "rq3_group_summary", "rq3_bootstrap_contrasts", "rq3_resolution_sensitivity"],
    )
    readiness = _read_optional_result(config, "rq3_publication_readiness")
    return _write_rq3_result_report(
        config,
        primary=stored["rq3_primary_estimand"],
        summary=stored["rq3_group_summary"],
        contrasts=stored["rq3_bootstrap_contrasts"],
        sensitivity=stored["rq3_resolution_sensitivity"],
        readiness=readiness,
        timing=_read_optional_result(config, "rq3_downvote_timing"),
        unconditional=_read_optional_result(config, "rq3_unconditional_downvotes"),
    )


# ---------------------------------------------------------------------------
# RQ2: directional prior contact with the same AI account
# ---------------------------------------------------------------------------


def _rq2_stage_guard(config: AnalysisConfig, stage: str) -> None:
    if config.analysis_stage != "rq2_m2":
        raise ValueError(f"{stage} requires effective analysis_stage: rq2_m2")


def _rq2_table_paths(config: AnalysisConfig, names: Iterable[str]) -> list[Path]:
    return [
        path
        for name in names
        for path in (
            config.tables_dir / f"{name}.parquet",
            config.tables_dir / f"{name}.csv",
        )
        if path.exists()
    ]


def _invalidate_rq2_artifacts(config: AnalysisConfig, *, stage: str) -> None:
    """Remove stale RQ2 surfaces while retaining resumable signed posteriors."""

    if stage not in {"prepare", "fit", "validate"}:
        raise ValueError(f"Unknown RQ2 invalidation stage {stage!r}")
    report = config.reports_dir / "rq2_result_report.md"
    report.unlink(missing_ok=True)
    report.with_suffix(report.suffix + ".manifest.json").unlink(missing_ok=True)
    prepare_names = set(RQ2_PREPARE_TABLE_NAMES)
    fit_names = set(RQ2_REQUIRED_FIT_TABLES)
    validation_names = set(RQ2_VALIDATION_TABLE_NAMES)
    if stage == "prepare":
        remove_names = prepare_names | fit_names | validation_names
    elif stage == "fit":
        remove_names = (fit_names | validation_names) - prepare_names
    else:
        remove_names = validation_names
    for path in _rq2_table_paths(config, remove_names):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".manifest.json").unlink(missing_ok=True)
    (config.run_dir / "manifest.json").unlink(missing_ok=True)
    (config.outputs_dir / "validation-gates.json").unlink(missing_ok=True)
    if stage in {"prepare", "fit"}:
        (config.outputs_dir / BUNDLE_FILENAME).unlink(missing_ok=True)
        (config.run_dir / "status.json").unlink(missing_ok=True)


def _rq2_model_data(model_frame: pd.DataFrame, term_manifest: pd.DataFrame) -> Any:
    from .model import prepare_model_data

    available_terms = included_terms_for_model("RQ2", term_manifest)
    numeric_terms = tuple(
        term for term in available_terms if term != "candidate_target_is_ai"
    )
    return prepare_model_data(
        model_frame,
        numeric_terms,
        ai_term="candidate_target_is_ai",
    )


def _rq2_frame_hash(frame: pd.DataFrame) -> str:
    """Hash an ordered RQ2 table independently of its on-disk format."""

    payload = frame.to_json(
        orient="split",
        date_format="iso",
        date_unit="ns",
        double_precision=15,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rq2_prepare_frames(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    from .rq2 import build_history_support

    data_tables = prepare_data(config)
    model_frame, term_manifest = build_model_frame(
        data_tables["network_history"], model_name="RQ2"
    )
    model_data = _rq2_model_data(model_frame, term_manifest)
    support = build_history_support(data_tables["network_history"])
    signature = pd.DataFrame(
        [
            {
                "analysis_stage": "rq2_m2",
                "model": "RQ2",
                "model_data_sha256": _prepared_data_hash(model_data),
                "model_frame_sha256": _rq2_frame_hash(model_frame),
                "term_manifest_sha256": _rq2_frame_hash(term_manifest),
                "history_support_sha256": _rq2_frame_hash(support),
                "implementation_sha256": _model_implementation_hash(config),
                "choice_sets": model_data.n_choices,
                "candidate_rows": int(model_data.mask.sum()),
                "participants": len(model_data.actor_labels),
                "terms": ",".join(model_data.terms),
            }
        ]
    )
    expected_days = analysis_set_manifest(config.analysis_set)[
        "analysis_set_valid_days"
    ]
    integrity = data_integrity_checks(
        data_tables["event_history"],
        data_tables["thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=data_tables["actor_label_qc"],
        group_database_provenance=data_tables["group_database_provenance"],
    )
    return {
        "rq2_event_history": data_tables["event_history"],
        "rq2_thread_choice_sets": data_tables["thread_choice_sets"],
        "rq2_network_history": data_tables["network_history"],
        "rq2_source_availability": data_tables["source_availability"],
        "rq2_choice_set_attrition": data_tables["choice_set_attrition"],
        "rq2_risk_set_composition": data_tables["risk_set_composition"],
        "rq2_actor_label_qc": data_tables["actor_label_qc"],
        "rq2_account_universe": data_tables["account_universe"],
        "rq2_group_database_provenance": data_tables[
            "group_database_provenance"
        ],
        "rq2_history_support": support,
        "rq2_model_frame": model_frame,
        "rq2_model_term_manifest": term_manifest,
        "rq2_analysis_data_signature": signature,
        "rq2_data_integrity_checks": integrity,
    }


def run_prepare_rq2(
    config: AnalysisConfig,
    config_path: Path,
) -> dict[str, pd.DataFrame]:
    """Build strict directional histories and the maintained RQ2 model frame."""

    _rq2_stage_guard(config, "prepare-rq2")
    _directories(config)
    _invalidate_rq2_artifacts(config, stage="prepare")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    snapshot = capture_source_snapshot(source_paths(config, config_path))
    tables = _rq2_prepare_frames(config)
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, tables)
    manifest = write_manifest(config, config_path, artifacts)
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {
            "stage": "prepare-rq2",
            "analysis_stage": "rq2_m2",
            "status": "completed",
            "run_id": config.run_id,
        },
        config.run_dir / "status.json",
    )
    return tables


def run_fit_rq2(
    config: AnalysisConfig,
    config_path: Path,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    resume: bool = True,
    estimand_draw_chunk_size: int = 64,
) -> dict[str, pd.DataFrame]:
    """Fit the AI-specific, condition-interacted RQ2 core model."""

    from .model import summarize_fit
    from .rq2 import add_estimand_denominators, standardized_estimands

    _rq2_stage_guard(config, "fit-rq2")
    _directories(config)
    _invalidate_rq2_artifacts(config, stage="fit")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    stable_prepare_names = {
        "rq2_history_support",
        "rq2_model_frame",
        "rq2_model_term_manifest",
    }
    snapshot = capture_source_snapshot(
        [
            *source_paths(config, config_path),
            *_rq2_table_paths(config, stable_prepare_names),
        ]
    )
    stored = _read_required_tables(
        config,
        [
            "rq2_analysis_data_signature",
            "rq2_history_support",
            "rq2_model_frame",
            "rq2_model_term_manifest",
        ],
    )
    model_frame = stored["rq2_model_frame"]
    term_manifest = stored["rq2_model_term_manifest"]
    data = _rq2_model_data(model_frame, term_manifest)
    signature_rows = stored["rq2_analysis_data_signature"]
    if len(signature_rows) != 1:
        raise ValueError("RQ2 preparation requires exactly one analysis signature")
    prepared_signature = signature_rows.iloc[0]
    preflight = {
        "analysis_stage": str(prepared_signature.get("analysis_stage"))
        == "rq2_m2",
        "model": str(prepared_signature.get("model")) == "RQ2",
        "model_data_sha256": str(prepared_signature.get("model_data_sha256"))
        == _prepared_data_hash(data),
        "model_frame_sha256": str(prepared_signature.get("model_frame_sha256"))
        == _rq2_frame_hash(model_frame),
        "term_manifest_sha256": str(
            prepared_signature.get("term_manifest_sha256")
        )
        == _rq2_frame_hash(term_manifest),
        "history_support_sha256": str(
            prepared_signature.get("history_support_sha256")
        )
        == _rq2_frame_hash(stored["rq2_history_support"]),
        "implementation_sha256": str(
            prepared_signature.get("implementation_sha256")
        )
        == _model_implementation_hash(config),
    }
    if not all(preflight.values()):
        failed = sorted(name for name, passed in preflight.items() if not passed)
        raise ValueError(
            "RQ2 prepared inputs failed the fit preflight: " + ", ".join(failed)
        )
    expected_draws = int(draws or config.sampling.draws)
    expected_tune = int(tune or config.sampling.tune)
    expected_chains = int(chains or config.sampling.chains)
    if expected_chains > 4:
        raise ValueError("RQ2 fitting is limited to at most four chains")

    model_path = config.models_dir / "M2_RQ2.nc"
    idata, signature, diagnostic = _fit_with_diagnostic_extension(
        config,
        data,
        "RQ2",
        model_path,
        role="rq2_main",
        draws=expected_draws,
        tune=expected_tune,
        chains=expected_chains,
        seed_offset=0,
        resume=resume,
    )
    try:
        parameters = summarize_fit(idata)
        parameters.insert(0, "model", "RQ2")
        estimands = standardized_estimands(
            idata,
            data,
            term_manifest,
            draw_chunk_size=estimand_draw_chunk_size,
        )
        estimands = add_estimand_denominators(estimands, data, config)
        influence = {
            unit: grouped_choice_psis(idata, model_frame, unit=unit)
            for unit in ("participant", "account", "thread")
        }
    finally:
        close_idata = getattr(idata, "close", None)
        if callable(close_idata):
            close_idata()

    diagnostics = pd.DataFrame(
        [{"model": "RQ2", "role": "rq2_main", **diagnostic}]
    )
    signature_table = pd.DataFrame([_signature_row(model_path, signature, data)])
    output_tables = {
        "rq2_model_parameter_summary": parameters,
        "rq2_primary_estimands": estimands,
        "rq2_model_diagnostics": diagnostics,
        "rq2_influence_participant": influence["participant"],
        "rq2_influence_account": influence["account"],
        "rq2_influence_thread": influence["thread"],
        "fit_run_signature": signature_table,
    }
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, output_tables)
    bundle = write_fit_result_bundle(
        config,
        source_snapshot=snapshot,
        model_data_sha256=_prepared_data_hash(data),
        implementation_sha256=_model_implementation_hash(config),
        table_artifacts=artifacts,
        posterior_paths=[model_path],
        fit_request={
            "analysis_stage": "rq2_m2",
            "model": "RQ2",
            "draws": expected_draws,
            "tune": expected_tune,
            "chains": expected_chains,
            "estimand": "joint zero-to-one directional contact at 24 valid-clock hours",
            "estimand_draw_chunk_size": int(estimand_draw_chunk_size),
            "model_frame_sha256": _rq2_frame_hash(model_frame),
            "term_manifest_sha256": _rq2_frame_hash(term_manifest),
            "history_support_sha256": _rq2_frame_hash(
                stored["rq2_history_support"]
            ),
            "influence_units": ["participant", "account", "thread"],
        },
    )
    metadata[bundle] = {
        "artifact": "authenticated RQ2 core-fit result bundle",
        "denominator": estimands["denominator"].iloc[0],
    }
    assert_source_snapshot_unchanged(snapshot)
    manifest_artifacts = [
        *_rq2_table_paths(config, RQ2_PREPARE_TABLE_NAMES),
        *artifacts,
        model_path,
        model_path.with_suffix(model_path.suffix + ".fit.json"),
        bundle,
    ]
    manifest = write_manifest(config, config_path, manifest_artifacts)
    write_artifact_manifests(config, manifest, metadata)
    write_json(
        {
            "stage": "fit-rq2",
            "analysis_stage": "rq2_m2",
            "status": (
                "completed"
                if bool(diagnostic["passed"])
                else "completed_with_failed_diagnostics"
            ),
            "run_id": config.run_id,
            "resource_limit": "at most four chains; estimands processed in bounded draw chunks",
        },
        config.run_dir / "status.json",
    )
    return output_tables


def _rq2_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return _rq2_frame_hash(left.reset_index(drop=True)) == _rq2_frame_hash(
        right.reset_index(drop=True)
    )


def run_validate_rq2(
    config: AnalysisConfig,
    config_path: Path,
) -> dict[str, pd.DataFrame]:
    """Rebuild, authenticate, and validate the full-length RQ2 core fit."""

    from .rq2 import build_core_readiness, write_report

    _rq2_stage_guard(config, "validate-rq2")
    _directories(config)
    _invalidate_rq2_artifacts(config, stage="validate")
    assert_source_snapshot_unchanged(PROCESS_CODE_SNAPSHOT)
    snapshot = capture_source_snapshot(source_paths(config, config_path))
    stored = _read_required_tables(
        config,
        RQ2_PREPARE_TABLE_NAMES
        - {"rq2_analysis_data_signature", "rq2_data_integrity_checks"},
    )
    rebuilt = _rq2_prepare_frames(config)
    stored_data = _rq2_model_data(
        stored["rq2_model_frame"], stored["rq2_model_term_manifest"]
    )
    rebuilt_data = _rq2_model_data(
        rebuilt["rq2_model_frame"], rebuilt["rq2_model_term_manifest"]
    )
    stored_hash = _prepared_data_hash(stored_data)
    rebuilt_hash = _prepared_data_hash(rebuilt_data)
    model_data_reproduced = stored_hash == rebuilt_hash
    model_frame_reproduced = _rq2_frames_equal(
        stored["rq2_model_frame"], rebuilt["rq2_model_frame"]
    )
    support_reproduced = _rq2_frames_equal(
        stored["rq2_history_support"], rebuilt["rq2_history_support"]
    )
    term_manifest_reproduced = _rq2_frames_equal(
        stored["rq2_model_term_manifest"],
        rebuilt["rq2_model_term_manifest"],
    )
    expected_days = analysis_set_manifest(config.analysis_set)[
        "analysis_set_valid_days"
    ]
    integrity = data_integrity_checks(
        rebuilt["rq2_event_history"],
        rebuilt["rq2_thread_choice_sets"],
        expected_experiment_days=expected_days,
        actor_label_qc=rebuilt["rq2_actor_label_qc"],
        group_database_provenance=rebuilt["rq2_group_database_provenance"],
        network_history=stored["rq2_network_history"],
        recomputed_network_history=rebuilt["rq2_network_history"],
    )
    integrity = pd.concat(
        [
            integrity,
            pd.DataFrame(
                [
                    {
                        "check": "rq2_model_frame_reproduces_from_strict_history",
                        "passed": model_data_reproduced and model_frame_reproduced,
                        "observed": (
                            f"model_data_hash_match={model_data_reproduced}; "
                            f"full_frame_match={model_frame_reproduced}; "
                            f"stored={stored_hash}; rebuilt={rebuilt_hash}"
                        ),
                        "required": "identical full RQ2 frame and prepared model arrays",
                    },
                    {
                        "check": "rq2_history_support_reproduces",
                        "passed": support_reproduced,
                        "observed": f"identical={support_reproduced}",
                        "required": "identical AI-account support table",
                    },
                    {
                        "check": "rq2_term_manifest_reproduces",
                        "passed": term_manifest_reproduced,
                        "observed": f"identical={term_manifest_reproduced}",
                        "required": "identical RQ2 term definitions and training scalers",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    analysis_signature = rebuilt["rq2_analysis_data_signature"]
    implementation_hash = _model_implementation_hash(config)
    bundle_authentication = authenticate_fit_result_bundle(
        config,
        current_model_data_sha256=rebuilt_hash,
        current_implementation_sha256=implementation_hash,
        required_tables=RQ2_REQUIRED_FIT_TABLES,
    )
    bundle_authenticated = bool(
        len(bundle_authentication) == 1
        and bundle_authentication["authenticated"].fillna(False).astype(bool).all()
    )

    def fitted_result(name: str) -> pd.DataFrame:
        return _read_optional_result(config, name) if bundle_authenticated else pd.DataFrame()

    signatures = authenticate_fit_signatures(
        _read_optional_result(config, "fit_run_signature")
    )
    main = signatures[
        signatures.get("model", pd.Series(dtype=object)).astype(str).eq("RQ2")
        & signatures.get("role", pd.Series(dtype=object)).astype(str).eq("rq2_main")
        & signatures.get(
            "live_posterior_authenticated",
            pd.Series(False, index=signatures.index),
        )
        .fillna(False)
        .astype(bool)
    ]
    validated_influence: dict[str, pd.DataFrame]
    if bundle_authenticated and len(main) == 1:
        import arviz as az

        idata = az.from_netcdf(Path(str(main["posterior_path"].iloc[0])))
        try:
            validated_influence = {
                unit: grouped_choice_psis(
                    idata, stored["rq2_model_frame"], unit=unit
                )
                for unit in ("participant", "account", "thread")
            }
        finally:
            idata.close()
    else:
        validated_influence = {
            unit: pd.DataFrame() for unit in ("participant", "account", "thread")
        }
    readiness = build_core_readiness(
        integrity=integrity,
        model_data_reproduced=(
            model_data_reproduced
            and model_frame_reproduced
            and support_reproduced
            and term_manifest_reproduced
        ),
        analysis_signature=analysis_signature,
        signatures=signatures,
        bundle_authentication=bundle_authentication,
        term_manifest=rebuilt["rq2_model_term_manifest"],
        support=rebuilt["rq2_history_support"],
        diagnostics=fitted_result("rq2_model_diagnostics"),
        estimands=fitted_result("rq2_primary_estimands"),
        influence_tables=validated_influence,
    )
    validation_tables = {
        "rq2_data_integrity_checks": integrity,
        "rq2_analysis_data_signature": analysis_signature,
        "rq2_fit_result_bundle_authentication": bundle_authentication,
        "rq2_fit_signature_authentication": signatures,
        "rq2_influence_participant_validated": validated_influence[
            "participant"
        ],
        "rq2_influence_account_validated": validated_influence["account"],
        "rq2_influence_thread_validated": validated_influence["thread"],
        "rq2_core_readiness": readiness,
    }
    assert_source_snapshot_unchanged(snapshot)
    artifacts, metadata = _write_tables(config, validation_tables)
    report = write_report(
        config,
        estimands=fitted_result("rq2_primary_estimands"),
        support=rebuilt["rq2_history_support"],
        diagnostics=fitted_result("rq2_model_diagnostics"),
        influence_tables=validated_influence,
        readiness=readiness,
        signatures=signatures,
    )
    metadata[report] = {
        "artifact": "validated RQ2 core-fit report",
        "denominator": (
            fitted_result("rq2_primary_estimands")["denominator"].iloc[0]
            if not fitted_result("rq2_primary_estimands").empty
            else "no authenticated RQ2 estimand"
        ),
    }
    core_passed = bool(
        not readiness.empty and readiness["passed"].fillna(False).astype(bool).all()
    )
    write_json(
        {
            "analysis_stage": "rq2_m2",
            "core_fit_passed": core_passed,
            "publication_admission_ready": False,
            "publication_checks_pending": [
                "doubled-prior sensitivity",
                "one-to-two-contact transition sensitivity",
                "rolling-origin RQ2 versus M1 prediction",
                "exact deletion refits for every grouped PSIS flag",
            ],
            "gates": readiness.to_dict("records"),
        },
        config.outputs_dir / "validation-gates.json",
    )
    write_json(
        {
            "stage": "validate-rq2",
            "analysis_stage": "rq2_m2",
            "status": "completed" if core_passed else "failed_core_fit_gates",
            "run_id": config.run_id,
            "core_fit_passed": core_passed,
            "publication_admission_ready": False,
        },
        config.run_dir / "status.json",
    )
    assert_source_snapshot_unchanged(snapshot)
    manifest = _refresh_manifest(config, config_path, metadata)
    write_artifact_manifests(config, manifest, metadata)
    return validation_tables


def run_report_rq2(config: AnalysisConfig, config_path: Path) -> Path:
    """Re-render the RQ2 core-fit report from stored tables."""

    from .rq2 import write_report

    _rq2_stage_guard(config, "report-rq2")
    stored = _read_required_tables(
        config,
        [
            "rq2_primary_estimands",
            "rq2_history_support",
            "rq2_model_diagnostics",
            "rq2_influence_participant",
            "rq2_influence_account",
            "rq2_influence_thread",
        ],
    )
    readiness = _read_optional_result(config, "rq2_core_readiness")
    signatures = _read_optional_result(config, "rq2_fit_signature_authentication")
    if signatures.empty:
        signatures = _read_optional_result(config, "fit_run_signature")
    path = write_report(
        config,
        estimands=stored["rq2_primary_estimands"],
        support=stored["rq2_history_support"],
        diagnostics=stored["rq2_model_diagnostics"],
        influence_tables={
            unit: stored[f"rq2_influence_{unit}"]
            for unit in ("participant", "account", "thread")
        },
        readiness=readiness,
        signatures=signatures,
    )
    _refresh_manifest(
        config,
        config_path,
        {
            path: {
                "artifact": "RQ2 core-fit report",
                "denominator": stored["rq2_primary_estimands"]["denominator"].iloc[0],
            }
        },
    )
    return path
