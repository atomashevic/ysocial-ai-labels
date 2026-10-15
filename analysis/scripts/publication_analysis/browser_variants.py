"""Staged M1b/M1c browser-variant analysis.

The prepare stage authenticates the frozen admitted M1 run, preserves its
candidate rows and transformed core terms, and adds browser covariates on one
complete reply-level cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .browser import prepare_browser
from .browser_choice_covariates import (
    BROWSER_TERM_MAP,
    KEY_COLUMNS,
    RAW_BROWSER_COLUMNS,
    TERMINAL_SENSITIVITY_COLUMNS,
    TERMINAL_SENSITIVITY_TERM_MAP,
    add_standardized_browser_terms,
    add_terminal_sensitivity_terms,
    build_browser_choice_covariates,
    candidate_universe_sha256,
    matched_choice_set_ids,
    stable_frame_sha256,
)
from .config import AnalysisConfig, load_config
from .io import ensure_directories, read_table, sha256_file, source_record, write_json, write_table
from .model import (
    PreparedModelData,
    build_primary_pymc_model,
    fit_model,
    prepare_model_data,
    standardized_estimands,
    summarize_fit,
)
from .provenance import assert_source_snapshot_unchanged, capture_source_snapshot
from .validation import grouped_choice_psis


ADMITTED_PRIMARY_ESTIMAND_SHA256 = (
    "149ce85d47f3751d4d2bef7557cfde8edf03aff9d723c9e00984c7eb2dfa0db7"
)
EXPECTED_M1_CHOICE_SETS = 2_143
EXPECTED_M1_CANDIDATE_ROWS = 17_489
VARIANT_SCHEMA = "ysocial.rq1-browser-variants.v1"
PREPARE_TABLES = (
    "rq1_browser_m1a_input_authentication",
    "rq1_browser_choice_covariates",
    "rq1_browser_matched_model_frame",
    "rq1_browser_cohort_attrition",
    "rq1_browser_term_manifest",
    "rq1_browser_gate_status",
    "rq1_browser_data_signature",
)
FIT_TABLES = (
    "rq1_browser_fit_run_signature",
    "rq1_browser_model_diagnostics",
    "rq1_browser_parameter_summary",
    "rq1_browser_estimands",
    "rq1_browser_estimand_comparison",
    "rq1_browser_posterior_correlations",
    "rq1_browser_grouped_influence",
)
PRIMARY_MODEL_TERMS = tuple(BROWSER_TERM_MAP.values())
COUNT_BROWSER_TERMS = (
    BROWSER_TERM_MAP["candidate_visit_count"],
    BROWSER_TERM_MAP["same_author_other_visit_count"],
)
TIME_BROWSER_TERMS = (
    BROWSER_TERM_MAP["candidate_dom_presence_seconds"],
    BROWSER_TERM_MAP["same_author_other_dom_presence_seconds"],
)
TERMINAL_BROWSER_TERMS = (
    BROWSER_TERM_MAP["candidate_visit_count"],
    BROWSER_TERM_MAP["same_author_other_visit_count"],
    *TERMINAL_SENSITIVITY_TERM_MAP.values(),
)
MODEL_VARIANTS = {
    "M1b_browser_matched": (),
    "M1c_browser_adjusted": PRIMARY_MODEL_TERMS,
    "M1c_count_only_sensitivity": COUNT_BROWSER_TERMS,
    "M1c_time_only_sensitivity": TIME_BROWSER_TERMS,
    "M1c_terminal_excluded_sensitivity": TERMINAL_BROWSER_TERMS,
}


def _prepared_data_sha256(data: PreparedModelData) -> str:
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


def _records_unchanged(records: Iterable[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    changed: list[str] = []
    for saved in records:
        path = Path(str(saved.get("path", "")))
        current = source_record(path)
        if any(current.get(key) != saved.get(key) for key in ("path", "exists", "size_bytes", "sha256")):
            changed.append(str(path))
    return not changed, changed


def _frozen_table(run_dir: Path, name: str) -> pd.DataFrame:
    return read_table(run_dir / "tables" / name)


def _signature_terms(signature: pd.DataFrame) -> tuple[str, ...]:
    if len(signature) != 1 or "terms" not in signature.columns:
        raise ValueError("Frozen M1 analysis signature must contain one terms field")
    terms = tuple(term.strip() for term in str(signature.iloc[0]["terms"]).split(",") if term.strip())
    if not terms or terms[0] != "candidate_target_is_ai":
        raise ValueError("Frozen M1 term order is missing candidate_target_is_ai")
    return terms


def authenticate_frozen_m1a(
    config: AnalysisConfig,
    m1a_run_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Authenticate the frozen admitted M1 inputs used by browser variants."""

    run_dir = m1a_run_dir.expanduser().resolve()
    bundle_path = run_dir / "fit_result_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    tables = {
        name: _frozen_table(run_dir, name)
        for name in (
            "thread_choice_sets",
            "model_frame",
            "model_term_manifest",
            "analysis_data_signature",
            "event_history",
        )
    }
    signature = tables["analysis_data_signature"]
    terms = _signature_terms(signature)
    prepared = prepare_model_data(
        tables["model_frame"],
        tuple(term for term in terms if term != "candidate_target_is_ai"),
        ai_term="candidate_target_is_ai",
    )
    rebuilt_hash = _prepared_data_sha256(prepared)
    saved_hash = str(signature.iloc[0]["model_data_sha256"])
    bundle_hash = str(bundle.get("model_data_sha256", ""))
    source_ok, changed_sources = _records_unchanged(bundle.get("sources", []))
    m1_records = [
        record
        for record in bundle.get("posteriors", [])
        if Path(str(record.get("path", ""))).name == "M1.nc"
    ]
    posterior_ok, changed_posteriors = _records_unchanged(m1_records)
    fit_signature_records = [
        record.get("fit_signature", {}) for record in m1_records if isinstance(record, Mapping)
    ]
    fit_signature_ok, changed_fit_signatures = _records_unchanged(fit_signature_records)
    primary_path = config.project_root / "results" / "tables" / "rq1_primary_estimand.csv"
    primary_hash = sha256_file(primary_path) if primary_path.exists() else ""
    model_frame = tables["model_frame"]
    choice_sets = tables["thread_choice_sets"]
    reaction_manifest = tables["model_term_manifest"]
    excluded_reactions = reaction_manifest.loc[
        reaction_manifest["term"].isin(("z_content_reaction_status", "z_author_reaction_status")),
        "status",
    ].astype(str)
    checks = {
        "bundle_schema_valid": bundle.get("schema") == "ysocial.publication-fit-bundle.v1",
        "bundle_run_id_matches": str(bundle.get("run_id")) == run_dir.name,
        "signature_model_is_m1": len(signature) == 1 and str(signature.iloc[0].get("model")) == "M1",
        "choice_set_count_matches": model_frame["choice_set_id"].nunique() == EXPECTED_M1_CHOICE_SETS,
        "candidate_row_count_matches": len(model_frame) == EXPECTED_M1_CANDIDATE_ROWS,
        "choice_rows_match_model_frame": (
            len(choice_sets) == len(model_frame)
            and candidate_universe_sha256(choice_sets) == candidate_universe_sha256(model_frame)
        ),
        "model_data_hash_matches": rebuilt_hash == saved_hash == bundle_hash,
        "frozen_sources_unchanged": source_ok,
        "m1_posterior_unchanged": len(m1_records) == 1 and posterior_ok and fit_signature_ok,
        "primary_estimand_hash_matches": primary_hash == ADMITTED_PRIMARY_ESTIMAND_SHA256,
        "reaction_terms_excluded": len(excluded_reactions) == 2
        and excluded_reactions.str.startswith("excluded").all(),
    }
    authenticated = all(checks.values())
    authentication = pd.DataFrame(
        [
            {
                "m1a_run_dir": str(run_dir),
                "authenticated": authenticated,
                **checks,
                "choice_sets": int(model_frame["choice_set_id"].nunique()),
                "candidate_rows": int(len(model_frame)),
                "participants": int(model_frame[["group", "actor_id"]].drop_duplicates().shape[0]),
                "saved_model_data_sha256": saved_hash,
                "rebuilt_model_data_sha256": rebuilt_hash,
                "implementation_sha256": str(signature.iloc[0]["implementation_sha256"]),
                "primary_estimand_sha256": primary_hash,
                "changed_sources": ",".join(changed_sources),
                "changed_posteriors": ",".join(changed_posteriors),
                "changed_fit_signatures": ",".join(changed_fit_signatures),
            }
        ]
    )
    details = {
        "bundle": bundle,
        "terms": terms,
        "prepared_data": prepared,
        "candidate_universe_sha256": candidate_universe_sha256(model_frame),
        "primary_estimand_path": primary_path,
    }
    if not authenticated:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Frozen M1a authentication failed: {failed}")
    return tables, authentication, details


def _variant_implementation_sha256(config: AnalysisConfig) -> str:
    digest = hashlib.sha256()
    for name in (
        "browser_choice_covariates.py",
        "browser_variants.py",
        "browser.py",
        "config.py",
        "io.py",
        "model.py",
        "provenance.py",
    ):
        path = config.project_root / "scripts" / "publication_analysis" / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _variant_source_paths(
    config: AnalysisConfig,
    config_path: Path,
    m1a_run_dir: Path,
) -> list[Path]:
    paths = [
        config_path,
        m1a_run_dir / "fit_result_bundle.json",
        m1a_run_dir / "models" / "M1.nc",
    ]
    paths.extend(
        m1a_run_dir / "tables" / f"{name}.parquet"
        for name in (
            "thread_choice_sets",
            "model_frame",
            "model_term_manifest",
            "analysis_data_signature",
            "event_history",
        )
    )
    for name in ("session_events", "session_qc", "browser_interval_audit"):
        for suffix in (".parquet", ".csv"):
            candidate = config.browser_dir / f"{name}{suffix}"
            if candidate.exists():
                paths.append(candidate)
                break
    for name in (
        "browser_choice_covariates.py",
        "browser_variants.py",
        "browser.py",
        "config.py",
        "io.py",
        "model.py",
        "provenance.py",
    ):
        paths.append(config.project_root / "scripts" / "publication_analysis" / name)
    paths.append(config.project_root / "results" / "tables" / "rq1_primary_estimand.csv")
    return paths


def _validate_run_target(config: AnalysisConfig, m1a_run_dir: Path) -> Path:
    if not config.run_id.startswith("rq1_browser_"):
        raise ValueError("Browser variant run IDs must start with 'rq1_browser_'")
    if config.run_id == m1a_run_dir.resolve().name:
        raise ValueError("Browser variants cannot use the admitted M1a run ID")
    marker = config.run_dir / "browser_variant_run.json"
    expected = {
        "schema": VARIANT_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config.contract_sha256,
        "m1a_run_dir": str(m1a_run_dir.resolve()),
    }
    if marker.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError("Existing browser variant run has a different input contract")
    elif config.outputs_dir.exists() and any(config.outputs_dir.rglob("*")):
        raise RuntimeError("Existing unmarked output directory will not be overwritten")
    ensure_directories([config.run_dir, config.tables_dir])
    write_json(expected, marker)
    return marker


def _cohort_attrition(
    choice_sets: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    by_choice = covariates.groupby("choice_set_id", sort=False).agg(
        group=("group", "first"),
        participant_rows=("candidate_content_id", "size"),
        session_status=("browser_covariate_session_status", "first"),
        candidate_quality=("candidate_interval_quality", lambda values: ",".join(sorted(set(map(str, values))))),
        author_quality=("same_author_interval_quality", lambda values: ",".join(sorted(set(map(str, values))))),
        retained=("browser_choice_set_complete", "all"),
        browser_session_id=("browser_session_id", "first"),
    ).reset_index()

    def reason(row: pd.Series) -> str:
        if bool(row["retained"]):
            return "retained"
        combined = f"{row['candidate_quality']},{row['author_quality']}"
        if row["session_status"] == "invalid_reply_timestamp":
            return "invalid_reply_timestamp"
        if row["session_status"] != "observed":
            return "unresolved_session"
        if "missing_author_metadata" in combined:
            return "missing_author_metadata"
        if "unresolved_interval" in combined:
            return "unresolved_interval"
        return "incomplete_other"

    by_choice["cohort_status"] = by_choice.apply(reason, axis=1)
    actors = choice_sets[["choice_set_id", "actor_id"]].drop_duplicates("choice_set_id")
    by_choice = by_choice.merge(actors, on="choice_set_id", how="left", validate="one_to_one")
    rows = []
    for (group, status), part in by_choice.groupby(["group", "cohort_status"], sort=True):
        rows.append(
            {
                "group": group,
                "cohort_status": status,
                "choice_sets": int(len(part)),
                "candidate_rows": int(part["participant_rows"].sum()),
                "participants": int(part["actor_id"].nunique()),
                "browser_sessions": int(part["browser_session_id"].dropna().astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows)


def _tracking_pass(qc: pd.DataFrame) -> bool:
    selected = qc.loc[
        qc.get("metric", pd.Series(dtype=object)).astype(str).eq("all_tracking_gates_pass")
    ]
    if len(selected) != 1:
        return False
    value = selected.iloc[0].get("gate_pass")
    return bool(False if pd.isna(value) else value)


def _signature_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def _data_signature(
    full_frame: pd.DataFrame,
    matched_frame: pd.DataFrame,
    core_terms: Sequence[str],
    tracking_passed: bool,
    implementation_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    browser_terms = tuple(BROWSER_TERM_MAP.values())
    centered = matched_frame.loc[:, browser_terms].astype(float) - matched_frame.groupby(
        "choice_set_id", sort=False
    )[list(browser_terms)].transform("mean")
    centered_rank = int(np.linalg.matrix_rank(centered.to_numpy(dtype=float)))
    variation = {
        term: int(matched_frame.groupby("choice_set_id", sort=False)[term].nunique().gt(1).sum())
        for term in browser_terms
    }
    raw_correlations = matched_frame.loc[:, list(RAW_BROWSER_COLUMNS)].corr(method="spearman")
    full_hash = candidate_universe_sha256(full_frame)
    cohort_hash = candidate_universe_sha256(matched_frame)
    core_hash = stable_frame_sha256(
        matched_frame,
        (*KEY_COLUMNS, "candidate_author_id", "chosen", *core_terms),
    )
    browser_hash = stable_frame_sha256(
        matched_frame,
        (*KEY_COLUMNS, *RAW_BROWSER_COLUMNS, *browser_terms),
    )
    status = "ready_for_exploratory_fit" if tracking_passed else "exploratory_browser_data_quality_failed"
    values = {
        "analysis_status": status,
        "full_choice_sets": int(full_frame["choice_set_id"].nunique()),
        "full_candidate_rows": int(len(full_frame)),
        "matched_choice_sets": int(matched_frame["choice_set_id"].nunique()),
        "matched_candidate_rows": int(len(matched_frame)),
        "matched_participants": int(matched_frame[["group", "actor_id"]].drop_duplicates().shape[0]),
        "matched_sessions": int(matched_frame["browser_session_id"].dropna().astype(str).nunique()),
        "full_candidate_universe_sha256": full_hash,
        "matched_candidate_universe_sha256": cohort_hash,
        "m1b_core_arrays_input_sha256": core_hash,
        "m1c_core_arrays_input_sha256": core_hash,
        "browser_covariates_sha256": browser_hash,
        "variant_implementation_sha256": implementation_hash,
        "centered_browser_design_rank": centered_rank,
        "centered_browser_design_columns": len(browser_terms),
        "tracking_quality_passed": tracking_passed,
    }
    verified = matched_frame.loc[
        matched_frame["browser_verified_choice_set_complete"].astype(bool)
    ]
    for group in ("G1", "G2"):
        part = verified[verified["group"].astype(str).eq(group)]
        values[f"verified_choice_sets_{group}"] = int(part["choice_set_id"].nunique())
        values[f"verified_candidate_rows_{group}"] = int(len(part))
    rows: list[dict[str, Any]] = [
        {"metric": key, "value": _signature_value(value), "details": ""}
        for key, value in values.items()
    ]
    for term, count in variation.items():
        rows.append(
            {
                "metric": f"within_choice_variation::{term}",
                "value": _signature_value(count),
                "details": "choice_sets",
            }
        )
    for left in RAW_BROWSER_COLUMNS:
        for right in RAW_BROWSER_COLUMNS:
            if RAW_BROWSER_COLUMNS.index(right) <= RAW_BROWSER_COLUMNS.index(left):
                continue
            rows.append(
                {
                    "metric": f"spearman::{left}::{right}",
                    "value": _signature_value(float(raw_correlations.loc[left, right])),
                    "details": "matched_candidate_rows",
                }
            )
    for raw_column in RAW_BROWSER_COLUMNS:
        rows.extend(
            [
                {
                    "metric": f"zero_rows::{raw_column}",
                    "value": _signature_value(
                        int(pd.to_numeric(matched_frame[raw_column]).eq(0).sum())
                    ),
                    "details": "matched_candidate_rows",
                },
                {
                    "metric": f"missing_rows::{raw_column}",
                    "value": _signature_value(int(matched_frame[raw_column].isna().sum())),
                    "details": "matched_candidate_rows",
                },
            ]
        )
    return pd.DataFrame(rows), values


def run_prepare_choice(
    config: AnalysisConfig,
    config_path: Path,
    m1a_run_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Prepare and sign the fixed M1b/M1c browser cohort."""

    run_marker = _validate_run_target(config, m1a_run_dir)
    frozen, authentication, m1a_details = authenticate_frozen_m1a(config, m1a_run_dir)
    source_paths = _variant_source_paths(config, config_path, m1a_run_dir.resolve())
    source_snapshot = capture_source_snapshot(source_paths)
    browser = prepare_browser(
        config,
        frozen["event_history"],
        frozen["thread_choice_sets"],
    )
    covariates = build_browser_choice_covariates(
        frozen["thread_choice_sets"],
        browser["browser_opportunity_episodes"],
        browser["browser_linked_actions"],
    )
    selected_ids = set(matched_choice_set_ids(covariates))
    matched_base = frozen["model_frame"].loc[
        frozen["model_frame"]["choice_set_id"].astype(str).isin(selected_ids)
    ].copy()
    selected_covariates = covariates.loc[
        covariates["choice_set_id"].astype(str).isin(selected_ids)
    ].copy()
    if matched_base.empty:
        raise RuntimeError("No browser-complete M1 choice sets remain")
    matched_base["_variant_row_order"] = np.arange(len(matched_base))
    feature_columns = [
        *KEY_COLUMNS,
        "browser_session_id",
        "backend_reply_timestamp_utc",
        "browser_covariate_session_status",
        "browser_link_ambiguous",
        "candidate_interval_quality",
        "same_author_interval_quality",
        "browser_covariate_row_complete",
        "browser_choice_set_complete",
        "browser_verified_row_complete",
        "browser_verified_choice_set_complete",
        *RAW_BROWSER_COLUMNS,
        *TERMINAL_SENSITIVITY_COLUMNS,
    ]
    matched = matched_base.merge(
        selected_covariates[feature_columns],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_variant_row_order", kind="stable").drop(columns="_variant_row_order")
    expected_counts = frozen["model_frame"].groupby("choice_set_id", sort=False).size()
    observed_counts = matched.groupby("choice_set_id", sort=False).size()
    if not observed_counts.equals(expected_counts.loc[observed_counts.index]):
        raise RuntimeError("Browser cohort changed candidate counts within a retained choice set")
    if not matched.groupby("choice_set_id")["chosen"].sum().eq(1).all():
        raise RuntimeError("Matched browser cohort requires exactly one chosen candidate per reply")
    matched, browser_term_manifest = add_standardized_browser_terms(matched)
    matched, terminal_term_manifest = add_terminal_sensitivity_terms(matched)
    browser_term_manifest = pd.concat(
        [browser_term_manifest, terminal_term_manifest], ignore_index=True
    )
    core_terms = tuple(m1a_details["terms"])
    implementation_hash = _variant_implementation_sha256(config)
    tracking_qc = browser["tracking_linkage_qc"].copy()
    tracking_passed = _tracking_pass(tracking_qc)
    analysis_status = (
        "ready_for_exploratory_fit" if tracking_passed else "exploratory_browser_data_quality_failed"
    )
    tracking_qc["analysis_status"] = analysis_status
    data_signature, signature_values = _data_signature(
        frozen["model_frame"],
        matched,
        core_terms,
        tracking_passed,
        implementation_hash,
    )
    attrition = _cohort_attrition(frozen["thread_choice_sets"], covariates)
    tables = {
        "rq1_browser_m1a_input_authentication": authentication,
        "rq1_browser_choice_covariates": covariates,
        "rq1_browser_matched_model_frame": matched,
        "rq1_browser_cohort_attrition": attrition,
        "rq1_browser_term_manifest": browser_term_manifest,
        "rq1_browser_gate_status": tracking_qc,
        "rq1_browser_data_signature": data_signature,
    }
    assert_source_snapshot_unchanged(source_snapshot)
    for name, frame in tables.items():
        write_table(frame, config.tables_dir / name)
    write_json(
        {
            "schema": VARIANT_SCHEMA,
            "run_id": config.run_id,
            "m1a_run_dir": str(m1a_run_dir.resolve()),
            "sources": source_snapshot,
            "variant_implementation_sha256": implementation_hash,
            "m1a_terms": list(core_terms),
            "data_signature": signature_values,
        },
        config.outputs_dir / "browser_variant_source_snapshot.json",
    )
    assert_source_snapshot_unchanged(source_snapshot)
    if sha256_file(m1a_details["primary_estimand_path"]) != ADMITTED_PRIMARY_ESTIMAND_SHA256:
        raise RuntimeError("Admitted M1a estimand changed during browser preparation")
    write_json(
        {
            "stage": "prepare-choice",
            "status": "completed",
            "analysis_status": analysis_status,
            "run_id": config.run_id,
            "m1a_run_dir": str(m1a_run_dir.resolve()),
            **signature_values,
            "m1a_terms": list(core_terms),
            "expected_tables": list(PREPARE_TABLES),
        },
        config.run_dir / "status.json",
    )
    return tables


def _load_prepare_status(config: AnalysisConfig) -> dict[str, Any]:
    path = config.run_dir / "status.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("stage") not in {
        "prepare-choice",
        "fit-choice",
        "validate-choice",
        "report",
    }:
        raise RuntimeError("Browser variant preparation is not complete")
    if value.get("run_id") != config.run_id:
        raise RuntimeError("Browser variant status belongs to another run")
    return value


def _load_marker(config: AnalysisConfig) -> dict[str, Any]:
    marker = json.loads((config.run_dir / "browser_variant_run.json").read_text(encoding="utf-8"))
    if marker.get("schema") != VARIANT_SCHEMA or marker.get("run_id") != config.run_id:
        raise RuntimeError("Invalid browser variant run marker")
    return marker


def _fit_source_paths(config: AnalysisConfig) -> list[Path]:
    paths = [
        config.project_root / "analysis.yaml",
        config.run_dir / "browser_variant_run.json",
        config.outputs_dir / "browser_variant_source_snapshot.json",
    ]
    paths.extend(config.tables_dir / f"{name}.parquet" for name in PREPARE_TABLES)
    for name in (
        "browser_choice_covariates.py",
        "browser_variants.py",
        "browser.py",
        "config.py",
        "io.py",
        "model.py",
        "provenance.py",
        "validation.py",
    ):
        paths.append(config.project_root / "scripts" / "publication_analysis" / name)
    paths.append(config.project_root / "results" / "tables" / "rq1_primary_estimand.csv")
    return paths


def _assert_paired_model_data(m1b: PreparedModelData, m1c: PreparedModelData) -> None:
    if tuple(m1c.terms[: len(m1b.terms)]) != tuple(m1b.terms):
        raise RuntimeError("M1c core term order differs from M1b")
    if not np.array_equal(m1b.x, m1c.x[:, :, : len(m1b.terms)]):
        raise RuntimeError("M1c core term arrays differ from M1b")
    for name in (
        "candidate_is_ai",
        "mask",
        "chosen_position",
        "receiver_index",
        "receiver_is_ai",
        "actor_index",
        "topic_index",
    ):
        if not np.array_equal(getattr(m1b, name), getattr(m1c, name)):
            raise RuntimeError(f"M1b/M1c paired array differs: {name}")
    for name in (
        "choice_ids",
        "receiver_labels",
        "receiver_groups",
        "actor_labels",
        "actor_groups",
        "topic_labels",
        "choice_groups",
    ):
        if tuple(getattr(m1b, name)) != tuple(getattr(m1c, name)):
            raise RuntimeError(f"M1b/M1c paired labels differ: {name}")


def _atomic_netcdf(idata: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.nc")
    try:
        idata.to_netcdf(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_or_resume(
    config: AnalysisConfig,
    frame: pd.DataFrame,
    core_numeric_terms: Sequence[str],
    browser_terms: Sequence[str],
    model_label: str,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed_offset: int,
) -> tuple[Any, PreparedModelData, dict[str, Any]]:
    import arviz as az

    numeric_terms = tuple(core_numeric_terms) + tuple(browser_terms)
    data = prepare_model_data(
        frame,
        numeric_terms,
        ai_term="candidate_target_is_ai",
    )
    data_hash = _prepared_data_sha256(data)
    implementation_hash = _variant_implementation_sha256(config)
    request = {
        "schema": VARIANT_SCHEMA,
        "run_id": config.run_id,
        "model": model_label,
        "model_spec": "M1",
        "model_data_sha256": data_hash,
        "variant_implementation_sha256": implementation_hash,
        "terms": list(data.terms),
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "target_accept": float(config.sampling.target_accept),
        "random_seed": int(config.seed + seed_offset),
        "analysis_status": "exploratory_browser_data_quality_failed",
    }
    path = config.models_dir / f"{model_label}.nc"
    sidecar = path.with_suffix(path.suffix + ".fit.json")
    if path.exists() or sidecar.exists():
        if not path.exists() or not sidecar.exists():
            raise RuntimeError(f"Incomplete resumable posterior pair for {model_label}")
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved.get("request") != request:
            raise RuntimeError(f"Existing posterior request differs for {model_label}")
        if saved.get("posterior_sha256") != sha256_file(path):
            raise RuntimeError(f"Existing posterior bytes changed for {model_label}")
        return az.from_netcdf(path), data, saved

    model = build_primary_pymc_model(data, config, model_spec="M1")
    idata = fit_model(
        model,
        config,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=config.seed + seed_offset,
        progressbar=False,
        cores=min(chains, 4),
    )
    _atomic_netcdf(idata, path)
    saved = {
        "request": request,
        "posterior_sha256": sha256_file(path),
        "posterior_size_bytes": path.stat().st_size,
    }
    write_json(saved, sidecar)
    return idata, data, saved


def _mcmc_diagnostic(idata: Any, model_label: str) -> dict[str, Any]:
    import arviz as az

    beta_rhat = az.rhat(idata, var_names=["beta"]).to_array().values.astype(float)
    beta_ess_bulk = az.ess(idata, var_names=["beta"], method="bulk").to_array().values.astype(float)
    beta_ess_tail = az.ess(idata, var_names=["beta"], method="tail").to_array().values.astype(float)
    divergences = int(idata.sample_stats["diverging"].sum().item())
    return {
        "model": model_label,
        "chains": int(idata.posterior.sizes.get("chain", 0)),
        "draws_per_chain": int(idata.posterior.sizes.get("draw", 0)),
        "divergences": divergences,
        "rhat_max": float(np.nanmax(beta_rhat)),
        "ess_bulk_min": float(np.nanmin(beta_ess_bulk)),
        "ess_tail_min": float(np.nanmin(beta_ess_tail)),
        "mcmc_release_pass": bool(
            divergences == 0
            and float(np.nanmax(beta_rhat)) <= 1.01
            and float(np.nanmin(beta_ess_bulk)) >= 400
        ),
        "analysis_status": "exploratory_browser_data_quality_failed",
    }


def _posterior_correlations(idata: Any, model_label: str) -> pd.DataFrame:
    beta = idata.posterior["beta"]
    terms = tuple(str(value) for value in beta.coords["term"].values)
    browser_terms = [term for term in terms if "visit_count" in term or "dom_presence" in term]
    if len(browser_terms) < 2:
        return pd.DataFrame(
            columns=("model", "term_a", "term_b", "posterior_correlation", "analysis_status")
        )
    values = (
        beta.sel(term=browser_terms)
        .stack(sample=("chain", "draw"))
        .transpose("sample", "term")
        .values
    )
    correlation = np.corrcoef(values, rowvar=False)
    rows = []
    for left in range(len(browser_terms)):
        for right in range(left + 1, len(browser_terms)):
            rows.append(
                {
                    "model": model_label,
                    "term_a": browser_terms[left],
                    "term_b": browser_terms[right],
                    "posterior_correlation": float(correlation[left, right]),
                    "analysis_status": "exploratory_browser_data_quality_failed",
                }
            )
    return pd.DataFrame(rows)


def _session_grouped_psis(
    idata: Any,
    frame: pd.DataFrame,
    choice_ids: Sequence[str],
    *,
    pareto_k_threshold: float = 0.70,
) -> pd.DataFrame:
    import arviz as az
    from scipy.special import logsumexp

    mapping = frame[["choice_set_id", "group", "browser_session_id"]].drop_duplicates()
    if mapping.groupby("choice_set_id").size().ne(1).any():
        raise ValueError("Each choice set must map to one browser session")
    mapping = mapping.assign(_choice=mapping["choice_set_id"].astype(str)).set_index("_choice")
    ordered = mapping.loc[list(choice_ids)]
    likelihood = idata.log_likelihood["choice"]
    dimension = "choice_set" if "choice_set" in likelihood.dims else likelihood.dims[-1]
    rows = []
    labels = ordered["group"].astype(str) + "::" + ordered["browser_session_id"].astype(str)
    for label in pd.unique(labels):
        selected = labels.eq(label).to_numpy()
        grouped = likelihood.isel({dimension: np.flatnonzero(selected)}).sum(dim=dimension)
        values = grouped.stack(sample=("chain", "draw")).transpose("sample").values
        weights, pareto_k = az.psislw(-values)
        group, session_id = str(label).split("::", maxsplit=1)
        rows.append(
            {
                "group": group,
                "influence_unit": "browser_session",
                "unit_id": session_id,
                "choice_sets": int(selected.sum()),
                "pareto_k": float(pareto_k),
                "elpd_loo": float(logsumexp(values + weights)),
                "needs_refit": bool(pareto_k > pareto_k_threshold),
                "threshold": pareto_k_threshold,
            }
        )
    return pd.DataFrame(rows)


def _model_influence(
    idata: Any,
    frame: pd.DataFrame,
    data: PreparedModelData,
    model_label: str,
) -> pd.DataFrame:
    parts = []
    for unit in ("participant", "account", "thread"):
        part = grouped_choice_psis(idata, frame, unit=unit)
        part.insert(0, "model", model_label)
        parts.append(part)
    session = _session_grouped_psis(idata, frame, data.choice_ids)
    session.insert(0, "model", model_label)
    parts.append(session)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["analysis_status"] = "exploratory_browser_data_quality_failed"
    return out


def _estimand_comparison(
    config: AnalysisConfig,
    estimands: pd.DataFrame,
    matched_frame: pd.DataFrame,
) -> pd.DataFrame:
    admitted = pd.read_csv(config.project_root / "results" / "tables" / "rq1_primary_estimand.csv")
    admitted = admitted.loc[
        admitted["estimand"].eq("rq1_opportunity_standardized_ai_target_selection_probability")
    ].copy()
    admitted["model_display"] = "M1a"
    admitted["comparison_role"] = "full_sample_reference"
    admitted["analysis_status"] = "admitted_primary"
    admitted["choice_sets"] = EXPECTED_M1_CHOICE_SETS
    admitted["candidate_rows"] = EXPECTED_M1_CANDIDATE_ROWS
    admitted_rows = admitted[
        [
            "model_display",
            "comparison_role",
            "group",
            "posterior_mean",
            "posterior_median",
            "ci_low",
            "ci_high",
            "posterior_probability_gt_zero",
            "posterior_draws",
            "choice_sets",
            "candidate_rows",
            "analysis_status",
        ]
    ]
    selected = estimands.loc[
        estimands["model"].isin(("M1b_browser_matched", "M1c_browser_adjusted"))
        & (
            (
                estimands["estimand"].eq("common_opportunity_ai_choice_probability")
                & estimands["group"].isin(("G1", "G2"))
            )
            | (
                estimands["estimand"].eq(
                    "G2_minus_G1_common_opportunity_ai_choice_probability"
                )
                & estimands["group"].eq("G2-G1")
            )
        )
    ].copy()
    selected["model_display"] = selected["model"].map(
        {"M1b_browser_matched": "M1b", "M1c_browser_adjusted": "M1c"}
    )
    selected["comparison_role"] = selected["model"].map(
        {
            "M1b_browser_matched": "browser_cohort_unadjusted",
            "M1c_browser_adjusted": "browser_cohort_adjusted",
        }
    )
    selected["choice_sets"] = int(matched_frame["choice_set_id"].nunique())
    selected["candidate_rows"] = int(len(matched_frame))
    selected["analysis_status"] = "exploratory_browser_data_quality_failed"
    selected_rows = selected[
        [
            "model_display",
            "comparison_role",
            "group",
            "posterior_mean",
            "posterior_median",
            "ci_low",
            "ci_high",
            "posterior_probability_gt_zero",
            "posterior_draws",
            "choice_sets",
            "candidate_rows",
            "analysis_status",
        ]
    ]
    return pd.concat([admitted_rows, selected_rows], ignore_index=True)


def run_fit_choice(
    config: AnalysisConfig,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit the paired exploratory M1b/M1c models and prespecified sensitivities."""

    status = _load_prepare_status(config)
    if config.profile == "release" and not bool(status.get("tracking_quality_passed")):
        raise RuntimeError("Release browser fitting is blocked by failed browser data-quality checks")
    source_snapshot = capture_source_snapshot(_fit_source_paths(config))
    frame = read_table(config.tables_dir / "rq1_browser_matched_model_frame")
    core_terms = tuple(status.get("m1a_terms", ()))
    if not core_terms or core_terms[0] != "candidate_target_is_ai":
        raise RuntimeError("Prepared browser run is missing the frozen M1 term order")
    core_numeric_terms = tuple(term for term in core_terms if term != "candidate_target_is_ai")
    expected_draws = int(draws or config.sampling.draws)
    expected_tune = int(tune or config.sampling.tune)
    expected_chains = int(chains or config.sampling.chains)
    idatas: dict[str, Any] = {}
    model_data: dict[str, PreparedModelData] = {}
    signatures: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    parameters: list[pd.DataFrame] = []
    estimands: list[pd.DataFrame] = []
    correlations: list[pd.DataFrame] = []
    for offset, (model_label, browser_terms) in enumerate(MODEL_VARIANTS.items()):
        idata, data, signature = _fit_or_resume(
            config,
            frame,
            core_numeric_terms,
            browser_terms,
            model_label,
            draws=expected_draws,
            tune=expected_tune,
            chains=expected_chains,
            seed_offset=100 * offset,
        )
        idatas[model_label] = idata
        model_data[model_label] = data
        signatures.append(
            {
                "model": model_label,
                "posterior_sha256": signature["posterior_sha256"],
                **signature["request"],
            }
        )
        diagnostics.append(_mcmc_diagnostic(idata, model_label))
        summary = summarize_fit(idata)
        summary.insert(0, "model", model_label)
        summary["analysis_status"] = "exploratory_browser_data_quality_failed"
        parameters.append(summary)
        model_estimands = standardized_estimands(idata, data, counterfactual_values={})
        model_estimands.insert(0, "model", model_label)
        model_estimands["analysis_status"] = "exploratory_browser_data_quality_failed"
        estimands.append(model_estimands)
        correlations.append(_posterior_correlations(idata, model_label))

    _assert_paired_model_data(
        model_data["M1b_browser_matched"],
        model_data["M1c_browser_adjusted"],
    )
    influence = pd.concat(
        [
            _model_influence(
                idatas[model_label], frame, model_data[model_label], model_label
            )
            for model_label in ("M1b_browser_matched", "M1c_browser_adjusted")
        ],
        ignore_index=True,
        sort=False,
    )
    estimand_table = pd.concat(estimands, ignore_index=True)
    nonempty_correlations = [frame for frame in correlations if not frame.empty]
    output = {
        "rq1_browser_fit_run_signature": pd.DataFrame(signatures),
        "rq1_browser_model_diagnostics": pd.DataFrame(diagnostics),
        "rq1_browser_parameter_summary": pd.concat(parameters, ignore_index=True),
        "rq1_browser_estimands": estimand_table,
        "rq1_browser_estimand_comparison": _estimand_comparison(config, estimand_table, frame),
        "rq1_browser_posterior_correlations": (
            pd.concat(nonempty_correlations, ignore_index=True)
            if nonempty_correlations
            else pd.DataFrame(
                columns=(
                    "model",
                    "term_a",
                    "term_b",
                    "posterior_correlation",
                    "analysis_status",
                )
            )
        ),
        "rq1_browser_grouped_influence": influence,
    }
    assert_source_snapshot_unchanged(source_snapshot)
    for name, table in output.items():
        write_table(table, config.tables_dir / name)
    fit_bundle = {
        "schema": VARIANT_SCHEMA,
        "run_id": config.run_id,
        "analysis_status": "exploratory_browser_data_quality_failed",
        "variant_implementation_sha256": _variant_implementation_sha256(config),
        "sources": source_snapshot,
        "tables": [
            source_record(config.tables_dir / f"{name}.parquet") for name in FIT_TABLES
        ],
        "posteriors": [
            {
                **source_record(config.models_dir / f"{model}.nc"),
                "fit_signature": source_record(
                    (config.models_dir / f"{model}.nc").with_suffix(".nc.fit.json")
                ),
            }
            for model in MODEL_VARIANTS
        ],
    }
    write_json(fit_bundle, config.outputs_dir / "browser_variant_fit_bundle.json")
    assert_source_snapshot_unchanged(source_snapshot)
    write_json(
        {
            **status,
            "stage": "fit-choice",
            "status": "completed",
            "analysis_status": "exploratory_browser_data_quality_failed",
            "fitted_models": list(MODEL_VARIANTS),
            "draws": expected_draws,
            "tune": expected_tune,
            "chains": expected_chains,
            "expected_fit_tables": list(FIT_TABLES),
        },
        config.run_dir / "status.json",
    )
    return output


def _bundle_authentication(bundle: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    source_ok, changed_sources = _records_unchanged(bundle.get("sources", []))
    table_ok, changed_tables = _records_unchanged(bundle.get("tables", []))
    posterior_records = bundle.get("posteriors", [])
    posterior_ok, changed_posteriors = _records_unchanged(posterior_records)
    fit_records = [
        record.get("fit_signature", {})
        for record in posterior_records
        if isinstance(record, Mapping)
    ]
    fit_ok, changed_fit_signatures = _records_unchanged(fit_records)
    details = {
        "source_bytes_unchanged": source_ok,
        "result_table_bytes_unchanged": table_ok,
        "posterior_bytes_unchanged": posterior_ok,
        "fit_signature_bytes_unchanged": fit_ok,
        "changed_sources": ",".join(changed_sources),
        "changed_tables": ",".join(changed_tables),
        "changed_posteriors": ",".join(changed_posteriors),
        "changed_fit_signatures": ",".join(changed_fit_signatures),
    }
    return source_ok and table_ok and posterior_ok and fit_ok, details


def run_validate_choice(config: AnalysisConfig) -> pd.DataFrame:
    """Authenticate paired browser outputs and apply pilot/release gates."""

    status = _load_prepare_status(config)
    if status.get("stage") not in {"fit-choice", "validate-choice", "report"}:
        raise RuntimeError("Browser variant fit is not complete")
    bundle_path = config.outputs_dir / "browser_variant_fit_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle_ok, bundle_details = _bundle_authentication(bundle)
    frame = read_table(config.tables_dir / "rq1_browser_matched_model_frame")
    signatures = read_table(config.tables_dir / "rq1_browser_fit_run_signature")
    diagnostics = read_table(config.tables_dir / "rq1_browser_model_diagnostics")
    comparison = read_table(config.tables_dir / "rq1_browser_estimand_comparison")
    core_terms = tuple(status.get("m1a_terms", ()))
    core_numeric = tuple(term for term in core_terms if term != "candidate_target_is_ai")
    m1b = prepare_model_data(frame, core_numeric, ai_term="candidate_target_is_ai")
    m1c = prepare_model_data(
        frame,
        (*core_numeric, *PRIMARY_MODEL_TERMS),
        ai_term="candidate_target_is_ai",
    )
    pair_ok = True
    try:
        _assert_paired_model_data(m1b, m1c)
    except RuntimeError:
        pair_ok = False
    signature_models = set(signatures["model"].astype(str))
    models_complete = signature_models == set(MODEL_VARIANTS)
    implementation_ok = (
        str(bundle.get("variant_implementation_sha256"))
        == _variant_implementation_sha256(config)
    )
    m1a_hash_ok = (
        sha256_file(config.project_root / "results" / "tables" / "rq1_primary_estimand.csv")
        == ADMITTED_PRIMARY_ESTIMAND_SHA256
    )
    candidate_sets_ok = bool(
        frame.groupby("choice_set_id").size().ge(2).all()
        and frame.groupby("choice_set_id")["chosen"].sum().eq(1).all()
        and frame["browser_choice_set_complete"].astype(bool).all()
    )
    finite_browser = bool(
        np.isfinite(frame.loc[:, list(RAW_BROWSER_COLUMNS)].to_numpy(dtype=float)).all()
    )
    both_arms = set(frame["group"].astype(str)) == {"G1", "G2"}
    tracking_quality = bool(status.get("tracking_quality_passed"))
    main_diagnostics = diagnostics[
        diagnostics["model"].isin(("M1b_browser_matched", "M1c_browser_adjusted"))
    ]
    mcmc_release = bool(
        len(main_diagnostics) == 2
        and main_diagnostics["mcmc_release_pass"].astype(bool).all()
    )
    comparison_groups = comparison.groupby("model_display")["group"].agg(
        lambda values: set(map(str, values))
    )
    comparison_complete = bool(
        set(comparison_groups.index.astype(str)) == {"M1a", "M1b", "M1c"}
        and all(groups == {"G1", "G2", "G2-G1"} for groups in comparison_groups)
    )
    checks = {
        "fit_bundle_authenticated": bundle_ok,
        "variant_implementation_matches": implementation_ok,
        "m1a_admitted_hash_unchanged": m1a_hash_ok,
        "m1b_m1c_exact_pair": pair_ok,
        "all_declared_models_present": models_complete,
        "candidate_sets_preserved": candidate_sets_ok,
        "browser_covariates_finite": finite_browser,
        "both_arms_present": both_arms,
        "estimand_comparison_complete": comparison_complete,
        "browser_data_quality_pass": tracking_quality,
        "main_model_mcmc_release_pass": mcmc_release,
    }
    internal_checks = [
        key for key in checks if key not in {"browser_data_quality_pass", "main_model_mcmc_release_pass"}
    ]
    internal_integrity = all(checks[key] for key in internal_checks)
    release_ready = internal_integrity and tracking_quality and mcmc_release
    rows = [
        {
            "check": name,
            "passed": bool(passed),
            "gate_scope": (
                "release" if name in {"browser_data_quality_pass", "main_model_mcmc_release_pass"}
                else "integrity"
            ),
            "analysis_status": "exploratory_browser_data_quality_failed",
            "details": "",
        }
        for name, passed in checks.items()
    ]
    rows.extend(
        [
            {
                "check": "internal_integrity_pass",
                "passed": internal_integrity,
                "gate_scope": "pilot",
                "analysis_status": "exploratory_browser_data_quality_failed",
                "details": "",
            },
            {
                "check": "browser_variant_release_ready",
                "passed": release_ready,
                "gate_scope": "release",
                "analysis_status": "exploratory_browser_data_quality_failed",
                "details": "",
            },
        ]
    )
    validation = pd.DataFrame(rows)
    write_table(validation, config.tables_dir / "rq1_browser_validation")
    write_json(
        {
            **status,
            "stage": "validate-choice",
            "status": "completed",
            "analysis_status": "exploratory_browser_data_quality_failed",
            "internal_integrity_pass": internal_integrity,
            "release_ready": release_ready,
            "bundle_authentication": bundle_details,
        },
        config.run_dir / "status.json",
    )
    return validation


def run_report(config: AnalysisConfig) -> Path:
    """Write an internal, explicitly exploratory browser-variant report."""

    status = _load_prepare_status(config)
    validation = read_table(config.tables_dir / "rq1_browser_validation")
    comparison = read_table(config.tables_dir / "rq1_browser_estimand_comparison")
    attrition = read_table(config.tables_dir / "rq1_browser_cohort_attrition")
    diagnostics = read_table(config.tables_dir / "rq1_browser_model_diagnostics")
    failures = validation.loc[~validation["passed"].astype(bool), "check"].astype(str).tolist()
    lines = [
        "# RQ1 browser-covariate sensitivity",
        "",
        f"- Run ID: `{config.run_id}`",
        "- Status: exploratory browser analysis",
        f"- Matched cohort: {status['matched_choice_sets']} replies and {status['matched_candidate_rows']} candidates",
        "- DOM-presence time is a browser-state proxy for opportunity to read. It does not measure attention or reading.",
        "- Candidate sets are the original earlier valid non-self same-thread candidates.",
        "",
        "## Estimands",
        "",
        "| Model | Role | Group | Mean | Median | 95% interval | P(>0) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict("records"):
        lines.append(
            "| {model_display} | {comparison_role} | {group} | {posterior_mean:.4f} | "
            "{posterior_median:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | "
            "{posterior_probability_gt_zero:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "M1a to M1b describes browser-cohort selection. M1b to M1c describes adjustment for recorded DOM activity within that cohort.",
            "",
            "## Cohort attrition",
            "",
            attrition.to_string(index=False),
            "",
            "## Pilot diagnostics",
            "",
            diagnostics.to_string(index=False),
            "",
            "## Failed release checks",
            "",
            *(f"- {failure}" for failure in failures),
            "",
            "These results are ineligible for promotion while browser data-quality or release-diagnostic checks fail.",
            "",
        ]
    )
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    path = config.reports_dir / "rq1_browser_variant_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        {
            **status,
            "stage": "report",
            "status": "completed",
            "report_path": str(path),
        },
        config.run_dir / "status.json",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.publication_analysis.browser_variants",
        description="Matched M1b/M1c browser-covariate sensitivity analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-choice", help="Prepare the paired browser cohort.")
    prepare.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    prepare.add_argument("--profile", choices=("pilot", "release"), default="pilot")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--m1a-run-dir", type=Path, required=True)
    fit = subparsers.add_parser("fit-choice", help="Fit paired exploratory M1b/M1c models.")
    fit.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    fit.add_argument("--profile", choices=("pilot", "release"), default="pilot")
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--draws", type=int)
    fit.add_argument("--tune", type=int)
    fit.add_argument("--chains", type=int)
    validate = subparsers.add_parser(
        "validate-choice", help="Authenticate browser variants and apply release gates."
    )
    validate.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    validate.add_argument("--profile", choices=("pilot", "release"), default="pilot")
    validate.add_argument("--run-id", required=True)
    report = subparsers.add_parser("report", help="Write the internal browser-variant report.")
    report.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    report.add_argument("--profile", choices=("pilot", "release"), default="pilot")
    report.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, profile=args.profile, run_id=args.run_id)
    if args.command == "prepare-choice":
        tables = run_prepare_choice(config, config_path, args.m1a_run_dir)
        print(
            f"Prepared {len(tables)} M1b/M1c browser tables in {config.tables_dir}",
            flush=True,
        )
        return 0
    if args.command == "fit-choice":
        tables = run_fit_choice(
            config,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
        )
        print(f"Fitted {len(MODEL_VARIANTS)} browser models and wrote {len(tables)} tables")
        return 0
    if args.command == "validate-choice":
        validation = run_validate_choice(config)
        release_ready = bool(
            validation.loc[
                validation["check"].eq("browser_variant_release_ready"), "passed"
            ].iloc[0]
        )
        print(
            f"Browser variant validation complete; release_ready={release_ready}",
            flush=True,
        )
        return 0 if config.profile == "pilot" or release_ready else 2
    if args.command == "report":
        path = run_report(config)
        print(f"Wrote browser variant report to {path}")
        return 0
    raise ValueError(f"Unsupported browser variant command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADMITTED_PRIMARY_ESTIMAND_SHA256",
    "EXPECTED_M1_CANDIDATE_ROWS",
    "EXPECTED_M1_CHOICE_SETS",
    "FIT_TABLES",
    "MODEL_VARIANTS",
    "PREPARE_TABLES",
    "authenticate_frozen_m1a",
    "build_parser",
    "main",
    "run_fit_choice",
    "run_prepare_choice",
    "run_report",
    "run_validate_choice",
]
