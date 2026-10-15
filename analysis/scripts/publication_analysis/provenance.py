"""Transactional provenance for fitted publication-result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .config import AnalysisConfig
from .io import source_record, write_json


BUNDLE_SCHEMA = "ysocial.publication-fit-bundle.v1"
BUNDLE_FILENAME = "fit_result_bundle.json"


def capture_source_snapshot(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Capture sorted file identities before a long-running fit begins."""

    unique = sorted({Path(path).resolve() for path in paths}, key=str)
    return [source_record(path) for path in unique]


def _records_match(records: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        return False, ["<invalid-record-list>"]
    changed: list[str] = []
    for saved in records:
        if not isinstance(saved, Mapping):
            changed.append("<invalid-record>")
            continue
        path = Path(str(saved.get("path", "")))
        current = source_record(path)
        comparable = ("path", "exists", "size_bytes", "sha256")
        if any(current.get(key) != saved.get(key) for key in comparable):
            changed.append(str(path))
    return not changed, changed


def assert_source_snapshot_unchanged(
    source_snapshot: Sequence[Mapping[str, Any]],
) -> None:
    """Stop publication if source or analysis code changed during execution."""

    unchanged, changed = _records_match(source_snapshot)
    if not unchanged:
        preview = ", ".join(changed[:5])
        suffix = "" if len(changed) <= 5 else f" and {len(changed) - 5} more"
        raise RuntimeError(
            "Fit sources changed while the process was running; results were not "
            f"published. Changed paths: {preview}{suffix}"
        )


def write_fit_result_bundle(
    config: AnalysisConfig,
    *,
    source_snapshot: Sequence[Mapping[str, Any]],
    model_data_sha256: str,
    implementation_sha256: str,
    table_artifacts: Iterable[Path],
    posterior_paths: Iterable[Path],
    fit_request: Mapping[str, Any],
) -> Path:
    """Bind exact result-table bytes to exact posterior and source bytes."""

    tables = []
    for path in sorted({Path(value).resolve() for value in table_artifacts}, key=str):
        record = source_record(path)
        record["table"] = path.stem
        record["format"] = path.suffix.removeprefix(".")
        tables.append(record)
    posteriors = []
    for path in sorted({Path(value).resolve() for value in posterior_paths}, key=str):
        record = source_record(path)
        fit_sidecar = path.with_suffix(path.suffix + ".fit.json")
        record["fit_signature"] = source_record(fit_sidecar)
        posteriors.append(record)
    value = {
        "schema": BUNDLE_SCHEMA,
        "run_id": config.run_id,
        "model_data_sha256": model_data_sha256,
        "implementation_sha256": implementation_sha256,
        "fit_request": dict(fit_request),
        "sources": list(source_snapshot),
        "tables": tables,
        "posteriors": posteriors,
    }
    path = config.outputs_dir / BUNDLE_FILENAME
    write_json(value, path)
    return path


def authenticate_fit_result_bundle(
    config: AnalysisConfig,
    *,
    current_model_data_sha256: str,
    current_implementation_sha256: str,
    required_tables: Iterable[str],
) -> pd.DataFrame:
    """Authenticate a completed fit bundle against live files and current data."""

    path = config.outputs_dir / BUNDLE_FILENAME
    status: list[str] = []
    value: Mapping[str, Any] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise TypeError("bundle root is not an object")
        value = loaded
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        status.append("missing_or_invalid_bundle")
    if value:
        if value.get("schema") != BUNDLE_SCHEMA:
            status.append("schema_mismatch")
        if str(value.get("run_id")) != config.run_id:
            status.append("run_id_mismatch")
        if str(value.get("model_data_sha256")) != current_model_data_sha256:
            status.append("model_data_hash_mismatch")
        if str(value.get("implementation_sha256")) != current_implementation_sha256:
            status.append("implementation_hash_mismatch")
        sources_ok, changed_sources = _records_match(value.get("sources", []))
        tables_ok, changed_tables = _records_match(value.get("tables", []))
        posterior_records = value.get("posteriors", [])
        posteriors_ok, changed_posteriors = _records_match(posterior_records)
        changed_fit_signatures: list[str] = []
        for record in posterior_records if isinstance(posterior_records, list) else []:
            if not isinstance(record, Mapping):
                changed_fit_signatures.append("<invalid-posterior-record>")
                continue
            fit_record = record.get("fit_signature", {})
            fit_ok, fit_changed = _records_match([fit_record])
            if not fit_ok:
                changed_fit_signatures.extend(fit_changed)
        if not sources_ok:
            status.append("source_bytes_changed")
        if not tables_ok:
            status.append("result_table_bytes_changed")
        if not posteriors_ok or changed_fit_signatures:
            status.append("posterior_or_signature_bytes_changed")
        table_records = value.get("tables", [])
        table_names = {
            str(record.get("table"))
            for record in table_records
            if isinstance(record, Mapping)
        } if isinstance(table_records, list) else set()
        missing_tables = sorted(set(required_tables).difference(table_names))
        if missing_tables:
            status.append("required_result_tables_missing")
    else:
        changed_sources = []
        changed_tables = []
        changed_posteriors = []
        changed_fit_signatures = []
        table_names = set()
        missing_tables = sorted(set(required_tables))
    authenticated = not status
    return pd.DataFrame(
        [
            {
                "bundle_path": str(path),
                "authenticated": authenticated,
                "authentication_status": (
                    "authenticated" if authenticated else ";".join(status)
                ),
                "authenticated_table_names": ",".join(sorted(table_names)),
                "required_tables_present": not missing_tables,
                "missing_required_tables": ",".join(missing_tables),
                "changed_sources": ",".join(changed_sources),
                "changed_tables": ",".join(changed_tables),
                "changed_posteriors": ",".join(changed_posteriors),
                "changed_fit_signatures": ",".join(changed_fit_signatures),
            }
        ]
    )


__all__ = [
    "BUNDLE_FILENAME",
    "assert_source_snapshot_unchanged",
    "authenticate_fit_result_bundle",
    "capture_source_snapshot",
    "write_fit_result_bundle",
]
