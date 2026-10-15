"""Run manifest construction for reproducible publication artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import AnalysisConfig
from .io import runtime_versions, source_record, write_json
from .study import GROUPS, analysis_set_manifest


def _git_revision(project_root: Path) -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_worktree_dirty": None}


def source_paths(config: AnalysisConfig, config_path: Path) -> list[Path]:
    def table_path(stem: Path) -> Path:
        for suffix in (".parquet", ".csv"):
            candidate = stem.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return stem.with_suffix(".parquet")

    source_manifest = config.source_manifest or (
        config.project_root
        / "data"
        / "raw"
        / "datasets"
        / config.processed_dir.name
        / "metadata"
        / "manifest.json"
    )
    paths = [
        config_path,
        source_manifest,
        table_path(config.processed_dir / "snapshot_events"),
        table_path(config.processed_dir / "snapshot_contents"),
        table_path(config.processed_dir / "snapshot_user_reputation"),
        config.project_root / "scripts" / "publication_analysis" / "study.py",
    ]
    if config.analysis_stage == "rq2_m2":
        paths.append(
            table_path(config.processed_dir / "snapshot_user_reputation_history")
        )
        if config.legacy_tables_dir is not None:
            for stem in (
                "22_interest_event_panel",
                "70_interest_topic_affordance_addon_event_panel",
            ):
                candidate = table_path(config.legacy_tables_dir / stem)
                if candidate.exists():
                    paths.append(candidate)
    elif config.analysis_stage not in {"rq1_m1", "rq3"}:
        paths.extend(
            [
                table_path(config.processed_dir / "snapshot_user_reputation_history"),
                table_path(config.browser_dir / "session_events"),
                table_path(config.browser_dir / "session_qc"),
            ]
        )
        for name in (
            "browser_interval_audit",
            "browser_manual_audit",
            "manual_tracking_audit",
            "session_manual_audit",
            "manual_audit",
        ):
            candidate = table_path(config.browser_dir / name)
            if candidate.exists():
                paths.append(candidate)
        if config.legacy_tables_dir is not None:
            for stem in (
                "22_interest_event_panel",
                "70_interest_topic_affordance_addon_event_panel",
            ):
                candidate = table_path(config.legacy_tables_dir / stem)
                if candidate.exists():
                    paths.append(candidate)
    package_dir = config.project_root / "scripts" / "publication_analysis"
    if package_dir.exists():
        paths.extend(sorted(package_dir.glob("*.py")))
    return paths


def write_manifest(config: AnalysisConfig, config_path: Path, artifacts: Iterable[Path] = ()) -> Path:
    """Record immutable inputs, filters, software versions, and generated artifacts."""

    group_manifest = {
        group: {
            "condition": value.label,
            "database": value.db_name,
            "experiment_id": value.experiment_id,
            "client_id": value.client_id,
            "expected_humans": value.human_count,
            "expected_ai": value.ai_count,
        }
        for group, value in GROUPS.items()
    }
    value = {
        "run_id": config.run_id,
        "analysis_stage": config.analysis_stage,
        "analysis_set": analysis_set_manifest(config.analysis_set),
        "start_time": config.start_time,
        "end_time": config.end_time or "snapshot extraction",
        "seed": config.seed,
        "configuration": {
            "sampling": asdict(config.sampling),
            "browser_gates": asdict(config.browser_gates),
            "latent_block_sizes": list(config.latent_block_sizes),
            "rq3": asdict(config.rq3),
        },
        "raw_data_mutated": False,
        "group_mapping": group_manifest,
        "sources": [source_record(path) for path in source_paths(config, config_path)],
        "artifacts": [source_record(path) for path in artifacts if path.exists() and path.is_file()],
        "software": runtime_versions(),
        "git": _git_revision(config.project_root),
        "invocation": {
            "argv": [str(value) for value in sys.argv],
            "working_directory": str(Path.cwd()),
        },
    }
    path = config.run_dir / "manifest.json"
    write_json(value, path)
    return path


def write_artifact_manifests(
    config: AnalysisConfig,
    run_manifest_path: Path,
    metadata: Mapping[Path, Mapping[str, Any]],
) -> list[Path]:
    """Write compact provenance sidecars for canonical publication artifacts."""

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    shared = {
        "run_id": run_manifest["run_id"],
        "analysis_stage": run_manifest.get("analysis_stage", "legacy"),
        "analysis_set": run_manifest["analysis_set"],
        "start_time": run_manifest["start_time"],
        "end_time": run_manifest["end_time"],
        "seed": run_manifest["seed"],
        "configuration": run_manifest["configuration"],
        "group_mapping": run_manifest["group_mapping"],
        "sources": run_manifest["sources"],
        "software": run_manifest["software"],
        "invocation": run_manifest["invocation"],
        "raw_data_mutated": False,
    }
    paths: list[Path] = []
    for artifact, artifact_metadata in metadata.items():
        if not artifact.exists() or not artifact.is_file():
            continue
        value = {
            **shared,
            "artifact": source_record(artifact),
            "artifact_metadata": dict(artifact_metadata),
            "run_manifest": str(run_manifest_path),
        }
        path = artifact.with_suffix(artifact.suffix + ".manifest.json")
        write_json(value, path)
        paths.append(path)
    return paths
