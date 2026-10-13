"""Artifact I/O and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing table {parquet} or {csv}")


def write_table(frame: pd.DataFrame, stem: Path) -> dict[str, str]:
    """Write analysis tables in Parquet and CSV without mutating inputs."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return {"parquet": str(parquet), "csv": str(csv)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ["pandas", "numpy", "pymc", "arviz", "statsmodels", "networkx", "pyarrow"]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not-installed"
    return versions
