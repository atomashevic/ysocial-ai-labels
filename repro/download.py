#!/usr/bin/env python3
"""Download the Zenodo record, unpack it into ``data/`` and verify every file.

By default only the two archives the figures and tables read are fetched
(``07_annotations.zip`` and ``08_analysis_ready.zip``, about 2 MB). ``--all``
fetches the complete record (about 130 MB). Unpacked files are checked
against the record's ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from repro.paths import DATA_DIR

RECORD_ID = "22964197"
API = "https://zenodo.org/api/records"
REQUIRED = ("07_annotations.zip", "08_analysis_ready.zip")
ALL = (
    "00_study.zip", "01_accounts.zip", "02_platform_activity.zip", "03_agent_memory.zip",
    "04_agent_interviews.zip", "05_browser_telemetry.zip", "06_experiment_config.zip",
    "07_annotations.zip", "08_analysis_ready.zip", "scripts.zip",
    "README.md", "CODEBOOK.md", "CHANGELOG.md", "LICENSE.md", "VALIDATION.md", "BUILD_LOG.txt",
    "datapackage.json", "row_counts.json",
)


def file_url(key: str, token: str | None) -> str:
    if token:  # unpublished draft opened through a Zenodo share link
        return f"{API}/{RECORD_ID}/draft/files/{key}/content?token={token}"
    return f"{API}/{RECORD_ID}/files/{key}/content"


def fetch(key: str, dest: Path, token: str | None) -> Path:
    target = dest / key
    if target.exists():
        return target
    print(f"downloading {key}", file=sys.stderr)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(file_url(key, token)) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.rename(target)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(dest: Path, prefixes: Sequence[str]) -> int:
    """Check unpacked files under ``prefixes`` against SHA256SUMS; return the number checked."""
    checked = 0
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if not name.startswith(tuple(prefixes)):
            continue
        path = dest / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} listed in SHA256SUMS but not unpacked")
        if sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {name}")
        checked += 1
    return checked


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="download the complete record")
    parser.add_argument("--dest", type=Path, default=DATA_DIR, help=f"target directory (default: {DATA_DIR})")
    parser.add_argument("--token", help="Zenodo share-link token, only needed before the record is published")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    keys = ALL if args.all else REQUIRED
    fetch("SHA256SUMS", args.dest, args.token)
    prefixes = []
    for key in keys:
        path = fetch(key, args.dest, args.token)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(args.dest)
            prefixes.append(path.stem + "/")
    print(f"verified {verify(args.dest, prefixes)} files in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
