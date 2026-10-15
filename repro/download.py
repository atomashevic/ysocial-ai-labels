#!/usr/bin/env python3
"""Download the Zenodo record, unpack it into ``data/`` and verify every file.

By default only the two archives the figures and tables read are fetched
(``07_annotations.zip`` and ``08_analysis_ready.zip``, about 3 MB). ``--fit``
adds the browser telemetry and experiment configuration that the model refits
in ``analysis/`` also read (about 70 MB). ``--all`` fetches the complete record (about 130 MB). The newest
published version of the record is used unless ``--record`` names a version.
Unpacked files are checked against the record's ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

from repro.paths import DATA_DIR

RECORD_ID = "22964197"  # version 1.0.0; later versions are found through /versions/latest
API = "https://zenodo.org/api/records"
REQUIRED = ("07_annotations.zip", "08_analysis_ready.zip")
FIT = ("05_browser_telemetry.zip", "06_experiment_config.zip", *REQUIRED)
ALL = (
    "00_study.zip", "01_accounts.zip", "02_platform_activity.zip", "03_agent_memory.zip",
    "04_agent_interviews.zip", "05_browser_telemetry.zip", "06_experiment_config.zip",
    "07_annotations.zip", "08_analysis_ready.zip", "scripts.zip",
    "README.md", "CODEBOOK.md", "LICENSE.md", "datapackage.json",
)


def latest_record() -> tuple[str, str]:
    """Return the id and version string of the newest published version."""
    with urllib.request.urlopen(f"{API}/{RECORD_ID}/versions/latest") as response:
        record = json.load(response)
    return str(record["id"]), str(record.get("metadata", {}).get("version", "unknown"))


def file_url(record: str, key: str, token: str | None) -> str:
    if token:  # unpublished draft opened through a Zenodo share link
        return f"{API}/{record}/draft/files/{key}/content?token={token}"
    return f"{API}/{record}/files/{key}/content"


def fetch(record: str, key: str, dest: Path, token: str | None) -> Path:
    target = dest / key
    if target.exists():
        return target
    print(f"downloading {key}", file=sys.stderr)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(file_url(record, key, token)) as response, partial.open("wb") as handle:
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
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true", help="download the complete record")
    scope.add_argument("--fit", action="store_true", help="also download the browser telemetry and configuration read by the model refits")
    parser.add_argument("--dest", type=Path, default=DATA_DIR, help=f"target directory (default: {DATA_DIR})")
    parser.add_argument("--record", help="Zenodo record id of one version (default: the newest published version)")
    parser.add_argument("--token", help="Zenodo share-link token for an unpublished draft; give its id with --record")
    args = parser.parse_args(argv)

    if args.record:
        record = args.record
    elif args.token:
        record = RECORD_ID
    else:
        record, version = latest_record()
        print(f"Zenodo record {record}, version {version}", file=sys.stderr)
    args.dest.mkdir(parents=True, exist_ok=True)
    keys = ALL if args.all else (FIT if args.fit else REQUIRED)
    fetch(record, "SHA256SUMS", args.dest, args.token)
    prefixes = []
    for key in keys:
        path = fetch(record, key, args.dest, args.token)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(args.dest)
            prefixes.append(path.stem + "/")
    print(f"verified {verify(args.dest, prefixes)} files in {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
