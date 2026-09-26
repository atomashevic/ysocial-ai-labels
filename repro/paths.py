"""Input and output locations.

The Zenodo record is unpacked into ``data/`` by ``python -m repro.download``.
Set ``YSOCIAL_DATA_DIR`` to read an existing copy from elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("YSOCIAL_DATA_DIR", ROOT / "data")).expanduser()
RESULTS_DIR = DATA_DIR / "08_analysis_ready" / "results_tables"
VALIDATION_DIR = DATA_DIR / "07_annotations" / "human_validation"

OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def require(path: Path) -> Path:
    """Return ``path`` or explain how to obtain the data."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m repro.download` first, "
            "or set YSOCIAL_DATA_DIR to an unpacked copy of the Zenodo record."
        )
    return path
