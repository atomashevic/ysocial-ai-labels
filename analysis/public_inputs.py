"""Lay out the Zenodo record so the unchanged fitting code can read it.

The scripts in ``scripts/`` were written against the study's private
directory layout. This module copies or converts the public files into that
layout below ``public_inputs/`` and writes one configuration per analysis
stage (``analysis.public-<stage>.yaml``). Every setting other than the input
and output paths is taken unchanged from ``analysis.yaml``.

Written below ``public_inputs/``: the canonical tables (CSV and Parquet), the
thread-topic panel under its legacy name, the browser session tables, the
community-to-database manifest, the criticism labels and downvote panel, and
the source folder of the sample-size simulation. The vendor thread-open export
is copied to the path ``rq1b.py`` reads (``data/raw/moveo/...``).

    python public_inputs.py --zenodo /path/to/unpacked/record
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = Path(__file__).resolve().parent
WORK = HERE / "public_inputs"
OUTPUT_ROOT = "public_runs"
VENDOR_THREAD_OPENS = HERE / "data/raw/moveo/feed_engagement_email_2026-07-15/full_dataset/thread_opens.csv"
SIMULATION_SOURCE = WORK / "reply_observed_design_source"

# The reported RQ1 M1 has no topic-varying AI slope, so its configuration
# leaves the topic panel out. RQ2 includes the slope and reads the panel.
STAGES = {
    "rq1": {"analysis_stage": "rq1_m1", "topics": False},
    "rq2": {"analysis_stage": "rq2_m2", "topics": True},
    "rq3": {"analysis_stage": "rq3", "topics": False},
}


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def canonical_tables(zenodo: Path) -> Path:
    source = zenodo / "08_analysis_ready/canonical"
    target = WORK / "processed"
    for path in sorted(source.glob("*.csv")):
        _copy(path, target / path.name)
        # Run manifests hash the Parquet form of these tables.
        pd.read_csv(path, low_memory=False).to_parquet(target / f"{path.stem}.parquet", index=False)
    return target


def topic_panel(zenodo: Path) -> Path:
    """Write thread topics under the legacy panel name the loader searches for."""

    topics = pd.read_csv(zenodo / "08_analysis_ready/model_inputs/thread_topics.csv")
    columns = ["group", "root_id", "primary_root_topic", "root_topics", "topic_assignment_source"]
    target = WORK / "legacy_tables"
    target.mkdir(parents=True, exist_ok=True)
    topics[columns].to_csv(target / "22_interest_event_panel.csv", index=False)
    return target


def browser_tables(zenodo: Path) -> Path:
    """Convert the released telemetry to the session tables read by rq1b.py."""

    source = zenodo / "05_browser_telemetry"
    target = WORK / "browser"
    target.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(source / "events.parquet").rename(columns={"user_id": "y_user_id"})
    events.to_parquet(target / "session_events.parquet", index=False)
    sessions = pd.read_csv(source / "sessions.csv")
    sessions.to_parquet(target / "session_qc.parquet", index=False)
    _copy(source / "vendor_feed_engagement/thread_opens.csv", VENDOR_THREAD_OPENS)
    return target


def source_manifest(zenodo: Path) -> Path:
    """Write the community-to-database manifest read by the data-integrity checks.

    The study read these identities from the private snapshot manifest. The record
    holds them too: each community's database folder in
    06_experiment_config/g*/server_config.json (data_path) and its experiment id,
    which is also the browser app version (05_browser_telemetry/sessions.csv).
    """

    sessions = pd.read_csv(zenodo / "05_browser_telemetry/sessions.csv", usecols=["group", "y_app_version"])
    groups = []
    for key, folder, group in (("grupa_1", "g1", "G1"), ("grupa_2", "g2", "G2")):
        server = json.loads((zenodo / f"06_experiment_config/{folder}/server_config.json").read_text())
        versions = sessions.loc[sessions["group"].eq(group), "y_app_version"].unique()
        if len(versions) != 1:
            raise ValueError(f"{group}: expected one experiment id in sessions.csv, found {versions}")
        database = "experiments_" + Path(server["data_path"].rstrip("/")).name
        groups.append({"group": key, "label": server["name"], "idexp": int(versions[0]), "db": database})
    manifest = {
        "dataset_name": "zenodo_record_08_analysis_ready_canonical",
        "source": "06_experiment_config/g*/server_config.json and 05_browser_telemetry/sessions.csv",
        "groups": groups,
    }
    path = WORK / "source_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def model_inputs(zenodo: Path) -> Path:
    """Copy the deposited criticism labels and downvote reply-event panel."""

    _copy(zenodo / "07_annotations/llm_annotations/criticism_labels.csv", WORK / "annotations/criticism_labels.csv")
    _copy(zenodo / "08_analysis_ready/model_inputs/rq3_vote_panel.csv", WORK / "model_inputs/rq3_vote_panel.csv")
    return WORK


def simulation_source(zenodo: Path) -> Path:
    """Rebuild the directory read by reply_observed_design.py.

    The script reads the M1 model frame and takes four variance scales from
    the posterior of the 15 July M1 fit. The record holds the frame and those
    scales; a one-draw posterior file carries the scales to the script.
    """

    inputs = zenodo / "08_analysis_ready/model_inputs"
    # round_trip parsing returns the exact float64 values of the original frame.
    frame = pd.read_csv(inputs / "reply_observed_design_frame.csv", float_precision="round_trip")
    spec = json.loads((inputs / "reply_observed_design_scales.json").read_text())
    (SIMULATION_SOURCE / "tables").mkdir(parents=True, exist_ok=True)
    (SIMULATION_SOURCE / "models").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(SIMULATION_SOURCE / "tables/model_frame.parquet", index=False)
    scales = spec["scales"]
    shape = {"chain": [0], "draw": [0]}
    posterior = xr.Dataset(
        {
            "sigma_receiver_effect": (("chain", "draw", "receiver_type"),
                                      np.array([[[scales["receiver_h"], scales["receiver_ai"]]]])),
            "sigma_actor_ai_effect": (("chain", "draw"), np.array([[scales["actor"]]])),
            "sigma_topic_ai_effect": (("chain", "draw"), np.array([[scales["topic"]]])),
        },
        coords={**shape, "receiver_type": ["H", "AI"], "term": spec["terms"]},
    )
    posterior.to_netcdf(SIMULATION_SOURCE / "models/M1.nc", group="posterior")
    return SIMULATION_SOURCE


def write_configs(processed: Path, legacy: Path | None, browser: Path) -> list[Path]:
    base = yaml.safe_load((HERE / "analysis.yaml").read_text())
    written = []
    for name, stage in STAGES.items():
        config = json.loads(json.dumps(base))
        analysis = config["analysis"]
        analysis["inputs"] = {
            "processed_dir": str(processed.relative_to(HERE)),
            "source_manifest": str((WORK / "source_manifest.json").relative_to(HERE)),
            "browser_dir": str(browser.relative_to(HERE)),
            "legacy_tables_dir": str(legacy.relative_to(HERE)) if stage["topics"] and legacy else None,
        }
        analysis["output_root"] = OUTPUT_ROOT
        analysis["analysis_stage"] = stage["analysis_stage"]
        config["reply_observed_design_sensitivity"]["source_directory"] = str(
            SIMULATION_SOURCE.relative_to(HERE)
        )
        path = HERE / f"analysis.public-{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zenodo", type=Path, required=True, help="Unpacked Zenodo record (version 1.1.0 or later).")
    args = parser.parse_args()
    zenodo = args.zenodo.expanduser().resolve()
    processed = canonical_tables(zenodo)
    legacy = topic_panel(zenodo)
    browser = browser_tables(zenodo)
    source_manifest(zenodo)
    model_inputs(zenodo)
    if (zenodo / "08_analysis_ready/model_inputs/reply_observed_design_frame.csv").exists():
        simulation_source(zenodo)
    else:
        print("record has no simulation inputs; skipped reply_observed_design_source")
    for path in write_configs(processed, legacy, browser):
        print(f"wrote {path.relative_to(HERE)}")
    print(f"inputs under {WORK.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
