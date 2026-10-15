"""Fit the direct-parent criticism model from the public record.

The study prepared this model with ``scripts/diagnostics/sol_criticism.py``,
which authenticates the private annotation run directories and cannot run
outside the study repository. This entry point builds the same panel from the
public canonical contents table and the deposited criticism labels
(``07_annotations/llm_annotations/criticism_labels.csv``), writes it in the
layout that ``scripts/diagnostics/annotation_models.py`` reads, and then calls
that module's unchanged fitting worker (model, priors, seed, sampler settings,
estimands and predictive checks).

    python public_inputs.py --zenodo /path/to/record
    python public_criticism.py prepare --output public_runs/criticism
    python public_criticism.py fit --output public_runs/criticism --variant primary
    python public_criticism.py fit --output public_runs/criticism --variant doubled_prior
    python public_criticism.py fit --output public_runs/criticism --variant equal_participant

``fit`` uses the sampler settings and seed in ``analysis.yaml`` (4 chains,
1,000 warm-up and 1,000 retained draws, 2 worker processes). ``--draws`` and
``--tune`` shorten a check run; results then go to a separate output folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

# Match annotation_models.py: one numerical thread per chain process.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "analysis.public-rq3.yaml"
DEFAULT_LABELS = HERE / "public_inputs/annotations/criticism_labels.csv"
CATEGORIES = ("CONTENT", "STYLE_CONDUCT", "PARTICIPATION")
PANEL_COLUMNS = ["row_id", "item_id", "group", "target_type", "participant", "context", "target", "machine_label"]
EXPECTED = {"rows": 2547, "positive": 299, "G1": 1508, "G2": 1039}


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(HERE))


def _source_files(config_path: Path, labels: Path, processed: Path) -> list[Path]:
    contents = processed / "snapshot_contents.parquet"
    if not contents.exists():
        contents = processed / "snapshot_contents.csv"
    scripts = HERE / "scripts"
    return [
        Path(__file__).resolve(),
        config_path,
        labels,
        contents,
        HERE / "analysis.yaml",
        *[scripts / "diagnostics" / name for name in (
            "annotation_models.py", "annotation_panels.py", "annotation_descriptives.py",
            "sol_criticism.py", "rq3_interaction.py")],
        *[scripts / "publication_analysis" / name for name in ("config.py", "data.py", "study.py", "io.py")],
    ]


def direct_parent_labels(labels: pd.DataFrame) -> pd.Series:
    """Recompute the model outcome with the study's rule (sol_criticism.parent_outcome)."""

    from scripts.diagnostics.sol_criticism import parent_outcome

    outcomes = []
    for row in labels.itertuples(index=False):
        present = [c for c in CATEGORIES if getattr(row, f"{c.lower()}_criticism") == "YES"]
        targets = {c: getattr(row, f"{c.lower()}_target") for c in present}
        outcomes.append(parent_outcome(row.criticism_label, ";".join(present), json.dumps(targets)))
    return pd.Series(outcomes, index=labels.index)


def build_criticism_panel(contents: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The criticism panel of annotation_panels.build_panels with the deposited labels.

    ``contents`` is the canonical frame from ``data._valid_contents`` with
    ``row_id`` equal to its position; ``contribution_index`` in the labels
    refers to the same positions.
    """

    from scripts.diagnostics.annotation_descriptives import response_links
    from scripts.diagnostics.annotation_panels import _codes

    f = contents.copy()
    if f[["group", "content_id", "author_id", "root_id"]].isna().any().any():
        raise ValueError("Missing contribution, account or thread identity")
    if f.duplicated(["group", "content_id"]).any():
        raise ValueError("Duplicate contribution identities")
    # The deposited labels must refer to the same rows and cover every participant comment.
    joined = labels.merge(f[["row_id", "group", "content_id", "author_type"]], left_on="contribution_index",
                          right_on="row_id", how="left", validate="one_to_one", suffixes=("", "_frame"))
    if joined.row_id.isna().any() or not joined.group.eq(joined.group_frame).all() \
            or not joined.content_id.eq(joined.content_id_frame).all() or not joined.author_type.eq("H").all():
        raise ValueError("criticism_labels.csv does not match the canonical contents frame")
    comments = f.loc[f.author_type.eq("H") & f.comment_to.notna() & f.comment_to.ne(-1)]
    if set(labels.contribution_index) != set(comments.row_id):
        raise ValueError("Criticism labels must cover the complete participant-comment cohort")
    labels = labels.assign(machine_label=direct_parent_labels(labels))
    in_model = labels.in_criticism_model.astype(str).str.lower().eq("true")
    if not labels.loc[in_model, "machine_label"].eq(labels.loc[in_model, "direct_parent_criticism"]).all():
        raise ValueError("Recomputed direct-parent outcome differs from the deposited one")

    # As in annotation_panels.build_panels (criticism panel only).
    f["account_code"] = _codes(f, "author_id")
    f["thread_code"] = _codes(f, "root_id")
    links = response_links(f)
    parents = f[["group", "content_id", "row_id", "account_code"]].rename(columns={
        "content_id": "parent_content_id", "row_id": "parent_row_id", "account_code": "parent_account_code"})
    links = links.merge(parents, on=["group", "parent_content_id"], validate="many_to_one")
    critique = links.loc[links.author_type.eq("H")].copy()
    critique["item_id"] = critique.row_id
    critique = critique.rename(columns={"account_code": "participant", "thread_code": "context",
                                        "parent_account_code": "target", "parent_author_type": "target_type"})
    # As in sol_criticism.attach_labels.
    critique = critique.merge(labels[["contribution_index", "machine_label"]], left_on="row_id",
                              right_on="contribution_index", how="left", validate="one_to_one")
    if critique.machine_label.isna().any():
        raise ValueError("An eligible reply lacks a criticism label")
    critique = critique[PANEL_COLUMNS].reset_index(drop=True)
    if set(critique.row_id) != set(labels.loc[in_model, "contribution_index"]):
        raise ValueError("Eligible replies differ from in_criticism_model in criticism_labels.csv")
    return critique


def prepare(args: argparse.Namespace) -> None:
    from scripts.diagnostics.annotation_panels import cell_summary, file_hash, save_json, select_panel
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new output folder")
    config_path = args.config.resolve()
    labels_path = args.labels.resolve()
    config = load_config(config_path, profile="release", run_id="public_criticism")
    contents = _valid_contents(config).reset_index(drop=True).rename_axis("row_id").reset_index()
    labels = pd.read_csv(labels_path)
    panel = build_criticism_panel(contents, labels)
    # annotation_models reads three panels; only the criticism panel is used here.
    panels = {"uptake": panel.iloc[:0], "criticism": panel, "rare": panel.iloc[:0]}
    frame, audit = select_panel(panels, "criticism", "primary")
    counts = frame.group.value_counts()
    observed = {"rows": len(frame), "positive": int(frame.y.sum()), "G1": int(counts.get("G1", 0)), "G2": int(counts.get("G2", 0))}
    if observed != EXPECTED:
        raise ValueError(f"Criticism panel {observed} differs from the reported panel {EXPECTED}")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, table in panels.items():
        path = output / f"{name}.parquet"
        table.to_parquet(path, index=False)
        hashes[path.name] = file_hash(path)
    cells = cell_summary(frame)
    cells.to_csv(output / "planned_cells.csv", index=False)
    audit.to_csv(output / "planned_attrition.csv", index=False)
    sources = {_relative(p): file_hash(p) for p in _source_files(config_path, labels_path, config.processed_dir)}
    save_json(output / "prepared.json", dict(
        sources=sources, panels=hashes, rows={k: len(v) for k, v in panels.items()},
        annotation_comments=len(labels), analysis="criticism",
        outcome="Any CONTENT, STYLE_CONDUCT or PARTICIPATION category directed at the direct parent",
        other_targets="Criticism solely of other, general or unresolved targets counts NO for the direct-parent outcome",
        inputs="public record: 08_analysis_ready/canonical and 07_annotations/llm_annotations/criticism_labels.csv",
        interpretation="Exploratory ecological associations; one community per condition"))
    print(cells[["cell", "observations", "positive", "participants", "contexts", "target_accounts"]].to_string(index=False), flush=True)


def fit(args: argparse.Namespace) -> int:
    from scripts.diagnostics import annotation_models

    if args.draws is not None or args.tune is not None:
        configuration = annotation_models._configuration

        def shortened():
            config = configuration()
            sampling = dict(config["sampling"])
            sampling["draws"] = args.draws or sampling["draws"]
            sampling["tune"] = args.tune or sampling["tune"]
            return {**config, "sampling": sampling}

        annotation_models._configuration = shortened
    worker_args = SimpleNamespace(output=args.output.resolve(), analysis="criticism", variant=args.variant,
                                  local_temp=(args.local_temp or args.output / "tmp").resolve(),
                                  benchmark=None, benchmark_draw=None)
    return annotation_models.worker(worker_args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "fit"))
    parser.add_argument("--output", type=Path, default=HERE / "public_runs/criticism")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--variant", choices=("primary", "doubled_prior", "equal_participant"), default="primary")
    parser.add_argument("--draws", type=int, help="Retained draws per chain for a short check (default: analysis.yaml).")
    parser.add_argument("--tune", type=int, help="Warm-up draws per chain for a short check (default: analysis.yaml).")
    parser.add_argument("--local-temp", type=Path, help="Folder for compiled model code (default: OUTPUT/tmp).")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
        return 0
    return fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
