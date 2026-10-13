#!/usr/bin/env python3
"""Complete the standalone RQ2 prediction and exact-refit admission checks.

The authenticated RQ2 core bundle includes ``scripts.publication_analysis``.
This command deliberately lives outside that package, reads the core bundle,
and writes only below ``<core run>/outputs/admission``.  Every fitted posterior
is compacted after its required prediction or estimand is calculated: parameter
draws and sample statistics are retained, while per-choice deterministic and
log-likelihood arrays are removed.

Long work is split into resumable foreground stages.  A normal release run is::

    timeout 24h python -m scripts.diagnostics.rq2_admission \
      --run-id CORE_RUN --stage forward
    timeout 24h python -m scripts.diagnostics.rq2_admission \
      --run-id CORE_RUN --stage refits
    timeout 24h python -m scripts.diagnostics.rq2_admission \
      --run-id CORE_RUN --stage finalize

If a timeout interrupts a stage, rerun the same command.  Completed signed
posteriors are reused.  Fits are sequential; each uses four chains, at most four
worker processes, and one BLAS/OpenMP thread per worker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ADMISSION_SCHEMA = "ysocial.rq2-admission.v1"
ADMISSION_IMPLEMENTATION_FIELD = "admission_implementation_sha256"
CORE_BUNDLE_FIELD = "core_bundle_sha256"
DRAW_CHUNK_SIZE = 32
FORWARD_MODELS = ("M1", "RQ2")
FORWARD_FOLDS = (1, 2, 3)
EXPECTED_REFIT_COUNTS: Mapping[str, int] = {
    "participant": 10,
    "account": 45,
    "thread": 1,
}
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
TABLE_NAMES = {
    "forward_assignments": "rq2_admission_forward_assignments",
    "forward_scalers": "rq2_admission_forward_scalers",
    "forward_predictions": "rq2_admission_forward_predictions",
    "forward_scores": "rq2_admission_forward_scores",
    "forward_comparisons": "rq2_admission_forward_comparisons",
    "forward_diagnostics": "rq2_admission_forward_diagnostics",
    "forward_signatures": "rq2_admission_forward_fit_signatures",
    "refit_plan": "rq2_admission_refit_plan",
    "refits": "rq2_admission_exact_refits",
    "refit_estimands": "rq2_admission_exact_refit_estimands",
    "refit_signatures": "rq2_admission_exact_refit_signatures",
    "forward_authentication": "rq2_admission_forward_authentication",
    "refit_authentication": "rq2_admission_refit_authentication",
    "readiness": "rq2_admission_readiness",
}
STAGE_TABLE_KEYS = tuple(
    key
    for key in TABLE_NAMES
    if key not in {"forward_authentication", "refit_authentication", "readiness"}
)


def configure_runtime(local_temp: Path | None = None) -> Path:
    """Use local temporary storage and one numerical thread per chain worker."""

    root = (
        Path("/tmp") / f"ysocial-rq2-admission-{os.getuid()}"
        if local_temp is None
        else Path(local_temp).expanduser()
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"
    for name in ("TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR"):
        os.environ[name] = str(root)
    tempfile.tempdir = str(root)

    compiledir = root / "pytensor"
    compiledir.mkdir(parents=True, exist_ok=True)
    existing = [
        value
        for value in os.environ.get("PYTENSOR_FLAGS", "").split(",")
        if value and not value.strip().startswith("base_compiledir=")
    ]
    existing.append(f"base_compiledir={compiledir}")
    os.environ["PYTENSOR_FLAGS"] = ",".join(existing)
    return root


def _admission_source_paths() -> tuple[Path, ...]:
    """Return every standalone source file used by admission-specific logic."""

    script = Path(__file__).resolve()
    return (script, script.with_name("rq2_sensitivity.py"))


def admission_implementation_sha256() -> str:
    """Hash admission code together with its standalone validation dependency."""

    digest = hashlib.sha256()
    for path in sorted(_admission_source_paths(), key=lambda item: item.name):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def qualitative_conclusion(ci_low: float, ci_high: float) -> str:
    """Classify one 95% credible interval without a stability threshold."""

    import numpy as np

    low = float(ci_low)
    high = float(ci_high)
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError("Qualitative conclusion requires a finite ordered interval")
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "uncertain"


def _validate_six_estimands(frame: Any, *, label: str) -> None:
    from scripts.diagnostics.rq2_sensitivity import _validate_six_estimands

    _validate_six_estimands(frame, label=label)


def add_qualitative_conclusions(refit: Any, primary: Any) -> Any:
    """Compare every refit cell with its authenticated primary interval class."""

    import pandas as pd

    _validate_six_estimands(primary, label="Authenticated RQ2 primary estimands")
    _validate_six_estimands(refit, label="RQ2 exact-refit estimands")
    keys = ["estimand", "direction", "group", "condition"]
    if primary.duplicated(keys).any() or refit.duplicated(keys).any():
        raise ValueError("RQ2 estimand comparison keys must be unique")
    reference = primary[keys + ["ci_low", "ci_high"]].copy()
    reference["primary_qualitative_conclusion"] = [
        qualitative_conclusion(low, high)
        for low, high in reference[["ci_low", "ci_high"]].itertuples(
            index=False, name=None
        )
    ]
    reference = reference.drop(columns=["ci_low", "ci_high"])
    result = refit.merge(reference, on=keys, how="left", validate="one_to_one")
    if result["primary_qualitative_conclusion"].isna().any():
        raise ValueError("An exact-refit estimand has no authenticated primary cell")
    result["qualitative_conclusion"] = [
        qualitative_conclusion(low, high)
        for low, high in result[["ci_low", "ci_high"]].itertuples(
            index=False, name=None
        )
    ]
    result["qualitative_class_changed"] = result[
        "qualitative_conclusion"
    ].ne(result["primary_qualitative_conclusion"])
    return pd.DataFrame(result)


def _normalized_ids(values: Any) -> Any:
    import numpy as np
    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype("string")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    normalized.loc[integer_like] = (
        numeric.loc[integer_like].round().astype("Int64").astype("string")
    )
    return normalized


def build_refit_plan(
    influence_tables: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int] = EXPECTED_REFIT_COUNTS,
) -> Any:
    """Return the exact, deterministic set of grouped PSIS deletions."""

    import numpy as np
    import pandas as pd

    expected_units = ("participant", "account", "thread")
    if set(expected_counts) != set(expected_units):
        raise ValueError("Expected refit counts must cover participant, account, thread")
    rows: list[dict[str, Any]] = []
    for unit in expected_units:
        frame = influence_tables.get(unit, pd.DataFrame()).copy()
        required = {
            "group",
            "influence_unit",
            "unit_id",
            "choice_sets",
            "pareto_k",
            "needs_refit",
            "threshold",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"RQ2 {unit} influence table is missing: {missing}")
        if not frame["influence_unit"].astype(str).eq(unit).all():
            raise ValueError(f"RQ2 {unit} influence table contains another unit type")
        flagged = frame[frame["needs_refit"].fillna(True).astype(bool)].copy()
        expected = int(expected_counts[unit])
        if len(flagged) != expected:
            raise ValueError(
                f"RQ2 admission requires exactly {expected} flagged {unit} rows; "
                f"observed {len(flagged)}"
            )
        flagged["unit_id"] = _normalized_ids(flagged["unit_id"])
        numeric = flagged[["choice_sets", "pareto_k", "threshold"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = bool(
            np.isfinite(numeric.to_numpy(dtype=float)).all()
            and numeric["choice_sets"].gt(0).all()
            and numeric["pareto_k"].gt(numeric["threshold"]).all()
        )
        if not valid:
            raise ValueError(f"RQ2 {unit} flagged rows contain invalid diagnostics")
        rows.extend(flagged.to_dict("records"))
    result = pd.DataFrame(rows)
    keys = ["influence_unit", "group", "unit_id"]
    if result.duplicated(keys).any():
        raise ValueError("RQ2 exact-refit plan contains duplicate grouped units")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def deletion_choice_ids(
    model_frame: Any,
    *,
    influence_unit: str,
    group: str,
    unit_id: str,
) -> set[str]:
    """Map one PSIS group to the choice events removed by its exact refit.

    Account PSIS groups are defined by the selected receiver.  The deletion
    therefore removes choice events in which that account was selected, rather
    than removing every candidate appearance of that account.
    """

    required = {"choice_set_id", "group", "chosen"}
    if influence_unit == "participant":
        column = "actor_id"
        source = model_frame
    elif influence_unit == "account":
        column = "candidate_author_id"
        source = model_frame[model_frame["chosen"].astype(bool)]
    elif influence_unit == "thread":
        column = "root_id"
        source = model_frame
    else:
        raise ValueError("Influence unit must be participant, account, or thread")
    required.add(column)
    missing = sorted(required.difference(model_frame.columns))
    if missing:
        raise ValueError(f"RQ2 model frame is missing deletion columns: {missing}")
    selected = source[
        source["group"].astype(str).eq(str(group))
        & _normalized_ids(source[column]).eq(str(unit_id))
    ]
    choice_ids = set(selected["choice_set_id"].astype(str))
    if not choice_ids:
        raise ValueError(
            f"No choice events match {influence_unit} deletion {group}::{unit_id}"
        )
    return choice_ids


def compact_inference_data(idata: Any) -> Any:
    """Keep parameter posterior and sample statistics; drop per-choice payload."""

    compact = idata.copy()
    choice_variables = [
        name
        for name, value in compact.posterior.data_vars.items()
        if name == "choice_log_likelihood" or "choice_set" in value.dims
    ]
    if choice_variables:
        compact.posterior = compact.posterior.drop_vars(choice_variables)
    for group in tuple(compact.groups()):
        if group not in {"posterior", "sample_stats"}:
            delattr(compact, group)
    if "beta" not in compact.posterior:
        raise ValueError("Compact RQ2 posterior must retain beta parameter draws")
    if "sample_stats" not in compact.groups():
        raise ValueError("Compact RQ2 posterior must retain sample statistics")
    return compact


def _is_compact(idata: Any) -> bool:
    return bool(
        set(idata.groups()).issubset({"posterior", "sample_stats"})
        and all(
            name != "choice_log_likelihood" and "choice_set" not in value.dims
            for name, value in idata.posterior.data_vars.items()
        )
    )


def _admission_paths(config: Any) -> dict[str, Path]:
    root = config.outputs_dir / "admission"
    return {
        "root": root,
        "tables": root / "tables",
        "models": root / "models",
        "forward_models": root / "models" / "forward",
        "refit_models": root / "models" / "refits",
        "reports": root / "reports",
        "report": root / "reports" / "rq2_admission_report.md",
        "bundle": root / "rq2_admission_bundle.json",
        "status": root / "status.json",
    }


def _table_path(paths: Mapping[str, Path], key: str) -> Path:
    return paths["tables"] / TABLE_NAMES[key]


def _ensure_paths(paths: Mapping[str, Path]) -> None:
    from scripts.publication_analysis.io import ensure_directories

    ensure_directories(
        [
            paths["tables"],
            paths["forward_models"],
            paths["refit_models"],
            paths["reports"],
        ]
    )


def _core_bundle_paths(config: Any, core: Mapping[str, Any]) -> list[Path]:
    """Resolve every file explicitly bound by the authenticated core bundle."""

    bundle_path = config.outputs_dir / "fit_result_bundle.json"
    value = json.loads(bundle_path.read_text(encoding="utf-8"))
    paths = {bundle_path.resolve(), *_admission_source_paths()}
    for section in ("sources", "tables", "posteriors"):
        records = value.get(section, [])
        if not isinstance(records, list):
            raise ValueError(f"Core fit bundle section {section!r} is invalid")
        for record in records:
            if not isinstance(record, Mapping) or not record.get("path"):
                raise ValueError(f"Core fit bundle has an invalid {section} record")
            paths.add(Path(str(record["path"])).resolve())
            fit_record = record.get("fit_signature")
            if isinstance(fit_record, Mapping) and fit_record.get("path"):
                paths.add(Path(str(fit_record["path"])).resolve())
    for path in (
        config.tables_dir / "rq2_model_frame.parquet",
        config.tables_dir / "rq2_model_term_manifest.parquet",
        config.tables_dir / "rq2_primary_estimands.parquet",
        config.tables_dir / "rq2_core_readiness.parquet",
        config.tables_dir / "fit_run_signature.parquet",
        core["main_posterior_path"],
        core["main_posterior_path"].with_suffix(".nc.fit.json"),
    ):
        paths.add(Path(path).resolve())
    return sorted(paths, key=str)


def _release_request(config: Any) -> tuple[int, int, int]:
    draws = int(config.sampling.draws)
    tune = int(config.sampling.tune)
    chains = int(config.sampling.chains)
    if draws < 1_000 or tune < 1_000 or chains != 4:
        raise ValueError(
            "RQ2 admission requires at least 1000 draws, 1000 tune steps, "
            "and exactly four chains"
        )
    return draws, tune, chains


def _admission_posterior_request(
    config: Any,
    data: Any,
    model_name: str,
    *,
    role: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
    core_bundle_sha256: str,
) -> dict[str, Any]:
    from scripts.publication_analysis.pipeline import _posterior_request_signature

    request = _posterior_request_signature(
        config,
        data,
        model=model_name,
        role=role,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )
    request.update(
        {
            ADMISSION_IMPLEMENTATION_FIELD: admission_implementation_sha256(),
            CORE_BUNDLE_FIELD: core_bundle_sha256,
            "core_run_id": config.run_id,
            "compact_posterior_policy": (
                "parameter_posterior_and_sample_stats_without_choice_payload"
            ),
        }
    )
    return request


def _sample_admission_model(
    config: Any,
    data: Any,
    model_name: str,
    *,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> Any:
    from scripts.publication_analysis.model import (
        build_primary_pymc_model,
        fit_model,
    )

    model = build_primary_pymc_model(data, config, model_spec=model_name)
    return fit_model(
        model,
        config,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        progressbar=False,
        cores=4,
    )


def _fit_or_resume(
    config: Any,
    data: Any,
    model_name: str,
    path: Path,
    *,
    role: str,
    seed_offset: int,
    resume: bool,
    core_bundle_sha256: str,
) -> tuple[Any, dict[str, Any], bool, dict[str, Any]]:
    """Return a compatible fit after at most one bounded MCMC extension.

    The release request is fitted first.  When its parameter diagnostics fail,
    a compact checkpoint is written and one fit with twice the requested draws
    and tuning is attempted.  A resumed posterior whose signed request already
    exceeds the release length is never extended again.
    """

    from scripts.publication_analysis.pipeline import _load_compatible_posterior
    from scripts.publication_analysis.validation import summarize_mcmc

    draws, tune, chains = _release_request(config)
    random_seed = config.seed + seed_offset
    request = _admission_posterior_request(
        config,
        data,
        model_name,
        role=role,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        core_bundle_sha256=core_bundle_sha256,
    )
    idata = None
    active_request = request
    if resume:
        idata = _load_compatible_posterior(
            path,
            draws=draws,
            chains=chains,
            request_signature=request,
        )
    if idata is not None:
        saved_signature = json.loads(
            path.with_suffix(path.suffix + ".fit.json").read_text(
                encoding="utf-8"
            )
        )
        active_request = dict(saved_signature["request"])
        reused = True
    else:
        idata = _sample_admission_model(
            config,
            data,
            model_name,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        reused = False

    diagnostic = dict(summarize_mcmc(idata))
    already_extended = bool(
        int(active_request.get("draws", 0)) > draws
        or int(active_request.get("tune", 0)) > tune
    )
    if bool(diagnostic["passed"]) or already_extended:
        diagnostic["sampling_extended"] = already_extended
        return idata, active_request, reused, diagnostic

    _persist_compact_posterior(idata, path, active_request, reused=reused)
    close = getattr(idata, "close", None)
    if callable(close):
        close()
    del idata
    gc.collect()
    print(
        f"[diagnostic extension] role={role} base_draws={draws} "
        f"base_tune={tune} mcmc=fail action=single_2x_refit",
        flush=True,
    )

    extended_request = _admission_posterior_request(
        config,
        data,
        model_name,
        role=role,
        draws=2 * draws,
        tune=2 * tune,
        chains=chains,
        random_seed=random_seed,
        core_bundle_sha256=core_bundle_sha256,
    )
    idata = _sample_admission_model(
        config,
        data,
        model_name,
        draws=2 * draws,
        tune=2 * tune,
        chains=chains,
        random_seed=random_seed,
    )
    diagnostic = dict(summarize_mcmc(idata))
    diagnostic["sampling_extended"] = True
    return idata, extended_request, False, diagnostic


def _persist_compact_posterior(
    idata: Any,
    path: Path,
    request: Mapping[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    """Atomically save or normalize the compact signed posterior."""

    from scripts.publication_analysis.pipeline import _write_posterior_atomic

    if reused and _is_compact(idata):
        return json.loads(
            path.with_suffix(path.suffix + ".fit.json").read_text(encoding="utf-8")
        )
    compact = compact_inference_data(idata)
    try:
        signature = _write_posterior_atomic(compact, path, request)
    finally:
        close = getattr(compact, "close", None)
        if callable(close):
            close()
    return signature


def _prepare_data(frame: Any, manifest: Any, model_name: str) -> Any:
    from scripts.publication_analysis.model import prepare_model_data
    from scripts.publication_analysis.terms import included_terms_for_model

    available = included_terms_for_model(model_name, manifest)
    numeric = tuple(term for term in available if term != "candidate_target_is_ai")
    return prepare_model_data(frame, numeric, ai_term="candidate_target_is_ai")


def _forward_assignments(model_frame: Any) -> Any:
    from scripts.publication_analysis.validation import rolling_origin_assignments

    assignments = rolling_origin_assignments(model_frame)
    if set(assignments["fold"].astype(int)) != set(FORWARD_FOLDS):
        raise ValueError("RQ2 forward validation requires exactly three folds")
    cells = assignments.groupby(["fold", "group", "role"])[
        "choice_set_id"
    ].nunique()
    expected = {
        (fold, group, role)
        for fold in FORWARD_FOLDS
        for group in ("G1", "G2")
        for role in ("train", "test")
    }
    if set(cells.index) != expected or not cells.gt(0).all():
        raise ValueError("RQ2 forward folds require nonempty train/test cells per arm")
    duplicate = assignments.duplicated(["fold", "choice_set_id"], keep=False)
    if duplicate.any():
        raise ValueError("A choice set has multiple roles within one forward fold")
    return assignments


def _forward_fold(
    model_frame: Any,
    assignments: Any,
    fold: int,
) -> tuple[Any, Any, Any, set[str], set[str]]:
    from scripts.publication_analysis.terms import (
        apply_model_term_manifest,
        build_model_frame,
    )

    split = assignments[assignments["fold"].astype(int).eq(int(fold))]
    train_ids = set(
        split.loc[split["role"].eq("train"), "choice_set_id"].astype(str)
    )
    test_ids = set(
        split.loc[split["role"].eq("test"), "choice_set_id"].astype(str)
    )
    if not train_ids or not test_ids or train_ids.intersection(test_ids):
        raise ValueError(f"RQ2 forward fold {fold} has invalid train/test choices")
    choice_text = model_frame["choice_set_id"].astype(str)
    train_raw = model_frame[choice_text.isin(train_ids)]
    _, manifest = build_model_frame(train_raw, model_name="RQ2")
    transformed = apply_model_term_manifest(model_frame, manifest)
    transformed_ids = transformed["choice_set_id"].astype(str)
    train = transformed[transformed_ids.isin(train_ids)].copy()
    test = transformed[transformed_ids.isin(test_ids)].copy()
    if set(train["choice_set_id"].astype(str)) != train_ids:
        raise RuntimeError(f"RQ2 forward fold {fold} lost training choices")
    if set(test["choice_set_id"].astype(str)) != test_ids:
        raise RuntimeError(f"RQ2 forward fold {fold} lost future choices")
    return train, test, manifest, train_ids, test_ids


def _validate_prediction_rows(predictions: Any, expected_ids: set[str]) -> None:
    import numpy as np

    required = {
        "choice_set_id",
        "group",
        "actor_label",
        "chosen_is_ai",
        "log_score",
        "selected_target_rank",
        "predicted_ai_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Forward predictions are missing fields: {missing}")
    observed = set(predictions["choice_set_id"].astype(str))
    if observed != expected_ids or predictions["choice_set_id"].astype(str).duplicated().any():
        raise ValueError("Forward predictions do not contain the exact future choices")
    numeric = predictions[
        ["log_score", "selected_target_rank", "predicted_ai_probability"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Forward predictions contain nonfinite values")


def _participant_clustered_total_se(values: Any, participants: Any) -> tuple[float, int]:
    import numpy as np
    import pandas as pd

    totals = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "participant": participants}
    ).groupby("participant", sort=False)["value"].sum()
    if len(totals) < 2:
        return float("nan"), int(len(totals))
    return float(np.sqrt(len(totals) * totals.var(ddof=1))), int(len(totals))


def _forward_scores(predictions: Any, assignments: Any) -> Any:
    import numpy as np
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for (fold, model), fitted in predictions.groupby(["fold", "model"], sort=True):
        split = assignments[assignments["fold"].astype(int).eq(int(fold))]
        train_counts = split[split["role"].eq("train")].groupby("group")[
            "choice_set_id"
        ].nunique()
        for group, part in [*fitted.groupby("group", sort=True), ("ALL", fitted)]:
            error = part["predicted_ai_probability"].astype(float) - part[
                "chosen_is_ai"
            ].astype(float)
            rows.append(
                {
                    "fold": int(fold),
                    "model": str(model),
                    "group": str(group),
                    "train_choice_sets": (
                        int(len(split[split["role"].eq("train")]["choice_set_id"].unique()))
                        if group == "ALL"
                        else int(train_counts.get(group, 0))
                    ),
                    "test_choice_sets": int(len(part)),
                    "mean_future_log_score": float(part["log_score"].mean()),
                    "median_selected_target_rank": float(
                        part["selected_target_rank"].median()
                    ),
                    "ai_target_brier_score": float(np.mean(error**2)),
                    "scaling_scope": "fold_training_candidate_rows_only",
                    "prediction_scope": (
                        "one_step_ahead_conditional_on_realized_earlier_heldout_history"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _forward_comparisons(predictions: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for fold_value in (*FORWARD_FOLDS, "ALL"):
        fitted = (
            predictions
            if fold_value == "ALL"
            else predictions[predictions["fold"].astype(int).eq(int(fold_value))]
        )
        reference = fitted[fitted["model"].eq("RQ2")]
        baseline = fitted[fitted["model"].eq("M1")]
        keys = ["fold", "choice_set_id", "group", "actor_label", "chosen_is_ai"]
        reference_keys = set(map(tuple, reference[keys].itertuples(index=False, name=None)))
        baseline_keys = set(map(tuple, baseline[keys].itertuples(index=False, name=None)))
        if reference_keys != baseline_keys:
            raise ValueError("M1 and RQ2 were not scored on identical future choices")
        paired = reference.merge(
            baseline,
            on=keys,
            suffixes=("_rq2", "_m1"),
            validate="one_to_one",
        )
        delta = paired["log_score_rq2"] - paired["log_score_m1"]
        difference = float(delta.sum())
        standard_error, clusters = _participant_clustered_total_se(
            delta, paired["actor_label"]
        )
        rows.append(
            {
                "fold": "ALL" if fold_value == "ALL" else str(int(fold_value)),
                "reference_model": "RQ2",
                "comparator_model": "M1",
                "test_choice_sets": int(len(paired)),
                "elpd_difference": difference,
                "elpd_difference_se": standard_error,
                "uncertainty_clusters": clusters,
                "uncertainty_unit": "participant_within_arm",
                "comparison_scope": (
                    "identical_future_choice_sets_with_fold_training_scalers"
                ),
            }
        )
    return pd.DataFrame(rows)


def _authenticate_signatures(
    signatures: Any,
    *,
    expected_rows: int,
    core_bundle_sha256: str,
) -> Any:
    from scripts.publication_analysis.validation import authenticate_fit_signatures

    authenticated = authenticate_fit_signatures(signatures)
    implementation = admission_implementation_sha256()
    valid = bool(
        len(authenticated) == expected_rows
        and authenticated["live_posterior_authenticated"].fillna(False).astype(bool).all()
        and authenticated["dimension_match"].fillna(False).astype(bool).all()
        and authenticated["publication_length_request"].fillna(False).astype(bool).all()
        and authenticated[ADMISSION_IMPLEMENTATION_FIELD]
        .astype(str)
        .eq(implementation)
        .all()
        and authenticated[CORE_BUNDLE_FIELD]
        .astype(str)
        .eq(core_bundle_sha256)
        .all()
        and authenticated["actual_chains"].astype(int).eq(4).all()
    )
    if not valid:
        raise ValueError("Admission posterior signatures failed authentication")
    return authenticated


def _release_signature_status(
    signature: Mapping[str, Any],
    *,
    expected_model_implementation_sha256: str,
) -> list[str]:
    """Reassert release-length and implementation requirements at finalize."""

    status: list[str] = []
    if not bool(signature["live_posterior_authenticated"]):
        status.append("signature_not_authenticated")
    if not bool(signature["dimension_match"]):
        status.append("posterior_dimension_mismatch")
    if not bool(signature["publication_length_request"]):
        status.append("nonpublication_length_request")
    if int(signature["actual_chains"]) != 4:
        status.append("actual_chain_count_mismatch")
    if (
        str(signature["implementation_sha256"])
        != str(expected_model_implementation_sha256)
    ):
        status.append("model_implementation_mismatch")
    return status


def _write_tables(paths: Mapping[str, Path], tables: Mapping[str, Any]) -> list[Path]:
    from scripts.publication_analysis.io import write_table

    artifacts: list[Path] = []
    for key, frame in tables.items():
        written = write_table(frame, _table_path(paths, key))
        artifacts.extend(Path(value) for value in written.values())
    return artifacts


def _normalize_structured_cells(frame: Any) -> Any:
    """Canonicalize dictionary/list cells that CSV stores as Python literals."""

    import ast

    normalized = frame.copy()

    def canonical(value: Any) -> Any:
        candidate = value
        if isinstance(candidate, str):
            try:
                candidate = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                return value
        if isinstance(candidate, (Mapping, list, tuple)):
            return json.dumps(candidate, sort_keys=True, default=str)
        return value

    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(canonical)
    return normalized


def _assert_table_pairs_semantically_equal(
    paths: Mapping[str, Path],
    keys: Sequence[str],
) -> None:
    """Require every generated Parquet/CSV pair to encode the same table."""

    import pandas as pd

    for key in keys:
        stem = _table_path(paths, key)
        parquet_path = stem.with_suffix(".parquet")
        csv_path = stem.with_suffix(".csv")
        if not parquet_path.is_file() or not csv_path.is_file():
            raise ValueError(f"Admission table pair is incomplete for {key!r}")
        parquet = _normalize_structured_cells(pd.read_parquet(parquet_path))
        csv = _normalize_structured_cells(pd.read_csv(csv_path))
        try:
            pd.testing.assert_frame_equal(
                parquet,
                csv,
                check_dtype=False,
                check_exact=False,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        except AssertionError as error:
            raise ValueError(
                f"Admission Parquet/CSV mismatch for {key!r}"
            ) from error


def run_forward_stage(
    config: Any,
    core: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    resume: bool,
    core_bundle_sha256: str,
) -> dict[str, Any]:
    """Fit M1 and RQ2 sequentially on the same three future-choice folds."""

    import pandas as pd

    from scripts.publication_analysis.pipeline import (
        _heldout_choice_predictions,
        _signature_row,
    )
    assignments = _forward_assignments(core["model_frame"])
    prediction_frames: list[Any] = []
    scaler_frames: list[Any] = []
    diagnostic_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    completed = 0
    total = len(FORWARD_FOLDS) * len(FORWARD_MODELS)
    for fold in FORWARD_FOLDS:
        train_frame, test_frame, manifest, train_ids, test_ids = _forward_fold(
            core["model_frame"], assignments, fold
        )
        scalers = manifest.copy()
        scalers.insert(0, "test_choice_sets", len(test_ids))
        scalers.insert(0, "train_choice_sets", len(train_ids))
        scalers.insert(0, "fold", fold)
        scaler_frames.append(scalers)
        for model_index, model_name in enumerate(FORWARD_MODELS):
            train_data = _prepare_data(train_frame, manifest, model_name)
            test_data = _prepare_data(test_frame, manifest, model_name)
            role = f"rq2_admission_forward_fold_{fold}_{model_name.lower()}"
            path = paths["forward_models"] / f"{model_name}_forward_fold{fold}.nc"
            idata, request, reused, diagnostic = _fit_or_resume(
                config,
                train_data,
                model_name,
                path,
                role=role,
                seed_offset=30_000 + 100 * fold + model_index,
                resume=resume,
                core_bundle_sha256=core_bundle_sha256,
            )
            try:
                predictions = _heldout_choice_predictions(
                    idata,
                    train_data,
                    test_data,
                    integrate_unseen_hierarchical=True,
                    random_seed=config.seed + 40_000 + 100 * fold + model_index,
                    include_population_mean_sensitivity=False,
                )
                _validate_prediction_rows(predictions, test_ids)
                predictions.insert(0, "model", model_name)
                predictions.insert(0, "fold", fold)
                prediction_frames.append(predictions)
                signature = _persist_compact_posterior(
                    idata, path, request, reused=reused
                )
                row = _signature_row(path.resolve(), signature, train_data)
                row.update(
                    {
                        "fold": fold,
                        "forward_training_data_match": True,
                        "compact_posterior": True,
                    }
                )
                signature_rows.append(row)
                diagnostic_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "role": role,
                        "resumed": reused,
                        **diagnostic,
                    }
                )
            finally:
                close = getattr(idata, "close", None)
                if callable(close):
                    close()
                del idata
                gc.collect()
            completed += 1
            print(
                f"[forward {completed}/{total}] fold={fold} model={model_name} "
                f"source={'resume' if reused else 'fit'} "
                f"extended={bool(diagnostic['sampling_extended'])} "
                f"mcmc={'pass' if diagnostic['passed'] else 'fail'}",
                flush=True,
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    signatures = _authenticate_signatures(
        pd.DataFrame(signature_rows),
        expected_rows=total,
        core_bundle_sha256=core_bundle_sha256,
    )
    tables = {
        "forward_assignments": assignments,
        "forward_scalers": pd.concat(scaler_frames, ignore_index=True),
        "forward_predictions": predictions,
        "forward_scores": _forward_scores(predictions, assignments),
        "forward_comparisons": _forward_comparisons(predictions),
        "forward_diagnostics": pd.DataFrame(diagnostic_rows),
        "forward_signatures": signatures,
    }
    _write_tables(paths, tables)
    return {
        "stage": "forward",
        "completed_fits": total,
        "expected_fits": total,
        "tables": [TABLE_NAMES[key] for key in tables],
    }


def _project_rq2_estimands(
    idata: Any,
    data: Any,
    term_manifest: Any,
    config: Any,
    *,
    draw_chunk_size: int = DRAW_CHUNK_SIZE,
) -> Any:
    from scripts.publication_analysis.rq2 import (
        add_estimand_denominators,
        standardized_estimands,
    )

    if draw_chunk_size != DRAW_CHUNK_SIZE:
        raise ValueError("RQ2 admission projections use a fixed 32-draw bound")
    estimands = standardized_estimands(
        idata,
        data,
        term_manifest,
        reference_hours=24.0,
        draw_chunk_size=draw_chunk_size,
    )
    estimands = add_estimand_denominators(estimands, data, config)
    _validate_six_estimands(estimands, label="RQ2 exact-refit estimands")
    return estimands


def _influence_tables(config: Any) -> dict[str, Any]:
    from scripts.publication_analysis.io import read_table

    return {
        unit: read_table(config.tables_dir / f"rq2_influence_{unit}")
        for unit in ("participant", "account", "thread")
    }


def run_refit_stage(
    config: Any,
    core: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    resume: bool,
    core_bundle_sha256: str,
    expected_counts: Mapping[str, int] = EXPECTED_REFIT_COUNTS,
) -> dict[str, Any]:
    """Run the exact grouped deletions sequentially and project all six cells."""

    import numpy as np
    import pandas as pd

    from scripts.publication_analysis.pipeline import (
        _prepared_data_hash,
        _rq2_model_data,
        _signature_row,
    )
    plan = build_refit_plan(
        _influence_tables(config), expected_counts=expected_counts
    )
    refit_rows: list[dict[str, Any]] = []
    estimand_frames: list[Any] = []
    signature_rows: list[dict[str, Any]] = []
    total = len(plan)
    for index, record in enumerate(plan.to_dict("records")):
        unit = str(record["influence_unit"])
        group = str(record["group"])
        unit_id = str(record["unit_id"])
        deleted = deletion_choice_ids(
            core["model_frame"],
            influence_unit=unit,
            group=group,
            unit_id=unit_id,
        )
        if len(deleted) != int(record["choice_sets"]):
            raise ValueError(
                f"{unit} deletion {group}::{unit_id} matched {len(deleted)} "
                f"choices; expected {int(record['choice_sets'])}"
            )
        retained = core["model_frame"][
            ~core["model_frame"]["choice_set_id"].astype(str).isin(deleted)
        ].copy()
        data = _rq2_model_data(retained, core["term_manifest"])
        key = f"{unit}::{group}::{unit_id}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
        role = f"rq2_admission_exact_delete_{unit}_{group}_{key_hash}"
        path = paths["refit_models"] / (
            f"RQ2_exact_delete_{unit}_{group}_{key_hash}.nc"
        )
        idata, request, reused, diagnostic = _fit_or_resume(
            config,
            data,
            "RQ2",
            path,
            role=role,
            seed_offset=50_000 + index,
            resume=resume,
            core_bundle_sha256=core_bundle_sha256,
        )
        try:
            estimands = _project_rq2_estimands(
                idata,
                data,
                core["term_manifest"],
                config,
                draw_chunk_size=DRAW_CHUNK_SIZE,
            )
            estimands = add_qualitative_conclusions(
                estimands, core["primary_estimands"]
            )
            estimands.insert(0, "deleted_unit_id", unit_id)
            estimands.insert(0, "deleted_unit_group", group)
            estimands.insert(0, "influence_unit", unit)
            estimands["deleted_choice_sets"] = len(deleted)
            estimands["psis_pareto_k"] = float(record["pareto_k"])
            finite = bool(
                np.isfinite(
                    estimands[
                        [
                            "posterior_mean",
                            "posterior_median",
                            "ci_low",
                            "ci_high",
                            "posterior_probability_gt_zero",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            )
            if not finite:
                raise ValueError(f"Exact refit {key} produced nonfinite estimands")
            estimand_frames.append(estimands)
            signature = _persist_compact_posterior(
                idata, path, request, reused=reused
            )
            signature_row = _signature_row(path.resolve(), signature, data)
            signature_row.update(
                {
                    "influence_unit": unit,
                    "deleted_unit_group": group,
                    "deleted_unit_id": unit_id,
                    "compact_posterior": True,
                }
            )
            signature_rows.append(signature_row)
            refit_rows.append(
                {
                    "influence_unit": unit,
                    "group": group,
                    "unit_id": unit_id,
                    "choice_sets": len(deleted),
                    "psis_pareto_k": float(record["pareto_k"]),
                    "refit_choice_sets": int(data.n_choices),
                    "refit_candidate_rows": int(data.mask.sum()),
                    "refit_data_sha256": _prepared_data_hash(data),
                    "estimand_rows": len(estimands),
                    "estimands_finite": finite,
                    "qualitative_class_changes": int(
                        estimands["qualitative_class_changed"].sum()
                    ),
                    "mcmc_passed": bool(diagnostic["passed"]),
                    "sampling_extended": bool(diagnostic["sampling_extended"]),
                    "max_rhat": diagnostic["max_rhat"],
                    "min_ess_bulk": diagnostic["min_ess_bulk"],
                    "min_ess_tail": diagnostic["min_ess_tail"],
                    "divergences": diagnostic["divergences"],
                    "posterior_path": str(path.resolve()),
                    "posterior_sha256": signature["actual"]["posterior_sha256"],
                    "signature_role": role,
                    "resumed": reused,
                    "exact_refit_completed": bool(diagnostic["passed"] and finite),
                    "deletion_scope": (
                        "choice_events_grouped_by_psis_with_observed_histories_fixed"
                    ),
                }
            )
        finally:
            close = getattr(idata, "close", None)
            if callable(close):
                close()
            del idata
            gc.collect()
        print(
            f"[refits {index + 1}/{total}] unit={unit} key={group}::{unit_id} "
            f"deleted={len(deleted)} source={'resume' if reused else 'fit'} "
            f"extended={bool(diagnostic['sampling_extended'])} "
            f"mcmc={'pass' if diagnostic['passed'] else 'fail'} "
            f"class_changes={int(estimands['qualitative_class_changed'].sum())}",
            flush=True,
        )
    signatures = _authenticate_signatures(
        pd.DataFrame(signature_rows),
        expected_rows=total,
        core_bundle_sha256=core_bundle_sha256,
    )
    refits = pd.DataFrame(refit_rows)
    estimands = pd.concat(estimand_frames, ignore_index=True)
    expected_estimand_rows = 6 * total
    if len(estimands) != expected_estimand_rows:
        raise RuntimeError(
            f"RQ2 exact refits require {expected_estimand_rows} estimand rows"
        )
    tables = {
        "refit_plan": plan,
        "refits": refits,
        "refit_estimands": estimands,
        "refit_signatures": signatures,
    }
    _write_tables(paths, tables)
    return {
        "stage": "refits",
        "completed_fits": total,
        "expected_fits": sum(expected_counts.values()),
        "estimand_rows": len(estimands),
        "qualitative_class_changes": int(
            estimands["qualitative_class_changed"].sum()
        ),
        "tables": [TABLE_NAMES[key] for key in tables],
    }


def _read_required_table(paths: Mapping[str, Path], key: str) -> Any:
    from scripts.publication_analysis.io import read_table

    return read_table(_table_path(paths, key))


def _allclose(left: Any, right: Any, columns: Sequence[str]) -> bool:
    import numpy as np

    return bool(
        len(left) == len(right)
        and np.allclose(
            left[list(columns)].to_numpy(dtype=float),
            right[list(columns)].to_numpy(dtype=float),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    )


def _frames_match(left: Any, right: Any, *, sort_keys: Sequence[str]) -> bool:
    """Compare regenerated tables with strict structure and numerical tolerance."""

    import pandas as pd

    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    try:
        pd.testing.assert_frame_equal(
            left.sort_values(list(sort_keys)).reset_index(drop=True),
            right.sort_values(list(sort_keys)).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    except AssertionError:
        return False
    return True


def _validate_forward_stage(
    config: Any,
    core: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[bool, Any]:
    import arviz as az
    import numpy as np
    import pandas as pd

    from scripts.publication_analysis.io import sha256_file
    from scripts.publication_analysis.pipeline import (
        _heldout_choice_predictions,
        _prepared_data_hash,
    )
    from scripts.publication_analysis.validation import (
        authenticate_fit_signatures,
        summarize_mcmc,
    )

    assignments = _read_required_table(paths, "forward_assignments")
    scalers = _read_required_table(paths, "forward_scalers")
    predictions = _read_required_table(paths, "forward_predictions")
    scores = _read_required_table(paths, "forward_scores")
    comparisons = _read_required_table(paths, "forward_comparisons")
    diagnostics = _read_required_table(paths, "forward_diagnostics")
    signatures = authenticate_fit_signatures(
        _read_required_table(paths, "forward_signatures")
    )
    core_bundle_sha256 = sha256_file(
        config.outputs_dir / "fit_result_bundle.json"
    )
    live_assignments = _forward_assignments(core["model_frame"])
    assignment_columns = ["fold", "group", "choice_set_id", "role"]

    def assignment_keys(frame: Any) -> set[tuple[int, str, str, str]]:
        return {
            (int(row.fold), str(row.group), str(row.choice_set_id), str(row.role))
            for row in frame[assignment_columns].itertuples(index=False)
        }

    assignments_match = bool(
        len(assignments) == len(live_assignments)
        and assignment_keys(assignments) == assignment_keys(live_assignments)
    )
    rows: list[dict[str, Any]] = []
    recomputed_prediction_frames: list[Any] = []
    expected_scaler_frames: list[Any] = []
    expected_pairs = {(fold, model) for fold in FORWARD_FOLDS for model in FORWARD_MODELS}
    observed_pairs = set(
        map(
            tuple,
            signatures[["fold", "model"]]
            .assign(fold=lambda value: value["fold"].astype(int))
            .itertuples(index=False, name=None),
        )
    )
    for fold in FORWARD_FOLDS:
        train, test, manifest, train_ids, test_ids = _forward_fold(
            core["model_frame"], assignments, fold
        )
        expected_scalers = manifest.copy()
        expected_scalers.insert(0, "test_choice_sets", len(test_ids))
        expected_scalers.insert(0, "train_choice_sets", len(train_ids))
        expected_scalers.insert(0, "fold", fold)
        expected_scaler_frames.append(expected_scalers)
        for model_index, model in enumerate(FORWARD_MODELS):
            data = _prepare_data(train, manifest, model)
            test_data = _prepare_data(test, manifest, model)
            signature = signatures[
                signatures["fold"].astype(int).eq(fold)
                & signatures["model"].astype(str).eq(model)
            ]
            status: list[str] = []
            if len(signature) != 1:
                status.append("missing_or_duplicate_signature")
                row = None
            else:
                row = signature.iloc[0]
                status.extend(
                    _release_signature_status(
                        row,
                        expected_model_implementation_sha256=core[
                            "implementation_sha256"
                        ],
                    )
                )
                if str(row["data_sha256"]) != _prepared_data_hash(data):
                    status.append("training_data_hash_mismatch")
                if (
                    str(row[ADMISSION_IMPLEMENTATION_FIELD])
                    != admission_implementation_sha256()
                ):
                    status.append("admission_implementation_mismatch")
                if str(row[CORE_BUNDLE_FIELD]) != core_bundle_sha256:
                    status.append("core_bundle_mismatch")
            predicted = predictions[
                predictions["fold"].astype(int).eq(fold)
                & predictions["model"].astype(str).eq(model)
            ]
            try:
                _validate_prediction_rows(predicted, test_ids)
            except ValueError as error:
                status.append(str(error))
            recomputed: dict[str, Any] = {}
            predictions_match = False
            if row is not None and not status:
                idata = az.from_netcdf(Path(str(row["posterior_path"])))
                try:
                    recomputed = summarize_mcmc(idata)
                    live_predictions = _heldout_choice_predictions(
                        idata,
                        data,
                        test_data,
                        integrate_unseen_hierarchical=True,
                        random_seed=(
                            config.seed + 40_000 + 100 * fold + model_index
                        ),
                        include_population_mean_sensitivity=False,
                    )
                    _validate_prediction_rows(live_predictions, test_ids)
                    live_predictions.insert(0, "model", model)
                    live_predictions.insert(0, "fold", fold)
                    recomputed_prediction_frames.append(live_predictions)
                    predictions_match = _frames_match(
                        predicted,
                        live_predictions,
                        sort_keys=("fold", "model", "choice_set_id"),
                    )
                finally:
                    idata.close()
                if not bool(recomputed["passed"]):
                    status.append("parameter_mcmc_failed")
                if not predictions_match:
                    status.append("heldout_predictions_recompute_mismatch")
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "authenticated": not status,
                    "authentication_status": (
                        "authenticated" if not status else ";".join(status)
                    ),
                    "recomputed_mcmc_passed": bool(recomputed.get("passed", False)),
                    "recomputed_predictions_match": predictions_match,
                }
            )
    expected_scalers = pd.concat(expected_scaler_frames, ignore_index=True)
    scaler_keys = ["fold", "term"]
    scalers_match = False
    if list(scalers.columns) == list(expected_scalers.columns):
        try:
            pd.testing.assert_frame_equal(
                scalers.sort_values(scaler_keys).reset_index(drop=True),
                expected_scalers.sort_values(scaler_keys).reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            scalers_match = True
        except AssertionError:
            pass

    expected_score_cells = {
        (fold, model, group)
        for fold in FORWARD_FOLDS
        for model in FORWARD_MODELS
        for group in ("G1", "G2", "ALL")
    }
    score_cells = set(
        (int(row.fold), str(row.model), str(row.group))
        for row in scores.itertuples(index=False)
    )
    comparison_cells = set(
        (str(row.fold), str(row.reference_model), str(row.comparator_model))
        for row in comparisons.itertuples(index=False)
    )
    finite = bool(
        np.isfinite(
            scores[
                [
                    "mean_future_log_score",
                    "median_selected_target_rank",
                    "ai_target_brier_score",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            comparisons[
                ["elpd_difference", "elpd_difference_se"]
            ].to_numpy(dtype=float)
        ).all()
    )
    expected_comparison_cells = {
        (str(fold), "RQ2", "M1") for fold in FORWARD_FOLDS
    } | {("ALL", "RQ2", "M1")}
    scores_match = False
    comparisons_match = False
    if len(recomputed_prediction_frames) == len(expected_pairs):
        recomputed_predictions = pd.concat(
            recomputed_prediction_frames, ignore_index=True
        )
        scores_match = _frames_match(
            scores,
            _forward_scores(recomputed_predictions, live_assignments),
            sort_keys=("fold", "model", "group"),
        )
        comparisons_match = _frames_match(
            comparisons,
            _forward_comparisons(recomputed_predictions),
            sort_keys=("fold", "reference_model", "comparator_model"),
        )
    result = pd.DataFrame(rows)
    for passed, message in (
        (assignments_match, "canonical_forward_assignments_mismatch"),
        (scalers_match, "fold_training_scalers_mismatch"),
        (scores_match, "forward_scores_recompute_mismatch"),
        (comparisons_match, "forward_comparisons_recompute_mismatch"),
    ):
        if not passed:
            result["authenticated"] = False
            result["authentication_status"] = result[
                "authentication_status"
            ].map(lambda value: f"{value};{message}")
    complete = bool(
        assignments_match
        and scalers_match
        and observed_pairs == expected_pairs
        and score_cells == expected_score_cells
        and comparison_cells == expected_comparison_cells
        and len(diagnostics) == 6
        and finite
        and scores_match
        and comparisons_match
        and result["authenticated"].all()
    )
    return complete, result


def _validate_refit_stage(
    config: Any,
    core: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[bool, Any]:
    import arviz as az
    import pandas as pd

    from scripts.publication_analysis.io import sha256_file
    from scripts.publication_analysis.pipeline import (
        _prepared_data_hash,
        _rq2_model_data,
    )
    from scripts.publication_analysis.validation import (
        authenticate_fit_signatures,
        summarize_mcmc,
    )

    live_plan = build_refit_plan(_influence_tables(config))
    saved_plan = _read_required_table(paths, "refit_plan")
    refits = _read_required_table(paths, "refits")
    saved_estimands = _read_required_table(paths, "refit_estimands")
    signatures = authenticate_fit_signatures(
        _read_required_table(paths, "refit_signatures")
    )
    core_bundle_sha256 = sha256_file(
        config.outputs_dir / "fit_result_bundle.json"
    )
    plan_keys = ["influence_unit", "group", "unit_id"]
    live_keys = set(
        map(tuple, live_plan[plan_keys].astype(str).itertuples(index=False, name=None))
    )
    saved_keys = set(
        map(tuple, saved_plan[plan_keys].astype(str).itertuples(index=False, name=None))
    )
    rows: list[dict[str, Any]] = []
    numeric_columns = (
        "posterior_mean",
        "posterior_median",
        "ci_low",
        "ci_high",
        "posterior_probability_gt_zero",
    )
    for record in live_plan.to_dict("records"):
        unit = str(record["influence_unit"])
        group = str(record["group"])
        unit_id = str(record["unit_id"])
        status: list[str] = []
        deleted = deletion_choice_ids(
            core["model_frame"],
            influence_unit=unit,
            group=group,
            unit_id=unit_id,
        )
        retained = core["model_frame"][
            ~core["model_frame"]["choice_set_id"].astype(str).isin(deleted)
        ]
        data = _rq2_model_data(retained, core["term_manifest"])
        refit = refits[
            refits["influence_unit"].astype(str).eq(unit)
            & refits["group"].astype(str).eq(group)
            & _normalized_ids(refits["unit_id"]).eq(unit_id)
        ]
        signature = signatures[
            signatures["influence_unit"].astype(str).eq(unit)
            & signatures["deleted_unit_group"].astype(str).eq(group)
            & _normalized_ids(signatures["deleted_unit_id"]).eq(unit_id)
        ]
        if len(refit) != 1 or len(signature) != 1:
            status.append("missing_or_duplicate_refit")
            signature_row = None
        else:
            signature_row = signature.iloc[0]
            status.extend(
                _release_signature_status(
                    signature_row,
                    expected_model_implementation_sha256=core[
                        "implementation_sha256"
                    ],
                )
            )
            expected_hash = _prepared_data_hash(data)
            if str(refit.iloc[0]["refit_data_sha256"]) != expected_hash:
                status.append("refit_data_hash_mismatch")
            if str(signature_row["data_sha256"]) != expected_hash:
                status.append("signature_data_hash_mismatch")
            if (
                str(signature_row[ADMISSION_IMPLEMENTATION_FIELD])
                != admission_implementation_sha256()
            ):
                status.append("admission_implementation_mismatch")
            if str(signature_row[CORE_BUNDLE_FIELD]) != core_bundle_sha256:
                status.append("core_bundle_mismatch")
        mcmc: dict[str, Any] = {}
        estimands_match = False
        qualitative_match = False
        if signature_row is not None and not status:
            idata = az.from_netcdf(Path(str(signature_row["posterior_path"])))
            try:
                mcmc = summarize_mcmc(idata)
                recomputed = _project_rq2_estimands(
                    idata,
                    data,
                    core["term_manifest"],
                    config,
                    draw_chunk_size=DRAW_CHUNK_SIZE,
                )
                recomputed = add_qualitative_conclusions(
                    recomputed, core["primary_estimands"]
                ).sort_values(["direction", "group"])
            finally:
                idata.close()
            stored = saved_estimands[
                saved_estimands["influence_unit"].astype(str).eq(unit)
                & saved_estimands["deleted_unit_group"].astype(str).eq(group)
                & _normalized_ids(saved_estimands["deleted_unit_id"]).eq(unit_id)
            ].sort_values(["direction", "group"])
            estimands_match = bool(
                len(stored) == 6
                and stored[["direction", "group"]].astype(str).values.tolist()
                == recomputed[["direction", "group"]].astype(str).values.tolist()
                and _allclose(stored, recomputed, numeric_columns)
            )
            qualitative_match = bool(
                len(stored) == 6
                and stored["qualitative_conclusion"].astype(str).tolist()
                == recomputed["qualitative_conclusion"].astype(str).tolist()
                and stored["primary_qualitative_conclusion"].astype(str).tolist()
                == recomputed["primary_qualitative_conclusion"].astype(str).tolist()
                and stored["qualitative_class_changed"].astype(bool).tolist()
                == recomputed["qualitative_class_changed"].astype(bool).tolist()
            )
        if not bool(mcmc.get("passed", False)):
            status.append("parameter_mcmc_failed")
        if not estimands_match:
            status.append("six_estimands_recompute_mismatch")
        if not qualitative_match:
            status.append("qualitative_classification_mismatch")
        rows.append(
            {
                "influence_unit": unit,
                "group": group,
                "unit_id": unit_id,
                "authenticated": not status,
                "authentication_status": (
                    "authenticated_and_recomputed" if not status else ";".join(status)
                ),
                "recomputed_mcmc_passed": bool(mcmc.get("passed", False)),
                "recomputed_estimands_match": estimands_match,
                "recomputed_qualitative_classification_match": qualitative_match,
            }
        )
        print(
            f"[finalize refit {len(rows)}/{len(live_plan)}] "
            f"unit={unit} key={group}::{unit_id} "
            f"auth={'pass' if not status else 'fail'}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    complete = bool(
        live_keys == saved_keys
        and len(refits) == sum(EXPECTED_REFIT_COUNTS.values())
        and len(saved_estimands) == 6 * sum(EXPECTED_REFIT_COUNTS.values())
        and len(signatures) == sum(EXPECTED_REFIT_COUNTS.values())
        and result["authenticated"].all()
    )
    return complete, result


def _report_text(
    *,
    core_run_id: str,
    readiness: Any,
    comparisons: Any,
    estimands: Any,
) -> str:
    class_changes = estimands[
        estimands["qualitative_class_changed"].astype(bool)
    ]
    comparison = comparisons[comparisons["fold"].astype(str).eq("ALL")].iloc[0]
    lines = [
        "# RQ2 admission diagnostic",
        "",
        f"- Core run: `{core_run_id}`",
        "- Forward models: M1 and RQ2 on three identical future-choice folds",
        "- Exact grouped deletions: 10 participants, 45 selected accounts, 1 thread",
        (
            "- Sampling: four one-thread chains, 1,000 tune and 1,000 retained "
            "draws by default, with one bounded 2x extension after a failed "
            "parameter-MCMC check"
        ),
        "- Stored fit policy: compact parameter posterior plus sample statistics",
        "- Projection memory bound: 32 posterior draws",
        "",
        "## Forward comparison",
        "",
        (
            f"Pooled RQ2 minus M1 ELPD: {float(comparison['elpd_difference']):.3f} "
            f"(participant-clustered s.e. {float(comparison['elpd_difference_se']):.3f})."
        ),
        "",
        "## Exact-refit qualitative conclusions",
        "",
        (
            f"The 56 refits produced {len(estimands)} estimand cells. "
            f"{len(class_changes)} cells changed 95% CrI class relative to the "
            "authenticated primary cell. Classification is positive when the lower "
            "limit exceeds zero, negative when the upper limit is below zero, and "
            "uncertain otherwise. No numerical stability threshold is applied."
        ),
        "",
        "## Admission gates",
        "",
        readiness.to_markdown(index=False),
        "",
        "## Inference boundary",
        "",
        (
            "The directional estimates are model-projected associations conditional "
            "on observed human replies and reconstructed same-thread opportunities. "
            "The between-community contrasts remain ecological."
        ),
        "",
    ]
    return "\n".join(lines)


def _artifact_paths(paths: Mapping[str, Path]) -> list[Path]:
    result: list[Path] = []
    for key in TABLE_NAMES:
        stem = _table_path(paths, key)
        for suffix in (".parquet", ".csv"):
            candidate = stem.with_suffix(suffix)
            if candidate.exists():
                result.append(candidate)
    for path in (paths["report"],):
        if path.exists():
            result.append(path)
    return sorted(result, key=str)


def finalize_admission(
    config: Any,
    core: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    source_snapshot: Sequence[Mapping[str, Any]],
    local_temp: Path,
) -> dict[str, Any]:
    import pandas as pd

    from scripts.publication_analysis.io import sha256_file, source_record, write_json
    from scripts.publication_analysis.provenance import assert_source_snapshot_unchanged

    _assert_table_pairs_semantically_equal(paths, STAGE_TABLE_KEYS)
    forward_pass, forward_auth = _validate_forward_stage(config, core, paths)
    refit_pass, refit_auth = _validate_refit_stage(config, core, paths)
    assert_source_snapshot_unchanged(source_snapshot)
    from scripts.diagnostics.rq2_sensitivity import authenticate_core_run

    core_after = authenticate_core_run(config)
    core_unchanged = bool(
        str(core_after["main_signature"]["posterior_sha256"])
        == str(core["main_signature"]["posterior_sha256"])
    )
    readiness = pd.DataFrame(
        [
            {
                "gate": "rq2_core_package_unchanged_and_authenticated",
                "passed": core_unchanged,
                "observed": str(core["main_signature"]["posterior_sha256"]),
            },
            {
                "gate": "rq2_three_origin_m1_vs_rq2_prediction",
                "passed": forward_pass,
                "observed": f"authenticated={int(forward_auth['authenticated'].sum())}/6",
            },
            {
                "gate": "rq2_exact_grouped_deletion_refits",
                "passed": refit_pass,
                "observed": (
                    f"authenticated={int(refit_auth['authenticated'].sum())}/"
                    f"{sum(EXPECTED_REFIT_COUNTS.values())}"
                ),
            },
        ]
    )
    _write_tables(
        paths,
        {
            "forward_authentication": forward_auth,
            "refit_authentication": refit_auth,
            "readiness": readiness,
        },
    )
    _assert_table_pairs_semantically_equal(paths, tuple(TABLE_NAMES))
    comparisons = _read_required_table(paths, "forward_comparisons")
    estimands = _read_required_table(paths, "refit_estimands")
    paths["report"].write_text(
        _report_text(
            core_run_id=config.run_id,
            readiness=readiness,
            comparisons=comparisons,
            estimands=estimands,
        ),
        encoding="utf-8",
    )
    assert_source_snapshot_unchanged(source_snapshot)
    artifacts = _artifact_paths(paths)
    signature_tables = (
        _read_required_table(paths, "forward_signatures"),
        _read_required_table(paths, "refit_signatures"),
    )
    posterior_paths = sorted(
        {
            Path(str(value)).resolve()
            for table in signature_tables
            for value in table["posterior_path"]
        },
        key=str,
    )
    expected_posteriors = (
        len(FORWARD_FOLDS) * len(FORWARD_MODELS)
        + sum(EXPECTED_REFIT_COUNTS.values())
    )
    if len(posterior_paths) != expected_posteriors or any(
        not path.is_relative_to(paths["models"].resolve())
        for path in posterior_paths
    ):
        raise RuntimeError("Admission posterior inventory is incomplete or out of scope")
    posterior_records: list[dict[str, Any]] = []
    for path in posterior_paths:
        record = source_record(path)
        record["fit_signature"] = source_record(
            path.with_suffix(path.suffix + ".fit.json")
        )
        posterior_records.append(record)
    bundle = {
        "schema": ADMISSION_SCHEMA,
        "core_run_id": config.run_id,
        "status": "passed" if readiness["passed"].all() else "failed",
        "admission_implementation_sha256": admission_implementation_sha256(),
        "core": {
            "bundle_sha256": sha256_file(config.outputs_dir / "fit_result_bundle.json"),
            "model_data_sha256": core["data_sha256"],
            "model_implementation_sha256": core["implementation_sha256"],
            "main_posterior": source_record(core["main_posterior_path"]),
            "authenticated_before_and_after": core_unchanged,
        },
        "sampling_policy": {
            "draws": int(config.sampling.draws),
            "tune": int(config.sampling.tune),
            "chains": 4,
            "chain_workers_maximum": 4,
            "numerical_threads_per_worker": 1,
            "progressbar": False,
            "diagnostic_extension": (
                "at_most_one_refit_with_twice_draws_and_tune_after_failed_mcmc"
            ),
        },
        "forward_validation": {
            "folds": list(FORWARD_FOLDS),
            "models": list(FORWARD_MODELS),
            "identical_future_choice_sets": True,
            "fold_specific_training_scalers": True,
        },
        "exact_refits": {
            "counts": dict(EXPECTED_REFIT_COUNTS),
            "total": sum(EXPECTED_REFIT_COUNTS.values()),
            "estimands_per_refit": 6,
            "projection_draw_chunk_size": DRAW_CHUNK_SIZE,
            "qualitative_rule": {
                "positive": "ci_low > 0",
                "negative": "ci_high < 0",
                "uncertain": "otherwise",
            },
            "qualitative_class_changes": int(
                estimands["qualitative_class_changed"].sum()
            ),
        },
        "storage_policy": (
            "compact_parameter_posterior_plus_sample_stats; "
            "per-choice deterministic and log-likelihood arrays dropped"
        ),
        "local_temp": str(local_temp),
        "source_snapshot": list(source_snapshot),
        "readiness": readiness.to_dict("records"),
        "artifacts": [source_record(path) for path in artifacts],
        "posteriors": posterior_records,
    }
    write_json(bundle, paths["bundle"])
    write_json(
        {
            "schema": ADMISSION_SCHEMA,
            "core_run_id": config.run_id,
            "stage": "finalize",
            "status": bundle["status"],
            "bundle": str(paths["bundle"]),
        },
        paths["status"],
    )
    assert_source_snapshot_unchanged(source_snapshot)
    return {
        "stage": "finalize",
        "status": bundle["status"],
        "admission_root": str(paths["root"]),
        "report": str(paths["report"]),
        "bundle": str(paths["bundle"]),
        "gates_passed": int(readiness["passed"].sum()),
        "gates_total": len(readiness),
    }


def command_plan(config_path: Path, run_id: str) -> list[str]:
    prefix = (
        "timeout 24h python -m scripts.diagnostics.rq2_admission "
        f"--config {config_path} --run-id {run_id}"
    )
    return [f"{prefix} --stage {stage}" for stage in ("forward", "refits", "finalize")]


def run_admission(
    *,
    config_path: Path,
    run_id: str,
    stage: str,
    profile: str = "release",
    resume: bool = True,
    local_temp: Path | None = None,
) -> dict[str, Any]:
    """Run one resumable stage while holding the authenticated core fixed."""

    local_root = configure_runtime(local_temp)
    from dataclasses import replace

    from scripts.diagnostics.rq2_sensitivity import authenticate_core_run
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.io import sha256_file
    from scripts.publication_analysis.provenance import (
        assert_source_snapshot_unchanged,
        capture_source_snapshot,
    )

    if profile != "release":
        raise ValueError("Formal RQ2 admission requires the release profile")
    config_path = Path(config_path).expanduser().resolve()
    config = replace(
        load_config(config_path, profile=profile, run_id=run_id),
        analysis_stage="rq2_m2",
    )
    _release_request(config)
    core = authenticate_core_run(config)
    paths = _admission_paths(config)
    _ensure_paths(paths)
    source_snapshot = capture_source_snapshot(_core_bundle_paths(config, core))
    core_bundle_sha256 = sha256_file(config.outputs_dir / "fit_result_bundle.json")
    plan = command_plan(config_path, run_id)
    if stage == "plan":
        return {
            "stage": "plan",
            "core_run_id": run_id,
            "admission_root": str(paths["root"]),
            "forward_fits": 6,
            "exact_refits": sum(EXPECTED_REFIT_COUNTS.values()),
            "commands": plan,
        }
    results: list[dict[str, Any]] = []
    if stage in {"all", "forward"}:
        results.append(
            run_forward_stage(
                config,
                core,
                paths,
                resume=resume,
                core_bundle_sha256=core_bundle_sha256,
            )
        )
        assert_source_snapshot_unchanged(source_snapshot)
    if stage in {"all", "refits"}:
        results.append(
            run_refit_stage(
                config,
                core,
                paths,
                resume=resume,
                core_bundle_sha256=core_bundle_sha256,
            )
        )
        assert_source_snapshot_unchanged(source_snapshot)
    if stage in {"all", "finalize"}:
        results.append(
            finalize_admission(
                config,
                core,
                paths,
                source_snapshot=source_snapshot,
                local_temp=local_root,
            )
        )
    return {
        "core_run_id": run_id,
        "stage": stage,
        "admission_root": str(paths["root"]),
        "results": results,
        "commands": plan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone, compact, resumable RQ2 admission diagnostics. "
            "Use one foreground stage per 24-hour command window."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    parser.add_argument("--run-id", required=True, help="Authenticated RQ2 core run ID")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "forward", "refits", "finalize", "all"),
    )
    parser.add_argument("--profile", default="release", choices=("release",))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore compatible signed compact posteriors and refit this stage.",
    )
    parser.add_argument(
        "--local-temp",
        type=Path,
        help="Local temporary and PyTensor compile directory (default: /tmp).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_admission(
        config_path=args.config,
        run_id=args.run_id,
        stage=args.stage,
        profile=args.profile,
        resume=not args.no_resume,
        local_temp=args.local_temp,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    final = result.get("results", [])
    if final and final[-1].get("stage") == "finalize":
        return 0 if final[-1].get("status") == "passed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
