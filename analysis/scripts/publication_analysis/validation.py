"""Validation gates for the streamlined publication analysis."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import sha256_file, write_json, write_table
from .study import GROUPS


def _id_text(series: pd.Series) -> pd.Series:
    """Normalize integer-like identifiers without collapsing missing values."""

    numeric = pd.to_numeric(series, errors="coerce")
    normalized = series.astype("string")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    normalized.loc[integer_like] = numeric.loc[integer_like].round().astype("Int64").astype("string")
    return normalized


def _group_database_mapping_check(
    event_history: pd.DataFrame,
    provenance: pd.DataFrame | None,
) -> tuple[str, bool, str, str]:
    """Reconcile realized arms with independently recorded source identities."""

    expected_groups = set(GROUPS)
    realized_groups = set(event_history["group"].dropna().astype(str))
    required = "; ".join(
        (
            f"{group}={GROUPS[group].db_name}, "
            f"experiment_id={GROUPS[group].experiment_id}"
        )
        for group in sorted(GROUPS)
    )
    if provenance is None:
        observed = f"event_groups={sorted(realized_groups)}; source provenance table unavailable"
        return "group_database_mapping", False, observed, required

    required_columns = {
        "group",
        "provenance_path",
        "provenance_status",
        "source_db_name",
        "source_experiment_id",
    }
    missing_columns = sorted(required_columns.difference(provenance.columns))
    if missing_columns:
        observed = (
            f"event_groups={sorted(realized_groups)}; source provenance missing columns "
            f"{missing_columns}"
        )
        return "group_database_mapping", False, observed, required

    source_groups = provenance["group"].fillna("").astype(str)
    group_counts = source_groups.value_counts().to_dict()
    provenance_shape_ok = (
        len(provenance) == len(expected_groups)
        and set(source_groups) == expected_groups
        and all(group_counts.get(group) == 1 for group in expected_groups)
    )
    details: list[str] = []
    identities_match = True
    provenance_established = True
    for group in sorted(GROUPS):
        matches = provenance[source_groups.eq(group)]
        if len(matches) != 1:
            identities_match = False
            provenance_established = False
            details.append(f"{group}: source_records={len(matches)}")
            continue
        record = matches.iloc[0]
        db_name = record["source_db_name"]
        experiment_id = pd.to_numeric(
            pd.Series([record["source_experiment_id"]]), errors="coerce"
        ).iloc[0]
        status = str(record["provenance_status"])
        provenance_path = record["provenance_path"]
        db_match = pd.notna(db_name) and str(db_name) == GROUPS[group].db_name
        experiment_match = pd.notna(experiment_id) and int(experiment_id) == GROUPS[group].experiment_id
        path_present = pd.notna(provenance_path) and bool(str(provenance_path).strip())
        status_established = status == "established_from_source_manifest" and path_present
        identities_match = identities_match and bool(db_match and experiment_match)
        provenance_established = provenance_established and status_established
        details.append(
            (
                f"{group}: db={db_name}, experiment_id={record['source_experiment_id']}, "
                f"status={status}, provenance={provenance_path}"
            )
        )

    passed = (
        realized_groups == expected_groups
        and provenance_shape_ok
        and identities_match
        and provenance_established
    )
    observed = f"event_groups={sorted(realized_groups)}; " + "; ".join(details)
    return "group_database_mapping", passed, observed, required


def data_integrity_checks(
    event_history: pd.DataFrame,
    choice_sets: pd.DataFrame,
    *,
    expected_experiment_days: Iterable[int] | None = None,
    actor_label_qc: pd.DataFrame | None = None,
    group_database_provenance: pd.DataFrame | None = None,
    network_history: pd.DataFrame | None = None,
    recomputed_network_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate publication-blocking invariants on the canonical risk sets."""

    required_events = {"group", "actor_type", "target_author_type", "valid_day_index"}
    required_choices = {
        "choice_set_id",
        "group",
        "actor_id",
        "candidate_author_id",
        "target_content_id",
        "target_author_id",
        "target_author_type",
        "candidate_content_id",
        "candidate_author_type",
        "event_timestamp",
        "candidate_created_at",
        "chosen",
    }
    missing_events = sorted(required_events.difference(event_history.columns))
    missing_choices = sorted(required_choices.difference(choice_sets.columns))
    if missing_events or missing_choices:
        raise ValueError(
            f"Integrity inputs are incomplete: event columns {missing_events}; "
            f"choice columns {missing_choices}"
        )

    grouped = choice_sets.groupby("choice_set_id", sort=False)
    chosen_count = grouped["chosen"].sum()
    candidate_count = grouped.size()
    event_time = pd.to_datetime(choice_sets["event_timestamp"], utc=True, errors="coerce")
    candidate_time = pd.to_datetime(choice_sets["candidate_created_at"], utc=True, errors="coerce")
    actor = _id_text(choice_sets["actor_id"])
    receiver = _id_text(choice_sets["candidate_author_id"])
    chosen = choice_sets[choice_sets["chosen"].astype(bool)]
    chosen_target_match = _id_text(chosen["candidate_content_id"]).eq(
        _id_text(chosen["target_content_id"])
    )
    chosen_author_match = _id_text(chosen["candidate_author_id"]).eq(
        _id_text(chosen["target_author_id"])
    )
    chosen_type_match = chosen["candidate_author_type"].astype(str).eq(
        chosen["target_author_type"].astype(str)
    )

    results = [
        _group_database_mapping_check(event_history, group_database_provenance),
        (
            "authoritative_actor_target_types",
            event_history["actor_type"].astype(str).isin(["H", "AI"]).all()
            and event_history["target_author_type"].astype(str).isin(["H", "AI"]).all(),
            "actor and target types in H,AI",
            "all rows",
        ),
        (
            "valid_analysis_day_filter",
            event_history["valid_day_index"].notna().all(),
            f"{int(event_history['valid_day_index'].isna().sum())} missing valid-day indexes",
            "0 missing",
        ),
        (
            "minimum_two_candidates",
            candidate_count.ge(2).all(),
            int(candidate_count.min()) if not candidate_count.empty else 0,
            ">=2",
        ),
        (
            "unique_chosen_candidate",
            chosen_count.eq(1).all(),
            int((chosen_count.ne(1)).sum()),
            "0 invalid sets",
        ),
        (
            "chosen_content_matches_backend_target",
            chosen_target_match.all(),
            int((~chosen_target_match).sum()),
            "0 mismatches",
        ),
        (
            "chosen_author_matches_backend_target",
            chosen_author_match.all() and chosen_type_match.all(),
            int((~chosen_author_match | ~chosen_type_match).sum()),
            "0 author or H/AI target mismatches",
        ),
        (
            "self_authored_candidates_excluded",
            actor.ne(receiver).all(),
            int(actor.eq(receiver).sum()),
            "0 self-authored candidates",
        ),
        (
            "candidate_strictly_precedes_event",
            event_time.notna().all()
            and candidate_time.notna().all()
            and candidate_time.lt(event_time).all(),
            int((candidate_time.ge(event_time) | event_time.isna() | candidate_time.isna()).sum()),
            "0 non-prior candidates",
        ),
    ]
    if expected_experiment_days is not None:
        if "experiment_day" not in event_history.columns:
            raise ValueError("Expected-day validation requires experiment_day")
        expected_days = {int(value) for value in expected_experiment_days}
        realized_days = set(
            pd.to_numeric(event_history["experiment_day"], errors="coerce").dropna().astype(int)
        )
        zero_event_days = sorted(expected_days - realized_days)
        extra_days = sorted(realized_days - expected_days)
        results.append(
            (
                "realized_days_within_configured_valid_set",
                not extra_days,
                (
                    f"realized={sorted(realized_days)}; "
                    f"configured_zero_event_days={zero_event_days}; extra={extra_days}"
                ),
                f"no realized day outside {sorted(expected_days)}; zero-event days allowed",
            )
        )
    if actor_label_qc is not None:
        required_qc = {
            "group",
            "account_type",
            "observed_accounts",
            "expected_accounts",
            "label_conflicts",
            "passed",
        }
        missing_qc = sorted(required_qc.difference(actor_label_qc.columns))
        if missing_qc:
            raise ValueError(f"Actor-label QC is missing columns: {missing_qc}")
        results.append(
            (
                "authoritative_account_labels_and_population_counts",
                actor_label_qc["passed"].fillna(False).astype(bool).all(),
                "; ".join(
                    (
                        f"{row.group}-{row.account_type}: "
                        f"{int(row.observed_accounts)}/{int(row.expected_accounts)}, "
                        f"conflicts={int(row.label_conflicts)}"
                    )
                    for row in actor_label_qc.itertuples(index=False)
                ),
                "all expected H/AI accounts present with zero cross-source label conflicts",
            )
        )
    if network_history is not None or recomputed_network_history is not None:
        if network_history is None or recomputed_network_history is None:
            raise ValueError("Network-history validation requires stored and recomputed frames")
        key_columns = ["group", "choice_set_id", "candidate_content_id"]
        dynamic_columns = [
            column
            for column in recomputed_network_history.columns
            if column.startswith("prior_")
            or column.startswith("receiver_popularity_")
            or column.startswith("candidate_reputation_")
        ]
        comparison_columns = [*key_columns, *dynamic_columns]
        missing_network = sorted(set(comparison_columns).difference(network_history.columns))
        reproduced = not missing_network
        detail = "exact pre-event dynamic-covariate reproduction"
        if reproduced:
            left = network_history[comparison_columns].sort_values(
                key_columns, kind="stable"
            ).reset_index(drop=True)
            right = recomputed_network_history[comparison_columns].sort_values(
                key_columns, kind="stable"
            ).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    left,
                    right,
                    check_dtype=False,
                    check_exact=False,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            except AssertionError as error:
                reproduced = False
                detail = str(error).splitlines()[0]
        else:
            detail = f"missing columns={missing_network}"
        results.append(
            (
                "pre_event_network_covariates_reproduce_from_ordered_events",
                reproduced,
                detail,
                "stored dynamic histories equal an independent strict-prior rebuild",
            )
        )
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(passed),
                "observed": str(observed),
                "required": str(required),
            }
            for name, passed, observed, required in results
        ]
    )


def authenticate_fit_signatures(signatures: pd.DataFrame) -> pd.DataFrame:
    """Recheck posterior bytes and signed requests at validation time."""

    if signatures.empty:
        return signatures.assign(
            live_posterior_authenticated=pd.Series(dtype=bool),
            authentication_status=pd.Series(dtype=object),
        )
    required = {
        "posterior_path",
        "posterior_sha256",
        "model",
        "role",
        "draws",
        "tune",
        "chains",
        "target_accept",
        "priors",
        "random_seed",
        "data_sha256",
        "implementation_sha256",
        "actual_draws",
        "actual_chains",
        "posterior_size_bytes",
    }
    missing = sorted(required.difference(signatures.columns))
    if missing:
        raise ValueError(f"Fit-signature table is missing authentication fields: {missing}")
    rows: list[dict[str, Any]] = []
    for record in signatures.to_dict("records"):
        path = Path(str(record["posterior_path"]))
        sidecar = path.with_suffix(path.suffix + ".fit.json")
        status = "authenticated"
        authenticated = False
        try:
            import xarray as xr

            signed = json.loads(sidecar.read_text(encoding="utf-8"))
            request = signed["request"]
            actual = signed["actual"]
            live_hash = sha256_file(path)
            with xr.open_dataset(path, group="posterior") as posterior:
                live_draws = int(posterior.sizes.get("draw", 0))
                live_chains = int(posterior.sizes.get("chain", 0))
            priors_match = json.dumps(
                request["priors"], sort_keys=True, default=str
            ) == json.dumps(record["priors"], sort_keys=True, default=str)
            fields_match = all(
                [
                    str(request["model"]) == str(record["model"]),
                    str(request["role"]) == str(record["role"]),
                    int(request["draws"]) == int(record["draws"]),
                    int(request["tune"]) == int(record["tune"]),
                    int(request["chains"]) == int(record["chains"]),
                    float(request["target_accept"])
                    == float(record["target_accept"]),
                    priors_match,
                    int(request["random_seed"]) == int(record["random_seed"]),
                    str(request["data_sha256"]) == str(record["data_sha256"]),
                    str(request["implementation_sha256"])
                    == str(record["implementation_sha256"]),
                ]
            )
            authenticated = bool(
                fields_match
                and live_hash == str(record["posterior_sha256"])
                and live_hash == str(actual["posterior_sha256"])
                and path.stat().st_size == int(actual["posterior_size_bytes"])
                and path.stat().st_size == int(record["posterior_size_bytes"])
                and live_draws == int(actual["draws"])
                and live_draws == int(record["actual_draws"])
                and live_chains == int(actual["chains"])
                and live_chains == int(record["actual_chains"])
            )
            if not authenticated:
                status = "hash_or_request_mismatch"
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            status = "missing_or_invalid_posterior_signature"
        rows.append(
            {
                **record,
                "live_posterior_authenticated": authenticated,
                "authentication_status": status,
            }
        )
    return pd.DataFrame(rows)


def _grouped_log_likelihood_efficiency(
    grouped_log_likelihood: Any,
    az: Any,
) -> tuple[float, float]:
    """Estimate MCMC efficiency from the grouped likelihood used by PSIS."""

    values = (
        grouped_log_likelihood.stack(sample=("chain", "draw"))
        .transpose("sample")
        .values.astype(float)
    )
    if not np.isfinite(values).all():
        raise ValueError("Participant-grouped log likelihood contains non-finite draws")
    sample_count = int(values.size)
    if sample_count < 1:
        raise ValueError("Participant-grouped log likelihood has no posterior draws")
    if np.ptp(values) <= np.finfo(float).eps:
        return 1.0, float(sample_count)
    result = az.ess(grouped_log_likelihood, method="mean")
    finite = result.to_array().values.astype(float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if not finite.size:
        return 1.0e-3, float(sample_count * 1.0e-3)
    effective_samples = float(np.min(finite))
    relative_efficiency = float(
        np.clip(effective_samples / sample_count, 1.0e-3, 1.0)
    )
    return relative_efficiency, effective_samples


def _choice_influence_units(
    choice_sets: pd.DataFrame,
    unit: str,
) -> pd.DataFrame:
    common = {"choice_set_id", "group"}
    if unit == "participant":
        required = common | {"actor_id"}
        unit_column = "actor_id"
        source = choice_sets
    elif unit == "account":
        required = common | {"candidate_author_id", "chosen"}
        unit_column = "candidate_author_id"
        source = choice_sets[choice_sets["chosen"].astype(bool)]
    elif unit == "thread":
        required = common | {"root_id"}
        unit_column = "root_id"
        source = choice_sets
    else:
        raise ValueError("Influence unit must be participant, account, or thread")
    missing = sorted(required.difference(choice_sets.columns))
    if missing:
        raise ValueError(f"Choice rows are missing {unit} influence columns: {missing}")
    choices = source[["choice_set_id", "group", unit_column]].drop_duplicates()
    counts = choices.groupby("choice_set_id", sort=False).size()
    if not counts.eq(1).all():
        raise ValueError(f"Each choice set must map to exactly one {unit} influence unit")
    return choices.rename(columns={unit_column: "unit_id"})


def grouped_choice_psis(
    idata: Any,
    choice_sets: pd.DataFrame,
    *,
    unit: str,
    pareto_k_threshold: float = 0.70,
) -> pd.DataFrame:
    """Compute grouped PSIS influence diagnostics for one declared choice unit."""

    import arviz as az
    from scipy.special import logsumexp

    if "log_likelihood" not in idata.groups() or "choice" not in idata.log_likelihood:
        raise ValueError("InferenceData needs log_likelihood.choice")
    choices = _choice_influence_units(choice_sets, unit)
    likelihood = idata.log_likelihood["choice"]
    dimension = "choice_set" if "choice_set" in likelihood.dims else likelihood.dims[-1]
    if likelihood.sizes[dimension] != len(choices):
        raise ValueError("Pointwise likelihood and choice-set metadata have different lengths")
    if dimension in likelihood.coords:
        ordered_ids = [str(value) for value in likelihood.coords[dimension].values]
        metadata = choices.assign(_id=choices["choice_set_id"].astype(str)).set_index("_id")
        if set(ordered_ids) != set(metadata.index):
            raise ValueError("Pointwise likelihood choice identifiers do not match choice rows")
        choices = metadata.loc[ordered_ids].reset_index(drop=True)

    labels = choices["group"].astype(str) + "::" + _id_text(choices["unit_id"])
    rows: list[dict[str, Any]] = []
    for label in pd.unique(labels):
        selected = labels.eq(label).to_numpy()
        grouped_likelihood = likelihood.isel(
            {dimension: np.flatnonzero(selected)}
        ).sum(dim=dimension)
        relative_efficiency, grouped_ess = _grouped_log_likelihood_efficiency(
            grouped_likelihood,
            az,
        )
        unit_log_likelihood = (
            grouped_likelihood.stack(sample=("chain", "draw"))
            .transpose("sample")
            .values
        )
        log_weights, pareto_k = az.psislw(
            -unit_log_likelihood,
            reff=relative_efficiency,
        )
        group, unit_id = str(label).split("::", maxsplit=1)
        rows.append(
            {
                "group": group,
                "influence_unit": unit,
                "unit_id": unit_id,
                "choice_sets": int(selected.sum()),
                "pareto_k": float(pareto_k),
                "elpd_loo": float(logsumexp(unit_log_likelihood + log_weights)),
                "needs_refit": bool(pareto_k > pareto_k_threshold),
                "threshold": pareto_k_threshold,
                "relative_efficiency": relative_efficiency,
                "grouped_log_likelihood_ess": grouped_ess,
                "relative_efficiency_source": (
                    f"{unit}_grouped_log_likelihood_mean_ess"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["needs_refit", "pareto_k"], ascending=[False, False]
    ).reset_index(drop=True)


def participant_grouped_psis(
    idata: Any,
    choice_sets: pd.DataFrame,
    *,
    pareto_k_threshold: float = 0.70,
) -> pd.DataFrame:
    """Compute PSIS diagnostics after grouping pointwise likelihood by participant."""

    result = grouped_choice_psis(
        idata,
        choice_sets,
        unit="participant",
        pareto_k_threshold=pareto_k_threshold,
    )
    return result.rename(columns={"unit_id": "actor_id"}).drop(
        columns="influence_unit"
    )


def rolling_origin_assignments(
    choice_sets: pd.DataFrame,
    cutpoints: Iterable[float] = (0.50, 0.65, 0.80),
    horizon: float = 0.15,
) -> pd.DataFrame:
    """Assign ordered choice sets to three forward train/test folds per arm."""

    required = {"choice_set_id", "group", "event_timestamp"}
    missing = sorted(required.difference(choice_sets.columns))
    if missing:
        raise ValueError(f"Choice sets are missing forward-validation columns: {missing}")
    events = choice_sets[list(required)].drop_duplicates("choice_set_id").copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True, errors="coerce")
    if events["event_timestamp"].isna().any():
        raise ValueError("Forward validation requires non-missing event timestamps")
    rows: list[dict[str, Any]] = []
    for group, arm in events.groupby("group", sort=True):
        arm = arm.sort_values(["event_timestamp", "choice_set_id"]).reset_index(drop=True)
        count = len(arm)
        for fold, cutpoint in enumerate(cutpoints, start=1):
            train_end = max(1, int(np.floor(count * float(cutpoint))))
            test_end = min(count, max(train_end + 1, int(np.floor(count * (float(cutpoint) + horizon)))))
            for role, subset in (("train", arm.iloc[:train_end]), ("test", arm.iloc[train_end:test_end])):
                for choice_id in subset["choice_set_id"]:
                    rows.append({"fold": fold, "group": group, "choice_set_id": choice_id, "role": role})
    return pd.DataFrame(rows)


def summarize_mcmc(idata: Any) -> dict[str, float | int | bool]:
    """Return publication diagnostics for an ArviZ InferenceData object."""

    import arviz as az

    summary = az.summary(idata, kind="diagnostics", round_to=8)
    rhat = pd.to_numeric(summary.get("r_hat"), errors="coerce")
    ess_bulk = pd.to_numeric(summary.get("ess_bulk"), errors="coerce")
    ess_tail = pd.to_numeric(summary.get("ess_tail"), errors="coerce")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())
    result: dict[str, float | int | bool] = {
        "max_rhat": float(rhat.max()) if rhat.notna().any() else np.nan,
        "min_ess_bulk": float(ess_bulk.min()) if ess_bulk.notna().any() else np.nan,
        "min_ess_tail": float(ess_tail.min()) if ess_tail.notna().any() else np.nan,
        "divergences": divergences,
    }
    result["passed"] = bool(
        result["max_rhat"] < 1.01
        and result["min_ess_bulk"] > 400
        and result["min_ess_tail"] > 400
        and divergences == 0
    )
    return result


def build_rq1_readiness(
    *,
    integrity: pd.DataFrame,
    analysis_signature: pd.DataFrame,
    diagnostics: pd.DataFrame,
    signatures: pd.DataFrame,
    primary_estimand: pd.DataFrame,
    prior_sensitivity: pd.DataFrame,
    forward_scores: pd.DataFrame,
    forward_diagnostics: pd.DataFrame,
    forward_comparisons: pd.DataFrame,
    influence_tables: Mapping[str, pd.DataFrame],
    fit_bundle_authentication: pd.DataFrame,
    exact_influence_refits: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Apply fail-closed release gates for the final RQ1 M1 analysis."""

    rows: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, observed: str) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "required_for_rq1_release": True,
                "observed": observed,
            }
        )

    integrity_pass = bool(
        not integrity.empty
        and integrity.get("passed", pd.Series(False, index=integrity.index))
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "rq1_canonical_data_integrity",
        integrity_pass,
        f"{int(integrity.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(integrity)} checks passed",
    )

    bundle_pass = bool(
        len(fit_bundle_authentication) == 1
        and fit_bundle_authentication.get(
            "authenticated", pd.Series(False, index=fit_bundle_authentication.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and fit_bundle_authentication.get(
            "required_tables_present",
            pd.Series(False, index=fit_bundle_authentication.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "rq1_fit_result_bundle_authentication",
        bundle_pass,
        (
            str(fit_bundle_authentication["authentication_status"].iloc[0])
            if len(fit_bundle_authentication) == 1
            else "missing authentication row"
        ),
    )

    current_data_hash = (
        str(analysis_signature["model_data_sha256"].iloc[0])
        if len(analysis_signature) == 1
        else None
    )
    current_implementation_hash = (
        str(analysis_signature["implementation_sha256"].iloc[0])
        if len(analysis_signature) == 1
        else None
    )
    main_signature = signatures[
        signatures.get("model", pd.Series(dtype=object)).eq("M1")
        & signatures.get("role", pd.Series(dtype=object)).eq("rq1_main")
    ]
    signature_pass = bool(
        len(main_signature) == 1
        and main_signature.get(
            "live_posterior_authenticated",
            pd.Series(False, index=main_signature.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and main_signature.get(
            "publication_length_request", pd.Series(False, index=main_signature.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and main_signature.get(
            "dimension_match", pd.Series(False, index=main_signature.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and main_signature.get("data_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_data_hash)
        .all()
        and main_signature.get("implementation_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_implementation_hash)
        .all()
    )
    add(
        "rq1_signed_publication_length_m1",
        signature_pass,
        f"authenticated main M1 rows={len(main_signature)}",
    )

    prior_signatures = signatures[
        signatures.get("role", pd.Series(dtype=object)).eq(
            "rq1_doubled_prior_scales"
        )
        & signatures.get("model", pd.Series(dtype=object)).eq("M1")
    ]
    forward_signatures = signatures[
        signatures.get("role", pd.Series(dtype=object))
        .astype(str)
        .str.startswith("forward_fold_")
        & signatures.get("model", pd.Series(dtype=object)).isin(["M0", "M1"])
    ]
    expected_forward_signatures = {
        (f"forward_fold_{fold}", model)
        for fold in (1, 2, 3)
        for model in ("M0", "M1")
    }
    observed_forward_signatures = set(
        zip(
            forward_signatures.get("role", pd.Series(dtype=object)).astype(str),
            forward_signatures.get("model", pd.Series(dtype=object)).astype(str),
        )
    )
    ancillary = pd.concat(
        [prior_signatures, forward_signatures], ignore_index=True
    )
    ancillary_signature_pass = bool(
        len(prior_signatures) == 1
        and observed_forward_signatures == expected_forward_signatures
        and len(forward_signatures) == len(expected_forward_signatures)
        and ancillary.get(
            "live_posterior_authenticated", pd.Series(False, index=ancillary.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and ancillary.get(
            "publication_length_request", pd.Series(False, index=ancillary.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and ancillary.get(
            "dimension_match", pd.Series(False, index=ancillary.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and ancillary.get("implementation_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_implementation_hash)
        .all()
        and forward_signatures.get(
            "forward_training_data_match",
            pd.Series(False, index=forward_signatures.index, dtype=bool),
        )
        .map(lambda value: bool(value) if pd.notna(value) else False)
        .all()
    )
    add(
        "rq1_signed_sensitivity_and_forward_posteriors",
        ancillary_signature_pass,
        (
            f"prior rows={len(prior_signatures)}; "
            f"forward rows={len(forward_signatures)}/6"
        ),
    )

    main_diagnostic = diagnostics[
        diagnostics.get("model", pd.Series(dtype=object)).eq("M1")
        & diagnostics.get("role", pd.Series(dtype=object)).eq("rq1_main")
    ]
    mcmc_pass = bool(
        len(main_diagnostic) == 1
        and main_diagnostic.get("passed", pd.Series(False, index=main_diagnostic.index))
        .fillna(False)
        .astype(bool)
        .all()
    )
    add("rq1_m1_mcmc", mcmc_pass, f"passing rows={int(mcmc_pass)}")

    expected_groups = {"G1", "G2", "G2-G1"}
    estimand_groups = set(
        primary_estimand.get("group", pd.Series(dtype=object)).astype(str)
    )
    numeric_columns = (
        "posterior_mean",
        "posterior_median",
        "ci_low",
        "ci_high",
        "posterior_probability_gt_zero",
    )
    estimand_finite = all(
        np.isfinite(
            pd.to_numeric(
                primary_estimand.get(column, pd.Series(dtype=float)), errors="coerce"
            )
        ).all()
        for column in numeric_columns
    )
    estimand_pass = bool(
        len(primary_estimand) == 3
        and estimand_groups == expected_groups
        and estimand_finite
        and primary_estimand.get(
            "standardization_scope", pd.Series(dtype=object)
        )
        .eq("pooled_common_candidate_sets")
        .all()
        and primary_estimand.get("inference_boundary", pd.Series(dtype=object))
        .eq("one_community_per_condition_ecological_contrast")
        .all()
    )
    add(
        "rq1_primary_estimand_complete",
        estimand_pass,
        f"groups={sorted(estimand_groups)}; finite={estimand_finite}",
    )

    prior_pass = bool(
        not prior_sensitivity.empty
        and prior_sensitivity.get(
            "passed", pd.Series(False, index=prior_sensitivity.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "rq1_prior_scale_sensitivity",
        prior_pass,
        f"{int(prior_sensitivity.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(prior_sensitivity)} rows passed",
    )

    expected_forward = {
        (fold, model, group)
        for fold in (1, 2, 3)
        for model in ("M0", "M1")
        for group in ("G1", "G2", "ALL")
    }
    observed_forward = {
        (int(row.fold), str(row.model), str(row.group))
        for row in forward_scores.itertuples(index=False)
    } if not forward_scores.empty else set()
    expected_diagnostics = {
        (fold, model) for fold in (1, 2, 3) for model in ("M0", "M1")
    }
    observed_diagnostics = {
        (int(row.fold), str(row.model))
        for row in forward_diagnostics.itertuples(index=False)
    } if not forward_diagnostics.empty else set()
    expected_comparisons = {
        (fold, "M1", "M0") for fold in ("1", "2", "3", "ALL")
    }
    observed_comparisons = {
        (
            "ALL" if str(row.fold) == "ALL" else str(int(float(row.fold))),
            str(row.reference_model),
            str(row.comparator_model),
        )
        for row in forward_comparisons.itertuples(index=False)
    } if not forward_comparisons.empty else set()
    forward_pass = bool(
        observed_forward == expected_forward
        and observed_diagnostics == expected_diagnostics
        and expected_comparisons == observed_comparisons
        and forward_diagnostics.get(
            "passed", pd.Series(False, index=forward_diagnostics.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and np.isfinite(
            pd.to_numeric(
                forward_scores.get("mean_future_log_score", pd.Series(dtype=float)),
                errors="coerce",
            )
        ).all()
        and np.isfinite(
            pd.to_numeric(
                forward_comparisons.get("elpd_difference", pd.Series(dtype=float)),
                errors="coerce",
            )
        ).all()
    )
    add(
        "rq1_three_origin_prediction",
        forward_pass,
        (
            f"score cells={len(observed_forward)}/{len(expected_forward)}; "
            f"diagnostics={len(observed_diagnostics)}/{len(expected_diagnostics)}; "
            f"comparisons={len(observed_comparisons)}/{len(expected_comparisons)}"
        ),
    )

    exact_influence_refits = exact_influence_refits or {}
    for unit in ("participant", "account", "thread"):
        influence = influence_tables.get(unit, pd.DataFrame())
        finite = bool(
            not influence.empty
            and np.isfinite(
                pd.to_numeric(
                    influence.get("pareto_k", pd.Series(dtype=float)), errors="coerce"
                )
            ).all()
        )
        flagged_rows = influence[
            influence.get("needs_refit", pd.Series(True, index=influence.index))
            .fillna(True)
            .astype(bool)
        ].copy()
        flagged_keys = set(
            flagged_rows.get("group", pd.Series(dtype=object)).astype(str)
            + "::"
            + _id_text(flagged_rows.get("unit_id", pd.Series(dtype=object)))
        )
        refits = exact_influence_refits.get(unit, pd.DataFrame()).copy()
        if refits.empty:
            refit_keys: set[str] = set()
            resolved_keys: set[str] = set()
            duplicate_refit_keys = False
        else:
            required_refit_columns = {
                "group",
                "unit_id",
                "exact_refit_completed",
                "authenticated_exact_refit",
                "recomputed_mcmc_passed",
                "recomputed_estimands_match",
            }
            missing_refit_columns = required_refit_columns.difference(
                refits.columns
            )
            if missing_refit_columns:
                refit_keys = set()
                resolved_keys = set()
                duplicate_refit_keys = True
            else:
                keys = (
                    refits["group"].astype(str)
                    + "::"
                    + _id_text(refits["unit_id"])
                )
                duplicate_refit_keys = bool(keys.duplicated(keep=False).any())
                refit_keys = set(keys)
                resolved = (
                    refits["exact_refit_completed"].fillna(False).astype(bool)
                    & refits["authenticated_exact_refit"]
                    .fillna(False)
                    .astype(bool)
                    & refits["recomputed_mcmc_passed"]
                    .fillna(False)
                    .astype(bool)
                    & refits["recomputed_estimands_match"]
                    .fillna(False)
                    .astype(bool)
                )
                resolved_keys = set(keys[resolved])
        unresolved = flagged_keys.difference(resolved_keys)
        unexpected = refit_keys.difference(flagged_keys)
        influence_pass = bool(
            finite
            and not duplicate_refit_keys
            and not unresolved
            and not unexpected
            and refit_keys == flagged_keys
        )
        add(
            f"rq1_{unit}_influence",
            influence_pass,
            (
                f"units={len(influence)}; problematic={len(flagged_keys)}; "
                f"exact_authenticated={len(resolved_keys)}; "
                f"unresolved={len(unresolved)}; unexpected={len(unexpected)}"
            ),
        )

    return pd.DataFrame(rows)


def browser_gate(linkage: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Evaluate browser linkage and interval gates without treating missing data as zero."""

    required = {
        "group",
        "manual_precision",
        "pre_action_linkage_rate",
        "verified_interval_rate",
    }
    missing = sorted(required.difference(linkage.columns))
    if missing:
        raise ValueError(f"Browser gate table is missing columns: {missing}")
    rows = linkage[list(required)].drop_duplicates("group").copy()
    expected = {"G1", "G2"}
    if set(rows["group"]) != expected:
        raise ValueError(f"Browser gates require exactly {sorted(expected)}")
    gates = config.browser_gates
    rows["manual_precision_pass"] = rows["manual_precision"] >= gates.min_manual_precision
    rows["pre_action_linkage_pass"] = rows["pre_action_linkage_rate"] >= gates.min_pre_action_linkage
    rows["interval_closure_pass"] = rows["verified_interval_rate"] >= gates.min_interval_closure
    arm_gap = float(rows["pre_action_linkage_rate"].max() - rows["pre_action_linkage_rate"].min())
    rows["arm_linkage_gap"] = arm_gap
    rows["arm_linkage_gap_pass"] = arm_gap <= gates.max_arm_linkage_gap
    rows["target_choice_browser_pass"] = (
        rows["manual_precision_pass"] & rows["pre_action_linkage_pass"] & rows["arm_linkage_gap_pass"]
    )
    rows["full_interval_hazard_pass"] = rows["target_choice_browser_pass"] & rows["interval_closure_pass"]
    return rows


def latent_block_gate(
    comparison: pd.DataFrame,
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the prespecified predictive, recovery, and stability gate by K."""

    required_comparison = {"k", "elpd_difference", "elpd_difference_se"}
    required_recovery = {"k", "recovery_rate", "median_ari"}
    required_stability = {"k", "stable_membership_share", "smallest_block_n", "degree_reducible"}
    for name, frame, required in [
        ("comparison", comparison, required_comparison),
        ("recovery", recovery, required_recovery),
        ("stability", stability, required_stability),
    ]:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Latent-block {name} table is missing columns: {missing}")
    merged = comparison.merge(recovery, on="k", validate="one_to_one").merge(
        stability, on="k", validate="one_to_one"
    )
    merged["predictive_pass"] = merged["elpd_difference"] >= 2 * merged["elpd_difference_se"]
    merged["recovery_pass"] = (merged["recovery_rate"] >= 0.80) & (merged["median_ari"] >= 0.80)
    merged["stability_pass"] = (
        (merged["stable_membership_share"] >= 0.90)
        & (merged["smallest_block_n"] >= 5)
        & ~merged["degree_reducible"].astype(bool)
    )
    merged["manuscript_prominence_pass"] = (
        merged["predictive_pass"] & merged["recovery_pass"] & merged["stability_pass"]
    )
    return merged


def observed_network_metrics(choice_sets: pd.DataFrame) -> pd.DataFrame:
    """Summarize the network quantities used in posterior predictive checks."""

    chosen = choice_sets[choice_sets["chosen"].astype(bool)].copy()
    required = {"group", "actor_id", "candidate_author_id", "candidate_author_type"}
    missing = sorted(required.difference(chosen.columns))
    if missing:
        raise ValueError(f"Chosen rows are missing network metric columns: {missing}")
    rows: list[dict[str, Any]] = []
    for group, arm in chosen.groupby("group", sort=True):
        ai = arm["candidate_author_type"].eq("AI")
        dyad_counts = arm.groupby(["actor_id", "candidate_author_id"]).size()
        repeat_events = int((dyad_counts - 1).clip(lower=0).sum())
        metrics = {
            "human_events": len(arm),
            "h_to_ai_share": float(ai.mean()),
            "distinct_ai_alters": int(arm.loc[ai, "candidate_author_id"].nunique()),
            "repeat_contact_share": repeat_events / len(arm) if len(arm) else np.nan,
        }
        if "prior_author_to_human_count" in arm:
            metrics["reciprocal_context_share"] = float(
                (pd.to_numeric(arm["prior_author_to_human_count"], errors="coerce") > 0).mean()
            )
        if "prior_two_path_count" in arm:
            metrics["two_path_closure_mean"] = float(
                pd.to_numeric(arm["prior_two_path_count"], errors="coerce").mean()
            )
        rows.extend({"group": group, "metric": metric, "observed": value} for metric, value in metrics.items())
    return pd.DataFrame(rows)


def write_network_audit(config: AnalysisConfig, choice_sets: pd.DataFrame, path: Path) -> None:
    """Write the required audit definition for the exact modeled network object."""

    chosen = choice_sets[choice_sets["chosen"].astype(bool)]
    groups = ", ".join(sorted(chosen["group"].astype(str).unique()))
    text = f"""# Network Audit: Opportunity-to-Network Publication Analysis

## Research question and network object

The model asks which prior same-thread content item a human selected at each observed reply event. The modeled object is a timestamped human-to-content choice; the content author supplies the account-level receiver and H/AI block.

## Node and edge definitions

- Human participants are senders. Human and AI content authors are receivers.
- A directed event runs from the human reply author to the selected content item and inherits that item's author.
- Repeated replies remain ordered events. They are not collapsed into a binary dyad.
- Candidate content authored by the sender is excluded.

## Time and treatment windows

- Analysis set: `{config.analysis_set}`.
- Start: `{config.start_time}`; end: `{config.end_time or 'snapshot extraction'}`.
- Groups present: {groups}.
- Warm-up and bug-confounded days are excluded from events, candidates, and the valid-analysis recency clock.

## Opportunity and projection

The primary risk set contains every earlier valid content item in the same root thread. Content remains explicit in the likelihood. Account-level summaries are derived after selection, so multiple eligible items by one author are preserved.

## Integrity checks

- Choice sets: {chosen['choice_set_id'].nunique():,}.
- Human reply events: {len(chosen):,}.
- Every retained set contains exactly one chosen candidate and at least two candidates.
- All history variables are calculated from events strictly earlier than the focal reply.
- Browser non-observation is represented separately from zero exposure.

## Model compatibility and claims

The relational-event likelihood matches the ordered choice object. Arm contrasts are ecological because each arm contains one community and a different cohort. Latent blocks describe residual receiver roles only after observed type, opportunity, activity, and receiver heterogeneity are adjusted.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_publication_readiness(
    *,
    integrity: pd.DataFrame,
    analysis_signature: pd.DataFrame,
    diagnostics: pd.DataFrame,
    signatures: pd.DataFrame,
    prior_sensitivity: pd.DataFrame,
    forward_scores: pd.DataFrame,
    forward_diagnostics: pd.DataFrame,
    forward_comparisons: pd.DataFrame,
    participant_psis: pd.DataFrame,
    participant_refits: pd.DataFrame,
    participant_reloo_summary: pd.DataFrame,
    predictive_checks: pd.DataFrame,
    event_timing_checks: pd.DataFrame,
    tracking_qc: pd.DataFrame,
    latent_status: pd.DataFrame,
    fit_bundle_authentication: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine computational, predictive, and claim-specific decision gates."""

    rows: list[dict[str, Any]] = []
    fit_bundle_authentication = (
        pd.DataFrame()
        if fit_bundle_authentication is None
        else fit_bundle_authentication
    )

    def add(gate: str, passed: bool, required: bool, observed: str) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "required_for_primary_release": required,
                "observed": observed,
            }
        )

    add(
        "canonical_data_integrity",
        not integrity.empty and integrity["passed"].fillna(False).astype(bool).all(),
        True,
        f"{int(integrity.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(integrity)} checks passed",
    )
    bundle_pass = bool(
        len(fit_bundle_authentication) == 1
        and fit_bundle_authentication.get(
            "authenticated",
            pd.Series(False, index=fit_bundle_authentication.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and fit_bundle_authentication.get(
            "required_tables_present",
            pd.Series(False, index=fit_bundle_authentication.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "fit_result_bundle_authentication",
        bundle_pass,
        True,
        (
            str(fit_bundle_authentication["authentication_status"].iloc[0])
            if len(fit_bundle_authentication) == 1
            else "missing fit-result bundle authentication"
        ),
    )
    required_models = {"M0", "M1", "M2", "M3"}
    main_signatures = signatures[signatures.get("role", pd.Series(dtype=object)).eq("main_nested_sequence")]
    signed_models = set(main_signatures.get("model", pd.Series(dtype=object)).astype(str))
    current_data_hash = (
        str(analysis_signature["model_data_sha256"].iloc[0])
        if len(analysis_signature) == 1 and "model_data_sha256" in analysis_signature
        else None
    )
    current_implementation_hash = (
        str(analysis_signature["implementation_sha256"].iloc[0])
        if len(analysis_signature) == 1 and "implementation_sha256" in analysis_signature
        else None
    )
    signature_pass = bool(
        required_models.issubset(signed_models)
        and main_signatures.get(
            "live_posterior_authenticated",
            pd.Series(False, index=main_signatures.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and main_signatures.get("dimension_match", pd.Series(False, index=main_signatures.index))
        .fillna(False)
        .astype(bool)
        .all()
        and main_signatures.get("data_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_data_hash)
        .all()
        and main_signatures.get("implementation_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_implementation_hash)
        .all()
        and main_signatures.get(
            "publication_length_request", pd.Series(False, index=main_signatures.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "signed_publication_length_main_posteriors",
        signature_pass,
        True,
        f"signed models={sorted(signed_models)}",
    )
    main_diagnostics = diagnostics[diagnostics.get("model", pd.Series(dtype=object)).isin(required_models)]
    main_mcmc_pass = bool(
        set(main_diagnostics.get("model", pd.Series(dtype=object))) == required_models
        and main_diagnostics.get("passed", pd.Series(False, index=main_diagnostics.index))
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "main_model_mcmc",
        main_mcmc_pass,
        True,
        f"{int(main_diagnostics.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/4 passed",
    )
    prior_diagnostic = diagnostics[
        diagnostics.get("model", pd.Series(dtype=object)).eq("M3_doubled_prior_scales")
    ]
    prior_mcmc_pass = bool(
        len(prior_diagnostic) == 1
        and prior_diagnostic["passed"].fillna(False).astype(bool).all()
    )
    prior_signature = signatures[
        signatures.get("role", pd.Series(dtype=object)).eq("doubled_prior_scales")
    ]
    prior_signed = bool(
        len(prior_signature) == 1
        and prior_signature.get(
            "live_posterior_authenticated",
            pd.Series(False, index=prior_signature.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and prior_signature.get("dimension_match", pd.Series(False, index=prior_signature.index))
        .fillna(False)
        .astype(bool)
        .all()
        and prior_signature.get(
            "publication_length_request", pd.Series(False, index=prior_signature.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and prior_signature.get("data_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_data_hash)
        .all()
        and prior_signature.get("implementation_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_implementation_hash)
        .all()
    )
    add(
        "doubled_prior_mcmc",
        prior_mcmc_pass and prior_signed,
        True,
        f"diagnostic_rows={len(prior_diagnostic)}; signed_rows={len(prior_signature)}",
    )
    prior_stability_pass = bool(
        not prior_sensitivity.empty
        and prior_sensitivity.get("passed", pd.Series(False, index=prior_sensitivity.index))
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "doubled_prior_substantive_stability",
        prior_stability_pass,
        True,
        f"{int(prior_sensitivity.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(prior_sensitivity)} estimands passed",
    )
    expected_forward = {(fold, model) for fold in (1, 2, 3) for model in required_models}
    expected_score_cells = {
        (fold, model, group)
        for fold in (1, 2, 3)
        for model in required_models
        for group in ("G1", "G2", "ALL")
    }
    observed_score_cells: list[tuple[int, str, str]] = []
    for row in forward_scores.itertuples(index=False):
        try:
            observed_score_cells.append(
                (int(row.fold), str(row.model), str(row.group))
            )
        except (AttributeError, TypeError, ValueError):
            continue
    observed_forward = {(fold, model) for fold, model, _ in observed_score_cells}
    forward_signatures = signatures[
        signatures.get("role", pd.Series(dtype=object)).astype(str).str.startswith("forward_fold_")
    ]
    signed_forward: set[tuple[int, str]] = set()
    for row in forward_signatures.itertuples(index=False):
        try:
            signed_forward.add(
                (int(str(row.role).rsplit("_", maxsplit=1)[-1]), str(row.model))
            )
        except (AttributeError, TypeError, ValueError):
            continue
    expected_comparisons = {
        (fold, comparator)
        for fold in ("1", "2", "3", "ALL")
        for comparator in ("M0", "M1", "M2")
    }
    comparison_cells: list[tuple[str, str]] = []
    for row in forward_comparisons.itertuples(index=False):
        try:
            raw_fold = str(row.fold)
            fold = "ALL" if raw_fold == "ALL" else str(int(float(raw_fold)))
            comparison_cells.append((fold, str(row.comparator_model)))
        except (AttributeError, TypeError, ValueError):
            continue
    score_values_finite = all(
        np.isfinite(
            pd.to_numeric(
                forward_scores.get(column, pd.Series(dtype=float)),
                errors="coerce",
            )
        ).all()
        for column in (
            "mean_future_log_score",
            "median_selected_target_rank",
            "ai_target_brier_score",
            "ai_calibration_difference",
        )
    )
    comparison_values_finite = all(
        np.isfinite(
            pd.to_numeric(
                forward_comparisons.get(column, pd.Series(dtype=float)),
                errors="coerce",
            )
        ).all()
        for column in ("elpd_difference", "elpd_difference_se")
    )
    forward_complete = bool(
        expected_forward == observed_forward
        and expected_score_cells == set(observed_score_cells)
        and len(observed_score_cells) == len(expected_score_cells)
        and expected_forward == signed_forward
        and len(forward_signatures) == len(expected_forward)
        and expected_comparisons == set(comparison_cells)
        and len(comparison_cells) == len(expected_comparisons)
        and score_values_finite
        and comparison_values_finite
        and forward_signatures.get(
            "live_posterior_authenticated",
            pd.Series(False, index=forward_signatures.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and forward_signatures.get(
            "publication_length_request", pd.Series(False, index=forward_signatures.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and forward_signatures.get(
            "dimension_match", pd.Series(False, index=forward_signatures.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and forward_signatures.get(
            "forward_training_data_match",
            pd.Series(False, index=forward_signatures.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and forward_signatures.get("implementation_sha256", pd.Series(dtype=object))
        .astype(str)
        .eq(current_implementation_hash)
        .all()
    )
    add(
        "three_origin_M0_M3_forward_validation",
        forward_complete,
        True,
        (
            f"score cells={len(observed_score_cells)}/36; "
            f"signed cells={len(signed_forward)}/12; comparisons={len(comparison_cells)}/12"
        ),
    )
    diagnostic_cells: list[tuple[int, str]] = []
    for row in forward_diagnostics.itertuples(index=False):
        try:
            diagnostic_cells.append((int(row.fold), str(row.model)))
        except (AttributeError, TypeError, ValueError):
            continue
    forward_mcmc_pass = bool(
        set(diagnostic_cells) == expected_forward
        and len(diagnostic_cells) == len(expected_forward)
        and forward_diagnostics.get(
            "passed", pd.Series(False, index=forward_diagnostics.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "forward_fit_mcmc",
        forward_mcmc_pass,
        True,
        f"{int(forward_diagnostics.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(forward_diagnostics)} passed",
    )
    expected_participants = (
        int(analysis_signature["participants"].iloc[0])
        if len(analysis_signature) == 1 and "participants" in analysis_signature
        else -1
    )
    expected_choice_sets = (
        int(analysis_signature["choice_sets"].iloc[0])
        if len(analysis_signature) == 1 and "choice_sets" in analysis_signature
        else -1
    )
    participant_keys_list = (
        participant_psis.get("group", pd.Series(dtype=object)).astype(str)
        + "::"
        + participant_psis.get("actor_id", pd.Series(dtype=object)).astype(str)
    )
    participant_numeric_finite = all(
        np.isfinite(
            pd.to_numeric(
                participant_psis.get(column, pd.Series(dtype=float)),
                errors="coerce",
            )
        ).all()
        for column in ("choice_sets", "pareto_k", "elpd_loo", "relative_efficiency")
    )
    participant_coverage_pass = bool(
        not participant_psis.empty
        and len(participant_keys_list) == participant_keys_list.nunique()
        and len(participant_psis) == expected_participants
        and int(
            pd.to_numeric(
                participant_psis.get("choice_sets", pd.Series(dtype=float)),
                errors="coerce",
            ).sum()
        )
        == expected_choice_sets
        and participant_numeric_finite
        and participant_psis.get(
            "recomputed_from_authenticated_posterior",
            pd.Series(False, index=participant_psis.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "participant_grouped_PSIS_coverage",
        participant_coverage_pass,
        True,
        (
            f"unique participants={participant_keys_list.nunique()}/{expected_participants}; "
            f"choice sets={int(pd.to_numeric(participant_psis.get('choice_sets', pd.Series(dtype=float)), errors='coerce').sum())}/{expected_choice_sets}"
        ),
    )
    flagged = participant_psis[
        participant_psis.get("needs_refit", pd.Series(False, index=participant_psis.index))
        .fillna(False)
        .astype(bool)
    ]
    flagged_keys = {
        (str(row.group), str(row.actor_id)) for row in flagged.itertuples(index=False)
    }
    refit_keys = {
        (str(row.group), str(row.actor_id))
        for row in participant_refits.itertuples(index=False)
    } if {"group", "actor_id"}.issubset(participant_refits.columns) else set()
    participant_signatures = signatures[
        signatures.get("role", pd.Series(dtype=object))
        .astype(str)
        .str.startswith("participant_deletion_")
    ]
    signed_refit_keys: set[tuple[str, str]] = set()
    for role in participant_signatures.get("role", pd.Series(dtype=object)).astype(str):
        label = role.removeprefix("participant_deletion_")
        if "::" in label:
            group, actor_id = label.split("::", maxsplit=1)
            signed_refit_keys.add((group, actor_id))
    refit_signatures_pass = bool(
        signed_refit_keys == flagged_keys
        and participant_signatures.get(
            "live_posterior_authenticated",
            pd.Series(False, index=participant_signatures.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and participant_signatures.get(
            "dimension_match", pd.Series(False, index=participant_signatures.index)
        )
        .fillna(False)
        .astype(bool)
        .all()
        and participant_signatures.get(
            "publication_length_request",
            pd.Series(False, index=participant_signatures.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and participant_signatures.get(
            "implementation_sha256", pd.Series(dtype=object)
        )
        .astype(str)
        .eq(current_implementation_hash)
        .all()
    )
    reloo_summary_pass = bool(
        len(participant_reloo_summary) == 1
        and participant_reloo_summary.get(
            "all_problematic_contributions_replaced",
            pd.Series(False, index=participant_reloo_summary.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
    )
    refit_rows_authenticated = bool(
        len(participant_refits) == len(refit_keys)
        and participant_refits.get(
            "authenticated_refit_contribution",
            pd.Series(False, index=participant_refits.index),
        )
        .fillna(False)
        .astype(bool)
        .all()
        and np.isfinite(
            pd.to_numeric(
                participant_refits.get(
                    "exact_refit_elpd", pd.Series(dtype=float)
                ),
                errors="coerce",
            )
        ).all()
    )
    exact_refit_pass = bool(
        participant_coverage_pass
        and reloo_summary_pass
        and (
            (flagged.empty and participant_refits.empty and not signed_refit_keys)
        or (
            flagged_keys == refit_keys
            and refit_signatures_pass
            and refit_rows_authenticated
            and participant_refits.get("refit_status", pd.Series(dtype=object))
            .eq("completed_exact_participant_deletion")
            .all()
            and participant_refits.get(
                "recomputed_mcmc_passed",
                pd.Series(False, index=participant_refits.index),
            )
            .fillna(False)
            .astype(bool)
            .all()
        )
        )
    )
    add(
        "participant_grouped_PSIS_exact_refits",
        exact_refit_pass,
        True,
        (
            f"flagged={len(flagged)}; exact_refits={len(participant_refits)}; "
            f"signed_refits={len(signed_refit_keys)}; reloo_summary={len(participant_reloo_summary)}"
        ),
    )
    conditional_scope_pass = bool(
        not predictive_checks.empty
        and predictive_checks.get("simulation_scope", pd.Series(dtype=object))
        .eq(
            "recursively_updated_network_history_conditional_on_observed_"
            "event_times_and_thread_risk_sets"
        )
        .all()
    )
    add(
        "recursive_network_check_scope_labeled",
        conditional_scope_pass,
        True,
        "network history updates recursively; event times and thread risk sets remain observed",
    )
    required_predictive_metrics = {
        "h_to_ai_share",
        "distinct_ai_alters",
        "repeat_contact_share",
        "top_alter_concentration",
        "reciprocal_context_share",
        "two_path_closure_mean",
        "receiver_in_degree_gini",
        "receiver_in_degree_median",
        "receiver_in_degree_q90",
        "receiver_in_degree_max",
    }
    predictive_cells = predictive_checks[
        predictive_checks.get("metric", pd.Series(dtype=object)).isin(
            required_predictive_metrics
        )
    ]
    expected_predictive_cells = {
        (group, metric)
        for group in ("G1", "G2")
        for metric in required_predictive_metrics
    }
    observed_predictive_cells = set(
        zip(
            predictive_cells.get("group", pd.Series(dtype=object)).astype(str),
            predictive_cells.get("metric", pd.Series(dtype=object)).astype(str),
        )
    )
    covered = predictive_cells.get(
        "covered", pd.Series(False, index=predictive_cells.index)
    ).fillna(False).astype(bool)
    coverage_by_arm = (
        predictive_cells.assign(_covered=covered)
        .groupby("group", sort=False)["_covered"]
        .sum()
        .to_dict()
        if "group" in predictive_cells.columns
        else {}
    )
    focal_predictive_metrics = {
        "h_to_ai_share",
        "repeat_contact_share",
        "top_alter_concentration",
        "reciprocal_context_share",
        "two_path_closure_mean",
    }
    focal_cells = predictive_cells[
        predictive_cells.get("metric", pd.Series(dtype=object)).isin(
            focal_predictive_metrics
        )
    ]
    expected_focal_cells = {
        (group, metric)
        for group in ("G1", "G2")
        for metric in focal_predictive_metrics
    }
    covered_focal_cells = {
        (str(row.group), str(row.metric))
        for row in focal_cells.itertuples(index=False)
        if pd.notna(row.covered) and bool(row.covered)
    }
    predictive_reproduction_pass = bool(
        expected_predictive_cells == observed_predictive_cells
        and expected_focal_cells == covered_focal_cells
        and int(covered.sum()) >= 16
        and all(int(coverage_by_arm.get(group, 0)) >= 8 for group in ("G1", "G2"))
    )
    add(
        "recursive_network_summary_reproduction",
        predictive_reproduction_pass,
        True,
        (
            f"covered={int(covered.sum())}/{len(expected_predictive_cells)} arm-metric cells; "
            f"G1={int(coverage_by_arm.get('G1', 0))}/10; "
            f"G2={int(coverage_by_arm.get('G2', 0))}/10; "
            f"focal covered={len(covered_focal_cells)}/{len(expected_focal_cells)}; "
            "thresholds=all focal cells, 16/20 overall, and 8/10 per arm"
        ),
    )
    expected_timing_metrics = {
        "median_gap_seconds",
        "q90_gap_seconds",
        "burst_gap_share_le_300_seconds",
        "long_gap_share_ge_3600_seconds",
        "consecutive_burst_pair_share",
        "lag1_log_gap_correlation",
    }
    timing_cells = event_timing_checks[
        event_timing_checks.get("metric", pd.Series(dtype=object)).isin(
            expected_timing_metrics
        )
    ]
    expected_timing_cells = {
        (group, metric)
        for group in ("G1", "G2")
        for metric in expected_timing_metrics
    }
    observed_timing_cells = set(
        zip(
            timing_cells.get("group", pd.Series(dtype=object)).astype(str),
            timing_cells.get("metric", pd.Series(dtype=object)).astype(str),
        )
    )
    timing_reproduction_pass = bool(
        expected_timing_cells == observed_timing_cells
        and timing_cells.get("covered", pd.Series(False, index=timing_cells.index))
        .fillna(False)
        .astype(bool)
        .all()
    )
    add(
        "event_timing_reproduction",
        timing_reproduction_pass,
        False,
        (
            f"covered={int(timing_cells.get('covered', pd.Series(dtype=bool)).fillna(False).sum())}"
            f"/{len(expected_timing_cells)} arm-metric cells; conditional on event count"
        ),
    )
    pooled_comparison = forward_comparisons[
        forward_comparisons.get("fold", pd.Series(dtype=object)).astype(str).eq("ALL")
        & forward_comparisons.get("comparator_model", pd.Series(dtype=object)).eq("M1")
    ]
    pooled_delta = (
        float(pooled_comparison["elpd_difference"].iloc[0])
        if len(pooled_comparison) == 1
        else np.nan
    )
    pooled_se = (
        float(pooled_comparison["elpd_difference_se"].iloc[0])
        if len(pooled_comparison) == 1
        else np.nan
    )
    m3_noninferior = bool(
        np.isfinite(pooled_delta)
        and np.isfinite(pooled_se)
        and pooled_se >= 0.0
        and pooled_delta >= -2.0 * pooled_se
    )
    add(
        "M3_future_predictive_noninferiority_to_M1",
        m3_noninferior,
        True,
        (
            f"pooled future ELPD M3-M1={pooled_delta:.3f}; SE={pooled_se:.3f}; "
            "required >= -2 SE"
        ),
    )
    network_gain = bool(
        np.isfinite(pooled_delta)
        and np.isfinite(pooled_se)
        and pooled_se >= 0.0
        and pooled_delta >= 2.0 * pooled_se
    )
    add(
        "network_history_predictive_gain_over_M1",
        network_gain,
        False,
        (
            "claim gate for repetition, reciprocity, popularity, and closure; "
            f"pooled future ELPD M3-M1={pooled_delta:.3f}; SE={pooled_se:.3f}"
        ),
    )
    target_tracking = tracking_qc[
        tracking_qc.get("metric", pd.Series(dtype=object)).eq(
            "target_choice_tracking_gates_pass"
        )
    ]
    target_browser_pass = bool(
        len(target_tracking) == 1
        and target_tracking["gate_pass"].fillna(False).astype(bool).all()
    )
    add(
        "browser_target_choice_inference",
        target_browser_pass,
        False,
        "required only for inferential strict rendered-risk-set comparisons",
    )
    interval_tracking = tracking_qc[
        tracking_qc.get("metric", pd.Series(dtype=object)).eq(
            "full_interval_hazard_gates_pass"
        )
    ]
    interval_browser_pass = bool(
        len(interval_tracking) == 1
        and interval_tracking["gate_pass"].fillna(False).astype(bool).all()
    )
    add(
        "browser_interval_hazard_inference",
        interval_browser_pass,
        False,
        "required only for interval hazards and standardized exposure-time conversion",
    )
    latent_pass = bool(
        not latent_status.empty
        and latent_status.get(
            "manuscript_prominence_pass", pd.Series(False, index=latent_status.index)
        )
        .fillna(False)
        .astype(bool)
        .any()
    )
    add(
        "latent_block_manuscript_prominence",
        latent_pass,
        False,
        "observed-type model remains primary when this gate is closed",
    )
    required_rows = [row for row in rows if row["required_for_primary_release"]]
    rows.append(
        {
            "gate": "primary_estimates_release",
            "passed": all(row["passed"] for row in required_rows),
            "required_for_primary_release": True,
            "observed": f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed",
        }
    )
    return pd.DataFrame(rows)


def run_validation(
    config: AnalysisConfig,
    choice_sets: pd.DataFrame,
    event_history: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write deterministic validation inputs and the network audit report."""

    from .timing import event_timing_posterior_checks

    assignments = rolling_origin_assignments(choice_sets)
    observed = observed_network_metrics(choice_sets)
    timing = event_timing_posterior_checks(
        event_history,
        random_seed=config.seed,
    )
    write_table(assignments, config.tables_dir / "forward_validation_assignments")
    write_table(observed, config.tables_dir / "observed_network_metrics")
    write_table(timing, config.tables_dir / "event_timing_posterior_checks")
    write_network_audit(config, choice_sets, config.reports_dir / "network-audit.md")
    write_json(
        {
            "browser_gates": asdict(config.browser_gates),
            "fit_result_bundle_gate": {
                "required_for_primary_release": True,
                "binds_source_table_posterior_and_fit_signature_bytes": True,
            },
            "mcmc_gate": {"max_rhat": 1.01, "min_ess_bulk": 400, "min_ess_tail": 400, "divergences": 0},
            "forward_validation_gate": {
                "rolling_origins": 3,
                "models_per_origin": 4,
                "score_groups": ["G1", "G2", "ALL"],
                "fresh_training_hash_required": True,
                "M3_vs_M1_primary_noninferiority": "ELPD difference >= -2 SE",
                "network_history_claim_gate": "ELPD difference >= +2 SE",
                "uncertainty_unit": "participant_within_arm",
            },
            "participant_grouped_reloo_gate": {
                "all_current_participants_required": True,
                "problematic_threshold": 0.70,
                "primary_gauss_hermite_nodes": 32,
                "validation_gauss_hermite_nodes": 64,
                "maximum_absolute_node_doubling_difference": 0.0001,
            },
            "posterior_check_gate": {
                "minimum_network_summary_arm_cells_covered_overall": "16/20",
                "minimum_network_summary_cells_covered_per_arm": "8/10",
                "all_focal_cells_required": [
                    "h_to_ai_share",
                    "repeat_contact_share",
                    "top_alter_concentration",
                    "reciprocal_context_share",
                    "two_path_closure_mean",
                ],
                "recursive_network_history": True,
                "event_timing_calibration_required_for_release": False,
                "timing_scope": "conditional_on_observed_event_count_and_valid_analysis_clock",
            },
            "latent_block_gate": {
                "minimum_elpd_se_multiples": 2,
                "minimum_recovery_rate": 0.80,
                "minimum_median_ari": 0.80,
                "minimum_recovery_replicates": 3,
                "minimum_stable_membership_share": 0.90,
                "minimum_pairwise_chain_ari": 0.80,
                "minimum_receiver_membership_probability": 0.80,
                "minimum_confident_membership_share": 0.90,
                "maximum_mean_normalized_membership_entropy": 0.25,
                "minimum_block_size": 5,
                "degree_reducible_when_cv_balanced_accuracy_at_least": 0.75,
                "degree_reducible_when_accuracy_gain_at_least": 0.20,
                "maximum_observed_type_block_nmi": 0.50,
                "minimum_network_slope_difference": 0.10,
                "ysocial_sized_recovery_required": True,
                "opportunity_activity_stress_test_required": True,
                "opportunity_activity_stress_test_current_status": "fail_closed_not_run",
                "all_components_required": True,
            },
        },
        config.outputs_dir / "validation-gates.json",
    )
    return {
        "forward_validation_assignments": assignments,
        "observed_network_metrics": observed,
        "event_timing_posterior_checks": timing,
    }


def build_rq3_readiness(
    *,
    integrity: pd.DataFrame,
    round_map: pd.DataFrame,
    reaction_coverage: pd.DataFrame,
    vote_source: pd.DataFrame,
    panel: pd.DataFrame,
    panel_signature: pd.DataFrame,
    rebuilt_panel_sha256: str | None,
    primary_estimand: pd.DataFrame,
    contrasts: pd.DataFrame,
    resolution_sensitivity: pd.DataFrame,
    expected_bootstrap_reps: int,
) -> pd.DataFrame:
    """Apply fail-closed release gates for the RQ3 windowed-downvote analysis."""

    rows: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, observed: str) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "required_for_rq3_release": True,
                "observed": observed,
            }
        )

    integrity_pass = bool(
        not integrity.empty
        and integrity.get("passed", pd.Series(False, index=integrity.index)).fillna(False).astype(bool).all()
    )
    add(
        "rq3_canonical_data_integrity",
        integrity_pass,
        f"{int(integrity.get('passed', pd.Series(dtype=bool)).fillna(False).sum())}/{len(integrity)} checks passed",
    )

    map_columns = {"within_one_hour", "closed_form_match", "source", "enforced"}.issubset(round_map.columns)
    enforced = round_map[round_map["enforced"].astype(bool)] if map_columns else round_map
    map_pass = bool(
        map_columns
        and not enforced.empty
        and enforced["within_one_hour"].fillna(False).astype(bool).all()
        and enforced["closed_form_match"].fillna(False).astype(bool).all()
    )
    observed_rounds = int(round_map["source"].eq("observed_comments").sum()) if map_columns else 0
    closed_rounds = int(round_map["source"].eq("closed_form").sum()) if map_columns else 0
    prestart_rounds = int(round_map["source"].eq("observed_prestart_unverified").sum()) if map_columns else 0
    add(
        "rq3_round_map_zero_residual",
        map_pass,
        f"{observed_rounds} observed rounds inside one hour and equal to the closed form; "
        f"{closed_rounds} reaction-only rounds from the closed form; "
        f"{prestart_rounds} pre-start rounds unverified and excluded",
    )

    coverage_pass = bool(
        not reaction_coverage.empty
        and "coverage_after" in reaction_coverage.columns
        and reaction_coverage["coverage_after"].fillna(0.0).ge(1.0).all()
    )
    add(
        "rq3_reaction_timestamp_coverage",
        coverage_pass,
        (
            f"{int(reaction_coverage['timestamped_after'].sum())}/{int(reaction_coverage['total_rows'].sum())} "
            "reaction rows timestamped"
            if "total_rows" in reaction_coverage.columns
            else "no reaction rows"
        ),
    )

    votes_pass = bool(
        not vote_source.empty
        and {"actor_type", "rows"}.issubset(vote_source.columns)
        and vote_source["actor_type"].astype(str).str.upper().eq("H").all()
        and vote_source["rows"].sum() > 0
    )
    add(
        "rq3_human_actor_only_votes",
        votes_pass,
        f"{int(vote_source['rows'].sum()) if 'rows' in vote_source.columns else 0} human reactions entered the windows",
    )

    signature_row = panel_signature.iloc[0] if len(panel_signature) == 1 else None
    stored_sha = str(signature_row["panel_sha256"]) if signature_row is not None else None
    reproduction_pass = bool(stored_sha is not None and rebuilt_panel_sha256 == stored_sha)
    add(
        "rq3_panel_reproduces_from_sources",
        reproduction_pass,
        f"stored={stored_sha}; rebuilt={rebuilt_panel_sha256}",
    )

    rq1_rows = panel[panel["in_rq1_analysis_set"].astype(bool)] if "in_rq1_analysis_set" in panel.columns else panel
    by_group = rq1_rows.groupby("group").size() if not rq1_rows.empty else pd.Series(dtype=int)
    expected = {
        "G1": int(signature_row["rq1_choice_sets_g1"]) if signature_row is not None else -1,
        "G2": int(signature_row["rq1_choice_sets_g2"]) if signature_row is not None else -1,
    }
    panel_pass = bool(
        signature_row is not None
        and all(int(by_group.get(group, 0)) == expected[group] for group in ("G1", "G2"))
        and sum(expected.values()) > 0
    )
    add(
        "rq3_panel_matches_rq1_analysis_set",
        panel_pass,
        f"panel rq1 rows G1={int(by_group.get('G1', 0))}, G2={int(by_group.get('G2', 0))}; "
        f"choice sets G1={expected['G1']}, G2={expected['G2']}",
    )

    numeric = ["estimate", "ci_low", "ci_high"]
    estimand_columns = {"group", "reaction_placement", *numeric}.issubset(primary_estimand.columns)
    estimand_pass = bool(
        estimand_columns
        and len(primary_estimand) == 3
        and set(primary_estimand["group"]) == {"G1", "G2", "G2-G1"}
        and np.isfinite(primary_estimand[numeric].to_numpy(dtype=float)).all()
        and primary_estimand["reaction_placement"].eq("round_start").all()
    )
    add(
        "rq3_primary_estimand_complete",
        estimand_pass,
        (
            f"{len(primary_estimand)} rows; groups={sorted(primary_estimand['group'].astype(str))}"
            if estimand_columns
            else "missing primary estimand"
        ),
    )

    focal = resolution_sensitivity[
        resolution_sensitivity["subset"].eq("rq1_analysis_set")
        & resolution_sensitivity["dyad"].eq("H->AI")
        & resolution_sensitivity["metric"].str.startswith("target_received_human_downvote")
    ] if {"subset", "dyad", "metric"}.issubset(resolution_sensitivity.columns) else resolution_sensitivity
    direction_pass = bool(not focal.empty and focal["direction_stable"].fillna(False).astype(bool).all())
    add(
        "rq3_resolution_direction_stable",
        direction_pass,
        (
            "; ".join(
                f"{getattr(row, 'comparison', 'round_start_vs_round_end')} {row.metric}: {int(row.rows_reclassified)} reclassified, "
                f"contrast {row.contrast_round_start:+.4f} -> {row.contrast_round_end:+.4f}"
                for row in focal.itertuples(index=False)
            )
            if not focal.empty
            else "no sensitivity rows"
        ),
    )

    reps = int(contrasts["bootstrap_reps"].min()) if "bootstrap_reps" in contrasts.columns and not contrasts.empty else 0
    add(
        "rq3_bootstrap_reps_release",
        reps >= int(expected_bootstrap_reps),
        f"{reps} replicates (required >= {int(expected_bootstrap_reps)})",
    )
    return pd.DataFrame(rows)
