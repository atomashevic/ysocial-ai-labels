#!/usr/bin/env python3
"""Run the two prespecified supporting diagnostics for the RQ2 core fit.

This module deliberately lives outside ``scripts.publication_analysis``. The
authenticated core-fit bundle binds that package byte for byte. The diagnostic
therefore reads and authenticates the core run, writes to a separate directory,
and leaves every core artifact unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SENSITIVITY_SCHEMA = "ysocial.rq2-sensitivity.v1"
SENSITIVITY_TABLE_NAMES = (
    "rq2_one_to_two_estimands",
    "rq2_doubled_prior_estimands",
    "rq2_prior_scale_sensitivity",
    "rq2_sensitivity_model_diagnostics",
    "rq2_sensitivity_fit_signature",
)
RQ2_CORE_GATE_NAMES = frozenset(
    {
        "rq2_canonical_data_and_strict_history",
        "rq2_ai_specific_condition_interacted_specification",
        "rq2_directional_first_and_repeat_contact_support",
        "rq2_fit_result_bundle_authentication",
        "rq2_signed_publication_length_core_posterior",
        "rq2_core_mcmc",
        "rq2_joint_probability_estimands",
        "rq2_grouped_influence_diagnostics_computed",
    }
)
ONE_TO_TWO_ESTIMANDS: Mapping[str, str] = {
    "human_to_ai": (
        "AI_account_selection_probability_difference_one_to_two_"
        "prior_human_to_AI_replies_24h"
    ),
    "ai_to_human": (
        "AI_account_selection_probability_difference_one_to_two_"
        "prior_AI_to_human_replies_24h"
    ),
}


def configure_conservative_threads() -> None:
    """Limit numerical libraries used by this standalone command to one thread."""

    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"


def doubled_prior_config(config: Any) -> Any:
    """Return the same configuration with each prior scale multiplied by two."""

    sampling = replace(
        config.sampling,
        prior_focal_sigma=2.0 * config.sampling.prior_focal_sigma,
        prior_adjustment_sigma=2.0 * config.sampling.prior_adjustment_sigma,
        prior_scale_sigma=2.0 * config.sampling.prior_scale_sigma,
    )
    return replace(config, sampling=sampling)


def _standardized_pair(
    manifest: Any,
    term: str,
    *,
    raw_low: float,
    raw_high: float,
) -> tuple[float, float]:
    import numpy as np

    rows = manifest[manifest["term"].astype(str).eq(term)]
    if len(rows) != 1:
        raise ValueError(f"RQ2 term manifest requires one row for {term!r}")
    row = rows.iloc[0]
    if str(row["status"]) != "included_standardized":
        raise ValueError(f"RQ2 term {term!r} must be included and standardized")
    mean = float(row["raw_mean"])
    standard_deviation = float(row["raw_sd"])
    if (
        not np.isfinite(mean)
        or not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        raise ValueError(f"RQ2 term {term!r} has an invalid training scaler")
    return (
        (raw_low - mean) / standard_deviation,
        (raw_high - mean) / standard_deviation,
    )


def one_to_two_scenarios(
    manifest: Any,
    *,
    reference_hours: float = 24.0,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Build count-one to count-two scenarios with recency fixed at 24 hours."""

    import numpy as np

    from scripts.publication_analysis.rq2 import (
        RQ2_AI_TERMS,
        RQ2_BASE_TERMS,
        RQ2_G2_AI_TERMS,
    )

    if not np.isfinite(reference_hours) or reference_hours <= 0:
        raise ValueError("RQ2 reference contact time must be positive and finite")
    fixed_recency = float(np.exp(-reference_hours / 24.0))
    raw_pairs = {
        "z_log_prior_contact_count": (
            float(np.log1p(1.0)),
            float(np.log1p(2.0)),
        ),
        "z_recent_prior_contact": (fixed_recency, fixed_recency),
        "z_log_reverse_contact_count": (
            float(np.log1p(1.0)),
            float(np.log1p(2.0)),
        ),
        "z_recent_reverse_contact": (fixed_recency, fixed_recency),
    }
    statuses = manifest.set_index("term")["status"].astype(str)
    scenarios: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for direction, base_terms in RQ2_BASE_TERMS.items():
        standardized = {
            base: _standardized_pair(
                manifest,
                base,
                raw_low=raw_pairs[base][0],
                raw_high=raw_pairs[base][1],
            )
            for base in base_terms
        }
        for base in base_terms:
            for interaction in (RQ2_AI_TERMS[base], RQ2_G2_AI_TERMS[base]):
                if statuses.get(interaction) != "included_unscaled":
                    raise ValueError(
                        f"RQ2 interaction {interaction!r} must be included unscaled"
                    )
        arm_values: dict[str, dict[str, tuple[float, float]]] = {}
        for group in ("G1", "G2"):
            values: dict[str, tuple[float, float]] = {}
            for base, pair in standardized.items():
                values[base] = pair
                values[RQ2_AI_TERMS[base]] = pair
                values[RQ2_G2_AI_TERMS[base]] = (
                    pair if group == "G2" else (0.0, 0.0)
                )
            arm_values[group] = values
        scenarios[direction] = arm_values
    return scenarios


def one_to_two_estimands(
    idata: Any,
    data: Any,
    term_manifest: Any,
    *,
    reference_hours: float = 24.0,
    draw_chunk_size: int = 32,
) -> Any:
    """Project the one-to-two contact contrast from an existing posterior."""

    import numpy as np
    import pandas as pd

    from scripts.publication_analysis.model import (
        _common_arm_frames,
        _posterior_summary,
        _posterior_utilities,
    )
    from scripts.publication_analysis.rq2 import (
        RQ2_DIRECTIONS,
        _beta_draws,
        _projection_from_utility,
    )

    if draw_chunk_size < 1:
        raise ValueError("RQ2 estimand draw chunk size must be positive")
    scenarios = one_to_two_scenarios(
        term_manifest,
        reference_hours=reference_hours,
    )
    g1_x, g2_x = _common_arm_frames(data)
    accumulated: dict[tuple[str, str], list[Any]] = {
        (direction, group): []
        for direction in RQ2_DIRECTIONS
        for group in ("G1", "G2")
    }
    opportunity_counts: dict[tuple[str, str], int] = {}
    chains = int(idata.posterior.sizes["chain"])
    draws = int(idata.posterior.sizes["draw"])
    for chain in range(chains):
        for start in range(0, draws, draw_chunk_size):
            stop = min(draws, start + draw_chunk_size)
            chunk = idata.isel(
                chain=slice(chain, chain + 1),
                draw=slice(start, stop),
            )
            beta, posterior_terms = _beta_draws(chunk)
            for group, arm_x in (("G1", g1_x), ("G2", g2_x)):
                utility = _posterior_utilities(chunk, data, x_override=arm_x)
                for direction in RQ2_DIRECTIONS:
                    values, opportunities = _projection_from_utility(
                        beta,
                        posterior_terms,
                        utility,
                        data,
                        arm_x,
                        scenarios[direction][group],
                    )
                    accumulated[(direction, group)].append(values)
                    key = (direction, group)
                    previous = opportunity_counts.setdefault(key, opportunities)
                    if previous != opportunities:
                        raise RuntimeError(
                            "RQ2 opportunity count changed across draw chunks"
                        )
                del utility

    rows: list[dict[str, Any]] = []
    for direction in RQ2_DIRECTIONS:
        g1 = np.concatenate(accumulated[(direction, "G1")])
        g2 = np.concatenate(accumulated[(direction, "G2")])
        if g1.shape != g2.shape:
            raise RuntimeError("RQ2 arm projections have different posterior lengths")
        opportunities = opportunity_counts[(direction, "G1")]
        if opportunities != opportunity_counts[(direction, "G2")]:
            raise RuntimeError("RQ2 arm projections have different opportunity counts")
        for group, values in (("G1", g1), ("G2", g2), ("G2-G1", g2 - g1)):
            row = _posterior_summary(values, ONE_TO_TWO_ESTIMANDS[direction], group)
            row.update(
                {
                    "direction": direction,
                    "condition": {
                        "G1": "labels_visible",
                        "G2": "labels_hidden",
                        "G2-G1": "labels_hidden_minus_visible",
                    }[group],
                    "estimate": float(np.median(values)),
                    "effect_scale": "absolute_selection_probability_difference",
                    "transition_low_count": 1,
                    "transition_high_count": 2,
                    "latest_contact_hours": float(reference_hours),
                    "reference_contact_hours": float(reference_hours),
                    "ai_receiver_opportunities": int(opportunities),
                    "standardization_scope": (
                        "pooled_choice_by_AI_receiver_opportunities"
                    ),
                    "counterfactual": (
                        "one eligible AI account at a time: one prior reply versus "
                        "two prior replies with latest contact fixed at "
                        f"{reference_hours:g} valid-clock hours; all eligible content "
                        "from that account moves jointly; opposite-direction history "
                        "and other covariates remain observed"
                    ),
                    "estimand_interpretation": (
                        "model_projected_association_not_causal_intervention"
                    ),
                    "inference_boundary": (
                        "one_community_per_condition_ecological_between_arm_contrast"
                    ),
                    "diagnostic_role": "supporting_projection_no_admission_gate",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def compare_prior_scales(primary: Any, doubled: Any) -> Any:
    """Compare zero-to-one estimands descriptively without a robustness gate."""

    import numpy as np
    import pandas as pd

    keys = ["estimand", "direction", "group", "condition"]
    summary_columns = [
        "posterior_median",
        "ci_low",
        "ci_high",
        "posterior_probability_gt_zero",
        "posterior_draws",
        "ai_receiver_opportunities",
    ]
    required = set(keys + summary_columns)
    for label, frame in (("primary", primary), ("doubled", doubled)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} estimands are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} estimands contain duplicate comparison keys")
        if len(frame) != 6:
            raise ValueError(f"{label} estimands must contain exactly six rows")

    left_keys = set(map(tuple, primary[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, doubled[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError("Primary and doubled-prior estimand keys differ")
    comparison = primary[keys + summary_columns].merge(
        doubled[keys + summary_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_doubled"),
    )
    comparison["median_shift_doubled_minus_primary"] = (
        comparison["posterior_median_doubled"]
        - comparison["posterior_median_primary"]
    )
    comparison["absolute_median_shift"] = comparison[
        "median_shift_doubled_minus_primary"
    ].abs()
    comparison["direction_stable"] = np.sign(
        comparison["posterior_median_primary"]
    ).eq(np.sign(comparison["posterior_median_doubled"]))
    comparison["credible_intervals_overlap"] = np.maximum(
        comparison["ci_low_primary"], comparison["ci_low_doubled"]
    ).le(np.minimum(comparison["ci_high_primary"], comparison["ci_high_doubled"]))
    comparison["prior_scale_multiplier"] = 2.0
    comparison["diagnostic_role"] = "supporting_diagnostic_only"
    return pd.DataFrame(comparison)


def validate_main_signature(
    signatures: Any,
    *,
    data_sha256: str,
    implementation_sha256: str,
    expected_path: Path | None = None,
) -> Any:
    """Return the unique authenticated full-length RQ2 main signature."""

    import pandas as pd

    required = {
        "model",
        "role",
        "posterior_path",
        "data_sha256",
        "implementation_sha256",
        "dimension_match",
        "publication_length_request",
        "live_posterior_authenticated",
    }
    missing = sorted(required.difference(signatures.columns))
    if missing:
        raise ValueError(f"RQ2 fit signatures are missing fields: {missing}")
    main = signatures[
        signatures["model"].astype(str).eq("RQ2")
        & signatures["role"].astype(str).eq("rq2_main")
    ]
    if len(main) != 1:
        raise ValueError("RQ2 sensitivity requires exactly one RQ2/rq2_main signature")
    row = main.iloc[0]
    checks = {
        "posterior_authentication": bool(row["live_posterior_authenticated"]),
        "dimension_match": bool(row["dimension_match"]),
        "publication_length_request": bool(row["publication_length_request"]),
        "model_data_sha256": str(row["data_sha256"]) == str(data_sha256),
        "implementation_sha256": str(row["implementation_sha256"])
        == str(implementation_sha256),
    }
    if expected_path is not None:
        checks["posterior_path"] = Path(str(row["posterior_path"])).resolve() == Path(
            expected_path
        ).resolve()
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise ValueError(
            "RQ2 main posterior failed sensitivity preflight: " + ", ".join(failed)
        )
    return pd.Series(row.copy())


def _validate_six_estimands(frame: Any, *, label: str) -> None:
    import numpy as np

    if len(frame) != 6:
        raise ValueError(f"{label} must contain exactly six estimand rows")
    expected = {
        (direction, group)
        for direction in ("human_to_ai", "ai_to_human")
        for group in ("G1", "G2", "G2-G1")
    }
    observed = set(
        map(
            tuple,
            frame[["direction", "group"]].astype(str).itertuples(
                index=False,
                name=None,
            ),
        )
    )
    if observed != expected:
        raise ValueError(f"{label} does not contain the six required cells")
    numeric = frame[
        [
            "posterior_mean",
            "posterior_median",
            "ci_low",
            "ci_high",
            "posterior_probability_gt_zero",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains nonfinite posterior summaries")


def authenticate_core_run(config: Any) -> dict[str, Any]:
    """Authenticate and load the minimum core artifacts needed by this diagnostic."""

    from scripts.publication_analysis.io import read_table
    from scripts.publication_analysis.pipeline import (
        RQ2_REQUIRED_FIT_TABLES,
        _model_implementation_hash,
        _prepared_data_hash,
        _rq2_model_data,
    )
    from scripts.publication_analysis.provenance import authenticate_fit_result_bundle
    from scripts.publication_analysis.validation import authenticate_fit_signatures

    model_frame = read_table(config.tables_dir / "rq2_model_frame")
    term_manifest = read_table(config.tables_dir / "rq2_model_term_manifest")
    data = _rq2_model_data(model_frame, term_manifest)
    data_hash = _prepared_data_hash(data)
    implementation_hash = _model_implementation_hash(config)
    bundle = authenticate_fit_result_bundle(
        config,
        current_model_data_sha256=data_hash,
        current_implementation_sha256=implementation_hash,
        required_tables=RQ2_REQUIRED_FIT_TABLES,
    )
    if not (
        len(bundle) == 1
        and bundle["authenticated"].fillna(False).astype(bool).all()
        and bundle["required_tables_present"].fillna(False).astype(bool).all()
    ):
        status = (
            str(bundle["authentication_status"].iloc[0])
            if len(bundle) == 1
            else "missing authentication row"
        )
        raise ValueError(f"RQ2 core bundle failed sensitivity preflight: {status}")

    core_readiness = read_table(config.tables_dir / "rq2_core_readiness")
    readiness_columns = {
        "gate",
        "passed",
        "required_for_rq2_core_fit",
    }
    readiness_names = (
        set(core_readiness["gate"].astype(str))
        if readiness_columns.issubset(core_readiness.columns)
        else set()
    )
    readiness_valid = bool(
        readiness_columns.issubset(core_readiness.columns)
        and len(core_readiness) == len(RQ2_CORE_GATE_NAMES)
        and core_readiness["gate"].astype(str).is_unique
        and readiness_names == RQ2_CORE_GATE_NAMES
        and core_readiness["passed"].fillna(False).astype(bool).all()
        and core_readiness["required_for_rq2_core_fit"]
        .fillna(False)
        .astype(bool)
        .all()
    )
    if not readiness_valid:
        raise ValueError("RQ2 sensitivity requires the exact eight passed core-fit gates")
    primary = read_table(config.tables_dir / "rq2_primary_estimands")
    _validate_six_estimands(primary, label="RQ2 primary estimands")
    signatures = authenticate_fit_signatures(
        read_table(config.tables_dir / "fit_run_signature")
    )
    main_path = config.models_dir / "M2_RQ2.nc"
    main = validate_main_signature(
        signatures,
        data_sha256=data_hash,
        implementation_sha256=implementation_hash,
        expected_path=main_path,
    )
    return {
        "model_frame": model_frame,
        "term_manifest": term_manifest,
        "data": data,
        "data_sha256": data_hash,
        "implementation_sha256": implementation_hash,
        "bundle_authentication": bundle,
        "core_readiness": core_readiness,
        "primary_estimands": primary,
        "main_signature": main,
        "main_posterior_path": main_path,
    }


def _sensitivity_paths(config: Any) -> dict[str, Path]:
    root = config.outputs_dir / "sensitivity"
    return {
        "root": root,
        "tables": root / "tables",
        "models": root / "models",
        "reports": root / "reports",
        "bundle": root / "rq2_sensitivity_bundle.json",
        "report": root / "reports" / "rq2_sensitivity_report.md",
        "posterior": root / "models" / "M2_RQ2_doubled_prior_scales.nc",
    }


def _clear_stale_outputs(paths: Mapping[str, Path]) -> None:
    for name in SENSITIVITY_TABLE_NAMES:
        for suffix in (".parquet", ".csv"):
            (paths["tables"] / f"{name}{suffix}").unlink(missing_ok=True)
    paths["report"].unlink(missing_ok=True)
    paths["bundle"].unlink(missing_ok=True)


def _markdown_table(frame: Any) -> str:
    return frame.to_markdown(index=False, floatfmt=".6f")


def _build_report(
    *,
    run_id: str,
    one_to_two: Any,
    comparison: Any,
    diagnostics: Any,
    primary_prior_scales: tuple[float, float, float],
    doubled_prior_scales: tuple[float, float, float],
) -> str:
    one_view = one_to_two[
        [
            "direction",
            "group",
            "posterior_median",
            "ci_low",
            "ci_high",
            "posterior_probability_gt_zero",
            "ai_receiver_opportunities",
        ]
    ]
    comparison_view = comparison[
        [
            "direction",
            "group",
            "posterior_median_primary",
            "posterior_median_doubled",
            "absolute_median_shift",
            "ci_low_primary",
            "ci_high_primary",
            "ci_low_doubled",
            "ci_high_doubled",
            "direction_stable",
            "credible_intervals_overlap",
        ]
    ]
    diagnostic_view = diagnostics[
        [
            "max_rhat",
            "min_ess_bulk",
            "min_ess_tail",
            "divergences",
            "passed",
            "sampling_extended",
        ]
    ]
    lines = [
        "# RQ2 supporting sensitivity diagnostics",
        "",
        f"- Core run ID: `{run_id}`",
        "- Role: supporting diagnostics with no added substantive admission gate",
        "- Resource policy: one sensitivity fit, four one-thread chains, bounded 32-draw projections",
        "",
        "## One-to-two prior-contact projection",
        "",
        (
            "This projection uses the authenticated main posterior. For one eligible AI "
            "account at a time, it changes the relevant prior-reply count from one to two "
            "and holds the latest reply at 24 valid-clock hours. All content from that "
            "account changes jointly. The opposite direction and all other covariates "
            "remain observed."
        ),
        "",
        _markdown_table(one_view),
        "",
        "## Doubled-prior refit",
        "",
        (
            "The refit doubles the focal fixed-effect prior scale, adjustment fixed-effect "
            "prior scale, and random-effect scale prior from "
            f"{primary_prior_scales[0]:g}, {primary_prior_scales[1]:g}, and "
            f"{primary_prior_scales[2]:g} to {doubled_prior_scales[0]:g}, "
            f"{doubled_prior_scales[1]:g}, and {doubled_prior_scales[2]:g}. "
            "Data, model terms, sampling length, chain count, and target acceptance "
            "remain unchanged."
        ),
        "",
        _markdown_table(diagnostic_view),
        "",
        "## Prior-scale comparison",
        "",
        (
            "The table compares the same zero-to-one probability estimands under the "
            "primary and doubled priors. Direction consistency, interval overlap, and "
            "median shifts are descriptive. No numerical robustness threshold is applied."
        ),
        "",
        _markdown_table(comparison_view),
        "",
        "## Interpretation boundary",
        "",
        (
            "Both diagnostics are model-based associations conditional on observed human "
            "replies and reconstructed same-thread opportunities. The G2 minus G1 contrast "
            "remains ecological because each condition contains one community."
        ),
        "",
    ]
    return "\n".join(lines)


def _diagnostic_implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _write_sensitivity_bundle(
    *,
    config: Any,
    paths: Mapping[str, Path],
    source_snapshot: list[dict[str, Any]],
    core: Mapping[str, Any],
    doubled_config: Any,
    diagnostics: Any,
    artifacts: list[Path],
) -> None:
    from scripts.publication_analysis.io import source_record, write_json

    diagnostic = diagnostics.iloc[0]
    bundle = {
        "schema": SENSITIVITY_SCHEMA,
        "run_id": config.run_id,
        "status": (
            "completed"
            if bool(diagnostic["passed"])
            else "completed_with_failed_mcmc_diagnostics"
        ),
        "diagnostic_role": "supporting_diagnostics_no_admission_gate",
        "core": {
            "bundle_authenticated_before_and_after": True,
            "core_gates_passed": 8,
            "model_data_sha256": core["data_sha256"],
            "implementation_sha256": core["implementation_sha256"],
            "main_posterior": source_record(core["main_posterior_path"]),
        },
        "fit_request": {
            "model": "RQ2",
            "role": "rq2_doubled_prior_scales",
            "draws": int(core["main_signature"]["draws"]),
            "tune": int(core["main_signature"]["tune"]),
            "chains": int(core["main_signature"]["chains"]),
            "target_accept": float(core["main_signature"]["target_accept"]),
            "prior_scale_multiplier": 2.0,
            "prior_focal_sigma": doubled_config.sampling.prior_focal_sigma,
            "prior_adjustment_sigma": doubled_config.sampling.prior_adjustment_sigma,
            "prior_scale_sigma": doubled_config.sampling.prior_scale_sigma,
            "seed_offset": 50,
        },
        "one_to_two_projection": {
            "source_posterior": "authenticated RQ2 main posterior",
            "transition_low_count": 1,
            "transition_high_count": 2,
            "latest_contact_hours": 24.0,
            "draw_chunk_size": 32,
        },
        "resource_policy": {
            "heavy_fits": 1,
            "maximum_chains": 4,
            "threads_per_chain": 1,
            "automatic_sampling_extensions_maximum": 1,
        },
        "diagnostic_implementation_sha256": _diagnostic_implementation_sha256(),
        "sources": source_snapshot,
        "artifacts": [source_record(path) for path in sorted(artifacts, key=str)],
    }
    write_json(bundle, paths["bundle"])


def run_sensitivity(
    *,
    config_path: Path,
    run_id: str,
    profile: str = "release",
    resume: bool = True,
    draw_chunk_size: int = 32,
) -> dict[str, Any]:
    """Authenticate the core run, execute one fit, and write separate diagnostics."""

    configure_conservative_threads()
    import arviz as az
    import pandas as pd

    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.io import (
        ensure_directories,
        source_record,
        write_table,
    )
    from scripts.publication_analysis.manifest import source_paths
    from scripts.publication_analysis.pipeline import (
        _fit_with_diagnostic_extension,
        _signature_row,
    )
    from scripts.publication_analysis.provenance import (
        assert_source_snapshot_unchanged,
        capture_source_snapshot,
    )
    from scripts.publication_analysis.rq2 import (
        add_estimand_denominators,
        standardized_estimands,
    )
    from scripts.publication_analysis.validation import authenticate_fit_signatures

    if draw_chunk_size != 32:
        raise ValueError("RQ2 sensitivity uses a fixed 32-draw memory bound")
    config_path = config_path.expanduser().resolve()
    config = replace(
        load_config(config_path, profile=profile, run_id=run_id),
        analysis_stage="rq2_m2",
    )
    if profile != "release":
        raise ValueError("Formal RQ2 sensitivity requires the release profile")
    paths = _sensitivity_paths(config)
    ensure_directories([paths["tables"], paths["models"], paths["reports"]])
    core = authenticate_core_run(config)
    input_paths = [
        *source_paths(config, config_path),
        Path(__file__).resolve(),
        config.tables_dir / "rq2_model_frame.parquet",
        config.tables_dir / "rq2_model_term_manifest.parquet",
        config.tables_dir / "rq2_primary_estimands.parquet",
        config.tables_dir / "rq2_core_readiness.parquet",
        config.tables_dir / "fit_run_signature.parquet",
        config.outputs_dir / "fit_result_bundle.json",
        core["main_posterior_path"],
        core["main_posterior_path"].with_suffix(".nc.fit.json"),
    ]
    source_snapshot = capture_source_snapshot(input_paths)
    _clear_stale_outputs(paths)

    main_idata = az.from_netcdf(core["main_posterior_path"])
    try:
        one_to_two = one_to_two_estimands(
            main_idata,
            core["data"],
            core["term_manifest"],
            reference_hours=24.0,
            draw_chunk_size=draw_chunk_size,
        )
        one_to_two = add_estimand_denominators(one_to_two, core["data"], config)
    finally:
        close_main = getattr(main_idata, "close", None)
        if callable(close_main):
            close_main()
        del main_idata
        gc.collect()
    _validate_six_estimands(one_to_two, label="RQ2 one-to-two estimands")

    main = core["main_signature"]
    draws = int(main["draws"])
    tune = int(main["tune"])
    chains = int(main["chains"])
    if draws < 1_000 or tune < 1_000 or chains != 4 or chains > 4:
        raise ValueError(
            "Formal doubled-prior sensitivity requires the authenticated "
            "1000/1000/four-chain core request"
        )
    doubled_config = doubled_prior_config(config)
    if float(doubled_config.sampling.target_accept) != float(main["target_accept"]):
        raise ValueError("Doubled-prior sensitivity must retain target acceptance")
    idata, signed_fit, diagnostic = _fit_with_diagnostic_extension(
        doubled_config,
        core["data"],
        "RQ2",
        paths["posterior"],
        role="rq2_doubled_prior_scales",
        draws=draws,
        tune=tune,
        chains=chains,
        seed_offset=50,
        resume=resume,
    )
    try:
        doubled_estimands = standardized_estimands(
            idata,
            core["data"],
            core["term_manifest"],
            reference_hours=24.0,
            draw_chunk_size=draw_chunk_size,
        )
        doubled_estimands = add_estimand_denominators(
            doubled_estimands,
            core["data"],
            config,
        )
    finally:
        close_sensitivity = getattr(idata, "close", None)
        if callable(close_sensitivity):
            close_sensitivity()
        del idata
        gc.collect()
    _validate_six_estimands(
        doubled_estimands,
        label="RQ2 doubled-prior estimands",
    )

    diagnostics = pd.DataFrame(
        [
            {
                "model": "RQ2",
                "role": "rq2_doubled_prior_scales",
                "diagnostic_scope": "computational_validity_only",
                **diagnostic,
            }
        ]
    )
    signature = pd.DataFrame(
        [_signature_row(paths["posterior"], signed_fit, core["data"])]
    )
    signature = authenticate_fit_signatures(signature)
    valid_sensitivity_signature = signature[
        signature["model"].astype(str).eq("RQ2")
        & signature["role"].astype(str).eq("rq2_doubled_prior_scales")
        & signature["live_posterior_authenticated"].fillna(False).astype(bool)
        & signature["dimension_match"].fillna(False).astype(bool)
        & signature["publication_length_request"].fillna(False).astype(bool)
        & signature["data_sha256"].astype(str).eq(core["data_sha256"])
        & signature["implementation_sha256"]
        .astype(str)
        .eq(core["implementation_sha256"])
    ]
    if len(valid_sensitivity_signature) != 1:
        raise ValueError("Doubled-prior posterior signature failed authentication")
    comparison = compare_prior_scales(
        core["primary_estimands"],
        doubled_estimands,
    )
    assert_source_snapshot_unchanged(source_snapshot)

    tables = {
        "rq2_one_to_two_estimands": one_to_two,
        "rq2_doubled_prior_estimands": doubled_estimands,
        "rq2_prior_scale_sensitivity": comparison,
        "rq2_sensitivity_model_diagnostics": diagnostics,
        "rq2_sensitivity_fit_signature": signature,
    }
    artifacts: list[Path] = []
    for name, frame in tables.items():
        written = write_table(frame, paths["tables"] / name)
        artifacts.extend(Path(value) for value in written.values())
    report_text = _build_report(
        run_id=config.run_id,
        one_to_two=one_to_two,
        comparison=comparison,
        diagnostics=diagnostics,
        primary_prior_scales=(
            float(config.sampling.prior_focal_sigma),
            float(config.sampling.prior_adjustment_sigma),
            float(config.sampling.prior_scale_sigma),
        ),
        doubled_prior_scales=(
            float(doubled_config.sampling.prior_focal_sigma),
            float(doubled_config.sampling.prior_adjustment_sigma),
            float(doubled_config.sampling.prior_scale_sigma),
        ),
    )
    paths["report"].write_text(report_text, encoding="utf-8")
    artifacts.extend(
        [
            paths["report"],
            paths["posterior"],
            paths["posterior"].with_suffix(".nc.fit.json"),
        ]
    )
    assert_source_snapshot_unchanged(source_snapshot)
    core_after = authenticate_core_run(config)
    if (
        str(core_after["main_signature"]["posterior_sha256"])
        != str(core["main_signature"]["posterior_sha256"])
    ):
        raise RuntimeError("RQ2 main posterior changed during sensitivity analysis")
    _write_sensitivity_bundle(
        config=config,
        paths=paths,
        source_snapshot=source_snapshot,
        core=core,
        doubled_config=doubled_config,
        diagnostics=diagnostics,
        artifacts=artifacts,
    )
    assert_source_snapshot_unchanged(source_snapshot)
    result = {
        "run_id": config.run_id,
        "sensitivity_root": str(paths["root"]),
        "report": str(paths["report"]),
        "bundle": str(paths["bundle"]),
        "doubled_posterior": source_record(paths["posterior"]),
        "doubled_mcmc_passed": bool(diagnostics["passed"].iloc[0]),
        "tables": list(tables),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded supporting sensitivity analysis for RQ2."
    )
    parser.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default="release", choices=("release",))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refit even when the exact signed doubled-prior posterior is reusable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_sensitivity(
        config_path=args.config,
        run_id=args.run_id,
        profile=args.profile,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    configure_conservative_threads()
    raise SystemExit(main())
