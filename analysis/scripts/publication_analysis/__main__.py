"""Command-line entry point for the publication analysis pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import load_config
from .pipeline import (
    run_fit,
    run_fit_choice,
    run_prepare,
    run_estimate_rq3,
    run_prepare_choice,
    run_prepare_rq2,
    run_prepare_rq3,
    run_fit_rq2,
    run_report_rq2,
    run_report_rq3,
    run_report_stage,
    run_validate,
    run_validate_choice,
    run_validate_rq2,
    run_validate_rq3,
)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis.yaml"),
        help="Analysis config YAML.",
    )
    parser.add_argument("--profile", choices=("pilot", "release"), default="release", help="Sampling profile.")
    parser.add_argument("--run-id", default="manual-run", help="Named run directory under output_root.")


def _add_sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", default="M0,M1,M2,M3", help="Comma-separated nested models.")
    parser.add_argument("--draws", type=int, help="Override retained posterior draws per chain.")
    parser.add_argument("--tune", type=int, help="Override warm-up draws per chain.")
    parser.add_argument("--chains", type=int, help="Override chain count.")
    parser.add_argument(
        "--skip-browser-sensitivity",
        action="store_true",
        help="Skip the strict pre-action rendered-risk-set sensitivity fit.",
    )
    parser.add_argument(
        "--skip-prior-sensitivity",
        action="store_true",
        help="Skip the doubled-prior-scale M3 sensitivity fit.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refit matching completed posterior files instead of resuming them.",
    )


def _add_choice_sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draws", type=int, help="Override retained draws per chain.")
    parser.add_argument("--tune", type=int, help="Override warm-up draws per chain.")
    parser.add_argument("--chains", type=int, help="Override chain count.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refit matching signed RQ1 posterior files intentionally.",
    )


def _add_latent_recovery(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--latent-recovery-replicates",
        type=int,
        default=3,
        help="Independent bounded recovery fixtures per K (minimum 3).",
    )
    parser.add_argument(
        "--latent-recovery-draws",
        type=int,
        default=500,
        help="Retained draws per chain for each bounded recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-actors",
        type=int,
        default=12,
        help="Human actors per conditional-choice recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-receivers",
        type=int,
        default=24,
        help="Receiver accounts per conditional-choice recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-choice-sets",
        type=int,
        default=240,
        help="Choice sets per conditional-choice recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-candidates",
        type=int,
        default=5,
        help="Candidates per conditional-choice recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-tune",
        type=int,
        default=500,
        help="Warm-up draws per chain for each bounded recovery fixture.",
    )
    parser.add_argument(
        "--latent-recovery-chains",
        type=int,
        default=2,
        help="Chains per bounded recovery fixture.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.publication_analysis",
        description="Opportunity-conditioned YSocial publication analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build canonical data and browser tables.")
    _add_config(prepare)

    prepare_choice = subparsers.add_parser(
        "prepare-choice",
        help="Build the canonical RQ1 same-thread choice tables.",
    )
    _add_config(prepare_choice)

    fit_choice = subparsers.add_parser(
        "fit-choice",
        help="Fit M1 and all declared RQ1 sensitivity and prediction checks.",
    )
    _add_config(fit_choice)
    _add_choice_sampling(fit_choice)

    validate_choice = subparsers.add_parser(
        "validate-choice",
        help="Authenticate and apply the fail-closed RQ1 release gates.",
    )
    _add_config(validate_choice)

    prepare_rq2 = subparsers.add_parser(
        "prepare-rq2",
        help="Build strict directional histories and the RQ2 model frame.",
    )
    _add_config(prepare_rq2)

    fit_rq2 = subparsers.add_parser(
        "fit-rq2",
        help="Fit the condition-interacted directional-contact RQ2 model.",
    )
    _add_config(fit_rq2)
    _add_choice_sampling(fit_rq2)
    fit_rq2.add_argument(
        "--estimand-draw-chunk-size",
        type=int,
        default=64,
        help="Posterior draws processed per probability-estimand chunk.",
    )

    validate_rq2 = subparsers.add_parser(
        "validate-rq2",
        help="Reproduce and authenticate the full-length RQ2 core fit.",
    )
    _add_config(validate_rq2)

    report_rq2 = subparsers.add_parser(
        "report-rq2",
        help="Render the validated RQ2 core-fit report.",
    )
    _add_config(report_rq2)

    prepare_rq3 = subparsers.add_parser(
        "prepare-rq3",
        help="Timestamp reactions from rounds and build the RQ3 sanction panel.",
    )
    _add_config(prepare_rq3)

    estimate_rq3 = subparsers.add_parser(
        "estimate-rq3",
        help="Compute windowed human-downvote outcomes, contrasts, and sensitivities.",
    )
    _add_config(estimate_rq3)
    estimate_rq3.add_argument("--bootstrap-reps", type=int, help="Override participant-bootstrap replicates.")

    validate_rq3 = subparsers.add_parser(
        "validate-rq3",
        help="Reproduce the RQ3 panel and apply the fail-closed release gates.",
    )
    _add_config(validate_rq3)

    report_rq3 = subparsers.add_parser("report-rq3", help="Render the RQ3 result report.")
    _add_config(report_rq3)

    fit = subparsers.add_parser("fit", help="Fit nested Bayesian relational-event models.")
    _add_config(fit)
    _add_sampling(fit)
    fit.add_argument("--include-latent", action="store_true", help="Fit gated latent K=2,3 models.")
    _add_latent_recovery(fit)
    fit.add_argument(
        "--run-forward-validation",
        action="store_true",
        help="Fit M0-M3 across all three rolling-origin validation folds.",
    )
    fit.add_argument(
        "--run-problematic-participant-refits",
        action="store_true",
        help="Exactly refit M3 for every participant with problematic grouped PSIS.",
    )

    validate = subparsers.add_parser("validate", help="Evaluate data, network, and model gates.")
    _add_config(validate)

    report = subparsers.add_parser("report", help="Render the canonical figures and report.")
    _add_config(report)

    manuscript_figures = subparsers.add_parser(
        "manuscript-figures",
        help="Render manuscript display figures from admitted tables in results/tables.",
    )
    manuscript_figures.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    manuscript_figures.add_argument("--figures-dir", type=Path, default=Path("artifacts/manuscript/figures"))

    all_parser = subparsers.add_parser("all", help="Run prepare, fit, validate, and report.")
    _add_config(all_parser)
    _add_sampling(all_parser)
    _add_latent_recovery(all_parser)
    all_parser.add_argument(
        "--skip-latent",
        action="store_true",
        help="Compatibility flag; latent receiver blocks are skipped by default.",
    )
    all_parser.add_argument(
        "--include-latent",
        action="store_true",
        help="Run the optional gated latent receiver-block extension.",
    )
    all_parser.add_argument(
        "--skip-forward-validation",
        action="store_true",
        help="Skip the three rolling-origin validation refits.",
    )
    return parser


def _models(value: str) -> tuple[str, ...]:
    result = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("--models cannot be empty")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "manuscript-figures":
        from .manuscript_figures import make_all

        paths = make_all(args.tables_dir.expanduser().resolve(), args.figures_dir.expanduser().resolve())
        for path in paths:
            print(f"Wrote {path}")
        return 0
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, profile=args.profile, run_id=args.run_id)
    if args.command in {
        "prepare-rq2",
        "fit-rq2",
        "validate-rq2",
        "report-rq2",
    }:
        config = replace(config, analysis_stage="rq2_m2")

    if args.command == "prepare":
        tables = run_prepare(config, config_path)
        print(f"Prepared {len(tables)} tables in {config.tables_dir}")
        return 0
    if args.command == "prepare-choice":
        tables = run_prepare_choice(config, config_path)
        print(f"Prepared {len(tables)} RQ1 tables in {config.tables_dir}")
        return 0
    if args.command == "fit-choice":
        tables = run_fit_choice(
            config,
            config_path,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            resume=not args.no_resume,
        )
        print(f"Fitted M1 and wrote {len(tables)} RQ1 result tables")
        return 0
    if args.command == "validate-choice":
        tables = run_validate_choice(config, config_path)
        readiness = tables["rq1_publication_readiness"]
        passed = bool(
            not readiness.empty
            and readiness["passed"].fillna(False).astype(bool).all()
        )
        print(f"RQ1 validation release_passed={passed}")
        return 0 if passed else 2
    if args.command == "prepare-rq2":
        tables = run_prepare_rq2(config, config_path)
        print(f"Prepared {len(tables)} RQ2 tables in {config.tables_dir}")
        return 0
    if args.command == "fit-rq2":
        tables = run_fit_rq2(
            config,
            config_path,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            resume=not args.no_resume,
            estimand_draw_chunk_size=args.estimand_draw_chunk_size,
        )
        print(f"Fitted RQ2 and wrote {len(tables)} core result tables")
        return 0
    if args.command == "validate-rq2":
        tables = run_validate_rq2(config, config_path)
        readiness = tables["rq2_core_readiness"]
        passed = bool(
            not readiness.empty
            and readiness["passed"].fillna(False).astype(bool).all()
        )
        print(f"RQ2 core validation passed={passed}")
        return 0 if passed else 2
    if args.command == "report-rq2":
        path = run_report_rq2(config, config_path)
        print(f"Wrote RQ2 core-fit report to {path}")
        return 0
    if args.command == "prepare-rq3":
        tables = run_prepare_rq3(config, config_path)
        print(f"Prepared {len(tables)} RQ3 tables in {config.tables_dir}")
        return 0
    if args.command == "estimate-rq3":
        tables = run_estimate_rq3(config, config_path, bootstrap_reps=args.bootstrap_reps)
        print(f"Estimated RQ3 outcomes and wrote {len(tables)} tables")
        return 0
    if args.command == "validate-rq3":
        tables = run_validate_rq3(config, config_path)
        readiness = tables["rq3_publication_readiness"]
        passed = bool(
            not readiness.empty
            and readiness["passed"].fillna(False).astype(bool).all()
        )
        print(f"RQ3 validation release_passed={passed}")
        return 0 if passed else 2
    if args.command == "report-rq3":
        path = run_report_rq3(config, config_path)
        print(f"Wrote RQ3 report to {path}")
        return 0
    if args.command == "fit":
        tables = run_fit(
            config,
            config_path,
            models=_models(args.models),
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            include_latent=args.include_latent,
            run_forward_validation=args.run_forward_validation,
            fit_browser_sensitivity=not args.skip_browser_sensitivity,
            run_prior_sensitivity=not args.skip_prior_sensitivity,
            resume=not args.no_resume,
            run_problematic_participant_refits=args.run_problematic_participant_refits,
            latent_recovery_replicates=args.latent_recovery_replicates,
            latent_recovery_actors=args.latent_recovery_actors,
            latent_recovery_receivers=args.latent_recovery_receivers,
            latent_recovery_choice_sets=args.latent_recovery_choice_sets,
            latent_recovery_candidates=args.latent_recovery_candidates,
            latent_recovery_draws=args.latent_recovery_draws,
            latent_recovery_tune=args.latent_recovery_tune,
            latent_recovery_chains=args.latent_recovery_chains,
        )
        print(f"Fitted models and wrote {len(tables)} result tables in {config.run_dir}")
        return 0
    if args.command == "validate":
        tables = run_validate(config, config_path)
        print(f"Validated {len(tables)} result surfaces in {config.run_dir}")
        return 0
    if args.command == "report":
        path = run_report_stage(config, config_path)
        print(f"Wrote publication report to {path}")
        return 0

    prepared = run_prepare(config, config_path)
    print(f"Prepared {len(prepared)} tables")
    fitted = run_fit(
        config,
        config_path,
        models=_models(args.models),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        include_latent=args.include_latent and not args.skip_latent,
        run_forward_validation=not args.skip_forward_validation,
        fit_browser_sensitivity=not args.skip_browser_sensitivity,
        run_prior_sensitivity=not args.skip_prior_sensitivity,
        resume=not args.no_resume,
        run_problematic_participant_refits=True,
        latent_recovery_replicates=args.latent_recovery_replicates,
        latent_recovery_actors=args.latent_recovery_actors,
        latent_recovery_receivers=args.latent_recovery_receivers,
        latent_recovery_choice_sets=args.latent_recovery_choice_sets,
        latent_recovery_candidates=args.latent_recovery_candidates,
        latent_recovery_draws=args.latent_recovery_draws,
        latent_recovery_tune=args.latent_recovery_tune,
        latent_recovery_chains=args.latent_recovery_chains,
    )
    print(f"Fitted models and wrote {len(fitted)} result tables")
    report_path = run_report_stage(config, config_path)
    print(f"Validated results and completed publication analysis: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
