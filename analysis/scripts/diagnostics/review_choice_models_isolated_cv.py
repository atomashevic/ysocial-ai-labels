"""RQ1 isolated forward-validation comparison of the fixed AI-authorship terms.

Review item 'Bayesian' #2. Reuses the reported rolling-origin design
(validation.rolling_origin_assignments: within each community, choice sets in
time order; folds train on the first 50/65/80% and test on the next 15%),
fold-specific training scalers, unseen-level integration (one normal draw per
unseen account/participant level and posterior draw) and participant-clustered
standard errors. Models per fold:

``M1``        reported M1 (fixed AI and G2-by-AI terms + account intercepts +
              participant AI slopes);
``M1_noAI``   M1 without the two fixed AI terms, same random-effect structure
              (account intercepts centred within community and author type,
              participant AI slopes centred within community);
``M0``        controls only (as reported);
``M0_AI``     M0 plus the two fixed AI terms, no random effects.

Comparisons: M1 - M1_noAI and M0_AI - M0 isolate the fixed AI terms with the
random-effect structure held constant; M1 - M0 reproduces the reported
comparison. ``--reuse-models DIR`` loads existing M0/M1 fold posteriors
(``{model}_forward_fold{k}.nc``, private host) after checking their training
data hashes; otherwise every model is fitted. Seeds follow the reported scheme
(config.seed + 300 + 10*fold + model index; new models use indexes 5 and 6).
No topic slope, as in the reported M1.

    python -m scripts.diagnostics.review_choice_models_isolated_cv \
        --config analysis.public-rq1.yaml --frame MODEL_FRAME.parquet --out RUN_DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.diagnostics.review_choice_models_common import limit_threads

limit_threads(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    RunLog, build_network_from_config, check_cohort, fit_spec, guard_output, load_analysis_config, mcmc_check,
    peak_rss_gib, save_json, sha256,
)

RUN_ID = 'review_choice_models_isolated_cv'
MODEL_INDEX = {'M0': 0, 'M1': 1, 'M1_noAI': 5, 'M0_AI': 6}
COMPARISONS = (('M1', 'M1_noAI'), ('M0_AI', 'M0'), ('M1', 'M0'), ('M1_noAI', 'M0'))


def specs():
    from scripts.publication_analysis.model import MODEL_SPECS, ModelSpec
    return {
        'M0': MODEL_SPECS['M0'],
        'M1': MODEL_SPECS['M1'],
        'M1_noAI': ModelSpec('M1_noAI', 'M1 random effects without fixed AI terms', ('opportunity_content',),
                             True, True, True),
        'M0_AI': ModelSpec('M0_AI', 'M0 plus fixed AI and G2-by-AI terms', ('opportunity_content', 'focal'),
                           False, False, False),
    }


def fold_data(frame, fold, assignments, model='M1'):
    """Training and test data with fold-training scalers, as in run_forward_validation_models.

    M0 uses its own term list (as reported); every other model uses the M1 terms,
    so the two fixed AI columns are present even when a model does not use them.
    """
    from scripts.publication_analysis.model import prepare_model_data
    from scripts.publication_analysis.terms import (apply_model_term_manifest, build_model_frame,
                                                    included_terms_for_model)
    split = assignments[assignments.fold.eq(fold)]
    train_ids = set(split.loc[split.role.eq('train'), 'choice_set_id'])
    test_ids = set(split.loc[split.role.eq('test'), 'choice_set_id'])
    _, manifest = build_model_frame(frame[frame.choice_set_id.isin(train_ids)], model_name='M1')
    transformed = apply_model_term_manifest(frame, manifest)
    terms = tuple(t for t in included_terms_for_model('M0' if model == 'M0' else 'M1', manifest)
                  if t != 'candidate_target_is_ai')
    train = prepare_model_data(transformed[transformed.choice_set_id.isin(train_ids)], terms,
                               ai_term='candidate_target_is_ai')
    test = prepare_model_data(transformed[transformed.choice_set_id.isin(test_ids)], terms,
                              ai_term='candidate_target_is_ai')
    return train, test


def load_reused(directory, model, fold, train):
    import arviz as az
    from scripts.publication_analysis.pipeline import _prepared_data_hash
    path = Path(directory) / f'{model}_forward_fold{fold}.nc'
    signature = Path(str(path) + '.fit.json')
    if not path.exists() or not signature.exists():
        return None
    expected = json.loads(signature.read_text())['request']['data_sha256']
    if _prepared_data_hash(train) != expected:
        raise RuntimeError(f'{path} was fitted on different training data')
    return az.from_netcdf(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frame', type=Path, help='RQ1 model_frame.parquet from prepare-choice; '
                                                    'rebuilt from --config when omitted')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', default='M1,M1_noAI,M0,M0_AI')
    parser.add_argument('--folds', default='1,2,3')
    parser.add_argument('--reuse-models', type=Path)
    parser.add_argument('--draws', type=int)
    parser.add_argument('--tune', type=int)
    parser.add_argument('--chains', type=int)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    guard_output(args.out)
    log = RunLog(args.out / 'run.log')
    start = time.monotonic()
    config = load_analysis_config(args.config, RUN_ID)
    log(f'{RUN_ID} models={args.models} folds={args.folds} reuse={args.reuse_models} '
        f'script_sha256={sha256(__file__)}')
    from scripts.publication_analysis.pipeline import (_heldout_choice_predictions,
                                                       _participant_clustered_total_se)
    from scripts.publication_analysis.validation import rolling_origin_assignments
    if args.frame:
        frame = pd.read_parquet(args.frame)
    else:
        from scripts.publication_analysis.terms import build_model_frame
        _, network = build_network_from_config(config, log)
        frame, _ = build_model_frame(network, model_name='M1')
    check_cohort(frame)
    frame = frame.copy()
    frame['primary_root_topic'] = np.nan  # reported M1 has no topic slope
    assignments = rolling_origin_assignments(frame)
    models = args.models.split(',')
    model_specs = specs()
    sampling = config.sampling
    draws, tune, chains = (args.draws or sampling.draws, args.tune or sampling.tune, args.chains or sampling.chains)
    predictions, diagnostics = [], []
    for fold in [int(f) for f in args.folds.split(',')]:
        for name in models:
            train, test = fold_data(frame, fold, assignments, name)
            if name == models[0]:
                log(f'fold {fold}: train {train.n_choices} replies, test {test.n_choices} replies')
            seed_offset = 300 + 10 * fold + MODEL_INDEX[name]
            idata = load_reused(args.reuse_models, name, fold, train) if args.reuse_models and name in ('M0', 'M1') \
                else None
            source = 'reused_reported_posterior' if idata is not None else 'fitted'
            if idata is None:
                fit_start = time.monotonic()
                idata = fit_spec(train, config, model_specs[name], seed=config.seed + seed_offset, draws=draws,
                                 tune=tune, chains=chains, cores=args.cores)
                check = mcmc_check(idata)
                if not check['passed'] and args.draws is None:
                    # Reported policy: one extension with doubled warm-up and draws.
                    log(f'fold {fold} {name}: checks failed {check}; extending')
                    idata = fit_spec(train, config, model_specs[name], seed=config.seed + seed_offset,
                                     draws=2 * draws, tune=2 * tune, chains=chains, cores=args.cores)
                    check = mcmc_check(idata)
                check['fit_seconds'] = time.monotonic() - fit_start
                idata.to_netcdf(args.out / f'{name}_fold{fold}.nc')
            else:
                check = mcmc_check(idata)
            diagnostics.append(dict(fold=fold, model=name, source=source, **check))
            log(f'fold {fold} {name}: {source} {check}')
            scored = _heldout_choice_predictions(idata, train, test, integrate_unseen_hierarchical=True,
                                                 random_seed=config.seed + 30_000 + 100 * fold + MODEL_INDEX[name],
                                                 include_population_mean_sensitivity=True)
            scored.insert(0, 'model', name)
            scored.insert(0, 'fold', fold)
            predictions.append(scored)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions[['fold', 'model', 'choice_set_id', 'group', 'actor_label', 'log_score',
                 'log_score_population_mean_sensitivity', 'predicted_ai_probability', 'chosen_is_ai']].to_parquet(
        args.out / 'predictions.parquet', index=False)
    pd.DataFrame(diagnostics).to_csv(args.out / 'diagnostics.csv', index=False)
    rows = []
    for reference, comparator in COMPARISONS:
        if reference not in models or comparator not in models:
            continue
        for fold in [*sorted(predictions.fold.unique()), 'ALL']:
            part = predictions if fold == 'ALL' else predictions[predictions.fold.eq(fold)]
            a = part[part.model.eq(reference)]
            b = part[part.model.eq(comparator)]
            paired = a.merge(b, on=['fold', 'choice_set_id', 'group', 'actor_label', 'chosen_is_ai'],
                             suffixes=('_ref', '_cmp'), validate='one_to_one')
            for policy, column in (('unseen_levels_integrated', 'log_score'),
                                   ('unseen_levels_at_zero', 'log_score_population_mean_sensitivity')):
                delta = paired[column + '_ref'] - paired[column + '_cmp']
                se, clusters = _participant_clustered_total_se(delta, paired.actor_label)
                rows.append(dict(fold=str(fold), reference_model=reference, comparator_model=comparator,
                                 unseen_policy=policy, test_choice_sets=len(paired),
                                 elpd_difference=float(delta.sum()), elpd_difference_se=se,
                                 participant_clusters=clusters))
    pd.DataFrame(rows).to_csv(args.out / 'comparisons.csv', index=False)
    save_json(args.out / 'exit_status.json', dict(exit_code=0, elapsed_seconds=time.monotonic() - start,
                                                  peak_rss_gib=peak_rss_gib()))
    log(f'complete; elapsed {time.monotonic() - start:.1f} s; peak RSS {peak_rss_gib():.2f} GiB')


if __name__ == '__main__':
    main()
