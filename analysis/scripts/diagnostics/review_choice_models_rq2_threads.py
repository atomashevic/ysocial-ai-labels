"""RQ2 sensitivity: prior contact from earlier threads and a direct-child indicator.

Review item 'Bayesian' #3.2. Refits the RQ2 directional-contact model with the
reported cohort (2,143 replies, 17,489 candidates), M1 controls, priors,
hierarchy (account intercepts by community and type, participant AI slopes,
topic AI slopes) and the general / AI-specific / G2-by-AI history structure.
Variants:

``earlier_threads``
    each of the four history measures (log count and 24-hour recency, both
    directions) is rebuilt from strictly prior replies in *other* root threads
    only; replies in the current root thread are ignored.
``direct_child``
    the reported histories plus ``direct_child`` (candidate is a direct reply
    to the focal participant's own earlier contribution), its AI-specific and
    G2-by-AI terms.
``earlier_threads_direct_child``
    both changes.
``split_direct_child``
    current-thread and other-thread histories as separate terms (as in the
    unadmitted run rq2_cross_thread_20260915) plus the direct-child terms.

Projections follow rq2.standardized_estimands: one eligible AI account at a
time moves from no prior reply to one reply 24 valid-clock hours earlier (in
the history type named by the variant), all its candidates jointly, averaged
equally over the 3,573 choice-by-AI-account opportunities with a competing
account; also equal-participant weighting. Associations, not causal effects.

Inputs: either the public-input layout (``--config analysis.public-rq2.yaml``,
which must include the thread-topic panel) or the private configuration on
the private host. Optional ``--reference-frame`` (an admitted rq2_model_frame.parquet)
checks the rebuilt histories.

    python -m scripts.diagnostics.review_choice_models_rq2_threads \
        --config analysis.public-rq2.yaml --variant earlier_threads --out RUN_DIR
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from scripts.diagnostics.review_choice_models_common import limit_threads

limit_threads(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, account_changes, account_layout, beta_rows, build_network_from_config,
    check_cohort, compare_reference_frame, fit_with_extension, guard_output, load_analysis_config,
    peak_rss_gib, save_json, sha256, summarize, weighted_mean,
)

RUN_ID = 'review_choice_models_rq2_threads'
VARIANTS = ('earlier_threads', 'direct_child', 'earlier_threads_direct_child', 'split_direct_child')
BASES = {
    'outgoing': ('z_log_prior_contact_count', 'z_recent_prior_contact'),
    'incoming': ('z_log_reverse_contact_count', 'z_recent_reverse_contact'),
}
DIRECTION_LABEL = {'outgoing': 'sent_participant_to_ai', 'incoming': 'received_ai_to_participant'}


def histories(frame, config, log):
    """Attach current-/other-thread histories and the direct-child flag."""
    from scripts.diagnostics.rq2_contact import partition_history
    from scripts.publication_analysis.data import build_event_history, load_contents
    events = build_event_history(config)
    out = partition_history(frame, events, load_contents(config.processed_dir))
    for direction, raw in (('outgoing', 'prior_human_to_author_count'), ('incoming', 'prior_author_to_human_count')):
        total = out[f'within_{direction}_count'] + out[f'cross_{direction}_count']
        if not total.eq(out[raw]).all():
            raise RuntimeError(f'Split {direction} counts do not reproduce the strict histories')
    log('split histories reproduce the strict prior-contact counts')
    return out


def design(frame, variant):
    """Return the modelled frame, the RQ2-family term list and history scales."""
    from scripts.publication_analysis.model import prepare_model_data
    from scripts.publication_analysis.terms import build_model_frame, included_terms_for_model
    modeled, manifest = build_model_frame(frame, model_name='RQ2')
    m1_terms = list(included_terms_for_model('M1', manifest))
    ai = modeled.candidate_target_is_ai.astype(float)
    g2 = modeled.group.astype(str).eq('G2').astype(float)
    terms, scales = list(m1_terms), {}
    scopes = {'earlier_threads': ('cross',), 'earlier_threads_direct_child': ('cross',),
              'split_direct_child': ('within', 'cross'), 'direct_child': (None,)}[variant]
    for scope in scopes:
        for direction, bases in BASES.items():
            if scope is None:
                count = modeled[{'outgoing': 'prior_human_to_author_count',
                                 'incoming': 'prior_author_to_human_count'}[direction]]
                hours = modeled[{'outgoing': 'prior_human_to_author_time_since_hours',
                                 'incoming': 'prior_author_to_human_time_since_hours'}[direction]]
            else:
                count = modeled[f'{scope}_{direction}_count']
                hours = modeled[f'{scope}_{direction}_hours']
            raws = (np.log1p(count.astype(float)), pd.Series(np.where(count.gt(0), np.exp(-hours / 24.0), 0.0),
                                                              index=modeled.index))
            for base, raw in zip(bases, raws):
                name = base if scope is None else base.replace('z_', f'z_{scope}_', 1)
                mean, sd = float(raw.mean()), float(raw.std(ddof=0))
                if not sd > 0:
                    raise ValueError(f'No variation in {name}')
                scales[name] = {'mean': mean, 'sd': sd}
                modeled[name] = (raw - mean) / sd
                modeled['candidate_ai_x_' + name] = ai * modeled[name]
                modeled['g2_x_candidate_ai_x_' + name] = g2 * ai * modeled[name]
                terms += [name, 'candidate_ai_x_' + name, 'g2_x_candidate_ai_x_' + name]
    if 'direct_child' in variant:
        modeled['direct_child'] = modeled['direct_reply'].astype(float)
        modeled['candidate_ai_x_direct_child'] = ai * modeled.direct_child
        modeled['g2_x_candidate_ai_x_direct_child'] = g2 * ai * modeled.direct_child
        terms += ['direct_child', 'candidate_ai_x_direct_child', 'g2_x_candidate_ai_x_direct_child']
    data = prepare_model_data(modeled, [t for t in terms if t != 'candidate_target_is_ai'],
                              ai_term='candidate_target_is_ai')
    return modeled, data, scales


def scenarios(data, scales, variant):
    """Zero-to-one (24 h) values for the variant's contact scope, per arm."""
    scope = {'earlier_threads': 'cross', 'earlier_threads_direct_child': 'cross',
             'split_direct_child': 'cross', 'direct_child': None}[variant]
    scopes = ['cross', 'within'] if variant == 'split_direct_child' else [scope]
    result = {}
    for sc in scopes:
        for direction, bases in BASES.items():
            for group in ('G1', 'G2'):
                values = {}
                for base, high in zip(bases, (math.log(2.0), math.exp(-1.0))):
                    name = base if sc is None else base.replace('z_', f'z_{sc}_', 1)
                    pair = ((0.0 - scales[name]['mean']) / scales[name]['sd'],
                            (high - scales[name]['mean']) / scales[name]['sd'])
                    values[name] = pair
                    values['candidate_ai_x_' + name] = pair
                    values['g2_x_candidate_ai_x_' + name] = pair if group == 'G2' else (0.0, 0.0)
                result[(sc or 'all_threads', direction, group)] = {data.terms.index(k): v for k, v in values.items()}
    return result


def project(idata, data, scales, variant, chunk, log):
    from scripts.publication_analysis.model import _common_arm_frames
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    g1, g2 = _common_arm_frames(data)
    arms = {'G1': ragged.rows(g1), 'G2': ragged.rows(g2)}
    layout = account_layout(data, ragged, 'AI')
    scen = scenarios(data, scales, variant)
    store = {}
    for draws in posterior.chunks(chunk):
        random = ragged.random_part(posterior, draws)
        for group, rows in arms.items():
            utility = ragged.utility(posterior, draws, rows, random)
            for (scope, direction, g), values in scen.items():
                if g != group:
                    continue
                idx = list(values)
                changes = account_changes(utility, posterior.beta[draws], rows, layout, idx,
                                          [values[i][0] for i in idx], [values[i][1] for i in idx])
                for weighting, key in (('equal_opportunity', None), ('equal_participant', 'actor')):
                    store.setdefault((scope, direction, group, weighting), np.empty(posterior.n))[draws] = \
                        weighted_mean(changes, None if key is None else layout[key])
    rows = []
    for (scope, direction, group, weighting), values in store.items():
        common = dict(variant=variant, contact_scope=scope, direction=DIRECTION_LABEL[direction],
                      weighting=weighting, opportunities=len(layout['choice']),
                      replies=len(np.unique(layout['choice'])))
        rows.append(summarize(values, group=group, **common))
        if group == 'G2':
            rows.append(summarize(values - store[(scope, direction, 'G1', weighting)], group='G2-G1', **common))
    log(f'projected {len(layout["choice"])} AI-account opportunities from {len(np.unique(layout["choice"]))} replies')
    return pd.DataFrame(rows)


def support(modeled):
    units = modeled.groupby(['group', 'choice_set_id', 'candidate_author_id']).agg(
        ai=('candidate_target_is_ai', 'first'), direct=('direct_reply', 'max'),
        **{c: (c, 'first') for c in modeled.columns if c.endswith('_count') and c.startswith(('within_', 'cross_'))})
    units = units.reset_index()
    competing = units.groupby('choice_set_id').candidate_author_id.transform('size').gt(1)
    units = units[competing & units.ai.astype(int).eq(1)]
    rows = []
    for group, part in units.groupby('group'):
        row = dict(group=group, ai_account_opportunities=len(part), direct_child=int(part.direct.sum()))
        for column in [c for c in part.columns if c.endswith('_count')]:
            row[f'{column}_positive'] = int(part[column].gt(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-frame', type=Path, help='admitted rq2_model_frame.parquet (optional check)')
    parser.add_argument('--draws', type=int)
    parser.add_argument('--tune', type=int)
    parser.add_argument('--chains', type=int)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    guard_output(args.out)
    log = RunLog(args.out / 'run.log')
    start = time.monotonic()
    config = load_analysis_config(args.config, RUN_ID)
    log(f'{RUN_ID} variant={args.variant} config={args.config} script_sha256={sha256(__file__)}')
    _, network = build_network_from_config(config, log)
    check_cohort(network)
    split = histories(network, config, log)
    modeled, data, scales = design(split, args.variant)
    if data.topic_labels == ('UNTAGGED',):
        log('WARNING: no topic panel; the reported RQ2 model has a topic-varying AI slope (45 levels)')
    log(f'{data.n_choices} replies, {int(data.mask.sum())} candidates, {len(data.terms)} terms, '
        f'{len(data.topic_labels)} topic levels')
    if args.reference_frame:
        reference = pd.read_parquet(args.reference_frame)
        diff = compare_reference_frame(modeled, reference, ['z_log_candidate_age', 'z_log_candidate_depth',
                                                            'prior_human_to_author_count',
                                                            'prior_author_to_human_count'])
        log(f'max abs difference from reference frame: {diff}')
    support(modeled).to_csv(args.out / 'support.csv', index=False)
    save_json(args.out / 'scales.json', scales)
    sampling = config.sampling
    meta = dict(run_id=RUN_ID, variant=args.variant, terms=list(data.terms), replies=data.n_choices,
                candidates=int(data.mask.sum()), topics=len(data.topic_labels), config=str(args.config),
                processed_dir=str(config.processed_dir), script_sha256=sha256(__file__),
                draws=args.draws or sampling.draws, tune=args.tune or sampling.tune,
                chains=args.chains or sampling.chains, target_accept=sampling.target_accept, seed=config.seed + 900)
    save_json(args.out / 'run.json', meta)
    if args.prepare_only:
        log(f'prepared only; elapsed {time.monotonic() - start:.1f} s')
        return
    from scripts.publication_analysis.model import MODEL_SPECS
    log(f'sampling {meta["chains"]} chains x ({meta["tune"]} warm-up + {meta["draws"]} draws), NUTS, '
        f'cores={min(args.cores, meta["chains"])}')
    idata, checks = fit_with_extension(data, config, MODEL_SPECS['RQ2'], seed=meta['seed'], draws=meta['draws'],
                                       tune=meta['tune'], chains=meta['chains'], cores=args.cores, log=log,
                                       extend=args.draws is None)
    idata.to_netcdf(args.out / 'posterior.nc')
    save_json(args.out / 'checks.json', checks)
    log(f'sampler checks: {checks}')
    pd.DataFrame(beta_rows(idata, args.variant)).to_csv(args.out / 'parameters.csv', index=False)
    estimands = project(idata, data, scales, args.variant, args.chunk, log)
    estimands['diagnostics_passed'] = checks['passed']
    estimands.to_csv(args.out / 'estimands.csv', index=False)
    save_json(args.out / 'exit_status.json', dict(exit_code=0, elapsed_seconds=time.monotonic() - start,
                                                  peak_rss_gib=peak_rss_gib(), diagnostics_passed=checks['passed']))
    log(f'complete; elapsed {time.monotonic() - start:.1f} s; peak RSS {peak_rss_gib():.2f} GiB')


if __name__ == '__main__':
    main()
