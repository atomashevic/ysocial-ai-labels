#!/usr/bin/env python3
"""Exploratory within/cross-thread contact analysis; core sources stay immutable.

Admission uses the existing authenticated implementation with a runtime-only
worker cap. The supplemental model keeps M1 controls and random effects and
uses fully interacted directional, thread-specific histories and direct replies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

CORE_RUN = 'rq2_m2_directional_contact_release_20260903'
RUN = 'rq2_contact_20260907'
SCOPES = ('within', 'cross')
DIRECTIONS = ('outgoing', 'incoming')
PREFIXES = ('', 'ai_x_', 'g2_x_', 'g2_ai_x_')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def config():
    from scripts.publication_analysis.config import load_config
    return replace(load_config(Path('analysis.yaml'), profile='release', run_id=CORE_RUN),
                   analysis_stage='rq2_m2')


def partition_history(frame, events, contents):
    """Use strict timestamp ordering and the canonical compressed valid-day clock."""
    import numpy as np
    import pandas as pd
    from scripts.publication_analysis.data import _id_key

    content = {}
    for row in contents.itertuples(index=False):
        key = (str(row.group), _id_key(row.content_id))
        value = (_id_key(row.author_id), str(row.author_type), _id_key(row.comment_to), _id_key(row.root_id))
        if key in content and content[key] != value:
            raise ValueError('Conflicting authoritative content records')
        content[key] = value
    if frame[['group', 'actor_id', 'candidate_author_id', 'root_id']].isna().any().any():
        raise ValueError('Missing source identifiers')
    out = frame.copy().reset_index(drop=True)
    rows = {}
    for group, focal in out.groupby('group', sort=False):
        history = events[(events.group.astype(str) == str(group)) &
                         (events.event_type == 'comments_replies')].sort_values('event_timestamp')
        history = list(history.itertuples(index=False))
        pointer = 0
        state = defaultdict(lambda: defaultdict(list))
        for _, choices in focal.sort_values('event_timestamp').groupby('choice_set_id', sort=False):
            now = pd.Timestamp(choices.event_timestamp.iloc[0])
            clock = float(choices.event_valid_clock_seconds.iloc[0])
            while pointer < len(history) and pd.Timestamp(history[pointer].event_timestamp) < now:
                event = history[pointer]
                a, b, root = map(_id_key, (event.actor_id, event.target_author_id, event.root_id))
                if a is None or b is None or root is None or not np.isfinite(event.valid_clock_seconds):
                    raise ValueError('Missing contact history source information')
                if a != b:
                    state[(a, b)][root].append(float(event.valid_clock_seconds))
                pointer += 1
            for row in choices.itertuples():
                a, b, root = map(_id_key, (row.actor_id, row.candidate_author_id, row.root_id))
                key = (str(group), _id_key(row.candidate_content_id))
                if key not in content:
                    raise ValueError('Unresolved candidate content')
                author, author_type, parent, candidate_root = content[key]
                if author != b or author_type != str(row.candidate_author_type) or candidate_root != root:
                    raise ValueError('Candidate mapping differs from authoritative content')
                direct = 0
                if key[1] != root:
                    parent_key = (str(group), parent)
                    if parent is None or parent_key not in content:
                        raise ValueError('Unresolved candidate parent link')
                    direct = int(content[parent_key][0] == a)
                values = {'direct_reply': direct}
                for direction, edge in (('outgoing', (a, b)), ('incoming', (b, a))):
                    by_thread = state[edge]
                    for scope in SCOPES:
                        prior = [v for r, times in by_thread.items()
                                 if (r == root) == (scope == 'within') for v in times]
                        values[f'{scope}_{direction}_count'] = len(prior)
                        age = (clock - max(prior)) / 3600 if prior else np.nan
                        if prior and (not np.isfinite(age) or age < 0):
                            raise ValueError('Invalid valid-clock contact age')
                        values[f'{scope}_{direction}_hours'] = age
                rows[row.Index] = values
    extra = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    if not extra.index.equals(out.index):
        raise ValueError('History enrichment lost candidate rows')
    out = pd.concat([out, extra], axis=1)
    for direction, original in (('outgoing', 'human_to_author'), ('incoming', 'author_to_human')):
        counts = sum(out[f'{s}_{direction}_count'] for s in SCOPES)
        age = out[[f'{s}_{direction}_hours' for s in SCOPES]].min(axis=1)
        if not np.array_equal(counts, out[f'prior_{original}_count']):
            raise ValueError(f'{direction} partition does not reproduce original counts')
        if not np.allclose(age, out[f'prior_{original}_time_since_hours'], equal_nan=True):
            raise ValueError(f'{direction} recency does not reproduce original history')
    return out


def design(frame, manifest, scalers=None):
    import numpy as np
    from scripts.publication_analysis.terms import included_terms_for_model
    from scripts.publication_analysis.model import prepare_model_data

    out = frame.copy()
    terms = list(included_terms_for_model('M1', manifest))
    scales = {} if scalers is None else dict(scalers)
    ai = out.candidate_target_is_ai.astype(float).to_numpy()
    g2 = out.group.eq('G2').astype(float).to_numpy()
    multipliers = (np.ones(len(out)), ai, g2, ai * g2)
    bases = {'direct_reply': out.direct_reply.to_numpy(dtype=float)}
    for scope in SCOPES:
        for direction in DIRECTIONS:
            bases[f'{scope}_{direction}_logcount'] = np.log1p(out[f'{scope}_{direction}_count'])
            bases[f'{scope}_{direction}_recency'] = np.exp(-out[f'{scope}_{direction}_hours'] / 24).fillna(0)
    for name, raw in bases.items():
        if name not in scales:
            scales[name] = {'mean': 0.0 if name == 'direct_reply' else float(np.mean(raw)),
                            'sd': 1.0 if name == 'direct_reply' else float(np.std(raw, ddof=0))}
        sd = scales[name]['sd']
        # A constant predictor has no identified association; fail instead of
        # publishing a projection determined solely by a prior.
        if sd <= 0 or not np.isfinite(sd):
            raise ValueError(f'No variation in {name}')
        z = (raw - scales[name]['mean']) / sd
        for prefix, mult in zip(PREFIXES, multipliers):
            term = prefix + name
            out[term] = z * mult
            terms.append(term)
    data = prepare_model_data(out, terms, ai_term='candidate_target_is_ai')
    # Explicit families retain the original focal-vs-adjustment prior policy.
    data = replace(data, term_families=tuple('focal' if t in
                   ('candidate_target_is_ai', 'g2_x_candidate_ai') else 'opportunity_content'
                   for t in data.terms))
    return out, data, scales


def prepare(root):
    import pandas as pd
    from scripts.diagnostics.rq2_sensitivity import authenticate_core_run
    from scripts.publication_analysis.data import build_event_history, load_contents
    from scripts.publication_analysis.io import sha256_file
    c = config()
    core = authenticate_core_run(c)
    frame = partition_history(core['model_frame'], build_event_history(c), load_contents(c.processed_dir))
    frame, data, scales = design(frame, core['term_manifest'])
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / 'model_frame.parquet', index=False)
    save_json(root / 'design.json', {'scalers': scales, 'terms': data.terms,
              'implementation_sha256': fingerprint(),
              'core_bundle_sha256': sha256_file(c.outputs_dir / 'fit_result_bundle.json'),
              'frame_sha256': sha256_file(root / 'model_frame.parquet'),
              'choice_sets': data.n_choices, 'candidate_rows': int(data.mask.sum()),
              'unresolved_parent_links': 0, 'history_reconstruction': 'passed'})
    print(f'Prepared {data.n_choices} complete choice sets, {data.mask.sum()} candidate rows', flush=True)


def load_design(root):
    import pandas as pd
    from scripts.publication_analysis.io import sha256_file
    from scripts.diagnostics.rq2_sensitivity import authenticate_core_run
    c = config()
    core = authenticate_core_run(c)
    metadata = json.loads((root / 'design.json').read_text())
    expected = {'implementation_sha256': fingerprint(),
                'core_bundle_sha256': sha256_file(c.outputs_dir / 'fit_result_bundle.json'),
                'frame_sha256': sha256_file(root / 'model_frame.parquet')}
    if any(metadata[k] != v for k, v in expected.items()):
        raise ValueError('Exploratory checkpoint is incompatible with current sources')
    frame, data, scales = design(pd.read_parquet(root / 'model_frame.parquet'),
                                core['term_manifest'], metadata['scalers'])
    return c, frame, data, scales


def request(c, data, multiplier, length):
    from scripts.publication_analysis.pipeline import _posterior_request_signature
    result = _posterior_request_signature(c, data, model='RQ2_contact',
                role=f'exploratory_contact_prior_{multiplier}', draws=length * c.sampling.draws,
                tune=length * c.sampling.tune, chains=4, random_seed=c.seed + 170)
    result['contact_implementation_sha256'] = fingerprint()
    return result


def fit(root, multiplier):
    from scripts.diagnostics.rq2_sensitivity import doubled_prior_config
    from scripts.diagnostics.rq2_admission import compact_inference_data
    from scripts.publication_analysis.model import build_primary_pymc_model, fit_model
    from scripts.publication_analysis.pipeline import _load_compatible_posterior, _write_posterior_atomic
    from scripts.publication_analysis.validation import summarize_mcmc
    c, _, data, _ = load_design(root)
    if multiplier == 2:
        c = doubled_prior_config(c)
    path = root / f'contact_prior_{multiplier}.nc'
    req = request(c, data, multiplier, 1)
    if path.exists() != path.with_suffix('.nc.fit.json').exists():
        raise ValueError('Incomplete posterior checkpoint requires inspection')
    idata = _load_compatible_posterior(path, draws=c.sampling.draws, chains=4, request_signature=req)
    if path.exists() and idata is None:
        raise ValueError('Existing exploratory posterior failed authentication')
    for length in (1, 2):
        if idata is None:
            model = build_primary_pymc_model(data, c, model_spec='M1')
            idata = fit_model(model, c, draws=length * c.sampling.draws,
                       tune=length * c.sampling.tune, chains=4, cores=2,
                       random_seed=c.seed + 170, progressbar=False)
            idata = compact_inference_data(idata)
            _write_posterior_atomic(idata, path, request(c, data, multiplier, length))
        diag = summarize_mcmc(idata)
        save_json(root / f'diagnostics_prior_{multiplier}.json', diag)
        if diag['passed']:
            break
        saved = json.loads(path.with_suffix('.nc.fit.json').read_text())['request']
        if saved['draws'] > c.sampling.draws or length == 2:
            break
        idata.close()
        idata = None
    # Keep failed diagnostics and fitted posterior for the final report as well.
    print(json.dumps(diag), flush=True)


def account_change(utility, target, low_shift, high_shift):
    """Shift every contribution of one account jointly; competitors stay fixed."""
    from scipy.special import expit, logsumexp
    if not target.any() or target.all():
        raise ValueError('Projection needs an account and a competing receiver')
    odds = logsumexp(utility[:, target], axis=1) - logsumexp(utility[:, ~target], axis=1)
    return expit(odds + high_shift) - expit(odds + low_shift)


def projection_units(frame, data, restricted):
    """Paired support requires both types after applying account restrictions."""
    import numpy as np
    units = []
    groups = {str(k): v for k, v in frame.groupby('choice_set_id', sort=False)}
    for i, choice in enumerate(data.choice_ids):
        part = groups[choice]
        receivers = data.receiver_index[i, data.mask[i]]
        accepted = []
        for receiver in np.unique(receivers):
            positions = receivers == receiver
            account = part.iloc[np.flatnonzero(positions)]
            if restricted and (account.direct_reply.any() or
               account.within_outgoing_count.any() or account.within_incoming_count.any()):
                continue
            accepted.append((i, receiver, int(data.receiver_is_ai[receiver]), positions,
                             account.iloc[0]))
        if {u[2] for u in accepted} == {0, 1}:
            units.extend(accepted)
    return units


def project(root, multiplier):
    import arviz as az
    import numpy as np
    import pandas as pd
    from scripts.publication_analysis.model import _posterior_utilities
    from scripts.publication_analysis.pipeline import _load_compatible_posterior
    from scripts.diagnostics.rq2_sensitivity import doubled_prior_config
    c, frame, data, scales = load_design(root)
    if multiplier == 2:
        c = doubled_prior_config(c)
    path = root / f'contact_prior_{multiplier}.nc'
    idata = _load_compatible_posterior(path, draws=c.sampling.draws, chains=4,
                                     request_signature=request(c, data, multiplier, 1))
    if idata is None:
        raise ValueError('Missing or incompatible exploratory posterior')
    terms = list(map(str, idata.posterior.beta.term.values))
    nchain, ndraw = idata.posterior.sizes['chain'], idata.posterior.sizes['draw']
    sampler = json.loads((root / f'diagnostics_prior_{multiplier}.json').read_text())
    rows, support_rows = [], []
    for restricted in (False, True):
        units = projection_units(frame, data, restricted)
        subset = 'no_within_contact_or_direct_reply' if restricted else 'mixed_author_events'
        for scope in (('cross',) if restricted else SCOPES):
            for direction in DIRECTIONS:
                for group in ('G1', 'G2'):
                    chosen = [u for u in units if data.choice_groups[u[0]] == group]
                    event_ids = sorted({u[0] for u in chosen})
                    event_lookup = {v: k for k, v in enumerate(event_ids)}
                    counts = np.zeros((len(event_ids), 2), int)
                    for i, _, ai, _, _ in chosen:
                        counts[event_lookup[i], ai] += 1
                    draws = np.empty((nchain, ndraw, 2))
                    for ai in (0, 1):
                        typed = [u for u in chosen if u[2] == ai]
                        support_rows.append({'subset': subset, 'scope': scope, 'direction': direction,
                          'group': group, 'account_type': ('H', 'AI')[ai], 'events': len(event_ids),
                          'account_opportunities': len(typed),
                          'unique_accounts': len({u[1] for u in typed}),
                          'observed_contact_opportunities': sum(u[4][f'{scope}_{direction}_count'] > 0 for u in typed),
                          'run_id': RUN})
                    if chosen:
                        for chain in range(nchain):
                            for start in range(0, ndraw, 16):
                                stop = min(start + 16, ndraw)
                                part = idata.sel(chain=[chain], draw=slice(
                                    idata.posterior.draw.values[start], idata.posterior.draw.values[stop - 1]))
                                utility = _posterior_utilities(part, data)
                                beta = part.posterior.beta.values.reshape(stop - start, len(terms))
                                sums = np.zeros((stop - start, len(event_ids), 2))
                                for i, _, ai, positions, row in chosen:
                                    shifts = []
                                    for count, recency in ((0.0, 0.0), (np.log(2), np.exp(-1))):
                                        delta = np.zeros(len(terms))
                                        for metric, raw in (('logcount', count), ('recency', recency)):
                                            base = f'{scope}_{direction}_{metric}'
                                            z = (raw - scales[base]['mean']) / scales[base]['sd']
                                            for prefix, mult in zip(PREFIXES, (1, ai, int(group == 'G2'), ai * int(group == 'G2'))):
                                                term = prefix + base
                                                delta[terms.index(term)] = z * mult - row[term]
                                        shifts.append(beta @ delta)
                                    change = account_change(utility[:, i, data.mask[i]], positions, *shifts)
                                    sums[:, event_lookup[i], ai] += change
                                draws[chain, start:stop] = (sums / counts[None, :, :]).mean(axis=1)
                    for kind in ('H', 'AI', 'AI_minus_H'):
                        row = {'subset': subset, 'scope': scope, 'direction': direction,
                               'group': group, 'account_type': kind, 'events': len(event_ids),
                               'prior_multiplier': multiplier, 'run_id': RUN,
                               'status': 'unavailable' if not chosen else 'estimated'}
                        if chosen:
                            values = draws[:, :, 1] - draws[:, :, 0] if kind == 'AI_minus_H' else draws[:, :, int(kind == 'AI')]
                            diagnostic = az.summary(az.from_dict(posterior={'change': values}), kind='diagnostics', round_to=8).iloc[0]
                            row.update(mean=float(values.mean()), ci_low=float(np.quantile(values, .025)),
                              ci_high=float(np.quantile(values, .975)), probability_positive=float((values > 0).mean()),
                              rhat=float(diagnostic.r_hat), ess_bulk=float(diagnostic.ess_bulk), ess_tail=float(diagnostic.ess_tail))
                            row['converged'] = bool(sampler['passed'] and row['rhat'] < 1.01 and
                                                   row['ess_bulk'] > 400 and row['ess_tail'] > 400)
                        rows.append(row)
    pd.DataFrame(rows).to_csv(root / f'projections_prior_{multiplier}.csv', index=False)
    pd.DataFrame(support_rows).to_csv(root / 'support.csv', index=False)
    idata.close()


def report(root):
    import pandas as pd
    import shutil
    for multiplier in (1, 2):
        project(root, multiplier)
    primary = pd.read_csv(root / 'projections_prior_1.csv')
    doubled = pd.read_csv(root / 'projections_prior_2.csv')
    keys = ['subset', 'scope', 'direction', 'group', 'account_type']
    comparison = primary.merge(doubled, on=keys, suffixes=('_primary', '_doubled'), validate='one_to_one')
    comparison['mean_shift'] = comparison.mean_doubled - comparison.mean_primary
    comparison.to_csv(root / 'prior_sensitivity.csv', index=False)
    diagnostics = []
    for multiplier in (1, 2):
        diagnostics.append({'prior_multiplier': multiplier, 'run_id': RUN,
                            **json.loads((root / f'diagnostics_prior_{multiplier}.json').read_text())})
    pd.DataFrame(diagnostics).to_csv(root / 'diagnostics.csv', index=False)
    # Exploratory results remain explicitly labelled and carry diagnostics,
    # including failed or sparse cases. Original tables require admission.
    for src, name in (('projections_prior_1.csv', 'estimands'), ('support.csv', 'support'),
                      ('prior_sensitivity.csv', 'prior_sensitivity'), ('diagnostics.csv', 'diagnostics')):
        shutil.copyfile(root / src, Path('results/tables') / f'rq2_contact_exploratory_{name}.csv')
    c = config()
    admission = pd.read_parquet(c.outputs_dir / 'admission/tables/rq2_admission_readiness.parquet')
    passed = bool(admission.passed.all())
    if passed:
        pd.read_parquet(c.tables_dir / 'rq2_primary_estimands.parquet').to_csv(
            'results/tables/rq2_primary_estimands.csv', index=False)
        for stem in ('rq2_admission_readiness', 'rq2_admission_forward_comparisons',
                     'rq2_admission_exact_refit_estimands'):
            pd.read_parquet(c.outputs_dir / f'admission/tables/{stem}.parquet').to_csv(
                f'results/tables/{stem}.csv', index=False)
    lines = [f'Exploratory RQ2 contact analysis: {RUN}',
             f'Original RQ2 admission: {"passed" if passed else "failed"}.',
             'Complete original choice sets are retained. Projections change every contribution from one account jointly.',
             'Both author types use the same eligible events, account averages within type, and equal event weights.',
             'The restricted cross-thread projection excludes accounts with any within-thread contact or direct reply.',
             'All associations are observational; community comparisons are ecological. These data do not establish exposure, notifications, memory, or causal learning.',
             '', primary.to_string(index=False)]
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')
    metadata = json.loads((root / 'design.json').read_text())
    notes = Path('notes/study.md')
    marker = f'### Exploratory RQ2 contact analysis ({RUN})'
    if marker not in notes.read_text():
        with notes.open('a') as stream:
            stream.write(f'\n{marker}\n\nRun `{RUN}` retains {metadata["choice_sets"]:,} observed human reply choice sets and {metadata["candidate_rows"]:,} candidate rows from `{CORE_RUN}`. Strictly prior directional histories partition into within-thread and cross-thread contact; reconstructed counts and recencies match the original inputs. Candidate direct-parent links are resolved from authoritative contents. The model retains M1 controls and hierarchical effects, with full account-type and community interactions for the new history and direct-reply terms.\n\nProjections compare no contact with one reply 24 valid-clock hours earlier. All contributions of a focal account change jointly. Paired human/AI comparisons use events containing eligible accounts of both types, average accounts within type, and weight events equally. The restricted cross-thread projection requires zero within-thread contact in either direction and no candidate directly replying to the participant; both types must remain eligible. Cell denominators and observed-contact support are in `results/tables/rq2_contact_exploratory_support.csv`. Estimates, diagnostics and doubled-prior sensitivity are in the other `rq2_contact_exploratory_*.csv` tables. Failed convergence is retained explicitly. Associations are exploratory; community comparisons are ecological and do not identify exposure, notifications, memory or causal learning. Original RQ2 admission: {"passed" if passed else "failed"}.\n')
    save_json(root / 'completion.json', {'status': 'complete', 'core_admission_passed': passed,
              'parameter_diagnostics_passed': all(d['passed'] for d in diagnostics),
              'report': str(root / 'report.txt')})


def admission(stage, local_temp):
    import scripts.publication_analysis.model as model
    import scripts.publication_analysis.pipeline as pipeline
    from scripts.diagnostics.rq2_admission import run_admission
    original = model.fit_model
    original_load = pipeline._load_compatible_posterior
    def limited(*args, **kwargs):
        kwargs['cores'] = 2
        print('Sampling with four chains, two workers, one numerical thread each', flush=True)
        return original(*args, **kwargs)
    def checked_load(path, **kwargs):
        result = original_load(path, **kwargs)
        if result is None and (path.exists() or path.with_suffix('.nc.fit.json').exists()):
            raise ValueError('Existing admission checkpoint failed authentication; stopping')
        return result
    model.fit_model = limited
    pipeline._load_compatible_posterior = checked_load
    result = run_admission(config_path=Path('analysis.yaml'), run_id=CORE_RUN,
                           stage=stage, local_temp=local_temp)
    print(json.dumps(result, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['forward', 'refits', 'finalize', 'prepare', 'fit', 'sensitivity', 'report'])
    parser.add_argument('--local-temp', type=Path, required=True)
    args = parser.parse_args()
    from scripts.diagnostics.rq2_admission import configure_runtime
    configure_runtime(args.local_temp)
    import os
    os.environ['XDG_CACHE_HOME'] = str(args.local_temp / 'cache')
    root = Path('runs') / RUN / 'outputs'
    if args.stage in ('forward', 'refits', 'finalize'):
        admission(args.stage, args.local_temp)
    elif args.stage == 'prepare':
        prepare(root)
    elif args.stage in ('fit', 'sensitivity'):
        fit(root, 1 if args.stage == 'fit' else 2)
    else:
        report(root)


if __name__ == '__main__':
    main()
