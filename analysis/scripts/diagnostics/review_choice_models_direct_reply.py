"""Direct-reply estimand from the split-history + direct-child RQ2 sensitivity fit.

Part of run review_choice_models_fits_20261015. Uses the posterior written by
``review_choice_models_rq2_threads --variant split_direct_child`` and rebuilds
its model data from the same configuration (public-input layout).

Estimand: among choice-by-AI-account opportunities with a competing account in
which the AI account has at least one candidate that directly replies to the
focal participant's own earlier contribution, the change in the account-level
selection probability (sum of its candidates' probabilities) when the
direct-reply indicator of those candidates is switched from 0 to 1 (the
direct_child, AI x direct_child and, under G2 coefficients, G2 x AI x
direct_child terms), with all histories and other covariates as observed.
Evaluated under each community's coefficients on the pooled opportunities and
averaged equally over opportunities (and, as a sensitivity, over participants).
Also reports observed selection rates of direct-reply and other AI accounts.

    python -m scripts.diagnostics.review_choice_models_direct_reply --config analysis.public-rq2.yaml \
        --fit-dir RUN_DIR --out OUT_DIR
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
from scipy.special import expit  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, account_layout, build_network_from_config, load_analysis_config, peak_rss_gib,
    sha256, summarize, weighted_mean,
)

RUN_ID = 'review_choice_models_fits_20261015'
TERMS = ('direct_child', 'candidate_ai_x_direct_child', 'g2_x_candidate_ai_x_direct_child')


def subset_layout(layout, keep):
    """Restrict an account layout to the opportunities flagged in ``keep``."""
    focal, other, fs, os_ = [], [], [], []
    f_ends = np.r_[layout['focal_starts'][1:], len(layout['focal'])]
    o_ends = np.r_[layout['other_starts'][1:], len(layout['other'])]
    for i in np.flatnonzero(keep):
        fs.append(len(focal))
        os_.append(len(other))
        focal.extend(layout['focal'][layout['focal_starts'][i]:f_ends[i]])
        other.extend(layout['other'][layout['other_starts'][i]:o_ends[i]])
    out = {k: v[keep] for k, v in layout.items() if k not in ('focal', 'other', 'focal_starts', 'other_starts')}
    out.update(focal=np.asarray(focal, dtype=np.intp), other=np.asarray(other, dtype=np.intp),
               focal_starts=np.asarray(fs, dtype=np.intp), other_starts=np.asarray(os_, dtype=np.intp))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument('--skip-reproduction', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.out / 'run.log')
    start = time.monotonic()
    import arviz as az
    from scripts.diagnostics.review_choice_models_rq2_threads import design, histories, project
    from scripts.publication_analysis.model import _common_arm_frames
    meta = json.loads((args.fit_dir / 'run.json').read_text())
    if meta['variant'] != 'split_direct_child':
        raise ValueError('Expected the split_direct_child fit')
    config = load_analysis_config(args.config, RUN_ID)
    log(f'{RUN_ID} direct-reply estimand; fit {args.fit_dir}; posterior sha256 {sha256(args.fit_dir / "posterior.nc")}; '
        f'script_sha256={sha256(__file__)}')
    _, network = build_network_from_config(config, log)
    split = histories(network, config, log)
    modeled, data, scales = design(split, meta['variant'])
    if list(data.terms) != meta['terms']:
        raise RuntimeError('Rebuilt terms differ from the fitted terms')
    saved = json.loads((args.fit_dir / 'scales.json').read_text())
    if any(abs(saved[k]['mean'] - scales[k]['mean']) > 1e-12 or abs(saved[k]['sd'] - scales[k]['sd']) > 1e-12
           for k in saved):
        raise RuntimeError('Rebuilt history scales differ from the fitted scales')
    idata = az.from_netcdf(args.fit_dir / 'posterior.nc')
    idata.posterior.load()
    if not args.skip_reproduction:
        mine = project(idata, data, scales, meta['variant'], args.chunk, log)
        stored = pd.read_csv(args.fit_dir / 'estimands.csv')
        keys = ['contact_scope', 'direction', 'group', 'weighting']
        merged = mine.merge(stored, on=keys, suffixes=('', '_stored'))
        difference = float((merged.posterior_median - merged.posterior_median_stored).abs().max())
        log(f'reproduced {len(merged)} stored estimands; max abs median difference {difference:.2e}')
        if difference > 1e-10:
            raise RuntimeError('Rebuilt data do not reproduce the stored estimands')
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    g1, g2 = _common_arm_frames(data)
    arms = {'G1': ragged.rows(g1), 'G2': ragged.rows(g2)}
    layout = account_layout(data, ragged, 'AI')
    direct_column = ragged.rows(data.x)[:, data.terms.index('direct_child')]
    ends = np.r_[layout['focal_starts'][1:], len(layout['focal'])]
    has_direct = np.array([direct_column[layout['focal'][s:e]].max() > 0
                           for s, e in zip(layout['focal_starts'], ends)])
    sub = subset_layout(layout, has_direct)
    log(f'{int(has_direct.sum())} of {len(has_direct)} AI-account opportunities contain a direct-reply candidate '
        f'(G1 {int((has_direct & (layout["group"] == "G1")).sum())}, '
        f'G2 {int((has_direct & (layout["group"] == "G2")).sum())})')
    indexes = [data.terms.index(t) for t in TERMS]
    store = {}
    for draws in posterior.chunks(args.chunk):
        random = ragged.random_part(posterior, draws)
        beta = posterior.beta[draws]
        for group, rows in arms.items():
            utility = ragged.utility(posterior, draws, rows, random)
            focal_u = utility[:, sub['focal']]
            removed = (rows[sub['focal']][:, indexes] @ beta[:, indexes].T).T
            log_other = np.logaddexp.reduceat(utility[:, sub['other']], sub['other_starts'], axis=1)
            observed = np.logaddexp.reduceat(focal_u, sub['focal_starts'], axis=1)
            switched_off = np.logaddexp.reduceat(focal_u - removed, sub['focal_starts'], axis=1)
            change = expit(observed - log_other) - expit(switched_off - log_other)
            for weighting, key in (('equal_opportunity', None), ('equal_participant', 'actor')):
                store.setdefault((group, weighting), np.empty(posterior.n))[draws] = weighted_mean(
                    change, None if key is None else sub[key])
    rows = []
    for (group, weighting), values in store.items():
        common = dict(estimand='AI_account_selection_probability_difference_direct_reply_1_vs_0',
                      weighting=weighting, opportunities=int(has_direct.sum()),
                      replies=int(len(np.unique(sub['choice']))), participants=int(len(np.unique(sub['actor']))))
        rows.append(summarize(values, group=group, **common))
        if group == 'G2':
            rows.append(summarize(values - store[('G1', weighting)], group='G2-G1', **common))
    pd.DataFrame(rows).to_csv(args.out / 'direct_reply_estimand.csv', index=False)
    observed_rows = []
    for group in ('G1', 'G2', 'ALL'):
        in_group = np.ones_like(has_direct) if group == 'ALL' else layout['group'] == group
        for label, mask in (('direct_reply_account', has_direct), ('other_ai_account', ~has_direct)):
            selected = layout['selected'][in_group & mask]
            observed_rows.append(dict(group=group, account=label, opportunities=int(selected.size),
                                      selected=int(selected.sum()), selection_rate=float(selected.mean())))
    pd.DataFrame(observed_rows).to_csv(args.out / 'direct_reply_observed.csv', index=False)
    log(f'complete; elapsed {time.monotonic() - start:.1f} s; peak RSS {peak_rss_gib():.2f} GiB')


if __name__ == '__main__':
    main()
