"""A14: criticism model adjusted for parent disagreement (CHB pre-submission review).

Named run ``chb_prereview_20261110``, item ``a14_criticism_parent_adjusted``.

The reported Sol direct-parent criticism model (``annotation_models.build_model``;
2,547 replies; crossed participant, thread and addressed-account intercepts;
reported priors and sampler settings) plus the A15 parent-disagreement indicator
``d`` (1 when GPT-5.6 Sol coded the parent as disagreeing with, criticizing or
antagonistic toward the replying participant; structurally 0 for root parents):

``baseline``        no covariate (reproduces the reported 13.3-point interaction)
``pooled``          + gamma * d
``by_authorship``   + gamma * d + delta * d * AI_parent

The indicator coefficients use Normal(0, prior_adjustment_sigma) (0.5), the
reported adjustment scale (as in review_choice_models_conduct.py).

Estimands follow the reported standardization: for cell (community g, target
type t) the reference rows are all rows of community g (their participant and
thread intercepts), the target-type shift is applied and addressed-account
effects are integrated by 20-point Gauss-Hermite. The covariate enters as a
mixture over d with weights given by the scheme:

``observed``        share of d among the rows of cell (g, t) (observed covariate
                    values; the analogue of the reported estimand)
``pooled_by_type``  share of d among rows of target type t pooled over both
                    communities (removes the between-community difference in
                    parent disagreement: the decomposition estimand)
``fixed_0``/``fixed_1``  d set to 0 or 1 in every cell (conditional on stance)

Interaction = (G1 AI - G1 H) - (G2 AI - G2 H): visible minus hidden difference in
the AI-minus-participant criticism gap, probability scale. Reported values are
posterior means with equal-tailed 95% intervals.

Mediator caveat: ``d`` is measured after assignment and can lie on the path from
the label condition to criticism (participants' earlier contributions shape how
the parent answers them). The adjusted contrasts are therefore descriptive
standardizations over ``d``, not total or direct causal effects of the label
condition; ``d`` is an unvalidated GPT-5.6 Sol label.

Sensitivity fits (fix-up of 2026-11-11):

``pooled_prior1``          ``pooled`` with gamma ~ Normal(0, 1.0)
``by_authorship_prior1``   ``by_authorship`` with gamma, delta ~ Normal(0, 1.0)
``pooled_nonroot``         ``pooled`` + rho * nonroot (1 when the parent is a comment),
                           rho ~ Normal(0, 0.5); d = 0 then separates root parents
                           (nonroot = 0) from non-root parents coded NO (nonroot = 1)

For ``pooled_nonroot`` the schemes mix over the three parent states (root;
non-root coded NO; non-root coded YES): ``observed`` and ``pooled_by_type`` use the
state shares of the cell or of the target type; ``fixed_0`` sets d = 0 and keeps
the cell's observed root/non-root share; ``fixed_0_root`` and ``fixed_0_nonroot``
set every parent to root or to non-root coded NO; ``fixed_1`` sets every parent to
non-root coded YES.

    flock LOCK python -m scripts.diagnostics.chb_prereview_a14_criticism_parent_adjusted fit --variant pooled
    python -m scripts.diagnostics.chb_prereview_a14_criticism_parent_adjusted collect
    python -m scripts.diagnostics.chb_prereview_a14_criticism_parent_adjusted collect-sensitivity
"""
from __future__ import annotations

import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.review_reception import (  # noqa: E402
    CELLS, LABELS, ROOT, criticism_frame, sampling_config, summarize,
)

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a14_criticism_parent_adjusted'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
A15_LABELS = ROOT / 'runs' / RUN_ID / 'a15_parent_disagreement/outputs/reply_parent_disagreement.csv'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
VARIANTS = ('baseline', 'pooled', 'by_authorship')
SENSITIVITY_VARIANTS = ('pooled_prior1', 'by_authorship_prior1', 'pooled_nonroot')
SCHEMES = ('observed', 'pooled_by_type', 'fixed_0', 'fixed_1')
NONROOT_SCHEMES = SCHEMES + ('fixed_0_root', 'fixed_0_nonroot')
STATES = ((0, 0), (0, 1), (1, 1))  # (d, nonroot): root parent; non-root coded NO; non-root coded YES
WIDE_ADJUSTMENT_SIGMA = 1.0
REPORTED = dict(interaction=0.1328, low=0.0507, high=0.2135)
MEDIATOR_CAVEAT = (
    'd (GPT-5.6 Sol parent-disagreement label, no human validation) is measured after assignment and can lie on '
    'the path from the label condition to criticism: participants\' earlier contributions and the condition can '
    'shape how the parent answers them. Adjusted contrasts are descriptive standardizations over d (observed, '
    'pooled_by_type, fixed d), not total or direct causal effects of the label condition, and they do not identify '
    'how much of the interaction d mediates.')


def log(message: str) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def adjusted_frame(labels_path: Path = A15_LABELS) -> pd.DataFrame:
    frame = criticism_frame()
    labels = pd.read_csv(labels_path)
    merged = frame.merge(labels[['group', 'content_id', 'parent_kind', 'parent_disagreement']],
                         on=['group', 'content_id'], how='left', validate='one_to_one')
    if merged.parent_disagreement.isna().any() or len(merged) != 2547:
        raise ValueError('Every cohort reply needs a parent-disagreement value')
    if merged.loc[merged.parent_kind.eq('root'), 'parent_disagreement'].ne(0).any():
        raise ValueError('Root parents must be structurally 0')
    merged['d'] = merged.parent_disagreement.astype(int)
    merged['nonroot'] = merged.parent_kind.ne('root').astype(int)
    return merged


def state_weights(frame: pd.DataFrame, scheme: str) -> dict[str, dict[tuple[int, int], float]]:
    """Weights over the (d, nonroot) parent states for each cell under a scheme (pooled_nonroot)."""
    state = frame.d.astype(int) * 2 + frame.nonroot.astype(int)
    if state.eq(2).any():
        raise ValueError('d = 1 requires a non-root parent')
    out = {}
    for cell in CELLS:
        group, typ = cell.split('_')
        in_cell = frame.cell.eq(cell)
        if scheme in ('observed', 'pooled_by_type'):
            part = state[in_cell] if scheme == 'observed' else state[frame.target_type.eq(typ)]
            out[cell] = {s: float(part.eq(s[0] * 2 + s[1]).mean()) for s in STATES}
        elif scheme == 'fixed_0':
            share_nonroot = float(frame.loc[in_cell, 'nonroot'].mean())
            out[cell] = {(0, 0): 1 - share_nonroot, (0, 1): share_nonroot, (1, 1): 0.0}
        elif scheme == 'fixed_0_root':
            out[cell] = {(0, 0): 1.0, (0, 1): 0.0, (1, 1): 0.0}
        elif scheme == 'fixed_0_nonroot':
            out[cell] = {(0, 0): 0.0, (0, 1): 1.0, (1, 1): 0.0}
        elif scheme == 'fixed_1':
            out[cell] = {(0, 0): 0.0, (0, 1): 0.0, (1, 1): 1.0}
        else:
            raise ValueError(scheme)
    return out


def as_state_weights(weights: dict[str, float]) -> dict[str, dict[tuple[int, int], float]]:
    """P(d = 1) weights of the models without the nonroot term, written as state weights (nonroot unused)."""
    return {cell: {(0, 0): 1 - share, (1, 1): share} for cell, share in weights.items()}


def scheme_weights(frame: pd.DataFrame, scheme: str) -> dict[str, float]:
    """P(d = 1) used for each cell under a scheme."""
    out = {}
    for cell in CELLS:
        group, typ = cell.split('_')
        if scheme == 'observed':
            out[cell] = float(frame.loc[frame.cell.eq(cell), 'd'].mean())
        elif scheme == 'pooled_by_type':
            out[cell] = float(frame.loc[frame.target_type.eq(typ), 'd'].mean())
        elif scheme in ('fixed_0', 'fixed_1'):
            out[cell] = float(scheme[-1])
        else:
            raise ValueError(scheme)
    return out


def build_model(frame: pd.DataFrame, sampling: dict, variant: str):
    """annotation_models.build_model (criticism) with the parent-disagreement terms."""
    import pymc as pm
    frame = frame.reset_index(drop=True)
    codes, coords = {}, {'observation': np.arange(len(frame))}
    for key in ('participant', 'context', 'target'):
        codes[key], levels = pd.factorize(frame[key], sort=True)
        coords[key] = np.arange(len(levels))
    visible = frame.group.eq('G1').to_numpy(dtype=float)
    ai = frame.target_type.eq('AI').to_numpy(dtype=float)
    d = frame.d.to_numpy(dtype=float)
    base_variant = variant.removesuffix('_prior1').removesuffix('_nonroot')
    indicator_sigma = WIDE_ADJUSTMENT_SIGMA if variant.endswith('_prior1') else sampling['prior_adjustment_sigma']
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal('alpha', mu=-2, sigma=1.5)
        b = {key: pm.Normal('beta_' + key, 0, sampling['prior_focal_sigma']) for key in ('visible', 'ai', 'interaction')}
        eta = alpha + b['visible'] * visible + b['ai'] * ai + b['interaction'] * visible * ai
        if base_variant in ('pooled', 'by_authorship'):
            eta = eta + pm.Normal('gamma_disagreement', 0, indicator_sigma) * d
        if base_variant == 'by_authorship':
            eta = eta + pm.Normal('delta_disagreement_ai', 0, indicator_sigma) * d * ai
        if variant.endswith('_nonroot'):
            nonroot = frame.nonroot.to_numpy(dtype=float)
            eta = eta + pm.Normal('rho_nonroot', 0, sampling['prior_adjustment_sigma']) * nonroot
        for key in codes:
            sigma = pm.HalfNormal('sigma_' + key, sampling['prior_scale_sigma'])
            z = pm.Normal('z_' + key, 0, 1, dims=key)
            eta = eta + sigma * z[codes[key]]
        pm.Bernoulli('outcome', p=pm.math.sigmoid(eta), observed=frame.y.to_numpy(), dims='observation')
    return model, codes


def draws_of(idata, name: str) -> np.ndarray:
    return idata.posterior[name].stack(sample=('chain', 'draw')).transpose('sample', ...).values


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def cell_probabilities(params: dict, frame: pd.DataFrame, codes: dict, weights_by_scheme: dict) -> dict:
    """Reference-row standardized cell probabilities for each scheme (draws x 4).

    Scheme weights are either P(d = 1) per cell (models without the nonroot term) or
    weights over the (d, nonroot) STATES per cell (``state_weights``).
    """
    a, bv, ba, bi = params['alpha'], params['beta_visible'], params['beta_ai'], params['beta_interaction']
    gamma = params.get('gamma_disagreement', np.zeros_like(a))
    delta = params.get('delta_disagreement_ai', np.zeros_like(a))
    rho = params.get('rho_nonroot', np.zeros_like(a))
    weights_by_scheme = {scheme: (as_state_weights(w) if not isinstance(next(iter(w.values())), dict) else w)
                         for scheme, w in weights_by_scheme.items()}
    up = params['sigma_participant'][:, None] * params['z_participant']
    uc = params['sigma_context'][:, None] * params['z_context']
    st = params['sigma_target']
    nodes, gh = np.polynomial.hermite.hermgauss(20)
    gh = gh / np.sqrt(np.pi)
    conditional = {}
    for cell in CELLS:
        group, typ = cell.split('_')
        ref = frame.group.eq(group).to_numpy()
        base = a[:, None] + bv[:, None] * float(group == 'G1') + up[:, codes['participant'][ref]] + uc[:, codes['context'][ref]]
        shift = (ba + bi * float(group == 'G1')) * float(typ == 'AI')
        for dv, nr in STATES:
            eta = base + (shift + dv * (gamma + delta * float(typ == 'AI')) + nr * rho)[:, None]
            integral = sum(wk * expit(eta + np.sqrt(2) * st[:, None] * xk) for xk, wk in zip(nodes, gh))
            conditional[(cell, (dv, nr))] = integral.mean(axis=1)
    out = {}
    for scheme, weights in weights_by_scheme.items():
        out[scheme] = np.stack([sum(w * conditional[(c, s)] for s, w in weights[c].items()) for c in CELLS], axis=1)
    return out


def contrasts(p: np.ndarray) -> dict[str, np.ndarray]:
    g1, g2 = p[:, 0] - p[:, 1], p[:, 2] - p[:, 3]
    values = {cell: p[:, i] for i, cell in enumerate(CELLS)}
    values.update(G1_AI_minus_H=g1, G2_AI_minus_H=g2, AI_visible_minus_hidden=p[:, 0] - p[:, 2],
                  H_visible_minus_hidden=p[:, 1] - p[:, 3], interaction_visible_minus_hidden_AI_gap=g1 - g2)
    return values


def diagnostics_of(idata) -> dict:
    import arviz as az
    summary = az.summary(idata, kind='diagnostics', round_to='none')
    out = dict(max_rhat=float(summary.r_hat.max()), min_bulk_ess=float(summary.ess_bulk.min()),
               min_tail_ess=float(summary.ess_tail.min()), divergences=int(idata.sample_stats.diverging.sum()),
               min_bfmi=float(np.min(az.bfmi(idata))), max_tree_depth=int(idata.sample_stats.tree_depth.max()))
    out['passed'] = bool(out['max_rhat'] < 1.01 and out['min_bulk_ess'] >= 400 and out['min_tail_ess'] >= 400
                         and out['divergences'] == 0 and out['min_bfmi'] > 0.3 and out['max_tree_depth'] < 10)
    return out


def fit(args) -> None:
    import pymc as pm
    sampling, base_seed = sampling_config()
    variant = args.variant
    tune = args.tune or sampling['tune']
    destination = OUT / 'fits' / (variant + ('' if tune == sampling['tune'] else f'_tune{tune}'))
    destination.mkdir(parents=True, exist_ok=False)
    frame = adjusted_frame()
    seed = base_seed + 7000 + 100 + 500 + (VARIANTS + SENSITIVITY_VARIANTS).index(variant) * 10
    log(f'fit {variant}: {len(frame)} replies, {int(frame.y.sum())} positives, d=1 in {int(frame.d.sum())}; seed {seed}; '
        f'draws {sampling["draws"]} tune {tune} chains 4 target_accept {sampling["target_accept"]}')
    model, codes = build_model(frame, sampling, variant)
    started = time.monotonic()
    with model:
        idata = pm.sample(draws=sampling['draws'], tune=tune, chains=4, cores=4,
                          target_accept=sampling['target_accept'], random_seed=seed, progressbar=False)
    elapsed = time.monotonic() - started
    checks = diagnostics_of(idata)
    checks.update(elapsed_seconds=elapsed, seed=seed, tune=tune, draws=sampling['draws'], chains=4,
                  target_accept=sampling['target_accept'])
    idata.to_netcdf(destination / 'posterior.nc')
    names = ['alpha', 'beta_visible', 'beta_ai', 'beta_interaction', 'sigma_participant', 'sigma_context', 'sigma_target',
             'z_participant', 'z_context'] + [n for n in ('gamma_disagreement', 'delta_disagreement_ai', 'rho_nonroot')
                                              if n in idata.posterior]
    params = {n: draws_of(idata, n) for n in names}
    if variant.endswith('_nonroot'):
        weights = {s: state_weights(frame, s) for s in NONROOT_SCHEMES}
    else:
        weights = {s: scheme_weights(frame, s) for s in SCHEMES}
    probabilities = cell_probabilities(params, frame, codes, weights)
    rows = []
    for scheme, p in probabilities.items():
        for name, values in contrasts(p).items():
            rows.append(dict(variant=variant, scheme=scheme, estimand=name, unit='probability', **summarize(values)))
    for n in ('beta_visible', 'beta_ai', 'beta_interaction', 'gamma_disagreement', 'delta_disagreement_ai',
              'rho_nonroot', 'sigma_participant', 'sigma_context', 'sigma_target'):
        if n in params:
            rows.append(dict(variant=variant, scheme='parameter', estimand=n, unit='log_odds' if not n.startswith('sigma')
                             else 'sd_log_odds', **summarize(params[n])))
    pd.DataFrame(rows).to_csv(destination / 'estimands.csv', index=False)
    np.savez_compressed(destination / 'interaction_draws.npz',
                        **{s: contrasts(p)['interaction_visible_minus_hidden_AI_gap'] for s, p in probabilities.items()})
    (destination / 'diagnostics.json').write_text(json.dumps(checks, indent=1) + '\n')
    serializable = {s: {c: ({f'd{k[0]}_nonroot{k[1]}': v for k, v in w.items()} if isinstance(w, dict) else w)
                        for c, w in cells.items()} for s, cells in weights.items()}
    (destination / 'scheme_weights.json').write_text(json.dumps(serializable, indent=1) + '\n')
    inter = [r for r in rows if r['estimand'] == 'interaction_visible_minus_hidden_AI_gap' and r['scheme'] == 'observed'][0]
    log(f'fit {variant}: done in {elapsed:.0f}s; diagnostics {json.dumps(checks)}; observed interaction '
        f'{inter["estimate"]:.4f} ({inter["ci_low"]:.4f}, {inter["ci_high"]:.4f})')
    if not checks['passed']:
        raise RuntimeError('Sampler diagnostics failed; refit with --tune 2000')


def passing_fit(variant: str) -> Path:
    """Latest passing fit folder of a variant (``variant`` or ``variant_tuneN``, not other variants)."""
    candidates = sorted(c for c in (OUT / 'fits').glob(variant + '*')
                        if c.name == variant or c.name.startswith(variant + '_tune'))
    passing = [c for c in candidates if json.loads((c / 'diagnostics.json').read_text())['passed']]
    if not passing:
        raise FileNotFoundError(f'No passing fit for {variant}')
    return passing[-1]


SENSITIVITY_DESCRIPTION = {
    'pooled': 'reference: gamma ~ Normal(0, 0.5)',
    'by_authorship': 'reference: gamma, delta ~ Normal(0, 0.5)',
    'pooled_prior1': 'pooled with gamma ~ Normal(0, 1.0)',
    'by_authorship_prior1': 'by_authorship with gamma, delta ~ Normal(0, 1.0)',
    'pooled_nonroot': 'pooled + rho * nonroot parent, rho ~ Normal(0, 0.5); d = 0 split into root and non-root coded NO',
}
FOCAL = 'interaction_visible_minus_hidden_AI_gap'


def collect_sensitivity(args) -> None:
    """Sensitivity table for the 2026-11-11 fix-up; adds ``fixups_20261111`` to run.json."""
    from scripts.diagnostics.chb_prereview_a15_parent_disagreement import sha256
    frame = adjusted_frame()
    fits = {variant: passing_fit(variant) for variant in ('pooled', 'by_authorship') + SENSITIVITY_VARIANTS}
    rows, diag = [], []
    for variant, folder in fits.items():
        table = pd.read_csv(folder / 'estimands.csv')
        keep = table.estimand.eq(FOCAL) | table.scheme.eq('parameter') & table.estimand.isin(
            ['beta_interaction', 'gamma_disagreement', 'delta_disagreement_ai', 'rho_nonroot'])
        rows.append(table.loc[keep].assign(role='reference' if variant in VARIANTS else 'sensitivity',
                                           specification=SENSITIVITY_DESCRIPTION[variant], fit=folder.name))
        draws = np.load(folder / 'interaction_draws.npz')
        for scheme in ('pooled_by_type', 'fixed_0'):
            rows.append(pd.DataFrame([dict(variant=variant, scheme=f'observed_minus_{scheme}', estimand=FOCAL,
                                           unit='probability', **summarize(draws['observed'] - draws[scheme]),
                                           role='reference' if variant in VARIANTS else 'sensitivity',
                                           specification=SENSITIVITY_DESCRIPTION[variant], fit=folder.name)]))
        if variant in SENSITIVITY_VARIANTS:
            d = json.loads((folder / 'diagnostics.json').read_text())
            diag.append(dict(variant=variant, fit=folder.name, **d))
    common = dict(run_id=RUN_ID, denominator_replies=len(frame), denominator_participants=frame.participant.nunique(),
                  positives=int(frame.y.sum()), parent_disagreement_replies_total=int(frame.d.sum()),
                  nonroot_parent_replies_total=int(frame.nonroot.sum()))
    sensitivity = pd.concat(rows, ignore_index=True).assign(**common)
    names = {n: TABLES / f'chb_prereview_a14_criticism_parent_adjusted_{n}.csv'
             for n in ('sensitivity', 'sensitivity_diagnostics')}
    sensitivity.to_csv(names['sensitivity'], index=False)
    pd.DataFrame(diag).assign(**common).to_csv(names['sensitivity_diagnostics'], index=False)
    focal = sensitivity.loc[sensitivity.estimand.eq(FOCAL) & sensitivity.scheme.isin(['observed', 'pooled_by_type',
                                                                                       'fixed_0', 'fixed_0_root',
                                                                                       'fixed_0_nonroot'])]
    record = json.loads((RUN / 'run.json').read_text())
    record['mediator_caveat'] = MEDIATOR_CAVEAT
    record['fixups_20261111'] = dict(
        changes=[
            'Headline values stated as posterior means with equal-tailed 95% intervals; by_authorship fixed_0 '
            'interaction is 11.8 points (posterior mean 0.11753).',
            'Summary leads with the pooled fit: pooled_by_type 13.4 (6.0 to 20.7), fixed_0 12.1 (5.4 to 19.0).',
            'Mediator caveat made explicit (mediator_caveat).',
            'Sensitivity fits: pooled_prior1 and by_authorship_prior1 (indicator priors Normal(0, 1.0) instead of '
            'Normal(0, 0.5)); pooled_nonroot (adds a non-root-parent indicator, so d = 0 separates root parents from '
            'non-root parents coded NO).',
            'collect() now selects only the named variant folder (or variant_tuneN), so the new sensitivity folders do '
            'not displace the reference fits; the existing tables were not regenerated.',
        ],
        mediator_caveat=MEDIATOR_CAVEAT,
        sign_convention='visible (G1) minus hidden (G2) AI-minus-participant criticism gap, probability scale; '
                        'posterior mean and equal-tailed 95% interval',
        command='env OMP_NUM_THREADS=1 python -m '
                'scripts.diagnostics.chb_prereview_a14_criticism_parent_adjusted fit --variant '
                '{pooled_prior1,by_authorship_prior1,pooled_nonroot}; ... collect-sensitivity',
        host=platform.node(),
        fits={v: str(f.relative_to(ROOT.parent)) for v, f in fits.items() if v in SENSITIVITY_VARIANTS},
        diagnostics=diag,
        focal_contrast={f'{r.variant}|{r.scheme}': [round(r.estimate, 6), round(r.ci_low, 6), round(r.ci_high, 6)]
                        for r in focal.itertuples()},
        new_outputs=[str(p.relative_to(ROOT.parent)) for p in names.values()],
        script_sha256=sha256(Path(__file__)),
        test_sha256=sha256(ROOT / 'tests/test_chb_prereview_a14_criticism_parent_adjusted.py'),
        collected=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
    )
    (RUN / 'run.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    log(f'collect-sensitivity: wrote {", ".join(p.name for p in names.values())} and run.json fixups_20261111')


def collect(args) -> None:
    import arviz
    import pymc
    from scripts.diagnostics.chb_prereview_a15_parent_disagreement import sha256
    frame = adjusted_frame()
    fits = {variant: passing_fit(variant) for variant in VARIANTS}
    rows, diag = [], []
    for variant, folder in fits.items():
        table = pd.read_csv(folder / 'estimands.csv')
        rows.append(table)
        d = json.loads((folder / 'diagnostics.json').read_text())
        diag.append(dict(variant=variant, fit=folder.name, **d))
    estimands = pd.concat(rows, ignore_index=True)
    base = np.load(fits['baseline'] / 'interaction_draws.npz')['observed']
    # Within-fit decomposition: observed-distribution interaction minus the interaction under each other
    # scheme, from the same posterior draws. observed - pooled_by_type is the part carried by the
    # between-community difference in parent disagreement. The baseline (independent fit) median is
    # reported for reference only.
    decomposition = []
    for variant in ('pooled', 'by_authorship'):
        adjusted = np.load(fits[variant] / 'interaction_draws.npz')
        for scheme in SCHEMES:
            if scheme == 'observed':
                continue
            decomposition.append(dict(variant=variant, comparison=f'observed_minus_{scheme}',
                                      baseline_fit_interaction_median=float(np.median(base)),
                                      **summarize(adjusted['observed'] - adjusted[scheme])))
    support = []
    for cell, part in frame.groupby('cell'):
        support.append(dict(cell=cell, replies=len(part), criticism_positive=int(part.y.sum()),
                            non_root_parent_replies=int(part.parent_kind.ne('root').sum()),
                            parent_disagreement_replies=int(part.d.sum()),
                            share_parent_disagreement=float(part.d.mean()),
                            criticism_rate_d1=float(part.loc[part.d.eq(1), 'y'].mean()) if part.d.any() else np.nan,
                            criticism_rate_d0=float(part.loc[part.d.eq(0), 'y'].mean())))
    common = dict(run_id=RUN_ID, denominator_replies=len(frame), denominator_participants=frame.participant.nunique(),
                  positives=int(frame.y.sum()), parent_disagreement_replies_total=int(frame.d.sum()))
    estimands.assign(**common).to_csv(TABLES / 'chb_prereview_a14_criticism_parent_adjusted_estimands.csv', index=False)
    pd.DataFrame(diag).assign(**common).to_csv(TABLES / 'chb_prereview_a14_criticism_parent_adjusted_diagnostics.csv',
                                               index=False)
    pd.DataFrame(support).assign(**common).to_csv(TABLES / 'chb_prereview_a14_criticism_parent_adjusted_support.csv',
                                                  index=False)
    pd.DataFrame(decomposition).assign(**common).to_csv(
        TABLES / 'chb_prereview_a14_criticism_parent_adjusted_decomposition.csv', index=False)
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='Reported criticism model plus the A15 parent-disagreement indicator (pooled and by target authorship).',
        host=platform.node(),
        command='python -m scripts.diagnostics.chb_prereview_a14_criticism_parent_adjusted '
                'fit --variant {baseline,pooled,by_authorship}; ... collect',
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        inputs={str(p.relative_to(ROOT.parent)): sha256(p) for p in (LABELS, A15_LABELS, ROOT / 'analysis.yaml',
                                                                ROOT.parent / 'data/08_analysis_ready/results_tables/rq3_sol_criticism_cells.csv')},
        sampling=sampling_config()[0], packages=dict(python=sys.version.split()[0], pymc=pymc.__version__,
                                                     arviz=arviz.__version__, numpy=np.__version__,
                                                     pandas=pd.__version__),
        fits={v: str(f.relative_to(ROOT.parent)) for v, f in fits.items()}, diagnostics=diag,
        start_end=dict(started=(RUN / 'run.log').read_text().split(' ', 1)[0], collected=dt.datetime.now().astimezone().isoformat(timespec='seconds')),
        reported=REPORTED,
        mediator_caveat=MEDIATOR_CAVEAT,
        tables=[f'runs/chb_prereview_20261110/tables/chb_prereview_a14_criticism_parent_adjusted_{n}.csv'
                for n in ('estimands', 'diagnostics', 'support', 'decomposition')],
    )
    previous = json.loads((RUN / 'run.json').read_text()) if (RUN / 'run.json').exists() else {}
    record.update({k: v for k, v in previous.items() if k.startswith('fixups_')})
    (RUN / 'run.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    log('collect: wrote tables and run.json')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--variant', choices=VARIANTS + SENSITIVITY_VARIANTS, required=True)
    f.add_argument('--tune', type=int)
    sub.add_parser('collect')
    sub.add_parser('collect-sensitivity')
    args = parser.parse_args(argv)
    {'fit': fit, 'collect': collect, 'collect-sensitivity': collect_sensitivity}[args.stage](args)


if __name__ == '__main__':
    main()
