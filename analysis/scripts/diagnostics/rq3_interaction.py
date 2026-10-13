#!/usr/bin/env python3
"""Bayesian other-participant downvote interaction on the existing RQ3 panel.

Run with the project's Python environment and a 24-hour external timeout.
This exploratory extension retains the original reply-event denominator. It
leaves the authenticated publication_analysis package and earlier fits intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

CELLS = ('G1_AI', 'G1_H', 'G2_AI', 'G2_H')
OUTCOME = 'target_other_downvote_24h'


def boolean(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if values.isna().any():
        raise ValueError(f'Invalid boolean in {series.name}')
    return values.astype(bool)


def prepare_panel(panel: pd.DataFrame, first_target: bool = False) -> pd.DataFrame:
    frame = panel.loc[boolean(panel.in_rq1_analysis_set) & panel.reaction_placement.eq('round_start')].copy()
    if frame.empty or frame.event_id.duplicated().any():
        raise ValueError('Expected one row per eligible RQ1 reply')
    frame['y'] = boolean(frame[OUTCOME]).astype(int)
    frame['cell'] = frame.group + '_' + frame.target_author_type
    if set(frame.cell) != set(CELLS):
        raise ValueError('Expected the four community-by-authorship cells')
    # Numeric IDs are local to each community; never merge across communities.
    for name, column in (('participant', 'actor_id'), ('target', 'selected_target_content_id')):
        frame[name + '_key'] = frame.group + ':' + pd.to_numeric(frame[column], errors='raise').astype('int64').astype(str)
    if frame.groupby('target_key').cell.nunique().max() != 1:
        raise ValueError('Target authorship is inconsistent')
    frame['event_timestamp'] = pd.to_datetime(frame.event_timestamp, utc=True)
    frame = frame.sort_values(['event_timestamp', 'event_id'], kind='stable')
    if first_target:
        frame = frame.drop_duplicates('target_key', keep='first')
    return frame.reset_index(drop=True)


def contrasts(p: np.ndarray) -> dict[str, np.ndarray]:
    """Last axis is G1 AI, G1 H, G2 AI, G2 H; probabilities in [0, 1]."""
    ai = p[..., 0] - p[..., 2]
    human = p[..., 1] - p[..., 3]
    return {'AI_visible_minus_hidden': ai, 'H_visible_minus_hidden': human,
            'interaction_AI_gap_minus_H_gap': ai - human}


def cell_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        part = frame[frame.cell.eq(cell)]
        rows.append(dict(cell=cell, events=len(part), positives=int(part.y.sum()),
                         participants=part.participant_key.nunique(), targets=part.target_key.nunique(),
                         positive_targets=part.loc[part.y.eq(1), 'target_key'].nunique(),
                         observed_probability=float(part.y.mean())))
    return pd.DataFrame(rows)


def build_model(frame: pd.DataFrame, sampling: dict, multiplier: float):
    import pymc as pm
    participant, participants = pd.factorize(frame.participant_key, sort=True)
    target, targets = pd.factorize(frame.target_key, sort=True)
    visible = frame.group.eq('G1').astype(float).to_numpy()
    ai = frame.target_author_type.eq('AI').astype(float).to_numpy()
    coords = {'participant': np.arange(len(participants)), 'target': np.arange(len(targets)), 'cell': CELLS}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal('alpha', mu=-2, sigma=1.5 * multiplier)
        focal = sampling['prior_focal_sigma'] * multiplier
        scale = sampling['prior_scale_sigma'] * multiplier
        beta_visible = pm.Normal('beta_visible', 0, focal)
        beta_ai = pm.Normal('beta_ai', 0, focal)
        beta_interaction = pm.Normal('beta_interaction', 0, focal)
        sigma_participant = pm.HalfNormal('sigma_participant', scale)
        z_participant = pm.Normal('z_participant', 0, 1, dims='participant')
        eta = (alpha + beta_visible * visible + beta_ai * ai + beta_interaction * visible * ai
               + sigma_participant * z_participant[participant])
        if not frame.target_key.is_unique:
            sigma_target = pm.HalfNormal('sigma_target', scale)
            z_target = pm.Normal('z_target', 0, 1, dims='target')
            eta = eta + sigma_target * z_target[target]
        prob = pm.math.sigmoid(eta)
        pm.Bernoulli('outcome', p=prob, observed=frame.y.to_numpy())
        pm.Deterministic('cell_probability', pm.math.stack([
            prob[np.flatnonzero(frame.cell.eq(cell))].mean() for cell in CELLS]), dims='cell')
    return model


def predictive_checks(frame: pd.DataFrame, replicated: np.ndarray) -> pd.DataFrame:
    """Check cell totals and concentration of positive events on shared targets."""
    checks = []
    for cell in CELLS:
        mask = frame.cell.eq(cell).to_numpy()
        part = frame.loc[mask]
        target, keys = pd.factorize(part.target_key, sort=True)
        observed_by_target = np.bincount(target, weights=part.y, minlength=len(keys))
        predicted_by_target = np.zeros((len(keys), len(replicated)), dtype=np.int32)
        np.add.at(predicted_by_target, target, replicated[:, mask].T)
        observed_metrics = dict(positive_events=int(part.y.sum()),
                                positive_targets=int((observed_by_target > 0).sum()),
                                positive_pairs=int((observed_by_target * (observed_by_target - 1) / 2).sum()))
        predicted_metrics = dict(positive_events=predicted_by_target.sum(axis=0),
                                 positive_targets=(predicted_by_target > 0).sum(axis=0),
                                 positive_pairs=(predicted_by_target * (predicted_by_target - 1) / 2).sum(axis=0))
        for metric, values in predicted_metrics.items():
            checks.append(dict(cell=cell, metric=metric, observed=observed_metrics[metric],
                               predicted_mean=float(values.mean()), predicted_low=float(np.quantile(values, .025)),
                               predicted_high=float(np.quantile(values, .975))))
    return pd.DataFrame(checks)


def sampler_diagnostics(idata, destination: Path) -> dict:
    import arviz as az
    cells = idata.posterior.cell_probability
    idata.posterior['probability_interaction'] = ((cells.sel(cell='G1_AI') - cells.sel(cell='G2_AI'))
                                                - (cells.sel(cell='G1_H') - cells.sel(cell='G2_H')))
    summary = az.summary(idata, kind='diagnostics', round_to='none')
    summary.to_csv(destination / 'parameter_diagnostics.csv')
    diagnostics = dict(max_rhat=float(summary.r_hat.max()), min_bulk_ess=float(summary.ess_bulk.min()),
                       min_tail_ess=float(summary.ess_tail.min()),
                       divergences=int(idata.sample_stats.diverging.sum()),
                       min_bfmi=float(np.min(az.bfmi(idata))),
                       max_tree_depth=int(idata.sample_stats.tree_depth.max()))
    diagnostics['passed'] = bool(diagnostics['max_rhat'] < 1.01 and diagnostics['min_bulk_ess'] >= 400
                                  and diagnostics['min_tail_ess'] >= 400 and diagnostics['divergences'] == 0
                                  and diagnostics['min_bfmi'] > 0.3 and diagnostics['max_tree_depth'] < 10)
    return diagnostics


def fit(frame: pd.DataFrame, sampling: dict, multiplier: float, seed: int, destination: Path):
    import arviz as az
    import pymc as pm
    model = build_model(frame, sampling, multiplier)
    with model:
        prior = pm.sample_prior_predictive(draws=1000, random_seed=seed + 2, var_names=['cell_probability'])
        prior_p = prior.prior.cell_probability.values.reshape(-1, 4)
        pd.DataFrame([dict(cell=cell, mean=float(prior_p[:, i].mean()),
                           low=float(np.quantile(prior_p[:, i], .025)),
                           high=float(np.quantile(prior_p[:, i], .975)))
                      for i, cell in enumerate(CELLS)]).to_csv(destination / 'prior_predictive.csv', index=False)
        idata = pm.sample(draws=sampling['draws'], tune=sampling['tune'], chains=sampling['chains'],
                          cores=min(4, sampling['chains']), target_accept=sampling['target_accept'],
                          random_seed=seed, progressbar=False)
    idata.to_netcdf(destination / 'posterior.nc')
    diagnostics = sampler_diagnostics(idata, destination)
    p = idata.posterior.cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
    values = {cell: p[:, i] for i, cell in enumerate(CELLS)}
    values.update(contrasts(p))
    values['log_odds_interaction'] = idata.posterior.beta_interaction.values.ravel()
    rows = [dict(estimand=name, estimate=float(v.mean()), ci_low=float(np.quantile(v, .025)),
                 ci_high=float(np.quantile(v, .975)), posterior_probability_gt_zero=float((v > 0).mean()),
                 interval_type='equal_tail_95', unit='log_odds' if name == 'log_odds_interaction' else 'probability')
            for name, v in values.items()]
    # Independent posterior predictions conditional on the fitted random effects.
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=['outcome'], random_seed=seed + 1, progressbar=False)
    replicated = ppc.posterior_predictive.outcome.values.reshape(-1, len(frame))
    predictive_checks(frame, replicated).to_csv(destination / 'predictive_checks.csv', index=False)
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('runs/rq3_downvote_roundts_20260902/outputs/tables/rq3_vote_panel.csv'))
    parser.add_argument('--config', type=Path, default=Path('analysis.yaml'))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    import yaml
    import pymc as pm
    import arviz as az
    config = yaml.safe_load(args.config.read_text())['analysis']
    sampling = config['sampling']
    destination = Path('runs') / args.run_id
    destination.mkdir(exist_ok=False, parents=True)
    panel = pd.read_csv(args.panel)
    metadata = dict(run_id=args.run_id, panel=str(args.panel), panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sampling=sampling,
                    priors='intercept Normal(-2,1.5); all three focal coefficients Normal(0,1); varying SD HalfNormal(0.5); sensitivity doubles all SDs',
                    seed=config['seed'], pymc=pm.__version__, arviz=az.__version__,
                    outcome=OUTCOME, timing='reply hour included through 24 h; reaction at round start',
                    estimand='within observed cell posterior mean probability; reply-event weighted; fitted varying intercepts',
                    contrast='(G1_AI-G2_AI)-(G1_H-G2_H)',
                    boundary='exploratory ecological contrast; one community per condition')
    (destination / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    outputs = destination / 'outputs'
    outputs.mkdir()
    tables, diagnostic_rows, support, checks = [], [], [], []
    for index, (name, multiplier, first) in enumerate((('primary', 1., False), ('doubled_prior', 2., False), ('first_target', 1., True))):
        frame = prepare_panel(panel, first)
        output = outputs / name
        output.mkdir()
        cell_summary(frame).assign(variant=name).to_csv(output / 'cell_summary.csv', index=False)
        support.append(cell_summary(frame).assign(variant=name))
        print(f'Fitting {name}: {len(frame)} replies, {frame.y.sum()} positives', flush=True)
        rows, diagnostics = fit(frame, sampling, multiplier, config['seed'] + 3000 + index, output)
        table = pd.DataFrame(rows).assign(variant=name, run_id=args.run_id, events=len(frame),
                                          participants=frame.participant_key.nunique(), targets=frame.target_key.nunique())
        table.to_csv(output / 'estimands.csv', index=False)
        tables.append(table)
        checks.append(pd.read_csv(output / 'predictive_checks.csv').assign(variant=name))
        diagnostic_rows.append(dict(variant=name, **diagnostics))
        print(json.dumps(diagnostic_rows[-1]), flush=True)
    pd.concat(tables, ignore_index=True).to_csv(outputs / 'rq3_interaction_estimands.csv', index=False)
    pd.concat(support, ignore_index=True).to_csv(outputs / 'rq3_interaction_cells.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(outputs / 'rq3_interaction_diagnostics.csv', index=False)
    pd.concat(checks, ignore_index=True).to_csv(outputs / 'rq3_interaction_predictive_checks.csv', index=False)
    if not all(row['passed'] for row in diagnostic_rows):
        raise RuntimeError('Sampler diagnostics failed; do not admit estimates')


if __name__ == '__main__':
    main()
