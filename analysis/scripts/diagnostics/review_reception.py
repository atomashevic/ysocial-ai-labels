"""Review-response reception analyses (named run review_reception_20261015).

Stages, each run separately with the project's fitting environment:

  validation      sample-level validation metrics and population-weighted
                  operating characteristics of the Sol direct-parent criticism
                  label in the 2,547-reply cohort; misclassification priors
  fit-criticism   one criticism model variant (--variant)
  fit-downvote    one target-level downvote model variant (--variant)
  pv-correction   cell-specific predictive-value correction of the admitted
                  criticism posterior (no sampling)
  scale           relative-scale summaries from the admitted posteriors
  denominators    Table 2 denominator checks from admitted tables
  collect         assemble runs/review_reception_20261015/tables/review_reception_*.csv

Inputs are the public Zenodo 1.1.0 staging tree (criticism_labels.csv,
rq3_vote_panel.csv), reported tables (08_analysis_ready/results_tables), and aggregate-only extracts from
the private host written by review_reception_private.py into
runs/review_reception_20261015/outputs/private_aggregates/. No text or participant
identifiers are written. Criticism model specification, priors, sampling and
standardization follow scripts/diagnostics/annotation_models.py; downvote models
follow scripts/diagnostics/rq3_interaction.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'review_reception_20261015'
RUN = ROOT / 'runs' / RUN_ID
OUT = RUN / 'outputs'
TABLES = RUN / 'tables'
TREE = ROOT.parent / 'data'
LABELS = TREE / '07_annotations/llm_annotations/criticism_labels.csv'
VOTES = TREE / '08_analysis_ready/model_inputs/rq3_vote_panel.csv'
ADMITTED = TREE / '08_analysis_ready/results_tables'
PRIVATE = TREE / '08_analysis_ready/further_checks/a10_content_misclass'
CELLS = ('G1_AI', 'G1_H', 'G2_AI', 'G2_H')
STRATA = ('POS_SCP', 'POS_CONTENT', 'NEG', 'NEG_NOTPARENT')
DRAWS = 20000
CRITICISM_VARIANTS = ('reproduce', 'misclass_common', 'misclass_cell', 'content_parent', 'conduct_participation_parent')
DOWNVOTE_VARIANTS = ('first_target', 'first_target_author')


def log(message):
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def summarize(values, **extra):
    values = np.asarray(values, dtype=float)
    return dict(estimate=float(np.mean(values)), median=float(np.median(values)),
                ci_low=float(np.quantile(values, .025)), ci_high=float(np.quantile(values, .975)),
                posterior_probability_gt_zero=float(np.mean(values > 0)), **extra)


def sampling_config():
    import yaml
    config = yaml.safe_load((ROOT / 'analysis.yaml').read_text())['analysis']
    sampling = config['sampling']
    if (sampling['chains'], sampling['draws'], sampling['tune'], sampling['target_accept']) != (4, 1000, 1000, .95):
        raise ValueError('Reported sampler settings changed')
    return sampling, config['seed']


# --------------------------------------------------------------------------- data

def criticism_frame():
    """Public direct-parent criticism panel; must reproduce the admitted cells."""
    labels = pd.read_csv(LABELS)
    frame = labels.loc[labels.in_criticism_model].copy()
    frame['target_type'] = frame.parent_author_type
    frame['cell'] = frame.group + '_' + frame.target_type
    frame['participant'] = frame.group + ':' + frame.author_code
    frame['context'] = frame.group + ':' + frame.root_id.astype(str)
    frame['target'] = frame.group + ':' + frame.parent_author_code
    frame['item_id'] = frame.group + ':' + frame.content_id.astype(str)
    frame['y'] = frame.direct_parent_criticism.eq('YES').astype(int)
    frame['content_parent'] = frame.content_target.eq('DIRECT_PARENT').astype(int)
    frame['conduct_participation_parent'] = (frame.style_conduct_target.eq('DIRECT_PARENT')
                                             | frame.participation_target.eq('DIRECT_PARENT')).astype(int)
    scp = frame.style_conduct_criticism.eq('YES') | frame.participation_criticism.eq('YES')
    frame['stratum'] = np.select(
        [frame.y.eq(1) & scp, frame.y.eq(1), frame.criticism_label.eq('YES')],
        ['POS_SCP', 'POS_CONTENT', 'NEG_NOTPARENT'], 'NEG')
    admitted = pd.read_csv(ADMITTED / 'rq3_sol_criticism_cells.csv').set_index('cell')
    for cell in CELLS:
        part = frame.loc[frame.cell.eq(cell)]
        observed = (len(part), int(part.y.sum()), part.participant.nunique(), part.context.nunique(), part.target.nunique())
        expected = tuple(int(admitted.loc[cell, k]) for k in ('observations', 'positive', 'participants', 'contexts', 'target_accounts'))
        if observed != expected:
            raise ValueError(f'Public panel does not reproduce admitted cell {cell}: {observed} != {expected}')
    if len(frame) != 2547 or frame.y.sum() != 299:
        raise ValueError('Public criticism cohort must contain 2,547 replies and 299 positives')
    return frame.reset_index(drop=True)


def validation_counts():
    """Population N and validated n/x per Sol stratum and cell (model cohort only)."""
    frame = criticism_frame()
    population = frame.groupby(['stratum', 'group', 'target_type']).size().rename('N')
    strata = pd.read_csv(PRIVATE / 'validation_criticism_strata.csv')
    sample = strata.loc[strata.in_criticism_model.astype(str).str.lower().eq('true')].copy()
    sample['target_type'] = sample.parent_author_type
    scp = sample.sol_stratum.eq('STYLE_OR_PARTICIPATION')
    sample['stratum'] = np.select(
        [sample.direct_parent_criticism.eq('YES') & scp, sample.direct_parent_criticism.eq('YES'),
         sample.criticism_label.eq('YES')], ['POS_SCP', 'POS_CONTENT', 'NEG_NOTPARENT'], 'NEG')
    sample['yes'] = np.where(sample.human.eq('YES'), sample['items'], 0)
    counts = sample.groupby(['stratum', 'group', 'target_type']).agg(n=('items', 'sum'), x=('yes', 'sum'))
    table = pd.concat([population, counts], axis=1).fillna(0).astype(int).reset_index()
    if (table.n > table.N).any():
        raise ValueError('Validated items exceed population counts')
    census = table.loc[table.stratum.eq('POS_SCP')]
    if not census.n.eq(census.N).all():
        raise ValueError('Style/participation positives are expected to be a census')
    return table, strata


# --------------------------------------------------------------------------- validation

def kappa(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def metric_row(task, domain, level, tp, tn, fp, fn, **extra):
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if n else np.nan)
    return dict(task=task, domain=domain, level=level, items=n, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp + tn) / n, precision_yes=precision, recall_yes=recall, f1_yes=f1,
                specificity_no=tn / (tn + fp) if tn + fp else np.nan, cohen_kappa=kappa(tp, tn, fp, fn), **extra)


def sample_metrics(strata):
    admitted = pd.read_csv(ROOT / 'results/human_validation_20260915/llm_majority_metrics.csv').set_index('task')
    rows = []

    def confusion(part, sol='criticism_label'):
        yes_h, yes_s = part.human.eq('YES'), part[sol].eq('YES')
        return (int(part['items'][yes_s & yes_h].sum()), int(part['items'][~yes_s & ~yes_h].sum()),
                int(part['items'][yes_s & ~yes_h].sum()), int(part['items'][~yes_s & yes_h].sum()))

    s = strata.copy()
    overall = confusion(s)
    if overall != tuple(int(admitted.loc['criticism', k]) for k in ('tp', 'tn', 'fp', 'fn')):
        raise ValueError('Criticism validation strata do not reproduce admitted confusion counts')
    rows.append(metric_row('criticism', 'all', 'sample', *overall,
                           krippendorff_alpha=admitted.loc['criticism', 'krippendorff_alpha'],
                           source='results/human_validation_20260915/llm_majority_metrics.csv; Table S12'))
    for g in ('G1', 'G2'):
        rows.append(metric_row('criticism', g, 'sample_by_community', *confusion(s.loc[s.group.eq(g)])))
    cohort = s.loc[s.in_criticism_model.astype(str).str.lower().eq('true')]
    rows.append(metric_row('criticism', 'model_cohort', 'sample_model_cohort', *confusion(cohort)))
    for t in ('AI', 'H'):
        rows.append(metric_row('criticism', 'direct_parent_' + t, 'sample_model_cohort_by_parent_authorship',
                               *confusion(cohort.loc[cohort.parent_author_type.eq(t)])))
    for g in ('G1', 'G2'):
        for t in ('AI', 'H'):
            rows.append(metric_row('criticism', f'{g}_{t}', 'sample_model_cohort_by_cell',
                                   *confusion(cohort.loc[cohort.group.eq(g) & cohort.parent_author_type.eq(t)])))
    rows.append(metric_row('criticism', 'outside_model_cohort', 'sample_outside_cohort',
                           *confusion(s.loc[s.in_criticism_model.astype(str).str.lower().eq('false')])))
    for stratum in ('STYLE_OR_PARTICIPATION', 'CONTENT_ONLY', 'NEGATIVE'):
        part = s.loc[s.sol_stratum.eq(stratum)]
        rows.append(metric_row('criticism', 'sol_' + stratum.lower(), 'sample_by_sol_stratum', *confusion(part)))
    ident = pd.read_csv(PRIVATE / 'validation_ai_identity_strata.csv').rename(columns={'llm_label': 'sol'})
    io = confusion(ident, 'sol')
    if io != tuple(int(admitted.loc['ai_identity', k]) for k in ('tp', 'tn', 'fp', 'fn')):
        raise ValueError('AI-identity strata do not reproduce admitted confusion counts')
    rows.append(metric_row('ai_identity', 'all', 'sample', *io,
                           krippendorff_alpha=admitted.loc['ai_identity', 'krippendorff_alpha'],
                           source='results/human_validation_20260915/llm_majority_metrics.csv; Table S12'))
    for g in ('G1', 'G2'):
        rows.append(metric_row('ai_identity', g, 'sample_by_community', *confusion(ident.loc[ident.group.eq(g)], 'sol')))
    reason = np.where(ident.selection_reason.str.startswith('All final Sol YES'), 'all_sol_positives',
                      np.where(ident.selection_reason.str.startswith('Consistency review'), 'consistency_review_negatives',
                               'style_matched_negatives'))
    for value in ('all_sol_positives', 'consistency_review_negatives', 'style_matched_negatives'):
        rows.append(metric_row('ai_identity', value, 'sample_by_selection', *confusion(ident.loc[reason == value], 'sol')))
    return pd.DataFrame(rows)


def stratum_draws(table, keys, prior, rng, finite=True):
    """Return dict key -> draws of human-YES totals (finite) or probabilities (superpopulation)."""
    a, b = prior
    grouped = table.groupby(keys)[['N', 'n', 'x']].sum()
    out = {}
    for key, row in grouped.iterrows():
        if row.N == 0 or key[0] == 'NEG_NOTPARENT':
            continue  # non-parent Sol criticism is assumed correctly negative for this endpoint
        if row.n == 0 and row.N > 0:
            raise ValueError(f'Unvalidated stratum {key}')
        p = rng.beta(a + row.x, b + row.n - row.x, DRAWS)
        out[key] = row.x + rng.binomial(row.N - row.n, p) if finite else p * row.N
    return out, grouped


def characteristics(table, keys, prior, rng, finite=True):
    """Operating characteristics by domain.

    Design strata are Sol stratum x community. Domains defined by target
    authorship are post-stratified within design strata. An empty `keys`
    gives the whole cohort.
    """
    order = ['group', 'target_type']
    cols = ['stratum', *[k for k in order if k == 'group' or k in keys]]
    draws, grouped = stratum_draws(table, cols, prior, rng, finite)
    domain_of = lambda key: tuple(key[cols.index(k)] for k in keys) or ('all',)
    results = {}
    for domain in sorted({domain_of(k) for k in grouped.index}):
        selected = [k for k in grouped.index if domain_of(k) == domain]
        n_pos = sum(grouped.N[k] for k in selected if k[0] in ('POS_SCP', 'POS_CONTENT'))
        n_neg = sum(grouped.N[k] for k in selected if k[0] in ('NEG', 'NEG_NOTPARENT'))
        zero = np.zeros(DRAWS)
        tp = sum((draws.get(k, zero) for k in selected if k[0] in ('POS_SCP', 'POS_CONTENT')), zero)
        # Sol criticism of a non-parent target is assumed correctly negative for this endpoint.
        fn = sum((draws.get(k, zero) for k in selected if k[0] == 'NEG'), zero)
        fp, tn = n_pos - tp, n_neg - fn
        with np.errstate(divide='ignore', invalid='ignore'):
            results[domain] = dict(
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                ppv=tp / n_pos if n_pos else np.full(DRAWS, np.nan), npv=tn / n_neg,
                true_prevalence=(tp + fn) / (n_pos + n_neg), sol_prevalence=np.full(DRAWS, n_pos / (n_pos + n_neg)),
                _counts=dict(N=int(n_pos + n_neg), sol_positive=int(n_pos), sol_negative=int(n_neg),
                             validated=int(sum(grouped.n[k] for k in selected)),
                             validated_human_yes=int(sum(grouped.x[k] for k in selected))))
    return results


def cell_error_draws(table, prior, pool_negatives, rng, finite):
    """Per-cell draws of Sol-positive agreement (PPV) and Sol-negative human-YES rate.

    Positive strata are style/participation positives (census) and content-only
    positives, sampled within community and post-stratified by authorship.
    Negatives are post-stratified by authorship (cell-specific) or pooled within
    community (pool_negatives=True), the design stratum of the matched sample.
    finite=True returns finite-population counts (TP, FN) in the 2,547 cohort;
    otherwise superpopulation rates (PPV, false-omission rate).
    """
    a, b = prior
    out = {}
    negatives = table.loc[table.stratum.eq('NEG')].groupby('group')[['N', 'n', 'x']].sum()
    pooled = {g: rng.beta(a + r.x, b + r.n - r.x, DRAWS) for g, r in negatives.iterrows()}
    for cell in CELLS:
        g, t = cell.split('_')
        part = table.loc[table.group.eq(g) & table.target_type.eq(t)].set_index('stratum')
        get = lambda s, k: int(part.loc[s, k]) if s in part.index else 0
        tp = np.zeros(DRAWS)
        for stratum in ('POS_SCP', 'POS_CONTENT'):
            N, n, x = get(stratum, 'N'), get(stratum, 'n'), get(stratum, 'x')
            if N:
                rate = rng.beta(a + x, b + n - x, DRAWS)
                tp = tp + (x + rng.binomial(N - n, rate) if finite else rate * N)
        N, n, x = get('NEG', 'N'), get('NEG', 'n'), get('NEG', 'x')
        rate = pooled[g] if pool_negatives else rng.beta(a + x, b + n - x, DRAWS)
        fn = x + rng.binomial(N - n, rate) if finite else rate * N
        n_pos = get('POS_SCP', 'N') + get('POS_CONTENT', 'N')
        n_neg = N + get('NEG_NOTPARENT', 'N')
        out[cell] = dict(tp=tp, fn=fn, n_pos=n_pos, n_neg=n_neg, ppv=tp / n_pos, false_omission=fn / n_neg)
    return out


ERROR_SCENARIOS = (('uniform', (1., 1.), False), ('uniform', (1., 1.), True),
                   ('jeffreys', (.5, .5), False), ('jeffreys', (.5, .5), True))


def beta_moments(values):
    m, v = float(np.mean(values)), float(np.var(values))
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def run_validation(args):
    table, strata = validation_counts()
    metrics = sample_metrics(strata)
    rows = []
    rng = np.random.default_rng(20261015)
    for prior_name, prior in (('uniform', (1., 1.)), ('jeffreys', (.5, .5))):
        for level, keys in (('cohort', []), ('community', ['group']), ('parent_authorship', ['target_type']),
                            ('cell', ['group', 'target_type'])):
            result = characteristics(table, keys, prior, rng)
            for domain, values in result.items():
                counts = values.pop('_counts')
                for measure, draw in values.items():
                    rows.append(dict(level=level, domain='_'.join(map(str, domain)), measure=measure, prior=prior_name,
                                     uncertainty='finite_population_beta_binomial', **counts,
                                     **{k: v for k, v in summarize(draw).items() if k != 'posterior_probability_gt_zero'}))
            if level == 'cell' and prior_name == 'uniform':
                prevalence = {'_'.join(k): v['true_prevalence'] for k, v in result.items()}
    # Design-based corrected cell prevalences and contrasts (no model).
    corrected = []
    for prior_name, prior, pool in ERROR_SCENARIOS:
        errors = cell_error_draws(table, prior, pool, rng, finite=True)
        p = np.column_stack([(errors[c]['tp'] + errors[c]['fn']) / (errors[c]['n_pos'] + errors[c]['n_neg']) for c in CELLS])
        for row in contrast_rows(p, 'design_based'):
            if row['unit'] == 'log_odds':
                continue
            corrected.append(dict(level='design_based_contrast', domain='cohort', measure=row['estimand'], prior=prior_name,
                                  negatives='pooled_within_community' if pool else 'cell_specific',
                                  uncertainty='finite_population_beta_binomial', N=2547,
                                  **{k: row[k] for k in ('estimate', 'median', 'ci_low', 'ci_high', 'posterior_probability_gt_zero')},
                                  posterior_probability_gt_one=row.get('posterior_probability_gt_one', np.nan)))
    oc = pd.concat([pd.DataFrame(rows), pd.DataFrame(corrected)], ignore_index=True)
    # Superpopulation priors for the misclassification models.
    sp_rng = np.random.default_rng(20261016)
    draws, grouped = stratum_draws(table, ['stratum', 'group'], (1., 1.), sp_rng, finite=False)
    n_pos = grouped.N.loc[['POS_SCP', 'POS_CONTENT']].sum()
    n_neg = grouped.N.loc[['NEG', 'NEG_NOTPARENT']].sum()
    tp = sum(v for k, v in draws.items() if k[0] in ('POS_SCP', 'POS_CONTENT'))
    fn = sum(v for k, v in draws.items() if k[0] == 'NEG')
    se, sp = tp / (tp + fn), (n_neg - fn) / (n_neg - fn + n_pos - tp)
    priors = dict(common=dict(sensitivity=beta_moments(se), specificity=beta_moments(sp),
                              sensitivity_draw_summary=summarize(se), specificity_draw_summary=summarize(sp),
                              derivation='superpopulation Beta(1+x,1+n-x) per Sol stratum x community, weighted by cohort counts; Beta moment-matched'),
                  cell=table.to_dict('records'),
                  cell_derivation='Beta(1+x,1+n-x) agreement probability per Sol stratum x cell; Se/Sp derived from cohort counts',
                  assumptions=['human majority is the reference', 'Sol-positive direct-parent target attribution taken as correct',
                               'human-confirmed criticism in Sol-negative items treated as direct-parent criticism (false negative)',
                               'three Sol criticism-YES replies with a non-parent target are true negatives for this endpoint',
                               'style-matched negatives treated as a random sample within community'])
    RUN.mkdir(parents=True, exist_ok=True)
    (OUT / 'misclassification_priors.json').write_text(json.dumps(priors, indent=2, default=float) + '\n')
    table.to_csv(OUT / 'validation_population_counts.csv', index=False)
    metrics.to_csv(OUT / 'validation_metrics.csv', index=False)
    oc.to_csv(OUT / 'operating_characteristics.csv', index=False)
    log(f'validation: {len(metrics)} metric rows, {len(oc)} operating-characteristic rows; '
        f'common Se prior Beta{tuple(round(v, 2) for v in priors["common"]["sensitivity"])}, '
        f'Sp prior Beta{tuple(round(v, 2) for v in priors["common"]["specificity"])}')


# --------------------------------------------------------------------------- criticism models

def build_criticism_model(frame, sampling, measurement=None, priors=None):
    """annotation_models.build_model with an optional misclassification layer.

    Without `measurement` the model is identical to the reported criticism model.
    With it, the linear predictor defines the probability of true criticism and
    the observed Sol label has P(Y*=1) = Se*pi + (1-Sp)*(1-pi).
    Standardized cell probabilities refer to true criticism.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from scripts.diagnostics.annotation_panels import reference_rows
    frame = frame.reset_index(drop=True)
    codes, coords = {}, {'cell': list(CELLS), 'observation': np.arange(len(frame))}
    for key in ('participant', 'context', 'target'):
        codes[key], levels = pd.factorize(frame[key], sort=True)
        coords[key] = np.arange(len(levels))
    refs, weights = reference_rows(frame, 'criticism', False)
    visible = frame.group.eq('G1').to_numpy(dtype=float)
    ai = frame.target_type.eq('AI').to_numpy(dtype=float)
    cell_index = frame.cell.map({c: i for i, c in enumerate(CELLS)}).to_numpy()
    nodes, gh_weights = np.polynomial.hermite.hermgauss(20)
    gh_weights = gh_weights / np.sqrt(np.pi)
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal('alpha', mu=-2, sigma=1.5)
        coefficients = {key: pm.Normal('beta_' + key, 0, sampling['prior_focal_sigma']) for key in ('visible', 'ai', 'interaction')}
        effects, sigmas = {}, {}
        for key in codes:
            sigmas[key] = pm.HalfNormal('sigma_' + key, sampling['prior_scale_sigma'])
            z = pm.Normal('z_' + key, 0, 1, dims=key)
            effects[key] = sigmas[key] * z[codes[key]]
        base = alpha + coefficients['visible'] * visible + effects['participant'] + effects['context']
        eta = base + coefficients['ai'] * ai + coefficients['interaction'] * visible * ai + effects['target']
        pi = pm.math.sigmoid(eta)
        if measurement is None:
            p_obs = pi
        elif measurement == 'common':
            se = pm.Beta('sensitivity', *priors['common']['sensitivity'])
            sp = pm.Beta('specificity', *priors['common']['specificity'])
            p_obs = se * pi + (1 - sp) * (1 - pi)
            se_cell, sp_cell = pt.stack([se] * 4), pt.stack([sp] * 4)
        elif measurement == 'cell':
            table = pd.DataFrame(priors['cell'])
            table = table.loc[table.N.gt(0)].reset_index(drop=True)
            table['cell'] = table.group + '_' + table.target_type
            model.add_coord('validation_stratum', (table.stratum + ':' + table.cell).tolist())
            agree = pm.Beta('agreement', alpha=1 + table.x.to_numpy(float), beta=1 + (table.n - table.x).to_numpy(float),
                            dims='validation_stratum')
            expected = agree * table.N.to_numpy(float)
            tp, fn, n_pos, n_neg = [], [], [], []
            for cell in CELLS:
                pos = table.index[table.cell.eq(cell) & table.stratum.isin(['POS_SCP', 'POS_CONTENT'])].to_numpy()
                neg = table.index[table.cell.eq(cell) & table.stratum.eq('NEG')].to_numpy()
                tp.append(pt.sum(expected[pos]))
                fn.append(pt.sum(expected[neg]))
                n_pos.append(float(table.N[pos].sum()))
                n_neg.append(float(table.N[table.cell.eq(cell) & table.stratum.isin(['NEG', 'NEG_NOTPARENT'])].sum()))
            tp, fn = pt.stack(tp), pt.stack(fn)
            n_pos, n_neg = np.array(n_pos), np.array(n_neg)
            se_cell = pm.Deterministic('sensitivity', tp / (tp + fn), dims='cell')
            sp_cell = pm.Deterministic('specificity', (n_neg - fn) / (n_neg - fn + n_pos - tp), dims='cell')
            p_obs = se_cell[cell_index] * pi + (1 - sp_cell[cell_index]) * (1 - pi)
        else:
            raise ValueError(measurement)
        pm.Bernoulli('outcome', p=p_obs, observed=frame.y.to_numpy(), dims='observation')
        standardized = []
        for cell in CELLS:
            group, typ = cell.split('_')
            keep = frame.iloc[refs].group.eq(group).to_numpy()
            focal = base[refs[keep]] + (coefficients['ai'] + coefficients['interaction'] * int(group == 'G1')) * int(typ == 'AI')
            probability = pm.math.sigmoid(focal[:, None] + np.sqrt(2) * sigmas['target'] * nodes[None, :])
            integrated = pm.math.sum(probability * gh_weights[None, :], axis=1)
            standardized.append(pm.math.sum(integrated * weights[keep]))
        p = pm.Deterministic('cell_probability', pm.math.stack(standardized), dims='cell')
        pm.Deterministic('probability_interaction', (p[0] - p[1]) - (p[2] - p[3]))
        if measurement is not None:
            pm.Deterministic('observed_label_cell_probability', se_cell * p + (1 - sp_cell) * (1 - p), dims='cell')
    return model


def contrast_rows(p, variant, extra=None, log_odds=None, unit_prefix=''):
    p = np.asarray(p)
    values = {cell: p[:, i] for i, cell in enumerate(CELLS)}
    values.update({
        'G1_AI_minus_H': p[:, 0] - p[:, 1], 'G2_AI_minus_H': p[:, 2] - p[:, 3],
        'AI_visible_minus_hidden': p[:, 0] - p[:, 2], 'H_visible_minus_hidden': p[:, 1] - p[:, 3],
        'interaction_visible_minus_hidden_AI_gap': (p[:, 0] - p[:, 1]) - (p[:, 2] - p[:, 3]),
        'risk_ratio_G1': p[:, 0] / p[:, 1], 'risk_ratio_G2': p[:, 2] / p[:, 3],
        'ratio_of_risk_ratios_G1_over_G2': (p[:, 0] / p[:, 1]) / (p[:, 2] / p[:, 3]),
    })
    odds = p / (1 - p)
    values['marginal_odds_ratio_G1'] = odds[:, 0] / odds[:, 1]
    values['marginal_odds_ratio_G2'] = odds[:, 2] / odds[:, 3]
    values['ratio_of_marginal_odds_ratios'] = values['marginal_odds_ratio_G1'] / values['marginal_odds_ratio_G2']
    if log_odds is not None:
        values['log_odds_interaction_coefficient'] = np.asarray(log_odds)
        values['odds_ratio_interaction_coefficient'] = np.exp(np.asarray(log_odds))
    rows = []
    for name, value in values.items():
        unit = ('ratio' if 'ratio' in name else 'log_odds' if name.startswith('log_odds') else 'probability')
        row = dict(variant=variant, estimand=name, unit=unit, **summarize(value))
        if unit == 'ratio':
            row['posterior_probability_gt_zero'] = np.nan
            row['posterior_probability_gt_one'] = float(np.mean(value > 1))
        rows.append({**row, **(extra or {})})
    return rows


def fit_criticism(args):
    import arviz as az
    import pymc as pm
    from scripts.diagnostics.rq3_interaction import predictive_checks, sampler_diagnostics
    sampling, base_seed = sampling_config()
    variant = args.variant
    destination = OUT / 'fits' / ('criticism_' + variant)
    destination.mkdir(parents=True, exist_ok=False)
    frame = criticism_frame()
    measurement, priors = None, None
    if variant in ('content_parent', 'conduct_participation_parent'):
        frame['y'] = frame[variant]
    elif variant.startswith('misclass'):
        measurement = variant.split('_', 1)[1]
        priors = json.loads((OUT / 'misclassification_priors.json').read_text())
    seed = base_seed + 7000 + 100 + CRITICISM_VARIANTS.index(variant) * 10
    cells = frame.groupby('cell').agg(observations=('y', 'size'), positive=('y', 'sum')).reset_index()
    cells.to_csv(destination / 'cells.csv', index=False)
    log(f'fit-criticism {variant}: {len(frame)} replies, {int(frame.y.sum())} positives, seed {seed}; '
        + ', '.join(f'{r.cell} {r.positive}/{r.observations}' for r in cells.itertuples()))
    started = time.monotonic()
    model = build_criticism_model(frame, sampling, measurement, priors)
    with model:
        idata = pm.sample(draws=sampling['draws'], tune=sampling['tune'], chains=4, cores=4,
                          target_accept=sampling['target_accept'], random_seed=seed, progressbar=False)
        predictive = pm.sample_posterior_predictive(idata, var_names=['outcome'], random_seed=seed + 2, progressbar=False)
    elapsed = time.monotonic() - started
    diagnostics = sampler_diagnostics(idata, destination)
    diagnostics['elapsed_seconds'] = elapsed
    idata.to_netcdf(destination / 'posterior.nc')
    checks = frame.assign(target_key=frame.context)
    predictive_checks(checks, predictive.posterior_predictive.outcome.values.reshape(-1, len(frame))).to_csv(
        destination / 'posterior_predictive_checks.csv', index=False)
    post = idata.posterior
    p = post.cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
    rows = contrast_rows(p, variant, dict(scale='true_criticism' if measurement else 'sol_label'),
                         log_odds=post.beta_interaction.values.ravel())
    if measurement:
        q = post.observed_label_cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
        rows += contrast_rows(q, variant, dict(scale='sol_label_implied'))[:9]
        for name in ('sensitivity', 'specificity'):
            values = post[name]
            if 'cell' in values.dims:
                for cell in CELLS:
                    rows.append(dict(variant=variant, estimand=f'{name}_{cell}', unit='probability', scale='measurement',
                                     **summarize(values.sel(cell=cell).values.ravel())))
            else:
                rows.append(dict(variant=variant, estimand=name, unit='probability', scale='measurement',
                                 **summarize(values.values.ravel())))
    table = pd.DataFrame(rows).assign(run_id=RUN_ID, replies=len(frame), positives=int(frame.y.sum()),
                                      participants=frame.participant.nunique())
    table.to_csv(destination / 'estimands.csv', index=False)
    (destination / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    interaction = table.loc[table.estimand.eq('interaction_visible_minus_hidden_AI_gap')].iloc[0]
    log(f'fit-criticism {variant}: done in {elapsed:.1f}s; diagnostics {json.dumps(diagnostics)}; '
        f'interaction {interaction.estimate:.4f} ({interaction.ci_low:.4f}, {interaction.ci_high:.4f})')
    if not diagnostics['passed']:
        raise RuntimeError('Sampler diagnostics failed')


def run_pv_correction(args):
    """Cell-specific predictive-value correction of the admitted posterior (two-stage, no sampling).

    True criticism probability in a cell = q*PPV + (1-q)*FOR, where q is the
    admitted standardized Sol-label probability. Linear, so it commutes with the
    reference-row standardization.
    """
    draws = pd.read_csv(PRIVATE / 'posterior_draws_criticism_primary.csv')
    table, _ = validation_counts()
    rng = np.random.default_rng(20261017)
    q = draws[['p_' + c for c in CELLS]].to_numpy()[rng.integers(0, len(draws), DRAWS)]
    rows = []
    for prior_name, prior, pool in ERROR_SCENARIOS:
        errors = cell_error_draws(table, prior, pool, rng, finite=False)
        true = np.column_stack([q[:, i] * errors[c]['ppv'] + (1 - q[:, i]) * errors[c]['false_omission']
                                for i, c in enumerate(CELLS)])
        variant = f'pv_correction_{prior_name}_{"pooled_negatives" if pool else "cell_negatives"}'
        rows += contrast_rows(true, variant, dict(scale='true_criticism'))
    table_out = pd.DataFrame(rows).assign(run_id=RUN_ID, replies=2547, positives=299, participants=36)
    table_out.to_csv(OUT / 'pv_correction_estimands.csv', index=False)
    for r in table_out.loc[table_out.estimand.eq('interaction_visible_minus_hidden_AI_gap')].itertuples():
        log(f'pv-correction {r.variant}: interaction {r.estimate:.4f} ({r.ci_low:.4f}, {r.ci_high:.4f})')


# --------------------------------------------------------------------------- downvote models

def build_author_model(frame, sampling):
    """rq3_interaction.build_model plus a target-author account intercept."""
    import pymc as pm
    participant, participants = pd.factorize(frame.participant_key, sort=True)
    author, authors = pd.factorize(frame.author_key, sort=True)
    visible = frame.group.eq('G1').astype(float).to_numpy()
    ai = frame.target_author_type.eq('AI').astype(float).to_numpy()
    coords = {'participant': np.arange(len(participants)), 'author': np.arange(len(authors)), 'cell': CELLS}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal('alpha', mu=-2, sigma=1.5)
        focal, scale = sampling['prior_focal_sigma'], sampling['prior_scale_sigma']
        beta_visible = pm.Normal('beta_visible', 0, focal)
        beta_ai = pm.Normal('beta_ai', 0, focal)
        beta_interaction = pm.Normal('beta_interaction', 0, focal)
        sigma_participant = pm.HalfNormal('sigma_participant', scale)
        z_participant = pm.Normal('z_participant', 0, 1, dims='participant')
        sigma_author = pm.HalfNormal('sigma_author', scale)
        z_author = pm.Normal('z_author', 0, 1, dims='author')
        eta = (alpha + beta_visible * visible + beta_ai * ai + beta_interaction * visible * ai
               + sigma_participant * z_participant[participant] + sigma_author * z_author[author])
        prob = pm.math.sigmoid(eta)
        pm.Bernoulli('outcome', p=prob, observed=frame.y.to_numpy())
        pm.Deterministic('cell_probability', pm.math.stack([
            prob[np.flatnonzero(frame.cell.eq(cell))].mean() for cell in CELLS]), dims='cell')
    return model


def fit_downvote(args):
    import pymc as pm
    from scripts.diagnostics import rq3_interaction as rq3
    sampling, base_seed = sampling_config()
    variant = args.variant
    destination = OUT / 'fits' / ('downvote_' + variant)
    destination.mkdir(parents=True, exist_ok=False)
    panel = pd.read_csv(VOTES)
    frame = rq3.prepare_panel(panel, first_target=True)
    frame['author_key'] = frame.group + ':' + pd.to_numeric(frame.target_author_id).astype('int64').astype(str)
    support = rq3.cell_summary(frame)
    admitted = pd.read_csv(ADMITTED / 'rq3_interaction_cells.csv')
    admitted = admitted.loc[admitted.variant.eq('first_target')].set_index('cell')
    for row in support.itertuples():
        if (row.events, row.positives, row.targets, row.positive_targets) != tuple(
                int(admitted.loc[row.cell, k]) for k in ('events', 'positives', 'targets', 'positive_targets')):
            raise ValueError(f'Public panel does not reproduce admitted first-target cell {row.cell}')
    if not frame.target_key.is_unique or len(frame) != 1322:
        raise ValueError('Target-level panel must contain one row for each of 1,322 targets')
    support['authors'] = [frame.loc[frame.cell.eq(c), 'author_key'].nunique() for c in CELLS]
    support.to_csv(destination / 'cell_summary.csv', index=False)
    seed = base_seed + 3000 + 2 + DOWNVOTE_VARIANTS.index(variant) * 10
    log(f'fit-downvote {variant}: {len(frame)} targets, {int(frame.y.sum())} positives, seed {seed}')
    started = time.monotonic()
    if variant == 'first_target':
        rows, diagnostics = rq3.fit(frame, sampling, 1., seed, destination)
    else:
        model = build_author_model(frame, sampling)
        with model:
            idata = pm.sample(draws=sampling['draws'], tune=sampling['tune'], chains=4, cores=4,
                              target_accept=sampling['target_accept'], random_seed=seed, progressbar=False)
            ppc = pm.sample_posterior_predictive(idata, var_names=['outcome'], random_seed=seed + 1, progressbar=False)
        idata.to_netcdf(destination / 'posterior.nc')
        diagnostics = rq3.sampler_diagnostics(idata, destination)
        rq3.predictive_checks(frame, ppc.posterior_predictive.outcome.values.reshape(-1, len(frame))).to_csv(
            destination / 'predictive_checks.csv', index=False)
    import arviz as az
    idata = az.from_netcdf(destination / 'posterior.nc')
    p = idata.posterior.cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
    rows = contrast_rows(p, variant, dict(scale='other_participant_downvote'),
                         log_odds=idata.posterior.beta_interaction.values.ravel())
    diagnostics['elapsed_seconds'] = time.monotonic() - started
    table = pd.DataFrame(rows).assign(run_id=RUN_ID, targets=len(frame), positives=int(frame.y.sum()),
                                      participants=frame.participant_key.nunique(), authors=frame.author_key.nunique())
    table.to_csv(destination / 'estimands.csv', index=False)
    (destination / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    value = table.loc[table.estimand.eq('interaction_visible_minus_hidden_AI_gap')].iloc[0]
    log(f'fit-downvote {variant}: diagnostics {json.dumps(diagnostics)}; interaction {value.estimate:.4f} '
        f'({value.ci_low:.4f}, {value.ci_high:.4f})')
    if not diagnostics['passed']:
        raise RuntimeError('Sampler diagnostics failed')


# --------------------------------------------------------------------------- relative scale

def observed_rows(counts, outcome, source):
    rng = np.random.default_rng(20261018)
    p = np.column_stack([rng.beta(.5 + x, .5 + n - x, DRAWS) for x, n in (counts[c] for c in CELLS)])
    rows = contrast_rows(p, 'observed_counts_jeffreys', dict(scale=outcome, source=source))
    point = {c: x / n for c, (x, n) in counts.items()}
    plug = {'risk_ratio_G1': point['G1_AI'] / point['G1_H'], 'risk_ratio_G2': point['G2_AI'] / point['G2_H']}
    plug['ratio_of_risk_ratios_G1_over_G2'] = plug['risk_ratio_G1'] / plug['risk_ratio_G2']
    odds = {c: v / (1 - v) for c, v in point.items()}
    plug['ratio_of_marginal_odds_ratios'] = (odds['G1_AI'] / odds['G1_H']) / (odds['G2_AI'] / odds['G2_H'])
    for row in rows:
        row['plug_in'] = plug.get(row['estimand'], point.get(row['estimand'], np.nan))
    return rows


def run_scale(args):
    rows = []
    admitted_crit = pd.read_csv(ADMITTED / 'rq3_sol_criticism_estimands.csv').set_index('estimand')
    admitted_dv = pd.read_csv(ADMITTED / 'rq3_interaction_estimands.csv')
    for name, outcome, admitted in (
            ('criticism_primary', 'criticism', admitted_crit),
            ('downvote_primary', 'other_participant_downvote', admitted_dv.loc[admitted_dv.variant.eq('primary')].set_index('estimand')),
            ('downvote_first_target', 'other_participant_downvote', admitted_dv.loc[admitted_dv.variant.eq('first_target')].set_index('estimand'))):
        draws = pd.read_csv(PRIVATE / f'posterior_draws_{name}.csv')
        p = draws[['p_' + c for c in CELLS]].to_numpy()
        for i, cell in enumerate(CELLS):
            if not np.isclose(p[:, i].mean(), admitted.loc[cell, 'estimate'], rtol=0, atol=1e-12):
                raise ValueError(f'{name} draws do not reproduce admitted {cell}')
        rows += contrast_rows(p, 'admitted_' + name, dict(scale=outcome, source=f'admitted posterior ({name})'),
                              log_odds=draws.beta_interaction.to_numpy())
    crit = pd.read_csv(ADMITTED / 'rq3_sol_criticism_cells.csv').set_index('cell')
    rows += observed_rows({c: (int(crit.loc[c, 'positive']), int(crit.loc[c, 'observations'])) for c in CELLS},
                          'criticism', 'rq3_sol_criticism_cells.csv')
    dv = pd.read_csv(ADMITTED / 'rq3_interaction_cells.csv')
    for variant in ('primary', 'first_target'):
        part = dv.loc[dv.variant.eq(variant)].set_index('cell')
        for row in observed_rows({c: (int(part.loc[c, 'positives']), int(part.loc[c, 'events'])) for c in CELLS},
                                 'other_participant_downvote', f'rq3_interaction_cells.csv ({variant})'):
            row['variant'] = f'observed_counts_jeffreys_{variant}'
            rows.append(row)
    pd.DataFrame(rows).assign(run_id=RUN_ID).to_csv(OUT / 'relative_scale.csv', index=False)
    log(f'scale: {len(rows)} rows from admitted posteriors and observed counts')


# --------------------------------------------------------------------------- denominators

def run_denominators(args):
    flow = pd.read_csv(ADMITTED / 'study_sample_flow.csv').pivot(index='stage', columns='group', values='n')
    crit = pd.read_csv(ADMITTED / 'rq3_sol_criticism_cells.csv').set_index('cell')
    tox = pd.read_csv(ADMITTED / 'rq3_toxicity_threshold_summary.csv').set_index(['group', 'target_author_type'])
    dv = pd.read_csv(ADMITTED / 'rq3_interaction_cells.csv')
    dv = dv.loc[dv.variant.eq('primary')].set_index('cell')
    labels = pd.read_csv(LABELS)
    votes = pd.read_csv(VOTES)
    votes = votes.loc[votes.reaction_placement.eq('round_start')]
    rows = []

    def check(quantity, g1, g2, expected_g1, expected_g2, source):
        rows.append(dict(quantity=quantity, G1=g1, G2=g2, total=g1 + g2, expected_G1=expected_g1, expected_G2=expected_g2,
                         expected_total=expected_g1 + expected_g2, passes=bool(g1 == expected_g1 and g2 == expected_g2), source=source))

    g = lambda stage: (int(flow.loc[stage, 'G1']), int(flow.loc[stage, 'G2']))
    check('annotated human comments', *g('annotated_human_comments'), 1518, 1056, 'study_sample_flow.csv')
    check('comments missing parent record', *g('comments_missing_parent_record'), 3, 0, 'study_sample_flow.csv')
    check('linked human reply events', *g('linked_human_reply_events'), 1515, 1056, 'study_sample_flow.csv')
    check('linked = annotated - missing parent', g('annotated_human_comments')[0] - g('comments_missing_parent_record')[0],
          g('annotated_human_comments')[1] - g('comments_missing_parent_record')[1], *g('linked_human_reply_events'), 'study_sample_flow.csv')
    check('self-replies excluded from criticism', *g('criticism_excluded_self_reply'), 7, 17, 'study_sample_flow.csv')
    check('criticism replies = linked - self-replies', g('linked_human_reply_events')[0] - g('criticism_excluded_self_reply')[0],
          g('linked_human_reply_events')[1] - g('criticism_excluded_self_reply')[1], 1508, 1039, 'study_sample_flow.csv')
    check('criticism replies (admitted cells)', int(crit.loc[['G1_AI', 'G1_H'], 'observations'].sum()),
          int(crit.loc[['G2_AI', 'G2_H'], 'observations'].sum()), *g('criticism_replies'), 'rq3_sol_criticism_cells.csv')
    check('toxicity replies (all linked)', int(tox.loc[('G1', 'AI'), 'n'] + tox.loc[('G1', 'H'), 'n']),
          int(tox.loc[('G2', 'AI'), 'n'] + tox.loc[('G2', 'H'), 'n']), *g('linked_human_reply_events'), 'rq3_toxicity_threshold_summary.csv')
    check('AI-target replies, toxicity = criticism', int(tox.loc[('G1', 'AI'), 'n']), int(tox.loc[('G2', 'AI'), 'n']),
          int(crit.loc['G1_AI', 'observations']), int(crit.loc['G2_AI', 'observations']), 'toxicity vs criticism tables')
    check('participant-target replies, toxicity - criticism = self-replies',
          int(tox.loc[('G1', 'H'), 'n'] - crit.loc['G1_H', 'observations']),
          int(tox.loc[('G2', 'H'), 'n'] - crit.loc['G2_H', 'observations']), *g('criticism_excluded_self_reply'),
          'toxicity vs criticism tables')
    check('choice exclusions: fewer than two candidates', *g('choice_excluded_fewer_than_two_eligible_candidates'), 223, 190, 'study_sample_flow.csv')
    check('choice exclusions: target absent', *g('choice_excluded_chosen_target_absent_after_filter'), 6, 9, 'study_sample_flow.csv')
    lg = g('linked_human_reply_events')
    f2 = g('choice_excluded_fewer_than_two_eligible_candidates')
    ta = g('choice_excluded_chosen_target_absent_after_filter')
    check('choice replies = linked - exclusions', lg[0] - f2[0] - ta[0], lg[1] - f2[1] - ta[1], *g('choice_replies'), 'study_sample_flow.csv')
    check('downvote reply events (admitted cells)', int(dv.loc[['G1_AI', 'G1_H'], 'events'].sum()),
          int(dv.loc[['G2_AI', 'G2_H'], 'events'].sum()), *g('choice_replies'), 'rq3_interaction_cells.csv')
    # Public-record cross-checks.
    by = labels.groupby('group')
    check('public: annotated comments', int(by.size()['G1']), int(by.size()['G2']), 1518, 1056, 'Zenodo criticism_labels.csv')
    selfrep = labels.loc[labels.same_author_as_parent].groupby(['group', 'parent_author_type']).size()
    check('public: self-replies (all to participant parents)', int(selfrep.get(('G1', 'H'), 0)), int(selfrep.get(('G2', 'H'), 0)),
          7, 17, 'Zenodo criticism_labels.csv')
    nonself = votes.groupby('group').size()
    check('public: non-self linked reply events in vote panel', int(nonself['G1']), int(nonself['G2']), 1508, 1039, 'Zenodo rq3_vote_panel.csv')
    rq1 = votes.loc[votes.in_rq1_analysis_set.astype(str).str.lower().eq('true')].groupby('group').size()
    check('public: reply events in RQ1/downvote set', int(rq1['G1']), int(rq1['G2']), 1286, 857, 'Zenodo rq3_vote_panel.csv')
    check('self-replies among RQ1 exclusions = (linked - choice) - (non-self - choice)',
          (lg[0] - 1286) - (int(nonself['G1']) - int(rq1['G1'])), (lg[1] - 857) - (int(nonself['G2']) - int(rq1['G2'])),
          7, 17, 'study_sample_flow.csv; Zenodo rq3_vote_panel.csv')
    table = pd.DataFrame(rows).assign(run_id=RUN_ID)
    table.to_csv(OUT / 'denominators.csv', index=False)
    log(f'denominators: {int(table.passes.sum())}/{len(table)} checks pass')
    if not table.passes.all():
        raise RuntimeError('Denominator check failed')


# --------------------------------------------------------------------------- collect

def run_collect(args):
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.read_csv(OUT / 'validation_metrics.csv').assign(run_id=RUN_ID).to_csv(TABLES / 'review_reception_validation_metrics.csv', index=False)
    pd.read_csv(OUT / 'operating_characteristics.csv').assign(run_id=RUN_ID).to_csv(TABLES / 'review_reception_operating_characteristics.csv', index=False)
    criticism = [pd.read_csv(OUT / 'pv_correction_estimands.csv')]
    diagnostics = []
    for variant in CRITICISM_VARIANTS:
        directory = OUT / 'fits' / ('criticism_' + variant)
        if not (directory / 'estimands.csv').exists():
            raise FileNotFoundError(directory)
        criticism.append(pd.read_csv(directory / 'estimands.csv'))
        diagnostics.append(dict(model='criticism', variant=variant, **json.loads((directory / 'diagnostics.json').read_text())))
    pd.concat(criticism, ignore_index=True).to_csv(TABLES / 'review_reception_criticism_estimands.csv', index=False)
    downvote = []
    for variant in DOWNVOTE_VARIANTS:
        directory = OUT / 'fits' / ('downvote_' + variant)
        downvote.append(pd.read_csv(directory / 'estimands.csv'))
        downvote.append(pd.read_csv(directory / 'cell_summary.csv').assign(variant=variant, estimand='cell_support'))
        diagnostics.append(dict(model='downvote', variant=variant, **json.loads((directory / 'diagnostics.json').read_text())))
    pd.concat(downvote, ignore_index=True).to_csv(TABLES / 'review_reception_downvote_estimands.csv', index=False)
    pd.DataFrame(diagnostics).assign(run_id=RUN_ID).to_csv(TABLES / 'review_reception_diagnostics.csv', index=False)
    pd.read_csv(OUT / 'relative_scale.csv').to_csv(TABLES / 'review_reception_relative_scale.csv', index=False)
    pd.read_csv(OUT / 'denominators.csv').to_csv(TABLES / 'review_reception_denominators.csv', index=False)
    import hashlib
    import arviz
    import pymc
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sampling, seed = sampling_config()
    metadata = dict(
        run_id=RUN_ID, script_sha256=digest(Path(__file__)),
        private_script_sha256=digest(Path(__file__).with_name('review_reception_private.py')),
        inputs={str(path.relative_to(ROOT.parent)): digest(path) for path in
                [LABELS, VOTES, *sorted(PRIVATE.glob('*.csv')), PRIVATE / 'private_checks.json',
                 ROOT / 'results/human_validation_20260915/llm_majority_metrics.csv',
                 *[ADMITTED / f'{name}.csv' for name in ('rq3_sol_criticism_cells', 'rq3_sol_criticism_estimands',
                                                        'rq3_interaction_cells', 'rq3_interaction_estimands',
                                                        'study_sample_flow', 'rq3_toxicity_threshold_summary')]]},
        sampling=sampling, base_seed=seed, pymc=pymc.__version__, arviz=arviz.__version__,
        private_host='aggregate-only extraction by review_reception_private.py from runs thread_criticism_sol_review_20260910, '
                     'thread_ai_identity_sol_review_20260910 and rq3_interaction_20260907',
        outputs=sorted(str(p.relative_to(ROOT.parent)) for p in TABLES.glob('review_reception_*.csv')),
        boundary='exploratory ecological comparisons; one community per condition; human majority is the validation reference')
    (RUN / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    log('collect: wrote runs/review_reception_20261015/tables/review_reception_*.csv and run.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('validation', 'fit-criticism', 'fit-downvote', 'pv-correction', 'scale',
                                          'denominators', 'collect'))
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.stage == 'fit-criticism' and args.variant not in CRITICISM_VARIANTS:
        parser.error(f'--variant must be one of {CRITICISM_VARIANTS}')
    if args.stage == 'fit-downvote' and args.variant not in DOWNVOTE_VARIANTS:
        parser.error(f'--variant must be one of {DOWNVOTE_VARIANTS}')
    {'validation': run_validation, 'fit-criticism': fit_criticism, 'fit-downvote': fit_downvote,
     'pv-correction': run_pv_correction, 'scale': run_scale, 'denominators': run_denominators,
     'collect': run_collect}[args.stage](args)


if __name__ == '__main__':
    main()
