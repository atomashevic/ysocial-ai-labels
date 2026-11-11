"""CHB pre-submission review A10: content criticism with classification error propagated.

Named run chb_prereview_20261110, item a10_content_misclass. Stages (run each
with the fitting environment, one PyMC fit at a time):

  fit --variant V   one criticism fit (variants below)
  collect           runs/chb_prereview_20261110/tables/chb_prereview_a10_content_misclass_*.csv and run.json
  record-fixups     add the fixups_20261111 entry to run.json (script hash, tables)

Variants
  content_parent_misclass_cell  Sol content criticism directed at the direct parent
                                (260 positives; the outcome of the admitted 11.4-point
                                `content_parent` row) with community x target-authorship
                                specific error rates
  content_only_misclass_cell    content-only positives (Sol stratum POS_CONTENT, 241)
                                with cell-specific error rates
  content_only_sol              content-only positives on the Sol-label scale (reference)

The model is review_reception.build_criticism_model with measurement='cell'
(the misclassification layer of the `misclass_cell` variant, 9.7 points for
all criticism): the linear predictor gives the probability of true criticism,
P(Sol = 1) = Se * pi + (1 - Sp) * (1 - pi), and Se/Sp per cell follow from
Beta(1 + x, 1 + n - x) agreement probabilities per validation stratum x cell
weighted by the stratum's cohort count N.

Human validation labels are binary (criticism present or absent) and carry no
subtype. Mapping onto content criticism:
  * a human YES on a Sol content-only positive confirms content criticism;
  * a human YES on a Sol style/conduct/participation positive that Sol also
    codes as content criticism of the direct parent confirms content criticism
    (content_parent outcome only);
  * a human YES on a Sol-negative item counts as missed content criticism
    (a false negative for both content outcomes);
  * Sol style/conduct/participation positives are not content-only criticism and
    count as correct negatives of the content-only outcome; those without a Sol
    content code at the parent count as correct negatives of content_parent;
  * the three Sol criticism-YES replies with a non-parent target are correct
    negatives (as in misclass_cell); two of them (G1, AI parent) were validated
    and both have a human YES, which this mapping, inherited from misclass_cell,
    does not count as an error.
Sampling: all style/conduct/participation positives (census), 89 of 245
content-only positives (87 of 241 in the 2,547-reply model cohort), and one
negative per validated positive matched within community on length and style.
The design treats validated items as a random sample within stratum x cell.

Reported sensitivity and specificity rows are posterior means of the fitted
error rates. They differ from the plug-in validation values
(implied_error_rates, Beta(1 + x, 1 + n - x) means weighted by N) because the
agreement probabilities enter the joint likelihood with the observed Sol labels,
so the Sol labels also pull the error rates, as in misclass_cell. The collect
stage writes both side by side (estimands column plug_in_validation).

Inputs: the public criticism panel (review_reception.criticism_frame) and the
aggregate validation strata extracted on the private host for review_reception_20261015
(deposited as 08_analysis_ready/further_checks/a10_content_misclass/validation_criticism_strata.csv).
No item text or identifiers are written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a10_content_misclass'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
CHECKS = ROOT.parent / 'data/08_analysis_ready/further_checks/a10_content_misclass'
STRATA_FILE = CHECKS / 'validation_criticism_strata.csv'
POPULATION_FILE = CHECKS / 'validation_population_counts.csv'
CELLS = ('G1_AI', 'G1_H', 'G2_AI', 'G2_H')
VARIANTS = ('content_parent_misclass_cell', 'content_only_misclass_cell', 'content_only_sol')
EXPECTED_POSITIVES = {'content_parent': 260, 'content_only': 241}
# Validation strata in the measurement layer's vocabulary: POS_* are Sol positives
# for the outcome, NEG are Sol negatives whose human YES counts as a false negative,
# NEG_NOTPARENT are Sol negatives assumed correct (only N is used).
POSITIVE, CONTENT_IN_SCP, FALSE_NEG, CORRECT_NEG = 'POS_CONTENT', 'POS_SCP', 'NEG', 'NEG_NOTPARENT'


def log(message):
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_versions():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'pymc', 'pytensor', 'arviz', 'scipy'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            versions[name] = 'missing'
    return versions


# --------------------------------------------------------------------------- validation mapping

def _true(series):
    return series.astype(str).str.lower().eq('true')


def outcome_column(frame, outcome):
    """0/1 Sol outcome on the criticism frame."""
    if outcome == 'content_parent':
        return frame.content_target.eq('DIRECT_PARENT').astype(int)
    if outcome == 'content_only':
        return frame.stratum.eq('POS_CONTENT').astype(int)
    raise ValueError(outcome)


def content_validation_table(frame, strata, outcome):
    """Validation strata x cell with cohort count N, validated n and human-YES x.

    `frame` is the 2,547-reply criticism frame (review_reception.criticism_frame,
    with columns group, target_type, stratum, content_target); `strata` is the
    aggregate validation table (one row per label pattern with an `items` count).
    Returns columns stratum, source, group, target_type, N, n, x; strata follow
    the measurement-layer vocabulary of build_criticism_model.
    """
    frame = frame.copy()
    content_at_parent = frame.content_target.eq('DIRECT_PARENT')
    scp = frame.stratum.eq('POS_SCP')
    if outcome == 'content_parent':
        source = np.select(
            [frame.stratum.eq('POS_CONTENT'), scp & content_at_parent, frame.stratum.eq('NEG')],
            ['content_only_positive', 'scp_positive_content_at_parent', 'sol_negative'], 'assumed_correct_negative')
    elif outcome == 'content_only':
        source = np.select([frame.stratum.eq('POS_CONTENT'), frame.stratum.eq('NEG')],
                           ['content_only_positive', 'sol_negative'], 'assumed_correct_negative')
    else:
        raise ValueError(outcome)
    frame['source'] = source
    mapping = {'content_only_positive': POSITIVE, 'scp_positive_content_at_parent': CONTENT_IN_SCP,
               'sol_negative': FALSE_NEG, 'assumed_correct_negative': CORRECT_NEG}
    population = frame.groupby(['source', 'group', 'target_type']).size().rename('N').reset_index()

    sample = strata.loc[_true(strata.in_criticism_model)].copy()
    sample['target_type'] = sample.parent_author_type
    sample['yes'] = np.where(sample.human.eq('YES'), sample['items'], 0)
    rows = []
    for record in population.itertuples():
        g, t = record.group, record.target_type
        part = sample.loc[sample.group.eq(g) & sample.target_type.eq(t)]
        if record.source == 'content_only_positive':
            sel = part.loc[part.sol_stratum.eq('CONTENT_ONLY') & part.direct_parent_criticism.eq('YES')]
            n, x = int(sel['items'].sum()), int(sel.yes.sum())
        elif record.source == 'sol_negative':
            sel = part.loc[part.sol_stratum.eq('NEGATIVE')]
            n, x = int(sel['items'].sum()), int(sel.yes.sum())
        elif record.source == 'scp_positive_content_at_parent':
            # Census stratum. The aggregate records content_criticism (any target),
            # not its target; resolve n/x only when unambiguous.
            sel = part.loc[part.sol_stratum.eq('STYLE_OR_PARTICIPATION') & part.direct_parent_criticism.eq('YES')
                           & part.content_criticism.eq('YES')]
            n_any, x_any = int(sel['items'].sum()), int(sel.yes.sum())
            if n_any == record.N:
                n, x = n_any, x_any
            elif x_any == n_any and n_any > record.N:
                n, x = int(record.N), int(record.N)  # every candidate item is human-YES
            else:
                raise ValueError(f'Cannot resolve content-at-parent census for {g}_{t}: {n_any}/{x_any} vs N={record.N}')
        else:
            n, x = 0, 0  # assumed correct; only N enters the error rates
        rows.append(dict(stratum=mapping[record.source], source=record.source, group=g, target_type=t,
                         N=int(record.N), n=n, x=x))
    table = (pd.DataFrame(rows).groupby(['stratum', 'source', 'group', 'target_type'], as_index=False)[['N', 'n', 'x']]
             .sum().sort_values(['stratum', 'group', 'target_type']).reset_index(drop=True))
    if (table.n > table.N).any() or (table.x > table.n).any():
        raise ValueError('Validated counts exceed population counts')
    positives = int(table.loc[table.stratum.isin([POSITIVE, CONTENT_IN_SCP]), 'N'].sum())
    if positives != int(outcome_column(frame, outcome).sum()):
        raise ValueError('Positive strata do not match the outcome')
    if table.N.sum() != len(frame):
        raise ValueError('Strata do not partition the cohort')
    return table


def implied_error_rates(table):
    """Plug-in Se, Sp, PPV and false-omission rate per cell (posterior-mean agreement)."""
    rows = []
    for cell in CELLS:
        g, t = cell.split('_')
        part = table.loc[table.group.eq(g) & table.target_type.eq(t)]
        pos = part.loc[part.stratum.isin([POSITIVE, CONTENT_IN_SCP])]
        neg = part.loc[part.stratum.eq(FALSE_NEG)]
        tp = float(((1 + pos.x) / (2 + pos.n) * pos.N).sum())
        fn = float(((1 + neg.x) / (2 + neg.n) * neg.N).sum())
        n_pos = float(pos.N.sum())
        n_neg = float(part.loc[part.stratum.isin([FALSE_NEG, CORRECT_NEG]), 'N'].sum())
        rows.append(dict(cell=cell, sol_positive=n_pos, sol_negative=n_neg, tp=tp, fn=fn,
                         sensitivity=tp / (tp + fn), specificity=(n_neg - fn) / (n_neg - fn + n_pos - tp),
                         ppv=tp / n_pos if n_pos else np.nan, false_omission=fn / n_neg))
    return pd.DataFrame(rows)


def attach_plug_in(estimands, rates):
    """Add the plug-in validation value beside each posterior-mean sensitivity/specificity row."""
    out = estimands.copy()
    lookup = {f'{name}_{r.cell}': getattr(r, name) for r in rates.itertuples() for name in ('sensitivity', 'specificity')}
    out['plug_in_validation'] = out.estimand.map(lookup).where(out.scale.eq('measurement'))
    return out


def load_inputs():
    from scripts.diagnostics.review_reception import criticism_frame
    frame = criticism_frame()
    strata = pd.read_csv(STRATA_FILE)
    # Content-only validation in the model cohort must reproduce 35/87 and 5/150 negatives.
    population = pd.read_csv(POPULATION_FILE)
    check = population.set_index(['stratum', 'group', 'target_type'])
    for outcome in EXPECTED_POSITIVES:
        table = content_validation_table(frame, strata, outcome)
        pos = table.loc[table.stratum.eq(POSITIVE)].set_index(['group', 'target_type'])
        neg = table.loc[table.stratum.eq(FALSE_NEG)].set_index(['group', 'target_type'])
        for (g, t), r in pos.iterrows():
            if (r.N, r.n, r.x) != tuple(int(check.loc[('POS_CONTENT', g, t), k]) for k in ('N', 'n', 'x')):
                raise ValueError(f'Content-only stratum {g}_{t} does not reproduce review_reception counts')
        for (g, t), r in neg.iterrows():
            if (r.N, r.n, r.x) != tuple(int(check.loc[('NEG', g, t), k]) for k in ('N', 'n', 'x')):
                raise ValueError(f'Negative stratum {g}_{t} does not reproduce review_reception counts')
    return frame, strata


# --------------------------------------------------------------------------- fitting

def fit(args):
    import pymc as pm
    from scripts.diagnostics.review_reception import build_criticism_model, contrast_rows, sampling_config
    from scripts.diagnostics.rq3_interaction import predictive_checks, sampler_diagnostics
    sampling, base_seed = sampling_config()
    variant = args.variant
    outcome = 'content_parent' if variant.startswith('content_parent') else 'content_only'
    measurement = 'cell' if variant.endswith('misclass_cell') else None
    tune = args.tune or sampling['tune']
    destination = OUT / 'fits' / variant
    destination.mkdir(parents=True, exist_ok=bool(args.overwrite))
    frame, strata = load_inputs()
    frame['y'] = outcome_column(frame, outcome)
    if int(frame.y.sum()) != EXPECTED_POSITIVES[outcome]:
        raise ValueError(f'{outcome} must have {EXPECTED_POSITIVES[outcome]} positives')
    table = content_validation_table(frame, strata, outcome)
    table.to_csv(destination / 'validation_table.csv', index=False)
    priors = {'cell': table.drop(columns='source').to_dict('records')} if measurement else None
    seed = base_seed + 7000 + 300 + VARIANTS.index(variant) * 10 + (1 if tune != sampling['tune'] else 0)
    cells = frame.groupby('cell').agg(observations=('y', 'size'), positive=('y', 'sum')).reset_index()
    cells.to_csv(destination / 'cells.csv', index=False)
    started_at = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    log(f'fit {variant}: {len(frame)} replies, {int(frame.y.sum())} positives, measurement {measurement}, '
        f'seed {seed}, tune {tune}; ' + ', '.join(f'{r.cell} {r.positive}/{r.observations}' for r in cells.itertuples()))
    started = time.monotonic()
    model = build_criticism_model(frame, sampling, measurement, priors)
    with model:
        idata = pm.sample(draws=sampling['draws'], tune=tune, chains=4, cores=4,
                          target_accept=sampling['target_accept'], random_seed=seed, progressbar=False)
        predictive = pm.sample_posterior_predictive(idata, var_names=['outcome'], random_seed=seed + 2, progressbar=False)
    elapsed = time.monotonic() - started
    diagnostics = sampler_diagnostics(idata, destination)
    diagnostics.update(elapsed_seconds=elapsed, started_at=started_at, tune=tune, draws=sampling['draws'], chains=4,
                       target_accept=sampling['target_accept'], seed=seed,
                       ended_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'))
    idata.to_netcdf(destination / 'posterior.nc')
    # Same posterior predictive check as misclass_cell: replicated observed Sol labels,
    # cell positives and their concentration within threads.
    checks = frame.assign(target_key=frame.context)
    predictive_checks(checks, predictive.posterior_predictive.outcome.values.reshape(-1, len(frame))).assign(
        variant=variant).to_csv(destination / 'posterior_predictive_checks.csv', index=False)
    post = idata.posterior
    p = post.cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
    rows = contrast_rows(p, variant, dict(scale='true_content_criticism' if measurement else 'sol_label'),
                         log_odds=post.beta_interaction.values.ravel())
    if measurement:
        q = post.observed_label_cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
        rows += contrast_rows(q, variant, dict(scale='sol_label_implied'))[:9]
        for name in ('sensitivity', 'specificity'):
            for cell in CELLS:
                values = post[name].sel(cell=cell).values.ravel()
                rows.append(dict(variant=variant, estimand=f'{name}_{cell}', unit='probability', scale='measurement',
                                 estimate=float(values.mean()), median=float(np.median(values)),
                                 ci_low=float(np.quantile(values, .025)), ci_high=float(np.quantile(values, .975)),
                                 posterior_probability_gt_zero=float(np.mean(values > 0))))
    out = pd.DataFrame(rows).assign(outcome=outcome, replies=len(frame), positives=int(frame.y.sum()),
                                    participants=frame.participant.nunique())
    out.to_csv(destination / 'estimands.csv', index=False)
    (destination / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    value = out.loc[out.estimand.eq('interaction_visible_minus_hidden_AI_gap')].iloc[0]
    log(f'fit {variant}: done in {elapsed:.1f}s; diagnostics {json.dumps(diagnostics)}; '
        f'interaction {value.estimate:.4f} ({value.ci_low:.4f}, {value.ci_high:.4f})')
    if not diagnostics['passed']:
        raise RuntimeError('Sampler diagnostics failed; refit with a longer warm-up (--tune)')


# --------------------------------------------------------------------------- collect

DENOMINATOR = ('2,547 human replies to strictly earlier same-thread non-self direct parents '
               '(G1 AI 432, G1 H 1,076; G2 AI 395, G2 H 644)')


def collect(args):
    TABLES.mkdir(exist_ok=True)
    estimands, checks, diagnostics, validation = [], [], [], []
    for variant in VARIANTS:
        d = OUT / 'fits' / variant
        e = pd.read_csv(d / 'estimands.csv')
        cells = pd.read_csv(d / 'cells.csv').set_index('cell')
        e['denominator'] = DENOMINATOR + f'; {int(e.positives.iloc[0])} Sol positives (' + ', '.join(
            f'{c} {int(cells.loc[c, "positive"])}/{int(cells.loc[c, "observations"])}' for c in CELLS) + ')'
        e['estimate_type'] = 'posterior_mean'
        checks.append(pd.read_csv(d / 'posterior_predictive_checks.csv'))
        diagnostics.append(dict(variant=variant, **json.loads((d / 'diagnostics.json').read_text())))
        if variant.endswith('misclass_cell'):
            v = pd.read_csv(d / 'validation_table.csv')
            outcome = e.outcome.iloc[0]
            plug = implied_error_rates(v).assign(variant=variant, outcome=outcome)
            e = attach_plug_in(e, plug)
        estimands.append(e)
        if variant.endswith('misclass_cell'):
            validation.append(v.assign(variant=variant, outcome=outcome, row_type='stratum'))
            validation.append(plug.assign(row_type='plug_in_error_rates'))
    sign = ('interaction = (G1_AI - G1_H) - (G2_AI - G2_H): visible minus hidden difference in the '
            'AI-minus-participant gap; G1 visible labels, G2 hidden labels')
    est = pd.concat(estimands, ignore_index=True).assign(run_id=RUN_ID, sign_convention=sign)
    est.to_csv(TABLES / f'chb_prereview_{ITEM}_estimands.csv', index=False)
    pd.concat(checks, ignore_index=True).assign(
        run_id=RUN_ID, denominator=DENOMINATOR,
        check='replicated Sol labels: cell positives, threads with a positive, positive pairs within thread').to_csv(
        TABLES / f'chb_prereview_{ITEM}_predictive_checks.csv', index=False)
    val = pd.concat(validation, ignore_index=True).assign(
        run_id=RUN_ID, denominator='2,547-reply model cohort (N); validated items in the cohort (n); human-majority YES (x)')
    val.to_csv(TABLES / f'chb_prereview_{ITEM}_validation.csv', index=False)
    pd.DataFrame(diagnostics).assign(run_id=RUN_ID, denominator=DENOMINATOR).to_csv(
        TABLES / f'chb_prereview_{ITEM}_diagnostics.csv', index=False)
    from scripts.diagnostics import review_reception
    from scripts.diagnostics.review_reception import LABELS, sampling_config
    sampling, seed = sampling_config()
    tables = sorted(TABLES.glob(f'chb_prereview_{ITEM}_*.csv'))
    metadata = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='A10: content criticism with classification error propagated, error rates specific to community x '
                'target authorship (CHB pre-submission review, decision D3)',
        host=socket.gethostname(),
        commands=[f'{sys.executable} -m scripts.diagnostics.chb_prereview_a10_content_misclass fit --variant {v}'
                  for v in VARIANTS] + [f'{sys.executable} -m scripts.diagnostics.chb_prereview_a10_content_misclass collect'],
        code_sha256={str(Path(p).relative_to(ROOT.parent)): sha256(p) for p in (
            Path(__file__), Path(review_reception.__file__), ROOT / 'scripts/diagnostics/rq3_interaction.py',
            ROOT / 'scripts/diagnostics/annotation_panels.py')},
        inputs={str(Path(p).relative_to(ROOT.parent)): sha256(p) for p in (
            LABELS, STRATA_FILE, POPULATION_FILE, ROOT.parent / 'data/08_analysis_ready/results_tables/rq3_sol_criticism_cells.csv', ROOT / 'analysis.yaml')},
        package_versions=package_versions(), sampling=sampling, base_seed=seed,
        priors='intercept Normal(-2,1.5); focal Normal(0,1); varying-intercept SDs HalfNormal(0.5); '
               'validation agreement Beta(1+x,1+n-x) per stratum x cell',
        validation_mapping=[
            'human majority is the reference; human labels are binary criticism without subtype',
            'human YES on a Sol content-only positive confirms content criticism',
            'content_parent: human YES on a Sol style/conduct/participation positive with a Sol content code at the '
            'direct parent confirms content criticism (census, 18 of 19 human YES)',
            'human YES on a Sol-negative item counts as missed content criticism (false negative)',
            'Sol style/conduct/participation positives are not content-only criticism (correct negatives for content_only; '
            'for content_parent when Sol has no content code at the parent)',
            'three Sol criticism-YES replies with a non-parent target are correct negatives; two of them (G1, AI parent) '
            'were validated and both have a human YES, which this mapping, inherited from misclass_cell, does not count '
            'as an error',
            'sampling: style/conduct/participation positives census (61); content-only positives 89 of 245 '
            '(87 of 241 in the model cohort); negatives matched one per positive within community (150, all in cohort); '
            'validated items treated as a random sample within stratum x cell'],
        error_rates=('sensitivity/specificity estimands are posterior means; plug_in_validation gives the plug-in '
                     'validation value (Beta(1+x,1+n-x) means weighted by N); they differ because the agreement '
                     'probabilities share the joint likelihood with the observed Sol labels, so the Sol labels also '
                     'pull the error rates, as in misclass_cell'),
        fits=diagnostics, outputs=[str(p.relative_to(ROOT.parent)) for p in tables],
        tables_sha256={p.name: sha256(p) for p in tables},
        boundary='exploratory ecological comparison; one community per condition; sensitivity analysis, not admitted primary')
    previous = json.loads((RUN / 'run.json').read_text()) if (RUN / 'run.json').exists() else {}
    metadata.update({k: v for k, v in previous.items() if k.startswith('fixups_')})
    (RUN / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    log(f'collect: wrote {len(tables)} tables and run.json')


FIXUPS_20261111 = dict(
    refit=False,
    changes=[
        'estimands table: new column estimate_type (posterior_mean for every row) and plug_in_validation beside the '
        'posterior-mean sensitivity/specificity rows of the misclass_cell variants (values from implied_error_rates, '
        'already in the validation table); other columns and values unchanged',
        'docstring and run.json validation_mapping state that the two validated non-parent Sol positives (G1, AI '
        'parent; both human YES) are treated as correct negatives, inherited from misclass_cell',
        'docstring and run.json error_rates note: the Sol labels also pull the error rates through the joint '
        'likelihood, so posterior-mean sensitivity exceeds the plug-in validation value',
        'collect keeps fixups_* entries when it rewrites run.json'],
    commands=['{py} -m scripts.diagnostics.chb_prereview_a10_content_misclass collect',
              '{py} -m scripts.diagnostics.chb_prereview_a10_content_misclass record-fixups'],
    new_outputs=[f'runs/chb_prereview_20261110/tables/chb_prereview_{ITEM}_estimands.csv (columns estimate_type, plug_in_validation)'])


def record_fixups(args):
    """Add the fixups_20261111 entry (script hash, regenerated tables) to run.json."""
    metadata = json.loads((RUN / 'run.json').read_text())
    tables = sorted(TABLES.glob(f'chb_prereview_{ITEM}_*.csv'))
    metadata['fixups_20261111'] = dict(
        FIXUPS_20261111, recorded_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        commands=[c.format(py=sys.executable) for c in FIXUPS_20261111['commands']],
        script_sha256={str(Path(__file__).relative_to(ROOT.parent)): sha256(Path(__file__))},
        tables_sha256={p.name: sha256(p) for p in tables})
    (RUN / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    log('record-fixups: added fixups_20261111 to run.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('fit', 'collect', 'record-fixups'))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--tune', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.stage == 'fit' and not args.variant:
        parser.error('--variant is required for fit')
    {'fit': fit, 'collect': collect, 'record-fixups': record_fixups}[args.stage](args)


if __name__ == '__main__':
    main()
