"""CHB pre-submission review A12: contribution-level downvote model.

Named run chb_prereview_20261110, item a12_contribution_downvotes. Stages:

  frame             build the contribution frame from the public inputs, reproduce the
                    SI raw counts (rq3_unconditional_downvotes.csv, split "all") and
                    tabulate the effect of excluding self-downvotes
  fit --variant V   author (primary), author_root (adds a root-thread intercept) or
                    author_root_wide (root-thread SD prior HalfNormal(1.0), fix-up 11 Nov)
  collect           runs/chb_prereview_20261110/tables/chb_prereview_a12_contribution_downvotes_*.csv, run.json
  record-fixups     add the fixups_20261111 entry to run.json (script and input hashes)

Frame: one row per contribution (root post or comment) by a participant or an AI
account created on a valid analysis day (publication_analysis.data._valid_contents,
the source of the SI counts). Outcome: at least one downvote from a participant other
than the contribution's author, cast on a valid day (publication_analysis.study.
filter_main_events); reactions carry only their hourly round and are placed at the
round start (the primary placement). No reply-relative window applies.

Follow-up exposure: valid hours from creation to the end of the last valid day,
21 * 24 - valid-clock hours at creation (excluded days are compressed out of the
valid clock), log-transformed and standardized over all rows.

Model: Bernoulli-logit with community, AI authorship and their interaction, the
exposure covariate and a varying intercept for the contribution's author
(sensitivity: plus a root-thread varying intercept). Priors follow the exploratory
downvote model (rq3_interaction): intercept Normal(-2, 1.5), focal coefficients
Normal(0, 1), covariate Normal(0, 0.5), varying-intercept SDs HalfNormal(0.5);
the author_root_wide variant keeps sigma_author ~ HalfNormal(0.5) and widens
sigma_root to HalfNormal(1.0), because the root-thread SD sits near 1 under the
HalfNormal(0.5) prior.
Cell probabilities average fitted probabilities (author and thread effects retained)
over the observed contributions in each community x authorship cell. Contrast:
(G1_AI - G1_H) - (G2_AI - G2_H), visible minus hidden difference in the
AI-minus-participant gap (G1 visible labels, G2 hidden labels), plus the ratio of
risk ratios (G1_AI / G1_H) / (G2_AI / G2_H).

Inputs are the public Zenodo processed snapshots
(public_inputs/processed, written by public_inputs.py), which rebuild the admitted RQ3 tables exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

from scripts.diagnostics.chb_prereview_a10_content_misclass import package_versions, sha256

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a12_contribution_downvotes'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
PUBLIC = ROOT
PUBLIC_CONFIG = PUBLIC / 'analysis.public-rq3.yaml'
FRAME = OUT / 'contribution_frame.csv'
CELLS = ('G1_AI', 'G1_H', 'G2_AI', 'G2_H')
VARIANTS = ('author', 'author_root', 'author_root_wide')
ROOT_SD_PRIOR = {'author_root': None, 'author_root_wide': 1.0}  # None: prior_scale_sigma (0.5)
TOPIC_STEMS = ('22_interest_event_panel', '70_interest_topic_affordance_addon_event_panel')
VALID_DAYS = 21
ADMITTED_UNCONDITIONAL = ROOT.parent / 'data/08_analysis_ready/results_tables/rq3_unconditional_downvotes.csv'
COMPARISON = {'reply_event_primary': (2.31, 0.26, 4.44), 'first_reply_target': (1.30, -1.04, 4.01)}


def log(message):
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def _key(group, values):
    return group.astype(str) + ':' + pd.to_numeric(values, errors='raise').astype('int64').astype(str)


# --------------------------------------------------------------------------- frame

def build_contribution_frame(contents, reactions, valid_days=VALID_DAYS):
    """One row per valid contribution with downvote outcomes and follow-up exposure.

    `contents`: group, content_id, author_id, author_type, comment_to, root_id,
    content_valid_clock_seconds. `reactions`: participant reactions on valid days
    with group, event_type, actor_id, target_content_id. Numeric IDs are local to
    each community and are always keyed with the community.
    """
    items = contents.copy()
    items['author_type'] = items.author_type.astype(str).str.upper()
    if not items.author_type.isin(['AI', 'H']).all():
        raise ValueError('Contribution author types must be AI or H')
    items['contribution_key'] = _key(items.group, items.content_id)
    if items.contribution_key.duplicated().any():
        raise ValueError('Duplicate contributions')
    items['author_key'] = _key(items.group, items.author_id)
    items['root_key'] = _key(items.group, items.root_id)
    items['is_root'] = pd.to_numeric(items.comment_to).eq(-1)
    items['cell'] = items.group.astype(str) + '_' + items.author_type
    downs = reactions.loc[reactions.event_type.eq('downvote')].copy()
    downs['contribution_key'] = _key(downs.group, downs.target_content_id)
    downs['voter_key'] = _key(downs.group, downs.actor_id)
    downs = downs.merge(items[['contribution_key', 'author_key']], on='contribution_key', how='inner',
                        validate='many_to_one')
    downs['self'] = downs.voter_key.eq(downs.author_key)
    counts = pd.DataFrame({
        'downvotes_any': downs.groupby('contribution_key').size(),
        'downvotes_self': downs.loc[downs.self].groupby('contribution_key').size(),
        'downvotes_other': downs.loc[~downs.self].groupby('contribution_key').size(),
        'other_voters': downs.loc[~downs.self].groupby('contribution_key').voter_key.nunique()})
    items = items.merge(counts, left_on='contribution_key', right_index=True, how='left')
    for column in counts.columns:
        items[column] = items[column].fillna(0).astype(int)
    items['y_any'] = items.downvotes_any.gt(0).astype(int)
    items['y'] = items.downvotes_other.gt(0).astype(int)
    items['followup_hours'] = valid_days * 24.0 - items.content_valid_clock_seconds.astype(float) / 3600.0
    if items.followup_hours.isna().any() or not items.followup_hours.gt(0).all():
        raise ValueError('Follow-up exposure must be positive for every contribution')
    log_hours = np.log(items.followup_hours)
    items['z_log_followup'] = (log_hours - log_hours.mean()) / log_hours.std(ddof=0)
    keep = ['group', 'cell', 'author_type', 'contribution_key', 'author_key', 'root_key', 'is_root',
            'downvotes_any', 'downvotes_self', 'downvotes_other', 'other_voters', 'y_any', 'y',
            'followup_hours', 'z_log_followup']
    return items[keep].sort_values('contribution_key').reset_index(drop=True)


def raw_counts(frame):
    rows = []
    for cell in CELLS:
        part = frame.loc[frame.cell.eq(cell)]
        rows.append(dict(cell=cell, contributions=len(part), root_posts=int(part.is_root.sum()),
                         comments=int((~part.is_root).sum()), authors=part.author_key.nunique(),
                         root_threads=part.root_key.nunique(),
                         downvoted_any=int(part.y_any.sum()), downvoted_other=int(part.y.sum()),
                         downvotes_any=int(part.downvotes_any.sum()), downvotes_self=int(part.downvotes_self.sum()),
                         downvotes_other=int(part.downvotes_other.sum()),
                         contributions_only_self_downvoted=int((part.y_any.eq(1) & part.y.eq(0)).sum()),
                         positive_authors=part.loc[part.y.eq(1), 'author_key'].nunique(),
                         share_any=float(part.y_any.mean()), share_other=float(part.y.mean()),
                         median_followup_hours=float(part.followup_hours.median())))
    return pd.DataFrame(rows)


def contrasts(p):
    """Columns G1_AI, G1_H, G2_AI, G2_H. Visible minus hidden throughout."""
    p = np.asarray(p, dtype=float)
    values = {cell: p[:, i] for i, cell in enumerate(CELLS)}
    values.update({
        'G1_AI_minus_H': p[:, 0] - p[:, 1], 'G2_AI_minus_H': p[:, 2] - p[:, 3],
        'AI_visible_minus_hidden': p[:, 0] - p[:, 2], 'H_visible_minus_hidden': p[:, 1] - p[:, 3],
        'interaction_visible_minus_hidden_AI_gap': (p[:, 0] - p[:, 1]) - (p[:, 2] - p[:, 3]),
        'risk_ratio_G1': p[:, 0] / p[:, 1], 'risk_ratio_G2': p[:, 2] / p[:, 3],
        'ratio_of_risk_ratios_G1_over_G2': (p[:, 0] / p[:, 1]) / (p[:, 2] / p[:, 3])})
    return values


def summary_rows(values, **extra):
    rows = []
    for name, v in values.items():
        v = np.asarray(v, dtype=float)
        ratio = 'ratio' in name
        rows.append(dict(estimand=name, unit='ratio' if ratio else ('log_odds' if name.startswith('beta') else 'probability'),
                         estimate=float(v.mean()), median=float(np.median(v)),
                         ci_low=float(np.quantile(v, .025)), ci_high=float(np.quantile(v, .975)),
                         posterior_probability_gt_zero=np.nan if ratio else float(np.mean(v > 0)),
                         posterior_probability_gt_one=float(np.mean(v > 1)) if ratio else np.nan, **extra))
    return rows


def concentration_checks(frame, replicated, key):
    """Observed vs replicated positives per cell and their concentration within `key`."""
    rows = []
    for cell in CELLS:
        mask = frame.cell.eq(cell).to_numpy()
        part = frame.loc[mask]
        codes, keys = pd.factorize(part[key], sort=True)
        observed = np.bincount(codes, weights=part.y, minlength=len(keys))
        predicted = np.zeros((len(keys), len(replicated)), dtype=np.int32)
        np.add.at(predicted, codes, replicated[:, mask].T)
        metrics = {'positive_contributions': (observed.sum(), predicted.sum(axis=0)),
                   f'positive_{key.replace("_key", "")}s': ((observed > 0).sum(), (predicted > 0).sum(axis=0)),
                   f'max_positives_per_{key.replace("_key", "")}': (observed.max(), predicted.max(axis=0))}
        for metric, (obs, pred) in metrics.items():
            rows.append(dict(cell=cell, metric=metric, observed=float(obs), predicted_mean=float(pred.mean()),
                             predicted_low=float(np.quantile(pred, .025)), predicted_high=float(np.quantile(pred, .975)),
                             posterior_predictive_p_ge_observed=float(np.mean(pred >= obs))))
    return pd.DataFrame(rows)


def public_inputs():
    from scripts.publication_analysis import rq3
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents
    from scripts.publication_analysis.study import filter_main_events, load_events
    config = load_config(PUBLIC_CONFIG, profile='release', run_id=f'{RUN_ID}_{ITEM}')
    raw = load_events(config.processed_dir)
    rounds = rq3.derive_round_timestamps(raw, round_start_time=config.rq3.round_start_time,
                                         first_experiment_round=config.rq3.first_experiment_round,
                                         verify_from=config.start_time)
    stamped = rq3.timestamp_reactions(raw, rounds, placement=rq3.PRIMARY_PLACEMENT)
    main = filter_main_events(stamped, start_time=config.start_time, end_time=config.end_time,
                              analysis_set=config.analysis_set)
    reactions = main.loc[main.event_type.isin(rq3.REACTION_EVENTS)
                         & main.actor_type.astype(str).str.upper().eq('H')].copy()
    reactions['event_timestamp'] = pd.to_datetime(reactions.event_timestamp, utc=True).dt.tz_convert('Europe/Belgrade')
    reactions['group'] = reactions.group.astype(str)
    contents = _valid_contents(config)
    return config, contents, reactions


def topic_inputs(legacy_tables_dir):
    """Files data._attach_topics can read via data._topic_table, with SHA-256.

    _topic_table returns an empty table when legacy_tables_dir is unset; otherwise it
    reads each stem in TOPIC_STEMS through io.read_table (Parquet before CSV) and
    stops at the first usable one. Every file it could read is listed.
    """
    record = dict(legacy_tables_dir=None if legacy_tables_dir is None else str(legacy_tables_dir),
                  stems=list(TOPIC_STEMS), files_sha256={})
    if legacy_tables_dir is None:
        record['note'] = ('legacy_tables_dir is null in the public config: _topic_table returns an empty table, '
                          'no topic file is read and the topic columns are all missing (unused by A12)')
        return record
    for stem in TOPIC_STEMS:
        for suffix in ('.parquet', '.csv'):
            path = Path(legacy_tables_dir) / f'{stem}{suffix}'
            if path.exists():
                record['files_sha256'][str(path)] = sha256(path)
                break
    return record


def run_frame(args):
    from scripts.publication_analysis import rq3
    config, contents, reactions = public_inputs()
    unconditional = rq3.unconditional_downvote_rates(contents, reactions, pd.DataFrame())
    admitted = pd.read_csv(ADMITTED_UNCONDITIONAL)
    admitted = admitted.loc[admitted.split.eq('all')].set_index(['group', 'author_type'])
    mine = unconditional.loc[unconditional.split.eq('all')].set_index(['group', 'author_type'])
    for index, row in admitted.iterrows():
        got = tuple(int(mine.loc[index, k]) for k in ('contents_n', 'downvoted_n', 'human_downvotes_total'))
        if got != tuple(int(row[k]) for k in ('contents_n', 'downvoted_n', 'human_downvotes_total')):
            raise ValueError(f'Public inputs do not reproduce rq3_unconditional_downvotes {index}: {got}')
    frame = build_contribution_frame(contents, reactions)
    counts = raw_counts(frame)
    for row in counts.itertuples():
        g, t = row.cell.split('_')
        if (row.contributions, row.downvoted_any) != (int(admitted.loc[(g, t), 'contents_n']),
                                                       int(admitted.loc[(g, t), 'downvoted_n'])):
            raise ValueError(f'Frame does not reproduce the SI raw counts in {row.cell}')
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FRAME, index=False)
    counts.to_csv(OUT / 'raw_counts.csv', index=False)
    log(f'frame: {len(frame)} contributions, {int(frame.y_any.sum())} with any participant downvote, '
        f'{int(frame.y.sum())} with a non-self participant downvote; SI counts reproduced; '
        + '; '.join(f'{r.cell} any {r.downvoted_any}/{r.contributions} other {r.downvoted_other}/{r.contributions} '
                    f'self-votes {r.downvotes_self}' for r in counts.itertuples()))


# --------------------------------------------------------------------------- model

def build_model(frame, sampling, thread=False, root_sd_prior=None):
    """`root_sd_prior` overrides the HalfNormal scale of sigma_root (default prior_scale_sigma)."""
    import pymc as pm
    author, authors = pd.factorize(frame.author_key, sort=True)
    visible = frame.group.eq('G1').astype(float).to_numpy()
    ai = frame.author_type.eq('AI').astype(float).to_numpy()
    exposure = frame.z_log_followup.to_numpy(float)
    coords = {'author': np.arange(len(authors)), 'cell': list(CELLS)}
    if thread:
        root, roots = pd.factorize(frame.root_key, sort=True)
        coords['root'] = np.arange(len(roots))
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal('alpha', mu=-2, sigma=1.5)
        focal, scale = sampling['prior_focal_sigma'], sampling['prior_scale_sigma']
        beta_visible = pm.Normal('beta_visible', 0, focal)
        beta_ai = pm.Normal('beta_ai', 0, focal)
        beta_interaction = pm.Normal('beta_interaction', 0, focal)
        beta_followup = pm.Normal('beta_followup', 0, sampling['prior_adjustment_sigma'])
        sigma_author = pm.HalfNormal('sigma_author', scale)
        z_author = pm.Normal('z_author', 0, 1, dims='author')
        eta = (alpha + beta_visible * visible + beta_ai * ai + beta_interaction * visible * ai
               + beta_followup * exposure + sigma_author * z_author[author])
        if thread:
            sigma_root = pm.HalfNormal('sigma_root', scale if root_sd_prior is None else root_sd_prior)
            z_root = pm.Normal('z_root', 0, 1, dims='root')
            eta = eta + sigma_root * z_root[root]
        prob = pm.math.sigmoid(eta)
        pm.Bernoulli('outcome', p=prob, observed=frame.y.to_numpy())
        pm.Deterministic('cell_probability', pm.math.stack([
            prob[np.flatnonzero(frame.cell.eq(cell))].mean() for cell in CELLS]), dims='cell')
    return model


def fit(args):
    import pymc as pm
    from scripts.diagnostics.review_reception import sampling_config
    from scripts.diagnostics.rq3_interaction import sampler_diagnostics
    sampling, base_seed = sampling_config()
    variant = args.variant
    tune = args.tune or sampling['tune']
    destination = OUT / 'fits' / variant
    destination.mkdir(parents=True, exist_ok=bool(args.overwrite))
    frame = pd.read_csv(FRAME)
    seed = base_seed + 9000 + VARIANTS.index(variant) * 10 + (1 if tune != sampling['tune'] else 0)
    started_at = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    log(f'fit {variant}: {len(frame)} contributions, {int(frame.y.sum())} positives, {frame.author_key.nunique()} authors, '
        f'{frame.root_key.nunique()} root threads, seed {seed}, tune {tune}')
    started = time.monotonic()
    thread = variant in ROOT_SD_PRIOR
    model = build_model(frame, sampling, thread=thread, root_sd_prior=ROOT_SD_PRIOR.get(variant))
    with model:
        idata = pm.sample(draws=sampling['draws'], tune=tune, chains=4, cores=4,
                          target_accept=sampling['target_accept'], random_seed=seed, progressbar=False)
        ppc = pm.sample_posterior_predictive(idata, var_names=['outcome'], random_seed=seed + 1, progressbar=False)
    elapsed = time.monotonic() - started
    idata.to_netcdf(destination / 'posterior.nc')
    diagnostics = sampler_diagnostics(idata, destination)
    diagnostics.update(elapsed_seconds=elapsed, started_at=started_at, tune=tune, draws=sampling['draws'], chains=4,
                       target_accept=sampling['target_accept'], seed=seed,
                       ended_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'))
    replicated = ppc.posterior_predictive.outcome.values.reshape(-1, len(frame))
    checks = [concentration_checks(frame, replicated, 'author_key')]
    checks.append(concentration_checks(frame, replicated, 'root_key'))
    pd.concat(checks, ignore_index=True).assign(variant=variant).to_csv(destination / 'predictive_checks.csv', index=False)
    post = idata.posterior
    p = post.cell_probability.transpose('chain', 'draw', 'cell').values.reshape(-1, 4)
    values = contrasts(p)
    for name in ('beta_interaction', 'beta_ai', 'beta_visible', 'beta_followup', 'sigma_author',
                 *(['sigma_root'] if thread else [])):
        values[name] = post[name].values.ravel()
    rows = summary_rows(values, variant=variant, scale='non_self_participant_downvote')
    for row in rows:
        if row['estimand'].startswith('sigma'):
            row['unit'] = 'log_odds_sd'
    table = pd.DataFrame(rows).assign(contributions=len(frame), positives=int(frame.y.sum()),
                                      authors=frame.author_key.nunique(), root_threads=frame.root_key.nunique())
    table.to_csv(destination / 'estimands.csv', index=False)
    (destination / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    value = table.loc[table.estimand.eq('interaction_visible_minus_hidden_AI_gap')].iloc[0]
    log(f'fit {variant}: done in {elapsed:.1f}s; diagnostics {json.dumps(diagnostics)}; '
        f'interaction {value.estimate:.4f} ({value.ci_low:.4f}, {value.ci_high:.4f})')
    if not diagnostics['passed']:
        raise RuntimeError('Sampler diagnostics failed; refit with a longer warm-up (--tune)')


# --------------------------------------------------------------------------- collect

def collect(args):
    frame = pd.read_csv(FRAME)
    counts = pd.read_csv(OUT / 'raw_counts.csv')
    denominator = ('6,088 valid-day contributions (root posts and comments): G1 AI 1,928, G1 H 1,693, '
                   'G2 AI 1,258, G2 H 1,209; outcome any downvote by a participant other than the author on a valid day')
    sign = ('interaction = (G1_AI - G1_H) - (G2_AI - G2_H): visible minus hidden difference in the AI-minus-participant '
            'gap; ratio_of_risk_ratios = (G1_AI/G1_H)/(G2_AI/G2_H); G1 visible labels, G2 hidden labels')
    counts.assign(run_id=RUN_ID, denominator=denominator, sign_convention=sign).to_csv(
        TABLES / f'chb_prereview_{ITEM}_raw_counts.csv', index=False)
    estimands, checks, diagnostics = [], [], []
    for variant in VARIANTS:
        d = OUT / 'fits' / variant
        estimands.append(pd.read_csv(d / 'estimands.csv'))
        checks.append(pd.read_csv(d / 'predictive_checks.csv'))
        diagnostics.append(dict(variant=variant, **json.loads((d / 'diagnostics.json').read_text())))
    # Observed-count reference (Jeffreys) on the same frame and outcome.
    rng = np.random.default_rng(20261110)
    obs = np.column_stack([rng.beta(.5 + r.downvoted_other, .5 + r.contributions - r.downvoted_other, 20000)
                           for r in counts.set_index('cell').loc[list(CELLS)].itertuples()])
    estimands.append(pd.DataFrame(summary_rows(contrasts(obs), variant='observed_counts_jeffreys',
                                               scale='non_self_participant_downvote')).assign(
        contributions=len(frame), positives=int(frame.y.sum()), authors=frame.author_key.nunique(),
        root_threads=frame.root_key.nunique()))
    comparison = pd.DataFrame([dict(variant=f'admitted_{name}', estimand='interaction_visible_minus_hidden_AI_gap',
                                    unit='probability', estimate=v[0] / 100, ci_low=v[1] / 100, ci_high=v[2] / 100,
                                    scale='other_participant_downvote_24h_reply_window',
                                    source='review_reception_downvote_estimands.csv / rq3_interaction_estimands.csv')
                               for name, v in COMPARISON.items()])
    est = pd.concat(estimands + [comparison], ignore_index=True).assign(run_id=RUN_ID, denominator=denominator,
                                                                         sign_convention=sign)
    est.to_csv(TABLES / f'chb_prereview_{ITEM}_estimands.csv', index=False)
    pd.concat(checks, ignore_index=True).assign(run_id=RUN_ID, denominator=denominator).to_csv(
        TABLES / f'chb_prereview_{ITEM}_predictive_checks.csv', index=False)
    pd.DataFrame(diagnostics).assign(run_id=RUN_ID, denominator=denominator).to_csv(
        TABLES / f'chb_prereview_{ITEM}_diagnostics.csv', index=False)
    from scripts.diagnostics.review_reception import sampling_config
    from scripts.publication_analysis.config import load_config
    sampling, seed = sampling_config()
    config = load_config(PUBLIC_CONFIG, profile='release', run_id=f'{RUN_ID}_{ITEM}')
    tables = sorted(TABLES.glob(f'chb_prereview_{ITEM}_*.csv'))
    processed = PUBLIC / 'public_inputs/processed'
    metadata = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='A12: contribution-level downvote model (one row per valid contribution; non-self participant '
                'downvote; author intercept; root-thread sensitivity)',
        host=socket.gethostname(),
        commands=[f'{sys.executable} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes frame']
                 + [f'{sys.executable} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes '
                    f'fit --variant {v}' for v in VARIANTS]
                 + [f'{sys.executable} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes collect'],
        code_sha256={str(p.relative_to(ROOT.parent)): sha256(p) for p in (
            Path(__file__), ROOT / 'scripts/diagnostics/chb_prereview_a10_content_misclass.py',
            ROOT / 'scripts/diagnostics/rq3_interaction.py', ROOT / 'scripts/diagnostics/review_reception.py',
            ROOT / 'scripts/publication_analysis/rq3.py', ROOT / 'scripts/publication_analysis/data.py',
            ROOT / 'scripts/publication_analysis/study.py')},
        inputs={**{str(p): sha256(p) for p in (PUBLIC_CONFIG, *sorted(processed.glob('snapshot_events.*')),
                                               *sorted(processed.glob('snapshot_contents.*')))},
                str(ADMITTED_UNCONDITIONAL.relative_to(ROOT.parent)): sha256(ADMITTED_UNCONDITIONAL),
                'analysis.yaml': sha256(ROOT / 'analysis.yaml')},
        topic_inputs=topic_inputs(config.legacy_tables_dir),
        intermediate={str(FRAME.relative_to(ROOT.parent)): sha256(FRAME)},
        intermediate_note='the contribution frame stays in the run outputs and is not copied to runs/chb_prereview_20261110/tables',
        package_versions=package_versions(), sampling=sampling, base_seed=seed,
        priors='intercept Normal(-2,1.5); focal Normal(0,1); log follow-up covariate Normal(0,0.5); SDs HalfNormal(0.5); '
               'author_root_wide: sigma_author HalfNormal(0.5), sigma_root HalfNormal(1.0)',
        exposure='log(21*24 - valid-clock hours at creation), standardized over the 6,088 rows',
        reaction_timing='reactions on valid days only; hourly round, placed at round start; no reply-relative window',
        fits=diagnostics, outputs=[str(p.relative_to(ROOT.parent)) for p in tables],
        tables_sha256={p.name: sha256(p) for p in tables},
        boundary='exploratory ecological comparison; one community per condition; sensitivity analysis, not admitted primary')
    previous = json.loads((RUN / 'run.json').read_text()) if (RUN / 'run.json').exists() else {}
    metadata.update({k: v for k, v in previous.items() if k.startswith('fixups_')})
    (RUN / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    log(f'collect: wrote {len(tables)} tables and run.json')


FIXUPS_20261111 = dict(
    changes=[
        'added fit variant author_root_wide: author + root-thread intercepts with sigma_root ~ HalfNormal(1.0) and '
        'sigma_author ~ HalfNormal(0.5) unchanged; other priors, data and sampler settings as author_root '
        '(4 chains x 1,000 draws, 1,000 tuning steps, target_accept 0.95); seed base + 9020',
        'run.json now records topic_inputs: the files data._attach_topics can read through data._topic_table, with '
        'SHA-256; the public config sets legacy_tables_dir to null, so no topic file is read',
        'confirmed that outputs/contribution_frame.csv stays in the run outputs and is not copied to runs/chb_prereview_20261110/tables',
        'collect keeps fixups_* entries when it rewrites run.json'],
    commands=['{py} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes fit '
              '--variant author_root_wide',
              '{py} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes collect',
              '{py} -m scripts.diagnostics.chb_prereview_a12_contribution_downvotes record-fixups'],
    new_outputs=[f'runs/{RUN_ID}/{ITEM}/outputs/fits/author_root_wide/',
                 f'runs/chb_prereview_20261110/tables/chb_prereview_{ITEM}_estimands.csv (variant author_root_wide rows)',
                 f'runs/chb_prereview_20261110/tables/chb_prereview_{ITEM}_diagnostics.csv (variant author_root_wide row)',
                 f'runs/chb_prereview_20261110/tables/chb_prereview_{ITEM}_predictive_checks.csv (variant author_root_wide rows)'])


def record_fixups(args):
    """Add the fixups_20261111 entry (script hash, tables, wide-prior fit summary) to run.json."""
    metadata = json.loads((RUN / 'run.json').read_text())
    tables = sorted(TABLES.glob(f'chb_prereview_{ITEM}_*.csv'))
    est = pd.read_csv(TABLES / f'chb_prereview_{ITEM}_estimands.csv')
    keep = est.estimand.isin(['interaction_visible_minus_hidden_AI_gap', 'sigma_root', 'sigma_author', 'beta_interaction'])
    summary = est.loc[keep & est.variant.isin(['author_root', 'author_root_wide']),
                      ['variant', 'estimand', 'estimate', 'ci_low', 'ci_high']].to_dict('records')
    diagnostics = json.loads((OUT / 'fits/author_root_wide/diagnostics.json').read_text())
    entry = dict(FIXUPS_20261111, recorded_at=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
                 commands=[c.format(py=sys.executable) for c in FIXUPS_20261111['commands']],
                 script_sha256={str(Path(__file__).relative_to(ROOT.parent)): sha256(Path(__file__))},
                 tables_sha256={p.name: sha256(p) for p in tables},
                 author_root_wide_diagnostics=diagnostics, sensitivity_summary=summary,
                 topic_inputs=metadata.get('topic_inputs'))
    metadata['fixups_20261111'] = entry
    (RUN / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    log('record-fixups: added fixups_20261111 to run.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('frame', 'fit', 'collect', 'record-fixups'))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--tune', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.stage == 'fit' and not args.variant:
        parser.error('--variant is required for fit')
    {'frame': run_frame, 'fit': fit, 'collect': collect, 'record-fixups': record_fixups}[args.stage](args)


if __name__ == '__main__':
    main()
