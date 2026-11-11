"""CHB pre-submission item A21: within-community RQ1 AI-authorship effect on the probability scale.

Run ``chb_prereview_20261110`` (item ``a21_rq1_within_community``); no model
fitting. Projections use the admitted M1 posterior of run
``rq1_m1_influence_refit_20260820``.

For each community separately, over that community's own analyzed replies
(each reply weighted equally), and for every posterior draw:

* ``P_obs``: probability of selecting an AI-authored candidate under the
  fitted model on the observed candidate rows. It equals the admitted
  ``observed_risk_set_ai_choice_probability``.
* ``P_cf``: the same probability with the community's total AI-authorship
  effect set to zero. For replier ``i`` in community ``G`` (G = 1 for the
  hidden-label community) that effect is ``beta_AI + G*delta_AI + b_i``,
  where ``b_i`` is the participant AI slope (constrained to average zero over
  the participants of each community). The counterfactual removes the whole
  sum, so AI-authored candidates carry the utility of participant-authored
  candidates with the same age, depth, root status, candidate-author effect
  and replier. A sensitivity counterfactual removes only
  ``beta_AI + G*delta_AI`` and keeps ``b_i``.
* ``P_obs - P_cf``: the AI-authorship effect within the community on the
  probability scale.

Set scopes: all of the community's own sets, and its sets mixed in authorship
(at least one AI-authored and one participant-authored candidate). In sets
of a single author type ``P_obs = P_cf``, so the all-set difference equals the
mixed share times the mixed-set difference. ``G2 - G1`` rows are the
hidden-minus-visible difference of the two within-community differences;
each is standardized over its own community's sets, so this is not the
admitted primary contrast, which standardizes both arms over the pooled sets.
Odds ratios ``exp(beta_AI)`` (visible labels) and ``exp(beta_AI + delta_AI)``
(hidden labels) are summarized from the same draws. Outputs are aggregate
tables only.

Example (one core)::

    OMP_NUM_THREADS=1 nice -n 10 python -m scripts.diagnostics.chb_prereview_a21_rq1_within_community \
        --rq1-dir RQ1_OUTPUTS --admitted-parameters review_choice_models_rq1_parameters.csv \
        --out OUT --chunk 200

From the public record, ``RQ1_OUTPUTS`` is ``public_runs/rq1/outputs`` after
``prepare-choice`` with ``analysis.public-rq1.yaml``; the posterior draws of the
reported M1 fit and the parameter table are then read from
``08_analysis_ready/further_checks/a19_a21_rq1_m1/`` in the Zenodo record.
"""
from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time
from pathlib import Path

from scripts.diagnostics.review_choice_models_common import limit_threads

limit_threads(int(os.environ.get('REVIEW_THREADS', '1')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, arm_frames, check_against_original, peak_rss_gib, save_json, sha256, summarize,
)

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a21_rq1_within_community'
SOURCE_RUN = 'rq1_m1_influence_refit_20260820'
AI_TERM = 'candidate_target_is_ai'
AI_X = 'g2_x_candidate_ai'
M1_PRIMARY_TERMS = (AI_X, 'g2_x_z_log_candidate_age', 'g2_x_z_log_candidate_age_squared',
                    'g2_x_z_log_candidate_depth', 'g2_x_candidate_is_root')
COMMUNITIES = ('G1', 'G2')
COMMUNITY_LABELS = {'G1': 'visible AI labels', 'G2': 'hidden AI labels',
                    'G2-G1': 'hidden minus visible AI labels'}
SCOPES = ('own_sets', 'own_mixed_sets')
PRIMARY_CF = 'zero_total_ai_effect'
SENSITIVITY_CF = 'zero_fixed_ai_effect_keep_participant_slopes'
COUNTERFACTUALS = {
    PRIMARY_CF: 'beta_AI + G*delta_AI + b_i set to zero (primary)',
    SENSITIVITY_CF: 'beta_AI + G*delta_AI set to zero; participant AI slopes b_i kept (sensitivity)',
}
DIFF_SIGN = 'P_obs - P_cf within the community'
DID_SIGN = ('(P_obs - P_cf in G2, hidden AI labels) minus (P_obs - P_cf in G1, visible AI labels); each '
            'standardized over its own community sets, not the admitted pooled-set primary contrast')


def counterfactual_frame(data) -> np.ndarray:
    """Model frame with both AI-authorship fixed-effect columns set to zero."""
    out = data.x.copy()
    for term in (AI_TERM, AI_X):
        out[:, :, data.terms.index(term)] = 0.0
    return out


def primary_terms(data) -> tuple[str, ...]:
    """Every G2 interaction term (the published Primary projection toggles all of them)."""
    return tuple(t for t in data.terms if t.startswith('g2_x_'))


def check_ai_columns(data) -> None:
    """The AI columns must equal the authorship indicator (G2 interaction only in G2 sets)."""
    ai = data.candidate_is_ai * data.mask
    g2 = (np.asarray(data.choice_groups) == 'G2')[:, None]
    if not np.array_equal(data.x[:, :, data.terms.index(AI_TERM)] * data.mask, ai):
        raise ValueError(f'{AI_TERM} differs from the candidate authorship indicator')
    if not np.array_equal(data.x[:, :, data.terms.index(AI_X)] * data.mask, ai * g2):
        raise ValueError(f'{AI_X} differs from G2 x candidate authorship')
    if set(data.choice_groups) != set(COMMUNITIES):
        raise ValueError(f'Expected communities {COMMUNITIES}; got {sorted(set(data.choice_groups))}')


def set_flags(ragged: Ragged) -> pd.DataFrame:
    """Per-set community, authorship mix and observed AI-target indicator (model order)."""
    ends = np.r_[ragged.starts[1:], ragged.n_rows]
    n_ai = np.add.reduceat(ragged.ai, ragged.starts)
    size = ends - ragged.starts
    chosen_ai = np.add.reduceat(ragged.ai * ragged.chosen, ragged.starts) > 0
    return pd.DataFrame({'group': ragged.group, 'n_candidates': size, 'n_ai': n_ai.astype(int),
                         'mixed': (n_ai > 0) & (n_ai < size), 'chosen_ai': chosen_ai})


def scope_masks(flags: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    group = flags.group.to_numpy()
    mixed = flags.mixed.to_numpy()
    return {(g, s): (group == g) & (mixed if s == 'own_mixed_sets' else True)
            for g in COMMUNITIES for s in SCOPES}


def set_probabilities(posterior: Posterior, ragged: Ragged, draws: slice, x_obs: np.ndarray,
                      x_cf: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw, per-set AI-target probability under the fitted model and both counterfactuals."""
    n = draws.stop - draws.start
    receiver = posterior.receiver[draws][:, ragged.receiver] if posterior.receiver is not None \
        else np.zeros((n, ragged.n_rows))
    slopes = posterior.actor[draws][:, ragged.actor] * ragged.ai if posterior.actor is not None \
        else np.zeros((n, ragged.n_rows))
    if posterior.topic is not None:
        raise ValueError('Topic AI slopes are not part of M1; extend the counterfactual before using them')
    beta = posterior.beta[draws]
    fixed_obs = beta @ x_obs.T
    fixed_cf = beta @ x_cf.T
    return {'P_obs': ragged.ai_probability(fixed_obs + receiver + slopes),
            PRIMARY_CF: ragged.ai_probability(fixed_cf + receiver),
            SENSITIVITY_CF: ragged.ai_probability(fixed_cf + receiver + slopes)}


def total_effect_error(posterior: Posterior, ragged: Ragged, data, x_obs, x_cf, draws: slice) -> float:
    """Max |u_obs - u_cf - (beta_AI + G*delta_AI + b_i) * AI| over rows of the given draws."""
    beta = posterior.beta[draws]
    g2 = (ragged.group[ragged.choice] == 'G2').astype(float)
    total = (beta[:, [data.terms.index(AI_TERM)]] + beta[:, [data.terms.index(AI_X)]] * g2[None, :]
             + posterior.actor[draws][:, ragged.actor])
    gap = beta @ x_obs.T - beta @ x_cf.T + posterior.actor[draws][:, ragged.actor] * ragged.ai
    return float(np.max(np.abs(gap - total * ragged.ai)))


def project(data, idata, chunk: int, log, limit: int | None = None):
    """Draw-level community averages of P_obs and both counterfactuals, plus the pooled primary projection."""
    check_ai_columns(data)
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    flags = set_flags(ragged)
    masks = scope_masks(flags)
    x_obs = ragged.rows(data.x)
    x_cf = ragged.rows(counterfactual_frame(data))
    primary = tuple(ragged.rows(f) for f in arm_frames(data, primary_terms(data)))
    total = posterior.n if limit is None else min(limit, posterior.n)
    store: dict[tuple, np.ndarray] = {}
    effect_error = 0.0
    single_type_error = 0.0
    for draws in posterior.chunks(chunk, limit):
        parts = set_probabilities(posterior, ragged, draws, x_obs, x_cf)
        if posterior.actor is not None:
            effect_error = max(effect_error, total_effect_error(posterior, ragged, data, x_obs, x_cf, draws))
        single = ~flags.mixed.to_numpy()
        for name in COUNTERFACTUALS:
            single_type_error = max(single_type_error,
                                    float(np.max(np.abs(parts['P_obs'][:, single] - parts[name][:, single]))))
        for (group, scope), include in masks.items():
            for name, values in parts.items():
                store.setdefault((name, group, scope), np.empty(total))[draws] = values[:, include].mean(axis=1)
        random = ragged.random_part(posterior, draws)
        for arm, rows in zip(COMMUNITIES, primary):
            store.setdefault(('primary_common_opportunity', arm, 'pooled_sets'), np.empty(total))[draws] = \
                ragged.ai_probability(ragged.utility(posterior, draws, rows, random)).mean(axis=1)
        if posterior.actor is not None:
            for group in COMMUNITIES:
                include = masks[(group, 'own_sets')]
                store.setdefault(('reply_weighted_mean_participant_ai_slope', group, 'own_sets'),
                                 np.empty(total))[draws] = posterior.actor[draws][:, data.actor_index[include]].mean(axis=1)
        log(f'a21 projections draws {draws.start}-{draws.stop}; peak RSS {peak_rss_gib():.2f} GiB')
    beta = posterior.beta[:total]
    store[('beta_AI', 'G1', 'parameter')] = beta[:, data.terms.index(AI_TERM)].copy()
    store[('beta_AI', 'G2', 'parameter')] = beta[:, data.terms.index(AI_TERM)] + beta[:, data.terms.index(AI_X)]
    store[('beta_AI', 'G2-G1', 'parameter')] = beta[:, data.terms.index(AI_X)].copy()
    checks = {'total_effect_identity_max_abs_error': effect_error,
              'single_author_type_sets_max_abs_P_obs_minus_P_cf': single_type_error}
    return flags, store, checks


def denominators(flags: pd.DataFrame, data, draws: int) -> dict[tuple[str, str], str]:
    tail = f'{draws} posterior draws of M1, run {SOURCE_RUN}'
    out = {}
    for group in COMMUNITIES:
        part = flags[flags.group.eq(group)]
        mixed = part[part.mixed]
        own = (f'{len(part)} observed human replies in {group} ({COMMUNITY_LABELS[group]}), each weighted equally; '
               f'{int(part.n_candidates.sum())} earlier valid non-self same-thread candidates ({int(part.n_ai.sum())} '
               f'AI-authored); {tail}')
        out[(group, 'own_sets')] = own
        out[(group, 'own_mixed_sets')] = (
            f'{len(mixed)} of {len(part)} observed human replies in {group} ({COMMUNITY_LABELS[group]}) whose '
            f'candidate set includes both AI-authored and participant-authored candidates, each weighted equally; '
            f'{int(mixed.n_candidates.sum())} candidates ({int(mixed.n_ai.sum())} AI-authored); {tail}')
    for scope in SCOPES:
        out[('G2-G1', scope)] = f'{out[("G2", scope)]} | {out[("G1", scope)]}'
    out[('parameter', 'parameter')] = (f'{len(flags)} observed human replies (G1 {int(flags.group.eq("G1").sum())}, '
                                       f'G2 {int(flags.group.eq("G2").sum())}); '
                                       f'{int(flags.n_candidates.sum())} candidates; {len(data.actor_labels)} '
                                       f'participants; {tail}')
    return out


def _row(values, *, null: float = 0.0, scale: float = 1.0, **extra) -> dict:
    values = np.asarray(values, dtype=float) * scale
    s = summarize(values)
    row = dict(run_id=RUN_ID, item=ITEM, **extra)
    row.update(posterior_mean=s['posterior_mean'], posterior_median=s['posterior_median'],
               posterior_sd=s['posterior_sd'], ci_low=s['ci_low'], ci_high=s['ci_high'], null_value=null,
               posterior_probability_gt_null=float(np.mean(values > null)), posterior_draws=s['posterior_draws'])
    return row


def results_table(store: dict, flags: pd.DataFrame, data) -> pd.DataFrame:
    draws = len(store[('beta_AI', 'G1', 'parameter')])
    denominator = denominators(flags, data, draws)
    rows = []
    for scope in SCOPES:
        for group in COMMUNITIES:
            part = flags[flags.group.eq(group)]
            part = part[part.mixed] if scope == 'own_mixed_sets' else part
            common = dict(community=group, community_label=COMMUNITY_LABELS[group], set_scope=scope,
                          choice_sets=len(part), observed_ai_target_replies=int(part.chosen_ai.sum()),
                          observed_ai_share=float(part.chosen_ai.mean()), denominator=denominator[(group, scope)])
            rows.append(dict(run_id=RUN_ID, item=ITEM, estimand='observed_ai_target_share', counterfactual='',
                             unit='percent', posterior_mean=100 * common['observed_ai_share'],
                             definition='observed share of replies whose target is AI-authored (descriptive, '
                                        'no posterior)', **common))
            rows.append(_row(store[('P_obs', group, scope)], scale=100, estimand='P_obs', counterfactual='',
                             unit='percent', definition='mean model P(AI-authored target) on observed candidate rows',
                             **common))
            for name, meaning in COUNTERFACTUALS.items():
                rows.append(_row(store[(name, group, scope)], scale=100, estimand='P_cf', counterfactual=name,
                                 unit='percent', definition=f'mean model P(AI-authored target) with {meaning}',
                                 **common))
                rows.append(_row(store[('P_obs', group, scope)] - store[(name, group, scope)], scale=100,
                                 estimand='P_obs_minus_P_cf', counterfactual=name, unit='percentage_points',
                                 contrast_sign=DIFF_SIGN, definition='within-community AI-authorship effect on the '
                                                                     f'probability scale ({meaning})', **common))
        for name, meaning in COUNTERFACTUALS.items():
            did = ((store[('P_obs', 'G2', scope)] - store[(name, 'G2', scope)])
                   - (store[('P_obs', 'G1', scope)] - store[(name, 'G1', scope)]))
            sets = int(flags.group.eq('G2').sum() + flags.group.eq('G1').sum()) if scope == 'own_sets' \
                else int(flags.mixed.sum())
            rows.append(_row(did, scale=100, estimand='G2_minus_G1_within_community_difference', counterfactual=name,
                             unit='percentage_points', community='G2-G1', community_label=COMMUNITY_LABELS['G2-G1'],
                             set_scope=scope, choice_sets=sets, contrast_sign=DID_SIGN,
                             definition=f'difference of the two within-community differences ({meaning}); '
                                        'not the admitted pooled-set primary contrast',
                             denominator=denominator[('G2-G1', scope)]))
    labels = {'G1': ('exp(beta_AI)', 'odds ratio for an AI-authored candidate, visible-label community'),
              'G2': ('exp(beta_AI + delta_AI)', 'odds ratio for an AI-authored candidate, hidden-label community'),
              'G2-G1': ('exp(delta_AI)', 'ratio of the hidden-label to the visible-label odds ratio')}
    for group, (formula, meaning) in labels.items():
        values = store[('beta_AI', group, 'parameter')]
        rows.append(_row(values, estimand='ai_authorship_log_odds', counterfactual='', unit='log_odds',
                         community=group, community_label=COMMUNITY_LABELS[group], set_scope='parameter',
                         choice_sets=len(flags), definition=f'log of {formula}',
                         denominator=denominator[('parameter', 'parameter')]))
        rows.append(_row(np.exp(values), null=1.0, estimand='ai_authorship_odds_ratio', counterfactual='',
                         unit='odds_ratio', community=group, community_label=COMMUNITY_LABELS[group],
                         set_scope='parameter', choice_sets=len(flags), definition=f'{formula}: {meaning}',
                         denominator=denominator[('parameter', 'parameter')]))
    for group in COMMUNITIES:
        key = ('reply_weighted_mean_participant_ai_slope', group, 'own_sets')
        if key in store:
            rows.append(_row(store[key], estimand='reply_weighted_mean_participant_ai_slope', counterfactual='',
                             unit='log_odds', community=group, community_label=COMMUNITY_LABELS[group],
                             set_scope='own_sets', choice_sets=int(flags.group.eq(group).sum()),
                             definition='mean of b_i over the community replies (b_i averages zero over '
                                        'participants, not over replies); separates the two counterfactuals',
                             denominator=denominator[(group, 'own_sets')]))
    columns = ['run_id', 'item', 'estimand', 'community', 'community_label', 'set_scope', 'counterfactual', 'unit',
               'posterior_mean', 'posterior_median', 'posterior_sd', 'ci_low', 'ci_high', 'null_value',
               'posterior_probability_gt_null', 'posterior_draws', 'choice_sets', 'observed_ai_target_replies',
               'observed_ai_share', 'contrast_sign', 'definition', 'denominator']
    return pd.DataFrame(rows).reindex(columns=columns)


ADMITTED_ESTIMANDS = (
    (('P_obs', 'G1', 'own_sets'), 'observed_risk_set_ai_choice_probability', 'G1'),
    (('P_obs', 'G2', 'own_sets'), 'observed_risk_set_ai_choice_probability', 'G2'),
    (('primary_common_opportunity', 'G1', 'pooled_sets'), 'common_opportunity_ai_choice_probability', 'G1'),
    (('primary_common_opportunity', 'G2', 'pooled_sets'), 'common_opportunity_ai_choice_probability', 'G2'),
    ('primary_contrast', 'G2_minus_G1_common_opportunity_ai_choice_probability', 'G2-G1'),
)
ADMITTED_PARAMETERS = (
    (('beta_AI', 'G1', 'parameter'), 'beta[candidate_target_is_ai]'),
    (('beta_AI', 'G2-G1', 'parameter'), 'beta[g2_x_candidate_ai]'),
    (('beta_AI', 'G2', 'parameter'),
     'G2 slope candidate_target_is_ai (= beta[candidate_target_is_ai] + beta[g2_x_candidate_ai])'),
)


def reproduction_table(store: dict, admitted: pd.DataFrame, parameters: pd.DataFrame | None) -> pd.DataFrame:
    rows = []

    def compare(values, ref, **extra):
        mine = summarize(values)
        row = dict(run_id=RUN_ID, item=ITEM, **extra)
        diff = 0.0
        for stat in ('posterior_mean', 'posterior_median', 'ci_low', 'ci_high'):
            row[f'reproduced_{stat}'] = mine[stat]
            row[f'admitted_{stat}'] = float(ref[stat])
            diff = max(diff, abs(mine[stat] - float(ref[stat])))
        row['max_abs_difference'] = diff
        rows.append(row)

    for key, estimand, group in ADMITTED_ESTIMANDS:
        if key == 'primary_contrast':
            values = (store[('primary_common_opportunity', 'G2', 'pooled_sets')]
                      - store[('primary_common_opportunity', 'G1', 'pooled_sets')])
        else:
            values = store[key]
        ref = admitted[admitted.model.eq('M1') & admitted.estimand.eq(estimand) & admitted.group.eq(group)]
        if len(ref) != 1:
            raise RuntimeError(f'Admitted row missing for {estimand} {group}')
        compare(values, ref.iloc[0], source='rq1_model_estimands.csv', admitted_estimand=estimand, group=group,
                denominator=f'2143 observed human replies; 4000 posterior draws of M1, run {SOURCE_RUN}')
    if parameters is not None:
        for key, name in ADMITTED_PARAMETERS:
            ref = parameters[parameters.parameter.eq(name)]
            if len(ref) != 1:
                raise RuntimeError(f'Admitted parameter row missing for {name}')
            compare(store[key], ref.iloc[0], source='review_choice_models_rq1_parameters.csv',
                    admitted_estimand=name, group=key[1], denominator=str(ref.iloc[0].denominator))
    return pd.DataFrame(rows)


def package_versions() -> dict[str, str]:
    import importlib
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'scipy', 'arviz', 'xarray', 'h5netcdf', 'pymc'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            continue
    return versions


def main(argv=None) -> int:
    from scripts.diagnostics.review_choice_models import load_rq1
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rq1-dir', type=Path, required=True)
    parser.add_argument('--admitted-parameters', type=Path, default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=200)
    parser.add_argument('--max-draws', type=int, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    out = args.out
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'outputs').mkdir(parents=True, exist_ok=True)
    log = RunLog(out / 'run.log')
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    clock = time.monotonic()
    timings = {}
    log(f'{RUN_ID}/{ITEM} start; argv {sys.argv}')

    frame, manifest, data, idata = load_rq1(args.rq1_dir, log)
    timings['load_seconds'] = time.monotonic() - clock
    if set(primary_terms(data)) != set(M1_PRIMARY_TERMS):
        raise RuntimeError(f'Unexpected M1 G2 interaction terms: {primary_terms(data)}')
    primary = arm_frames(data, primary_terms(data))
    from scripts.publication_analysis.model import _common_arm_frames
    if not all(np.array_equal(a, b) for a, b in zip(primary, _common_arm_frames(data))):
        raise RuntimeError('Primary frames differ from the published frame builder')
    checked = check_against_original(idata, data, {'observed': data.x, 'primary_G1': primary[0],
                                                   'primary_G2': primary[1]})
    log(f'ragged projection matches published code (max abs diff): {checked}')

    started_projection = time.monotonic()
    flags, store, checks = project(data, idata, args.chunk, log, args.max_draws)
    timings['projection_seconds'] = time.monotonic() - started_projection
    draws = len(store[('beta_AI', 'G1', 'parameter')])
    log(f'identity checks: {checks}')
    if checks['total_effect_identity_max_abs_error'] > 1e-10 or \
            checks['single_author_type_sets_max_abs_P_obs_minus_P_cf'] > 1e-12:
        raise RuntimeError('Counterfactual identity checks failed')
    for group in COMMUNITIES:
        part = flags[flags.group.eq(group)]
        log(f'{group}: {len(part)} replies, {int(part.chosen_ai.sum())} AI-authored targets, '
            f'{int(part.mixed.sum())} mixed sets')

    from scripts.diagnostics.review_choice_models import reported_rq1_estimands, rq1_input_paths
    admitted = pd.read_csv(reported_rq1_estimands(args.rq1_dir))
    parameters = pd.read_csv(args.admitted_parameters) if args.admitted_parameters else None
    reproduction = reproduction_table(store, admitted, parameters)
    log('reproduction of admitted values:\n' + reproduction[
        ['source', 'admitted_estimand', 'group', 'reproduced_posterior_mean', 'admitted_posterior_mean',
         'max_abs_difference']].to_string(index=False))
    if args.max_draws is None and reproduction.max_abs_difference.max() > 1e-9:
        raise RuntimeError('Admitted values were not reproduced')

    table = results_table(store, flags, data)
    written = {}
    for name, part in (('', table), ('_reproduction_check', reproduction)):
        path = out / 'tables' / f'chb_prereview_{ITEM}{name}.csv'
        part.to_csv(path, index=False)
        written[path.name] = sha256(path)
    np.savez_compressed(out / 'outputs/draw_level_community_averages.npz',
                        **{'|'.join(k): v for k, v in store.items()})
    show = table[table.estimand.isin(['P_obs', 'P_cf', 'P_obs_minus_P_cf', 'G2_minus_G1_within_community_difference',
                                      'observed_ai_target_share', 'ai_authorship_odds_ratio',
                                      'reply_weighted_mean_participant_ai_slope'])]
    log('within-community results:\n' + show[
        ['estimand', 'community', 'set_scope', 'counterfactual', 'choice_sets', 'posterior_mean', 'ci_low',
         'ci_high', 'posterior_probability_gt_null']].to_string(index=False))

    code = {p: sha256(Path(__file__).resolve().parents[2] / p) for p in (
        f'scripts/diagnostics/chb_prereview_{ITEM}.py', 'scripts/diagnostics/review_choice_models_common.py',
        'scripts/diagnostics/review_choice_models.py', 'scripts/publication_analysis/model.py',
        'scripts/publication_analysis/terms.py')}
    rq1 = args.rq1_dir
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in rq1_input_paths(rq1).items()}
    if args.admitted_parameters:
        inputs['admitted_parameters'] = {'path': str(args.admitted_parameters),
                                         'sha256': sha256(args.admitted_parameters)}
    timings['total_seconds'] = time.monotonic() - clock
    primary_rows = reproduction[reproduction.admitted_estimand.str.startswith(('common_opportunity',
                                                                                'G2_minus_G1_common'))]
    save_json(out / 'run.json', {
        'run_id': RUN_ID, 'item': ITEM,
        'purpose': 'CHB pre-submission review A21: RQ1 AI-authorship effect within each community on the '
                   'probability scale (P_obs - P_cf over the community own replies), projected from the admitted '
                   'M1 posterior (no refit), with the AI-authorship odds ratios',
        'host': socket.gethostname(), 'threads': os.environ.get('OMP_NUM_THREADS'), 'nice': os.nice(0),
        'command': ' '.join([f'python -m scripts.diagnostics.chb_prereview_{ITEM}', *sys.argv[1:]]),
        'started': started, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'timings_seconds': timings,
        'code_sha256': code, 'inputs': inputs, 'source_run': SOURCE_RUN, 'package_versions': package_versions(),
        'counterfactual_decision': {
            'primary': f'{PRIMARY_CF}: {COUNTERFACTUALS[PRIMARY_CF]}',
            'sensitivity': f'{SENSITIVITY_CF}: {COUNTERFACTUALS[SENSITIVITY_CF]}',
            'kept_in_both': 'candidate age, age squared, depth, root status and their G2 interactions; '
                            'candidate-author (receiver) effects, centered within community and author type',
            'reason': 'The AI-authorship effect for replier i is beta_AI + G*delta_AI + b_i; b_i averages zero over '
                      'the participants of a community but not over replies, so removing the whole sum is the '
                      'counterfactual in which AI authorship carries no utility for any replier.'},
        'diagnostics': {'check_against_original_max_abs_diff': checked, **checks,
                        'reproduction_max_abs_difference': float(reproduction.max_abs_difference.max()),
                        'primary_projection_reproduction_max_abs_difference':
                            float(primary_rows.max_abs_difference.max()),
                        'posterior_draws': draws, 'peak_rss_gib': peak_rss_gib()},
        'sign_convention': {'within': DIFF_SIGN, 'between': DID_SIGN},
        'tables_sha256': written,
    })
    log(f'{RUN_ID}/{ITEM} done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
