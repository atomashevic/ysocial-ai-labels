"""CHB pre-submission items A1 and A2 from the split_direct_child refit posterior.

Run ``chb_prereview_20261110`` (item ``a1_a2``); no model fitting. Uses the
posterior written by ``review_choice_models_rq2_threads --variant
split_direct_child`` and rebuilds its model data from the same configuration.

A1, direct-reply projection by addressed-account type. Opportunities are
reply-by-account pairs with a competing account (``account_layout``) in which
the account has at least one candidate with D = 1 (the candidate directly
replies to the focal participant's own earlier contribution). For each such
opportunity the estimand is the account's selection probability with D as
observed minus the same probability with D = 0 for all of its candidates (the
``direct_child``, ``candidate_ai_x_direct_child`` and, under G2 coefficients,
``g2_x_candidate_ai_x_direct_child`` terms removed). Each community's
coefficients are applied to all pooled opportunities of the type; opportunities
are averaged equally (primary) or equally within and then across participants.

* AI accounts: reproduces ``review_choice_models_direct_reply``.
* Participant accounts: only ``direct_child`` (gamma_D) applies, so the G2-G1
  contrast is not a community difference in the participant direct-reply
  association; it reflects the G2 terms of the other covariates acting on the
  baseline probabilities. D = 1 for a participant candidate includes every
  top-level comment under the replier's own root post, so the participant set is
  broader than the AI set.

A2, RQ1 standardized AI-target contrast (2,143 replies, 17,489 candidates;
hidden minus visible) from the same posterior, with the history and
direct-reply terms as structural controls: (a) all 14 ``g2_x_`` terms toggled;
(b) only the 5 M1 G2 terms toggled, the 9 history/direct-reply G2 terms held at
G1 (zero) in both arms; (c) the authorship term only, published convention
(other G2 terms as observed). Two sensitivity conventions are added.

Example (one core)::

    OMP_NUM_THREADS=1 nice -n 10 python -m scripts.diagnostics.chb_prereview_a1_direct_reply_by_type run \
        --config analysis.public-rq2.yaml --fit-dir FIT_DIR --admitted-dir ADMITTED \
        --original-direct-reply ORIGINAL_OUT --out OUT
"""
from __future__ import annotations

import argparse
import json
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
from scipy.special import expit  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, account_layout, arm_frames, peak_rss_gib, save_json, sha256, summarize,
    weighted_mean,
)
from scripts.diagnostics.review_choice_models_direct_reply import TERMS as DIRECT_TERMS  # noqa: E402
from scripts.diagnostics.review_choice_models_direct_reply import subset_layout  # noqa: E402

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a1_a2'
SOURCE_FIT = 'rq2_threads_split_direct_child refit, private host, finished 2026-11-11T00:33'
SIGN = 'G2 (hidden AI labels) minus G1 (visible AI labels)'
TYPES = {'AI': 'AI', 'participant': 'HUMAN'}
AI_X = 'g2_x_candidate_ai'
M1_G2 = ('g2_x_z_log_candidate_age', 'g2_x_z_log_candidate_age_squared', 'g2_x_z_log_candidate_depth',
         'g2_x_candidate_is_root', AI_X)
D_DEFINITION = {
    'AI': 'D=1: the AI candidate directly replies to the focal participant\'s own earlier contribution',
    'participant': ('D=1: the participant candidate directly replies to the focal participant\'s own earlier '
                    'contribution; includes every top-level comment under the replier\'s own root post, so the '
                    'participant set is broader than the AI set'),
}
CONTRAST_NOTE = {
    'AI': 'G2-G1 includes the community-specific AI direct-reply term g2_x_candidate_ai_x_direct_child',
    'participant': ('G2-G1 is not a community difference in the participant direct-reply association: participant '
                    'accounts have only gamma_D (direct_child), shared by both communities; the contrast arises from '
                    'the G2 terms of the other covariates acting on the baseline probabilities'),
}


# ---------------------------------------------------------------------------
# A1: opportunity selection and switch-off
# ---------------------------------------------------------------------------

def direct_reply_mask(layout, direct_rows) -> np.ndarray:
    """True for opportunities whose account has at least one candidate with D = 1."""
    ends = np.r_[layout['focal_starts'][1:], len(layout['focal'])]
    return np.array([bool(direct_rows[layout['focal'][s:e]].max() > 0)
                     for s, e in zip(layout['focal_starts'], ends)], dtype=bool)


def switch_off_change(utility, x_rows, beta, sub, indexes) -> np.ndarray:
    """Draws x opportunities: P(account | D observed) - P(account | D = 0 on all its candidates).

    Same computation as ``review_choice_models_direct_reply``; the account
    probability is expit(logsumexp(focal) - logsumexp(other)).
    """
    focal_u = utility[:, sub['focal']]
    removed = (x_rows[sub['focal']][:, indexes] @ beta[:, indexes].T).T
    log_other = np.logaddexp.reduceat(utility[:, sub['other']], sub['other_starts'], axis=1)
    observed = np.logaddexp.reduceat(focal_u, sub['focal_starts'], axis=1)
    switched_off = np.logaddexp.reduceat(focal_u - removed, sub['focal_starts'], axis=1)
    return expit(observed - log_other) - expit(switched_off - log_other)


def type_subsets(data, ragged):
    """Per account type: full layout, D=1 mask and the D=1 sub-layout."""
    direct_rows = ragged.rows(data.x)[:, data.terms.index('direct_child')]
    if not np.isin(direct_rows, (0.0, 1.0)).all():
        raise ValueError('direct_child must be an unscaled 0/1 term')
    out = {}
    for label, author_type in TYPES.items():
        layout = account_layout(data, ragged, author_type)
        mask = direct_reply_mask(layout, direct_rows)
        out[label] = dict(layout=layout, mask=mask, sub=subset_layout(layout, mask))
    return out


# ---------------------------------------------------------------------------
# A2: RQ1 arm frames
# ---------------------------------------------------------------------------

def a2_frames(data) -> dict[str, dict]:
    g2_all = [t for t in data.terms if t.startswith('g2_x_')]
    others = [t for t in g2_all if t not in M1_G2]

    def zero(frames, terms):
        g1, g2 = (f.copy() for f in frames)
        for term in terms:
            g1[:, :, data.terms.index(term)] = 0.0
            g2[:, :, data.terms.index(term)] = 0.0
        return g1, g2

    return {
        'a_all_G2_terms': dict(
            frames=arm_frames(data, g2_all), toggled=g2_all, held_at_G1=[], held_observed=[], role='primary',
            label='(a) all 14 G2 terms toggled'),
        'b_M1_G2_terms_others_at_G1': dict(
            frames=zero(arm_frames(data, M1_G2), others), toggled=list(M1_G2), held_at_G1=others,
            held_observed=[], role='primary',
            label='(b) 5 M1 G2 terms toggled; history and direct-reply G2 terms held at G1'),
        'c_authorship_term_only': dict(
            frames=arm_frames(data, [AI_X]), toggled=[AI_X], held_at_G1=[],
            held_observed=[t for t in g2_all if t != AI_X], role='primary',
            label='(c) authorship term only; other G2 terms as observed (published convention)'),
        'b_alt_M1_G2_terms_others_observed': dict(
            frames=arm_frames(data, M1_G2), toggled=list(M1_G2), held_at_G1=[], held_observed=others,
            role='sensitivity', label='(b, alt) 5 M1 G2 terms toggled; history and direct-reply G2 terms as observed'),
        'c_alt_authorship_term_only_others_at_G1': dict(
            frames=zero(arm_frames(data, [AI_X]), [t for t in g2_all if t != AI_X]), toggled=[AI_X],
            held_at_G1=[t for t in g2_all if t != AI_X], held_observed=[], role='sensitivity',
            label='(c, alt) authorship term only; other G2 terms held at G1'),
    }


# ---------------------------------------------------------------------------
# Projection loop
# ---------------------------------------------------------------------------

def project(idata, data, chunk: int, log, limit: int | None = None):
    from scripts.publication_analysis.model import _common_arm_frames
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    subsets = type_subsets(data, ragged)
    common = _common_arm_frames(data)
    arms = {'G1': ragged.rows(common[0]), 'G2': ragged.rows(common[1])}
    frames = a2_frames(data)
    a2_rows = {k: tuple(ragged.rows(f) for f in v['frames']) for k, v in frames.items()}
    indexes = [data.terms.index(t) for t in DIRECT_TERMS]
    total = posterior.n if limit is None else min(limit, posterior.n)
    a1: dict[tuple, np.ndarray] = {}
    a2: dict[tuple, np.ndarray] = {}
    for draws in posterior.chunks(chunk, limit):
        random = ragged.random_part(posterior, draws)
        beta = posterior.beta[draws]
        for group, rows in arms.items():
            utility = ragged.utility(posterior, draws, rows, random)
            for label, part in subsets.items():
                change = switch_off_change(utility, rows, beta, part['sub'], indexes)
                for weighting, key in (('equal_opportunity', None), ('equal_participant', 'actor')):
                    a1.setdefault((label, group, weighting), np.empty(total))[draws] = weighted_mean(
                        change, None if key is None else part['sub'][key])
        for key, (g1, g2) in a2_rows.items():
            for group, rows in (('G1', g1), ('G2', g2)):
                a2.setdefault((key, group), np.empty(total))[draws] = ragged.ai_probability(
                    ragged.utility(posterior, draws, rows, random)).mean(axis=1)
        log(f'a1/a2 projections draws {draws.start}-{draws.stop}; peak RSS {peak_rss_gib():.2f} GiB')
    return subsets, frames, a1, a2


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _counts(sub):
    groups = pd.Series(sub['group'])
    return dict(opportunities=int(len(sub['choice'])), opportunities_G1=int(groups.eq('G1').sum()),
                opportunities_G2=int(groups.eq('G2').sum()), replies=int(len(np.unique(sub['choice']))),
                participants=int(len(np.unique(sub['actor']))))


def a1_denominator(label, counts, draws, posterior_sha):
    return (f"{counts['opportunities']} reply-by-{label}-account opportunities with a competing account in which "
            f"the {label} account has a direct-reply candidate (G1 {counts['opportunities_G1']}, "
            f"G2 {counts['opportunities_G2']}), from {counts['replies']} replies and {counts['participants']} "
            f"participants; {draws} posterior draws of the split_direct_child refit (posterior.nc sha256 "
            f"{posterior_sha[:12]})")


def a1_estimand_table(subsets, a1, posterior_sha) -> pd.DataFrame:
    rows = []
    for label, part in subsets.items():
        counts = _counts(part['sub'])
        for weighting in ('equal_opportunity', 'equal_participant'):
            g1 = a1[(label, 'G1', weighting)]
            g2 = a1[(label, 'G2', weighting)]
            for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
                rows.append(summarize(
                    values, run_id=RUN_ID, account_type=label,
                    estimand=f'{label}_account_selection_probability_difference_direct_reply_observed_vs_0',
                    group=group, coefficients=('G1 (visible labels)' if group == 'G1' else
                                               'G2 (hidden labels)' if group == 'G2' else SIGN),
                    weighting=weighting, role='primary' if weighting == 'equal_opportunity' else 'sensitivity',
                    terms_switched_off=';'.join(DIRECT_TERMS) if label == 'AI' else 'direct_child',
                    contrast_note=CONTRAST_NOTE[label] if group == 'G2-G1' else '',
                    direct_reply_definition=D_DEFINITION[label], source_fit=SOURCE_FIT, **counts,
                    denominator=a1_denominator(label, counts, len(g1), posterior_sha)))
    return pd.DataFrame(rows)


def a1_observed_table(subsets) -> pd.DataFrame:
    rows = []
    for label, part in subsets.items():
        layout, mask = part['layout'], part['mask']
        groups = pd.Series(layout['group'])
        total = (f"{len(mask)} reply-by-{label}-account opportunities with a competing account "
                 f"(G1 {int(groups.eq('G1').sum())}, G2 {int(groups.eq('G2').sum())}) from "
                 f"{len(np.unique(layout['choice']))} replies")
        for group in ('G1', 'G2', 'ALL'):
            in_group = np.ones_like(mask) if group == 'ALL' else (layout['group'] == group)
            for account, select in (('direct_reply_account', mask), (f'other_{label}_account', ~mask)):
                chosen = layout['selected'][in_group & select]
                rows.append(dict(run_id=RUN_ID, account_type=label, group=group, account=account,
                                 opportunities=int(chosen.size), selected=int(chosen.sum()),
                                 selection_rate=float(chosen.mean()) if chosen.size else float('nan'),
                                 direct_reply_definition=D_DEFINITION[label], denominator=total))
    return pd.DataFrame(rows)


def a2_table(frames, a2, m1_reference: pd.DataFrame | None, data) -> pd.DataFrame:
    reference = {}
    if m1_reference is not None:
        for key, estimand in (('primary', 'G2_minus_G1_common_opportunity_ai_choice_probability'),
                              ('authorship', 'direct_G2xAI_common_opportunity_probability')):
            ref = m1_reference[m1_reference.model.eq('M1') & m1_reference.estimand.eq(estimand)
                               & m1_reference.group.eq('G2-G1')]
            if len(ref) != 1:
                raise RuntimeError(f'M1 reference row missing for {estimand}')
            reference[key] = ref.iloc[0]
    candidates = int(data.mask.sum())
    rows = []
    for key, spec in frames.items():
        g1, g2 = a2[(key, 'G1')], a2[(key, 'G2')]
        ref = reference.get('authorship' if key.startswith('c') else 'primary')
        for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
            row = summarize(values, run_id=RUN_ID, projection=key, label=spec['label'], role=spec['role'],
                            group=group, contrast_sign=SIGN if group == 'G2-G1' else '',
                            toggled_terms=';'.join(spec['toggled']), held_at_G1_terms=';'.join(spec['held_at_G1']),
                            held_observed_terms=';'.join(spec['held_observed']),
                            weighting='reply_event_weighted_equal_per_set', replies=data.n_choices,
                            candidates=candidates, source_fit=SOURCE_FIT)
            if ref is not None and group == 'G2-G1':
                row.update(m1_reference_estimand=ref.estimand, m1_reference_posterior_mean=float(ref.posterior_mean),
                           m1_reference_posterior_median=float(ref.posterior_median),
                           m1_reference_ci_low=float(ref.ci_low), m1_reference_ci_high=float(ref.ci_high))
            row['denominator'] = (f'{data.n_choices} observed human replies with {candidates} earlier valid non-self '
                                  f'same-thread candidates; {len(g1)} posterior draws of the split_direct_child refit')
            rows.append(row)
    return pd.DataFrame(rows)


def interval_class(low: float, high: float) -> str:
    if low > 0:
        return 'excludes_zero_positive'
    if high < 0:
        return 'excludes_zero_negative'
    return 'includes_zero'


def compare_rows(check, mine: pd.DataFrame, admitted: pd.DataFrame, keys, label_cols, denominator) -> list[dict]:
    merged = mine.merge(admitted, on=keys, suffixes=('_refit', '_admitted'), validate='one_to_one')
    if len(merged) != len(admitted) or len(merged) != len(mine):
        raise RuntimeError(f'{check}: {len(mine)} refit rows, {len(admitted)} admitted rows, {len(merged)} matched')
    rows = []
    for _, r in merged.iterrows():
        row = dict(run_id=RUN_ID, check=check, quantity='|'.join(str(r[c]) for c in label_cols))
        for stat in ('posterior_median', 'ci_low', 'ci_high'):
            row[f'admitted_{stat}'] = float(r[f'{stat}_admitted'])
            row[f'refit_{stat}'] = float(r[f'{stat}_refit'])
            row[f'abs_diff_{stat}'] = abs(row[f'refit_{stat}'] - row[f'admitted_{stat}'])
        row['admitted_interval_class'] = interval_class(row['admitted_ci_low'], row['admitted_ci_high'])
        row['refit_interval_class'] = interval_class(row['refit_ci_low'], row['refit_ci_high'])
        row['classification_unchanged'] = row['admitted_interval_class'] == row['refit_interval_class']
        row['denominator'] = denominator
        rows.append(row)
    return rows


def reproduction_table(fit_dir: Path, admitted_dir: Path, original: pd.DataFrame, a1_table: pd.DataFrame
                       ) -> pd.DataFrame:
    fit_denominator = ('fit: 2143 replies, 17489 candidates; projections: 3573 reply-by-AI-account opportunities '
                       'with a competing account (G1 2158, G2 1415) from 1597 replies; 4000 posterior draws')
    params = pd.read_csv(fit_dir / 'parameters.csv')
    admitted_params = pd.read_csv(admitted_dir / 'review_choice_models_fits_rq2_split_direct_child_parameters.csv')
    rows = compare_rows('parameters_refit_vs_admitted', params, admitted_params, ['parameter'], ['parameter'],
                        fit_denominator)
    keys = ['contact_scope', 'direction', 'group', 'weighting']
    est = pd.read_csv(fit_dir / 'estimands.csv')
    admitted_est = pd.read_csv(admitted_dir / 'review_choice_models_fits_rq2_split_direct_child_estimands.csv')
    rows += compare_rows('history_estimands_refit_vs_admitted', est, admitted_est[keys + [
        'posterior_median', 'ci_low', 'ci_high']], keys, keys, fit_denominator)
    dkeys = ['group', 'weighting']
    direct_denominator = ('626 reply-by-AI-account opportunities with a competing account in which the AI account has '
                          'a direct-reply candidate (G1 346, G2 280), from 469 replies and 35 participants; 4000 '
                          'posterior draws')
    admitted_direct = pd.read_csv(admitted_dir / 'review_choice_models_fits_rq2_split_direct_reply_estimand.csv')
    rows += compare_rows('direct_reply_AI_refit_original_script_vs_admitted', original[dkeys + [
        'posterior_median', 'ci_low', 'ci_high']], admitted_direct[dkeys + ['posterior_median', 'ci_low', 'ci_high']],
        dkeys, dkeys, direct_denominator)
    mine = a1_table[a1_table.account_type.eq('AI')][dkeys + ['posterior_median', 'ci_low', 'ci_high']]
    rows += compare_rows('direct_reply_AI_a1_module_vs_admitted', mine, admitted_direct[dkeys + [
        'posterior_median', 'ci_low', 'ci_high']], dkeys, dkeys, direct_denominator)
    exact = compare_rows('direct_reply_AI_a1_module_vs_original_script_same_posterior', mine, original[dkeys + [
        'posterior_median', 'ci_low', 'ci_high']], dkeys, dkeys, direct_denominator)
    for row in exact:  # same posterior: the "admitted" columns hold the original script's values
        row.update({k.replace('admitted_', 'original_script_'): row.pop(k) for k in list(row)
                    if k.startswith('admitted_')})
    return pd.DataFrame(rows + exact)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def package_versions() -> dict[str, str]:
    import importlib
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'scipy', 'arviz', 'xarray', 'netCDF4', 'h5netcdf', 'pymc'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            continue
    return versions


def run(args) -> int:
    import arviz as az
    from scripts.diagnostics.review_choice_models_common import (build_network_from_config, check_against_original,
                                                                 check_cohort, load_analysis_config)
    from scripts.diagnostics.review_choice_models_rq2_threads import design, histories
    from scripts.publication_analysis.model import _common_arm_frames, _direct_ai_arm_frames
    out = args.out
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    log = RunLog(out / 'run.log')
    started, clock = time.strftime('%Y-%m-%dT%H:%M:%S%z'), time.monotonic()
    wall = clock
    timings = {}
    log(f'{RUN_ID}/{ITEM} start; argv {sys.argv}')
    posterior_sha = sha256(args.fit_dir / 'posterior.nc')
    log(f'posterior {args.fit_dir / "posterior.nc"} sha256 {posterior_sha}')
    meta = json.loads((args.fit_dir / 'run.json').read_text())
    if meta['variant'] != 'split_direct_child':
        raise ValueError('Expected the split_direct_child fit')
    config = load_analysis_config(args.config, RUN_ID)
    _, network = build_network_from_config(config, log)
    check_cohort(network)
    split = histories(network, config, log)
    _, data, scales = design(split, meta['variant'])
    if list(data.terms) != meta['terms']:
        raise RuntimeError('Rebuilt terms differ from the fitted terms')
    saved = json.loads((args.fit_dir / 'scales.json').read_text())
    scale_diff = max(max(abs(saved[k]['mean'] - scales[k]['mean']), abs(saved[k]['sd'] - scales[k]['sd']))
                     for k in saved)
    if scale_diff > 1e-12:
        raise RuntimeError('Rebuilt history scales differ from the fitted scales')
    timings['rebuild_seconds'] = time.monotonic() - clock
    idata = az.from_netcdf(args.fit_dir / 'posterior.nc')
    idata.posterior.load()

    frames = a2_frames(data)
    published = {'a_all_G2_terms': _common_arm_frames(data), 'c_authorship_term_only': _direct_ai_arm_frames(data)}
    for key, original in published.items():
        if not all(np.array_equal(a, b) for a, b in zip(frames[key]['frames'], original)):
            raise RuntimeError(f'{key} frames differ from the published frame builder')
    checked = check_against_original(idata, data, {
        'a_G1': frames['a_all_G2_terms']['frames'][0], 'a_G2': frames['a_all_G2_terms']['frames'][1],
        'b_G1': frames['b_M1_G2_terms_others_at_G1']['frames'][0],
        'b_G2': frames['b_M1_G2_terms_others_at_G1']['frames'][1],
        'c_G1': frames['c_authorship_term_only']['frames'][0], 'c_G2': frames['c_authorship_term_only']['frames'][1]})
    log(f'ragged projection matches published code (max abs diff): {checked}')

    clock = time.monotonic()
    subsets, frames, a1, a2 = project(idata, data, args.chunk, log, args.max_draws)
    timings['projection_seconds'] = time.monotonic() - clock
    for label, part in subsets.items():
        log(f'{label}: {int(part["mask"].sum())} of {len(part["mask"])} opportunities have a direct-reply '
            f'candidate; {_counts(part["sub"])}')

    a1_est = a1_estimand_table(subsets, a1, posterior_sha)
    tables = {
        'chb_prereview_a1_direct_reply_by_type_estimands': a1_est,
        'chb_prereview_a1_direct_reply_by_type_observed': a1_observed_table(subsets),
        'chb_prereview_a2_rq1_from_split_refit': a2_table(frames, a2, pd.read_csv(
            args.admitted_dir / 'rq1_model_estimands.csv'), data),
    }
    original = pd.read_csv(args.original_direct_reply / 'direct_reply_estimand.csv')
    reproduction = reproduction_table(args.fit_dir, args.admitted_dir, original, a1_est)
    tables['chb_prereview_a1a2_reproduction_check'] = reproduction
    exact = reproduction[reproduction.check.eq('direct_reply_AI_a1_module_vs_original_script_same_posterior')]
    exact_max = float(exact[['abs_diff_posterior_median', 'abs_diff_ci_low', 'abs_diff_ci_high']].to_numpy().max())
    log(f'A1 AI vs original direct-reply script on the same posterior: max abs diff {exact_max:.3e}')
    if args.max_draws is None and exact_max > 1e-12:
        raise RuntimeError('A1 AI projection does not reproduce the original direct-reply script')
    written = {}
    for name, table in tables.items():
        path = out / 'tables' / f'{name}.csv'
        table.to_csv(path, index=False)
        written[path.name] = sha256(path)
    summary = {}
    for check, part in reproduction.groupby('check', sort=False):
        summary[check] = {
            'rows': int(len(part)),
            'max_abs_diff_median': float(part.abs_diff_posterior_median.max()),
            'max_abs_diff_interval_bounds': float(part[['abs_diff_ci_low', 'abs_diff_ci_high']].to_numpy().max()),
            'all_interval_classifications_unchanged': bool(part.classification_unchanged.all()),
        }
    log('reproduction summary: ' + json.dumps(summary))
    log('A1 estimands:\n' + a1_est[['account_type', 'group', 'weighting', 'posterior_median', 'ci_low', 'ci_high',
                                    'opportunities']].to_string(index=False))
    a2t = tables['chb_prereview_a2_rq1_from_split_refit']
    log('A2 contrasts:\n' + a2t[a2t.group.eq('G2-G1')][['projection', 'posterior_mean', 'posterior_median', 'ci_low',
                                                        'ci_high', 'posterior_probability_gt_zero']].to_string(index=False))

    root = Path(__file__).resolve().parents[2]
    code = {p: sha256(root / p) for p in (
        'scripts/diagnostics/chb_prereview_a1_direct_reply_by_type.py',
        'scripts/diagnostics/review_choice_models_common.py', 'scripts/diagnostics/review_choice_models_direct_reply.py',
        'scripts/diagnostics/review_choice_models_rq2_threads.py', 'scripts/publication_analysis/model.py',
        'scripts/publication_analysis/terms.py', 'scripts/publication_analysis/data.py')}
    inputs = {'posterior': {'path': str(args.fit_dir / 'posterior.nc'), 'sha256': posterior_sha}}
    for name in ('run.json', 'scales.json', 'parameters.csv', 'estimands.csv', 'checks.json'):
        inputs[f'fit_{name}'] = {'path': str(args.fit_dir / name), 'sha256': sha256(args.fit_dir / name)}
    inputs['config'] = {'path': str(args.config), 'sha256': sha256(args.config)}
    inputs['original_direct_reply_estimand'] = {
        'path': str(args.original_direct_reply / 'direct_reply_estimand.csv'),
        'sha256': sha256(args.original_direct_reply / 'direct_reply_estimand.csv')}
    for path in sorted(args.admitted_dir.glob('*.csv')):
        inputs[f'admitted_{path.stem}'] = {'path': str(path), 'sha256': sha256(path)}
    timings['total_seconds'] = time.monotonic() - wall
    save_json(out / 'run.json', {
        'run_id': RUN_ID, 'item': ITEM,
        'purpose': ('CHB pre-submission review A1 (direct-reply projection for AI and participant accounts) and A2 '
                    '(RQ1 standardized AI-target contrast with history and direct-reply terms as structural '
                    'controls), both from the split_direct_child refit posterior; no model fitting'),
        'host': socket.gethostname(), 'threads': os.environ.get('OMP_NUM_THREADS'), 'nice': os.nice(0),
        'command': ' '.join(['python -m scripts.diagnostics.chb_prereview_a1_direct_reply_by_type', *sys.argv[1:]]),
        'started': started, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'timings_seconds': timings,
        'source_fit': SOURCE_FIT, 'fit_meta': {k: meta[k] for k in meta if k != 'terms'},
        'code_sha256': code, 'inputs': inputs, 'package_versions': package_versions(),
        'diagnostics': {'history_scales_max_abs_diff': scale_diff, 'check_against_original_max_abs_diff': checked,
                        'a1_ai_vs_original_script_max_abs_diff': exact_max, 'posterior_draws': len(a2[
                            ('a_all_G2_terms', 'G1')]), 'peak_rss_gib': peak_rss_gib()},
        'reproduction_check': summary, 'sign_convention': SIGN, 'tables_sha256': written,
    })
    log(f'{RUN_ID}/{ITEM} done')
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='A1 and A2 projections from the split_direct_child posterior (private host)')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--fit-dir', type=Path, required=True)
    p.add_argument('--admitted-dir', type=Path, required=True,
                   help='copies of the admitted split_direct_child and rq1_model_estimands tables')
    p.add_argument('--original-direct-reply', type=Path, required=True,
                   help='output directory of review_choice_models_direct_reply run on the same posterior')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chunk', type=int, default=100)
    p.add_argument('--max-draws', type=int, default=None)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
