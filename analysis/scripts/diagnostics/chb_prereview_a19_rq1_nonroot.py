"""CHB pre-submission item A19: RQ1 non-root projection from the admitted M1 posterior.

Run ``chb_prereview_20261110`` (item ``a19_rq1_nonroot``); no model fitting.

For every posterior draw and choice set the M1 candidate probabilities give

* ``P_AI``: probability that the reply selects an AI-authored candidate;
* ``p_root``: probability that it selects the root post (0 when the set has none);
* ``P_nr``: probability of an AI-authored target conditional on the reply not
  targeting the root. It is computed by dropping the root alternative and
  renormalizing, which under IIA equals the multinomial logit restricted to the
  non-root alternatives.

The identity ``P_AI = p_root * 1[root AI] + (1 - p_root) * P_nr`` holds per set
and draw. Projections use the published Primary (all G2 terms) and
Authorship-term-only arm frames; contrasts are G2 (hidden labels) minus G1
(visible labels). Outputs are aggregate CSV tables only.

Example (one core)::

    OMP_NUM_THREADS=1 nice -n 10 python -m scripts.diagnostics.chb_prereview_a19_rq1_nonroot \
        --rq1-dir RQ1_OUTPUTS --out OUT --chunk 200

From the public record, ``RQ1_OUTPUTS`` is ``public_runs/rq1/outputs`` after
``prepare-choice`` with ``analysis.public-rq1.yaml``; the posterior draws of the
reported M1 fit are then read from
``08_analysis_ready/further_checks/a19_a21_rq1_m1/M1.nc`` in the Zenodo record.

Headline table in percentage points from the stored draw-level set averages
(fix-up of 2026-11-11; no new projection; ``p_root`` is reported both over all
2,143 sets and over the 1,677 sets that contain the root)::

    python -m scripts.diagnostics.chb_prereview_a19_rq1_nonroot headline \
        --npz runs/chb_prereview_20261110/a19_rq1_nonroot/outputs/draw_level_set_averages.npz \
        --support runs/chb_prereview_20261110/tables/chb_prereview_a19_rq1_nonroot_support.csv \
        --out runs/chb_prereview_20261110/tables/chb_prereview_a19_rq1_nonroot_headline.csv
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

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, arm_frames, check_against_original, peak_rss_gib, save_json, sha256, summarize,
)

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a19_rq1_nonroot'
SOURCE_RUN = 'rq1_m1_influence_refit_20260820'
ROOT_TERM = 'candidate_is_root'
AI_X = 'g2_x_candidate_ai'
PRIMARY_TERMS = (AI_X, 'g2_x_z_log_candidate_age', 'g2_x_z_log_candidate_age_squared',
                 'g2_x_z_log_candidate_depth', 'g2_x_candidate_is_root')
PROJECTIONS = {'primary_all_G2_terms': PRIMARY_TERMS, 'authorship_term_only': (AI_X,)}
LABELS = {'primary_all_G2_terms': 'Primary (all G2 terms)', 'authorship_term_only': 'Authorship term only'}
ADMITTED = {'primary_all_G2_terms': ('common_opportunity_ai_choice_probability',
                                     'G2_minus_G1_common_opportunity_ai_choice_probability'),
            'authorship_term_only': ('direct_G2xAI_common_opportunity_probability',
                                     'direct_G2xAI_common_opportunity_probability')}
SIGN = 'G2 (hidden AI labels) minus G1 (visible AI labels)'
COMPONENTS = ('P_AI', 'p_root', 'root_ai_component', 'nonroot_component', 'P_nr', 'p_root_given_root_present')


def root_rows(data, ragged: Ragged) -> np.ndarray:
    """Binary root indicator on the valid candidate rows (unscaled model term)."""
    values = ragged.rows(data.x[:, :, data.terms.index(ROOT_TERM)])
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f'{ROOT_TERM} must be an unscaled 0/1 term')
    return values.astype(bool)


def set_support(ragged: Ragged, root: np.ndarray) -> pd.DataFrame:
    """Per-set root and non-root composition (one row per choice set, model order)."""
    ends = np.r_[ragged.starts[1:], ragged.n_rows]
    rows = []
    for start, end in zip(ragged.starts, ends):
        is_root = root[start:end]
        ai = ragged.ai[start:end].astype(bool)
        chosen = ragged.chosen[start:end]
        if is_root.sum() > 1:
            raise ValueError('A choice set holds more than one root candidate')
        if (~is_root).sum() < 1:
            raise ValueError('A choice set has no non-root candidate')
        nr_ai = ai[~is_root]
        rows.append(dict(has_root=bool(is_root.any()), root_ai=bool((is_root & ai).any()),
                         n_candidates=int(end - start), n_nonroot=int((~is_root).sum()),
                         n_nonroot_ai=int(nr_ai.sum()),
                         mixed_nonroot=bool(nr_ai.any() and (~nr_ai).any()),
                         mixed_all=bool(ai.any() and (~ai).any()),
                         chosen_root=bool((chosen & is_root).any()),
                         chosen_ai=bool((chosen & ai).any())))
    out = pd.DataFrame(rows)
    out['group'] = ragged.group[ragged.choice[ragged.starts]]
    return out


def set_components(ragged: Ragged, utility: np.ndarray, root: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw, per-set P_AI, p_root, P_nr and the two terms of the exact decomposition."""
    probability = ragged.probabilities(utility)
    p_ai = np.add.reduceat(probability * ragged.ai, ragged.starts, axis=1)
    p_root = np.add.reduceat(probability * root, ragged.starts, axis=1)
    root_ai = np.add.reduceat(probability * (root * ragged.ai), ragged.starts, axis=1)
    # Drop the root alternative and renormalize on the logit scale (stable when p_root ~ 1).
    p_nr = ragged.ai_probability(np.where(root[None, :], -np.inf, utility))
    return {'P_AI': p_ai, 'p_root': p_root, 'root_ai_component': root_ai,
            'nonroot_component': (1.0 - p_root) * p_nr, 'P_nr': p_nr}


def project(data, idata, chunk: int, log: RunLog, limit: int | None = None):
    """Draw-level set averages of every component, by projection and arm."""
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    root = root_rows(data, ragged)
    support = set_support(ragged, root)
    scopes = {'all_sets': np.ones(len(support), bool), 'mixed_nonroot_sets': support.mixed_nonroot.to_numpy()}
    has_root = support.has_root.to_numpy()
    frames = {key: tuple(ragged.rows(f) for f in arm_frames(data, terms)) for key, terms in PROJECTIONS.items()}
    total = posterior.n if limit is None else min(limit, posterior.n)
    store: dict[tuple, np.ndarray] = {}
    identity_error = 0.0
    for draws in posterior.chunks(chunk, limit):
        random = ragged.random_part(posterior, draws)
        for key, arms in frames.items():
            for arm, rows in zip(('G1', 'G2'), arms):
                parts = set_components(ragged, ragged.utility(posterior, draws, rows, random), root)
                identity = parts['root_ai_component'] + parts['nonroot_component'] - parts['P_AI']
                identity_error = max(identity_error, float(np.max(np.abs(identity))))
                for scope, include in scopes.items():
                    for name, values in parts.items():
                        store.setdefault((key, arm, scope, name), np.empty(total))[draws] = \
                            values[:, include].mean(axis=1)
                    store.setdefault((key, arm, scope, 'p_root_given_root_present'), np.empty(total))[draws] = \
                        parts['p_root'][:, include & has_root].mean(axis=1)
        log(f'a19 projections draws {draws.start}-{draws.stop}; peak RSS {peak_rss_gib():.2f} GiB')
    return support, store, identity_error


def denominators(support: pd.DataFrame, draws: int) -> dict[str, str]:
    counts = support.group.value_counts()
    mixed = support[support.mixed_nonroot].group.value_counts()
    base = (f'{len(support)} observed human replies (G1 {counts.get("G1", 0)}, G2 {counts.get("G2", 0)}); '
            f'{int(support.n_candidates.sum())} earlier valid non-self same-thread candidates, of which '
            f'{int(support.n_nonroot.sum())} non-root; {int(support.has_root.sum())} sets contain the root; '
            f'{draws} posterior draws of M1, run {SOURCE_RUN}')
    return {
        'all_sets': base,
        'mixed_nonroot_sets': (f'{int(support.mixed_nonroot.sum())} replies whose non-root candidates include both '
                               f'AI-authored and participant-authored content (G1 {mixed.get("G1", 0)}, '
                               f'G2 {mixed.get("G2", 0)}), from ' + base),
    }


def projection_table(store, support, draws: int) -> pd.DataFrame:
    denominator = denominators(support, draws)
    sets = {'all_sets': len(support), 'mixed_nonroot_sets': int(support.mixed_nonroot.sum())}
    rows = []
    for key, terms in PROJECTIONS.items():
        for scope in ('all_sets', 'mixed_nonroot_sets'):
            for estimand in ('P_nr', 'P_AI'):
                g1 = store[(key, 'G1', scope, estimand)]
                g2 = store[(key, 'G2', scope, estimand)]
                for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
                    rows.append(summarize(
                        values, run_id=RUN_ID, projection=key, label=LABELS[key], estimand=estimand,
                        set_scope=scope, choice_sets=sets[scope], group=group,
                        contrast_sign=SIGN if group == 'G2-G1' else '', toggled_terms=';'.join(terms),
                        weighting='reply_event_weighted_equal_per_set', denominator=denominator[scope]))
    return pd.DataFrame(rows)


def decomposition_table(store, support, draws: int) -> pd.DataFrame:
    denominator = denominators(support, draws)['all_sets']
    meaning = {'P_AI': 'mean P(AI-authored target)',
               'p_root': 'mean P(root target); 0 for sets without a root',
               'root_ai_component': 'mean p_root*1[root AI-authored]',
               'nonroot_component': 'mean (1-p_root)*P_nr',
               'P_nr': 'mean P(AI-authored target | non-root target)',
               'p_root_given_root_present': f'mean P(root target) over the {int(support.has_root.sum())} sets with a root'}
    rows = []
    for key in PROJECTIONS:
        for name in COMPONENTS:
            g1 = store[(key, 'G1', 'all_sets', name)]
            g2 = store[(key, 'G2', 'all_sets', name)]
            for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
                rows.append(summarize(values, run_id=RUN_ID, projection=key, label=LABELS[key], component=name,
                                      definition=meaning[name], community_spec=group,
                                      contrast_sign=SIGN if group == 'G2-G1' else '',
                                      identity='P_AI = root_ai_component + nonroot_component',
                                      denominator=denominator))
    return pd.DataFrame(rows)


def support_table(support: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in [*support.groupby('group'), ('pooled', support)]:
        not_root = part[~part.chosen_root]
        rows.append(dict(run_id=RUN_ID, group=group, choice_sets=len(part),
                         sets_with_root=int(part.has_root.sum()), sets_with_ai_root=int(part.root_ai.sum()),
                         sets_ge2_nonroot=int((part.n_nonroot >= 2).sum()),
                         sets_mixed_nonroot=int(part.mixed_nonroot.sum()), sets_mixed_any=int(part.mixed_all.sum()),
                         sets_all_nonroot_ai=int((part.n_nonroot_ai == part.n_nonroot).sum()),
                         sets_all_nonroot_participant=int((part.n_nonroot_ai == 0).sum()),
                         candidates=int(part.n_candidates.sum()), nonroot_candidates=int(part.n_nonroot.sum()),
                         nonroot_ai_candidates=int(part.n_nonroot_ai.sum()),
                         replies_targeting_root=int(part.chosen_root.sum()),
                         replies_not_targeting_root=len(not_root),
                         observed_ai_share_not_targeting_root=float(not_root.chosen_ai.mean()),
                         replies_not_targeting_root_in_mixed_nonroot_sets=int(not_root.mixed_nonroot.sum()),
                         observed_ai_share_not_targeting_root_mixed_nonroot=float(
                             not_root[not_root.mixed_nonroot].chosen_ai.mean()),
                         denominator=f'{len(part)} observed human replies in {group}; candidate rows from M1 frame '
                                     f'of run {SOURCE_RUN}'))
    return pd.DataFrame(rows)


def reproduction_table(store, admitted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (arm_estimand, contrast_estimand) in ADMITTED.items():
        g1 = store[(key, 'G1', 'all_sets', 'P_AI')]
        g2 = store[(key, 'G2', 'all_sets', 'P_AI')]
        for group, values, estimand in (('G1', g1, arm_estimand), ('G2', g2, arm_estimand),
                                        ('G2-G1', g2 - g1, contrast_estimand)):
            ref = admitted[admitted.model.eq('M1') & admitted.estimand.eq(estimand) & admitted.group.eq(group)]
            if len(ref) != 1:
                raise RuntimeError(f'Admitted row missing for {estimand} {group}')
            ref = ref.iloc[0]
            mine = summarize(values)
            row = dict(run_id=RUN_ID, projection=key, admitted_estimand=estimand, group=group)
            diff = 0.0
            for stat in ('posterior_mean', 'posterior_median', 'ci_low', 'ci_high'):
                row[f'reproduced_{stat}'] = mine[stat]
                row[f'admitted_{stat}'] = float(ref[stat])
                diff = max(diff, abs(mine[stat] - float(ref[stat])))
            row['max_abs_difference'] = diff
            row['denominator'] = f'2143 observed human replies; 4000 posterior draws of M1, run {SOURCE_RUN}'
            rows.append(row)
    return pd.DataFrame(rows)


HEADLINE = (  # (component, set scope label, choice-set column of the pooled support row)
    ('P_nr', 'all_sets', 'choice_sets'),
    ('P_AI', 'all_sets', 'choice_sets'),
    ('p_root', 'all_sets', 'choice_sets'),
    ('p_root_given_root_present', 'sets_with_root', 'sets_with_root'),
)


def headline_table(store: dict, support: pd.DataFrame) -> pd.DataFrame:
    """Headline values in percentage points from the stored draw-level set averages (no new projection).

    ``p_root`` averages over all choice sets (0 for sets without a root);
    ``p_root_given_root_present`` averages over the sets that contain the root.
    """
    pooled = support.set_index('group').loc['pooled']
    rows = []
    for key in PROJECTIONS:
        for component, scope, column in HEADLINE:
            g1 = store[(key, 'G1', 'all_sets', component)]
            g2 = store[(key, 'G2', 'all_sets', component)]
            for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
                s = summarize(100 * values)
                rows.append(dict(run_id=RUN_ID, projection=key, label=LABELS[key], component=component,
                                 set_scope=scope, choice_sets=int(pooled[column]), group=group,
                                 unit='percent' if group != 'G2-G1' else 'percentage_points',
                                 posterior_mean=s['posterior_mean'], ci_low=s['ci_low'], ci_high=s['ci_high'],
                                 posterior_probability_gt_zero=s['posterior_probability_gt_zero'],
                                 contrast_sign=SIGN if group == 'G2-G1' else '',
                                 denominator=(f'{int(pooled[column])} of {int(pooled.choice_sets)} observed human '
                                              f'replies (choice sets); {len(g1)} posterior draws of M1, run '
                                              f'{SOURCE_RUN}')))
    return pd.DataFrame(rows)


def headline_main(npz: Path, support_csv: Path, out_csv: Path) -> pd.DataFrame:
    with np.load(npz) as stored:
        store = {tuple(k.split('|')): stored[k] for k in stored.files}
    table = headline_table(store, pd.read_csv(support_csv))
    table.to_csv(out_csv, index=False)
    return table


def package_versions() -> dict[str, str]:
    import importlib
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'scipy', 'arviz', 'xarray', 'netCDF4', 'h5netcdf', 'pymc'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            continue
    return versions


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['headline']:
        sub = argparse.ArgumentParser(description='Headline table from stored draw-level set averages (public record)')
        sub.add_argument('--npz', type=Path, required=True)
        sub.add_argument('--support', type=Path, required=True)
        sub.add_argument('--out', type=Path, required=True)
        args = sub.parse_args(argv[1:])
        headline_main(args.npz, args.support, args.out)
        return 0
    from scripts.diagnostics.review_choice_models import load_rq1
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rq1-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=200)
    parser.add_argument('--max-draws', type=int, default=None)
    args = parser.parse_args(argv)
    out = args.out
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'outputs').mkdir(parents=True, exist_ok=True)
    log = RunLog(out / 'run.log')
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    log(f'{RUN_ID}/{ITEM} start; argv {sys.argv}')

    frame, manifest, data, idata = load_rq1(args.rq1_dir, log)
    primary = arm_frames(data, PROJECTIONS['primary_all_G2_terms'])
    authorship = arm_frames(data, PROJECTIONS['authorship_term_only'])
    from scripts.publication_analysis.model import _common_arm_frames, _direct_ai_arm_frames
    for mine, original in ((primary, _common_arm_frames(data)), (authorship, _direct_ai_arm_frames(data))):
        if not all(np.array_equal(a, b) for a, b in zip(mine, original)):
            raise RuntimeError('Projection frames differ from the published frame builders')
    checked = check_against_original(idata, data, {'primary_G1': primary[0], 'primary_G2': primary[1],
                                                   'authorship_G1': authorship[0], 'authorship_G2': authorship[1]})
    log(f'ragged projection matches published code (max abs diff): {checked}')

    support, store, identity_error = project(data, idata, args.chunk, log, args.max_draws)
    draws = len(next(iter(store.values())))
    log(f'max |P_AI - root_ai_component - nonroot_component| over sets and draws: {identity_error:.3e}')
    if identity_error > 1e-10:
        raise RuntimeError('Decomposition identity failed')

    from scripts.diagnostics.review_choice_models import reported_rq1_estimands, rq1_input_paths
    admitted_path = reported_rq1_estimands(args.rq1_dir)
    reproduction = reproduction_table(store, pd.read_csv(admitted_path))
    log('reproduction of admitted projections:\n' + reproduction.to_string(index=False))
    if args.max_draws is None and reproduction.max_abs_difference.max() > 1e-9:
        raise RuntimeError('Admitted projections were not reproduced')

    tables = {
        'projections': projection_table(store, support, draws),
        'decomposition': decomposition_table(store, support, draws),
        'support': support_table(support),
        'reproduction_check': reproduction,
    }
    written = {}
    for name, table in tables.items():
        path = out / 'tables' / f'chb_prereview_{ITEM}_{name}.csv'
        table.to_csv(path, index=False)
        written[path.name] = sha256(path)
    np.savez_compressed(out / 'outputs/draw_level_set_averages.npz',
                        **{'|'.join(k): v for k, v in store.items()})
    contrasts = tables['projections']
    log('non-root projections:\n' + contrasts[contrasts.group.eq('G2-G1')][
        ['projection', 'set_scope', 'estimand', 'posterior_mean', 'posterior_median', 'ci_low', 'ci_high',
         'posterior_probability_gt_zero']].to_string(index=False))

    code = {p: sha256(Path(__file__).resolve().parents[2] / p) for p in (
        f'scripts/diagnostics/chb_prereview_{ITEM}.py', 'scripts/diagnostics/review_choice_models_common.py',
        'scripts/diagnostics/review_choice_models.py', 'scripts/publication_analysis/model.py',
        'scripts/publication_analysis/terms.py')}
    rq1 = args.rq1_dir
    save_json(out / 'run.json', {
        'run_id': RUN_ID, 'item': ITEM,
        'purpose': 'CHB pre-submission review A19: RQ1 AI-target probability conditional on a non-root target, '
                   'projected from the admitted M1 posterior (no refit), with the exact root/non-root '
                   'decomposition of P_AI',
        'host': socket.gethostname(), 'threads': os.environ.get('OMP_NUM_THREADS'), 'nice': os.nice(0),
        'command': ' '.join(['python -m scripts.diagnostics.chb_prereview_a19_rq1_nonroot', *sys.argv[1:]]),
        'started': started, 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'code_sha256': code,
        'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in rq1_input_paths(rq1).items()},
        'source_run': SOURCE_RUN, 'package_versions': package_versions(),
        'diagnostics': {'check_against_original_max_abs_diff': checked,
                        'decomposition_identity_max_abs_error': identity_error,
                        'reproduction_max_abs_difference': float(reproduction.max_abs_difference.max()),
                        'posterior_draws': draws, 'peak_rss_gib': peak_rss_gib()},
        'sign_convention': SIGN,
        'tables_sha256': written,
    })
    log(f'{RUN_ID}/{ITEM} done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
