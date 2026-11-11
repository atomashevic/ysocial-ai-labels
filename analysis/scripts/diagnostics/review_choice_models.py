"""Review diagnostics for the RQ1 M1 and RQ2 choice models (no model fitting).

Run ``review_choice_models_20261015``. Every stage reads admitted posteriors
and model frames (on the private host) and writes aggregate CSV tables only:

* ``rq1``: reproduces the Primary and Authorship-term-only projections and
  decomposes their difference by community-specific control term; summarizes
  M1 parameters, sampler settings and candidate composition.
* ``forward``: per-fold M1-versus-M0 forward-validation summaries.
* ``rq2``: RQ2 history coefficients, AI and participant-account projections
  under alternative weightings, the count/recency parameterization, the
  projection-opportunity rule and same-thread/direct-child composition.
* ``split``: same-thread and other-thread projections from the existing
  (unadmitted) split-history posterior ``rq2_cross_thread_20260915``.
* ``reactions``: availability of strictly prior-hour reaction scores.

Example (one core)::

    python -m scripts.diagnostics.review_choice_models --stage all --out OUT \
        --rq1-dir RQ1_OUTPUTS --rq2-dir RQ2_OUTPUTS --split-dir SPLIT_RUN \
        --events PROCESSED/snapshot_events.parquet
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

from scripts.diagnostics.review_choice_models_common import limit_threads

limit_threads(int(os.environ.get('REVIEW_THREADS', '1')))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.review_choice_models_common import (  # noqa: E402
    Posterior, Ragged, RunLog, account_changes, account_layout, arm_frames,
    check_against_original, peak_rss_gib, save_json, sha256, summarize, weighted_mean,
)

RUN_ID = 'review_choice_models_20261015'
M1_SHA = '17dc31e01eff3e508c9d440055c5611834cbf6914c85f05641ee082a052b10ad'
# The same posterior draws as deposited in the Zenodo record (posterior group only, without the
# pointwise log-likelihood): 08_analysis_ready/further_checks/a19_a21_rq1_m1/models/M1.nc.
PUBLIC_RECORD = Path(__file__).resolve().parents[3] / 'data/08_analysis_ready'
M1_DEPOSITED = PUBLIC_RECORD / 'further_checks/a19_a21_rq1_m1/M1.nc'
M1_PUBLIC_SHA = '06a676a7705c81ca177739f880fe8b90417cbcc2b059f3f82a5ef346efa8fecb'
RQ2_SHA = '1c9024b62823473ba868e5b0e795aca771fc80c13b737d4a6aa3811b52ede324'
CONTROLS = {
    'age': 'g2_x_z_log_candidate_age',
    'age_squared': 'g2_x_z_log_candidate_age_squared',
    'depth': 'g2_x_z_log_candidate_depth',
    'root': 'g2_x_candidate_is_root',
}
AI_X = 'g2_x_candidate_ai'


def _load_idata(path: Path, expected: str | None, log: RunLog):
    import arviz as az
    digest = sha256(path)
    log(f'{path.name} sha256 {digest}')
    if expected and digest not in ((expected, M1_PUBLIC_SHA) if expected == M1_SHA else (expected,)):
        raise RuntimeError(f'{path} does not match the admitted posterior hash')
    idata = az.from_netcdf(path)
    idata.posterior.load()
    return idata


def reported_rq1_estimands(rq1_dir: Path) -> Path:
    """rq1_model_estimands.csv of the reported fit: in the fit folder, else in the public record."""
    path = rq1_dir / 'tables/rq1_model_estimands.csv'
    return path if path.exists() else PUBLIC_RECORD / 'results_tables/rq1_model_estimands.csv'


def rq1_input_paths(rq1_dir: Path) -> dict[str, Path]:
    """Files a projection from the reported M1 fit reads, for the run record."""
    posterior = rq1_dir / 'models/M1.nc'
    return {'m1_posterior': posterior if posterior.exists() else M1_DEPOSITED,
            'model_frame': rq1_dir / 'tables/model_frame.parquet',
            'term_manifest': rq1_dir / 'tables/model_term_manifest.parquet',
            'admitted_estimands': reported_rq1_estimands(rq1_dir)}


def load_rq1(rq1_dir: Path, log: RunLog):
    from scripts.publication_analysis.model import prepare_model_data
    from scripts.publication_analysis.terms import included_terms_for_model
    frame = pd.read_parquet(rq1_dir / 'tables/model_frame.parquet')
    manifest = pd.read_parquet(rq1_dir / 'tables/model_term_manifest.parquet')
    terms = included_terms_for_model('M1', manifest)
    data = prepare_model_data(frame, tuple(t for t in terms if t != 'candidate_target_is_ai'),
                              ai_term='candidate_target_is_ai')
    # With tables from prepare-choice on the public inputs, the deposited posterior is read.
    idata = _load_idata(rq1_input_paths(rq1_dir)['m1_posterior'], M1_SHA, log)
    if data.n_choices != 2143 or int(data.mask.sum()) != 17489:
        raise RuntimeError('M1 frame does not match the admitted denominators')
    return frame, manifest, data, idata


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------

def rq1_projections(data, idata, chunk: int, log: RunLog, limit: int | None = None):
    """Per-draw AI-target probabilities for every subset of toggled G2 terms."""
    from scripts.publication_analysis.model import _common_arm_frames, _direct_ai_arm_frames
    primary = arm_frames(data, [AI_X, *CONTROLS.values()])
    authorship = arm_frames(data, [AI_X])
    for mine, original in ((primary, _common_arm_frames(data)), (authorship, _direct_ai_arm_frames(data))):
        if not all(np.array_equal(a, b) for a, b in zip(mine, original)):
            raise RuntimeError('Projection frames differ from the published frame builders')
    checked = check_against_original(idata, data, {'primary_G1': primary[0], 'primary_G2': primary[1],
                                                   'authorship_G1': authorship[0],
                                                   'authorship_G2': authorship[1]})
    log(f'ragged projection matches published code: {checked}')

    names = list(CONTROLS)
    scenarios: dict[str, tuple[str, ...]] = {}
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            scenarios['ai+' + '+'.join(subset) if subset else 'ai'] = (AI_X, *(CONTROLS[n] for n in subset))
    scenarios['controls_only'] = tuple(CONTROLS.values())
    for name in names:
        scenarios[f'control_only_{name}'] = (CONTROLS[name],)
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    frames = {key: tuple(ragged.rows(f) for f in arm_frames(data, toggled)) for key, toggled in scenarios.items()}
    observed_rows = ragged.rows(data.x)
    total = posterior.n if limit is None else min(limit, posterior.n)
    out = {key: (np.empty(total), np.empty(total)) for key in scenarios}
    observed = {group: np.empty(total) for group in ('G1', 'G2')}
    derivatives = {name: np.empty(total) for name in names}
    groups = np.asarray(data.choice_groups)
    base_index = {name: data.terms.index(term.removeprefix('g2_x_')) for name, term in CONTROLS.items()}
    for draws in posterior.chunks(chunk, limit):
        random = ragged.random_part(posterior, draws)
        for key, (g1_rows, g2_rows) in frames.items():
            out[key][0][draws] = ragged.ai_probability(ragged.utility(posterior, draws, g1_rows, random)).mean(axis=1)
            out[key][1][draws] = ragged.ai_probability(ragged.utility(posterior, draws, g2_rows, random)).mean(axis=1)
        fitted = ragged.ai_probability(ragged.utility(posterior, draws, observed_rows, random))
        for group in ('G1', 'G2'):
            observed[group][draws] = fitted[:, groups == group].mean(axis=1)
        # dP_AI/d beta_c under the G1 specification, averaged over pooled sets:
        # sum_j p_j 1[AI_j](x_jc - sum_k p_k x_kc) = Cov_p(1[AI], x_c) per set.
        g1_rows = frames['ai+age+age_squared+depth+root'][0]
        probability = ragged.probabilities(ragged.utility(posterior, draws, g1_rows, random))
        for name, index in base_index.items():
            x = g1_rows[:, index]
            mean_x = np.add.reduceat(probability * x, ragged.starts, axis=1)
            p_ai = np.add.reduceat(probability * ragged.ai, ragged.starts, axis=1)
            mean_ai_x = np.add.reduceat(probability * ragged.ai * x, ragged.starts, axis=1)
            derivatives[name][draws] = (mean_ai_x - p_ai * mean_x).mean(axis=1)
        log(f'rq1 projections draws {draws.start}-{draws.stop}; peak RSS {peak_rss_gib():.2f} GiB')
    return scenarios, out, observed, derivatives


def rq1_tables(data, idata, chunk, log, limit=None):
    scenarios, out, observed, derivatives = rq1_projections(data, idata, chunk, log, limit)
    rows = []
    labels = {'ai': 'Authorship term only', 'ai+age+age_squared+depth+root': 'Primary (all G2 terms)'}
    for key, toggled in scenarios.items():
        g1, g2 = out[key]
        for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
            rows.append(summarize(values, projection=key, label=labels.get(key, ''), group=group,
                                  toggled_terms=';'.join(toggled), weighting='reply_event_weighted',
                                  standardization='pooled 2143 choice sets, 17489 candidates'))
    for group, values in observed.items():
        rows.append(summarize(values, projection='observed_sets_fitted', label='Observed choice sets, fitted',
                              group=group, toggled_terms='', weighting='reply_event_weighted',
                              standardization='own-community choice sets'))
    projections = pd.DataFrame(rows)

    contrast = {key: out[key][1] - out[key][0] for key in scenarios}
    names = list(CONTROLS)
    gap = contrast['ai+' + '+'.join(names)] - contrast['ai']
    shap_rows = []
    for name in names:
        others = [n for n in names if n != name]
        phi = np.zeros_like(gap)
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(len(names) - size - 1) / math.factorial(len(names))
            for subset in itertools.combinations(others, size):
                with_c = sorted([*subset, name], key=names.index)
                key_with = 'ai+' + '+'.join(with_c)
                key_without = 'ai+' + '+'.join(subset) if subset else 'ai'
                phi += weight * (contrast[key_with] - contrast[key_without])
        single = contrast['ai+' + name] - contrast['ai']
        drop = contrast['ai+' + '+'.join(names)] - contrast['ai+' + '+'.join(others)]
        shap_rows.append(summarize(phi, control=name, quantity='shapley_share_of_primary_minus_authorship_pp'))
        shap_rows.append(summarize(single, control=name, quantity='add_to_authorship_only_change_pp'))
        shap_rows.append(summarize(drop, control=name, quantity='drop_from_primary_change_pp'))
    age_pair = contrast['ai+age+age_squared'] - contrast['ai']
    shap_rows.append(summarize(age_pair, control='age+age_squared', quantity='add_to_authorship_only_change_pp'))
    shap_rows.append(summarize(gap, control='all', quantity='primary_minus_authorship_only_pp'))
    beta = Posterior(idata, data).beta[:len(gap)]
    for name, values in derivatives.items():
        interaction = beta[:, data.terms.index(CONTROLS[name])]
        shap_rows.append(summarize(values, control=name, quantity='dPAI_dbeta_G1_spec_mean_over_sets'))
        shap_rows.append(summarize(values * interaction, control=name,
                                   quantity='first_order_change_dPAI_times_G2_interaction'))
    decomposition = pd.DataFrame(shap_rows)
    return projections, decomposition


def rq1_parameters(data, idata):
    post = idata.posterior
    beta = post['beta']
    rows = []
    for term in beta.coords['term'].values:
        rows.append(summarize(beta.sel(term=term).values, parameter=f'beta[{term}]'))
    for base, interaction in [(t.removeprefix('g2_x_'), t) for t in CONTROLS.values()] + [('candidate_target_is_ai', AI_X)]:
        rows.append(summarize((beta.sel(term=base) + beta.sel(term=interaction)).values,
                              parameter=f'G2 slope {base} (= beta[{base}] + beta[{interaction}])'))
    for author_type in post['sigma_receiver_effect'].coords['author_type'].values:
        rows.append(summarize(post['sigma_receiver_effect'].sel(author_type=author_type).values,
                              parameter=f'sigma_receiver_effect[{author_type}]'))
    rows.append(summarize(post['sigma_actor_ai_effect'].values, parameter='sigma_actor_ai_effect'))
    actor = post['actor_ai_effect']
    groups = np.asarray([str(label).split('::')[0] for label in actor.coords['actor'].values])
    for group in ('G1', 'G2'):
        rows.append(summarize(actor.values[..., groups == group].sum(axis=-1),
                              parameter=f'sum of participant AI slopes within {group} (constraint check)'))
    stats = idata.sample_stats
    sampler = {
        'inference_library': stats.attrs.get('inference_library'),
        'inference_library_version': stats.attrs.get('inference_library_version'),
        'sampler_variables': ';'.join(sorted(stats.data_vars)),
        'chains': int(post.sizes['chain']), 'retained_draws_per_chain': int(post.sizes['draw']),
        'tuning_steps': int(stats.attrs.get('tuning_steps', -1)),
        'sampling_time_seconds': float(stats.attrs.get('sampling_time', float('nan'))),
        'divergences': int(stats['diverging'].sum()),
        'max_tree_depth': int(stats['tree_depth'].max()),
        'reached_max_treedepth': int(stats['reached_max_treedepth'].sum()),
        'mean_step_size': float(stats['step_size'].mean()),
        'mean_acceptance_rate': float(stats['acceptance_rate'].mean()),
        'receivers': int(post.sizes['receiver']), 'participants': int(post.sizes['actor']),
    }
    return pd.DataFrame(rows), pd.DataFrame([sampler])


def rq1_composition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = frame.copy()
    f['author'] = np.where(f.candidate_target_is_ai.astype(int).eq(1), 'AI', 'participant')
    f['log_age'] = np.log1p(f.candidate_age_hours.clip(lower=0))
    f['log_depth'] = np.log1p(f.candidate_depth.clip(lower=0))
    f['chosen'] = f.chosen.astype(int)
    rows = []
    for (group, author), part in f.groupby(['group', 'author']):
        rows.append(dict(group=group, candidate_author=author, candidates=len(part),
                         choice_sets=part.choice_set_id.nunique(), chosen=int(part.chosen.sum()),
                         share_of_candidates=len(part) / (f.group == group).sum(),
                         median_age_valid_hours=float(part.candidate_age_hours.median()),
                         q25_age_valid_hours=float(part.candidate_age_hours.quantile(.25)),
                         q75_age_valid_hours=float(part.candidate_age_hours.quantile(.75)),
                         mean_log1p_age=float(part.log_age.mean()), mean_depth=float(part.candidate_depth.mean()),
                         median_depth=float(part.candidate_depth.median()),
                         share_root=float(part.candidate_is_root.astype(int).mean()),
                         roots=int(part.candidate_is_root.astype(int).sum()),
                         chosen_share_root=float(part.loc[part.chosen.eq(1), 'candidate_is_root'].astype(int).mean())
                         if part.chosen.sum() else float('nan'),
                         selection_rate=float(part.chosen.mean())))
    composition = pd.DataFrame(rows)

    # Within-set contrasts: conditional logit identifies within-set variation only.
    sets = []
    for (group, choice), part in f.groupby(['group', 'choice_set_id'], sort=False):
        ai = part[part.author.eq('AI')]
        hu = part[part.author.eq('participant')]
        sets.append(dict(group=group, mixed=bool(len(ai) and len(hu)), n=len(part), n_ai=len(ai),
                         root_ai=int((ai.candidate_is_root.astype(int) == 1).any()),
                         root_present=int(part.candidate_is_root.astype(int).eq(1).any()),
                         d_log_age=ai.log_age.mean() - hu.log_age.mean() if len(ai) and len(hu) else np.nan,
                         d_log_depth=ai.log_depth.mean() - hu.log_depth.mean() if len(ai) and len(hu) else np.nan,
                         d_root=ai.candidate_is_root.astype(int).mean() - hu.candidate_is_root.astype(int).mean()
                         if len(ai) and len(hu) else np.nan,
                         chosen_ai=int(part.loc[part.chosen.eq(1), 'author'].eq('AI').any()),
                         chosen_root=int(part.loc[part.chosen.eq(1), 'candidate_is_root'].astype(int).eq(1).any())))
    sets = pd.DataFrame(sets)
    within = []
    for group, part in sets.groupby('group'):
        mixed = part[part.mixed]
        within.append(dict(group=group, choice_sets=len(part), mixed_sets=len(mixed),
                           sets_with_root_candidate=int(part.root_present.sum()),
                           sets_with_root_candidate_ai_authored=int(part.root_ai.sum()),
                           mean_candidates=float(part.n.mean()), mean_ai_share=float((part.n_ai / part.n).mean()),
                           observed_ai_target_share=float(part.chosen_ai.mean()),
                           observed_root_target_share=float(part.chosen_root.mean()),
                           mixed_mean_ai_minus_participant_log1p_age=float(mixed.d_log_age.mean()),
                           mixed_mean_ai_minus_participant_log1p_depth=float(mixed.d_log_depth.mean()),
                           mixed_mean_ai_minus_participant_root_share=float(mixed.d_root.mean())))
    return composition, pd.DataFrame(within)


def stage_rq1(args, log):
    frame, manifest, data, idata = load_rq1(args.rq1_dir, log)
    out = args.out
    manifest[['term', 'family', 'status', 'raw_mean', 'raw_sd']].to_csv(out / 'rq1_term_scaling.csv', index=False)
    parameters, sampler = rq1_parameters(data, idata)
    parameters.to_csv(out / 'rq1_parameters.csv', index=False)
    sampler.to_csv(out / 'rq1_sampler.csv', index=False)
    composition, within = rq1_composition(frame)
    composition.to_csv(out / 'rq1_candidate_composition.csv', index=False)
    within.to_csv(out / 'rq1_within_set_composition.csv', index=False)
    projections, decomposition = rq1_tables(data, idata, args.chunk, log, args.max_draws)
    projections.to_csv(out / 'rq1_projection_decomposition.csv', index=False)
    decomposition.to_csv(out / 'rq1_control_attribution.csv', index=False)
    admitted = pd.read_csv(args.rq1_dir / 'tables/rq1_model_estimands.csv')
    check = []
    for estimand, key in (('common_opportunity_ai_choice_probability', 'ai+age+age_squared+depth+root'),
                          ('direct_G2xAI_common_opportunity_probability', 'ai')):
        for group in ('G1', 'G2'):
            mine = projections[projections.projection.eq(key) & projections.group.eq(group)].iloc[0]
            ref = admitted[admitted.estimand.eq(estimand) & admitted.group.eq(group)].iloc[0]
            check.append(dict(estimand=estimand, group=group, reproduced_median=mine.posterior_median,
                              admitted_median=ref.posterior_median,
                              abs_difference=abs(mine.posterior_median - ref.posterior_median)))
    pd.DataFrame(check).to_csv(out / 'rq1_reproduction_check.csv', index=False)
    log('rq1 reproduction: ' + pd.DataFrame(check).to_string(index=False))


# ---------------------------------------------------------------------------
# Forward validation
# ---------------------------------------------------------------------------

def stage_forward(args, log):
    tables = args.rq1_dir / 'tables'
    scores = pd.read_csv(tables / 'rq1_forward_validation_scores.csv')
    predictions = pd.read_parquet(tables / 'rq1_forward_validation_predictions.parquet')
    comparisons = pd.read_csv(tables / 'rq1_forward_validation_comparisons.csv')
    frame = pd.read_parquet(tables / 'model_frame.parquet')
    from scripts.publication_analysis.validation import rolling_origin_assignments
    assign = rolling_origin_assignments(frame)
    times = frame.drop_duplicates('choice_set_id').set_index('choice_set_id').event_timestamp
    days = frame.drop_duplicates('choice_set_id').set_index('choice_set_id').event_valid_day_index
    rows = []
    for (fold, group, role), part in assign.groupby(['fold', 'group', 'role']):
        ids = part.choice_set_id
        rows.append(dict(fold=fold, group=group, role=role, choice_sets=len(ids),
                         first_event=str(pd.to_datetime(times.loc[ids]).min()),
                         last_event=str(pd.to_datetime(times.loc[ids]).max()),
                         first_valid_day_index=int(days.loc[ids].min()), last_valid_day_index=int(days.loc[ids].max())))
    folds = pd.DataFrame(rows)
    unseen = (predictions.groupby(['fold', 'model'])
              [['unseen_receiver_levels', 'unseen_actor_levels', 'unseen_topic_levels']].max().reset_index())
    m1 = predictions[predictions.model.eq('M1')].set_index(['fold', 'choice_set_id'])
    m0 = predictions[predictions.model.eq('M0')].set_index(['fold', 'choice_set_id'])
    paired = m1[['group', 'actor_label', 'log_score', 'chosen_is_ai']].join(
        m0[['log_score']], rsuffix='_m0')
    paired['delta'] = paired.log_score - paired.log_score_m0
    by_group = (paired.reset_index().groupby(['fold', 'group'])
                .agg(test_choice_sets=('delta', 'size'), elpd_difference=('delta', 'sum')).reset_index())
    pooled_group = paired.reset_index().groupby('group').agg(
        test_choice_sets=('delta', 'size'), elpd_difference=('delta', 'sum')).reset_index().assign(fold='ALL')
    by_type = (paired.reset_index().groupby(['fold', 'chosen_is_ai'])
               .agg(test_choice_sets=('delta', 'size'), elpd_difference=('delta', 'sum')).reset_index())
    folds.to_csv(args.out / 'forward_fold_windows.csv', index=False)
    unseen.to_csv(args.out / 'forward_unseen_levels.csv', index=False)
    pd.concat([by_group, pooled_group]).to_csv(args.out / 'forward_elpd_by_group.csv', index=False)
    by_type.to_csv(args.out / 'forward_elpd_by_selected_type.csv', index=False)
    comparisons.to_csv(args.out / 'forward_admitted_comparisons.csv', index=False)
    scores.to_csv(args.out / 'forward_admitted_scores.csv', index=False)
    log('forward stage written')


# ---------------------------------------------------------------------------
# RQ2
# ---------------------------------------------------------------------------

def load_rq2(rq2_dir: Path, log: RunLog):
    from scripts.publication_analysis.model import prepare_model_data
    from scripts.publication_analysis.terms import included_terms_for_model
    frame = pd.read_parquet(rq2_dir / 'tables/rq2_model_frame.parquet')
    manifest = pd.read_parquet(rq2_dir / 'tables/rq2_model_term_manifest.parquet')
    terms = included_terms_for_model('RQ2', manifest)
    data = prepare_model_data(frame, tuple(t for t in terms if t != 'candidate_target_is_ai'),
                              ai_term='candidate_target_is_ai')
    idata = _load_idata(rq2_dir / 'models/M2_RQ2.nc', RQ2_SHA, log)
    return frame, manifest, data, idata


BASES = {
    'received_ai_to_participant': ('z_log_reverse_contact_count', 'z_recent_reverse_contact'),
    'sent_participant_to_ai': ('z_log_prior_contact_count', 'z_recent_prior_contact'),
}


def _pair(manifest, term, raw_low, raw_high):
    row = manifest.set_index('term').loc[term]
    return ((raw_low - row.raw_mean) / row.raw_sd, (raw_high - row.raw_mean) / row.raw_sd)


def rq2_scenarios(manifest, data, kind: str, group: str, author: str):
    """Term values (low, high) for the focal account's rows."""
    recent = math.exp(-1.0)
    raw = {
        'zero_to_one': {'count': (0.0, math.log(2.0)), 'recency': (0.0, recent)},
        'count_only': {'count': (0.0, math.log(2.0)), 'recency': (0.0, 0.0)},
        'recency_only': {'count': (0.0, 0.0), 'recency': (0.0, recent)},
        'one_to_two': {'count': (math.log(2.0), math.log(3.0)), 'recency': (recent, recent)},
    }[kind]
    result = {}
    for direction, (count, recency) in BASES.items():
        values = {}
        for base, part in ((count, 'count'), (recency, 'recency')):
            pair = _pair(manifest, base, *raw[part])
            values[base] = pair
            if author == 'AI':
                values['candidate_ai_x_' + base] = pair
                values['g2_x_candidate_ai_x_' + base] = pair if group == 'G2' else (0.0, 0.0)
            elif author == 'AI_general_slopes':
                # Hold the AI-specific terms at their low value in both states, so
                # only the general (participant-account) history slopes change.
                values['candidate_ai_x_' + base] = (pair[0], pair[0])
                values['g2_x_candidate_ai_x_' + base] = (pair[0], pair[0]) if group == 'G2' else (0.0, 0.0)
        result[direction] = {data.terms.index(term): value for term, value in values.items()}
    return result


def rq2_parameters(idata):
    beta = idata.posterior['beta']
    rows = []
    for term in beta.coords['term'].values:
        rows.append(summarize(beta.sel(term=term).values, parameter=f'beta[{term}]'))
    for direction, bases in BASES.items():
        for base in bases:
            general = beta.sel(term=base).values
            ai = beta.sel(term='candidate_ai_x_' + base).values
            g2 = beta.sel(term='g2_x_candidate_ai_x_' + base).values
            for label, values in (('participant-account (general) slope', general),
                                  ('AI-specific deviation', ai), ('G2 x AI deviation', g2),
                                  ('AI-account slope, G1', general + ai),
                                  ('AI-account slope, G2', general + ai + g2)):
                rows.append(summarize(values, parameter=f'{base}: {label}', direction=direction))
        count, recency = bases
        for group, extra in (('G1', 0.0), ('G2', 1.0)):
            c = (beta.sel(term=count) + beta.sel(term='candidate_ai_x_' + count)
                 + extra * beta.sel(term='g2_x_candidate_ai_x_' + count)).values.ravel()
            r = (beta.sel(term=recency) + beta.sel(term='candidate_ai_x_' + recency)
                 + extra * beta.sel(term='g2_x_candidate_ai_x_' + recency)).values.ravel()
            rows.append(dict(parameter=f'posterior correlation of AI-account count and recency slopes, {group}',
                             direction=direction, posterior_mean=float(np.corrcoef(c, r)[0, 1])))
        c = beta.sel(term=count).values.ravel()
        r = beta.sel(term=recency).values.ravel()
        rows.append(dict(parameter='posterior correlation of general count and recency slopes',
                         direction=direction, posterior_mean=float(np.corrcoef(c, r)[0, 1])))
    for name in ('sigma_receiver_effect', 'sigma_actor_ai_effect', 'sigma_topic_ai_effect'):
        values = idata.posterior[name]
        if 'author_type' in values.dims:
            for t in values.coords['author_type'].values:
                rows.append(summarize(values.sel(author_type=t).values, parameter=f'{name}[{t}]'))
        else:
            rows.append(summarize(values.values, parameter=name))
    return pd.DataFrame(rows)


def rq2_linear_components(idata, manifest):
    beta = idata.posterior['beta']
    recent = math.exp(-1.0)
    rows = []
    for direction, (count, recency) in BASES.items():
        dc = np.diff(_pair(manifest, count, 0.0, math.log(2.0)))[0]
        dr = np.diff(_pair(manifest, recency, 0.0, recent))[0]
        d12 = np.diff(_pair(manifest, count, math.log(2.0), math.log(3.0)))[0]
        for group, extra in (('G1', 0.0), ('G2', 1.0)):
            for account, ai in (('AI', 1.0), ('participant', 0.0)):
                bc = (beta.sel(term=count) + ai * beta.sel(term='candidate_ai_x_' + count)
                      + ai * extra * beta.sel(term='g2_x_candidate_ai_x_' + count)).values
                br = (beta.sel(term=recency) + ai * beta.sel(term='candidate_ai_x_' + recency)
                      + ai * extra * beta.sel(term='g2_x_candidate_ai_x_' + recency)).values
                for part, values in (('count component of 0->1 shift', bc * dc),
                                     ('recency component of 0->1 shift', br * dr),
                                     ('total 0->1 shift', bc * dc + br * dr),
                                     ('1->2 shift (count only; recency fixed at 24 h)', bc * d12)):
                    rows.append(summarize(values, direction=direction, group=group, account=account,
                                          quantity=part, scale='linear predictor (log odds of the account vs '
                                                               'competitors)',
                                          standardized_step=float({'count component of 0->1 shift': dc,
                                                                   'recency component of 0->1 shift': dr,
                                                                   '1->2 shift (count only; recency fixed at 24 h)': d12
                                                                   }.get(part, float('nan')))))
    return pd.DataFrame(rows)


def rq2_projection_table(data, idata, manifest, chunk, log, limit=None, tag='rq2_admitted'):
    from scripts.publication_analysis.model import _common_arm_frames
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    g1, g2 = _common_arm_frames(data)
    check = check_against_original(idata, data, {'G1': g1, 'G2': g2})
    log(f'{tag}: ragged utilities match published code: {check}')
    arms = {'G1': ragged.rows(g1), 'G2': ragged.rows(g2)}
    layouts = {'AI': account_layout(data, ragged, 'AI'), 'participant': account_layout(data, ragged, 'H')}
    for key, layout in layouts.items():
        log(f'{tag}: {key} opportunities {len(layout["choice"])} from '
            f'{len(np.unique(layout["choice"]))} replies')
    specs = [('AI', 'AI', kind) for kind in ('zero_to_one', 'count_only', 'recency_only', 'one_to_two')]
    specs += [('AI', 'AI_general_slopes', 'zero_to_one'), ('participant', 'participant', 'zero_to_one'),
              ('participant', 'participant', 'one_to_two')]
    weightings = {'equal_opportunity': None, 'equal_reply': 'choice', 'equal_participant': 'actor'}
    total = posterior.n if limit is None else min(limit, posterior.n)
    store = {}
    for draws in posterior.chunks(chunk, limit):
        random = ragged.random_part(posterior, draws)
        for group, rows in arms.items():
            utility = ragged.utility(posterior, draws, rows, random)
            for layout_key, author, kind in specs:
                layout = layouts[layout_key]
                scen = rq2_scenarios(manifest, data, kind, group, 'H' if author == 'participant' else author)
                for direction, values in scen.items():
                    indexes = list(values)
                    changes = account_changes(utility, posterior.beta[draws], rows, layout, indexes,
                                              [values[i][0] for i in indexes], [values[i][1] for i in indexes])
                    for weighting, key in weightings.items():
                        k = (author, kind, direction, group, weighting)
                        store.setdefault(k, np.empty(total))[draws] = weighted_mean(
                            changes, None if key is None else layout[key])
        log(f'{tag} projections draws {draws.start}-{draws.stop}; peak RSS {peak_rss_gib():.2f} GiB')
    rows = []
    for (author, kind, direction, group, weighting), values in store.items():
        layout = layouts['participant' if author == 'participant' else 'AI']
        rows.append(summarize(values, account=author, scenario=kind, direction=direction, group=group,
                              weighting=weighting, opportunities=len(layout['choice']),
                              replies=len(np.unique(layout['choice'])), participants=len(np.unique(layout['actor']))))
        if group == 'G2':
            other = store[(author, kind, direction, 'G1', weighting)]
            rows.append(summarize(values - other, account=author, scenario=kind, direction=direction,
                                  group='G2-G1', weighting=weighting, opportunities=len(layout['choice']),
                                  replies=len(np.unique(layout['choice'])),
                                  participants=len(np.unique(layout['actor']))))
    return pd.DataFrame(rows), layouts


def rq2_opportunity_rule(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f['is_ai'] = f.candidate_author_type.astype(str).str.upper().eq('AI')
    sets = f.groupby(['group', 'choice_set_id']).agg(
        candidates=('candidate_content_id', 'size'), authors=('candidate_author_id', 'nunique'),
        ai_candidates=('is_ai', 'sum'),
        ai_authors=('candidate_author_id', lambda s: s[f.loc[s.index, 'is_ai']].nunique())).reset_index()
    sets['category'] = np.select(
        [sets.ai_candidates.eq(0), sets.authors.eq(1)],
        ['no AI-authored candidate', 'AI candidates from one account only (no competing account)'],
        'eligible: AI account with a competing account')
    sets['opportunities'] = np.where(sets.category.str.startswith('eligible'), sets.ai_authors, 0)
    rows = []
    for (group, category), part in sets.groupby(['group', 'category']):
        rows.append(dict(group=group, category=category, replies=len(part),
                         opportunities=int(part.opportunities.sum()),
                         single_author_sets=int(part.authors.eq(1).sum())))
    for category, part in sets.groupby('category'):
        rows.append(dict(group='ALL', category=category, replies=len(part),
                         opportunities=int(part.opportunities.sum()),
                         single_author_sets=int(part.authors.eq(1).sum())))
    eligible = sets[sets.category.str.startswith('eligible')]
    for group, part in list(eligible.groupby('group')) + [('ALL', eligible)]:
        rows.append(dict(group=group, category='eligible: AI accounts per reply (median; max)',
                         replies=len(part), opportunities=int(part.opportunities.sum()),
                         median_accounts=float(part.ai_authors.median()), max_accounts=int(part.ai_authors.max())))
    return pd.DataFrame(rows)


def rq2_thread_composition(split_frame: pd.DataFrame, rq2_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ['group', 'choice_set_id', 'candidate_content_id']
    merged = rq2_frame[keys + ['candidate_author_id', 'candidate_author_type', 'actor_id', 'chosen',
                               'prior_human_to_author_count', 'prior_author_to_human_count']].merge(
        split_frame[keys + ['direct_reply', 'within_outgoing_count', 'cross_outgoing_count',
                            'within_incoming_count', 'cross_incoming_count', 'within_outgoing_hours',
                            'cross_outgoing_hours', 'within_incoming_hours', 'cross_incoming_hours']],
        on=keys, how='left', validate='one_to_one')
    if merged.direct_reply.isna().any():
        raise RuntimeError('Split-history frame does not cover the RQ2 frame')
    if not ((merged.within_outgoing_count + merged.cross_outgoing_count).eq(merged.prior_human_to_author_count).all()
            and (merged.within_incoming_count + merged.cross_incoming_count).eq(
                merged.prior_author_to_human_count).all()):
        raise RuntimeError('Split histories do not reproduce the RQ2 histories')
    merged['chosen'] = merged.chosen.astype(int)
    merged['author'] = np.where(merged.candidate_author_type.astype(str).str.upper().eq('AI'), 'AI', 'participant')
    # Candidate level: direct children of the focal participant's own contribution.
    cand = []
    for (group, author), part in merged.groupby(['group', 'author']):
        for direct, sub in part.groupby('direct_reply'):
            cand.append(dict(unit='candidate', group=group, account=author, direct_child=int(direct),
                             candidates=len(sub), choice_sets=sub.choice_set_id.nunique(),
                             selected=int(sub.chosen.sum()), selection_rate=float(sub.chosen.mean())))
    # Account level (choice x account), competing-account opportunities only.
    acc = merged.groupby(['group', 'choice_set_id', 'candidate_author_id']).agg(
        author=('author', 'first'), actor=('actor_id', 'first'), selected=('chosen', 'max'),
        direct_child=('direct_reply', 'max'),
        **{c: (c, 'first') for c in ['within_outgoing_count', 'cross_outgoing_count', 'within_incoming_count',
                                     'cross_incoming_count', 'within_outgoing_hours', 'cross_outgoing_hours',
                                     'within_incoming_hours', 'cross_incoming_hours']}).reset_index()
    authors = merged.groupby('choice_set_id').candidate_author_id.nunique()
    acc = acc[acc.choice_set_id.map(authors).gt(1)]
    rows = []
    for (group, author), part in list(acc.groupby(['group', 'author'])) + [
            (('ALL', a), p) for a, p in acc.groupby('author')]:
        for direction, prefix in (('received_account_to_participant', 'incoming'),
                                  ('sent_participant_to_account', 'outgoing')):
            within = part[f'within_{prefix}_count']
            cross = part[f'cross_{prefix}_count']
            positive = (within + cross).gt(0)
            wh = part[f'within_{prefix}_hours']
            ch = part[f'cross_{prefix}_hours']
            latest_within = positive & within.gt(0) & (cross.eq(0) | (wh <= ch))
            for label, mask in (('any prior contact', positive),
                                ('latest prior contact in current root thread', latest_within),
                                ('latest prior contact in another root thread', positive & ~latest_within),
                                ('current-thread contact only', within.gt(0) & cross.eq(0)),
                                ('other-thread contact only', within.eq(0) & cross.gt(0)),
                                ('both current- and other-thread contact', within.gt(0) & cross.gt(0)),
                                ('any prior contact and account content is a direct child',
                                 positive & part.direct_child.eq(1)),
                                ('no prior contact', ~positive)):
                sub = part[mask]
                rows.append(dict(unit='choice_by_account_opportunity', group=group, account=author,
                                 direction=direction, category=label, opportunities=len(sub),
                                 share_of_positive=(float(mask.sum() / positive.sum())
                                                    if positive.sum() and label != 'no prior contact' else np.nan),
                                 denominator_opportunities=len(part), denominator_positive=int(positive.sum()),
                                 selected=int(sub.selected.sum()),
                                 selection_rate=float(sub.selected.mean()) if len(sub) else np.nan))
        for direct, sub in part.groupby('direct_child'):
            rows.append(dict(unit='choice_by_account_opportunity', group=group, account=author,
                             direction='any', category=f'direct child of focal participant = {int(direct)}',
                             opportunities=len(sub), denominator_opportunities=len(part),
                             selected=int(sub.selected.sum()), selection_rate=float(sub.selected.mean())))
    return pd.DataFrame(cand), pd.DataFrame(rows)


def stage_rq2(args, log):
    frame, manifest, data, idata = load_rq2(args.rq2_dir, log)
    out = args.out
    rq2_parameters(idata).to_csv(out / 'rq2_parameters.csv', index=False)
    rq2_linear_components(idata, manifest).to_csv(out / 'rq2_linear_predictor_components.csv', index=False)
    rq2_opportunity_rule(frame).to_csv(out / 'rq2_opportunity_rule.csv', index=False)
    manifest[['term', 'family', 'status', 'raw_mean', 'raw_sd']].to_csv(out / 'rq2_term_scaling.csv', index=False)
    terms = list(data.terms)
    x = frame[[t for t in terms if t.startswith('z_')]]
    x.corr().to_csv(out / 'rq2_history_term_correlations.csv')
    support = []
    for direction, (count, recency) in BASES.items():
        raw = {'received_ai_to_participant': 'prior_author_to_human_count',
               'sent_participant_to_ai': 'prior_human_to_author_count'}[direction]
        n = frame[raw]
        positive = n.gt(0)
        support.append(dict(direction=direction, candidate_rows=len(frame), positive_rows=int(positive.sum()),
                            positive_rows_with_count_one=int(n.eq(1).sum()),
                            share_of_positive_with_count_one=float(n.eq(1).sum() / positive.sum()),
                            corr_count_recency_all_rows=float(frame[count].corr(frame[recency])),
                            corr_count_recency_positive_rows=float(frame.loc[positive, count].corr(
                                frame.loc[positive, recency]))))
    pd.DataFrame(support).to_csv(out / 'rq2_count_recency_support.csv', index=False)
    projections, _ = rq2_projection_table(data, idata, manifest, args.chunk, log, args.max_draws)
    projections.to_csv(out / 'rq2_projections.csv', index=False)
    admitted = pd.read_csv(args.rq2_dir / 'tables/rq2_primary_estimands.csv')
    rows = []
    for direction, name in (('ai_to_human', 'received_ai_to_participant'), ('human_to_ai', 'sent_participant_to_ai')):
        for group in ('G1', 'G2', 'G2-G1'):
            ref = admitted[admitted.direction.eq(direction) & admitted.group.eq(group)].iloc[0]
            mine = projections[projections.account.eq('AI') & projections.scenario.eq('zero_to_one')
                               & projections.direction.eq(name) & projections.group.eq(group)
                               & projections.weighting.eq('equal_opportunity')].iloc[0]
            rows.append(dict(direction=name, group=group, reproduced_median=mine.posterior_median,
                             admitted_median=ref.posterior_median, reproduced_low=mine.ci_low,
                             admitted_low=ref.ci_low, abs_difference=abs(mine.posterior_median
                                                                         - ref.posterior_median)))
    pd.DataFrame(rows).to_csv(out / 'rq2_reproduction_check.csv', index=False)
    log('rq2 reproduction: ' + pd.DataFrame(rows).to_string(index=False))
    if args.split_dir:
        split = pd.read_parquet(args.split_dir / 'model_frame.parquet')
        candidate, account = rq2_thread_composition(split, frame)
        candidate.to_csv(out / 'rq2_direct_child_candidates.csv', index=False)
        account.to_csv(out / 'rq2_thread_composition.csv', index=False)


def stage_split(args, log):
    """Same- and other-thread projections from the existing split posterior."""
    from scripts.publication_analysis.model import _common_arm_frames, prepare_model_data
    meta = json.loads((args.split_dir / 'run.json').read_text())
    scales = json.loads((args.split_dir / 'scales.json').read_text())
    frame = pd.read_parquet(args.split_dir / 'model_frame.parquet')
    data = prepare_model_data(frame, meta['terms'])
    idata = _load_idata(args.split_dir / 'posterior.nc', None, log)
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    g1, g2 = _common_arm_frames(data)
    log(f'split: utilities match published code: {check_against_original(idata, data, {"G1": g1, "G2": g2})}')
    arms = {'G1': ragged.rows(g1), 'G2': ragged.rows(g2)}
    layout = account_layout(data, ragged, 'AI')
    bases = {'outgoing': ('z_log_prior_contact_count', 'z_recent_prior_contact'),
             'incoming': ('z_log_reverse_contact_count', 'z_recent_reverse_contact')}

    def scenario(scope, direction, group):
        values = {}
        for base, high in zip(bases[direction], (math.log(2.0), math.exp(-1.0))):
            name = base.replace('z_', f'z_{scope}_', 1)
            mean, sd = scales[name]['mean'], scales[name]['sd']
            pair = (-mean / sd, (high - mean) / sd)
            values[name] = pair
            values['candidate_ai_x_' + name] = pair
            values['g2_x_candidate_ai_x_' + name] = pair if group == 'G2' else (0.0, 0.0)
        return {data.terms.index(k): v for k, v in values.items()}

    total = posterior.n if args.max_draws is None else min(args.max_draws, posterior.n)
    store = {}
    for draws in posterior.chunks(args.chunk, args.max_draws):
        random = ragged.random_part(posterior, draws)
        for group, rows in arms.items():
            utility = ragged.utility(posterior, draws, rows, random)
            for scope in ('within', 'cross'):
                for direction in bases:
                    values = scenario(scope, direction, group)
                    idx = list(values)
                    changes = account_changes(utility, posterior.beta[draws], rows, layout, idx,
                                              [values[i][0] for i in idx], [values[i][1] for i in idx])
                    store.setdefault((scope, direction, group), np.empty(total))[draws] = changes.mean(axis=1)
        log(f'split projections draws {draws.start}-{draws.stop}')
    rows = []
    for (scope, direction, group), values in store.items():
        rows.append(summarize(values, scope=scope, direction=direction, group=group,
                              opportunities=len(layout['choice']), replies=len(np.unique(layout['choice']))))
        if group == 'G2':
            rows.append(summarize(values - store[(scope, direction, 'G1')], scope=scope, direction=direction,
                                  group='G2-G1', opportunities=len(layout['choice']),
                                  replies=len(np.unique(layout['choice']))))
    pd.DataFrame(rows).to_csv(args.out / 'rq2_split_projections.csv', index=False)
    beta = idata.posterior['beta']
    prows = [summarize(beta.sel(term=t).values, parameter=f'beta[{t}]') for t in beta.coords['term'].values
             if 'within' in str(t) or 'cross' in str(t)]
    pd.DataFrame(prows).to_csv(args.out / 'rq2_split_parameters.csv', index=False)


# ---------------------------------------------------------------------------
# Reaction-score feasibility
# ---------------------------------------------------------------------------

def stage_reactions(args, log):
    from scripts.publication_analysis import rq3
    from scripts.publication_analysis.data import _add_reaction_history
    events = pd.read_parquet(args.events)
    events['event_timestamp'] = pd.to_datetime(events['event_timestamp'], utc=True, errors='coerce')
    round_map = rq3.derive_round_timestamps(events, round_start_time='2026-05-09T16:00:00+02:00',
                                            first_experiment_round=2, verify_from='2026-05-11 08:00:00+02:00')
    reactions = events[events.event_type.isin(['upvote', 'downvote'])]
    log(f'reactions {len(reactions)}; with exact timestamp {int(reactions.event_timestamp.notna().sum())}; '
        f'with round {int(reactions["round"].notna().sum())}')
    frame = pd.read_parquet(args.rq1_dir / 'tables/model_frame.parquet')
    frame['event_timestamp'] = pd.to_datetime(frame.event_timestamp, utc=True)
    out_rows = []
    scored = {}
    for placement in ('round_end', 'round_start'):
        stamped = rq3.timestamp_reactions(events, round_map, placement=placement)
        stamped = stamped[stamped.event_type.isin(['upvote', 'downvote'])]
        scored[placement] = _add_reaction_history(frame.drop(columns=[c for c in frame.columns if 'reaction' in c
                                                                      and c.startswith('candidate_')],
                                                             errors='ignore'), stamped)
    strict = scored['round_end']
    loose = scored['round_start']
    strict['ai'] = np.where(strict.candidate_target_is_ai.astype(int).eq(1), 'AI', 'participant')
    same_hour = (loose.candidate_content_prior_upvotes + loose.candidate_content_prior_downvotes
                 - strict.candidate_content_prior_upvotes - strict.candidate_content_prior_downvotes)
    strict['same_hour_reactions'] = same_hour.to_numpy()
    for (group, author), part in strict.groupby(['group', 'ai']):
        votes = part.candidate_content_prior_upvotes + part.candidate_content_prior_downvotes
        out_rows.append(dict(group=group, candidate_author=author, candidates=len(part),
                             share_any_prior_hour_reaction=float(votes.gt(0).mean()),
                             share_any_prior_hour_downvote=float(part.candidate_content_prior_downvotes.gt(0).mean()),
                             mean_prior_hour_net_score=float(part.candidate_content_prior_reaction_score.mean()),
                             median_prior_hour_net_score=float(part.candidate_content_prior_reaction_score.median()),
                             mean_author_prior_hour_net_score=float(part.candidate_author_prior_reaction_score.mean()),
                             share_with_same_hour_reaction_excluded=float(part.same_hour_reactions.gt(0).mean()),
                             chosen_mean_prior_hour_net_score=float(
                                 part.loc[part.chosen.astype(int).eq(1), 'candidate_content_prior_reaction_score'].mean())))
    pd.DataFrame(out_rows).to_csv(args.out / 'reaction_score_feasibility.csv', index=False)
    round_map[['round', 'source', 'closed_form_match', 'within_one_hour', 'enforced']].groupby(
        ['source', 'closed_form_match', 'within_one_hour', 'enforced']).size().rename('rounds').reset_index().to_csv(
        args.out / 'reaction_round_map_summary.csv', index=False)
    log('reaction feasibility written')




# ---------------------------------------------------------------------------
# Plug-in ablation of the fixed AI terms in the reported forward folds
# ---------------------------------------------------------------------------

def stage_ablation(args, log):
    """Score each reported M1 fold posterior with and without its fixed AI terms.

    No refit: the fixed AI and G2-by-AI coefficients are set to zero in every
    draw while account intercepts and participant AI slopes keep their fitted
    values. It indicates how much held-out density the fixed AI terms carry
    inside M1; the refitted comparison is review_choice_models_isolated_cv.
    """
    import arviz as az
    from scripts.diagnostics.review_choice_models_isolated_cv import fold_data
    from scripts.publication_analysis.pipeline import (_heldout_choice_predictions, _prepared_data_hash,
                                                       _participant_clustered_total_se)
    from scripts.publication_analysis.validation import rolling_origin_assignments
    tables = args.rq1_dir / 'tables'
    frame = pd.read_parquet(tables / 'model_frame.parquet')
    admitted = pd.read_parquet(tables / 'rq1_forward_validation_predictions.parquet')
    assignments = rolling_origin_assignments(frame)
    rows, seed = [], 20260715
    for fold in (1, 2, 3):
        train, test = fold_data(frame, fold, assignments)
        path = args.rq1_dir / 'models' / f'M1_forward_fold{fold}.nc'
        expected = json.loads(Path(str(path) + '.fit.json').read_text())['request']['data_sha256']
        if _prepared_data_hash(train) != expected:
            raise RuntimeError(f'fold {fold} training data differ from the reported fit')
        idata = az.from_netcdf(path)
        idata.posterior.load()
        prediction_seed = seed + 30_000 + 100 * fold + 1
        full = _heldout_choice_predictions(idata, train, test, integrate_unseen_hierarchical=True,
                                           random_seed=prediction_seed, include_population_mean_sensitivity=True)
        reference = admitted[admitted.fold.eq(fold) & admitted.model.eq('M1')].set_index('choice_set_id')
        reproduced = float(np.max(np.abs(full.set_index('choice_set_id').log_score - reference.log_score)))
        ablated_idata = idata.copy()
        beta = idata.posterior['beta'].copy(deep=True)
        for term in ('candidate_target_is_ai', 'g2_x_candidate_ai'):
            beta.loc[dict(term=term)] = 0.0
        ablated_idata.posterior['beta'] = beta
        if float(np.abs(idata.posterior['beta'].sel(term='candidate_target_is_ai')).max()) == 0.0:
            raise RuntimeError('Ablation modified the reported posterior in memory')
        ablated = _heldout_choice_predictions(ablated_idata, train, test, integrate_unseen_hierarchical=True,
                                              random_seed=prediction_seed, include_population_mean_sensitivity=True)
        delta = full.log_score - ablated.log_score
        se, clusters = _participant_clustered_total_se(delta, full.actor_label)
        rows.append(dict(fold=fold, test_choice_sets=len(full), m1_reproduces_admitted_max_abs=reproduced,
                         elpd_m1_minus_m1_without_fixed_ai_terms=float(delta.sum()), se=se,
                         participant_clusters=clusters, ai_target_choices=int(full.chosen_is_ai.sum()),
                         elpd_difference_ai_target_choices=float(delta[full.chosen_is_ai.astype(bool)].sum()),
                         elpd_difference_participant_target_choices=float(
                             delta[~full.chosen_is_ai.astype(bool)].sum())))
        rows[-1]['_delta'] = delta.to_numpy()
        rows[-1]['_actor'] = full.actor_label.to_numpy()
        log(f'ablation fold {fold}: {rows[-1]["elpd_m1_minus_m1_without_fixed_ai_terms"]:.3f} '
            f'(se {se:.3f}); admitted reproduction {reproduced:.2e}')
    delta = np.concatenate([r.pop('_delta') for r in rows])
    actors = np.concatenate([r.pop('_actor') for r in rows])
    se, clusters = _participant_clustered_total_se(pd.Series(delta), pd.Series(actors))
    rows.append(dict(fold='ALL', test_choice_sets=len(delta), elpd_m1_minus_m1_without_fixed_ai_terms=float(delta.sum()),
                     se=se, participant_clusters=clusters))
    pd.DataFrame(rows).to_csv(args.out / 'forward_plugin_ablation.csv', index=False)

# ---------------------------------------------------------------------------
# Promotion of aggregate tables
# ---------------------------------------------------------------------------

M1_DENOMINATOR = ('2143 observed human replies (G1 1286, G2 857); 17489 earlier valid non-self same-thread '
                  'candidates (G1 10171, G2 7318); 36 participants; 4000 posterior draws of M1, run '
                  'rq1_m1_influence_refit_20260820')
RQ2_DENOMINATOR = ('2143 replies, 17489 candidates; AI-account projections average 3573 reply-by-AI-account '
                   'opportunities with a competing account (G1 2158, G2 1415) from 1597 replies and 36 '
                   'participants; participant-account projections average 8868 opportunities from 1978 replies; '
                   '4000 posterior draws, run rq2_m2_directional_contact_release_20260903')
PROMOTE = {
    'rq1_projection_decomposition': M1_DENOMINATOR,
    'rq1_control_attribution': M1_DENOMINATOR,
    'rq1_parameters': M1_DENOMINATOR,
    'rq1_sampler': M1_DENOMINATOR,
    'rq1_term_scaling': '17489 candidate rows pooled across communities (training set of the full fit)',
    'rq1_candidate_composition': '17489 candidate rows in 2143 choice sets',
    'rq1_within_set_composition': '2143 choice sets (G1 1286, G2 857)',
    'rq1_reproduction_check': M1_DENOMINATOR,
    'forward_fold_windows': '964 held-out replies in three rolling-origin folds (321/321/322)',
    'forward_unseen_levels': '964 held-out replies in three rolling-origin folds (321/321/322)',
    'forward_elpd_by_group': '964 held-out replies (G1 578, G2 386)',
    'forward_elpd_by_selected_type': '964 held-out replies',
    'forward_plugin_ablation': '964 held-out replies in three rolling-origin folds (321/321/322); reported M1 fold posteriors',
    'rq2_parameters': RQ2_DENOMINATOR,
    'rq2_projections': RQ2_DENOMINATOR,
    'rq2_linear_predictor_components': RQ2_DENOMINATOR,
    'rq2_count_recency_support': '17489 candidate rows (all account types)',
    'rq2_opportunity_rule': '2143 replies; 3573 AI-account opportunities from 1597 replies',
    'rq2_thread_composition': ('reply-by-account opportunities with a competing account: 3573 AI-account '
                               '(G1 2158, G2 1415) and 8868 participant-account'),
    'rq2_direct_child_candidates': '17489 candidate rows',
    'rq2_reproduction_check': RQ2_DENOMINATOR,
    'rq2_term_scaling': '17489 candidate rows pooled across communities',
    'rq2_split_projections': ('3573 AI-account opportunities from 1597 replies; 4000 posterior draws of the '
                              'unadmitted split-history run rq2_cross_thread_20260915'),
    'rq2_split_parameters': '2143 replies, 17489 candidates; unadmitted run rq2_cross_thread_20260915',
    'reaction_score_feasibility': ('17489 candidate rows; 3890 reactions (G1 2485, G2 1405) placed by round '
                                   '(421 rounds mapped to clock hours)'),
}


def stage_promote(args, log):
    destination = args.results
    destination.mkdir(parents=True, exist_ok=True)
    for name, denominator in PROMOTE.items():
        table = pd.read_csv(args.out / f'{name}.csv')
        if name == 'forward_fold_windows':
            for column in ('first_event', 'last_event'):
                table[column] = pd.to_datetime(table[column], utc=True).dt.tz_convert(
                    'Europe/Belgrade').dt.strftime('%Y-%m-%d')
            table = table.rename(columns={'first_event': 'first_date', 'last_event': 'last_date'})
        table.insert(0, 'run_id', RUN_ID)
        table['denominator'] = denominator
        path = destination / f'review_choice_models_{name}.csv'
        table.to_csv(path, index=False)
        log(f'promoted {path.name} ({len(table)} rows) sha256 {sha256(path)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['rq1', 'forward', 'ablation', 'rq2', 'split', 'reactions', 'all', 'promote'],
                        required=True)
    parser.add_argument('--results', type=Path, help='tables directory (promote stage)')
    parser.add_argument('--rq1-dir', type=Path)
    parser.add_argument('--rq2-dir', type=Path)
    parser.add_argument('--split-dir', type=Path)
    parser.add_argument('--events', type=Path, help='snapshot_events.parquet (reaction rounds)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=200)
    parser.add_argument('--max-draws', type=int, default=None, help='testing only')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.out / 'run.log')
    log(f'{RUN_ID} stage={args.stage} max_draws={args.max_draws} python={sys.version.split()[0]} '
        f'threads={os.environ.get("OMP_NUM_THREADS")} script_sha256={sha256(__file__)}')
    stages = (['rq1', 'forward', 'ablation', 'rq2', 'split', 'reactions'] if args.stage == 'all'
              else [args.stage])
    for stage in stages:
        globals()['stage_' + stage](args, log)
        log(f'stage {stage} complete; peak RSS {peak_rss_gib():.2f} GiB')
    save_json(args.out / f'complete_{args.stage}.json', dict(run_id=RUN_ID, stage=args.stage,
                                                             max_draws=args.max_draws))


if __name__ == '__main__':
    main()
