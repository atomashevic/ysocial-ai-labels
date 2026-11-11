"""A21 within-community RQ1 projection: counterfactual utilities, set scopes and table contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from scripts.diagnostics.chb_prereview_a21_rq1_within_community import (  # noqa: E402
    AI_TERM, AI_X, COUNTERFACTUALS, PRIMARY_CF, SENSITIVITY_CF, check_ai_columns, project, results_table,
    set_flags,
)
from scripts.diagnostics.review_choice_models_common import Ragged  # noqa: E402
from scripts.publication_analysis.model import (  # noqa: E402
    _ai_choice_probability, _posterior_utilities, choice_utilities, prepare_model_data,
)

DRAWS = 6


def _fixture(seed: int = 11, n_choices: int = 40):
    rng = np.random.default_rng(seed)
    rows = []
    for choice in range(n_choices):
        group = 'G1' if choice % 2 else 'G2'
        size = int(rng.integers(2, 7))
        authors = rng.choice(12, size=size)
        if choice % 5 == 0:  # single-author-type sets
            authors = authors - authors % 2 + (choice % 2)
        chosen = int(rng.integers(size))
        for position, author in enumerate(authors):
            rows.append(dict(choice_set_id=f'c{choice}', group=group, chosen=int(position == chosen),
                             candidate_author_id=int(author), actor_id=f'{group}{choice % 3}',
                             primary_root_topic='UNTAGGED', candidate_target_is_ai=int(author % 2),
                             candidate_is_root=float(position == 0 and choice % 3 != 0),
                             z_log_candidate_age=rng.normal(), z_log_candidate_depth=rng.normal()))
    frame = pd.DataFrame(rows)
    g2 = frame.group.eq('G2').astype(float)
    frame['g2_x_z_log_candidate_age'] = g2 * frame.z_log_candidate_age
    frame['g2_x_candidate_ai'] = g2 * frame.candidate_target_is_ai
    terms = ('z_log_candidate_age', 'z_log_candidate_depth', 'candidate_is_root', 'g2_x_z_log_candidate_age',
             'g2_x_candidate_ai')
    data = prepare_model_data(frame, terms, ai_term='candidate_target_is_ai')
    actor = rng.normal(scale=0.5, size=(1, DRAWS, len(data.actor_labels)))
    groups = np.asarray([label[:2] for label in data.actor_labels])
    for g in ('G1', 'G2'):  # participant AI slopes average zero within each community, as in M1
        actor[..., groups == g] -= actor[..., groups == g].mean(axis=-1, keepdims=True)
    posterior = {
        'beta': rng.normal(size=(1, DRAWS, len(data.terms))),
        'receiver_effect': rng.normal(scale=0.3, size=(1, DRAWS, len(data.receiver_labels))),
        'actor_ai_effect': actor,
    }
    idata = az.from_dict(posterior=posterior,
                         coords={'term': list(data.terms), 'receiver': list(data.receiver_labels),
                                 'actor': list(data.actor_labels)},
                         dims={'beta': ['term'], 'receiver_effect': ['receiver'], 'actor_ai_effect': ['actor']})
    return frame, data, idata


def _brute(data, idata, *, drop_slopes: bool, zero_ai: bool = True):
    """Per-draw set AI probability from model.choice_utilities with the AI coefficients removed."""
    post = idata.posterior
    out = np.empty((DRAWS, data.n_choices))
    for d in range(DRAWS):
        beta = post.beta.values[0, d].copy()
        if zero_ai:
            beta[list(post.beta.coords['term'].values).index(AI_TERM)] = 0.0
            beta[list(post.beta.coords['term'].values).index(AI_X)] = 0.0
        u = choice_utilities(data, beta, receiver_effect=post.receiver_effect.values[0, d],
                             actor_ai_effect=None if drop_slopes else post.actor_ai_effect.values[0, d])
        p = np.exp(u - u.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out[d] = (p * data.candidate_is_ai).sum(axis=1)
    return out


def _means(values, mask):
    return values[:, mask].mean(axis=1)


def test_projection_matches_published_utilities_and_brute_force():
    _, data, idata = _fixture()
    check_ai_columns(data)
    flags, store, checks = project(data, idata, chunk=4, log=lambda message: None)
    assert checks['total_effect_identity_max_abs_error'] < 1e-12
    assert checks['single_author_type_sets_max_abs_P_obs_minus_P_cf'] < 1e-12
    observed = _ai_choice_probability(_posterior_utilities(idata, data), data)
    primary_cf = _brute(data, idata, drop_slopes=True)
    keep_slopes = _brute(data, idata, drop_slopes=False)
    np.testing.assert_allclose(_brute(data, idata, drop_slopes=False, zero_ai=False), observed, atol=1e-12)
    group = flags.group.to_numpy()
    mixed = flags.mixed.to_numpy()
    for g in ('G1', 'G2'):
        for scope, mask in (('own_sets', group == g), ('own_mixed_sets', (group == g) & mixed)):
            np.testing.assert_allclose(store[('P_obs', g, scope)], _means(observed, mask), atol=1e-12)
            np.testing.assert_allclose(store[(PRIMARY_CF, g, scope)], _means(primary_cf, mask), atol=1e-12)
            np.testing.assert_allclose(store[(SENSITIVITY_CF, g, scope)], _means(keep_slopes, mask), atol=1e-12)


def test_single_type_sets_scale_all_set_difference():
    _, data, idata = _fixture()
    flags, store, _ = project(data, idata, chunk=DRAWS, log=lambda message: None)
    assert (~flags.mixed).any() and flags.mixed.any()
    for g in ('G1', 'G2'):
        part = flags[flags.group.eq(g)]
        share = part.mixed.mean()
        for name in COUNTERFACTUALS:
            own = store[('P_obs', g, 'own_sets')] - store[(name, g, 'own_sets')]
            mixed = store[('P_obs', g, 'own_mixed_sets')] - store[(name, g, 'own_mixed_sets')]
            np.testing.assert_allclose(own, share * mixed, atol=1e-12)


def test_set_flags_against_frame():
    frame, data, _ = _fixture()
    flags = set_flags(Ragged(data))
    expected = frame.groupby('choice_set_id', sort=False).apply(
        lambda part: pd.Series(dict(group=part.group.iloc[0], mixed=part.candidate_target_is_ai.nunique() == 2,
                                    chosen_ai=bool(part.loc[part.chosen.eq(1), 'candidate_target_is_ai'].iloc[0]))),
        include_groups=False)
    assert flags.group.tolist() == expected.group.tolist()
    assert flags.mixed.tolist() == expected.mixed.tolist()
    assert flags.chosen_ai.tolist() == expected.chosen_ai.tolist()


def test_results_table_contrasts_and_odds_ratios():
    _, data, idata = _fixture()
    flags, store, _ = project(data, idata, chunk=DRAWS, log=lambda message: None)
    table = results_table(store, flags, data)
    assert table.run_id.eq('chb_prereview_20261110').all() and table.denominator.notna().all()
    for scope in ('own_sets', 'own_mixed_sets'):
        did = table[(table.estimand == 'G2_minus_G1_within_community_difference') & (table.set_scope == scope)
                    & (table.counterfactual == PRIMARY_CF)].iloc[0]
        expected = 100 * ((store[('P_obs', 'G2', scope)] - store[(PRIMARY_CF, 'G2', scope)])
                          - (store[('P_obs', 'G1', scope)] - store[(PRIMARY_CF, 'G1', scope)]))
        assert did.posterior_mean == pytest.approx(expected.mean())
        assert did.ci_low == pytest.approx(np.quantile(expected, 0.025))
        assert did.unit == 'percentage_points'
    beta = idata.posterior.beta.values[0]
    terms = list(idata.posterior.beta.coords['term'].values)
    hidden = beta[:, terms.index(AI_TERM)] + beta[:, terms.index(AI_X)]
    row = table[(table.estimand == 'ai_authorship_odds_ratio') & (table.community == 'G2')].iloc[0]
    assert row.posterior_mean == pytest.approx(np.exp(hidden).mean())
    assert row.ci_high == pytest.approx(np.quantile(np.exp(hidden), 0.975))
    assert row.posterior_probability_gt_null == pytest.approx(np.mean(hidden > 0))
    share = table[(table.estimand == 'observed_ai_target_share') & (table.community == 'G1')
                  & (table.set_scope == 'own_sets')].iloc[0]
    g1 = flags[flags.group.eq('G1')]
    assert share.choice_sets == len(g1) and share.observed_ai_target_replies == int(g1.chosen_ai.sum())
    assert share.posterior_mean == pytest.approx(100 * g1.chosen_ai.mean())


def test_zero_ai_effect_gives_zero_difference():
    _, data, idata = _fixture()
    terms = list(idata.posterior.beta.coords['term'].values)
    idata.posterior['beta'].values[..., terms.index(AI_TERM)] = 0.0
    idata.posterior['beta'].values[..., terms.index(AI_X)] = 0.0
    idata.posterior['actor_ai_effect'].values[...] = 0.0
    _, store, _ = project(data, idata, chunk=DRAWS, log=lambda message: None)
    for g in ('G1', 'G2'):
        for name in COUNTERFACTUALS:
            np.testing.assert_allclose(store[('P_obs', g, 'own_sets')], store[(name, g, 'own_sets')], atol=1e-14)
