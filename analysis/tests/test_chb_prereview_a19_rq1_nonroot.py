"""A19 non-root projection: per-set components, decomposition identity and set support."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from scripts.diagnostics.chb_prereview_a19_rq1_nonroot import (  # noqa: E402
    PROJECTIONS, project, root_rows, set_components, set_support,
)
from scripts.diagnostics.review_choice_models_common import Posterior, Ragged  # noqa: E402
from scripts.publication_analysis.model import prepare_model_data  # noqa: E402


def _fixture(seed: int = 5, n_choices: int = 30):
    rng = np.random.default_rng(seed)
    rows = []
    for choice in range(n_choices):
        group = 'G1' if choice % 2 else 'G2'
        size = int(rng.integers(2, 7))
        has_root = choice % 4 != 0
        authors = rng.choice(10, size=size)
        chosen = int(rng.integers(size))
        for position, author in enumerate(authors):
            rows.append(dict(choice_set_id=f'c{choice}', group=group, chosen=int(position == chosen),
                             candidate_author_id=int(author), actor_id=choice % 4, primary_root_topic='UNTAGGED',
                             candidate_target_is_ai=int(author % 2),
                             candidate_is_root=float(has_root and position == 0),
                             z_log_candidate_age=rng.normal(), z_log_candidate_age_squared=rng.normal(),
                             z_log_candidate_depth=rng.normal()))
    frame = pd.DataFrame(rows)
    g2 = frame.group.eq('G2').astype(float)
    for base in ('z_log_candidate_age', 'z_log_candidate_age_squared', 'z_log_candidate_depth', 'candidate_is_root'):
        frame[f'g2_x_{base}'] = g2 * frame[base]
    frame['g2_x_candidate_ai'] = g2 * frame.candidate_target_is_ai
    terms = ('z_log_candidate_age', 'z_log_candidate_age_squared', 'z_log_candidate_depth', 'candidate_is_root',
             'g2_x_z_log_candidate_age', 'g2_x_z_log_candidate_age_squared', 'g2_x_z_log_candidate_depth',
             'g2_x_candidate_is_root', 'g2_x_candidate_ai')
    data = prepare_model_data(frame, terms, ai_term='candidate_target_is_ai')
    draws = 5
    posterior = {
        'beta': rng.normal(size=(1, draws, len(data.terms))) * 2,
        'receiver_effect': rng.normal(scale=0.3, size=(1, draws, len(data.receiver_labels))),
        'actor_ai_effect': rng.normal(scale=0.3, size=(1, draws, len(data.actor_labels))),
    }
    idata = az.from_dict(posterior=posterior, coords={'term': list(data.terms), 'receiver': list(data.receiver_labels),
                                                      'actor': list(data.actor_labels)},
                         dims={'beta': ['term'], 'receiver_effect': ['receiver'], 'actor_ai_effect': ['actor']})
    return frame, data, idata


def _brute(utility_row, ai, root):
    p = np.exp(utility_row - utility_row.max())
    p /= p.sum()
    nr = ~root
    q = p[nr] / p[nr].sum()
    return p @ ai, p[root].sum(), (q * ai[nr]).sum()


def test_components_match_brute_force_and_identity():
    _, data, idata = _fixture()
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    root = root_rows(data, ragged)
    utility = ragged.utility(posterior, slice(0, posterior.n), ragged.rows(data.x))
    parts = set_components(ragged, utility, root)
    ends = np.r_[ragged.starts[1:], ragged.n_rows]
    for d in range(posterior.n):
        for s, (a, b) in enumerate(zip(ragged.starts, ends)):
            p_ai, p_root, p_nr = _brute(utility[d, a:b], ragged.ai[a:b], root[a:b])
            assert parts['P_AI'][d, s] == pytest.approx(p_ai, abs=1e-12)
            assert parts['p_root'][d, s] == pytest.approx(p_root, abs=1e-12)
            assert parts['P_nr'][d, s] == pytest.approx(p_nr, abs=1e-12)
    identity = parts['root_ai_component'] + parts['nonroot_component'] - parts['P_AI']
    assert np.max(np.abs(identity)) < 1e-12
    # Sets without a root: P_nr equals P_AI.
    no_root = ~np.add.reduceat(root.astype(float), ragged.starts).astype(bool)
    np.testing.assert_allclose(parts['P_nr'][:, no_root], parts['P_AI'][:, no_root], atol=1e-12)


def test_nonroot_probability_stable_when_root_dominates():
    _, data, idata = _fixture()
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    root = root_rows(data, ragged)
    utility = ragged.utility(posterior, slice(0, 1), ragged.rows(data.x)) + 800.0 * root[None, :]
    parts = set_components(ragged, utility, root)
    assert np.isfinite(parts['P_nr']).all()
    assert ((parts['P_nr'] >= -1e-12) & (parts['P_nr'] <= 1 + 1e-12)).all()
    unshifted = set_components(ragged, utility - 800.0 * root[None, :], root)
    np.testing.assert_allclose(parts['P_nr'], unshifted['P_nr'], atol=1e-12)


def test_set_support_flags():
    frame, data, _ = _fixture()
    ragged = Ragged(data)
    support = set_support(ragged, root_rows(data, ragged))
    expected = []
    for _, part in frame.groupby('choice_set_id', sort=False):
        nr = part[part.candidate_is_root.eq(0)]
        expected.append(dict(has_root=bool(part.candidate_is_root.eq(1).any()),
                             n_nonroot=len(nr), mixed_nonroot=nr.candidate_target_is_ai.nunique() == 2,
                             chosen_root=bool(part.loc[part.chosen.eq(1), 'candidate_is_root'].eq(1).any()),
                             group=part.group.iloc[0]))
    expected = pd.DataFrame(expected)
    for column in expected:
        assert support[column].tolist() == expected[column].tolist()


def test_project_scopes_and_arms():
    _, data, idata = _fixture()
    support, store, error = project(data, idata, chunk=2, log=lambda message: None)
    assert error < 1e-12
    for key in PROJECTIONS:
        for arm in ('G1', 'G2'):
            p_ai = store[(key, arm, 'all_sets', 'P_AI')]
            parts = store[(key, arm, 'all_sets', 'root_ai_component')] + store[(key, arm, 'all_sets', 'nonroot_component')]
            np.testing.assert_allclose(p_ai, parts, atol=1e-12)
            assert store[(key, arm, 'mixed_nonroot_sets', 'P_nr')].shape == (5,)
    # Sets whose non-root candidates are all of one author type have P_nr fixed at 0 or 1 in both arms,
    # so the all-set contrast equals the mixed-set contrast scaled by the mixed share.
    share = support.mixed_nonroot.mean()
    for key in PROJECTIONS:
        all_c = store[(key, 'G2', 'all_sets', 'P_nr')] - store[(key, 'G1', 'all_sets', 'P_nr')]
        mixed_c = store[(key, 'G2', 'mixed_nonroot_sets', 'P_nr')] - store[(key, 'G1', 'mixed_nonroot_sets', 'P_nr')]
        np.testing.assert_allclose(all_c, share * mixed_c, atol=1e-12)


def test_headline_table_reports_root_probability_on_both_set_scopes():
    from scripts.diagnostics.chb_prereview_a19_rq1_nonroot import headline_table
    rng = np.random.default_rng(1)
    store = {}
    for key in PROJECTIONS:
        for group in ('G1', 'G2'):
            for component in ('P_nr', 'P_AI', 'p_root', 'p_root_given_root_present'):
                store[(key, group, 'all_sets', component)] = rng.uniform(0.2, 0.7, size=50)
    support = pd.DataFrame([dict(group='pooled', choice_sets=10, sets_with_root=7)])
    table = headline_table(store, support)
    assert len(table) == len(PROJECTIONS) * 4 * 3
    row = table[(table.projection == 'primary_all_G2_terms') & (table.component == 'p_root_given_root_present')
                & (table.group == 'G2-G1')].iloc[0]
    diff = 100 * (store[('primary_all_G2_terms', 'G2', 'all_sets', 'p_root_given_root_present')]
                  - store[('primary_all_G2_terms', 'G1', 'all_sets', 'p_root_given_root_present')])
    assert row.posterior_mean == pytest.approx(diff.mean())
    assert row.ci_high == pytest.approx(np.quantile(diff, .975))
    assert row.set_scope == 'sets_with_root' and row.choice_sets == 7
    assert set(table.loc[table.component == 'p_root', 'choice_sets']) == {10}
