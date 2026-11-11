"""A1/A2 helpers: direct-reply opportunities, the D switch-off and the A2 arm frames on toy data."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.diagnostics.chb_prereview_a1_direct_reply_by_type import (
    M1_G2, a2_frames, direct_reply_mask, switch_off_change,
)
from scripts.diagnostics.review_choice_models_common import account_layout
from scripts.diagnostics.review_choice_models_direct_reply import subset_layout

# Rows of three choice sets. Set 2 holds one account only, so it yields no opportunity.
CHOICE = np.array([0, 0, 0, 0, 1, 1, 2, 2])
RECEIVER = np.array([10, 10, 20, 30, 10, 20, 20, 20])
AI = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
D = np.array([1, 0, 0, 1, 0, 1, 1, 1], dtype=float)
CHOSEN = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
TERMS = ['z', 'direct_child', 'candidate_ai_x_direct_child']


def _ragged():
    return SimpleNamespace(starts=np.array([0, 4, 6]), n_rows=8, receiver=RECEIVER, ai=AI,
                           actor=np.array([0, 0, 0, 0, 1, 1, 0, 0]), group=np.array(['G1', 'G2', 'G1'], dtype=object),
                           chosen=CHOSEN)


def _x():
    z = np.array([0.3, -1.0, 0.5, 1.2, -0.4, 0.8, 0.1, 0.2])
    return np.column_stack([z, D, AI * D])


def _brute(beta, x, focal, rows):
    u = x[rows] @ beta
    p = np.exp(u - u.max())
    p /= p.sum()
    return p[np.isin(rows, focal)].sum()


def test_opportunities_and_mask():
    data = SimpleNamespace(n_choices=3)
    ai = account_layout(data, _ragged(), 'AI')
    human = account_layout(data, _ragged(), 'HUMAN')
    assert ai['choice'].tolist() == [0, 1] and ai['receiver'].tolist() == [10, 10]
    assert human['choice'].tolist() == [0, 0, 1] and human['receiver'].tolist() == [20, 30, 20]
    assert direct_reply_mask(ai, D).tolist() == [True, False]
    assert direct_reply_mask(human, D).tolist() == [False, True, True]
    sub = subset_layout(human, direct_reply_mask(human, D))
    assert sub['focal'].tolist() == [3, 5] and sub['other'].tolist() == [0, 1, 2, 4]
    assert sub['selected'].tolist() == [False, False]


def test_switch_off_matches_brute_force():
    rng = np.random.default_rng(3)
    beta = rng.normal(size=(4, len(TERMS)))
    x = _x()
    utility = beta @ x.T
    indexes = [TERMS.index('direct_child'), TERMS.index('candidate_ai_x_direct_child')]
    data = SimpleNamespace(n_choices=3)
    sets = {0: np.arange(0, 4), 1: np.arange(4, 6)}
    for author_type in ('AI', 'HUMAN'):
        layout = account_layout(data, _ragged(), author_type)
        sub = subset_layout(layout, direct_reply_mask(layout, D))
        change = switch_off_change(utility, x, beta, sub, indexes)
        ends = np.r_[sub['focal_starts'][1:], len(sub['focal'])]
        for j, (s, e) in enumerate(zip(sub['focal_starts'], ends)):
            focal = sub['focal'][s:e]
            rows = sets[int(sub['choice'][j])]
            off = x.copy()
            off[np.ix_(focal, indexes)] = 0.0  # D = 0 for this account's candidates only
            for d in range(beta.shape[0]):
                expected = _brute(beta[d], x, focal, rows) - _brute(beta[d], off, focal, rows)
                assert np.isclose(change[d, j], expected, atol=1e-12)
    # A participant account's switch-off involves gamma_D only: zeroing the AI term changes nothing.
    layout = account_layout(data, _ragged(), 'HUMAN')
    sub = subset_layout(layout, direct_reply_mask(layout, D))
    only_gamma = switch_off_change(utility, x, beta, sub, [TERMS.index('direct_child')])
    assert np.allclose(only_gamma, switch_off_change(utility, x, beta, sub, indexes))


def test_a2_frames_conventions():
    rng = np.random.default_rng(1)
    bases = ['z_log_candidate_age', 'z_log_candidate_age_squared', 'z_log_candidate_depth', 'candidate_is_root',
             'candidate_ai_x_z_hist', 'candidate_ai_x_direct_child']
    terms = bases + [f'g2_x_{b}' for b in bases] + ['g2_x_candidate_ai']
    is_ai = rng.integers(0, 2, size=(3, 4)).astype(float)
    x = rng.normal(size=(3, 4, len(terms)))
    g2 = np.array([0.0, 1.0, 1.0])[:, None]
    for b in bases:
        x[:, :, terms.index(f'g2_x_{b}')] = g2 * x[:, :, terms.index(b)]
    x[:, :, terms.index('g2_x_candidate_ai')] = g2 * is_ai
    data = SimpleNamespace(terms=terms, x=x, candidate_is_ai=is_ai)
    frames = a2_frames(data)
    others = ['g2_x_candidate_ai_x_z_hist', 'g2_x_candidate_ai_x_direct_child']
    col = terms.index
    a1, a2 = frames['a_all_G2_terms']['frames']
    b1, b2 = frames['b_M1_G2_terms_others_at_G1']['frames']
    c1, c2 = frames['c_authorship_term_only']['frames']
    for t in [t for t in terms if t.startswith('g2_x_')]:
        assert np.all(a1[:, :, col(t)] == 0)
    assert np.array_equal(a2[:, :, col('g2_x_candidate_ai')], is_ai)
    assert np.array_equal(a2[:, :, col('g2_x_candidate_ai_x_z_hist')], x[:, :, col('candidate_ai_x_z_hist')])
    for t in M1_G2:
        assert np.array_equal(b2[:, :, col(t)], a2[:, :, col(t)]) and np.all(b1[:, :, col(t)] == 0)
    for t in others:
        assert np.all(b1[:, :, col(t)] == 0) and np.all(b2[:, :, col(t)] == 0)
        assert np.array_equal(c1[:, :, col(t)], x[:, :, col(t)]) and np.array_equal(c2[:, :, col(t)], x[:, :, col(t)])
    assert np.all(c1[:, :, col('g2_x_candidate_ai')] == 0)
    assert np.array_equal(c2[:, :, col('g2_x_candidate_ai')], is_ai)
    assert np.array_equal(c1[:, :, col('g2_x_z_log_candidate_age')], x[:, :, col('g2_x_z_log_candidate_age')])
