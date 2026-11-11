import numpy as np
import pandas as pd

from scripts.diagnostics import chb_prereview_a14_criticism_parent_adjusted as a14


def _frame():
    rows = []
    for group, typ, n, d1 in (('G1', 'AI', 10, 4), ('G1', 'H', 10, 1), ('G2', 'AI', 10, 2), ('G2', 'H', 10, 1)):
        for i in range(n):
            rows.append(dict(group=group, target_type=typ, cell=f'{group}_{typ}', d=int(i < d1),
                             participant=f'{group}:p{i % 3}', context=f'{group}:c{i % 4}'))
    return pd.DataFrame(rows)


def test_scheme_weights():
    frame = _frame()
    observed = a14.scheme_weights(frame, 'observed')
    assert observed == {'G1_AI': 0.4, 'G1_H': 0.1, 'G2_AI': 0.2, 'G2_H': 0.1}
    pooled = a14.scheme_weights(frame, 'pooled_by_type')
    assert pooled['G1_AI'] == pooled['G2_AI'] == 0.3 and pooled['G1_H'] == pooled['G2_H'] == 0.1
    assert set(a14.scheme_weights(frame, 'fixed_1').values()) == {1.0}


def test_cell_probabilities_closed_form_without_random_effects():
    frame = _frame()
    codes = {k: pd.factorize(frame[k], sort=True)[0] for k in ('participant', 'context')}
    s = 2
    params = dict(alpha=np.full(s, -2.0), beta_visible=np.full(s, 0.3), beta_ai=np.full(s, 1.0),
                  beta_interaction=np.full(s, 0.5), gamma_disagreement=np.full(s, 0.8),
                  delta_disagreement_ai=np.full(s, -0.4),
                  sigma_participant=np.zeros(s), z_participant=np.zeros((s, 6)),
                  sigma_context=np.zeros(s), z_context=np.zeros((s, 8)), sigma_target=np.zeros(s))
    weights = {k: a14.scheme_weights(frame, k) for k in a14.SCHEMES}
    p = a14.cell_probabilities(params, frame, codes, weights)
    ex = a14.expit
    w = weights['observed']
    g1_ai = (1 - w['G1_AI']) * ex(-2 + .3 + 1 + .5) + w['G1_AI'] * ex(-2 + .3 + 1 + .5 + .8 - .4)
    g2_h = (1 - w['G2_H']) * ex(-2) + w['G2_H'] * ex(-2 + .8)
    assert np.allclose(p['observed'][:, 0], g1_ai) and np.allclose(p['observed'][:, 3], g2_h)
    assert np.allclose(p['fixed_0'][:, 2], ex(-2 + 1))
    inter = a14.contrasts(p['fixed_0'])['interaction_visible_minus_hidden_AI_gap']
    assert np.allclose(inter, (ex(-1.7 + 1.5) - ex(-1.7)) - (ex(-1) - ex(-2)))


def _nonroot_frame():
    frame = _frame()
    # Non-root parents: every d = 1 row plus the next two rows of each cell.
    frame['nonroot'] = (frame.groupby('cell').cumcount() < frame.groupby('cell').d.transform('sum') + 2).astype(int)
    return frame


def test_state_weights_sum_to_one_and_match_scheme_definitions():
    frame = _nonroot_frame()
    for scheme in a14.NONROOT_SCHEMES:
        weights = a14.state_weights(frame, scheme)
        for cell in a14.CELLS:
            assert abs(sum(weights[cell].values()) - 1) < 1e-12
    observed = a14.state_weights(frame, 'observed')
    assert observed['G1_AI'] == {(0, 0): 0.4, (0, 1): 0.2, (1, 1): 0.4}
    fixed_0 = a14.state_weights(frame, 'fixed_0')
    assert fixed_0['G1_AI'] == {(0, 0): 0.4, (0, 1): 0.6, (1, 1): 0.0}
    pooled = a14.state_weights(frame, 'pooled_by_type')
    assert pooled['G1_AI'] == pooled['G2_AI'] and pooled['G1_AI'][(1, 1)] == 0.3
    # P(d = 1) under the state weights equals the scheme_weights of the model without the nonroot term.
    for scheme in a14.SCHEMES:
        assert {c: w[(1, 1)] for c, w in a14.state_weights(frame, scheme).items()} == a14.scheme_weights(frame, scheme)


def test_cell_probabilities_with_nonroot_term_closed_form():
    frame = _nonroot_frame()
    codes = {k: pd.factorize(frame[k], sort=True)[0] for k in ('participant', 'context')}
    s = 2
    params = dict(alpha=np.full(s, -2.0), beta_visible=np.full(s, 0.3), beta_ai=np.full(s, 1.0),
                  beta_interaction=np.full(s, 0.5), gamma_disagreement=np.full(s, 0.8), rho_nonroot=np.full(s, -0.3),
                  sigma_participant=np.zeros(s), z_participant=np.zeros((s, 6)),
                  sigma_context=np.zeros(s), z_context=np.zeros((s, 8)), sigma_target=np.zeros(s))
    weights = {k: a14.state_weights(frame, k) for k in a14.NONROOT_SCHEMES}
    p = a14.cell_probabilities(params, frame, codes, weights)
    ex = a14.expit
    g1_ai = -2 + .3 + 1 + .5
    expected = .4 * ex(g1_ai) + .2 * ex(g1_ai - .3) + .4 * ex(g1_ai - .3 + .8)
    assert np.allclose(p['observed'][:, 0], expected)
    assert np.allclose(p['fixed_0_root'][:, 3], ex(-2))
    assert np.allclose(p['fixed_0_nonroot'][:, 3], ex(-2 - .3))
    assert np.allclose(p['fixed_1'][:, 2], ex(-2 + 1 - .3 + .8))
    # Without rho the state weights reproduce the P(d = 1) mixture of the original variants.
    params['rho_nonroot'] = np.zeros(s)
    old = a14.cell_probabilities(params, frame, codes, {k: a14.scheme_weights(frame, k) for k in a14.SCHEMES})
    for scheme in ('observed', 'pooled_by_type', 'fixed_1'):
        assert np.allclose(p_new := a14.cell_probabilities(params, frame, codes, {scheme: weights[scheme]})[scheme],
                           old[scheme]), p_new


def test_passing_fit_ignores_other_variant_folders(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(a14, 'OUT', tmp_path)
    for name in ('pooled', 'pooled_nonroot', 'pooled_prior1', 'pooled_tune2000'):
        (tmp_path / 'fits' / name).mkdir(parents=True)
        (tmp_path / 'fits' / name / 'diagnostics.json').write_text(json.dumps({'passed': name != 'pooled_tune2000'}))
    assert a14.passing_fit('pooled').name == 'pooled'
    assert a14.passing_fit('pooled_nonroot').name == 'pooled_nonroot'
