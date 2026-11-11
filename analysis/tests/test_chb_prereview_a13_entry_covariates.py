"""Tests for the A13 direct-thread-entry covariate refit (feed rank, toxic persona, estimands)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics import chb_prereview_a13_entry_covariates as a13


def _raw(rows):
    return pd.DataFrame(rows, columns=['action', 'id', 'type', 'semantic_group', 'section'])


FEED = ('appear', None, 'post', 'feed_container', '/39/rfeed/all')


def _card(post_id, kind='post', group='feed_container', action='appear'):
    return (action, f'post_{post_id}', kind, group, '/39/rfeed/all')


def test_visit_feed_ranks_counts_all_feed_cards_by_first_appearance():
    raw = _raw([
        (None, None, None, None, None),                          # 1 blank row
        ('page_view', 'pv', 'page', 'global', '/39/rfeed/all'),  # 2 visit entry
        _card(10),                                               # 3 rank 1 (not eligible)
        _card(20),                                               # 4 rank 2
        _card(10, action='disappear'),                           # 5
        _card(10),                                               # 6 repeat, ignored
        _card(30, group='thread_container'),                     # 7 not a feed card
        _card(40),                                               # 8 rank 3
        ('page_view', 'pv', 'page', 'global', '/39/rthread/40'), # 9 visit exit
        _card(50),                                               # 10 outside visit
        ('page_view', 'pv', 'page', 'global', '/39/rfeed/all'),  # 11 open visit
        _card(40),                                               # 12 rank 1 in open visit
        _card(20),                                               # 13 rank 2
    ])
    visits = pd.DataFrame({
        'opportunity_id': ['a', 'b', 'c', 'd'],
        'feed_entry_event_seq': [2, 2, 11, 11],
        'feed_exit_event_seq': pd.array([9, 9, pd.NA, pd.NA], dtype='Int64'),
        'post_id': [20, 40, 40, 20],
    })
    out = a13.visit_feed_ranks(raw, visits).set_index('opportunity_id')
    assert out.loc['a', 'feed_rank'] == 2
    assert out.loc['b', 'feed_rank'] == 3
    assert out.loc['c', 'feed_rank'] == 1
    assert out.loc['d', 'feed_rank'] == 2
    assert out.loc['a', 'feed_visit_cards'] == 3
    assert out.loc['c', 'feed_visit_cards'] == 2
    assert out.loc['b', 'raw_first_appear_seq'] == 8
    assert out.raw_entry_is_feed_page_view.all() and out.raw_exit_is_page_view.all()
    assert out.raw_page_views_inside_visit.eq(0).all()


def test_eligible_rank_orders_within_visit():
    panel = pd.DataFrame({'feed_spell_key': ['v1', 'v1', 'v2', 'v1'], 'first_appear_event_seq': [8, 4, 3, 6]})
    assert a13.eligible_feed_rank(panel).tolist() == [3, 1, 1, 2]


def test_persona_lookup_flags_only_toxic_ai():
    accounts = pd.DataFrame({'group': ['G1'] * 4, 'user_id': [1, 2, 3, 4],
                             'actor_class': ['ai', 'ai', 'ai', 'human'],
                             'toxicity': ['moderately', 'Extremely ', 'absolutely no', None]})
    persona = a13.persona_lookup(accounts)
    assert persona.to_dict() == {'G1:1': 1.0, 'G1:2': 1.0, 'G1:3': 0.0, 'G1:4': 0.0}


def test_post_type_atoms_weights():
    frame = pd.DataFrame({'toxic_persona': [1, 1, 0, 0, 0], 'z_root_toxicity': np.linspace(-1, 1, 5)})
    mask = np.ones(5, dtype=bool)
    atoms = a13.post_type_atoms(frame, mask, ['toxic_persona'])
    assert sorted((v['toxic_persona'], w) for v, w in atoms) == [(0.0, 0.6), (1.0, 0.4)]
    atoms = a13.post_type_atoms(frame, mask, ['toxic_persona', 'z_root_toxicity'])
    assert a13.N_TOXICITY_ATOMS == 40
    assert len(atoms) == 80 and np.isclose(sum(w for _, w in atoms), 1.0)
    atoms = a13.post_type_atoms(frame, mask, ['toxic_persona', 'z_root_toxicity'], n_atoms=10)
    assert len(atoms) == 20 and np.isclose(sum(w for _, w in atoms), 1.0)
    assert a13.post_type_atoms(frame, mask, ['z_log_feed_rank']) == [({}, 1.0)]


def _synthetic(seed=0, n=80, draws=6):
    import arviz as az

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'opportunity_id': [f'o{i}' for i in range(n)],
        'group': np.where(np.arange(n) < n // 2, 'G1', 'G2'),
        'author_type': rng.choice(['H', 'AI'], n),
        'participant_key': [f'p{i % 6}' for i in range(n)],
        'session_id': [f's{i % 10}' for i in range(n)],
        'post_key': [f'q{i % 15}' for i in range(n)],
        'experiment_day': rng.choice([2, 3, 18], n),
        'post_age_hours': rng.exponential(5, n),
        'feed_spell_card_opportunities': rng.integers(1, 9, n),
        'direct_thread_entry': rng.integers(0, 2, n),
    })
    # participants nested in groups as in the study
    frame['participant_key'] = frame.group + ':' + frame.participant_key
    from scripts.publication_analysis.rq1b import build_model_frame
    frame, _ = build_model_frame(frame)
    frame['z_log_feed_rank'] = rng.normal(size=n)
    frame['toxic_persona'] = np.where(frame.author_type.eq('AI'), rng.integers(0, 2, n), 0).astype(float)
    frame['z_root_toxicity'] = rng.normal(size=n)
    sizes = {'participant': frame.participant_index.nunique(), 'session': frame.session_index.nunique(),
             'post': frame.post_index.nunique(), 'valid_day': frame.day_index.nunique()}
    posterior = {name: rng.normal(scale=0.5, size=(1, draws)) for name in (
        'alpha', 'beta_group', 'beta_ai', 'beta_group_ai', 'beta_log_age', 'beta_log_feed_spell_cards',
        'beta_z_log_feed_rank', 'beta_toxic_persona', 'beta_z_root_toxicity', 'sigma_participant',
        'sigma_participant_ai', 'sigma_session', 'sigma_post', 'sigma_day')}
    dims = {}
    for name, dim in (('participant_effect', 'participant'), ('participant_ai_effect', 'participant'),
                      ('session_effect', 'session'), ('post_effect', 'post'), ('day_effect', 'valid_day')):
        posterior[name] = rng.normal(scale=0.3, size=(1, draws, sizes[dim]))
        dims[name] = [dim]
    idata = az.from_dict(posterior=posterior, dims=dims)
    return frame, idata, posterior


def test_cell_draws_without_extra_terms_matches_admitted_estimands():
    from scripts.publication_analysis.rq1b import posterior_estimands

    frame, idata, _ = _synthetic()
    admitted = posterior_estimands(idata, frame).set_index(['estimand', 'group', 'author_type',
                                                            'standardization_scope'])
    for scheme in ('community', 'pooled'):
        rows = pd.DataFrame(a13.estimand_rows(a13.cell_draws(idata, frame, (), scheme, chunk=4)))
        rows = rows.set_index(['estimand', 'group', 'author_type', 'standardization_scope'])
        joined = rows.join(admitted, rsuffix='_admitted')
        assert np.allclose(joined.posterior_mean, joined.posterior_mean_admitted, atol=1e-12)
        assert np.allclose(joined.ci_low, joined.ci_low_admitted, atol=1e-12)


def test_cell_draws_persona_mixture_and_focal_arithmetic():
    from scipy.special import expit

    frame, idata, post = _synthetic(seed=3)
    cells = a13.cell_draws(idata, frame, ('z_log_feed_rank', 'toxic_persona'), 'community', chunk=4)
    equal = cells['equal_participant_observed_opportunities']
    d = 0
    for g in ('G1', 'G2'):
        rows = frame[frame.group.eq(g)]
        share = rows.loc[rows.author_type.eq('AI'), 'toxic_persona'].mean()
        base = (post['alpha'][0, d] + post['beta_group'][0, d] * (g == 'G2') + post['beta_ai'][0, d]
                + post['beta_group_ai'][0, d] * (g == 'G2')
                + post['beta_log_age'][0, d] * rows.z_log_post_age
                + post['beta_log_feed_spell_cards'][0, d] * rows.z_log_feed_spell_cards
                + post['beta_z_log_feed_rank'][0, d] * rows.z_log_feed_rank
                + post['participant_effect'][0, d][rows.participant_index]
                + post['participant_ai_effect'][0, d][rows.participant_index]
                + post['session_effect'][0, d][rows.session_index]
                + post['post_effect'][0, d][rows.post_index] + post['day_effect'][0, d][rows.day_index])
        p = share * expit(base + post['beta_toxic_persona'][0, d]) + (1 - share) * expit(base)
        expected = pd.Series(np.asarray(p), index=rows.index).groupby(rows.participant_index).mean().mean()
        assert np.isclose(equal[f'{g}_AI'][d], expected)
    rows = pd.DataFrame(a13.estimand_rows(cells))
    focal = rows[(rows.estimand == 'G2_minus_G1_change_in_AI_minus_H_gap')
                 & (rows.standardization_scope == 'equal_participant_observed_opportunities')]
    manual = equal['G2_AI'] - equal['G2_H'] - (equal['G1_AI'] - equal['G1_H'])
    assert np.isclose(focal.posterior_mean.iloc[0], manual.mean())


def test_build_model_without_terms_has_admitted_logp():
    pytest.importorskip('pymc')
    from scripts.publication_analysis.rq1b import build_pymc_model

    frame, _, _ = _synthetic(seed=5)
    sampling = SimpleNamespace(prior_focal_sigma=1.0, prior_adjustment_sigma=0.5, prior_scale_sigma=0.5)
    ours = a13.build_model(frame, sampling, ())
    theirs = build_pymc_model(frame, SimpleNamespace(sampling=sampling))
    point = theirs.initial_point()
    rng = np.random.default_rng(1)
    point = {k: v + rng.normal(scale=0.1, size=np.shape(v)) for k, v in point.items()}
    assert set(ours.initial_point()) == set(point)
    assert np.isclose(ours.compile_logp()(point), theirs.compile_logp()(point))
    extended = a13.build_model(frame, sampling, ('z_log_feed_rank', 'toxic_persona'))
    assert {'beta_z_log_feed_rank', 'beta_toxic_persona'} <= set(extended.named_vars)


def test_prefinal_rank_caps_last_two_cards():
    rank = [1, 2, 3, 4, 5, 1, 2, 1]
    cards = [5, 5, 5, 5, 5, 2, 2, 1]
    assert a13.prefinal_feed_rank(rank, cards).tolist() == [1, 2, 3, 4, 4, 1, 1, 1]
    with pytest.raises(ValueError):
        a13.prefinal_feed_rank([3], [2])


def test_public_feed_ranks_match_raw_rule_and_crosscheck():
    raw = _raw([
        ('page_view', 'pv', 'page', 'global', '/39/rfeed/all'),  # 1 visit entry
        _card(10),                                               # 2
        _card(20),                                               # 3
        _card(10),                                               # 4 repeat
        ('page_view', 'pv', 'page', 'global', '/39/rthread/20'), # 5 exit
        ('page_view', 'pv', 'page', 'global', '/39/rfeed/all'),  # 6 open visit
        _card(30),                                               # 7
        _card(20),                                               # 8
    ])
    visits = pd.DataFrame({'opportunity_id': ['a', 'b', 'c'], 'session_id': ['s1'] * 3,
                           'feed_entry_event_seq': [1, 6, 6],
                           'feed_exit_event_seq': pd.array([5, pd.NA, pd.NA], dtype='Int64'),
                           'post_id': [20, 20, 30], 'first_appear_event_seq': [3, 8, 7]})
    raw_ranks = a13.visit_feed_ranks(raw, visits[['opportunity_id', 'feed_entry_event_seq',
                                                   'feed_exit_event_seq', 'post_id']])
    events = raw.assign(session_id='s1', event_seq=np.arange(1, len(raw) + 1))
    other = events.assign(session_id='s2')                       # other sessions are ignored
    public = a13.public_feed_ranks(pd.concat([events, other]), visits).set_index('opportunity_id')
    assert public.public_feed_rank.to_dict() == {'a': 2, 'b': 2, 'c': 1}
    assert public.public_feed_visit_cards.to_dict() == {'a': 2, 'b': 2, 'c': 2}
    checks = a13.rank_crosscheck(public.reset_index(), raw_ranks, visits)
    assert checks['rank_equal'] == checks['cards_equal'] == checks['first_appear_seq_equal_panel'] == 3
    assert checks['max_abs_rank_diff'] == 0.0
    shifted = raw_ranks.assign(feed_rank=raw_ranks.feed_rank + 1)
    assert a13.rank_crosscheck(public.reset_index(), shifted, visits)['rank_equal'] == 0


def test_no_visit_size_model_and_estimands():
    pytest.importorskip('pymc')
    frame, idata, post = _synthetic(seed=7)
    sampling = SimpleNamespace(prior_focal_sigma=1.0, prior_adjustment_sigma=0.5, prior_scale_sigma=0.5)
    model = a13.build_model(frame, sampling, ('z_log_feed_rank',), visit_size=False)
    assert 'beta_log_feed_spell_cards' not in model.named_vars
    assert 'beta_log_feed_spell_cards' in a13.build_model(frame, sampling, ('z_log_feed_rank',)).named_vars
    dropped = idata.copy()
    dropped.posterior = dropped.posterior.drop_vars('beta_log_feed_spell_cards')
    zeroed = idata.copy()
    zeroed.posterior['beta_log_feed_spell_cards'] = zeroed.posterior['beta_log_feed_spell_cards'] * 0.0
    for scheme in ('community', 'pooled'):
        a = a13.cell_draws(dropped, frame, ('z_log_feed_rank',), scheme, chunk=4)
        b = a13.cell_draws(zeroed, frame, ('z_log_feed_rank',), scheme, chunk=4)
        for scope in a:
            for cell in a[scope]:
                assert np.allclose(a[scope][cell], b[scope][cell])
    assert 'beta_log_feed_spell_cards' not in a13.parameter_names('position_disposition_no_visit_size')
    assert 'beta_z_log_feed_rank_prefinal' in a13.parameter_names('prefinal_position_disposition')


def test_parameter_table_uses_equal_tailed_quantiles():
    frame, idata, post = _synthetic(seed=9, draws=200)
    table = a13.parameter_table(idata, 'position_disposition_detox').set_index('parameter')
    for name in ('beta_z_log_feed_rank', 'beta_toxic_persona', 'beta_log_age'):
        v = post[name].ravel()
        assert np.isclose(table.loc[name, 'eti_2.5%'], np.quantile(v, 0.025))
        assert np.isclose(table.loc[name, 'eti_97.5%'], np.quantile(v, 0.975))
        assert np.isclose(table.loc[name, 'mean'], v.mean())
    assert not any(c.startswith('hdi') for c in table.columns)
