import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics import chb_prereview_a20_analysis as a20a


def _frame(effect=0.0, contexts=6, draws=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(contexts):
        stratum = 'extremely' if c % 2 else 'absolutely no'
        for d in range(draws):
            base = rng.normal()
            for arm in ('on', 'off'):
                rows.append(dict(key=f'c{c:02d}|{arm}|{d}', context_id=f'c{c:02d}', arm=arm, draw=d, stratum=stratum,
                                 seeds_sha256=f's{c}{d}', y=base + (effect if arm == 'on' else 0.0),
                                 w=int(not (arm == 'off' and d == 0)), selected_style='TROLL' if d % 2 else 'QUESTION'))
    return pd.DataFrame(rows)


def test_pair_frame_aligns_arms_and_checks_seeds():
    frame = _frame().sample(frac=1, random_state=1)
    wide = a20a.pair_frame(frame)
    assert len(wide) == 24
    assert (wide.key_on.str.replace('|on|', '|') == wide.key_off.str.replace('|off|', '|')).all()
    bad = frame.copy()
    bad.loc[bad.key.eq('c00|off|1'), 'seeds_sha256'] = 'other'
    with pytest.raises(ValueError, match='share seeds'):
        a20a.pair_frame(bad)
    with pytest.raises(ValueError, match='both arms'):
        a20a.pair_frame(frame.loc[frame.key.ne('c01|on|2')])


def test_conditional_difference_and_subset_counts():
    wide = a20a.pair_frame(_frame(effect=0.5))
    engine = a20a.Engine(wide)
    est = engine.estimate('y', 'w')
    on = wide.y_on.mean()
    off = wide.loc[wide.w_off.eq(1), 'y_off'].mean()
    assert est['difference'] == pytest.approx(on - off)
    assert (est['n_on'], est['n_off'], est['pairs_both_in_subset']) == (24, 18, 18)
    assert est['ci_low'] < est['difference'] < est['ci_high']


def test_permutation_swaps_exchange_whole_records():
    wide = a20a.pair_frame(_frame(effect=0.0))
    arrays = a20a.arm_arrays(wide, 'y', 'w')
    swaps = np.zeros((2, len(wide)))
    swaps[1, :] = 1.0  # swapping every pair negates the difference
    stats = a20a.permuted(arrays, swaps)
    observed = a20a.diff_stat(arrays['s_on'].sum(), arrays['n_on'].sum(), arrays['s_off'].sum(), arrays['n_off'].sum())
    assert stats[0] == pytest.approx(observed)
    assert stats[1] == pytest.approx(-observed)
    one = np.zeros((1, len(wide)))
    one[0, 0] = 1.0  # swapping pair 0 moves its OFF record (w = 0) into ON
    s_on = arrays['s_on'].sum() - arrays['s_on'][0] + arrays['s_off'][0]
    n_on = arrays['n_on'].sum() - arrays['n_on'][0] + arrays['n_off'][0]
    s_off = arrays['s_off'].sum() - arrays['s_off'][0] + arrays['s_on'][0]
    n_off = arrays['n_off'].sum() - arrays['n_off'][0] + arrays['n_on'][0]
    assert a20a.permuted(arrays, one)[0] == pytest.approx(s_on / n_on - s_off / n_off)


def test_permutation_p_detects_effect_and_not_null():
    strong = a20a.Engine(a20a.pair_frame(_frame(effect=3.0, contexts=10))).estimate('y', 'w')
    null = a20a.Engine(a20a.pair_frame(_frame(effect=0.0, contexts=10))).estimate('y', 'w')
    assert strong['perm_p_pair_swap'] < 0.001
    assert null['perm_p_pair_swap'] > 0.05
    assert a20a.perm_pvalue(1.0, np.array([0.5, -2.0, 1.0])) == pytest.approx(3 / 4)


def test_context_swaps_move_all_draws_of_a_context_together():
    clusters = np.array([0, 0, 1, 1, 1, 2])
    swaps = a20a.context_swaps(clusters, 3, n=50)
    for row in swaps:
        assert row[0] == row[1] and row[2] == row[3] == row[4]


def test_bootstrap_counts_resample_clusters():
    counts = a20a.bootstrap_counts(5, n=200)
    assert counts.shape == (200, 5)
    assert (counts.sum(1) == 5).all()


def test_style_chi_square_and_agreement():
    frame = _frame()
    wide = a20a.pair_frame(frame)
    chi, p = a20a.style_permutation(wide, ['QUESTION', 'TROLL'], a20a.swap_matrix(len(wide), 200))
    assert chi == pytest.approx(0.0) and p == pytest.approx(1.0)
    assert a20a.chi_square(np.array([10, 0]), np.array([0, 10]))[0] == pytest.approx(20.0)
    rows = {r['comparison']: r for r in a20a.agreement_rows(frame, 'selected_style', a20a.bootstrap_counts(6, n=20),
                                                            sorted(frame.context_id.unique()))}
    assert rows['crn_pair']['share'] == 1.0 and rows['crn_pair']['denominator'] == 24
    assert rows['within_on']['denominator'] == 36  # 6 contexts x C(4, 2)


def test_marker_strip_and_masking():
    text = 'IDENTITY MARKERS:\nSome entries may include [human], meaning the account is a human participant.\n' \
           '@cool_cat12 [human] you are wrong, ask @nobody_here or G1-P03'
    assert a20a.marker_echo(text) == {'token_echo': True, 'note_echo': True}
    clean = a20a.strip_marker_text(text)
    assert a20a.marker_echo(clean) == {'token_echo': False, 'note_echo': False}
    masked, counts = a20a.mask_draft(clean, {'7': 'A', '9': 'B'}, {'cool_cat12': '7'})
    assert masked == '@Speaker A you are wrong, ask @another user or another user'
    assert counts == {'bare_name_matches': 0, 'at_name_matches': 1, 'other_handles': 1}
    assert a20a.leaks(masked, {'cool_cat12': '7'}) == []


def test_speaker_letters_follow_a15_order():
    posts = {('G1', 1): dict(post_id=1, comment_to=-1, author_code='9'),
             ('G1', 2): dict(post_id=2, comment_to=1, author_code='5'),
             ('G1', 3): dict(post_id=3, comment_to=2, author_code='7')}
    chain = a20a.draft_chain(posts, 'G1', 3)
    assert [r['post_id'] for r in chain] == [3, 2, 1]
    assert a20a.speaker_letters(chain, '7', '5') == {'7': 'A', '9': 'B', '5': 'C'}


def test_blind_plan_is_deterministic_and_mixes_arms():
    keys = [f'c{c:02d}|{arm}|{d}' for c in range(10) for arm in ('on', 'off') for d in range(4)]
    key_of, plan = a20a.blind_plan(keys)
    assert key_of == a20a.blind_plan(list(reversed(keys)))[0]
    assert sorted(key_of.values()) == sorted(keys)
    first = [key_of[i] for i in plan['main'][0]]
    assert {k.split('|')[1] for k in first} == {'on', 'off'}
    assert sum(map(len, plan['selfcheck'])) == 8
