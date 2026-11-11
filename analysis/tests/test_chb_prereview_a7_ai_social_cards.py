"""Card classification, decay, summaries and snapshot timing for CHB pre-review item A7."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a7_ai_social_cards import (
    bootstrap_final, classify, completed_valid_days, decayed, snapshot_round, snapshot_time, summarize,
    trajectory_summary,
)

ACCOUNTS = pd.DataFrame(dict(group=['G1'] * 4 + ['G2'] * 3, user_id=[1, 2, 3, 4, 1, 2, 3],
                             actor_class=['ai', 'ai', 'human', 'admin', 'ai', 'ai', 'human']))


def _cards():
    return pd.DataFrame(dict(
        group=['G1', 'G1', 'G1', 'G2', 'G2'], agent_user_id=[1, 1, 2, 1, 2], other_user_id=[2, 3, 3, 3, 1],
        affinity=[0.5, -0.3, 1.0, 0.2, 0.0], conflict=[0.1, 0.6, -0.1, 0.4, 0.0], humor=[0.0] * 5,
        trust=[0.2, -0.2, 0.3, 0.0, 0.1], last_updated_round=[10, 20, 30, 30, 10], event_count=[1, 0, 2, 1, 3]))


def test_classify_uses_account_table_and_checks_stored_dyad():
    cards = classify(_cards(), ACCOUNTS)
    assert cards.dyad.tolist() == ['AI->AI', 'AI->H', 'AI->H', 'AI->H', 'AI->AI']
    bad = _cards().assign(dyad=['AI->H'] * 5)
    with pytest.raises(AssertionError):
        classify(bad, ACCOUNTS)
    with pytest.raises(AssertionError):  # a card about an admin account is not expected
        classify(_cards().assign(other_user_id=[4, 3, 3, 3, 1]), ACCOUNTS)


def test_decay_matches_client_formula():
    out = decayed(pd.Series([1.0, 2.0]), pd.Series([10, 40]), 30)
    assert out.tolist() == pytest.approx([math.exp(-0.05 * 20), 2.0])


def test_summarize_shares_and_counts():
    rows = {(r['group'], r['dyad']): r for r in summarize(classify(_cards(), ACCOUNTS), at_round=30)}
    g1h = rows[('G1', 'AI->H')]
    assert g1h['n_cards'] == 2 and g1h['n_ai_accounts'] == 2 and g1h['n_cards_event_count_0'] == 1
    assert g1h['share_conflict_gt_affinity'] == pytest.approx(0.5)
    assert g1h['affinity_mean'] == pytest.approx(0.35)
    assert rows[('G2', 'AI->AI')]['share_conflict_gt_affinity'] == 0.0


def test_bootstrap_point_values_and_sign():
    cards = classify(_cards(), ACCOUNTS)
    boot = bootstrap_final(cards, np.random.default_rng(0), n_boot=500).set_index(['dyad', 'statistic'])
    row = boot.loc[('AI->H', 'affinity_mean')]
    assert row.value_g1 == pytest.approx(0.35) and row.value_g2 == pytest.approx(0.2)
    assert row.difference == pytest.approx(0.2 - 0.35)  # hidden minus visible
    assert row.bootstrap_low <= row.bootstrap_high


def test_snapshot_timing():
    when = snapshot_time('ysocial_grupa1_grupa2_snapshot_2026-06-10T09-44-59_0200')
    assert when.isoformat() == '2026-06-10T09:44:59+02:00'
    assert snapshot_time('ysocial_grupa1_grupa2_snapshot_2026-05-30T12-59-35_0200_roundts') is None
    rounds = pd.DataFrame(dict(round=[761, 762, 763], round_start=['2026-06-10T07:00:00+02:00',
                                                                   '2026-06-10T08:00:00+02:00',
                                                                   '2026-06-10T09:00:00+02:00']))
    assert snapshot_round(when, rounds) == 763
    windows = pd.DataFrame(dict(window_end=['2026-06-09T08:00:00.000+02:00', '2026-06-10T08:00:00.000+02:00',
                                            '2026-06-11T08:00:00.000+02:00']))
    assert completed_valid_days(when, windows) == 2


def test_trajectory_summary_reports_path_not_only_endpoints():
    snaps = ['s1', 's2', 's3']
    rows = []
    for i, snap in enumerate(snaps):
        for group, shares in (('G1', [0.5, 0.2, 0.3]), ('G2', [0.6, 0.3, 0.4])):
            rows.append(dict(snapshot=snap, group=group, dyad='AI->H', taken_in_excluded_window=(i == 1),
                             share_conflict_gt_affinity=shares[i], conflict_mean=shares[i] / 2,
                             affinity_mean=1 - shares[i], trust_mean=0.1))
    out = trajectory_summary(pd.DataFrame(rows))['share_conflict_gt_affinity']
    assert out['G1'] == dict(first=0.5, min=0.2, min_snapshot='s2', min_in_excluded_window=True, final=0.3)
    assert out['G2']['min'] == 0.3 and out['snapshots'] == 3 and out['snapshots_g2_above_g1'] == 3
    aff = trajectory_summary(pd.DataFrame(rows))['affinity_mean']
    assert aff['G1']['max'] == pytest.approx(0.8) and aff['G1']['max_snapshot'] == 's2'
