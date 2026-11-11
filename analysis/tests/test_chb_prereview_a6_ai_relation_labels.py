"""Interval arithmetic, clustered bootstrap and 1:1 event linkage for CHB pre-review item A6."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a6_ai_relation_labels import (
    cluster_bootstrap_shares, link_events, newcombe, outcome_indicators, reconciliation,
)


def test_newcombe_matches_statsmodels_and_admitted_table():
    sm = pytest.importorskip('statsmodels.stats.proportion')
    d, lo, hi = newcombe(229, 451, 144, 271)
    ref = sm.confint_proportions_2indep(144, 271, 229, 451, method='newcomb', compare='diff')
    assert d == pytest.approx(144 / 271 - 229 / 451)
    assert (lo, hi) == pytest.approx(ref, abs=1e-12)
    # The admitted proportion_difference_intervals row (73/451 vs 41/271).
    d, lo, hi = newcombe(73, 451, 41, 271)
    assert 100 * d == pytest.approx(-1.05710148010571)
    assert 100 * lo == pytest.approx(-6.332152570997648)
    assert 100 * hi == pytest.approx(4.622623314893967)


def test_cluster_bootstrap_ratio_of_sums():
    rng = np.random.default_rng(1)
    counts = np.array([[1, 0], [1, 0], [1, 0]])
    sizes = np.array([2, 2, 2])
    shares = cluster_bootstrap_shares(counts, sizes, rng, n_boot=200)
    assert shares.shape == (200, 2)
    assert np.allclose(shares[:, 0], 0.5) and np.allclose(shares[:, 1], 0.0)
    # Unequal clusters: the bootstrap mean is near the pooled share, not the mean of cluster shares.
    counts = np.array([[9], [0]])
    sizes = np.array([10, 1])
    shares = cluster_bootstrap_shares(counts, sizes, np.random.default_rng(2), n_boot=20000)
    assert set(np.round(np.unique(shares), 6)) <= {0.0, 0.818182, 0.9}


def test_link_events_one_to_one_with_shared_key():
    cohort = pd.DataFrame(dict(group=['G1', 'G1', 'G2'], content_id=[11, 12, 13], author_id=[3, 3, 4],
                               comment_to=[7, 7, 8], round=[5, 5, 9],
                               created_at=['2026-05-12 10:01', '2026-05-12 10:30', '2026-05-12 11:00']))
    events = pd.DataFrame(dict(group=['G1', 'G1', 'G2', 'G2'], memory_event_id=[101, 100, 200, 201],
                               actor_user_id=[3, 3, 4, 4], target_post_id=[7.0, 7.0, 8.0, 9.0],
                               round_id=[5, 5, 9, 9], relation_label=['agree', 'disagree', 'hostile', 'agree'],
                               tone_label=['neutral'] * 4))
    linked = link_events(cohort, events)
    assert linked._merge.eq('both').all()
    pairs = dict(zip(linked.content_id, linked.memory_event_id))
    assert pairs == {11: 100, 12: 101, 13: 200}


def test_outcome_indicators():
    frame = pd.DataFrame(dict(relation_label=['disagree', 'hostile', 'agree', 'curious'],
                              tone_label=['snarky', 'angry', 'supportive', 'curious']))
    ind = outcome_indicators(frame)
    assert ind[('derived', 'disagree_or_hostile')].tolist() == [1, 1, 0, 0]
    assert ind[('relation_label', 'hostile')].tolist() == [0, 1, 0, 0]
    assert ind[('derived', 'tone_angry_or_snarky')].tolist() == [1, 1, 0, 0]
    assert ind[('relation_label', 'curious')].sum() == 1


def test_reconciliation_separates_missing_root_timestamp_from_off_day_comments():
    # Events 1-4 in G1: linked; outside the 08:00 windows; root timestamp missing; experiment day 31.
    events = pd.DataFrame(dict(group=['G1'] * 4 + ['G2'], memory_event_id=[1, 2, 3, 4, 5],
                               event_type=['comment'] * 5, dyad=['AI->H'] * 5, actor_user_id=[9] * 5,
                               target_post_id=[50.0, 51.0, 52.0, 53.0, 60.0], round_id=[10, 11, 12, 13, 10],
                               valid_day_index=[1.0, np.nan, 2.0, 21.0, 1.0]))
    linked = pd.DataFrame(dict(group=['G1', 'G2'], memory_event_id=[1, 5]))
    posts = pd.DataFrame(dict(group=['G1'] * 4 + ['G2'], post_id=[100, 101, 102, 103, 200],
                              author_user_id=[9] * 5, comment_to=[50, 51, 52, 53, 60], round=[10, 11, 12, 13, 10]))
    reasons = pd.DataFrame(dict(
        group=['G1'] * 3, content_id=[101, 102, 103],
        created_at=['2026-05-12 10:00:00+02:00', '2026-05-13 10:00:00+02:00', '2026-06-09 11:12:00+02:00'],
        root_thread_created_at=['2026-05-12 09:00:00+02:00', None, '2026-06-08 09:00:00+02:00'],
        experiment_day=[3.0, 4.0, 31.0], invalid_reason=['x', 'missing_timestamp', 'invalid_event_day'],
        analysis_phase=['pre_bug_valid', 'pre_bug_valid', 'outside_revision_window']))
    checks = {}
    out = reconciliation(events, linked, posts, set(), reasons, valid_days=[2, 3, 4, 30], checks=checks)
    g1 = out.loc[out.group.eq('G1')].set_index('category')['count']
    assert g1['linked_to_admitted_cohort'] == 1
    assert g1['not_in_cohort_event_round_outside_valid_day_windows'] == 1
    assert g1['not_in_cohort_root_thread_timestamp_missing'] == 1
    assert g1['not_in_cohort_comment_on_experiment_day_outside_configured_valid_days'] == 1
    assert g1['not_in_cohort_other'] == 0
    assert out.category_definition.notna().all()
    assert checks['reconciliation_off_configured_day_span']['G1']['experiment_days'] == [31]
    assert checks['reconciliation_off_configured_day_span']['G1']['public_valid_day_index'] == [21]
