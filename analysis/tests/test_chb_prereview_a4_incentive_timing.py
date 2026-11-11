from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a4_incentive_timing import (
    band, direct_reply_flag, newcombe, own_prior_counts, prior_counts, wilson)


def test_bands_follow_voucher_thresholds() -> None:
    assert [band(c) for c in (0, 55, 56, 83, 84, 200)] == ['lt56', 'lt56', '56_to_83', '56_to_83', 'ge84', 'ge84']


def test_wilson_and_newcombe_match_reference_values() -> None:
    lo, hi = wilson(30, 432)
    assert (lo, hi) == pytest.approx((0.04906, 0.09748), abs=1e-4)
    # proportion_difference_intervals.csv: 30/432 vs 20/395 gives -1.9 (-5.2 to 1.4) points.
    d, dlo, dhi = newcombe(30, 432, 20, 395)
    assert (100 * d, 100 * dlo, 100 * dhi) == pytest.approx((-1.881, -5.2, 1.4), abs=0.06)
    assert all(math.isnan(v) for v in wilson(0, 0))


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz='Europe/Belgrade')


def test_prior_counts_are_strict_and_per_actor_and_group() -> None:
    comments = pd.DataFrame({'group': ['G1', 'G1', 'G1', 'G2'], 'author_id': [1, 1, 2, 1],
                             'created': [_ts('2026-05-11 09:00'), _ts('2026-05-11 10:00'),
                                         _ts('2026-05-11 08:30'), _ts('2026-05-11 08:00')]})
    events = pd.DataFrame({'group': ['G1', 'G1', 'G2'], 'actor_id': [1, 1, 1],
                           'event_time': [_ts('2026-05-11 10:00'), _ts('2026-05-11 11:00'), _ts('2026-05-11 09:00')]})
    assert prior_counts(events, comments).tolist() == [1, 2, 1]
    assert own_prior_counts(events).tolist() == [0, 1, 0]


def test_direct_reply_flag_uses_parent_author_of_selected_target() -> None:
    contents = pd.DataFrame({'group': ['G1'] * 4, 'content_id': [1, 2, 3, 4], 'author_id': [10, 20, 30, 20],
                             'comment_to': [-1, 1, 2, 3]})
    events = pd.DataFrame({'group': ['G1'] * 4, 'actor_id': [20, 20, 10, 30], 'target_id': [3, 1, 2, 4]})
    # Target 3 replies to 2 (author 20): D = 1 for actor 20. Root target: D = 0.
    # Target 2 replies to the root by 10: D = 1 for actor 10. Target 4 replies to 3 by 30: D = 1.
    assert direct_reply_flag(events, contents).tolist() == [1, 0, 1, 1]
    with pytest.raises(ValueError):
        direct_reply_flag(events.assign(target_id=[3, 1, 2, 4]),
                          contents.loc[contents.content_id.ne(2)])
