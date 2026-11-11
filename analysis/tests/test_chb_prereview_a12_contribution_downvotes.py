"""Contribution frame and estimand arithmetic for A12."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a12_contribution_downvotes import (
    ROOT_SD_PRIOR, TOPIC_STEMS, VARIANTS, build_contribution_frame, build_model, concentration_checks, contrasts,
    raw_counts, topic_inputs,
)


def _inputs():
    contents = pd.DataFrame([
        # group, content_id, author_id, author_type, comment_to, root_id, valid-clock seconds
        ('G1', 1, 10, 'AI', -1, 1, 0.5 * 3600),
        ('G1', 2, 11, 'H', 1, 1, 24 * 3600),
        ('G1', 3, 11, 'h', 1, 1, 100 * 3600),
        ('G2', 1, 10, 'AI', -1, 1, 10 * 3600),   # same numeric IDs as G1: must stay separate
        ('G2', 2, 12, 'H', 1, 1, 503 * 3600),
    ], columns=['group', 'content_id', 'author_id', 'author_type', 'comment_to', 'root_id', 'content_valid_clock_seconds'])
    reactions = pd.DataFrame([
        ('G1', 'downvote', 11, 1),   # participant downvote on an AI root post
        ('G1', 'downvote', 11, 2),   # self-downvote only
        ('G1', 'downvote', 11, 3),   # self-downvote ...
        ('G1', 'downvote', 13, 3),   # ... plus another participant
        ('G1', 'upvote', 13, 2),     # upvotes never count
        ('G2', 'downvote', 12, 2),   # self-downvote only
        ('G1', 'downvote', 13, 99),  # not a valid contribution
    ], columns=['group', 'event_type', 'actor_id', 'target_content_id'])
    return contents, reactions


def test_frame_excludes_self_downvotes_and_keys_by_community():
    frame = build_contribution_frame(*_inputs(), valid_days=21).set_index('contribution_key')
    assert frame.loc['G1:1', ['y_any', 'y']].tolist() == [1, 1]
    assert frame.loc['G1:2', ['y_any', 'y', 'downvotes_self']].tolist() == [1, 0, 1]
    assert frame.loc['G1:3', ['y_any', 'y', 'downvotes_other', 'other_voters']].tolist() == [1, 1, 1, 1]
    assert frame.loc['G2:1', 'y_any'] == 0  # the G1 downvote on content 1 does not leak across communities
    assert frame.loc['G2:2', ['y_any', 'y']].tolist() == [1, 0]
    assert frame.loc['G1:3', 'cell'] == 'G1_H'
    assert bool(frame.loc['G1:1', 'is_root']) and not bool(frame.loc['G1:2', 'is_root'])


def test_followup_exposure_is_valid_hours_to_end_of_last_day():
    frame = build_contribution_frame(*_inputs(), valid_days=21).set_index('contribution_key')
    assert frame.loc['G1:1', 'followup_hours'] == pytest.approx(503.5)
    assert frame.loc['G2:2', 'followup_hours'] == pytest.approx(1.0)
    z = frame.z_log_followup
    assert z.mean() == pytest.approx(0, abs=1e-12) and z.std(ddof=0) == pytest.approx(1)
    assert z.loc['G2:2'] < z.loc['G1:1']


def test_nonpositive_followup_raises():
    contents, reactions = _inputs()
    contents.loc[0, 'content_valid_clock_seconds'] = 21 * 24 * 3600
    with pytest.raises(ValueError):
        build_contribution_frame(contents, reactions, valid_days=21)


def test_raw_counts_and_contrasts_signs():
    frame = build_contribution_frame(*_inputs(), valid_days=21)
    counts = raw_counts(frame).set_index('cell')
    assert counts.loc['G1_H', ['downvoted_any', 'downvoted_other', 'contributions_only_self_downvoted']].tolist() == [2, 1, 1]
    p = np.array([[0.04, 0.01, 0.02, 0.005]])
    values = contrasts(p)
    assert values['interaction_visible_minus_hidden_AI_gap'][0] == pytest.approx((0.04 - 0.01) - (0.02 - 0.005))
    assert values['ratio_of_risk_ratios_G1_over_G2'][0] == pytest.approx(4 / 4)
    assert values['AI_visible_minus_hidden'][0] == pytest.approx(0.02)


def test_concentration_checks_counts_positive_authors():
    frame = pd.DataFrame(dict(cell=['G1_AI'] * 3 + ['G1_H', 'G2_AI', 'G2_H'], y=[1, 1, 0, 0, 1, 0],
                              author_key=['a', 'a', 'b', 'c', 'd', 'e']))
    replicated = np.array([[1, 1, 0, 0, 1, 0], [0, 1, 1, 1, 0, 0]])
    checks = concentration_checks(frame, replicated, 'author_key').set_index(['cell', 'metric'])
    assert checks.loc[('G1_AI', 'positive_authors'), 'observed'] == 1
    assert checks.loc[('G1_AI', 'positive_authors'), 'predicted_mean'] == pytest.approx(1.5)
    assert checks.loc[('G1_AI', 'max_positives_per_author'), 'observed'] == 2


def test_topic_inputs_records_null_dir_and_hashes_files(tmp_path):
    empty = topic_inputs(None)
    assert empty['legacy_tables_dir'] is None and empty['files_sha256'] == {} and 'note' in empty
    (tmp_path / f'{TOPIC_STEMS[0]}.csv').write_text('group,root_id\n')
    (tmp_path / f'{TOPIC_STEMS[0]}.parquet').write_bytes(b'x')
    record = topic_inputs(tmp_path)
    assert list(record['files_sha256']) == [str(tmp_path / f'{TOPIC_STEMS[0]}.parquet')]  # Parquet first, as read_table


def test_wide_root_prior_changes_only_sigma_root():
    pm = pytest.importorskip('pymc')
    assert VARIANTS[:2] == ('author', 'author_root')  # existing seeds keep their variant index
    assert ROOT_SD_PRIOR == {'author_root': None, 'author_root_wide': 1.0}
    frame = build_contribution_frame(*_inputs(), valid_days=21)
    sampling = dict(prior_focal_sigma=1.0, prior_adjustment_sigma=0.5, prior_scale_sigma=0.5)

    def logp_at_one(model, name):
        return float(pm.logp(model[name], 1.0).eval())

    narrow = build_model(frame, sampling, thread=True)
    wide = build_model(frame, sampling, thread=True, root_sd_prior=1.0)
    half_normal = lambda sigma: 0.5 * np.log(2 / np.pi) - np.log(sigma) - 0.5 / sigma ** 2
    assert logp_at_one(narrow, 'sigma_root') == pytest.approx(half_normal(0.5))
    assert logp_at_one(wide, 'sigma_root') == pytest.approx(half_normal(1.0))
    assert logp_at_one(wide, 'sigma_author') == pytest.approx(half_normal(0.5))
