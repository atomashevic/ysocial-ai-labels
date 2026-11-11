"""Validation-stratum mapping for the A10 content-criticism measurement layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a10_content_misclass import (
    attach_plug_in, content_validation_table, implied_error_rates, outcome_column,
)


def _frame():
    rows = []

    def add(n, group, target, stratum, content_target=None):
        for _ in range(n):
            rows.append(dict(group=group, target_type=target, stratum=stratum, content_target=content_target))

    for group in ('G1', 'G2'):
        for target in ('AI', 'H'):
            add(20, group, target, 'NEG')
            add(6, group, target, 'POS_CONTENT', 'DIRECT_PARENT')
            add(3, group, target, 'POS_SCP', 'DIRECT_PARENT')
            add(2, group, target, 'POS_SCP')
    add(1, 'G1', 'AI', 'NEG_NOTPARENT')
    frame = pd.DataFrame(rows)
    frame['cell'] = frame.group + '_' + frame.target_type
    return frame


def _strata():
    rows = []
    for group in ('G1', 'G2'):
        for target in ('AI', 'H'):
            base = dict(group=group, parent_author_type=target, in_criticism_model=True)
            rows += [dict(base, sol_stratum='NEGATIVE', direct_parent_criticism='NO', content_criticism='NO', human='NO', items=4),
                     dict(base, sol_stratum='NEGATIVE', direct_parent_criticism='NO', content_criticism='NO', human='YES', items=1),
                     dict(base, sol_stratum='CONTENT_ONLY', direct_parent_criticism='YES', content_criticism='YES', human='YES', items=2),
                     dict(base, sol_stratum='CONTENT_ONLY', direct_parent_criticism='YES', content_criticism='YES', human='NO', items=1),
                     dict(base, sol_stratum='STYLE_OR_PARTICIPATION', direct_parent_criticism='YES', content_criticism='YES', human='YES', items=3),
                     dict(base, sol_stratum='STYLE_OR_PARTICIPATION', direct_parent_criticism='YES', content_criticism='NO', human='YES', items=2)]
    # Items outside the model cohort never enter.
    rows.append(dict(group='G1', parent_author_type='AI', in_criticism_model=False, sol_stratum='CONTENT_ONLY',
                     direct_parent_criticism='', content_criticism='YES', human='YES', items=5))
    return pd.DataFrame(rows)


def test_content_parent_mapping_partitions_cohort_and_keeps_scp_content_positives():
    frame = _frame()
    table = content_validation_table(frame, _strata(), 'content_parent')
    assert table.N.sum() == len(frame)
    cell = table.loc[table.group.eq('G1') & table.target_type.eq('AI')].set_index('stratum')
    assert (cell.loc['POS_CONTENT', ['N', 'n', 'x']] == [6, 3, 2]).all()
    assert (cell.loc['POS_SCP', ['N', 'n', 'x']] == [3, 3, 3]).all()  # SCP positives with content at parent
    assert (cell.loc['NEG', ['N', 'n', 'x']] == [20, 5, 1]).all()
    assert cell.loc['NEG_NOTPARENT', 'N'] == 3  # two SCP-only positives and one non-parent Sol positive
    positives = table.loc[table.stratum.isin(['POS_CONTENT', 'POS_SCP']), 'N'].sum()
    assert positives == outcome_column(frame, 'content_parent').sum() == 36


def test_content_only_mapping_treats_all_scp_positives_as_correct_negatives():
    frame = _frame()
    table = content_validation_table(frame, _strata(), 'content_only')
    assert set(table.stratum) == {'POS_CONTENT', 'NEG', 'NEG_NOTPARENT'}
    cell = table.loc[table.group.eq('G2') & table.target_type.eq('H')].set_index('stratum')
    assert cell.loc['NEG_NOTPARENT', 'N'] == 5 and cell.loc['NEG_NOTPARENT', 'n'] == 0
    assert table.loc[table.stratum.eq('POS_CONTENT'), 'N'].sum() == outcome_column(frame, 'content_only').sum() == 24


def test_implied_error_rates_match_hand_calculation():
    table = content_validation_table(_frame(), _strata(), 'content_only')
    rates = implied_error_rates(table).set_index('cell')
    tp = (1 + 2) / (2 + 3) * 6
    fn = (1 + 1) / (2 + 5) * 20
    n_neg = 20 + 5
    r = rates.loc['G2_H']
    assert r.sensitivity == pytest.approx(tp / (tp + fn))
    assert r.specificity == pytest.approx((n_neg - fn) / (n_neg - fn + 6 - tp))
    assert r.ppv == pytest.approx(tp / 6)


def test_ambiguous_scp_census_raises():
    strata = _strata()
    # One content-coded SCP item rated NO while counts differ from N: cannot assign it to the parent stratum.
    extra = strata.loc[strata.sol_stratum.eq('STYLE_OR_PARTICIPATION') & strata.content_criticism.eq('YES')].iloc[[0]].copy()
    extra['human'], extra['items'] = 'NO', 1
    with pytest.raises(ValueError):
        content_validation_table(_frame(), pd.concat([strata, extra], ignore_index=True), 'content_parent')


def test_plug_in_values_sit_beside_measurement_rows_only():
    rates = implied_error_rates(content_validation_table(_frame(), _strata(), 'content_only'))
    estimands = pd.DataFrame(dict(
        estimand=['G1_AI', 'sensitivity_G2_H', 'specificity_G1_AI'], scale=['true_content_criticism', 'measurement', 'measurement'],
        estimate=[0.2, 0.5, 0.9]))
    out = attach_plug_in(estimands, rates).set_index('estimand')
    assert np.isnan(out.loc['G1_AI', 'plug_in_validation'])
    assert out.loc['sensitivity_G2_H', 'plug_in_validation'] == pytest.approx(rates.set_index('cell').loc['G2_H', 'sensitivity'])
    assert out.loc['specificity_G1_AI', 'plug_in_validation'] == pytest.approx(rates.set_index('cell').loc['G1_AI', 'specificity'])
    assert (out.estimate == estimands.set_index('estimand').estimate).all()
