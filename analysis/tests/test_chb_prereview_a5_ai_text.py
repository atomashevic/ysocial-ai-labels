from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.chb_prereview_a5_ai_text import (
    cluster_bootstrap, mtld, segment_ttrs, text_statistics, tokenize, ttr_first)


def test_tokenize_lowercases_and_drops_urls_and_handles() -> None:
    assert tokenize("Don't @G1-P03 see https://x.org/a WHY?") == ["don't", 'see', 'why']
    assert tokenize('It’s fine') == ["it's", 'fine']


def test_mtld_hand_computed_values() -> None:
    # Forward and backward: a, a -> TTR 0.5 completes a factor, twice; 4 tokens / 2 factors.
    assert mtld(['a'] * 4) == pytest.approx(2.0)
    # No repeated token: no factor in either direction, so MTLD is undefined.
    assert math.isnan(mtld(['a', 'b', 'c']))
    # a b a: forward TTR 1, 1, 2/3 -> one factor, 3 tokens; backward identical.
    assert mtld(['a', 'b', 'a']) == pytest.approx(3.0)
    # a b c a: forward TTR 3/4 at end -> partial factor (1-.75)/.28; backward same sequence reversed.
    assert mtld(['a', 'b', 'c', 'a']) == pytest.approx(4 / (0.25 / 0.28))


def test_ttr_first50_and_segments() -> None:
    tokens = [f't{i % 25}' for i in range(120)]
    assert ttr_first(tokens) == pytest.approx(25 / 50)
    assert math.isnan(ttr_first(tokens[:49]))
    assert segment_ttrs(tokens) == pytest.approx([0.5, 0.5])


def test_cluster_bootstrap_sign_is_hidden_minus_visible() -> None:
    texts = pd.DataFrame({'group': ['G1'] * 4 + ['G2'] * 4, 'cluster': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                          'chars': [100] * 4 + [90] * 4, 'words': [20] * 8, 'mtld': [50.] * 8,
                          'ttr50': [np.nan] * 8, 'compound': [0.] * 8, 'question': [False] * 4 + [True] * 4})
    accounts = pd.DataFrame({'group': ['G1', 'G1', 'G2', 'G2'], 'cluster': ['a', 'b', 'c', 'd'],
                             'account_mtld': [10., 10., 12., 12.], 'segment_ttrs': [[.8], [.8], [.9], [.9]]})
    boot = cluster_bootstrap(texts, accounts, replicates=20, seed=1)
    assert boot.median_characters.eq(-10).all()
    assert boot.share_question_mark.eq(1).all()
    assert boot.mean_account_mtld.eq(2).all()
    assert boot.msttr50.tolist() == pytest.approx([0.1] * 20)
    stats = text_statistics(texts.loc[texts.group.eq('G1')])
    assert stats['share_vader_negative'] == 0 and math.isnan(stats['mean_ttr_first50'])
