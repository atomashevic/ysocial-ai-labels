from __future__ import annotations

import pandas as pd

from scripts.diagnostics.chb_prereview_a3_downvote_targets import cell_rows, events_per_target_rows


def _frame() -> pd.DataFrame:
    rows = [  # group, target, author type, author code, participant, y
        ('G1', 1, 'AI', 'A1', 1, 1), ('G1', 1, 'AI', 'A1', 2, 1), ('G1', 1, 'AI', 'A1', 3, 0),
        ('G1', 2, 'AI', 'A1', 1, 1), ('G1', 3, 'H', 'P1', 2, 0), ('G2', 1, 'AI', 'A9', 4, 1),
    ]
    frame = pd.DataFrame(rows, columns=['group', 'tid', 'target_author_type', 'target_author_code', 'pid', 'y'])
    frame['target_key'] = frame.group + ':' + frame.tid.astype(str)
    frame['participant_key'] = frame.group + ':' + frame.pid.astype(str)
    frame['cell'] = frame.group + '_' + frame.target_author_type
    return frame


def test_cells_count_distinct_targets_within_community() -> None:
    cells = pd.DataFrame(cell_rows(_frame(), 'primary', 'toy')).set_index(['group', 'target_author_type'])
    g1 = cells.loc[('G1', 'AI')]
    assert (g1.reply_events, g1.positive_events, g1.targets, g1.positive_targets) == (4, 3, 2, 2)
    assert (g1.positive_target_authors, g1.max_positive_events_per_target) == (1, 2)
    assert g1.positive_events_on_targets_with_more_than_one == 2
    # Content id 1 exists in both communities; keys keep them apart.
    total = cells.loc[('ALL', 'ALL')]
    assert (total.targets, total.positive_targets, total.positive_target_authors) == (4, 3, 2)


def test_events_per_target_distribution() -> None:
    dist = pd.DataFrame(events_per_target_rows(_frame(), 'toy'))
    g1 = dist.loc[dist.group.eq('G1')].set_index('positive_events_on_target')
    assert g1.positive_targets.to_dict() == {1: 1, 2: 1}
    assert g1.loc[2, 'reply_events_on_these_targets'] == 3
