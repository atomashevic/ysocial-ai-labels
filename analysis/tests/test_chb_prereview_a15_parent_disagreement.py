import math

import numpy as np
import pandas as pd

from scripts.diagnostics import chb_prereview_a15_parent_disagreement as a15


def _posts():
    rows = [
        dict(group='G1', post_id=1, comment_to=-1, author_code='G1-P01', text='TITLE: root by P01'),
        dict(group='G1', post_id=2, comment_to=1, author_code='G1-A05', text='first reply'),
        dict(group='G1', post_id=3, comment_to=2, author_code='G1-P02', text='second'),
        dict(group='G1', post_id=4, comment_to=3, author_code='G1-A07', text='third'),
        dict(group='G1', post_id=5, comment_to=4, author_code='G1-P03', text='fourth'),
        dict(group='G1', post_id=6, comment_to=5, author_code='G1-A05', text='@G1-P02 wrong, ask cool_cat12'),
    ]
    return {(r['group'], r['post_id']): r for r in rows}


def test_context_window_adds_focal_contribution_above_window():
    posts = _posts()
    chain = a15.ancestors(posts, 'G1', 6)
    assert [r['post_id'] for r in chain] == [5, 4, 3, 2, 1]
    root, shown, gap, extra, structural = a15.context_rows(chain, 'G1-A05')
    assert root['post_id'] == 1
    assert [r['post_id'] for r in shown] == [3, 4, 5]  # oldest first: answered (5) plus two earlier
    assert gap and extra['post_id'] == 2
    assert structural == 'participant_earlier_in_branch'
    _, _, _, _, structural = a15.context_rows(chain, 'G1-P03')
    assert structural == 'answers_participant'
    _, _, _, extra, structural = a15.context_rows(chain, 'G1-P09')
    assert extra is None and structural == 'participant_not_in_branch'


def test_masking_replaces_codes_and_usernames():
    names = {'cool_cat12': 'G1-A09'}
    speaker_of = {'G1-P02': 'A', 'G1-A05': 'B'}
    text = a15.mask_text('@G1-P02 wrong, ask cool_cat12 or G1-P11', speaker_of, names)
    assert text == '@Speaker A wrong, ask another user or another user'
    assert a15.mask_text('cool_cat12', {'G1-A09': 'C'}, names) == 'Speaker C'


def test_build_item_never_shows_focal_reply_and_letters_focal_a():
    posts = _posts()
    pair = pd.Series(dict(group='G1', parent_content_id=4, author_code='G1-P03', pair_id='p0001'))
    item = a15.build_item(pair, posts, {})
    shown_texts = [c['text'] for c in item['context']] + [item['target']['text']]
    assert 'fourth' not in shown_texts  # post 5 is the focal reply to parent 4
    assert item['target']['text'] == 'third'
    assert item['answered_speaker'] != 'A'  # parent 4 answers P02, not P03
    assert item['structural'] == 'participant_not_in_branch'


def test_wilson_and_kappa():
    low, high = a15.wilson(10, 100)
    assert math.isclose(low, 0.05523, abs_tol=1e-4) and math.isclose(high, 0.17437, abs_tol=1e-4)
    a = np.array(['YES', 'YES', 'NO', 'NO'])
    assert a15.cohen_kappa(a, a) == 1.0
    assert math.isclose(a15.cohen_kappa(a, np.array(['YES', 'NO', 'NO', 'NO'])), 0.5)


def test_reply_indicator_root_is_structural_zero():
    frame = pd.DataFrame(dict(group=['G1'] * 3, parent_content_id=[1, 4, 4], author_code=['P', 'P', 'Q'],
                              parent_kind=['root', 'comment', 'comment']))
    pairs = frame.assign(pair_id=['p0', 'p1', 'p2'])
    labels = pd.DataFrame(dict(item_id=['p1', 'p2'], label=['YES', 'NO']))
    assert a15.reply_indicator(frame, pairs, labels).tolist() == [0, 1, 0]


def test_validate_rejects_inconsistent_rows():
    good = {'results': [dict(item_id='p1', label='YES', category='content', evidence='not true')]}
    assert a15.validate(good, ['p1'])[0]['label'] == 'YES'
    bad = {'results': [dict(item_id='p1', label='YES', category='none', evidence='x')]}
    try:
        a15.validate(bad, ['p1'])
    except ValueError:
        pass
    else:
        raise AssertionError('inconsistent label accepted')


def test_clipping_table_counts_clipped_texts_by_role():
    long = 'x' * (a15.TEXT_CHARS + 10)
    items = [
        dict(item_id='p0000', target=dict(text='short'),
             context=[dict(position='thread_root', role='contribution answered by the target',
                           text=a15.clip(long, a15.TEXT_CHARS))]),
        dict(item_id='p0001', target=dict(text=a15.clip(long, a15.TEXT_CHARS)),
             context=[dict(position='thread_root', role='thread root post', text=a15.clip(long, a15.ROOT_CHARS)),
                      dict(position='omitted', role='earlier contributions in this branch omitted', text=''),
                      dict(position='ancestor', role='contribution answered by the target', text='ok')]),
    ]
    pairs = pd.DataFrame(dict(pair_id=['p0000', 'p0001'], group=['G1', 'G1'], parent_author_type=['AI', 'AI']))
    labels = pd.DataFrame(dict(item_id=['p0000', 'p0001'], label=['YES', 'NO']))
    table = a15.clipping_table(items, pairs, labels).set_index('text_role')
    assert table.loc['answered', 'clipped_texts'] == 1 and table.loc['answered', 'texts_shown'] == 2
    assert table.loc['answered', 'clipped_texts_in_yes_pairs'] == 1
    assert table.loc['target', 'clipped_texts'] == 1 and table.loc['target', 'clipped_texts_in_yes_pairs'] == 0
    assert table.loc['thread_root_context', 'limit_chars'] == a15.ROOT_CHARS
    assert set(table.index) == {'answered', 'target', 'thread_root_context'}
    assert (table.annotated_pairs == 2).all()
