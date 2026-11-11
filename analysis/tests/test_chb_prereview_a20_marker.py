import hashlib
import inspect
import json

import pytest

from scripts.diagnostics import chb_prereview_a20_marker as a20

FRAME_COUNTS = {'absolutely no': 219, 'no': 10, 'moderately': 94, 'extremely': 128}


def frame_rows(counts=FRAME_COUNTS):
    rows, cid = [], 1000
    for level in a20.DISPOSITIONS:
        for _ in range(counts[level]):
            rows.append({'content_id': cid, 'disposition': level})
            cid += 7
    return rows


def test_allocation_is_proportional_and_sums_to_n():
    alloc = a20.allocate(FRAME_COUNTS, 60)
    assert alloc == {'absolutely no': 29, 'no': 1, 'moderately': 13, 'extremely': 17}
    assert sum(alloc.values()) == 60
    total = sum(FRAME_COUNTS.values())
    for level, k in alloc.items():
        assert abs(k - 60 * FRAME_COUNTS[level] / total) < 1


def test_allocation_rejects_oversampling():
    with pytest.raises(ValueError):
        a20.allocate({'no': 2}, 3, order=('no',))


def test_sample_is_deterministic_stratified_and_order_free():
    rows = frame_rows()
    first, alloc = a20.stratified_sample(rows, 60, seed=a20.SEED)
    again, _ = a20.stratified_sample(list(reversed(rows)), 60, seed=a20.SEED)
    assert [r['content_id'] for r in first] == [r['content_id'] for r in again]
    assert [r['context_id'] for r in first] == [f'c{i:02d}' for i in range(60)]
    assert len({r['content_id'] for r in first}) == 60
    by_level = {}
    for r in first:
        by_level[r['disposition']] = by_level.get(r['disposition'], 0) + 1
    assert by_level == alloc
    other, _ = a20.stratified_sample(rows, 60, seed=a20.SEED + 1)
    assert [r['content_id'] for r in other] != [r['content_id'] for r in first]


def test_draw_seeds_are_shared_by_arms_and_vary_by_draw_and_context():
    assert 'arm' not in inspect.signature(a20.draw_seeds).parameters
    s = a20.draw_seeds(a20.SEED, 'c00', 0)
    assert set(s) == {'python', 'numpy', 'style_llm', 'shuffle', 'override'}
    assert s == a20.draw_seeds(a20.SEED, 'c00', 0)
    assert s != a20.draw_seeds(a20.SEED, 'c00', 1)
    assert s != a20.draw_seeds(a20.SEED, 'c01', 0)
    expected = int.from_bytes(hashlib.sha256(f'{a20.SEED}|c00|0|shuffle'.encode()).digest()[:4], 'big') & 0x7FFFFFFF
    assert s['shuffle'] == expected
    assert all(0 <= v < 2**31 for v in s.values())
    assert len(set(s.values())) == len(s)


def test_jobs_keep_arms_adjacent_and_skip_done():
    jobs = a20.plan_jobs(['c00', 'c01'], 2)
    assert len(jobs) == 8
    for i in range(0, len(jobs), 2):
        (c1, d1, a1), (c2, d2, a2) = jobs[i], jobs[i + 1]
        assert (c1, d1) == (c2, d2) and (a1, a2) == a20.ARMS
    done = {a20.record_key('c00', 'on', 0), a20.record_key('c00', 'off', 0)}
    rest = a20.plan_jobs(['c00', 'c01'], 2, done)
    assert ('c00', 0, 'on') not in rest and len(rest) == 6


def test_completed_keys_ignores_errors_and_truncated_lines(tmp_path):
    path = tmp_path / 'draws.jsonl'
    lines = [json.dumps({'key': 'c00|on|0', 'status': 'posted'}),
             json.dumps({'key': 'c00|off|0', 'status': 'error'}),
             json.dumps({'key': 'c01|on|0', 'status': 'not_posted'}),
             '{"key": "c01|off|0", "sta']
    path.write_text('\n'.join(lines))
    assert a20.completed_keys(path) == {'c00|on|0', 'c01|on|0'}
    assert a20.completed_keys(tmp_path / 'missing.jsonl') == set()


def _prompt(marker: bool, branch: str) -> str:
    tag = a20.MARKER_TOKEN if marker else ''
    sections = ['TARGET PARENT:\n@ann' + tag + ' [rep: top 25%] - is the sequel better?']
    if marker:
        sections.append(a20.MARKER_NOTE)
    sections.append('LOCAL SIBLINGS:\n@bot_7 [rep: middle 50%] - ' + branch)
    return '\n\n'.join(sections)


def test_marker_check_on_matched_prompts():
    on, off = _prompt(True, 'no, the pacing drags'), _prompt(False, 'no, the pacing drags')
    res = a20.compare_prompts(on, off)
    assert res['on_has_marker_token'] and res['on_has_note']
    assert not res['off_has_marker_token'] and not res['off_has_note']
    assert res['identical_after_strip']


def test_marker_check_flags_leaks_and_classifies_truncation():
    leaked = a20.compare_prompts(_prompt(True, 'x'), _prompt(True, 'x'))
    assert leaked['off_has_marker_token'] and leaked['off_has_note']
    full = 'no, the pacing drags in the middle act'
    truncated = a20.compare_prompts(_prompt(True, full[:12].rstrip() + '...'), _prompt(False, full))
    assert not truncated['identical_after_strip']
    assert truncated['difference'] == 'truncation_only'
    changed = a20.compare_prompts(_prompt(True, 'yes'), _prompt(False, 'no'))
    assert changed['difference'] == 'other'
