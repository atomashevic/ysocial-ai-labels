"""A3: distinct targets behind the other-participant downvote reply events.

Named run chb_prereview_20261110, item a3_downvote_targets. Descriptive only, no
model is fitted. The input is the public Zenodo vote panel (record version 1.1.0 or later)
(08_analysis_ready/model_inputs/rq3_vote_panel.csv), restricted with the same
code as the admitted exploratory interaction model
(scripts/diagnostics/rq3_interaction.py, ``prepare_panel``): the 2,143 reply
events in the RQ1 analysis set with round-start reaction placement, outcome
``target_other_downvote_24h`` (a downvote by a participant other than the
replier, from the reply's own hour through 24 h after the reply).

Outputs (aggregates only):

* runs/chb_prereview_20261110/tables/chb_prereview_a3_downvote_targets_cells.csv: reply events,
  positive events, distinct targets, distinct positive targets and distinct
  authors of positive targets by community and target authorship, for the
  primary reply-event cohort and the first-eligible-reply-per-target variant.
* runs/chb_prereview_20261110/tables/chb_prereview_a3_downvote_targets_events_per_target.csv:
  distribution of positive reply events (and all reply events) per positive
  target, by community and target authorship.

Usage::

    python -m scripts.diagnostics.chb_prereview_a3_downvote_targets
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a3_downvote_targets'
RUN = ROOT / 'runs' / RUN_ID / ITEM
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
PANEL = ROOT.parent / 'data/08_analysis_ready/model_inputs/rq3_vote_panel.csv'
CONDITION = {'G1': 'labels_visible', 'G2': 'labels_hidden'}
TARGET_TYPE = {'AI': 'AI-authored', 'H': 'participant-authored'}
# Reported cells (08_analysis_ready/results_tables/rq3_interaction_cells.csv, run rq3_interaction_20260907).
EXPECTED = {
    'primary': {'G1_AI': (340, 18, 10), 'G1_H': (946, 19, 8), 'G2_AI': (308, 5, 4), 'G2_H': (549, 2, 1)},
    'first_target': {'G1_AI': (264, 10, 10), 'G1_H': (491, 8, 8), 'G2_AI': (218, 4, 4), 'G2_H': (349, 1, 1)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per target: reply events, positive events and authorship cell."""
    keys = frame.target_author_code if 'target_author_code' in frame else frame.target_author_id
    out = frame.assign(author_key=frame.group + ':' + keys.astype(str)).groupby('target_key').agg(
        group=('group', 'first'), target_author_type=('target_author_type', 'first'),
        author_key=('author_key', 'first'), events=('y', 'size'), positive_events=('y', 'sum')).reset_index()
    return out


def cell_rows(frame: pd.DataFrame, variant: str, denominator: str) -> list[dict]:
    targets = target_summary(frame)
    rows = []
    for (group, kind), part in targets.groupby(['group', 'target_author_type'], sort=True):
        positive = part.loc[part.positive_events.gt(0)]
        rows.append(dict(
            run_id=RUN_ID, item=ITEM, variant=variant, group=group, condition=CONDITION[group],
            target_author_type=kind, target_authorship=TARGET_TYPE[kind],
            reply_events=int(part.events.sum()), positive_events=int(part.positive_events.sum()),
            targets=len(part), positive_targets=len(positive),
            positive_target_authors=int(positive.author_key.nunique()),
            max_positive_events_per_target=int(positive.positive_events.max()) if len(positive) else 0,
            positive_events_on_targets_with_more_than_one=int(positive.loc[positive.positive_events.gt(1), 'positive_events'].sum()),
            participants=int(frame.loc[frame.cell.eq(f'{group}_{kind}'), 'participant_key'].nunique()),
            denominator=denominator))
    total = targets.loc[targets.positive_events.gt(0)]
    rows.append(dict(
        run_id=RUN_ID, item=ITEM, variant=variant, group='ALL', condition='both', target_author_type='ALL',
        target_authorship='all', reply_events=int(targets.events.sum()), positive_events=int(targets.positive_events.sum()),
        targets=len(targets), positive_targets=len(total), positive_target_authors=int(total.author_key.nunique()),
        max_positive_events_per_target=int(total.positive_events.max()),
        positive_events_on_targets_with_more_than_one=int(total.loc[total.positive_events.gt(1), 'positive_events'].sum()),
        participants=int(frame.participant_key.nunique()), denominator=denominator))
    return rows


def events_per_target_rows(frame: pd.DataFrame, denominator: str) -> list[dict]:
    targets = target_summary(frame)
    positive = targets.loc[targets.positive_events.gt(0)]
    rows = []
    for (group, kind), part in positive.groupby(['group', 'target_author_type'], sort=True):
        for value, n in part.positive_events.value_counts().sort_index().items():
            sub = part.loc[part.positive_events.eq(value)]
            rows.append(dict(run_id=RUN_ID, item=ITEM, variant='primary', group=group, condition=CONDITION[group],
                             target_author_type=kind, positive_events_on_target=int(value), positive_targets=int(n),
                             positive_events=int(value * n), reply_events_on_these_targets=int(sub.events.sum()),
                             cell_positive_targets=len(part), cell_positive_events=int(part.positive_events.sum()),
                             denominator=denominator))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=PANEL)
    args = parser.parse_args()
    from scripts.diagnostics import rq3_interaction as rq3

    started = dt.datetime.now().astimezone()
    log(f'{ITEM}: start, panel {args.panel}')
    panel = pd.read_csv(args.panel)
    frames = {'primary': rq3.prepare_panel(panel), 'first_target': rq3.prepare_panel(panel, first_target=True)}
    checks = {}
    for variant, frame in frames.items():
        support = rq3.cell_summary(frame).set_index('cell')
        for cell, (events, positives, positive_targets) in EXPECTED[variant].items():
            observed = (int(support.events[cell]), int(support.positives[cell]), int(support.positive_targets[cell]))
            checks[f'{variant}:{cell}:events_positives_positive_targets'] = observed == (events, positives, positive_targets)
    checks['primary_events_2143'] = len(frames['primary']) == 2143
    checks['primary_positives_44'] = int(frames['primary'].y.sum()) == 44
    checks['primary_positive_targets_23'] = frames['primary'].loc[frames['primary'].y.eq(1), 'target_key'].nunique() == 23
    checks['first_target_rows_1322'] = len(frames['first_target']) == 1322
    checks['first_target_positives_23'] = int(frames['first_target'].y.sum()) == 23
    # Every positive target in the primary cohort is positive at its first eligible reply.
    first_pos = set(frames['first_target'].loc[frames['first_target'].y.eq(1), 'target_key'])
    prim_pos = set(frames['primary'].loc[frames['primary'].y.eq(1), 'target_key'])
    checks['first_target_positive_set_equals_primary_positive_set'] = first_pos == prim_pos
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        log(f'{ITEM}: checks failed {failed}')
        raise SystemExit(1)
    denom_primary = ('2,143 reply events in the RQ1 analysis set (G1 AI 340, G1 H 946, G2 AI 308, G2 H 549) '
                     'to 1,322 distinct targets; 44 positive events')
    denom_first = '1,322 distinct targets, first eligible reply per target (G1 AI 264, G1 H 491, G2 AI 218, G2 H 349); 23 positives'
    cells = pd.DataFrame(cell_rows(frames['primary'], 'primary', denom_primary)
                         + cell_rows(frames['first_target'], 'first_target', denom_first))
    per_target = pd.DataFrame(events_per_target_rows(frames['primary'], denom_primary))
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = {'cells': TABLES / f'chb_prereview_{ITEM}_cells.csv',
               'events_per_target': TABLES / f'chb_prereview_{ITEM}_events_per_target.csv'}
    cells.to_csv(outputs['cells'], index=False)
    per_target.to_csv(outputs['events_per_target'], index=False)
    log('\n' + cells.drop(columns=['denominator', 'run_id', 'item']).to_string(index=False))
    log('\n' + per_target.drop(columns=['denominator', 'run_id', 'item']).to_string(index=False))
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='Distinct targets behind the 44 positive other-participant downvote reply events (CHB pre-review A3).',
        host=socket.gethostname(), command=' '.join([sys.executable, '-m', 'scripts.diagnostics.chb_prereview_a3_downvote_targets'] + sys.argv[1:]),
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        imported_modules={'scripts/diagnostics/rq3_interaction.py': sha256(ROOT / 'scripts/diagnostics/rq3_interaction.py')},
        inputs={str(args.panel.relative_to(ROOT.parent) if args.panel.is_relative_to(ROOT.parent) else args.panel): sha256(args.panel)},
        outputs={k: dict(path=str(v.relative_to(ROOT.parent)), sha256=sha256(v)) for k, v in outputs.items()},
        packages=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        started=started.isoformat(timespec='seconds'), finished=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        diagnostics=dict(checks=checks, model_fitted=False))
    (RUN / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    log(f'{ITEM}: done, all {len(checks)} checks pass')


if __name__ == '__main__':
    main()
