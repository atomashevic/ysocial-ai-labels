"""A4: timing of participant comments and direct-reply targets by voucher progress.

Named run chb_prereview_20261110, item a4_incentive_timing. Descriptive only.

(i) Hour of day (Europe/Belgrade) of participant comments, by community, on two
    bases: the analysed cohort (``_valid_contents`` on the 21 analysed calendar
    dates, 1,518 / 1,056 comments) and the released 08:00-08:00 activity
    windows (Zenodo ``posts.csv``, 1,717 / 1,223). Shares created 07:00-08:00 and
    08:00-09:00 carry Wilson 95% intervals; the hidden-minus-visible difference
    carries a Newcombe hybrid-score interval.

(ii) Among the 2,143 analysed reply-target choices, the share whose selected
    target is a direct reply to the replier's own earlier contribution
    (D = 1 for the chosen candidate; definition of ``direct_reply`` in
    scripts/diagnostics/rq2_contact.py: the target is a comment whose parent was
    authored by the replier), by the replier's cumulative comment count at reply
    time (<56, 56-83, >=84; the voucher thresholds) and community, with Wilson
    95% intervals. The count is the number of the replier's comment rows created
    strictly before the reply. The dashboard counted every non-removed comment row
    (no length or quality rule), so the dashboard basis counts all comment rows on
    the whole platform timeline, including warm-up and the excluded interval; it is
    computed on the private host by the ``private-counts`` stage from the full
    10 June snapshot (aggregate-free per-event counts keyed by public content ids).
    The released-window basis counts only comment rows in the public record.

Stages::

    # on the private host (single thread, pandas only, no repository imports)
    python chb_prereview_a4_incentive_timing.py private-counts --snapshot S --panel P --removed R --out O
    # from the public record
    python -m scripts.diagnostics.chb_prereview_a4_incentive_timing run --private-counts O/event_counts.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import sys
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a4_incentive_timing'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
TREE = ROOT.parent / 'data'
POSTS = TREE / '02_platform_activity/posts.csv'
PANEL = TREE / '08_analysis_ready/model_inputs/rq3_vote_panel.csv'
FRAME = TREE / '08_analysis_ready/model_inputs/reply_observed_design_frame.csv'
# Direct-child candidate rows (same table, column candidates, direct_child = 1).
EXPECTED_DIRECT_CANDIDATES = {'G1': 404 + 1477, 'G2': 350 + 1217}
PUBLIC_CONFIG = ROOT / 'analysis.public-rq3.yaml'
TZ = 'Europe/Belgrade'
GROUPS = ('G1', 'G2')
CONDITION = {'G1': 'labels_visible', 'G2': 'labels_hidden'}
TIER1, UPPER = 56, 84
BANDS = ('lt56', '56_to_83', 'ge84')
BAND_LABEL = {'lt56': '<56', '56_to_83': '56-83', 'ge84': '>=84'}
Z95 = 1.959963984540054
# Selected direct-child candidates (study table review_choice_models_rq2_direct_child_candidates.csv).
EXPECTED_DIRECT_SELECTED = {'G1': 132 + 271, 'G2': 111 + 213}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def log(message: str, run: Path = RUN) -> None:
    run.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (run / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


# --------------------------------------------------------------------------- arithmetic

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p2 - p1 with the Newcombe hybrid-score 95% interval."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def band(count: int) -> str:
    if count < TIER1:
        return 'lt56'
    if count < UPPER:
        return '56_to_83'
    return 'ge84'


def is_comment(comment_to: pd.Series) -> pd.Series:
    values = pd.to_numeric(comment_to, errors='coerce')
    return values.notna() & values.ne(-1)


def naive_utc(values: pd.Series) -> np.ndarray:
    """Tz-aware timestamps as naive UTC datetime64 for sorting and searching."""
    return pd.to_datetime(values, utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')


def prior_counts(events: pd.DataFrame, comments: pd.DataFrame) -> pd.Series:
    """Number of the actor's comment rows created strictly before each event.

    ``events``: group, actor_id, event_time (tz-aware), indexed by event.
    ``comments``: group, author_id, created (tz-aware); one row per counted comment.
    """
    out = pd.Series(0, index=events.index, dtype='int64')
    for (group, actor), part in events.groupby(['group', 'actor_id']):
        times = np.sort(naive_utc(comments.loc[comments.group.eq(group) & comments.author_id.eq(actor), 'created']))
        out.loc[part.index] = np.searchsorted(times, naive_utc(part.event_time), side='left')
    return out


def own_prior_counts(events: pd.DataFrame) -> pd.Series:
    """Number of the actor's other events (by exact event time) strictly before each event."""
    out = pd.Series(0, index=events.index, dtype='int64')
    for _, part in events.groupby(['group', 'actor_id']):
        times = np.sort(naive_utc(part.event_time))
        out.loc[part.index] = np.searchsorted(times, naive_utc(part.event_time), side='left')
    return out


def direct_reply_flag(events: pd.DataFrame, contents: pd.DataFrame) -> pd.Series:
    """D = 1 when the selected target is a comment whose parent the replier authored."""
    lookup = contents.set_index(['group', 'content_id'])
    if not lookup.index.is_unique:
        raise ValueError('Duplicate content ids')
    flags = []
    for row in events.itertuples(index=False):
        target = lookup.loc[(row.group, int(row.target_id))]
        parent = target.comment_to
        if pd.isna(parent) or int(parent) == -1:
            flags.append(0)
            continue
        key = (row.group, int(parent))
        if key not in lookup.index:
            raise ValueError(f'Unresolved parent of selected target {row.group}:{row.target_id}')
        flags.append(int(int(lookup.loc[key, 'author_id']) == int(row.actor_id)))
    return pd.Series(flags, index=events.index, dtype='int64')


# --------------------------------------------------------------------------- private stage

def private_counts(args) -> None:
    """Runs on the private host. Writes per-event counts keyed by public content ids only."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now().astimezone()
    snap = pd.read_parquet(args.snapshot, columns=['group', 'content_id', 'author_id', 'author_type', 'comment_to', 'created_at'])
    snap['group'] = snap.group.astype(str)
    snap['created'] = pd.to_datetime(snap.created_at, utc=True, format='mixed')
    removed = pd.read_csv(args.removed)
    removed_keys = set(zip(removed.group.astype(str), removed.post_id.astype(int)))
    snap['removed_in_release'] = [(g, int(c)) in removed_keys for g, c in zip(snap.group, snap.content_id)]
    panel = pd.read_csv(args.panel)
    flag = panel.in_rq1_analysis_set.astype(str).str.lower().eq('true') & panel.reaction_placement.eq('round_start')
    ev = panel.loc[flag, ['group', 'event_id', 'actor_id', 'reply_content_id', 'event_timestamp']].copy()
    ev['event_time'] = pd.to_datetime(ev.event_timestamp, utc=True, format='mixed')
    ev['actor_id'] = ev.actor_id.astype(int)
    checks = {'events_2143': len(ev) == 2143}
    key = snap.set_index(['group', 'content_id'])
    found = [(g, int(c)) in key.index for g, c in zip(ev.group, ev.reply_content_id)]
    checks['all_replies_in_snapshot'] = all(found)
    match = key.loc[list(zip(ev.group, ev.reply_content_id.astype(int)))]
    checks['reply_author_matches_actor'] = bool((match.author_id.astype(int).to_numpy() == ev.actor_id.to_numpy()).all())
    checks['reply_time_matches_event_time'] = bool((np.abs(naive_utc(match.created.reset_index(drop=True)) - naive_utc(ev.event_time.reset_index(drop=True)))
                                                 < np.timedelta64(1, 'ms')).all())
    comments = snap.loc[is_comment(snap.comment_to)].copy()
    comments['author_id'] = comments.author_id.astype(int)
    nonremoved = comments.loc[~comments.removed_in_release]
    ev['cum_comment_rows_before_all'] = prior_counts(ev, comments)
    ev['cum_comment_rows_before_nonremoved'] = prior_counts(ev, nonremoved)
    ev[['group', 'event_id', 'reply_content_id', 'cum_comment_rows_before_all', 'cum_comment_rows_before_nonremoved']].to_csv(
        out / 'event_counts.csv', index=False)
    actors = set(zip(ev.group, ev.actor_id))
    mine = comments.loc[[(g, a) in actors for g, a in zip(comments.group, comments.author_id)]]
    start, fault_start, fault_end = (pd.Timestamp(t, tz=TZ) for t in ('2026-05-11 08:00', '2026-05-19 08:00', '2026-05-28 08:00'))
    period = np.select([mine.created.lt(start), mine.created.ge(fault_start) & mine.created.lt(fault_end)],
                       ['before_2026-05-11T08', 'excluded_2026-05-19T08_to_05-28T08'], 'released_windows_or_after')
    summary = dict(
        snapshot=str(args.snapshot), snapshot_sha256=sha256(Path(args.snapshot)), panel_sha256=sha256(Path(args.panel)),
        removed_sha256=sha256(Path(args.removed)), host=socket.gethostname(), started=started.isoformat(timespec='seconds'),
        finished=dt.datetime.now().astimezone().isoformat(timespec='seconds'), checks=checks,
        snapshot_rows=len(snap), replier_comment_rows_by_group_period=mine.assign(period=period).groupby(['group', 'period']).size()
        .rename('rows').reset_index().to_dict(orient='records'),
        removed_rows_matched_in_snapshot=int(snap.removed_in_release.sum()),
        packages=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))
    (out / 'private_summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    if not all(checks.values()):
        raise SystemExit('private checks failed')


# --------------------------------------------------------------------------- public stage

def hour_tables(analysed: pd.DataFrame, released: pd.DataFrame):
    rows, summary = [], []
    bases = {'analysed_calendar_dates': ('participant comments on the 21 analysed calendar dates (_valid_contents)', analysed),
             'released_activity_windows': ('participant comments in the 21 released 08:00-08:00 windows (Zenodo posts.csv)', released)}
    for basis, (definition, frame) in bases.items():
        counts = {}
        for group in GROUPS:
            hours = frame.loc[frame.group.eq(group), 'hour']
            n = len(hours)
            vc = hours.value_counts().reindex(range(24), fill_value=0)
            for hour, k in vc.items():
                lo, hi = wilson(int(k), n)
                rows.append(dict(run_id=RUN_ID, item=ITEM, basis=basis, basis_definition=definition, group=group,
                                 condition=CONDITION[group], hour_start=int(hour), comments=int(k), denominator_comments=n,
                                 share=k / n, wilson_low=lo, wilson_high=hi, denominator=f'{n} participant comments'))
            counts[group] = (n, int(vc[7]), int(vc[8]), int(vc[0:8].sum()))
        for label, idx in (('07:00-08:00', 1), ('08:00-09:00', 2), ('00:00-08:00', 3)):
            (n1, *c1), (n2, *c2) = counts['G1'], counts['G2']
            k1, k2 = c1[idx - 1], c2[idx - 1]
            d, dlo, dhi = newcombe(k1, n1, k2, n2)
            for group, k, n in (('G1', k1, n1), ('G2', k2, n2)):
                lo, hi = wilson(k, n)
                summary.append(dict(run_id=RUN_ID, item=ITEM, basis=basis, interval=label, group=group,
                                    condition=CONDITION[group], comments=k, denominator_comments=n, share=k / n,
                                    wilson_low=lo, wilson_high=hi, hidden_minus_visible_pp=100 * d,
                                    hidden_minus_visible_low_pp=100 * dlo, hidden_minus_visible_high_pp=100 * dhi,
                                    contrast_sign='G2 (hidden) minus G1 (visible)',
                                    denominator=f'{n1} (G1) and {n2} (G2) participant comments; {definition}'))
    return pd.DataFrame(rows), pd.DataFrame(summary)


def band_table(events: pd.DataFrame, count_column: str, basis: str, definition: str) -> pd.DataFrame:
    rows = []
    frame = events.assign(band=events[count_column].map(band))
    total = {g: int(frame.group.eq(g).sum()) for g in GROUPS}
    for group in GROUPS + ('ALL',):
        part = frame if group == 'ALL' else frame.loc[frame.group.eq(group)]
        for b in BANDS + ('all',):
            sub = part if b == 'all' else part.loc[part.band.eq(b)]
            n, k = len(sub), int(sub.direct_reply.sum())
            lo, hi = wilson(k, n)
            avail = sub.loc[sub.direct_candidates.gt(0)]
            na, ka = len(avail), int(avail.direct_reply.sum())
            alo, ahi = wilson(ka, na)
            rows.append(dict(run_id=RUN_ID, item=ITEM, count_basis=basis, count_basis_definition=definition,
                             group=group, condition=CONDITION.get(group, 'both'), cumulative_comment_band=BAND_LABEL.get(b, 'all'),
                             reply_choices=n, direct_reply_targets=k, share=k / n if n else math.nan,
                             wilson_low=lo, wilson_high=hi,
                             uniform_choice_expected_share=float(sub.direct_candidate_fraction.mean()) if n else math.nan,
                             mean_candidates=float(sub.candidates.mean()) if n else math.nan,
                             choices_with_direct_candidate=na,
                             share_given_direct_candidate=ka / na if na else math.nan,
                             given_direct_candidate_wilson_low=alo, given_direct_candidate_wilson_high=ahi,
                             participants=int(sub.actor_key.nunique()),
                             denominator=f'2,143 analysed reply-target choices (G1 {total["G1"]}, G2 {total["G2"]})'))
    return pd.DataFrame(rows)


def run(args) -> None:
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents

    started = dt.datetime.now().astimezone()
    log(f'{ITEM}: run stage start')
    checks = {}
    config = load_config(PUBLIC_CONFIG, profile='release', run_id=RUN_ID)
    valid = _valid_contents(config)
    valid['is_comment'] = is_comment(valid.comment_to)
    analysed = valid.loc[valid.author_type.eq('H') & valid.is_comment].copy()
    analysed['hour'] = pd.to_datetime(analysed.created_at, utc=True, format='mixed').dt.tz_convert(TZ).dt.hour
    checks['analysed_participant_comments_1518_1056'] = analysed.group.value_counts().to_dict() == {'G1': 1518, 'G2': 1056}

    posts = pd.read_csv(POSTS, usecols=['group', 'post_id', 'post_kind', 'author_user_id', 'author_class', 'comment_to',
                                        'created_at', 'admin_removed'])
    posts['created'] = pd.to_datetime(posts.created_at, utc=True, format='mixed')
    released = posts.loc[posts.author_class.eq('human') & posts.post_kind.eq('comment')].copy()
    released['hour'] = released.created.dt.tz_convert(TZ).dt.hour
    checks['released_participant_comments_1717_1223'] = released.group.value_counts().to_dict() == {'G1': 1717, 'G2': 1223}
    checks['released_comment_rows_none_removed'] = not bool(released.admin_removed.any())
    hours, hour_summary = hour_tables(analysed, released)

    panel = pd.read_csv(PANEL)
    flag = panel.in_rq1_analysis_set.astype(str).str.lower().eq('true') & panel.reaction_placement.eq('round_start')
    ev = panel.loc[flag].copy()
    ev['event_time'] = pd.to_datetime(ev.event_timestamp, utc=True, format='mixed')
    ev['actor_id'] = ev.actor_id.astype(int)
    ev['target_id'] = ev.selected_target_content_id.astype(int)
    ev['actor_key'] = ev.group + ':' + ev.actor_id.astype(str)
    checks['events_2143'] = len(ev) == 2143
    contents = posts.rename(columns={'post_id': 'content_id', 'author_user_id': 'author_id'})[
        ['group', 'content_id', 'author_id', 'comment_to']]
    ev['direct_reply'] = direct_reply_flag(ev, contents)
    observed = ev.groupby('group').direct_reply.sum().to_dict()
    checks['direct_reply_selected_matches_review_choice_models'] = observed == EXPECTED_DIRECT_SELECTED
    # Candidate availability: D for every earlier valid non-self same-thread candidate.
    cand = pd.read_csv(FRAME, usecols=['choice_set_id', 'group', 'actor_id', 'candidate_content_id', 'chosen'])
    cand = cand.rename(columns={'candidate_content_id': 'target_id'})
    cand['direct'] = direct_reply_flag(cand, contents)
    checks['candidate_rows_17489'] = len(cand) == 17489
    checks['direct_candidates_match_review_choice_models'] = cand.groupby('group').direct.sum().to_dict() == EXPECTED_DIRECT_CANDIDATES
    chosen = cand.loc[cand.chosen.astype(str).str.lower().eq('true')]
    checks['one_chosen_per_set'] = chosen.choice_set_id.is_unique and len(chosen) == 2143
    per_set = cand.groupby('choice_set_id').agg(candidates=('direct', 'size'), direct_candidates=('direct', 'sum'))
    per_set['direct_candidate_fraction'] = per_set.direct_candidates / per_set.candidates
    ev['choice_set_id'] = ev.group + ':' + ev.event_id
    ev = ev.join(per_set, on='choice_set_id', validate='one_to_one')
    checks['every_event_has_choice_set'] = bool(ev.candidates.notna().all())
    chosen_flag = chosen.set_index('choice_set_id').direct.reindex(ev.choice_set_id).to_numpy()
    checks['chosen_candidate_direct_equals_event_flag'] = bool((chosen_flag == ev.direct_reply.to_numpy()).all())
    # Public released-window basis.
    rel_comments = posts.loc[is_comment(posts.comment_to) & ~posts.admin_removed].rename(columns={'author_user_id': 'author_id'})
    # Released timestamps carry milliseconds while reply event times carry
    # microseconds, so a reply's own row can sort just before its event time;
    # drop each reply's own row before counting.
    own = set(zip(ev.group, ev.reply_content_id.astype(int)))
    others = rel_comments.loc[[(g, int(c)) not in own for g, c in zip(rel_comments.group, rel_comments.post_id)]]
    own_rows = rel_comments.loc[[(g, int(c)) in own for g, c in zip(rel_comments.group, rel_comments.post_id)]]
    checks['released_contains_every_reply_row'] = len(own_rows) == len(ev)
    ev['cum_released'] = prior_counts(ev, others[['group', 'author_id', 'created']])
    # Other analysed replies by the same actor count as prior when created earlier.
    ev['cum_released'] += own_prior_counts(ev)
    tables = [band_table(ev, 'cum_released', 'released_windows',
                         "replier's comment rows in the public record (21 released 08:00-08:00 windows) created before the reply")]
    private_meta = None
    if args.private_counts:
        private = pd.read_csv(args.private_counts)
        merged = ev.merge(private, on=['group', 'event_id'], how='left', validate='one_to_one', suffixes=('', '_private'))
        checks['private_counts_cover_all_events'] = bool(merged.cum_comment_rows_before_nonremoved.notna().all())
        checks['private_reply_ids_match'] = bool((merged.reply_content_id == merged.reply_content_id_private).all())
        checks['private_counts_at_least_released'] = bool((merged.cum_comment_rows_before_nonremoved >= merged.cum_released).all())
        ev = merged
        tables.insert(0, band_table(ev, 'cum_comment_rows_before_nonremoved', 'dashboard_all_comment_rows',
                                    "replier's non-removed comment rows on the whole platform timeline (warm-up, released "
                                    "windows and excluded interval) created before the reply; dashboard counting basis"))
        summary_path = Path(args.private_counts).with_name('private_summary.json')
        private_meta = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if private_meta:
            checks.update({f'private:{k}': v for k, v in private_meta['checks'].items()})
    bands = pd.concat(tables, ignore_index=True)
    # Participant-level reach of the thresholds (dashboard basis when available).
    count_col = 'cum_comment_rows_before_nonremoved' if args.private_counts else 'cum_released'
    reach = ev.groupby(['group', 'actor_key'])[count_col].max().reset_index()
    if not all(checks.values()):
        log(f'{ITEM}: checks failed {[k for k, v in checks.items() if not v]}')
        raise SystemExit(1)
    outputs = {'hour_of_day': TABLES / f'chb_prereview_{ITEM}_hour_of_day.csv',
               'boundary_hours': TABLES / f'chb_prereview_{ITEM}_boundary_hours.csv',
               'direct_reply_by_band': TABLES / f'chb_prereview_{ITEM}_direct_reply_by_band.csv'}
    hours.to_csv(outputs['hour_of_day'], index=False)
    hour_summary.to_csv(outputs['boundary_hours'], index=False)
    bands.to_csv(outputs['direct_reply_by_band'], index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    reach.groupby('group')[count_col].describe().to_csv(OUT / 'max_prior_count_by_participant_summary.csv')
    log('\n' + hour_summary.drop(columns=['run_id', 'item', 'denominator']).to_string(index=False))
    log('\n' + bands.drop(columns=['run_id', 'item', 'denominator', 'count_basis_definition']).to_string(index=False))
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='Hour of day of participant comments relative to the 08:00 boundary and direct-reply target share by '
                'voucher-threshold band (CHB pre-review A4).',
        host=socket.gethostname(), command=' '.join([sys.executable, '-m', 'scripts.diagnostics.chb_prereview_a4_incentive_timing'] + sys.argv[1:]),
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        imported_modules={p: sha256(ROOT / p) for p in ('scripts/publication_analysis/data.py', 'scripts/publication_analysis/config.py')},
        inputs={str(POSTS.relative_to(ROOT.parent)): sha256(POSTS), str(PANEL.relative_to(ROOT.parent)): sha256(PANEL),
                str(FRAME.relative_to(ROOT.parent)): sha256(FRAME),
                str(PUBLIC_CONFIG): sha256(PUBLIC_CONFIG),
                str(Path(config.processed_dir) / 'snapshot_contents.csv'): sha256(Path(config.processed_dir) / 'snapshot_contents.csv'),
                **({os.path.relpath(args.private_counts, ROOT.parent): sha256(Path(args.private_counts))} if args.private_counts else {})},
        private_stage=private_meta,
        private_stage_host_and_command=(
            'private host: cd WORK/a4_incentive_timing && OMP_NUM_THREADS=1 nice -n 10 '
            'python -I code/chb_prereview_a4_incentive_timing.py private-counts '
            '--snapshot PRIVATE_PROCESSED/ysocial_grupa1_grupa2_snapshot_2026-06-10T09-44-59_0200/'
            'snapshot_contents.parquet --panel inputs/rq3_vote_panel.csv --removed inputs/released_removed_rows.csv --out out'),
        # The private stage ran from an earlier revision of this file; only the public stage changed afterwards.
        private_stage_script_sha256='5cc9a4d5659cdd5885bdab580d7fd3f1783aff3ad66540ed29ee24132d52750f',
        outputs={k: dict(path=str(v.relative_to(ROOT.parent)), sha256=sha256(v)) for k, v in outputs.items()},
        packages=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        started=started.isoformat(timespec='seconds'), finished=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        dashboard_counting_basis='every non-removed comment row counts toward the 56 and 84 thresholds; no length or quality '
                                 'rule (task brief, item F7); the live dashboard used midnight calendar days whereas the '
                                 'protocol activity day ran 08:00-08:00',
        diagnostics=dict(checks=checks, model_fitted=False))
    (RUN / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    log(f'{ITEM}: done, all {len(checks)} checks pass')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('private-counts')
    p.add_argument('--snapshot', required=True)
    p.add_argument('--panel', required=True)
    p.add_argument('--removed', required=True)
    p.add_argument('--out', required=True)
    r = sub.add_parser('run')
    r.add_argument('--private-counts', type=Path, default=TREE / '08_analysis_ready/further_checks/a4_incentive_timing/event_counts.csv',
                   help='per-event counts written by the private-counts stage (deposited in the Zenodo record)')
    r.add_argument('--write-removed', action='store_true', help='write the public removed-row list for the private stage and exit')
    args = parser.parse_args()
    if args.stage == 'private-counts':
        private_counts(args)
    elif args.write_removed:
        posts = pd.read_csv(POSTS, usecols=['group', 'post_id', 'admin_removed'])
        OUT.mkdir(parents=True, exist_ok=True)
        posts.loc[posts.admin_removed, ['group', 'post_id']].to_csv(OUT / 'released_removed_rows.csv', index=False)
        print(OUT / 'released_removed_rows.csv')
    else:
        run(args)


if __name__ == '__main__':
    main()
