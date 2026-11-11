"""AI accounts' social cards about participants and other AI accounts (CHB pre-review item A7).

Named run ``chb_prereview_20261110``, item ``a7_ai_social_cards``.

Each AI account keeps one social card per other account it has commented to or
voted on. A card holds affinity, conflict, humor and trust, each clipped to
[-5, 5]. On every update the YSocial client first decays the stored values by
exp(-0.05 x rounds since the card's last update) (``memory_social_decay_lambda``
0.05 per hourly round), then adds the deltas from the AI account's own memory
note (comments) or fixed vote deltas (upvote: affinity +0.5, trust +0.2,
conflict -0.1; downvote: conflict +0.6, affinity -0.3, trust -0.2). Votes update
the values without counting as events (``memory_vote_signal_only`` true). The
stored values are therefore post-decay as of each card's last update; reads for
prompts and relationship ranking use the four stored values without further
decay. Relationship ranking also adds a separate recency term, exp(-0.05 x
rounds since the last interaction), to its priority score (YSocial client
``base_agent.py``, about l.1998-2012 and l.11586-11613 in the run source on the
private host); that term does not alter the stored values.

The final snapshot (10 June 09:44) is taken in round 763 and so includes
rounds 762-763, after the last public valid-day window closed at 08:00; cards
last updated in those rounds are counted in the run record (none is an
AI-to-participant card).

Two stages:

``extract`` (private host, one thread): reads ``snapshot_memory_social_cards.csv``
from every daily snapshot directory and writes aggregate rows only (counts,
means, quantiles, shares) by snapshot, community and dyad. Account classes come
from the public ``01_accounts/accounts.csv`` and must agree with the snapshot's
own type columns.

``finalize`` (public record): reads the extracted aggregates, recomputes the final
snapshot from the public Zenodo cards, checks that the two agree, adds an
AI-account clustered bootstrap (10,000 replicates) for hidden-minus-visible
differences at the final snapshot and writes the tables.

Sign convention: hidden minus visible (G2 minus G1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a7_ai_social_cards'
GROUPS = ('G1', 'G2')
DYADS = ('AI->H', 'AI->AI')
METRICS = ('affinity', 'conflict', 'humor', 'trust')
DECAY_LAMBDA = 0.05
EXPECTED_FINAL = {('G1', 'AI->H'): 405, ('G2', 'AI->H'): 234, ('G1', 'AI->AI'): 625, ('G2', 'AI->AI'): 422}
SNAPSHOT_RE = re.compile(r'^ysocial_grupa1_grupa2_snapshot_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_0200$')
CONTRAST = 'G2_minus_G1 (hidden minus visible)'
N_BOOT = 10_000
SEED = 20261110
CARD_COLUMNS = ['group', 'agent_user_id', 'other_user_id', 'agent_type', 'other_type', 'dyad', *METRICS,
                'last_updated_round', 'event_count']
TZ = timezone(timedelta(hours=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot_time(name: str) -> datetime | None:
    match = SNAPSHOT_RE.match(name)
    if not match:
        return None
    return datetime.strptime(match.group(1), '%Y-%m-%dT%H-%M-%S').replace(tzinfo=TZ)


def snapshot_round(when: datetime, rounds: pd.DataFrame) -> int:
    """Hourly round whose start is the latest one at or before ``when``."""
    starts = pd.to_datetime(rounds.round_start, utc=True)
    keep = starts.le(pd.Timestamp(when).tz_convert('UTC'))
    return int(rounds.loc[keep, 'round'].max())


def completed_valid_days(when: datetime, windows: pd.DataFrame) -> int:
    ends = pd.to_datetime(windows.window_end, utc=True)
    return int(ends.le(pd.Timestamp(when).tz_convert('UTC')).sum())


def in_excluded_window(when: datetime, excluded: pd.DataFrame) -> bool:
    ts = pd.Timestamp(when).tz_convert('UTC')
    starts = pd.to_datetime(excluded.window_start, utc=True)
    ends = pd.to_datetime(excluded.window_end, utc=True)
    return bool((starts.le(ts) & ends.gt(ts)).any())


def classify(cards: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative account classes and check them against the stored types."""
    cls = accounts[['group', 'user_id', 'actor_class']]
    out = cards.merge(cls.rename(columns={'user_id': 'agent_user_id', 'actor_class': 'agent_class'}),
                      on=['group', 'agent_user_id'], how='left', validate='many_to_one')
    out = out.merge(cls.rename(columns={'user_id': 'other_user_id', 'actor_class': 'other_class'}),
                    on=['group', 'other_user_id'], how='left', validate='many_to_one')
    assert out.agent_class.eq('ai').all(), 'every card owner must be an AI account'
    assert out.other_class.isin({'ai', 'human'}).all(), 'card subjects must be AI or participant accounts'
    out['dyad_authoritative'] = np.where(out.other_class.eq('ai'), 'AI->AI', 'AI->H')
    if 'dyad' in out.columns:
        assert out.dyad.eq(out.dyad_authoritative).all(), 'stored dyad disagrees with account classes'
    out['dyad'] = out.dyad_authoritative
    return out.drop(columns=['dyad_authoritative'])


def decayed(values: pd.Series, last_round: pd.Series, at_round: int) -> pd.Series:
    delta = (at_round - last_round.astype(float)).clip(lower=0)
    return values * np.exp(-DECAY_LAMBDA * delta)


def summarize(cards: pd.DataFrame, at_round: int) -> list[dict]:
    """Aggregate rows by community and dyad. ``at_round`` is the snapshot's hourly round."""
    rows = []
    for group in GROUPS:
        for dyad in DYADS:
            c = cards.loc[cards.group.eq(group) & cards.dyad.eq(dyad)]
            row = dict(group=group, dyad=dyad, n_cards=len(c), n_ai_accounts=int(c.agent_user_id.nunique()),
                       n_other_accounts=int(c.other_user_id.nunique()),
                       n_cards_event_count_0=int(c.event_count.fillna(0).eq(0).sum()),
                       event_count_mean=float(c.event_count.fillna(0).mean()) if len(c) else np.nan,
                       share_conflict_gt_affinity=float((c.conflict > c.affinity).mean()) if len(c) else np.nan,
                       share_conflict_gt_0=float((c.conflict > 0).mean()) if len(c) else np.nan,
                       share_affinity_gt_0=float((c.affinity > 0).mean()) if len(c) else np.nan,
                       share_trust_gt_0=float((c.trust > 0).mean()) if len(c) else np.nan,
                       rounds_since_update_median=float((at_round - c.last_updated_round).median()) if len(c) else np.nan)
            for m in METRICS:
                v = c[m].astype(float)
                q1, med, q3 = (v.quantile([0.25, 0.5, 0.75]).tolist() if len(v) else [np.nan] * 3)
                row.update({f'{m}_mean': float(v.mean()) if len(v) else np.nan, f'{m}_median': med,
                            f'{m}_q1': q1, f'{m}_q3': q3, f'{m}_sd': float(v.std(ddof=1)) if len(v) > 1 else np.nan})
                dv = decayed(v, c.last_updated_round, at_round)
                row[f'{m}_mean_decayed_to_snapshot'] = float(dv.mean()) if len(v) else np.nan
            dc = decayed(c.conflict.astype(float), c.last_updated_round, at_round)
            da = decayed(c.affinity.astype(float), c.last_updated_round, at_round)
            row['share_conflict_gt_affinity_decayed_to_snapshot'] = float((dc > da).mean()) if len(c) else np.nan
            rows.append(row)
    return rows


def bootstrap_final(cards: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Hidden-minus-visible differences at one snapshot, AI accounts resampled within community."""
    stats = {f'{m}_mean': (lambda c, m=m: c[m].to_numpy(float)) for m in METRICS}
    stats['share_conflict_gt_affinity'] = lambda c: (c.conflict > c.affinity).to_numpy(float)
    rows = []
    for dyad in DYADS:
        per_group = {}
        for group in GROUPS:
            c = cards.loc[cards.group.eq(group) & cards.dyad.eq(dyad)]
            agents = np.sort(c.agent_user_id.unique())
            index = pd.Index(agents)
            pos = index.get_indexer(c.agent_user_id)
            sizes = np.bincount(pos, minlength=len(agents)).astype(float)
            sums = {k: np.bincount(pos, weights=f(c), minlength=len(agents)) for k, f in stats.items()}
            draws = rng.integers(0, len(agents), size=(n_boot, len(agents)))
            weights = np.zeros((n_boot, len(agents)))
            np.add.at(weights, (np.repeat(np.arange(n_boot), len(agents)), draws.ravel()), 1.0)
            den = weights @ sizes
            per_group[group] = dict(n=len(c), clusters=len(agents),
                                    point={k: float(v.sum() / sizes.sum()) for k, v in sums.items()},
                                    boot={k: (weights @ v) / den for k, v in sums.items()})
        for k in stats:
            diff = per_group['G2']['boot'][k] - per_group['G1']['boot'][k]
            rows.append(dict(dyad=dyad, statistic=k, value_g1=per_group['G1']['point'][k],
                             value_g2=per_group['G2']['point'][k],
                             difference=per_group['G2']['point'][k] - per_group['G1']['point'][k],
                             bootstrap_low=float(np.quantile(diff, 0.025)),
                             bootstrap_high=float(np.quantile(diff, 0.975)),
                             n_cards_g1=per_group['G1']['n'], n_cards_g2=per_group['G2']['n'],
                             ai_accounts_g1=per_group['G1']['clusters'], ai_accounts_g2=per_group['G2']['clusters']))
    return pd.DataFrame(rows)


def trajectory_summary(series: pd.DataFrame, dyad: str = 'AI->H') -> dict:
    """First, extreme and final values of the daily card aggregates by community.

    ``series`` holds one row per snapshot, community and dyad (the extracted
    aggregates). Minimum for conflict and the conflict-above-affinity share,
    maximum for affinity and trust; ``snapshots_g2_above_g1`` counts snapshots
    where the hidden-label value exceeds the visible-label one.
    """
    sub = series.loc[series.dyad.eq(dyad)].sort_values('snapshot')
    out = {}
    for stat, extreme in (('share_conflict_gt_affinity', 'min'), ('conflict_mean', 'min'),
                          ('affinity_mean', 'max'), ('trust_mean', 'max')):
        wide = sub.pivot(index='snapshot', columns='group', values=stat)
        entry = {}
        for group in GROUPS:
            col = wide[group]
            at = col.idxmin() if extreme == 'min' else col.idxmax()
            entry[group] = {'first': float(col.iloc[0]), extreme: float(col.loc[at]), f'{extreme}_snapshot': at,
                            f'{extreme}_in_excluded_window': bool(
                                sub.loc[sub.snapshot.eq(at), 'taken_in_excluded_window'].iloc[0]),
                            'final': float(col.iloc[-1])}
        entry['snapshots'] = int(len(wide))
        entry['snapshots_g2_above_g1'] = int((wide['G2'] > wide['G1']).sum())
        out[stat] = entry
    return out


def load_public_frames(zenodo: Path):
    accounts = pd.read_csv(zenodo / '01_accounts/accounts.csv', usecols=['group', 'user_id', 'actor_class'])
    rounds = pd.read_csv(zenodo / '00_study/rounds.csv')
    windows = pd.read_csv(zenodo / '00_study/valid_day_windows.csv')
    excluded = pd.read_csv(zenodo / '00_study/excluded_windows.csv')
    return accounts, rounds, windows, excluded


def extract(args) -> int:
    accounts, rounds, windows, excluded = load_public_frames(args.public)
    rows, hashes, seen = [], {}, {}
    dirs = sorted(p for p in args.snapshots_root.iterdir() if p.is_dir() and 'snapshot' in p.name)
    for directory in dirs:
        path = directory / 'snapshot_memory_social_cards.csv'
        if not path.exists():
            continue
        digest = sha256(path)
        hashes[directory.name] = digest
        when = snapshot_time(directory.name)
        if when is None:  # suffixed copies: must duplicate a dated snapshot
            seen.setdefault('suffixed', []).append(directory.name)
            continue
        cards = pd.read_csv(path, usecols=CARD_COLUMNS)
        cards = classify(cards, accounts)
        at_round = snapshot_round(when, rounds)
        assert cards.last_updated_round.max() <= at_round, (directory.name, cards.last_updated_round.max(), at_round)
        for row in summarize(cards, at_round):
            row.update(snapshot=directory.name[len('ysocial_grupa1_grupa2_snapshot_'):],
                       snapshot_time=when.isoformat(), snapshot_round=at_round,
                       max_last_updated_round=int(cards.last_updated_round.max()),
                       completed_valid_days=completed_valid_days(when, windows),
                       taken_in_excluded_window=in_excluded_window(when, excluded), sha256=digest)
            rows.append(row)
    for name in seen.get('suffixed', []):
        stem = name.split('_0200')[0] + '_0200'
        assert hashes.get(stem) == hashes[name], f'{name} differs from {stem}'
    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / 'social_card_snapshot_aggregates.csv', index=False)
    (args.out / 'extract.json').write_text(json.dumps(dict(
        host=platform.node(), python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__,
        snapshot_sha256=hashes, suffixed_duplicates=seen.get('suffixed', []),
        distinct_snapshots=int(out.snapshot.nunique()),
        public_inputs={str(p): sha256(p) for p in [args.public / '01_accounts/accounts.csv',
                                                   args.public / '00_study/rounds.csv',
                                                   args.public / '00_study/valid_day_windows.csv',
                                                   args.public / '00_study/excluded_windows.csv']},
        finished_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
    print(out[['snapshot', 'group', 'dyad', 'n_cards', 'conflict_mean', 'affinity_mean',
               'share_conflict_gt_affinity']].to_string(index=False))
    return 0


def finalize(args) -> int:
    root = Path(__file__).resolve().parents[2]
    start = datetime.now(timezone.utc)

    def rel(path: Path) -> str:
        return os.path.relpath(path, root.parent)

    accounts, rounds, windows, excluded = load_public_frames(args.zenodo)
    series = pd.read_csv(args.extracted / 'social_card_snapshot_aggregates.csv')
    extract_record = json.loads((args.extracted / 'extract.json').read_text())
    public_path = args.zenodo / '08_analysis_ready/canonical/snapshot_memory_social_cards.csv'
    raw_path = args.zenodo / '03_agent_memory/memory_social_cards.csv'
    public = classify(pd.read_csv(public_path, usecols=CARD_COLUMNS), accounts)
    raw = pd.read_csv(raw_path, usecols=['group', 'agent_user_id', 'other_user_id', *METRICS,
                                         'last_updated_round', 'event_count'])
    both = public.merge(raw, on=['group', 'agent_user_id', 'other_user_id'], suffixes=('', '_raw'),
                        validate='one_to_one')
    assert len(both) == len(public) == len(raw)
    for col in [*METRICS, 'last_updated_round', 'event_count']:
        assert np.allclose(both[col].astype(float), both[f'{col}_raw'].astype(float)), col
    checks = {}
    for (group, dyad), n in EXPECTED_FINAL.items():
        got = int((public.group.eq(group) & public.dyad.eq(dyad)).sum())
        assert got == n, (group, dyad, got)
    checks['final_cards'] = {f'{g} {d}': n for (g, d), n in EXPECTED_FINAL.items()}

    final_name = series.snapshot.max()
    final_time = datetime.fromisoformat(series.loc[series.snapshot.eq(final_name), 'snapshot_time'].iloc[0])
    at_round = snapshot_round(final_time, rounds)
    assert completed_valid_days(final_time, windows) == 21
    recomputed = pd.DataFrame(summarize(public, at_round))
    stored = series.loc[series.snapshot.eq(final_name)].set_index(['group', 'dyad'])
    for row in recomputed.itertuples(index=False):
        ref = stored.loc[(row.group, row.dyad)]
        for col in recomputed.columns.drop(['group', 'dyad']):
            a, b = float(getattr(row, col)), float(ref[col])
            assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-9, (row.group, row.dyad, col, a, b)
    checks['final_snapshot_private_equals_public'] = True
    checks['distinct_snapshots'] = int(series.snapshot.nunique())
    checks['suffixed_duplicate_directories'] = extract_record['suffixed_duplicates']
    checks['vote_only_cards_final'] = {f'{g} {d}': int(stored.loc[(g, d), 'n_cards_event_count_0'])
                                       for g in GROUPS for d in DYADS}
    last_valid_round = int(rounds.loc[rounds.valid_day_index.notna(), 'round'].max())
    late = public.last_updated_round.gt(last_valid_round)
    checks['final_snapshot_rounds_after_last_valid_window'] = dict(
        last_valid_window_round=last_valid_round, snapshot_round=at_round,
        rounds=[r for r in range(last_valid_round + 1, at_round + 1)],
        cards_last_updated_in_those_rounds={f'{g} {d}': int((late & public.group.eq(g) & public.dyad.eq(d)).sum())
                                            for g in GROUPS for d in DYADS})
    checks['trajectory_ai_to_participant'] = trajectory_summary(series)

    series = series.assign(run_id=RUN_ID, item=ITEM,
                           community=lambda f: np.where(f.group.eq('G1'), 'visible', 'hidden'),
                           denominator_definition=('social cards held by AI accounts in the snapshot '
                                                   '(n_cards), all rounds since the start, not valid-day restricted'))
    lead = ['run_id', 'item', 'snapshot', 'snapshot_time', 'snapshot_round', 'completed_valid_days',
            'taken_in_excluded_window', 'group', 'community', 'dyad', 'n_cards']
    series = series[lead + [c for c in series.columns if c not in lead and c != 'sha256']]

    final = series.loc[series.snapshot.eq(final_name)].copy()
    final['snapshot_role'] = ('first snapshot after the last valid day (window ended 2026-06-10 08:00); taken in '
                              'round 763, so it includes rounds 762-763 after that window (no AI-to-participant '
                              'card was updated in them)')
    assert all(v == 0 for k, v in checks['final_snapshot_rounds_after_last_valid_window'][
        'cards_last_updated_in_those_rounds'].items() if k.endswith('AI->H'))
    boot = bootstrap_final(public, np.random.default_rng(SEED))
    boot.insert(0, 'contrast', CONTRAST)
    boot.insert(0, 'snapshot', final_name)
    boot.insert(0, 'item', ITEM)
    boot.insert(0, 'run_id', RUN_ID)
    boot['interval'] = (f'AI-account clustered percentile bootstrap, 95%, {N_BOOT} replicates, '
                        f'accounts resampled within community, seed {SEED}')
    boot['value_basis'] = 'stored card values (post-decay at each card\'s last update)'
    boot['denominator_definition'] = 'social cards in the final snapshot (n_cards); AI accounts holding them (clusters)'

    out_paths = {
        'snapshots': args.tables / f'chb_prereview_{ITEM}_snapshots.csv',
        'final': args.tables / f'chb_prereview_{ITEM}_final.csv',
        'final_contrasts': args.tables / f'chb_prereview_{ITEM}_final_contrasts.csv',
    }
    args.tables.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_paths['snapshots'], index=False)
    final.to_csv(out_paths['final'], index=False)
    boot.to_csv(out_paths['final_contrasts'], index=False)

    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose=('Describe affinity, conflict, humor and trust on AI accounts\' social cards about participants '
                 '(and about other AI accounts) across the daily snapshots by community; final-snapshot '
                 'hidden-minus-visible differences with an AI-account clustered bootstrap.'),
        host_finalize=platform.node(), host_extract=extract_record['host'],
        command_extract=('cd WORK/a7_ai_social_cards && OMP_NUM_THREADS=1 nice -n 10 '
                         'python chb_prereview_a7_ai_social_cards.py extract '
                         '--snapshots-root PRIVATE_PROCESSED --public public --out extract'),
        command_finalize=' '.join(['python', '-m', 'scripts.diagnostics.chb_prereview_a7_ai_social_cards', *sys.argv[1:]]),
        script=os.path.relpath(Path(__file__).resolve(), root.parent), script_sha256=sha256(Path(__file__)),
        inputs_private_snapshot_sha256=extract_record['snapshot_sha256'],
        inputs_public={rel(p): sha256(p) for p in [public_path, raw_path, args.zenodo / '01_accounts/accounts.csv',
                                                   args.zenodo / '00_study/rounds.csv',
                                                   args.zenodo / '00_study/valid_day_windows.csv',
                                                   args.zenodo / '00_study/excluded_windows.csv']},
        extracted_aggregates={'path': rel(args.extracted / 'social_card_snapshot_aggregates.csv'),
                              'sha256': sha256(args.extracted / 'social_card_snapshot_aggregates.csv')},
        packages_finalize={'numpy': np.__version__, 'pandas': pd.__version__},
        packages_extract={'numpy': extract_record['numpy'], 'pandas': extract_record['pandas'],
                          'python': extract_record['python']},
        decay=dict(memory_social_decay_lambda=DECAY_LAMBDA, unit='hourly round', applied='at update only',
                   ranking_recency_term=('relationship ranking adds exp(-0.05 x rounds since the last interaction) '
                                         'to the priority score; stored values are not further decayed '
                                         '(base_agent.py about l.1998-2012, l.11586-11613, run source on the '
                                         'private host)')),
        seed=SEED, bootstrap_replicates=N_BOOT, contrast=CONTRAST,
        started_utc=start.isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
        extract_finished_utc=extract_record['finished_utc'],
        outputs={k: rel(v) for k, v in out_paths.items()}, diagnostics=checks)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    previous = args.run_dir / 'run.json'
    if previous.exists():  # keep fix-up records across reruns
        record.update({k: v for k, v in json.loads(previous.read_text()).items() if k.startswith('fixups_')})
    (args.run_dir / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    pd.set_option('display.width', 250)
    print(final[['group', 'dyad', 'n_cards', 'n_ai_accounts', 'n_other_accounts', 'n_cards_event_count_0',
                 'affinity_mean', 'affinity_median', 'conflict_mean', 'conflict_median', 'humor_mean',
                 'trust_mean', 'trust_median', 'share_conflict_gt_affinity']].to_string(index=False))
    print(boot[['dyad', 'statistic', 'value_g1', 'value_g2', 'difference', 'bootstrap_low',
                'bootstrap_high']].to_string(index=False))
    print(json.dumps(checks, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('extract')
    ex.add_argument('--snapshots-root', type=Path, required=True)
    ex.add_argument('--public', type=Path, required=True, help='folder with the public 00_study and 01_accounts')
    ex.add_argument('--out', type=Path, required=True)
    fi = sub.add_parser('finalize')
    root = Path(__file__).resolve().parents[2]
    fi.add_argument('--zenodo', type=Path, default=root.parent / 'data')
    fi.add_argument('--extracted', type=Path, default=root.parent / f'data/08_analysis_ready/further_checks/{ITEM}')
    fi.add_argument('--tables', type=Path, default=root / 'runs/chb_prereview_20261110/tables')
    fi.add_argument('--run-dir', type=Path, default=root / f'runs/{RUN_ID}/{ITEM}')
    args = parser.parse_args(argv)
    return extract(args) if args.stage == 'extract' else finalize(args)


if __name__ == '__main__':
    raise SystemExit(main())
