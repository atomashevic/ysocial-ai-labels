"""AI accounts' own exchange labels on their replies to participants (CHB pre-review item A6).

Named run ``chb_prereview_20261110``, item ``a6_ai_relation_labels``.

After an AI account comments, the same language model writes a memory note in
the account's role (prompt ``handler_memory_interaction_note``, identical in both
communities): ``relation_label`` is one of agree, disagree, funny, helpful,
hostile, neutral and ``tone_label`` one of supportive, snarky, angry, curious,
neutral, describing the AI account's reply relative to the contribution it
answered. Votes write no events (``memory_vote_signal_only`` true).

The cohort is the admitted set of AI comments whose direct parent is a
participant contribution on the 21 valid days (451 visible-label, 271
hidden-label). Each comment is linked 1:1 to its comment memory event on
(community, actor account, parent contribution, hourly round); events do not
store the AI comment's own identifier. Inputs are the public Zenodo 1.1.0 tree
and the public analysis configuration, read through the unchanged
``scripts.publication_analysis`` loaders.

Contrasts are hidden minus visible (G2 minus G1), in percentage points, with
Newcombe hybrid-score 95% intervals (independent comments) and an AI-account
clustered percentile bootstrap (accounts resampled with replacement within
community; 10,000 replicates).

Day definition: the cohort uses the admitted calendar-day definition (configured
experiment days 2-9 and 18-30, strict thread validity: the comment and its root
thread both need a timestamp on a configured day), as every admitted analysis
does. The public ``valid_day_index`` on memory events uses the 08:00-to-08:00
windows of ``00_study/valid_day_windows.csv``; window 21 runs from 9 June 08:00
to 10 June 08:00 and so covers comments on experiment days 31 and 32 that the
calendar-day definition excludes. The reconciliation table counts those
separately.

    python -m scripts.diagnostics.chb_prereview_a6_ai_relation_labels \
        --zenodo ../data \
        --config analysis.public-rq3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a6_ai_relation_labels'
ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('G1', 'G2')
EXPECTED_COHORT = {'G1': 451, 'G2': 271}
EXPECTED_AI_COMMENTS = {'G1': 1432, 'G2': 931}
RELATION_VALUES = ('agree', 'disagree', 'hostile', 'neutral', 'funny', 'helpful', 'curious')
TONE_VALUES = ('snarky', 'curious', 'neutral', 'supportive', 'angry')
TOXIC = {'moderately', 'extremely'}
NONTOXIC = {'absolutely no', 'no'}
STRATA = ('all', 'toxic_disposition', 'nontoxic_disposition')
CONTRAST = 'G2_minus_G1 (hidden minus visible)'
N_BOOT = 10_000
SEED = 20261110
Z = NormalDist().inv_cdf(0.975)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    centre = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return centre - half, centre + half


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p2 - p1 with the Newcombe hybrid-score 95% interval (as in study_description)."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def cluster_bootstrap_shares(counts: np.ndarray, sizes: np.ndarray, rng: np.random.Generator,
                             n_boot: int = N_BOOT) -> np.ndarray:
    """Ratio-of-sums shares over clusters resampled with replacement.

    ``counts`` is (clusters, outcomes) of positives, ``sizes`` (clusters,) of
    denominators. Returns (n_boot, outcomes) bootstrap shares.
    """
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    m = len(sizes)
    draws = rng.integers(0, m, size=(n_boot, m))
    weights = np.zeros((n_boot, m))
    np.add.at(weights, (np.repeat(np.arange(n_boot), m), draws.ravel()), 1.0)
    num = weights @ counts
    den = weights @ sizes
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den[:, None]


def link_events(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Link each AI comment to one comment memory event, 1:1.

    Key: community, actor account, parent contribution, hourly round. Within a
    key that holds several comments, comments in creation order are paired with
    events in identifier order.
    """
    key = ['group', 'author_id', 'comment_to', 'round']
    left = cohort.sort_values(['group', 'created_at', 'content_id']).copy()
    left['_slot'] = left.groupby(key, sort=False).cumcount()
    right = events.rename(columns={'actor_user_id': 'author_id', 'target_post_id': 'comment_to',
                                   'round_id': 'round'}).copy()
    right['comment_to'] = right.comment_to.astype('int64')
    right = right.sort_values(['group', 'memory_event_id'])
    right['_slot'] = right.groupby(key, sort=False).cumcount()
    linked = left.merge(right, on=key + ['_slot'], how='left', validate='one_to_one', indicator=True)
    return linked.drop(columns='_slot')


def load_cohort(zenodo: Path, config_path: Path, checks: dict):
    sys.path.insert(0, str(ROOT))
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents

    from scripts.publication_analysis.study import analysis_set_config

    config = load_config(config_path, profile='release', run_id=f'{RUN_ID}_{ITEM}')
    set_cfg = analysis_set_config(config.analysis_set)
    checks['configured_valid_days'] = list(set_cfg['valid_days'])
    checks['strict_thread_validity'] = bool(set_cfg['strict_thread_validity'])
    checks['processed_dir'] = str(config.processed_dir)
    contents = _valid_contents(config).reset_index(drop=True)
    assert len(contents) == 6088, len(contents)
    contents['is_root'] = contents.comment_to.isna() | contents.comment_to.eq(-1)
    parents = contents[['group', 'content_id', 'author_type', 'author_id']].rename(
        columns={'content_id': 'parent_id', 'author_type': 'parent_type', 'author_id': 'parent_author'})
    contents = contents.merge(parents, left_on=['group', 'comment_to'], right_on=['group', 'parent_id'],
                              how='left', validate='many_to_one')

    accounts = pd.read_csv(zenodo / '01_accounts/accounts.csv')
    universe = accounts[['group', 'user_id', 'actor_class', 'toxicity']].rename(columns={'user_id': 'author_id'})
    contents = contents.merge(universe, on=['group', 'author_id'], how='left', validate='many_to_one')
    # Authoritative account classes agree with the canonical author type.
    assert contents.actor_class.notna().all()
    assert (contents.author_type.eq('AI') == contents.actor_class.eq('ai')).all()

    posts = pd.read_csv(zenodo / '02_platform_activity/posts.csv',
                        usecols=['group', 'post_id', 'author_user_id', 'comment_to', 'thread_id', 'round'])
    contents = contents.merge(posts.rename(columns={'post_id': 'content_id', 'comment_to': 'comment_to_posts'}),
                              on=['group', 'content_id'], how='left', validate='one_to_one')
    assert contents['round'].notna().all()
    assert contents.author_user_id.eq(contents.author_id).all()
    assert contents.thread_id.eq(contents.root_id).all()

    ai_comments = contents.loc[contents.author_type.eq('AI') & ~contents.is_root]
    for group in GROUPS:
        assert len(ai_comments.loc[ai_comments.group.eq(group)]) == EXPECTED_AI_COMMENTS[group]
    assert ai_comments.parent_type.notna().all()
    cohort = ai_comments.loc[ai_comments.parent_type.eq('H')].copy()
    cohort['comment_to'] = cohort.comment_to.astype('int64')
    cohort['round'] = cohort['round'].astype('int64')
    for group in GROUPS:
        assert len(cohort.loc[cohort.group.eq(group)]) == EXPECTED_COHORT[group]
    assert cohort.toxicity.isin(TOXIC | NONTOXIC).all()
    checks['cohort'] = {g: int(cohort.group.eq(g).sum()) for g in GROUPS}
    checks['ai_comments'] = {g: int(ai_comments.group.eq(g).sum()) for g in GROUPS}
    admitted = set(zip(contents.group, contents.content_id.astype('int64')))
    return cohort, posts, admitted


def load_events(zenodo: Path) -> pd.DataFrame:
    events = pd.read_csv(zenodo / '03_agent_memory/memory_interaction_events.csv', low_memory=False,
                         usecols=['group', 'memory_event_id', 'round_id', 'actor_user_id', 'target_user_id',
                                  'thread_root_id', 'target_post_id', 'actor_post_id', 'event_type',
                                  'relation_label', 'tone_label', 'valid_day_index'])
    canonical = pd.read_csv(zenodo / '08_analysis_ready/canonical/snapshot_memory_interaction_events.csv',
                            usecols=['group', 'memory_event_id', 'relation_label', 'tone_label', 'dyad'])
    merged = events.merge(canonical, on=['group', 'memory_event_id'], how='outer', validate='one_to_one',
                          suffixes=('', '_canonical'), indicator=True)
    assert merged._merge.eq('both').all()
    for col in ('relation_label', 'tone_label'):
        assert merged[col].fillna('').eq(merged[f'{col}_canonical'].fillna('')).all(), col
    return merged.drop(columns=['_merge', 'relation_label_canonical', 'tone_label_canonical'])


def outcome_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for value in RELATION_VALUES:
        out[('relation_label', value)] = frame.relation_label.eq(value)
    for value in TONE_VALUES:
        out[('tone_label', value)] = frame.tone_label.eq(value)
    out[('derived', 'disagree_or_hostile')] = frame.relation_label.isin({'disagree', 'hostile'})
    out[('derived', 'tone_angry_or_snarky')] = frame.tone_label.isin({'angry', 'snarky'})
    return pd.DataFrame(out, index=frame.index).astype(int)


def stratum_frame(linked: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum == 'all':
        return linked
    keep = TOXIC if stratum == 'toxic_disposition' else NONTOXIC
    return linked.loc[linked.toxicity.isin(keep)]


def build_tables(linked: pd.DataFrame, rng: np.random.Generator):
    count_rows, contrast_rows = [], []
    for stratum in STRATA:
        frame = stratum_frame(linked, stratum)
        ind = outcome_indicators(frame)
        cols = list(ind.columns)
        boot, cells = {}, {}
        for group in GROUPS:
            g = frame.group.eq(group)
            per_acc = ind.loc[g].groupby(frame.loc[g, 'author_id']).sum()
            sizes = frame.loc[g].groupby('author_id').size().reindex(per_acc.index)
            boot[group] = cluster_bootstrap_shares(per_acc.to_numpy(), sizes.to_numpy(), rng)
            n = int(g.sum())
            cells[group] = (ind.loc[g].sum(), n, len(sizes), int(frame.loc[g, 'parent_author'].nunique()))
            for j, (field, value) in enumerate(cols):
                k = int(cells[group][0][(field, value)])
                lo, hi = wilson(k, n)
                count_rows.append(dict(
                    run_id=RUN_ID, item=ITEM, stratum=stratum, group=group,
                    community='visible' if group == 'G1' else 'hidden', label_field=field, label_value=value,
                    count=k, denominator=n, share=k / n, wilson_low=lo, wilson_high=hi,
                    ai_accounts_contributing=len(sizes), participants_replied_to=cells[group][3],
                    denominator_definition=('AI comments replying directly to a participant contribution on '
                                            'valid days, linked 1:1 to their memory event')))
        diff = boot['G2'] - boot['G1']
        for j, (field, value) in enumerate(cols):
            k1, n1 = int(cells['G1'][0][(field, value)]), cells['G1'][1]
            k2, n2 = int(cells['G2'][0][(field, value)]), cells['G2'][1]
            d, lo, hi = newcombe(k1, n1, k2, n2)
            valid = diff[:, j][np.isfinite(diff[:, j])]
            contrast_rows.append(dict(
                run_id=RUN_ID, item=ITEM, stratum=stratum, label_field=field, label_value=value,
                contrast=CONTRAST, positive_g1=k1, n_g1=n1, positive_g2=k2, n_g2=n2,
                share_g1=k1 / n1, share_g2=k2 / n2, difference_pp=100 * d,
                newcombe_low_pp=100 * lo, newcombe_high_pp=100 * hi,
                bootstrap_low_pp=100 * float(np.quantile(valid, 0.025)),
                bootstrap_high_pp=100 * float(np.quantile(valid, 0.975)),
                bootstrap_sd_pp=100 * float(np.std(valid, ddof=1)),
                bootstrap_replicates=len(valid), bootstrap_clusters_g1=cells['G1'][2],
                bootstrap_clusters_g2=cells['G2'][2],
                interval_newcombe='Newcombe hybrid score (Wilson components), 95%, comments independent',
                interval_bootstrap=f'AI-account clustered percentile bootstrap, 95%, {N_BOOT} replicates, '
                                   f'accounts resampled within community, seed {SEED}',
                denominator_definition=('AI comments replying directly to a participant contribution on valid '
                                        'days (n_g1, n_g2); AI accounts with at least one such comment '
                                        '(bootstrap clusters)')))
    return pd.DataFrame(count_rows), pd.DataFrame(contrast_rows)


RECON_DEFINITIONS = {
    'ai_to_participant_comment_events_all_rounds':
        'all comment memory events whose target is a participant account',
    'linked_to_admitted_cohort': 'linked 1:1 to an admitted AI comment (the analysed cohort)',
    'not_in_cohort_event_round_outside_valid_day_windows':
        'event round outside the 21 public 08:00-to-08:00 valid-day windows (valid_day_index empty)',
    'not_in_cohort_root_thread_timestamp_missing':
        ('comment has created_at on a configured day but its root thread has no root_thread_created_at; '
         'strict thread validity excludes it'),
    'not_in_cohort_comment_on_experiment_day_outside_configured_valid_days':
        ('comment and root timestamps present, but the comment was created on an experiment day outside the '
         'configured calendar days 2-9 and 18-30 (inside public 08:00 window 21)'),
    'not_in_cohort_other': 'any remaining event (expected none)',
}


def reconciliation(events: pd.DataFrame, linked: pd.DataFrame, posts: pd.DataFrame,
                   admitted: set, reasons: pd.DataFrame, valid_days, checks: dict | None = None) -> pd.DataFrame:
    """Account for every AI-to-participant comment event outside the admitted cohort.

    ``admitted`` holds (community, contribution id) of the 6,088 admitted
    valid-day contributions; ``posts`` is the public post table; ``reasons``
    the canonical per-contribution ``created_at``, ``root_thread_created_at``,
    ``experiment_day``, ``invalid_reason`` and ``analysis_phase``;
    ``valid_days`` the configured calendar experiment days.
    """
    ai_h = events.loc[events.event_type.eq('comment') & events.dyad.eq('AI->H')].copy()
    used = set(zip(linked.group, linked.memory_event_id))
    ai_h['in_cohort'] = [(g, e) in used for g, e in zip(ai_h.group, ai_h.memory_event_id)]
    rest = ai_h.loc[~ai_h.in_cohort].copy()
    rest['target_post_id'] = rest.target_post_id.astype('int64')
    replies = posts.loc[posts.comment_to.notna() & posts.comment_to.ne(-1)].copy()
    replies['comment_to'] = replies.comment_to.astype('int64')
    rest = rest.merge(replies[['group', 'post_id', 'author_user_id', 'comment_to', 'round']],
                      left_on=['group', 'actor_user_id', 'target_post_id', 'round_id'],
                      right_on=['group', 'author_user_id', 'comment_to', 'round'], how='left')
    rest['comment_admitted'] = [(g, i) in admitted for g, i in zip(rest.group, rest.post_id)]
    assert not rest.comment_admitted.any()
    rest = rest.merge(reasons.rename(columns={'content_id': 'post_id'}), on=['group', 'post_id'], how='left')
    created = pd.to_datetime(rest.created_at, utc=True, format='mixed')
    rows, spans = [], {}
    for group in GROUPS:
        g = ai_h.loc[ai_h.group.eq(group)]
        sel = rest.group.eq(group)
        r = rest.loc[sel]
        on_day = r.valid_day_index.notna()
        root_missing = on_day & r.root_thread_created_at.isna()
        # Every comment here has its own timestamp; only the root thread's is missing.
        assert r.loc[root_missing, 'created_at'].notna().all()
        assert root_missing.eq(on_day & r.invalid_reason.eq('missing_timestamp')).all()
        off_day = (on_day & ~root_missing & r.created_at.notna()
                   & ~r.experiment_day.isin(list(valid_days)))
        assert off_day.eq(on_day & r.analysis_phase.eq('outside_revision_window')).all()
        if off_day.any():
            when = created.loc[sel][off_day].dt.tz_convert('Europe/Belgrade')
            spans[group] = dict(experiment_days=sorted(int(d) for d in r.loc[off_day, 'experiment_day'].unique()),
                                public_valid_day_index=sorted(int(d) for d in r.loc[off_day, 'valid_day_index'].unique()),
                                created_first=when.min().isoformat(), created_last=when.max().isoformat())
        rows += [
            dict(category='ai_to_participant_comment_events_all_rounds', count=len(g)),
            dict(category='linked_to_admitted_cohort', count=int(g.in_cohort.sum())),
            dict(category='not_in_cohort_event_round_outside_valid_day_windows', count=int((~on_day).sum())),
            dict(category='not_in_cohort_root_thread_timestamp_missing', count=int(root_missing.sum())),
            dict(category='not_in_cohort_comment_on_experiment_day_outside_configured_valid_days',
                 count=int(off_day.sum())),
            dict(category='not_in_cohort_other', count=int((on_day & ~root_missing & ~off_day).sum())),
        ]
        for row in rows[-6:]:
            row.update(run_id=RUN_ID, item=ITEM, group=group)
        assert rows[-6]['count'] == sum(x['count'] for x in rows[-5:])
    if checks is not None:
        checks['reconciliation_off_configured_day_span'] = spans
    out = pd.DataFrame(rows)[['run_id', 'item', 'group', 'category', 'count']]
    out['category_definition'] = out.category.map(RECON_DEFINITIONS)
    out['denominator_definition'] = ('comment memory events whose target is a participant account '
                                     '(all rounds, warm-up and excluded days included)')
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--zenodo', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'analysis.public-rq3.yaml')
    parser.add_argument('--tables', type=Path, default=ROOT / 'runs/chb_prereview_20261110/tables')
    parser.add_argument('--run-dir', type=Path, default=ROOT / f'runs/{RUN_ID}/{ITEM}')
    args = parser.parse_args(argv)
    start = datetime.now(timezone.utc)
    checks: dict = {}
    rng = np.random.default_rng(SEED)

    cohort, posts, admitted = load_cohort(args.zenodo, args.config, checks)
    events = load_events(args.zenodo)
    comment_events = events.loc[events.event_type.eq('comment')]
    linked = link_events(cohort, comment_events)
    checks['linked'] = {g: int((linked.group.eq(g) & linked._merge.eq('both')).sum()) for g in GROUPS}
    checks['unlinked'] = {g: int((linked.group.eq(g) & ~linked._merge.eq('both')).sum()) for g in GROUPS}
    assert sum(checks['unlinked'].values()) == 0, checks['unlinked']
    assert not linked.duplicated(['group', 'memory_event_id']).any()
    # The linked event points at the same participant and thread as the comment's parent.
    assert linked.target_user_id.astype(int).eq(linked.parent_author.astype(int)).all()
    assert linked.thread_root_id.astype(int).eq(linked.root_id.astype(int)).all()
    assert linked.dyad.eq('AI->H').all()
    assert linked.relation_label.notna().all() and linked.tone_label.notna().all()
    key = ['group', 'author_id', 'comment_to', 'round']
    checks['cohort_keys_shared_by_two_comments'] = int(cohort.duplicated(key).sum())
    checks['labels_outside_prompt_list'] = linked.relation_label.loc[
        ~linked.relation_label.isin({'agree', 'disagree', 'funny', 'helpful', 'hostile', 'neutral'})
    ].groupby(linked.group).size().to_dict()

    counts, contrasts = build_tables(linked, rng)
    reasons = pd.read_csv(args.zenodo / '08_analysis_ready/canonical/snapshot_contents.csv',
                          usecols=['group', 'content_id', 'created_at', 'root_thread_created_at', 'experiment_day',
                                   'invalid_reason', 'analysis_phase'])
    recon = reconciliation(events, linked, posts, admitted, reasons, checks['configured_valid_days'], checks)
    checks['newcombe_excludes_zero'] = [
        f"{r.stratum} {r.label_field}={r.label_value}: {r.difference_pp:+.1f} pp "
        f"({r.newcombe_low_pp:.1f} to {r.newcombe_high_pp:.1f})"
        for r in contrasts.itertuples() if r.newcombe_low_pp > 0 or r.newcombe_high_pp < 0]
    checks['bootstrap_excludes_zero'] = int(((contrasts.bootstrap_low_pp > 0)
                                             | (contrasts.bootstrap_high_pp < 0)).sum())
    headline = contrasts.loc[contrasts.stratum.eq('all') & contrasts.label_value.eq('disagree_or_hostile')].iloc[0]
    assert (headline.positive_g1, headline.n_g1, headline.positive_g2, headline.n_g2) == (229, 451, 144, 271)

    args.tables.mkdir(parents=True, exist_ok=True)
    out_paths = {
        'counts': args.tables / f'chb_prereview_{ITEM}_counts.csv',
        'contrasts': args.tables / f'chb_prereview_{ITEM}_contrasts.csv',
        'reconciliation': args.tables / f'chb_prereview_{ITEM}_reconciliation.csv',
    }
    counts.to_csv(out_paths['counts'], index=False)
    contrasts.to_csv(out_paths['contrasts'], index=False)
    recon.to_csv(out_paths['reconciliation'], index=False)

    # Per-account intermediate (public pseudonymous codes only) for audit.
    out_dir = args.run_dir / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    per_account = linked.assign(**{'disagree_or_hostile': linked.relation_label.isin({'disagree', 'hostile'}).astype(int),
                                   'hostile': linked.relation_label.eq('hostile').astype(int)}).groupby(
        ['group', 'author_code', 'toxicity']).agg(n=('content_id', 'size'), disagree_or_hostile=('disagree_or_hostile', 'sum'),
                                                  hostile=('hostile', 'sum')).reset_index()
    per_account.to_csv(out_dir / 'per_account_labels.csv', index=False)

    inputs = [args.zenodo / '01_accounts/accounts.csv', args.zenodo / '02_platform_activity/posts.csv',
              args.zenodo / '03_agent_memory/memory_interaction_events.csv',
              args.zenodo / '08_analysis_ready/canonical/snapshot_contents.csv',
              args.zenodo / '08_analysis_ready/canonical/snapshot_memory_interaction_events.csv',
              args.zenodo / '06_experiment_config/g1/prompts.json',
              args.zenodo / '06_experiment_config/g2/prompts.json', args.config]
    # The loader (scripts.publication_analysis.study.read_table) prefers the parquet beside the CSV.
    processed = Path(checks['processed_dir'])
    loader_read = next(p for p in (processed / 'snapshot_contents.parquet', processed / 'snapshot_contents.csv')
                       if p.exists())
    processed_inputs = {
        'loader_read': str(loader_read), 'loader_read_sha256': sha256(loader_read),
        'rule': 'read_table(processed_dir / "snapshot_contents") reads .parquet when present, else .csv',
        'siblings': {str(p): sha256(p) for p in (processed / 'snapshot_contents.parquet',
                                                 processed / 'snapshot_contents.csv') if p.exists()},
    }
    import scipy
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose=('Tabulate the AI accounts\' own relation and tone labels of their replies to participants on '
                 'valid days by community and toxicity disposition; hidden-minus-visible differences with '
                 'Newcombe and AI-account clustered bootstrap intervals.'),
        host=platform.node(), python=sys.version.split()[0],
        command=' '.join(['python', '-m', 'scripts.diagnostics.chb_prereview_a6_ai_relation_labels', *sys.argv[1:]]),
        script=str(Path(__file__).resolve().relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        inputs={str(p): sha256(p) for p in inputs},
        processed_snapshot_contents=processed_inputs,
        packages={'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        seed=SEED, bootstrap_replicates=N_BOOT, contrast=CONTRAST,
        started_utc=start.isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
        outputs={k: os.path.relpath(v, ROOT.parent) for k, v in out_paths.items()},
        diagnostics=checks,
        headline={k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v))
                  for k, v in headline.items()},
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    previous = args.run_dir / 'run.json'
    if previous.exists():  # keep fix-up records across reruns
        record.update({k: v for k, v in json.loads(previous.read_text()).items() if k.startswith('fixups_')})
    (args.run_dir / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    view = contrasts.loc[contrasts.label_value.isin(['disagree_or_hostile', 'hostile', 'agree', 'disagree'])]
    print(view[['stratum', 'label_value', 'positive_g1', 'n_g1', 'positive_g2', 'n_g2', 'difference_pp',
                'newcombe_low_pp', 'newcombe_high_pp', 'bootstrap_low_pp', 'bootstrap_high_pp']].to_string(index=False))
    print(recon.to_string(index=False))
    print(json.dumps(checks, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
