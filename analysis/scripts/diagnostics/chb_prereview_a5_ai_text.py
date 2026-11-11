"""A5: AI comment text features by community beyond Detoxify.

Named run chb_prereview_20261110, item a5_ai_text. Descriptive only, no model.

Cohort: the admitted analysis cohort (``_valid_contents``) rebuilt from the
public Zenodo 1.1.0 canonical tables through the public-route configuration
(analysis.public-rq3.yaml). Subsets:

* ``ai_all``: all AI comments (1,432 visible-label / 931 hidden-label);
* ``ai_to_participant``: AI comments whose direct parent in the cohort is a
  participant contribution (451 / 271);
* ``participant_all``: participant comments (1,518 / 1,056), reference only.

Text is the stored comment text, stripped (as in review_ai_conduct.py). Per text:
characters, whitespace words, tokens (lower-cased word tokens with internal
apostrophes; URLs and @handles removed), MTLD (McCarthy and Jarvis 2010,
threshold 0.72, mean of forward and backward passes; undefined when a text has
no completed or partial factor, i.e. no repeated token), type-token ratio of the
first 50 tokens for texts with at least 50 tokens, VADER compound score
(vaderSentiment) and whether the text contains a question mark. Per account, the
account's comments are concatenated in creation order and summarized with MTLD
and MSTTR-50 (mean TTR over consecutive complete 50-token segments), because
single comments are short (median about 20 words) and per-text MTLD is unstable.

Hidden-minus-visible differences (G2 minus G1) carry 95% percentile intervals from
2,000 bootstrap replicates that resample accounts with replacement within each
community (AI accounts for AI subsets, participants for the participant
reference). Intervals are descriptive and treat accounts as exchangeable.

Usage::

    python -m scripts.diagnostics.chb_prereview_a5_ai_text
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a5_ai_text'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
PUBLIC_CONFIG = ROOT / 'analysis.public-rq3.yaml'
CONDUCT_COMMENTS = ROOT.parent / 'data/08_analysis_ready/further_checks/a5_ai_text/review_ai_conduct_comments.csv'
GROUPS = ('G1', 'G2')
CONDITION = {'G1': 'labels_visible', 'G2': 'labels_hidden'}
SUBSETS = ('ai_all', 'ai_to_participant', 'participant_all')
EXPECTED_N = {'ai_all': {'G1': 1432, 'G2': 931}, 'ai_to_participant': {'G1': 451, 'G2': 271},
              'participant_all': {'G1': 1518, 'G2': 1056}}
MTLD_THRESHOLD = 0.72
SEGMENT = 50
NEGATIVE, POSITIVE = -0.05, 0.05
REPLICATES = 2000
SEED = 20261110
URL = re.compile(r'https?://\S+|www\.\S+')
HANDLE = re.compile(r'@[\w-]+')
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


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


# --------------------------------------------------------------------------- text measures

def tokenize(text: str) -> list[str]:
    text = HANDLE.sub(' ', URL.sub(' ', str(text))).lower().replace('’', "'")
    return TOKEN.findall(text)


def _mtld_pass(tokens: list[str], threshold: float = MTLD_THRESHOLD) -> float:
    factors, types, count = 0.0, set(), 0
    for token in tokens:
        count += 1
        types.add(token)
        if len(types) / count <= threshold:
            factors += 1
            types, count = set(), 0
    if count:
        factors += (1 - len(types) / count) / (1 - threshold)
    return len(tokens) / factors if factors > 0 else math.nan


def mtld(tokens: list[str], threshold: float = MTLD_THRESHOLD) -> float:
    """Bidirectional MTLD; NaN when undefined (no factor in either direction)."""
    if not tokens:
        return math.nan
    forward, backward = _mtld_pass(tokens, threshold), _mtld_pass(tokens[::-1], threshold)
    if math.isnan(forward) or math.isnan(backward):
        return math.nan
    return (forward + backward) / 2


def ttr_first(tokens: list[str], n: int = SEGMENT) -> float:
    return len(set(tokens[:n])) / n if len(tokens) >= n else math.nan


def segment_ttrs(tokens: list[str], n: int = SEGMENT) -> list[float]:
    return [len(set(tokens[i:i + n])) / n for i in range(0, len(tokens) - n + 1, n)]


# --------------------------------------------------------------------------- statistics

def text_statistics(frame: pd.DataFrame) -> dict[str, float]:
    """Community-level statistics from per-text features (frame rows are texts)."""
    defined = frame.mtld.dropna()
    eligible = frame.ttr50.dropna()
    return {
        'median_characters': float(frame.chars.median()),
        'mean_characters': float(frame.chars.mean()),
        'median_words': float(frame.words.median()),
        'median_text_mtld': float(defined.median()) if len(defined) else math.nan,
        'share_text_mtld_undefined': float(frame.mtld.isna().mean()),
        'mean_ttr_first50': float(eligible.mean()) if len(eligible) else math.nan,
        'vader_compound_mean': float(frame.compound.mean()),
        'share_vader_negative': float(frame.compound.lt(NEGATIVE).mean()),
        'share_vader_positive': float(frame.compound.gt(POSITIVE).mean()),
        'share_question_mark': float(frame.question.mean()),
    }


def account_statistics(accounts: pd.DataFrame) -> dict[str, float]:
    """Statistics over account-level concatenations (rows are accounts)."""
    with_mtld = accounts.account_mtld.dropna()
    segments = np.concatenate([np.asarray(s, dtype=float) for s in accounts.segment_ttrs]) if len(accounts) else np.array([])
    return {'mean_account_mtld': float(with_mtld.mean()) if len(with_mtld) else math.nan,
            'msttr50': float(segments.mean()) if segments.size else math.nan}


TEXT_METRICS = tuple(text_statistics(pd.DataFrame({'chars': [1.], 'words': [1.], 'mtld': [1.], 'ttr50': [1.],
                                                   'compound': [0.], 'question': [False]})))
ACCOUNT_METRICS = ('mean_account_mtld', 'msttr50')


def all_statistics(texts: pd.DataFrame, accounts: pd.DataFrame) -> dict[str, float]:
    return {**text_statistics(texts), **account_statistics(accounts)}


def cluster_bootstrap(texts: pd.DataFrame, accounts: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Hidden-minus-visible differences, resampling clusters within community.

    ``texts`` and ``accounts`` carry ``group`` and ``cluster`` columns.
    Returns one row per replicate with one column per statistic.
    """
    rng = np.random.default_rng(seed)
    pieces = {}
    for group in GROUPS:
        t = texts.loc[texts.group.eq(group)]
        a = accounts.loc[accounts.group.eq(group)].set_index('cluster')
        ids = np.array(sorted(t.cluster.unique()))
        by = {c: t.loc[t.cluster.eq(c)] for c in ids}
        pieces[group] = (ids, by, a)
    rows = []
    for _ in range(replicates):
        values = {}
        for group in GROUPS:
            ids, by, a = pieces[group]
            draw = rng.choice(ids, size=len(ids), replace=True)
            values[group] = all_statistics(pd.concat([by[c] for c in draw], ignore_index=True),
                                           a.loc[[c for c in draw if c in a.index]])
        rows.append({k: values['G2'][k] - values['G1'][k] for k in values['G1']})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- cohort

def load_cohort() -> tuple[pd.DataFrame, dict]:
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents

    config = load_config(PUBLIC_CONFIG, profile='release', run_id=RUN_ID)
    contents = _valid_contents(config).reset_index(drop=True)
    contents['group'] = contents.group.astype(str)
    contents['author_key'] = contents.author_id.astype(int)
    contents['is_root'] = contents.comment_to.isna() | contents.comment_to.eq(-1)
    contents['text'] = contents.tweet.fillna('').astype(str).str.strip()
    parents = contents[['group', 'content_id', 'author_type']].rename(
        columns={'content_id': 'parent_id', 'author_type': 'parent_type'})
    contents = contents.merge(parents, left_on=['group', 'comment_to'], right_on=['group', 'parent_id'],
                              how='left', validate='many_to_one')
    source = Path(config.processed_dir) / 'snapshot_contents.csv'
    return contents, {str(source): sha256(source), str(PUBLIC_CONFIG): sha256(PUBLIC_CONFIG)}


def features(frame: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    out = frame[['group', 'content_id', 'author_key', 'author_type', 'created_at', 'text']].copy()
    out['chars'] = out.text.str.len()
    out['words'] = out.text.str.split().str.len()
    out['tokens'] = out.text.map(tokenize)
    out['n_tokens'] = out.tokens.str.len()
    out['mtld'] = out.tokens.map(mtld)
    out['ttr50'] = out.tokens.map(ttr_first)
    out['compound'] = out.text.map(lambda t: analyzer.polarity_scores(t)['compound'])
    out['question'] = out.text.str.contains('?', regex=False)
    return out


def account_frame(texts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = texts.assign(t=pd.to_datetime(texts.created_at, utc=True, format='mixed')).sort_values(['t', 'content_id'])
    for (group, cluster), part in ordered.groupby(['group', 'cluster'], sort=True):
        tokens = [tok for toks in part.tokens for tok in toks]
        rows.append(dict(group=group, cluster=cluster, texts=len(part), tokens=len(tokens),
                         account_mtld=mtld(tokens) if len(tokens) >= SEGMENT else math.nan,
                         segment_ttrs=segment_ttrs(tokens)))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replicates', type=int, default=REPLICATES)
    args = parser.parse_args()
    import importlib.metadata as metadata

    started = dt.datetime.now().astimezone()
    log(f'{ITEM}: start ({args.replicates} replicates, seed {SEED})')
    contents, inputs = load_cohort()
    checks = {'cohort_6088': len(contents) == 6088}
    comments = contents.loc[~contents.is_root]
    subsets = {'ai_all': comments.loc[comments.author_type.eq('AI')],
               'ai_to_participant': comments.loc[comments.author_type.eq('AI') & comments.parent_type.eq('H')],
               'participant_all': comments.loc[comments.author_type.eq('H')]}
    for name, frame in subsets.items():
        checks[f'{name}_counts'] = frame.group.value_counts().to_dict() == EXPECTED_N[name]
    admitted = pd.read_csv(CONDUCT_COMMENTS).set_index(['subset', 'group'])
    anchors = {}
    for name, admitted_name in (('ai_all', 'all_ai_comments'), ('ai_to_participant', 'ai_to_participant')):
        f = subsets[name]
        for group in GROUPS:
            g = f.loc[f.group.eq(group)]
            # Released text has usernames replaced by account codes (scripts/zenodo/scrub.py), so
            # character lengths may differ slightly from the private-host values.
            public_mean, public_median = float(g.text.str.len().mean()), float(g.text.str.len().median())
            anchors[f'{name}_{group}'] = dict(
                public_median_chars=public_median, admitted_median_chars=float(admitted.loc[(admitted_name, group), 'chars_median']),
                public_mean_chars=public_mean, admitted_mean_chars=float(admitted.loc[(admitted_name, group), 'chars_mean']))
            checks[f'{name}_{group}_chars_within_scrub_tolerance_of_review_ai_conduct'] = (
                abs(public_mean - float(admitted.loc[(admitted_name, group), 'chars_mean'])) < 0.1
                and abs(public_median - float(admitted.loc[(admitted_name, group), 'chars_median'])) <= 1)
            checks[f'{name}_{group}_median_words_reproduces_review_ai_conduct'] = (
                float(g.text.str.split().str.len().median()) == float(admitted.loc[(admitted_name, group), 'words_median']))
    if not all(checks.values()):
        log(f'{ITEM}: anchor checks failed {[k for k, v in checks.items() if not v]}')
        raise SystemExit(1)
    log(f'{ITEM}: anchors reproduced: {json.dumps(checks)}; lengths {json.dumps(anchors)}')

    summary_rows, diff_rows = [], []
    for index, (name, frame) in enumerate(subsets.items()):
        texts = features(frame).assign(cluster=lambda d: d.group + ':' + d.author_key.astype(str))
        accounts = account_frame(texts)
        cluster_unit = 'participant' if name == 'participant_all' else 'AI account'
        n_texts = texts.group.value_counts().to_dict()
        n_clusters = texts.groupby('group').cluster.nunique().to_dict()
        denominator = (f'{n_texts["G1"]} (G1) and {n_texts["G2"]} (G2) comments from {n_clusters["G1"]} and '
                       f'{n_clusters["G2"]} {cluster_unit}s')
        point = {}
        for group in GROUPS:
            t, a = texts.loc[texts.group.eq(group)], accounts.loc[accounts.group.eq(group)]
            point[group] = all_statistics(t, a)
            eligible = {'median_text_mtld': int(t.mtld.notna().sum()), 'mean_ttr_first50': int(t.ttr50.notna().sum()),
                        'mean_account_mtld': int(a.account_mtld.notna().sum()),
                        'msttr50': int(sum(len(s) for s in a.segment_ttrs))}
            for metric, value in point[group].items():
                summary_rows.append(dict(run_id=RUN_ID, item=ITEM, subset=name, group=group, condition=CONDITION[group],
                                         metric=metric, value=value, n_texts=len(t), n_clusters=len(a),
                                         cluster_unit=cluster_unit,
                                         n_eligible=eligible.get(metric, len(t)),
                                         eligible_unit={'mean_account_mtld': 'accounts with >=50 tokens',
                                                        'msttr50': '50-token segments'}.get(metric, 'texts'),
                                         denominator=denominator))
        light = texts[['group', 'cluster', 'chars', 'words', 'mtld', 'ttr50', 'compound', 'question']]
        boot = cluster_bootstrap(light, accounts, args.replicates, SEED + index)
        OUT.mkdir(parents=True, exist_ok=True)
        boot.to_csv(OUT / f'bootstrap_{name}.csv', index=False)
        for metric in point['G1']:
            values = boot[metric].dropna()
            diff_rows.append(dict(
                run_id=RUN_ID, item=ITEM, subset=name, metric=metric, visible_G1=point['G1'][metric],
                hidden_G2=point['G2'][metric], hidden_minus_visible=point['G2'][metric] - point['G1'][metric],
                ci_low=float(values.quantile(.025)) if len(values) else math.nan,
                ci_high=float(values.quantile(.975)) if len(values) else math.nan,
                bootstrap_replicates=args.replicates, finite_replicates=int(len(values)),
                cluster_unit=cluster_unit, contrast_sign='G2 (hidden labels) minus G1 (visible labels)',
                interval='95% percentile, cluster bootstrap within community', denominator=denominator))
        log(f'{ITEM}: {name} done')
    summary, diffs = pd.DataFrame(summary_rows), pd.DataFrame(diff_rows)
    outputs = {'summary': TABLES / f'chb_prereview_{ITEM}_summary.csv',
               'differences': TABLES / f'chb_prereview_{ITEM}_differences.csv'}
    summary.to_csv(outputs['summary'], index=False)
    diffs.to_csv(outputs['differences'], index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        log('\n' + diffs[['subset', 'metric', 'visible_G1', 'hidden_G2', 'hidden_minus_visible', 'ci_low', 'ci_high',
                          'finite_replicates']].round(4).to_string(index=False))
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='AI comment text features (length, lexical diversity, VADER sentiment, questions) by community, '
                'with participant comments as reference (CHB pre-review A5).',
        host=socket.gethostname(),
        command=' '.join([sys.executable, '-m', 'scripts.diagnostics.chb_prereview_a5_ai_text'] + sys.argv[1:]),
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        imported_modules={p: sha256(ROOT / p) for p in ('scripts/publication_analysis/data.py', 'scripts/publication_analysis/config.py')},
        inputs={**inputs, '08_analysis_ready/further_checks/a5_ai_text/review_ai_conduct_comments.csv': sha256(CONDUCT_COMMENTS)},
        outputs={k: dict(path=str(v.relative_to(ROOT.parent)), sha256=sha256(v)) for k, v in outputs.items()},
        packages=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                      vaderSentiment=metadata.version('vaderSentiment')),
        settings=dict(replicates=args.replicates, seed=SEED, mtld_threshold=MTLD_THRESHOLD, segment=SEGMENT,
                      vader_negative_below=NEGATIVE, vader_positive_above=POSITIVE, token_pattern=TOKEN.pattern),
        started=started.isoformat(timespec='seconds'), finished=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        diagnostics=dict(checks=checks, length_anchors=anchors, model_fitted=False))
    (RUN / 'run.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    log(f'{ITEM}: done, all {len(checks)} checks pass')


if __name__ == '__main__':
    main()
