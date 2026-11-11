"""A15: parent-contribution disagreement with the replying participant (CHB pre-submission review).

Named run ``chb_prereview_20261110``, item ``a15_parent_disagreement``.

Unit: a (parent contribution, replying participant) pair from the 2,547-reply
criticism cohort (public ``criticism_labels.csv`` rows with ``in_criticism_model``).
Every pair whose parent is a comment (non-root) is annotated by GPT-5.6 Sol through
the Codex CLI; root parents receive the structural value 0 (a root post cannot be
answering the replying participant).

The annotator sees, for each pair, the thread root, the contribution the parent was
answering (the parent's own parent) plus up to two earlier ancestors, the replying
participant's nearest earlier contribution in the branch when it lies above that
window, and the parent text. The focal participant reply (the criticism outcome) is
never shown. Account identities are replaced by neutral speaker letters: the replying
participant is always Speaker A; mentions of usernames or account codes become
"@Speaker X" for accounts shown in the item and "@another user" otherwise. Community
and account type are never shown.

Stages::

    python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement build
    python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement annotate --pass main [--batches 0]
    python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement annotate --pass selfcheck
    python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement collect
    python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement clipping

Shares are per (parent contribution, replying participant) pair, not per contribution.
Texts longer than the limits are clipped (``clip``); the ``clipping`` stage counts
them from the stored items.

``build``, ``collect`` and ``clipping`` are deterministic and need no network access. ``annotate``
calls ``codex exec`` with at most two concurrent processes and retries malformed
batches. Prompts and raw outputs are kept in the run's ``outputs/`` directory.

From the public record, run ``build``, ``collect`` and ``clipping``: when this run has
no ``outputs/annotation/`` folder, ``collect`` reads the labels of the reported passes
from ``08_analysis_ready/further_checks/a15_parent_disagreement/`` in the Zenodo record
after checking that they cover the rebuilt batch plan. Running ``annotate`` instead
produces a new annotation, which a language model does not repeat exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a15_parent_disagreement'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TREE = ROOT.parent / 'data'
POSTS = TREE / '02_platform_activity/posts.csv'
ACCOUNTS = TREE / '01_accounts/accounts.csv'
LABELS = TREE / '07_annotations/llm_annotations/criticism_labels.csv'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
# Labels of the reported annotation passes, deposited in the Zenodo record (version 1.2.0 or later).
DEPOSITED = TREE / '08_analysis_ready/further_checks/a15_parent_disagreement'
SEED = 20261110
BATCH_SIZE = 25
SELFCHECK_SHARE = 0.10
MAX_PARALLEL = 2
MAX_ATTEMPTS = 4
MODEL = 'gpt-5.6-sol'
REASONING = 'medium'
CATEGORIES = ('content', 'style_conduct', 'participation', 'antagonistic_only')
ROOT_CHARS = 800
TEXT_CHARS = 1500
EARLIER_ANCESTORS = 2
CODE_PATTERN = re.compile(r'\bG[12]-(?:ADMIN|[PA]\d+)\b', re.IGNORECASE)
SPEAKERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


# --------------------------------------------------------------------------- utilities

def log(message: str) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (RUN / 'run.log').open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float('nan'), float('nan')
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a) | set(b))
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == k)) * float(np.mean(b == k)) for k in labels)
    return float('nan') if expected == 1 else (observed - expected) / (1 - expected)


# --------------------------------------------------------------------------- frame construction

def cohort(labels: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """The 2,547 criticism-model replies with the parent kind attached."""
    frame = labels.loc[labels.in_criticism_model.astype(str).str.lower().eq('true')].copy()
    kinds = posts.set_index(['group', 'post_id']).post_kind
    frame['parent_kind'] = kinds.reindex(list(zip(frame.group, frame.parent_content_id))).to_numpy()
    if frame.parent_kind.isna().any():
        raise ValueError('Cohort parent missing from posts.csv')
    return frame.reset_index(drop=True)


def pairs_of(frame: pd.DataFrame) -> pd.DataFrame:
    """Unique (group, parent contribution, replying participant) pairs."""
    keys = ['group', 'parent_content_id', 'author_code']
    pairs = (frame.groupby(keys, sort=True)
             .agg(parent_author_code=('parent_author_code', 'first'), parent_author_type=('parent_author_type', 'first'),
                  parent_kind=('parent_kind', 'first'), root_id=('root_id', 'first'), replies=('content_id', 'size'))
             .reset_index())
    pairs['pair_id'] = ['p%04d' % i for i in range(len(pairs))]
    return pairs


def ancestors(posts_by_id: dict, group: str, post_id: int) -> list[dict]:
    """Ancestors of ``post_id`` nearest first, ending with the root."""
    chain, current = [], posts_by_id[(group, post_id)]
    seen = {post_id}
    while int(current['comment_to']) != -1:
        parent_id = int(current['comment_to'])
        if parent_id in seen or (group, parent_id) not in posts_by_id:
            raise ValueError(f'Broken reply chain at {group}:{post_id}')
        seen.add(parent_id)
        current = posts_by_id[(group, parent_id)]
        chain.append(current)
    return chain


def context_rows(chain: list[dict], focal_code: str, window: int = EARLIER_ANCESTORS):
    """Select the shown ancestors from a parent's ancestor chain (nearest first, root last).

    Returns (root, shown_non_root_oldest_first, gap_before_window, extra_focal, structural):
    the answered contribution is chain[0]; up to ``window`` earlier non-root ancestors follow it;
    if the replying participant authored a non-root ancestor above the window, the nearest such
    one is added. ``structural`` describes whether the participant appears anywhere in the chain.
    """
    root = chain[-1]
    non_root = chain[:-1]
    shown = non_root[:1 + window]
    hidden = non_root[1 + window:]
    extra = next((row for row in hidden if row['author_code'] == focal_code), None)
    gap = len(hidden) > 0
    answered_by_focal = chain[0]['author_code'] == focal_code
    focal_in_chain = any(row['author_code'] == focal_code for row in chain)
    structural = 'answers_participant' if answered_by_focal else (
        'participant_earlier_in_branch' if focal_in_chain else 'participant_not_in_branch')
    return root, list(reversed(shown)), gap, extra, structural


def mask_text(text: str, speaker_of: dict, names: dict) -> str:
    """Replace account codes and usernames by speaker letters (or 'another user')."""
    text = '' if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)

    def code_sub(match):
        code = match.group(0).upper()
        letter = speaker_of.get(code)
        return f'Speaker {letter}' if letter else 'another user'

    text = CODE_PATTERN.sub(code_sub, text)
    for name, code in names.items():
        pattern = re.compile(r'(?<![\w])' + re.escape(name) + r'(?![\w])', re.IGNORECASE)
        if pattern.search(text):
            letter = speaker_of.get(code)
            text = pattern.sub(f'Speaker {letter}' if letter else 'another user', text)
    return text


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + ' [...]'


def build_item(pair, posts_by_id: dict, names: dict) -> dict:
    group, parent_id, focal = pair.group, int(pair.parent_content_id), pair.author_code
    parent = posts_by_id[(group, parent_id)]
    chain = ancestors(posts_by_id, group, parent_id)
    root, shown, gap, extra, structural = context_rows(chain, focal)
    speaker_of = {focal.upper(): 'A'}

    def letter(code):
        code = str(code).upper()
        if code not in speaker_of:
            speaker_of[code] = SPEAKERS[len(speaker_of)]
        return speaker_of[code]

    ordered = [root] + ([extra] if extra is not None else []) + shown + [parent]
    for row in ordered:
        letter(row['author_code'])
    answered_id = int(chain[0]['post_id'])
    context = []
    root_answered = int(root['post_id']) == answered_id
    context.append(dict(position='thread_root', speaker=letter(root['author_code']),
                        role='contribution answered by the target' if root_answered else 'thread root post',
                        text=clip(mask_text(root['text'], speaker_of, names), TEXT_CHARS if root_answered else ROOT_CHARS)))
    if extra is not None:
        context.append(dict(position='earlier_in_branch', speaker=letter(extra['author_code']),
                            role="Speaker A's nearest earlier contribution in this branch",
                            text=clip(mask_text(extra['text'], speaker_of, names), TEXT_CHARS)))
    if gap:
        context.append(dict(position='omitted', speaker='', role='earlier contributions in this branch omitted',
                            text=''))
    for row in shown:
        is_answered = int(row['post_id']) == answered_id
        context.append(dict(position='ancestor', speaker=letter(row['author_code']),
                            role='contribution answered by the target' if is_answered else 'earlier contribution',
                            text=clip(mask_text(row['text'], speaker_of, names), TEXT_CHARS)))
    target = dict(speaker=letter(parent['author_code']), text=clip(mask_text(parent['text'], speaker_of, names), TEXT_CHARS))
    return dict(item_id=pair.pair_id, structural=structural, context=context, target=target,
                answered_speaker=letter(chain[0]['author_code']))


# --------------------------------------------------------------------------- prompt and schema

CODEBOOK = """You are annotating contributions in online discussion threads for a research study.
Work only from the text in this message. Do not run commands, open files or browse.

For each item, decide whether the TARGET contribution (written by the speaker named in `target.speaker`)
disagrees with, criticizes, or acts antagonistically toward SPEAKER A or toward a contribution that
Speaker A made earlier in the shown thread. Speaker A is always the same person within an item;
other letters are other accounts. All speaker letters are neutral and say nothing about who the
accounts are.

Label YES when any of these categories applies to Speaker A or Speaker A's contribution; otherwise NO.
Give the single most prominent category for YES:
- content: an expressed challenge, disagreement or negative evaluation of the accuracy, reasoning,
  relevance or usefulness of Speaker A's contribution. It can be polite, blunt, sarcastic or phrased
  as a question. Ordinary, polite disagreement qualifies without any hostility ("I don't think that's
  right", "Not really, it was the opposite").
- style_conduct: an objection to how Speaker A writes or behaves in the conversation, such as
  rudeness, aggression, condescension, repetitive posting, writing style or inappropriate intervention.
- participation: a request that Speaker A stop replying, leave or disengage, or a denial of Speaker A's
  standing to take part.
- antagonistic_only: insults, mockery, contempt or hostility aimed at Speaker A personally, without
  any of the three categories above. An insult alone counts here.

Label NO for: agreement; ordinary answers, recommendations or added information; neutral
clarification or follow-up questions; offering an alternative without challenging Speaker A;
criticism solely of outside people, media, products, institutions or tools (even when Speaker A
mentioned them) unless it challenges Speaker A's claim or evaluation; disagreement or criticism aimed
only at speakers other than Speaker A; profanity or jokes not aimed at Speaker A.

Target rule. The target usually answers a contribution by Speaker A (its role is "contribution
answered by the target"). If the target answers another speaker, label YES only if the target
disagrees with or criticizes Speaker A's earlier contribution shown in the context, or Speaker A
personally; otherwise NO. If Speaker A neither appears in the context nor is addressed in the target,
label NO.

Classify only the target text. Criticism that appears only in the context does not make the target
YES. The context may be incomplete; do not invent missing exchanges. Do not consider whether any
account is a person or an automated account.

For each item return: item_id; label (YES or NO); category (one of content, style_conduct,
participation, antagonistic_only for YES; none for NO); evidence: for YES a phrase of at most 20 words
quoted from the target that shows the stance; for NO a reason of at most 20 words.
Return exactly one result per item, in the order given, as JSON matching the schema.
"""


def schema() -> dict:
    result = {
        'type': 'object', 'additionalProperties': False,
        'required': ['item_id', 'label', 'category', 'evidence'],
        'properties': {
            'item_id': {'type': 'string'},
            'label': {'type': 'string', 'enum': ['YES', 'NO']},
            'category': {'type': 'string', 'enum': [*CATEGORIES, 'none']},
            'evidence': {'type': 'string'},
        },
    }
    return {'type': 'object', 'additionalProperties': False, 'required': ['results'],
            'properties': {'results': {'type': 'array', 'items': result}}}


def prompt_for(items: list[dict]) -> str:
    shown = [dict(item_id=item['item_id'], context=[{k: v for k, v in c.items() if k != 'position'} for c in item['context']],
                  target=item['target']) for item in items]
    return CODEBOOK + '\nItems (JSON):\n' + json.dumps(shown, ensure_ascii=False, indent=1) + '\n'


def validate(result: dict, item_ids: list[str]) -> list[dict]:
    rows = result.get('results') if isinstance(result, dict) else None
    if not isinstance(rows, list) or [r.get('item_id') for r in rows] != item_ids:
        if isinstance(rows, list) and sorted(r.get('item_id') for r in rows) == sorted(item_ids):
            rows = sorted(rows, key=lambda r: item_ids.index(r['item_id']))
        else:
            raise ValueError('Result item ids do not match the batch')
    for row in rows:
        if row.get('label') not in ('YES', 'NO'):
            raise ValueError('Bad label')
        if (row['label'] == 'YES') != (row.get('category') in CATEGORIES) or row.get('category') not in (*CATEGORIES, 'none'):
            raise ValueError('Label and category disagree')
        if not isinstance(row.get('evidence'), str) or len(row['evidence'].split()) > 30:
            raise ValueError('Evidence missing or too long')
    return rows


# --------------------------------------------------------------------------- stages

def stage_build(args) -> None:
    posts = pd.read_csv(POSTS, low_memory=False)
    accounts = pd.read_csv(ACCOUNTS)
    labels = pd.read_csv(LABELS)
    frame = cohort(labels, posts)
    if len(frame) != 2547:
        raise ValueError('Cohort must contain 2,547 replies')
    pairs = pairs_of(frame)
    counts = pairs.groupby(['parent_kind', 'parent_author_type']).size()
    log(f'build: {len(frame)} replies -> {len(pairs)} pairs; by parent kind and type {counts.to_dict()}')
    posts_by_id = {(r['group'], int(r['post_id'])): r for r in posts.to_dict('records')}
    names = {}
    for row in accounts.itertuples():
        if isinstance(row.display_name, str) and not CODE_PATTERN.fullmatch(row.display_name):
            names[row.display_name] = str(row.account_code).upper()
    names = dict(sorted(names.items(), key=lambda kv: -len(kv[0])))
    items, structural = [], {}
    for pair in pairs.itertuples():
        if pair.parent_kind == 'root':
            structural[pair.pair_id] = 'root_parent'
            continue
        if posts_by_id[(pair.group, int(pair.parent_content_id))]['author_code'] != pair.parent_author_code:
            raise ValueError('Parent author mismatch')
        item = build_item(pair, posts_by_id, names)
        structural[pair.pair_id] = item['structural']
        items.append(item)
    pairs['structural'] = pairs.pair_id.map(structural)
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT / 'pairs.csv', index=False)
    leak = [i['item_id'] for i in items if CODE_PATTERN.search(json.dumps(i))
            or any(re.search(r'(?<![\w])' + re.escape(n) + r'(?![\w])', json.dumps(i, ensure_ascii=False), re.I)
                   for n in names)]
    if leak:
        raise ValueError(f'Identity tokens survive masking in {leak[:5]}')
    with (OUT / 'items.jsonl').open('w') as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    (OUT / 'schema.json').write_text(json.dumps(schema(), indent=1) + '\n')
    rng = random.Random(SEED)
    order = list(range(len(items)))
    batches = [order[i:i + BATCH_SIZE] for i in range(0, len(order), BATCH_SIZE)]
    n_check = round(SELFCHECK_SHARE * len(items))
    check = rng.sample(order, n_check)
    rng.shuffle(check)
    check_batches = [check[i:i + BATCH_SIZE] for i in range(0, len(check), BATCH_SIZE)]
    plan = dict(main=[[items[i]['item_id'] for i in b] for b in batches],
                selfcheck=[[items[i]['item_id'] for i in b] for b in check_batches])
    (OUT / 'batches.json').write_text(json.dumps(plan, indent=1) + '\n')
    log(f'build: {len(items)} annotated items in {len(batches)} batches; self-check {n_check} items in '
        f'{len(check_batches)} batches (seed {SEED}); structural {pd.Series(structural).value_counts().to_dict()}')


def run_batch(pass_name: str, index: int, items: list[dict]) -> dict:
    folder = OUT / 'annotation' / pass_name
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / f'batch_{index:03d}.result.json'
    ids = [i['item_id'] for i in items]
    if done.exists():
        return dict(batch=index, status='cached')
    prompt = prompt_for(items)
    (folder / f'batch_{index:03d}.prompt.txt').write_text(prompt)
    workdir = OUT / 'codex_empty_workdir'
    workdir.mkdir(exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = folder / f'batch_{index:03d}.attempt{attempt}.raw.json'
        stderr = folder / f'batch_{index:03d}.attempt{attempt}.stderr.txt'
        command = ['codex', 'exec', '-m', MODEL, '-c', f'model_reasoning_effort="{REASONING}"', '-s', 'read-only',
                   '--ephemeral', '--skip-git-repo-check', '-C', str(workdir), '--output-schema', str(OUT / 'schema.json'),
                   '-o', str(raw), '-']
        started = time.monotonic()
        with stderr.open('w') as err:
            proc = subprocess.run(command, input=prompt, text=True, stdout=err, stderr=subprocess.STDOUT, timeout=1800)
        elapsed = time.monotonic() - started
        try:
            if proc.returncode != 0:
                raise ValueError(f'codex exit {proc.returncode}')
            rows = validate(json.loads(raw.read_text()), ids)
        except Exception as error:  # noqa: BLE001 - retry any malformed output
            log(f'annotate {pass_name} batch {index} attempt {attempt}: failed ({error}) after {elapsed:.0f}s')
            continue
        done.write_text(json.dumps(dict(batch=index, attempt=attempt, elapsed_seconds=elapsed, results=rows),
                                   ensure_ascii=False, indent=1) + '\n')
        log(f'annotate {pass_name} batch {index}: ok on attempt {attempt} in {elapsed:.0f}s; '
            f'YES {sum(r["label"] == "YES" for r in rows)}/{len(rows)}')
        return dict(batch=index, status='ok', attempt=attempt)
    return dict(batch=index, status='failed')


def stage_annotate(args) -> None:
    items = {json.loads(line)['item_id']: json.loads(line) for line in (OUT / 'items.jsonl').read_text().splitlines()}
    plan = json.loads((OUT / 'batches.json').read_text())[args.pass_name]
    wanted = range(len(plan)) if args.batches is None else args.batches
    jobs = [(i, [items[k] for k in plan[i]]) for i in wanted]
    if args.limit_items:
        jobs = [(i, batch[:args.limit_items]) for i, batch in jobs]
        args.pass_name = args.pass_name + '_test'
    log(f'annotate {args.pass_name}: {len(jobs)} batches, {MAX_PARALLEL} parallel, model {MODEL} ({REASONING})')
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        outcomes = list(pool.map(lambda job: run_batch(args.pass_name, *job), jobs))
    failed = [o['batch'] for o in outcomes if o['status'] == 'failed']
    log(f'annotate {args.pass_name}: {len(outcomes) - len(failed)} batches ok, failed {failed}')


def read_pass(name: str, plan: list[list[str]]) -> pd.DataFrame:
    """Validated labels of one pass: this run's batch results, else the deposited labels."""
    if not (OUT / 'annotation' / name).exists() and (DEPOSITED / f'labels_{name}.csv').exists():
        labels = pd.read_csv(DEPOSITED / f'labels_{name}.csv')
        expected = [item for batch in plan for item in batch]
        if sorted(labels.item_id) != sorted(expected):
            raise ValueError(f'Deposited {name} labels do not match the rebuilt batch plan')
        log(f'collect: {name} labels read from {DEPOSITED.name}/labels_{name}.csv (no new annotation)')
        return labels
    rows = []
    for index in range(len(plan)):
        path = OUT / 'annotation' / name / f'batch_{index:03d}.result.json'
        if not path.exists():
            raise FileNotFoundError(f'{name} batch {index} has no validated result')
        payload = json.loads(path.read_text())
        rows += [dict(r, batch=index, attempt=payload['attempt']) for r in payload['results']]
    return pd.DataFrame(rows)


def reply_indicator(frame: pd.DataFrame, pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Per-reply 0/1 parent-disagreement indicator; root parents are structurally 0."""
    merged = frame.merge(pairs[['group', 'parent_content_id', 'author_code', 'pair_id', 'parent_kind']],
                         on=['group', 'parent_content_id', 'author_code', 'parent_kind'], how='left', validate='many_to_one')
    if merged.pair_id.isna().any():
        raise ValueError('Reply without pair')
    value = merged.pair_id.map(labels.set_index('item_id').label.eq('YES').astype(int))
    value = value.where(merged.parent_kind.ne('root'), 0)
    if value.isna().any():
        raise ValueError('Non-root pair without a label')
    return value.astype(int)


def stage_collect(args) -> None:
    posts = pd.read_csv(POSTS, low_memory=False)
    labels = pd.read_csv(LABELS)
    frame = cohort(labels, posts)
    pairs = pd.read_csv(OUT / 'pairs.csv')
    plan = json.loads((OUT / 'batches.json').read_text())
    main = read_pass('main', plan['main'])
    check = read_pass('selfcheck', plan['selfcheck'])
    if main.item_id.duplicated().any() or set(main.item_id) != set(pairs.loc[pairs.parent_kind.ne('root'), 'pair_id']):
        raise ValueError('Main pass must cover each non-root pair once')
    main.to_csv(OUT / 'labels_main.csv', index=False)
    check.to_csv(OUT / 'labels_selfcheck.csv', index=False)
    labelled = pairs.merge(main[['item_id', 'label', 'category', 'evidence']], left_on='pair_id', right_on='item_id',
                           how='left')
    labelled['disagree'] = labelled.label.eq('YES').astype(int)
    frame['parent_disagreement'] = reply_indicator(frame, pairs, main).to_numpy()
    frame[['group', 'content_id', 'parent_content_id', 'author_code', 'parent_author_type', 'parent_kind',
           'parent_disagreement']].to_csv(OUT / 'reply_parent_disagreement.csv', index=False)
    community = {'G1': 'visible', 'G2': 'hidden'}
    rows = []
    for unit, data, flag in (('pair', labelled, 'disagree'), ('reply', frame, 'parent_disagreement')):
        for scope in ('non_root_parents', 'all_parents'):
            part = data if scope == 'all_parents' else data.loc[data.parent_kind.ne('root')]
            for (group, typ), cell in part.groupby(['group', 'parent_author_type']):
                k, n = int(cell[flag].sum()), len(cell)
                low, high = wilson(k, n)
                rows.append(dict(run_id=RUN_ID, unit=unit, scope=scope, group=group, community=community[group],
                                 parent_author_type=typ, yes=k, denominator=n, share=k / n, wilson_low=low,
                                 wilson_high=high))
    shares = pd.DataFrame(rows)
    categories = (labelled.loc[labelled.label.eq('YES')].groupby(['group', 'parent_author_type', 'category']).size()
                  .rename('pairs').reset_index())
    categories['denominator_yes_pairs'] = categories.groupby(['group', 'parent_author_type']).pairs.transform('sum')
    categories['run_id'] = RUN_ID
    structural = (labelled.groupby(['group', 'parent_author_type', 'structural'])
                  .agg(pairs=('pair_id', 'size'), yes=('disagree', 'sum')).reset_index())
    structural['run_id'] = RUN_ID
    both = check.merge(main, on='item_id', suffixes=('_selfcheck', '_main'), validate='one_to_one')
    agree = float((both.label_selfcheck == both.label_main).mean())
    kappa = cohen_kappa(both.label_selfcheck, both.label_main)
    table = pd.crosstab(both.label_main, both.label_selfcheck)
    reliability = pd.DataFrame([dict(run_id=RUN_ID, comparison='selfcheck_vs_main_label', items=len(both),
                                     agreement=agree, cohen_kappa=kappa,
                                     yes_main=int(both.label_main.eq('YES').sum()),
                                     yes_selfcheck=int(both.label_selfcheck.eq('YES').sum()),
                                     yes_yes=int(table.loc['YES', 'YES']) if 'YES' in table.index and 'YES' in table.columns else 0,
                                     category_agreement_among_yes_yes=float(
                                         (both.loc[both.label_main.eq('YES') & both.label_selfcheck.eq('YES'), 'category_main']
                                          == both.loc[both.label_main.eq('YES') & both.label_selfcheck.eq('YES'), 'category_selfcheck']).mean()),
                                     sample_seed=SEED, denominator_non_root_pairs=int(pairs.parent_kind.ne('root').sum()),
                                     human_validation='none')])
    TABLES.mkdir(parents=True, exist_ok=True)
    shares.to_csv(TABLES / 'chb_prereview_a15_parent_disagreement_shares.csv', index=False)
    categories.to_csv(TABLES / 'chb_prereview_a15_parent_disagreement_categories.csv', index=False)
    structural.to_csv(TABLES / 'chb_prereview_a15_parent_disagreement_structure.csv', index=False)
    reliability.to_csv(TABLES / 'chb_prereview_a15_parent_disagreement_selfconsistency.csv', index=False)
    log(f'collect: shares {shares.loc[shares.unit.eq("pair") & shares.scope.eq("non_root_parents")].to_dict("records")}')
    log(f'collect: self-consistency {len(both)} items, agreement {agree:.3f}, kappa {kappa:.3f}')
    attempts = pd.concat([main.assign(pass_name='main'), check.assign(pass_name='selfcheck')])
    record = dict(
        run_id=RUN_ID, item=ITEM,
        purpose='GPT-5.6 Sol (Codex CLI) annotation of whether the parent contribution disagrees with, criticizes or '
                'is antagonistic toward the replying participant; covariate for A14.',
        host=platform.node(), command=' '.join(['python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement',
                                                 'build|annotate --pass main|annotate --pass selfcheck|collect']),
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        inputs={str(p.relative_to(ROOT.parent)): sha256(p) for p in (POSTS, ACCOUNTS, LABELS)},
        annotation=dict(tool='codex-cli 0.167.0 (ChatGPT login)', model=MODEL, reasoning_effort=REASONING,
                        sandbox='read-only', batch_size=BATCH_SIZE, max_parallel=MAX_PARALLEL,
                        main_batches=len(plan['main']), selfcheck_batches=len(plan['selfcheck']),
                        attempts_distribution=attempts.groupby(['pass_name', 'batch']).attempt.first().value_counts().to_dict(),
                        schema_sha256=sha256(OUT / 'schema.json'), items_sha256=sha256(OUT / 'items.jsonl')),
        denominators=dict(replies=len(frame), pairs=len(pairs), non_root_pairs=int(pairs.parent_kind.ne('root').sum()),
                          root_pairs=int(pairs.parent_kind.eq('root').sum()),
                          by_kind_type={f'{k[0]}_{k[1]}': int(v) for k, v in
                                        pairs.groupby(['parent_kind', 'parent_author_type']).size().items()}),
        diagnostics=dict(selfcheck_items=len(both), agreement=agree, cohen_kappa=kappa, human_validation='none'),
        packages=dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__),
        start_end=dict(started=(RUN / 'run.log').read_text().split(' ', 1)[0],
                       collected=dt.datetime.now().astimezone().isoformat(timespec='seconds')),
        test_batch='annotation/main_test: batch 3, first 3 items, before the main pass',
        tables=[str((TABLES / f'chb_prereview_a15_parent_disagreement_{n}.csv').relative_to(ROOT.parent))
                for n in ('shares', 'categories', 'structure', 'selfconsistency')],
    )
    (RUN / 'run.json').write_text(json.dumps(record, indent=1, default=str) + '\n')


CLIP_MARK = ' [...]'


def clipping_table(items: list[dict], pairs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Counts of shown texts clipped by ``clip`` (suffix ' [...]'), by text role and (group, parent type).

    Roles: ``target`` (the annotated parent), ``answered`` (the contribution the parent answers, limit
    TEXT_CHARS), ``earlier`` (other non-root context, TEXT_CHARS) and ``thread_root_context`` (an unanswered
    thread root, ROOT_CHARS). ``items`` is the number of annotated pairs in the cell.
    """
    meta = pairs.set_index('pair_id')[['group', 'parent_author_type']]
    yes = labels.set_index('item_id').label.eq('YES')
    rows = []
    for item in items:
        group, typ = meta.loc[item['item_id']]
        texts = [('target', TEXT_CHARS, item['target']['text'])]
        for entry in item['context']:
            if entry['position'] == 'omitted':
                continue
            if entry['role'] == 'contribution answered by the target':
                texts.append(('answered', TEXT_CHARS, entry['text']))
            elif entry['position'] == 'thread_root':
                texts.append(('thread_root_context', ROOT_CHARS, entry['text']))
            else:
                texts.append(('earlier', TEXT_CHARS, entry['text']))
        for role, limit, text in texts:
            clipped = text.endswith(CLIP_MARK)
            if clipped and len(text) > limit + len(CLIP_MARK):
                raise ValueError('Clipped text longer than its limit')
            rows.append(dict(group=group, parent_author_type=typ, text_role=role, limit_chars=limit,
                             clipped=int(clipped), clipped_yes=int(clipped and yes[item['item_id']])))
    frame = pd.DataFrame(rows)
    table = (frame.groupby(['group', 'parent_author_type', 'text_role', 'limit_chars'])
             .agg(texts_shown=('clipped', 'size'), clipped_texts=('clipped', 'sum'),
                  clipped_texts_in_yes_pairs=('clipped_yes', 'sum')).reset_index())
    items_per_cell = meta.loc[[i['item_id'] for i in items]].groupby(['group', 'parent_author_type']).size()
    table['annotated_pairs'] = [int(items_per_cell[(g, t)]) for g, t in zip(table.group, table.parent_author_type)]
    table['run_id'] = RUN_ID
    return table


def stage_clipping(args) -> None:
    """Fix-up of 2026-11-11: disclose texts clipped in the annotation items (no new annotation)."""
    items = [json.loads(line) for line in (OUT / 'items.jsonl').read_text().splitlines()]
    pairs = pd.read_csv(OUT / 'pairs.csv')
    labels = pd.read_csv(OUT / 'labels_main.csv')
    table = clipping_table(items, pairs, labels)
    path = TABLES / 'chb_prereview_a15_parent_disagreement_clipping.csv'
    table.to_csv(path, index=False)
    totals = table.groupby('text_role').clipped_texts.sum().to_dict()
    log(f'clipping: clipped texts by role {totals} over {len(items)} annotated pairs')
    record = json.loads((RUN / 'run.json').read_text())
    record['fixups_20261111'] = dict(
        changes=[
            'Summary wording: the AI-parent disagreement shares are reported as pair shares with their difference '
            '(visible 95/200 = 47.5%, hidden 84/176 = 47.7%, visible minus hidden -0.2 points), not as "did not '
            'differ"; GPT-5.6 Sol labels without human validation; the unit is the (non-root AI parent, replying '
            'participant) pair, '
            'not the contribution (183 and 156 distinct AI parent contributions).',
            'Disclosure: texts clipped in the annotation items, from the existing items.jsonl (no new annotation).',
        ],
        clipped_texts_by_role={k: int(v) for k, v in totals.items()},
        clipping_limits=dict(answered_target_earlier=TEXT_CHARS, unanswered_thread_root=ROOT_CHARS),
        distinct_non_root_parents={f'{g}_{t}': int(n) for (g, t), n in
                                   pairs.loc[pairs.parent_kind.ne('root')].groupby(['group', 'parent_author_type'])
                                   .parent_content_id.nunique().items()},
        command='python -m scripts.diagnostics.chb_prereview_a15_parent_disagreement clipping',
        new_outputs=[str(path.relative_to(ROOT.parent))],
        script_sha256=sha256(Path(__file__)),
        test_sha256=sha256(ROOT / 'tests/test_chb_prereview_a15_parent_disagreement.py'),
        collected=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
    )
    (RUN / 'run.json').write_text(json.dumps(record, indent=1, default=str) + '\n')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    sub.add_parser('build')
    annotate = sub.add_parser('annotate')
    annotate.add_argument('--pass', dest='pass_name', choices=['main', 'selfcheck'], required=True)
    annotate.add_argument('--batches', type=int, nargs='*')
    annotate.add_argument('--limit-items', type=int, help='test: send only the first N items of each batch')
    sub.add_parser('collect')
    sub.add_parser('clipping')
    args = parser.parse_args(argv)
    {'build': stage_build, 'annotate': stage_annotate, 'collect': stage_collect,
     'clipping': stage_clipping}[args.stage](args)


if __name__ == '__main__':
    main()
