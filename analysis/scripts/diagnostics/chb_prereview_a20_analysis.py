"""A20 analysis: paired marker ON minus OFF outcomes of the regenerated G1 AI replies.

Named run ``chb_prereview_20261110``, item ``a20_marker``. Input: the 480 draws of the
regeneration run on the private host (``full_20261111``: 60 contexts x 2 arms x 4 draws). The two
arms of a (context, draw) pair share every seed (common random numbers), so every
estimand is computed on the 240 pairs and its uncertainty comes from a cluster
bootstrap over the 60 contexts and a permutation test that swaps the arm labels within
each pair.

Stages::

    # private host (one core, nice 10): per-draw frame without text, plus masked drafts
    python chb_prereview_a20_analysis.py frame --src FULL_DIR --dest ANALYSIS_DIR
    # from the public record
    python -m scripts.diagnostics.chb_prereview_a20_analysis items
    python -m scripts.diagnostics.chb_prereview_a20_analysis annotate --pass main
    python -m scripts.diagnostics.chb_prereview_a20_analysis annotate --pass selfcheck
    python -m scripts.diagnostics.chb_prereview_a20_analysis collect

``frame`` strips any echoed "[human]" token or IDENTITY MARKERS note from the drafts,
replaces every username or account code (with or without "@") by the A15 speaker letter
of that account in the annotation item, or by "another user", and writes only
``frame.csv`` (per-draw outcomes, no text) and ``drafts_masked.jsonl``. ``items`` builds
A15-format annotation items: the context (thread root, the participant contribution the
draft answers and up to two earlier ancestors) comes from the public release, the target
is the masked draft. Drafts are shuffled across arms and contexts and get neutral item
ids; the arm never reaches the annotator. ``annotate`` runs the A15 codebook, schema and
batching through the Codex CLI (two processes at most). ``collect`` merges the labels and
writes the three result tables and the ``analysis`` section of the run record.

From the public record, run ``items`` and ``collect``. Both copy ``frame.csv``,
``drafts_masked.jsonl`` and ``frame_summary.json`` from
``08_analysis_ready/further_checks/a20_marker/`` in the Zenodo record when this run has
none, and ``collect`` reads the deposited labels when this run has no
``outputs/annotation/`` folder. The ``frame`` stage and the regeneration itself
(``chb_prereview_a20_marker.py``) need the experiment database, the archived client and
the generator model, which are not public.
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
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.diagnostics import chb_prereview_a15_parent_disagreement as a15
except ImportError:  # private host: both files are copied side by side
    import chb_prereview_a15_parent_disagreement as a15

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'chb_prereview_20261110'
ITEM = 'a20_marker'
RUN = ROOT / 'runs' / RUN_ID / ITEM
OUT = RUN / 'outputs'
TABLES = ROOT / 'runs/chb_prereview_20261110/tables'
TREE = ROOT.parent / 'data'
# Per-draw frame, masked drafts and labels of the reported run, deposited in the Zenodo record (1.2.0 or later).
DEPOSITED = TREE / '08_analysis_ready/further_checks/a20_marker'
POSTS = TREE / '02_platform_activity/posts.csv'
ACCOUNTS = TREE / '01_accounts/accounts.csv'
SEED = 20261110
N_BOOT = 10_000
N_PERM = 10_000
Z_MDD = 1.959963984540054 + 0.8416212335729143  # two-sided alpha 0.05, power 0.80
ARMS = ('on', 'off')
ADVERSARIAL = ('TROLL', 'TROLL_SOFT', 'SHITPOST', 'QUICK_DISAGREE')
STRATA = {'high_moderately_extremely': ('moderately', 'extremely'), 'low_absolutely_no_no': ('absolutely no', 'no')}
SYNTHETIC_ID = 10 ** 9
TOKEN_RE = re.compile(r'[ \t]*\[\s*human\s*\]', re.IGNORECASE)
NOTE_LINES = ('IDENTITY MARKERS:',
              'Some entries may include [human], meaning the account is a human participant.',
              'Entries without this marker have no identity marker.')
NOTE_RE = re.compile(r'identity markers?\b|meaning the account is a human participant|have no identity marker',
                     re.IGNORECASE)
HANDLE_RE = re.compile(r'@(?!Speaker [A-Z]\b|another user\b)[A-Za-z0-9_][\w.\-]*')
FRAME_COLUMNS = ('key', 'context_id', 'arm', 'draw', 'stratum', 'path', 'status', 'would_post', 'not_posted_reason',
                 'selected_style', 'final_text_source', 'has_text', 'detoxify_toxicity', 'length_chars',
                 'vader_compound', 'alignment_rewrite_called', 'length_rewrite_used', 'writing_actions_this_round',
                 'seconds', 'llm_seconds', 'llm_calls', 'seeds_sha256', 'text_sha256', 'token_echo', 'note_echo',
                 'ai_post_id', 'parent_post_id', 'thread_root_id')


# --------------------------------------------------------------------------- utilities

def log(message: str, path: Path | None = None) -> None:
    path = path or RUN / 'analysis.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f'{dt.datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with path.open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def sha256(path: Path) -> str:
    return a15.sha256(Path(path))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- marker echo and masking

def marker_echo(text: str) -> dict:
    """Whether a draft echoes the per-line "[human]" marker or the IDENTITY MARKERS note."""
    text = text or ''
    return {'token_echo': bool(TOKEN_RE.search(text)), 'note_echo': bool(NOTE_RE.search(text))}


def strip_marker_text(text: str) -> str:
    """Remove echoed note lines and "[human]" tokens before annotation."""
    kept = []
    for line in (text or '').splitlines():
        for note in NOTE_LINES:
            line = re.sub(re.escape(note), '', line, flags=re.IGNORECASE)
        if re.search(r'identity markers?\b', line, re.IGNORECASE):
            continue
        kept.append(line)
    out = TOKEN_RE.sub('', '\n'.join(kept))
    return re.sub(r'[ \t]{2,}', ' ', out).strip()


def draft_chain(posts_by_id: dict, group: str, parent_id: int) -> list[dict]:
    """The participant contribution a draft answers followed by its ancestors (root last)."""
    parent = posts_by_id[(group, parent_id)]
    return [parent] + a15.ancestors(posts_by_id, group, parent_id)


def speaker_letters(chain: list[dict], focal_code: str, target_code: str) -> dict:
    """A15 letter assignment for a draft written by ``target_code`` in reply to ``chain[0]``.

    Speaker A is the participant; letters follow the A15 order root, earlier focal
    contribution, shown ancestors (oldest first), target.
    """
    root, shown, _, extra, _ = a15.context_rows(chain, focal_code)
    letters = {str(focal_code).upper(): 'A'}
    for row in [root] + ([extra] if extra is not None else []) + shown:
        code = str(row['author_code']).upper()
        letters.setdefault(code, a15.SPEAKERS[len(letters)])
    letters.setdefault(str(target_code).upper(), a15.SPEAKERS[len(letters)])
    return letters


def mask_draft(text: str, letters: dict, names: dict) -> tuple[str, dict]:
    """Mask usernames and account codes as A15 does; any remaining @handle becomes "@another user"."""
    counts = {'bare_name_matches': 0, 'at_name_matches': 0, 'other_handles': 0}
    for name in names:
        pattern = re.compile(r'(?<![\w])(@?)' + re.escape(name) + r'(?![\w])', re.IGNORECASE)
        for match in pattern.finditer(text):
            counts['at_name_matches' if match.group(1) else 'bare_name_matches'] += 1
    masked = a15.mask_text(text, letters, names)
    counts['other_handles'] = len(HANDLE_RE.findall(masked))
    masked = HANDLE_RE.sub('@another user', masked)
    return masked, counts


def leaks(text: str, names: dict) -> list[str]:
    found = []
    if a15.CODE_PATTERN.search(text):
        found.append('account_code')
    if HANDLE_RE.search(text):
        found.append('handle')
    for name in names:
        if re.search(r'(?<![\w])' + re.escape(name) + r'(?![\w])', text, re.IGNORECASE):
            found.append('username')
            break
    return found


# --------------------------------------------------------------------------- frame stage (private host)

def frame_row(rec: dict, ctx: dict) -> dict:
    text = rec.get('final_text')
    echo = marker_echo(text)
    row = {k: rec.get(k) for k in FRAME_COLUMNS if k in rec}
    row.update(
        has_text=bool(text), llm_calls=len(rec.get('llm_calls') or []),
        seeds_sha256=sha256_text(json.dumps(rec['seeds'], sort_keys=True)),
        text_sha256=sha256_text(text) if text else None, **echo,
        ai_post_id=int(ctx['ai_post_id']), parent_post_id=int(ctx['parent_post_id']),
        thread_root_id=int(ctx['thread_root_id']))
    return {k: row.get(k) for k in FRAME_COLUMNS}


def stage_frame(args) -> None:
    started = time.monotonic()
    src, dest = Path(args.src).expanduser(), Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    host_log = dest / 'analysis_frame.log'
    recs = [json.loads(line) for line in (src / 'draws.jsonl').read_text().splitlines() if line.strip()]
    keys = [r['key'] for r in recs]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate draw keys')
    if any(r['status'] == 'error' for r in recs):
        raise ValueError('error rows in draws.jsonl')
    contexts = {p.stem: json.loads(p.read_text()) for p in sorted((src / 'contexts').glob('c*.json'))}
    users = {}
    for ctx in contexts.values():
        users.update({int(k): v for k, v in ctx['usernames'].items()})
    names = {name: str(uid) for uid, name in users.items() if isinstance(name, str) and len(name) >= 3}
    names = dict(sorted(names.items(), key=lambda kv: -len(kv[0])))
    short = sum(1 for name in users.values() if isinstance(name, str) and len(name) < 3)
    rows, drafts, mask_totals = [], [], {'bare_name_matches': 0, 'at_name_matches': 0, 'other_handles': 0}
    stripped = 0
    for rec in recs:
        ctx = contexts[rec['context_id']]
        rows.append(frame_row(rec, ctx))
        text = rec.get('final_text')
        if not text:
            continue
        posts_by_id = {('G1', int(p['post_id'])): {
            'post_id': int(p['post_id']), 'comment_to': -1 if p['comment_to'] is None else int(p['comment_to']),
            'author_code': str(p['user_id'])} for p in ctx['thread_posts']}
        chain = draft_chain(posts_by_id, 'G1', int(ctx['parent_post_id']))
        letters = speaker_letters(chain, str(ctx['other_user_id']), str(ctx['agent_user_id']))
        clean = strip_marker_text(text)
        stripped += int(clean != text.strip())
        masked, counts = mask_draft(clean, letters, names)
        for k, v in counts.items():
            mask_totals[k] += v
        found = leaks(masked, names) + (['marker'] if any(marker_echo(masked).values()) else [])
        if found:
            raise ValueError(f'{rec["key"]}: masked draft still has {found}')
        drafts.append({'key': rec['key'], 'text': masked,
                       'chain_post_ids': [int(r['post_id']) for r in chain],
                       'letters_by_post': {str(r['post_id']): letters[str(r['author_code']).upper()] for r in chain},
                       'target_letter': letters[str(ctx['agent_user_id'])]})
    frame = pd.DataFrame(rows)
    stamps = [line.split(' ', 1)[0] for line in (src / 'run.log').read_text().splitlines()
              if re.match(r'\d{4}-\d\d-\d\dT', line)]
    generation = {'run_log_first': stamps[0], 'run_log_last': stamps[-1],
                  'draw_seconds_sum': round(float(frame.seconds.sum()), 1),
                  'draw_seconds_median': round(float(frame.seconds.median()), 2),
                  'status_counts': {str(k): int(v) for k, v in frame.status.value_counts().items()}}
    frame.to_csv(dest / 'frame.csv', index=False)
    with (dest / 'drafts_masked.jsonl').open('w') as handle:
        for d in drafts:
            handle.write(json.dumps(d, ensure_ascii=False) + '\n')
    summary = {
        'host': platform.node(), 'python': sys.version.split()[0], 'pandas': pd.__version__,
        'inputs': {str(p.relative_to(src.parent)): sha256(p) for p in
                   [src / 'draws.jsonl', src / 'build.json', src / 'summary_by_arm.csv', src / 'summary_styles_by_arm.csv']},
        'contexts_sha256': sha256_text(''.join(sha256(p) for p in sorted((src / 'contexts').glob('c*.json')))),
        'generation_run': generation, 'draws': len(recs), 'drafts_with_text': len(drafts), 'drafts_changed_by_marker_strip': stripped,
        'token_echo_by_arm': frame.groupby('arm').token_echo.sum().astype(int).to_dict(),
        'note_echo_by_arm': frame.groupby('arm').note_echo.sum().astype(int).to_dict(),
        'usernames_masked': len(names), 'usernames_shorter_than_3_masked_only_as_handles': short,
        'masking_matches': mask_totals, 'leak_check': 'no username, account code, @handle or marker text left',
        'seconds': round(time.monotonic() - started, 2),
        'outputs': {n: sha256(dest / n) for n in ('frame.csv', 'drafts_masked.jsonl')},
    }
    (dest / 'frame_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    log('frame: ' + json.dumps(summary), host_log)


# --------------------------------------------------------------------------- items stage (public record)

def release_tables():
    posts = pd.read_csv(POSTS, low_memory=False)
    accounts = pd.read_csv(ACCOUNTS)
    posts_by_id = {(r['group'], int(r['post_id'])): r for r in posts.to_dict('records')}
    names = {}
    for row in accounts.itertuples():
        if isinstance(row.display_name, str) and not a15.CODE_PATTERN.fullmatch(row.display_name):
            names[row.display_name] = str(row.account_code).upper()
    return posts_by_id, dict(sorted(names.items(), key=lambda kv: -len(kv[0])))


def draft_item(item_id: str, draft: dict, ai_post_id: int, posts_by_id: dict, names: dict) -> dict:
    """An A15 item whose target is the masked draft, built by ``a15.build_item`` itself."""
    parent_id = int(draft['chain_post_ids'][0])
    parent = posts_by_id[('G1', parent_id)]
    agent_code = posts_by_id[('G1', int(ai_post_id))]['author_code']
    chain = draft_chain(posts_by_id, 'G1', parent_id)
    if [int(r['post_id']) for r in chain] != draft['chain_post_ids']:
        raise ValueError(f'{draft["key"]}: release chain differs from the host chain')
    letters = speaker_letters(chain, parent['author_code'], agent_code)
    if {str(r['post_id']): letters[str(r['author_code']).upper()] for r in chain} != draft['letters_by_post'] \
            or letters[str(agent_code).upper()] != draft['target_letter']:
        raise ValueError(f'{draft["key"]}: speaker letters differ between host and release')
    synthetic = dict(post_id=SYNTHETIC_ID, comment_to=parent_id, author_code=agent_code, text=draft['text'])
    view = dict(posts_by_id)
    view[('G1', SYNTHETIC_ID)] = synthetic
    pair = pd.Series(dict(group='G1', parent_content_id=SYNTHETIC_ID, author_code=parent['author_code'],
                          pair_id=item_id))
    item = a15.build_item(pair, view, names)
    if item['structural'] != 'answers_participant' or item['target']['speaker'] != draft['target_letter']:
        raise ValueError(f'{draft["key"]}: unexpected item structure')
    return item


def blind_plan(keys: list[str], seed: int = SEED, batch_size: int = a15.BATCH_SIZE,
               selfcheck_share: float = a15.SELFCHECK_SHARE) -> tuple[dict, dict]:
    """Shuffle drafts across arms and contexts, assign neutral item ids, and batch as A15."""
    rng = random.Random(seed)
    order = sorted(keys)
    rng.shuffle(order)
    key_of = {f'd{i:04d}': key for i, key in enumerate(order)}
    ids = list(key_of)
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    check = rng.sample(ids, round(selfcheck_share * len(ids)))
    rng.shuffle(check)
    plan = dict(main=batches, selfcheck=[check[i:i + batch_size] for i in range(0, len(check), batch_size)])
    return key_of, plan


def seed_outputs() -> None:
    """Copy the deposited outputs of the private-host ``frame`` stage into this run when absent."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('frame.csv', 'drafts_masked.jsonl', 'frame_summary.json'):
        if not (OUT / name).exists() and (DEPOSITED / name).exists():
            shutil.copyfile(DEPOSITED / name, OUT / name)
            log(f'copied {name} from the Zenodo record ({DEPOSITED.name}/)')


def stage_items(args) -> None:
    seed_outputs()
    frame = pd.read_csv(OUT / 'frame.csv')
    drafts = [json.loads(line) for line in (OUT / 'drafts_masked.jsonl').read_text().splitlines()]
    posts_by_id, names = release_tables()
    ai_post = frame.set_index('key').ai_post_id
    key_of, plan = blind_plan([d['key'] for d in drafts])
    by_key = {d['key']: d for d in drafts}
    items = [draft_item(item_id, by_key[key], int(ai_post[key]), posts_by_id, names) for item_id, key in key_of.items()]
    blob = json.dumps(items, ensure_ascii=False)
    if any(any(marker_echo(i['target']['text']).values()) for i in items) or re.search(r'\b(on|off)\|\d', blob):
        raise ValueError('arm information or marker text in the items')
    if a15.CODE_PATTERN.search(blob):
        raise ValueError('account code in the items')
    with (OUT / 'items.jsonl').open('w') as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    pd.DataFrame({'item_id': list(key_of), 'key': list(key_of.values())}).to_csv(OUT / 'item_keys.csv', index=False)
    (OUT / 'schema.json').write_text(json.dumps(a15.schema(), indent=1) + '\n')
    (OUT / 'batches.json').write_text(json.dumps(plan, indent=1) + '\n')
    clipped = sum(i['target']['text'].endswith(a15.CLIP_MARK) for i in items)
    log(f'items: {len(items)} drafts -> {len(plan["main"])} main batches, {sum(map(len, plan["selfcheck"]))} '
        f'self-check items in {len(plan["selfcheck"])} batches (seed {SEED}); clipped targets {clipped}')


# --------------------------------------------------------------------------- annotate stage (Codex CLI)

def run_batch(pass_name: str, index: int, items: list[dict]) -> dict:
    folder = OUT / 'annotation' / pass_name
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / f'batch_{index:03d}.result.json'
    ids = [i['item_id'] for i in items]
    if done.exists():
        return dict(batch=index, status='cached')
    prompt = a15.prompt_for(items)
    (folder / f'batch_{index:03d}.prompt.txt').write_text(prompt)
    workdir = OUT / 'codex_empty_workdir'
    workdir.mkdir(exist_ok=True)
    for attempt in range(1, a15.MAX_ATTEMPTS + 1):
        raw = folder / f'batch_{index:03d}.attempt{attempt}.raw.json'
        stderr = folder / f'batch_{index:03d}.attempt{attempt}.stderr.txt'
        command = ['codex', 'exec', '-m', a15.MODEL, '-c', f'model_reasoning_effort="{a15.REASONING}"', '-s',
                   'read-only', '--ephemeral', '--skip-git-repo-check', '-C', str(workdir), '--output-schema',
                   str(OUT / 'schema.json'), '-o', str(raw), '-']
        started = time.monotonic()
        with stderr.open('w') as err:
            proc = subprocess.run(command, input=prompt, text=True, stdout=err, stderr=subprocess.STDOUT, timeout=1800)
        elapsed = time.monotonic() - started
        try:
            if proc.returncode != 0:
                raise ValueError(f'codex exit {proc.returncode}')
            rows = a15.validate(json.loads(raw.read_text()), ids)
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
    log(f'annotate {args.pass_name}: {len(jobs)} batches, {a15.MAX_PARALLEL} parallel, model {a15.MODEL} '
        f'({a15.REASONING})')
    with ThreadPoolExecutor(max_workers=a15.MAX_PARALLEL) as pool:
        outcomes = list(pool.map(lambda job: run_batch(args.pass_name, *job), jobs))
    failed = [o['batch'] for o in outcomes if o['status'] == 'failed']
    log(f'annotate {args.pass_name}: {len(outcomes) - len(failed)} batches ok, failed {failed}')


def codex_version() -> str:
    try:
        out = subprocess.run(['codex', '--version'], capture_output=True, text=True).stdout.strip().split()
    except OSError:
        out = []
    return 'codex-cli ' + out[-1] if out else 'codex-cli not installed (deposited labels used)'


def read_pass(name: str, plan: list[list[str]]) -> pd.DataFrame:
    """Validated labels of one pass: this run's batch results, else the deposited labels."""
    if not (OUT / 'annotation' / name).exists() and (DEPOSITED / f'labels_{name}.csv').exists():
        labels = pd.read_csv(DEPOSITED / f'labels_{name}.csv').drop(columns='key', errors='ignore')
        if sorted(labels.item_id) != sorted(item for batch in plan for item in batch):
            raise ValueError(f'Deposited {name} labels do not match the rebuilt batch plan')
        log(f'collect: {name} labels read from {DEPOSITED.name}/labels_{name}.csv (no new annotation)')
        return labels
    rows = []
    for index in range(len(plan)):
        payload = json.loads((OUT / 'annotation' / name / f'batch_{index:03d}.result.json').read_text())
        rows += [dict(r, batch=index, attempt=payload['attempt'], elapsed_seconds=payload['elapsed_seconds'])
                 for r in payload['results']]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- paired estimation

def pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (context, draw) with ``*_on`` and ``*_off`` columns; checks the pairing."""
    if frame.duplicated(['context_id', 'draw', 'arm']).any():
        raise ValueError('duplicate (context, draw, arm)')
    if set(frame.arm) != set(ARMS):
        raise ValueError('arms must be on and off')
    on = frame.loc[frame.arm.eq('on')].set_index(['context_id', 'draw'])
    off = frame.loc[frame.arm.eq('off')].set_index(['context_id', 'draw'])
    if set(on.index) != set(off.index):
        raise ValueError('every (context, draw) needs both arms')
    off = off.loc[on.index]
    if 'seeds_sha256' in frame and not on.seeds_sha256.eq(off.seeds_sha256).all():
        raise ValueError('paired draws must share seeds')
    if not on.stratum.eq(off.stratum).all():
        raise ValueError('stratum differs within a pair')
    wide = on.add_suffix('_on').join(off.add_suffix('_off')).reset_index()
    wide['stratum'] = wide.pop('stratum_on')
    return wide.sort_values(['context_id', 'draw']).reset_index(drop=True)


def arm_arrays(wide: pd.DataFrame, outcome: str, subset: str) -> dict:
    """Per-pair weighted sums for a conditional mean: membership w and w * y in each arm."""
    out = {}
    for arm in ARMS:
        w = wide[f'{subset}_{arm}'].astype(float).to_numpy()
        y = wide[f'{outcome}_{arm}'].astype(float).to_numpy()
        y = np.where(w > 0, y, 0.0)
        if np.isnan(y).any():
            raise ValueError(f'{outcome} missing inside subset {subset}')
        out[f'n_{arm}'], out[f's_{arm}'] = w, w * y
    return out


def diff_stat(s_on, n_on, s_off, n_off):
    with np.errstate(invalid='ignore', divide='ignore'):
        return s_on / n_on - s_off / n_off


def swap_matrix(n_pairs: int, n: int = N_PERM, seed: int = SEED) -> np.ndarray:
    """Random within-pair arm swaps (1 = swap), one row per permutation."""
    return np.random.default_rng(seed).integers(0, 2, size=(n, n_pairs)).astype(float)


def permuted(arrays: dict, swaps: np.ndarray) -> np.ndarray:
    """Statistic under within-pair swaps: swapped pairs exchange their ON and OFF records."""
    res = {}
    for kind in ('s', 'n'):
        on, off = arrays[f'{kind}_on'], arrays[f'{kind}_off']
        res[f'{kind}_on'] = on.sum() + swaps @ (off - on)
        res[f'{kind}_off'] = on.sum() + off.sum() - res[f'{kind}_on']
    return diff_stat(res['s_on'], res['n_on'], res['s_off'], res['n_off'])


def perm_pvalue(observed: float, null: np.ndarray) -> float:
    null = null[~np.isnan(null)]
    if np.isnan(observed) or null.size == 0:
        return float('nan')
    return float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + null.size))


def bootstrap_counts(n_clusters: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Multiplicity of each cluster in each bootstrap replicate (resampling clusters with replacement)."""
    draws = np.random.default_rng(seed).integers(0, n_clusters, size=(n, n_clusters))
    counts = np.zeros((n, n_clusters))
    np.add.at(counts, (np.repeat(np.arange(n), n_clusters), draws.ravel()), 1)
    return counts


def cluster_sums(values: np.ndarray, clusters: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(clusters, weights=values, minlength=n_clusters)


def bootstrapped(arrays: dict, clusters: np.ndarray, counts: np.ndarray) -> np.ndarray:
    k = counts.shape[1]
    agg = {name: counts @ cluster_sums(v, clusters, k) for name, v in arrays.items()}
    return diff_stat(agg['s_on'], agg['n_on'], agg['s_off'], agg['n_off'])


def context_swaps(clusters: np.ndarray, n_clusters: int, n: int = N_PERM, seed: int = SEED + 1) -> np.ndarray:
    """Swaps of whole contexts (all draws of a context together), expanded to pairs."""
    return swap_matrix(n_clusters, n, seed)[:, clusters]


class Engine:
    """Shared bootstrap replicates and permutation swaps for one set of pairs."""

    def __init__(self, wide: pd.DataFrame):
        self.wide = wide.reset_index(drop=True)
        codes, uniques = pd.factorize(self.wide.context_id, sort=True)
        self.clusters, self.n_clusters = codes, len(uniques)
        self.counts = bootstrap_counts(self.n_clusters)
        self.swaps = swap_matrix(len(self.wide))
        self.cswaps = context_swaps(self.clusters, self.n_clusters)

    def estimate(self, outcome: str, subset: str) -> dict:
        arrays = arm_arrays(self.wide, outcome, subset)
        n_on, n_off = arrays['n_on'].sum(), arrays['n_off'].sum()
        on = arrays['s_on'].sum() / n_on if n_on else float('nan')
        off = arrays['s_off'].sum() / n_off if n_off else float('nan')
        diff = on - off
        boot = bootstrapped(arrays, self.clusters, self.counts)
        boot = boot[~np.isnan(boot)]
        se = float(np.std(boot, ddof=1)) if boot.size > 1 else float('nan')
        low, high = (np.quantile(boot, [0.025, 0.975]) if boot.size else (float('nan'),) * 2)
        both = int(((arrays['n_on'] > 0) & (arrays['n_off'] > 0)).sum())
        return dict(on=on, off=off, difference=diff, boot_se=se, ci_low=float(low), ci_high=float(high),
                    perm_p_pair_swap=perm_pvalue(diff, permuted(arrays, self.swaps)),
                    perm_p_context_swap=perm_pvalue(diff, permuted(arrays, self.cswaps)),
                    mdd_80=Z_MDD * se, n_on=int(n_on), n_off=int(n_off), pairs=len(self.wide),
                    pairs_both_in_subset=both, contexts=self.n_clusters, boot_valid=int(boot.size))


def chi_square(on_counts: np.ndarray, off_counts: np.ndarray) -> np.ndarray:
    """Pearson chi-square of 2 x K tables (rows of on_counts / off_counts are tables)."""
    on_counts, off_counts = np.atleast_2d(on_counts), np.atleast_2d(off_counts)
    total = on_counts + off_counts
    keep = total.sum(axis=0) > 0
    on_counts, off_counts, total = on_counts[:, keep], off_counts[:, keep], total[:, keep]
    n_on, n_off = on_counts.sum(1, keepdims=True), off_counts.sum(1, keepdims=True)
    grand = n_on + n_off
    e_on, e_off = total * n_on / grand, total * n_off / grand
    return (((on_counts - e_on) ** 2 / e_on) + ((off_counts - e_off) ** 2 / e_off)).sum(1)


def style_permutation(wide: pd.DataFrame, styles: list[str], swaps: np.ndarray) -> tuple[float, float]:
    on = np.stack([wide.selected_style_on.eq(s).to_numpy(float) for s in styles], 1)
    off = np.stack([wide.selected_style_off.eq(s).to_numpy(float) for s in styles], 1)
    observed = float(chi_square(on.sum(0), off.sum(0))[0])
    perm_on = on.sum(0) + swaps @ (off - on)
    perm_off = on.sum(0) + off.sum(0) - perm_on
    null = chi_square(perm_on, perm_off)
    return observed, float((1 + np.sum(null >= observed - 1e-9)) / (1 + null.size))


def agreement_rows(frame: pd.DataFrame, column: str, counts: np.ndarray, contexts: list[str]) -> list[dict]:
    """Shares of draw pairs with the same value of ``column``.

    ``crn_pair``: ON and OFF of the same draw (shared seeds). ``across_arms_other_draw``: ON draw i
    with OFF draw j != i (different seeds). ``within_on`` and ``within_off``: two draws of one arm.
    """
    kinds = {'crn_pair': [], 'across_arms_other_draw': [], 'within_on': [], 'within_off': []}
    for cid in contexts:
        part = frame.loc[frame.context_id.eq(cid)]
        on = part.loc[part.arm.eq('on')].set_index('draw')[column]
        off = part.loc[part.arm.eq('off')].set_index('draw')[column]
        pairs = {'crn_pair': [(on[d], off[d]) for d in on.index],
                 'across_arms_other_draw': [(on[i], off[j]) for i in on.index for j in off.index if i != j],
                 'within_on': [(on[i], on[j]) for i in on.index for j in on.index if i < j],
                 'within_off': [(off[i], off[j]) for i in off.index for j in off.index if i < j]}
        for kind, values in pairs.items():
            same = [a == b for a, b in values if pd.notna(a) and pd.notna(b)]
            kinds[kind].append((sum(same), len(same)))
    rows = []
    for kind, per in kinds.items():
        num, den = np.array([p[0] for p in per], float), np.array([p[1] for p in per], float)
        reps = (counts @ num) / (counts @ den)
        rows.append(dict(comparison=kind, same=int(num.sum()), denominator=int(den.sum()), share=num.sum() / den.sum(),
                         ci_low=float(np.quantile(reps, 0.025)), ci_high=float(np.quantile(reps, 0.975))))
    return rows


OUTCOMES = [
    # (subset, outcome, label)
    ('all_draws', 'posted', 'share posted'),
    ('all_draws', 'adversarial_style', 'share adversarial style (TROLL, TROLL_SOFT, SHITPOST, QUICK_DISAGREE)'),
    ('with_text', 'detoxify_toxicity', 'mean Detoxify toxicity'),
    ('with_text', 'detox_gt_05', 'share Detoxify > 0.5'),
    ('with_text', 'length_chars', 'mean length (characters)'),
    ('with_text', 'vader_compound', 'mean VADER compound'),
    ('with_text', 'disagree', 'share disagreeing with the participant (A15 codebook)'),
    ('with_text', 'adversarial_style', 'share adversarial style'),
    ('with_text', 'token_echo', 'share echoing the [human] token'),
    ('with_text', 'note_echo', 'share echoing the IDENTITY MARKERS note'),
    ('posted', 'detoxify_toxicity', 'mean Detoxify toxicity'),
    ('posted', 'detox_gt_05', 'share Detoxify > 0.5'),
    ('posted', 'length_chars', 'mean length (characters)'),
    ('posted', 'vader_compound', 'mean VADER compound'),
    ('posted', 'disagree', 'share disagreeing with the participant (A15 codebook)'),
    ('posted', 'adversarial_style', 'share adversarial style'),
]
SUBSET_TEXT = {'all_draws': 'all draws', 'with_text': 'drafts with text', 'posted': 'posted drafts'}


def outcome_frame(frame: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Per-draw outcome and subset columns used by the estimands."""
    f = frame.copy()
    for column in ('would_post', 'has_text', 'token_echo', 'note_echo'):
        f[column] = f[column].astype(str).str.lower().eq('true')
    f['all_draws'] = 1
    f['with_text'] = f.has_text.astype(int)
    f['posted'] = f.would_post.astype(int)
    f['adversarial_style'] = f.selected_style.isin(ADVERSARIAL).astype(int)
    f['detox_gt_05'] = (f.detoxify_toxicity > 0.5).astype(float).where(f.detoxify_toxicity.notna())
    f['not_posted_reason'] = f.not_posted_reason.fillna('posted')
    for reason in sorted(f.not_posted_reason.unique()):
        if reason != 'posted':
            f[f'reason_{reason.strip("_")}'] = f.not_posted_reason.eq(reason).astype(int)
    if labels is not None:
        lab = labels.set_index('key').label.eq('YES').astype(int)
        f['disagree'] = f.key.map(lab).astype(float)
        if f.loc[f.has_text, 'disagree'].isna().any():
            raise ValueError('draft with text but no label')
    for column in ('token_echo', 'note_echo'):
        f[column] = f[column].astype(int)
    return f


def estimand_rows(engine: Engine, frame: pd.DataFrame, scope: str) -> list[dict]:
    reasons = [c for c in frame.columns if c.startswith('reason_')]
    plan = list(OUTCOMES) + [('all_draws', r, f'share not posted: {r[len("reason_"):]}') for r in reasons]
    rows = []
    for subset, outcome, label in plan:
        if outcome not in frame.columns:
            continue
        est = engine.estimate(outcome, subset)
        denominator = (f'{est["n_on"]} ON and {est["n_off"]} OFF {SUBSET_TEXT[subset]} in {est["pairs"]} '
                       f'(context, draw) pairs from {est["contexts"]} contexts; {est["pairs_both_in_subset"]} pairs '
                       f'have both arms in the subset')
        rows.append(dict(run_id=RUN_ID, scope=scope, subset=subset, outcome=outcome, label=label, **est,
                         denominator=denominator))
    return rows


def style_table(engine: Engine, frame: pd.DataFrame) -> pd.DataFrame:
    wide = engine.wide
    styles = sorted(set(frame.selected_style.dropna()))
    rows = []
    for style in styles:
        for arm in ARMS:
            wide[f'style_{style}_{arm}'] = wide[f'selected_style_{arm}'].eq(style).astype(int)
            wide[f'all_draws_{arm}'] = 1
        est = engine.estimate(f'style_{style}', 'all_draws')
        rows.append(dict(run_id=RUN_ID, row_type='style_share', item=style, on_draws=int(round(est['on'] * est['n_on'])),
                         off_draws=int(round(est['off'] * est['n_off'])), on=est['on'], off=est['off'],
                         difference=est['difference'], boot_se=est['boot_se'], ci_low=est['ci_low'],
                         ci_high=est['ci_high'], perm_p_pair_swap=est['perm_p_pair_swap'],
                         perm_p_context_swap=est['perm_p_context_swap'], statistic=float('nan'),
                         denominator=f'{est["n_on"]} ON and {est["n_off"]} OFF draws in {est["pairs"]} pairs from '
                                     f'{est["contexts"]} contexts'))
    chi, p_pair = style_permutation(wide, styles, engine.swaps)
    _, p_ctx = style_permutation(wide, styles, engine.cswaps)
    rows.append(dict(run_id=RUN_ID, row_type='omnibus_chi_square', item='style distribution ON vs OFF',
                     statistic=chi, perm_p_pair_swap=p_pair, perm_p_context_swap=p_ctx,
                     denominator=f'{len(wide)} pairs, {len(styles)} styles, Pearson chi-square of the 2 x '
                                 f'{len(styles)} table; {N_PERM} within-pair (or within-context) arm swaps'))
    contexts = sorted(frame.context_id.unique())
    for column, name in (('selected_style', 'same_style'), ('text_sha256', 'same_final_text'),
                         ('not_posted_reason', 'same_status')):
        for row in agreement_rows(frame, column, engine.counts, contexts):
            rows.append(dict(run_id=RUN_ID, row_type=name, item=row['comparison'], on=row['share'],
                             ci_low=row['ci_low'], ci_high=row['ci_high'], on_draws=row['same'],
                             denominator=f'{row["same"]} of {row["denominator"]} draw pairs ({row["comparison"]}); '
                                         f'bootstrap over {len(contexts)} contexts'))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- collect stage

def stage_collect(args) -> None:
    started = time.monotonic()
    seed_outputs()
    frame = pd.read_csv(OUT / 'frame.csv')
    keys = pd.read_csv(OUT / 'item_keys.csv')
    plan = json.loads((OUT / 'batches.json').read_text())
    main = read_pass('main', plan['main'])
    check = read_pass('selfcheck', plan['selfcheck'])
    if main.item_id.duplicated().any() or set(main.item_id) != set(keys.item_id):
        raise ValueError('main pass must label each draft once')
    labels = main.merge(keys, on='item_id', validate='one_to_one')
    labels.to_csv(OUT / 'labels_main.csv', index=False)
    check.to_csv(OUT / 'labels_selfcheck.csv', index=False)
    f = outcome_frame(frame, labels)
    wide = pair_frame(f)
    engine = Engine(wide)
    estimands = pd.DataFrame(estimand_rows(engine, f, 'all_contexts'))
    styles = style_table(engine, f)
    strata_rows, by_stratum = [], {}
    for name, levels in STRATA.items():
        sub = Engine(wide.loc[wide.stratum.isin(levels)])
        rows = estimand_rows(sub, f, name)
        strata_rows += rows
        same = agreement_rows(f.loc[f.stratum.isin(levels)], 'selected_style', sub.counts,
                              sorted(f.loc[f.stratum.isin(levels), 'context_id'].unique()))[0]
        strata_rows.append(dict(run_id=RUN_ID, scope=name, subset='all_draws', outcome='same_style_crn_pair',
                                label='share of (context, draw) pairs with the same style in both arms',
                                on=same['share'], ci_low=same['ci_low'], ci_high=same['ci_high'],
                                pairs=same['denominator'], denominator=f'{same["same"]} of {same["denominator"]} pairs'))
        by_stratum[name] = (sub, rows)
    (high, high_rows), (low, low_rows) = by_stratum['high_moderately_extremely'], by_stratum['low_absolutely_no_no']
    for hr, lr in zip(high_rows, low_rows):
        hb = bootstrapped(arm_arrays(high.wide, hr['outcome'], hr['subset']), high.clusters, high.counts)
        lb = bootstrapped(arm_arrays(low.wide, lr['outcome'], lr['subset']), low.clusters, low.counts)
        d = hb - lb
        d = d[~np.isnan(d)]
        strata_rows.append(dict(run_id=RUN_ID, scope='high_minus_low', subset=hr['subset'], outcome=hr['outcome'],
                                label=hr['label'], difference=hr['difference'] - lr['difference'],
                                boot_se=float(np.std(d, ddof=1)), ci_low=float(np.quantile(d, 0.025)),
                                ci_high=float(np.quantile(d, 0.975)), mdd_80=Z_MDD * float(np.std(d, ddof=1)),
                                pairs=hr['pairs'] + lr['pairs'], contexts=hr['contexts'] + lr['contexts'],
                                boot_valid=int(d.size),
                                denominator=f'difference of the stratum differences; independent cluster bootstraps '
                                            f'within strata ({hr["contexts"]} and {lr["contexts"]} contexts)'))
    strata = pd.DataFrame(strata_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = {name: TABLES / f'chb_prereview_a20_marker_{name}.csv' for name in ('estimands', 'styles', 'strata')}
    estimands.to_csv(paths['estimands'], index=False)
    styles.to_csv(paths['styles'], index=False)
    strata.to_csv(paths['strata'], index=False)
    both = check.merge(main, on='item_id', suffixes=('_selfcheck', '_main'), validate='one_to_one')
    agree = float((both.label_selfcheck == both.label_main).mean())
    kappa = a15.cohen_kappa(both.label_selfcheck, both.label_main)
    lab = f.set_index('key')
    same_text = wide.loc[wide.text_sha256_on.notna() & wide.text_sha256_on.eq(wide.text_sha256_off)]
    same_text_label = float((lab.loc[same_text.key_on, 'disagree'].to_numpy()
                             == lab.loc[same_text.key_off, 'disagree'].to_numpy()).mean()) if len(same_text) else None
    attempts = pd.concat([main.assign(pass_name='main'), check.assign(pass_name='selfcheck')])
    per_batch = attempts.groupby(['pass_name', 'batch']).agg(attempt=('attempt', 'first'),
                                                              seconds=('elapsed_seconds', 'first'))
    host = json.loads((OUT / 'frame_summary.json').read_text())
    record = json.loads((RUN / 'run.json').read_text()) if (RUN / 'run.json').exists() else {}
    analysis_log = (RUN / 'analysis.log').read_text().splitlines() if (RUN / 'analysis.log').exists() else []
    annotate_stamps = [line.split(' ', 1)[0] for line in analysis_log if ' annotate ' in line]
    record['status'] = (f'full run done ({host["generation_run"]["run_log_first"]} to '
                        f'{host["generation_run"]["run_log_last"]}, {host["draws"]} draws, no errors); paired analysis '
                        f'and blind disagreement coding done ({dt.date.today().isoformat()})')
    record['open_items'] = ['human validation of the disagreement labels (none; GPT-5.6 Sol self-consistency only)']
    record['analysis'] = dict(
        script=str(Path(__file__).relative_to(ROOT.parent)), script_sha256=sha256(Path(__file__)),
        test='tests/test_chb_prereview_a20_analysis.py',
        test_sha256=sha256(ROOT / 'tests/test_chb_prereview_a20_analysis.py'),
        commands=['private host: cd WORK/a20_marker && nice -n 10 env OMP_NUM_THREADS=1 '
                  'python analysis/code/chb_prereview_a20_analysis.py frame '
                  '--src full_20261111 --dest analysis',
                  'local: copy WORK/a20_marker/analysis/{frame.csv,drafts_masked.jsonl,frame_summary.json} '
                  'runs/chb_prereview_20261110/a20_marker/outputs/',
                  'python -m scripts.diagnostics.chb_prereview_a20_analysis items',
                  'python -m scripts.diagnostics.chb_prereview_a20_analysis annotate --pass main',
                  'python -m scripts.diagnostics.chb_prereview_a20_analysis annotate --pass selfcheck',
                  'python -m scripts.diagnostics.chb_prereview_a20_analysis collect'],
        host_frame=host,
        inputs={**{f'private host:{k}': v for k, v in host['inputs'].items()},
                'private host:full_20261111/contexts (sha256 of the 60 file hashes)': host['contexts_sha256'],
                **{str(p.relative_to(ROOT.parent)): sha256(p) for p in (POSTS, ACCOUNTS)},
                **{f'outputs/{n}': sha256(OUT / n) for n in ('frame.csv', 'drafts_masked.jsonl', 'items.jsonl',
                                                              'item_keys.csv', 'batches.json', 'schema.json')},
                'scripts/diagnostics/chb_prereview_a15_parent_disagreement.py':
                    sha256(ROOT / 'scripts/diagnostics/chb_prereview_a15_parent_disagreement.py')},
        local_outputs='runs/chb_prereview_20261110/a20_marker/outputs/ (gitignored): frame.csv, drafts_masked.jsonl, '
                       'frame_summary.json, items.jsonl, item_keys.csv, batches.json, schema.json, annotation/, '
                       'labels_main.csv, labels_selfcheck.csv',
        marker_echo=dict(token_echo_by_arm=host['token_echo_by_arm'], note_echo_by_arm=host['note_echo_by_arm'],
                         drafts_with_text=host['drafts_with_text'],
                         drafts_changed_by_strip=host['drafts_changed_by_marker_strip']),
        annotation=dict(codebook='A15 CODEBOOK, schema and validate() imported from '
                                 'scripts/diagnostics/chb_prereview_a15_parent_disagreement.py; item = A15 build_item '
                                 'with the masked draft as target; context from the public release',
                        tool=codex_version(), model=a15.MODEL, reasoning_effort=a15.REASONING,
                        sandbox='read-only', batch_size=a15.BATCH_SIZE, max_parallel=a15.MAX_PARALLEL,
                        blinding=f'drafts shuffled across arms and contexts (random.Random({SEED})), neutral item ids '
                                 'd0000.., arm and context never shown',
                        main_batches=len(plan['main']), selfcheck_batches=len(plan['selfcheck']),
                        items=len(keys), selfcheck_items=len(both),
                        attempts_distribution={str(k): int(v) for k, v in per_batch.attempt.value_counts().items()},
                        batch_seconds=dict(total=round(float(per_batch.seconds.sum()), 1),
                                           median=round(float(per_batch.seconds.median()), 1),
                                           max=round(float(per_batch.seconds.max()), 1)),
                        selfcheck_agreement=agree, selfcheck_cohen_kappa=kappa,
                        identical_text_pairs=int(len(same_text)), identical_text_pairs_label_agreement=same_text_label,
                        yes_main=int(main.label.eq('YES').sum()), human_validation='none'),
        uncertainty=dict(bootstrap=f'cluster bootstrap over contexts, {N_BOOT} replicates, seed {SEED}, percentile 95% '
                                   'interval; strata resampled within stratum',
                         permutation=f'{N_PERM} random within-pair arm swaps (seed {SEED}); context-level swaps '
                                     f'(seed {SEED + 1}) as a check; two-sided p = (1 + #|T*| >= |T|) / (1 + B)',
                         mdd='minimum detectable difference = (1.96 + 0.84) x bootstrap SE (alpha 0.05, power 0.80)'),
        tables=[str(p.relative_to(ROOT.parent)) for p in paths.values()],
        headline={f'{r.subset}:{r.outcome}': dict(on=r.on, off=r.off, on_minus_off=r.difference,
                                                   ci95=[r.ci_low, r.ci_high], perm_p_pair_swap=r.perm_p_pair_swap,
                                                   mdd_80=r.mdd_80, n_on=r.n_on, n_off=r.n_off)
                  for r in estimands.itertuples() if r.outcome in ('disagree', 'detoxify_toxicity', 'detox_gt_05',
                                                                   'adversarial_style', 'posted')}
        | {'same_style_crn_pairs': styles.loc[styles.row_type.eq('same_style') & styles['item'].eq('crn_pair'),
                                              'denominator'].iloc[0],
           'style_omnibus_perm_p_pair_swap': float(styles.loc[styles.row_type.eq('omnibus_chi_square'),
                                                              'perm_p_pair_swap'].iloc[0])},
        timings=dict(generation_run=host['generation_run'], host_frame_seconds=host['seconds'],
                     annotation_first_log=annotate_stamps[0] if annotate_stamps else None,
                     annotation_last_log=annotate_stamps[-1] if annotate_stamps else None,
                     collect_seconds=round(time.monotonic() - started, 1),
                     log=str((RUN / 'analysis.log').relative_to(ROOT.parent))),
        collected=dt.datetime.now().astimezone().isoformat(timespec='seconds'),
    )
    (RUN / 'run.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    log(f'collect: wrote {", ".join(record["analysis"]["tables"])}; self-check agreement {agree:.3f}, kappa {kappa:.3f}')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    frame = sub.add_parser('frame')
    frame.add_argument('--src', required=True)
    frame.add_argument('--dest', required=True)
    sub.add_parser('items')
    annotate = sub.add_parser('annotate')
    annotate.add_argument('--pass', dest='pass_name', choices=['main', 'selfcheck'], required=True)
    annotate.add_argument('--batches', type=int, nargs='*')
    sub.add_parser('collect')
    args = parser.parse_args(argv)
    {'frame': stage_frame, 'items': stage_items, 'annotate': stage_annotate, 'collect': stage_collect}[args.stage](args)


if __name__ == '__main__':
    main()
