"""A20: paired regeneration of G1 AI replies with and without the "[human]" marker.

Runs on the private host only: it needs the experiment database dumps, the archived
YSocial client and a local Ollama server with the generator model, none of which are in
the public record. It is published to document the procedure; its aggregate outputs and
the masked drafts are deposited in the Zenodo record
(``08_analysis_ready/further_checks/a20_marker/``) and analysed by
``chb_prereview_a20_analysis.py``. For a stratified sample of admitted G1 AI comments that reply
directly to a participant, the script rebuilds the server responses that the archived
YSocial client (``src_run``) saw when it wrote the comment, then calls the client's own
``Agent.comment()`` twice per draw: once with ``show_user_type_labels`` True (G1, marker
ON) and once False (G2 behaviour, marker OFF). Everything else is held fixed, and the
two arms of a draw share the Ollama seeds, the style-list shuffle seed and the
adversarial-override seed (common random numbers).

The client code is imported from a private copy of ``src_run`` under the work
directory. All HTTP traffic to the YSocial server is answered offline from the
experiment database dumps (``OfflineServer``); the only network peer is the local Ollama.
The final ``/comment`` POST is intercepted, so nothing is written anywhere.

Outputs under ``--out`` (private host only; they contain participant text and AI text):
``build.json``, ``contexts/cNN.json``, ``draws.jsonl`` (append-only, one line per
context, arm and draw), ``pilot/`` (prompt check and saved prompts) and
``summary_by_arm.csv`` (aggregates without text).

Usage on the private host::

    A20_PRIVATE_REPO=... A20_SRC_RUN=... A20_WORK=... nice -n 10 python chb_prereview_a20_marker.py \
        --out OUT --contexts 60 --draws 4 --resume
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

RUN_ID = 'chb_prereview_20261110_a20_marker'
SEED = 20261110
ARMS = ('on', 'off')
DISPOSITIONS = ('absolutely no', 'no', 'moderately', 'extremely')
G1_FOLDER = '75c440fb_c89d_4d17_abbb_a70b935de62a'
MODEL = 'ikiru/Dolphin-Mistral-24B-Venice-Edition:latest'
OLLAMA = 'http://127.0.0.1:11434'
OFFLINE_BASE = 'http://a20-offline.invalid/'
OFFLINE_EMAIL = 'offline@a20.invalid'
MAX_THREAD_READING = 15  # agents.max_length_thread_reading in the G1 client config
POLITE_GPU_LIMIT_MIB = 8 * 1024
# Copied from base_agent.py:10587-10590 to verify the arms, never to build prompts.
MARKER_NOTE = ('IDENTITY MARKERS:\n'
               'Some entries may include [human], meaning the account is a human participant.\n'
               'Entries without this marker have no identity marker.')
MARKER_TOKEN = ' [human]'
# Order of the post-generation filters in comment() (base_agent.py:13355-13420).
FILTER_ORDER = ('_looks_like_verbatim_reply_copy', '_reply_targets_wrong_user', '_has_recent_identical_comment',
                '_is_thread_duplicate', '_looks_repetitive_comment')

# Private inputs, not in the public record: the study's analysis repository with the raw
# snapshots and database dumps (A20_PRIVATE_REPO), the archived YSocial client source the
# experiment ran (A20_SRC_RUN) and a work directory (A20_WORK).
REPO = Path(os.environ.get('A20_PRIVATE_REPO', 'private/ysocial-experiment'))
SRC_RUN = Path(os.environ.get('A20_SRC_RUN', 'private/src_run'))
WORK = Path(os.environ.get('A20_WORK', 'private/a20_marker'))
SNAPSHOTS = REPO / 'data/raw/snapshots'
FULLSUITE = REPO / 'data/raw/datasets/ysocial_grupa1_grupa2_clean21_fullsuite_ready_2026-06-10T12-02-34_0200'
FINAL_SNAPSHOT = FULLSUITE / 'data/raw/datasets/ysocial_grupa1_grupa2_snapshot_2026-06-10T09-44-59_0200'
FINAL_DUMP = FINAL_SNAPSHOT / f'db_dumps/experiments_{G1_FOLDER}.dump'
G1_EXP_DIR = SNAPSHOTS / f'ysocial_grupa1_grupa2_snapshot_2026-06-09T08-47-44_0200/artifacts/y_web_experiments/{G1_FOLDER}'
SNAPSHOT_CONTENTS = REPO / 'data/processed/ysocial_grupa1_grupa2_snapshot_2026-06-10T09-44-59_0200/snapshot_contents.parquet'
DETOX_SCORES = REPO / 'outputs/tables/primary_analysis_2026-06-10T09-44-59_0200_valid21/08b_detoxify_scores.csv'


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested).
# ---------------------------------------------------------------------------

def allocate(counts: dict, n: int, order=DISPOSITIONS) -> dict:
    """Proportional allocation by largest remainder; ties go to the earlier stratum."""
    total = sum(int(counts.get(k, 0)) for k in order)
    if n > total:
        raise ValueError(f'cannot sample {n} from {total}')
    quotas = {k: n * int(counts.get(k, 0)) / total for k in order}
    out = {k: int(math.floor(q)) for k, q in quotas.items()}
    remainder = n - sum(out.values())
    ranked = sorted(order, key=lambda k: (-(quotas[k] - out[k]), order.index(k)))
    for k in ranked[:remainder]:
        out[k] += 1
    return out


def stratified_sample(rows, n: int, seed: int = SEED, key='content_id', stratum='disposition'):
    """Sample n rows proportionally by stratum, then shuffle the sample order.

    ``rows`` is a list of dicts. Returns a list of dicts with ``context_id`` (c00, c01, ...)
    in the order the contexts will be run; a pilot of k contexts takes the first k.
    """
    import numpy as np
    rows = sorted(rows, key=lambda r: int(r[key]))
    counts = {}
    for r in rows:
        counts[r[stratum]] = counts.get(r[stratum], 0) + 1
    alloc = allocate(counts, n)
    rng = np.random.default_rng(seed)
    chosen = []
    for level in DISPOSITIONS:
        pool = [r for r in rows if r[stratum] == level]
        if alloc[level]:
            idx = sorted(rng.choice(len(pool), size=alloc[level], replace=False).tolist())
            chosen.extend(pool[i] for i in idx)
    order = rng.permutation(len(chosen)).tolist()
    out = []
    for i, j in enumerate(order):
        row = dict(chosen[j])
        row['context_id'] = f'c{i:02d}'
        out.append(row)
    return out, alloc


def derive_seed(*parts) -> int:
    digest = hashlib.sha256('|'.join(str(p) for p in parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big') & 0x7FFFFFFF


def draw_seeds(base_seed: int, context_id: str, draw: int) -> dict:
    """Seeds for one draw. The arm is deliberately not an input: both arms share them."""
    return {name: derive_seed(base_seed, context_id, int(draw), name)
            for name in ('python', 'numpy', 'style_llm', 'shuffle', 'override')}


def record_key(context_id: str, arm: str, draw: int) -> str:
    return f'{context_id}|{arm}|{int(draw)}'


def completed_keys(path: Path) -> set:
    """Keys already recorded without error; a truncated last line is ignored."""
    done = set()
    if not Path(path).exists():
        return done
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('status') != 'error':
                done.add(rec['key'])
    return done


def plan_jobs(context_ids, draws: int, done=frozenset()):
    """(context_id, draw, arm) jobs, arms adjacent so pairs complete together."""
    return [(c, d, arm) for c in context_ids for d in range(int(draws)) for arm in ARMS
            if record_key(c, arm, d) not in done]


def strip_marker(text: str) -> str:
    """Remove the identity note and per-line markers from a marker-ON prompt."""
    out = text.replace(MARKER_NOTE + '\n\n', '').replace('\n\n' + MARKER_NOTE, '').replace(MARKER_NOTE, '')
    return out.replace(MARKER_TOKEN, '')


def compare_prompts(on: str, off: str) -> dict:
    """Marker check for one paired prompt."""
    import difflib
    stripped = strip_marker(on)
    res = {
        'on_has_marker_token': MARKER_TOKEN in on,
        'on_has_note': MARKER_NOTE in on,
        'off_has_marker_token': MARKER_TOKEN in off or '[human]' in off,
        'off_has_note': 'IDENTITY MARKERS' in off,
        'identical_after_strip': stripped == off,
        'on_chars': len(on), 'off_chars': len(off),
    }
    if not res['identical_after_strip']:
        diff = [l for l in difflib.unified_diff(stripped.splitlines(), off.splitlines(), lineterm='', n=0)
                if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        res['differing_lines'] = len(diff)
        res['difference'] = classify_difference(stripped, off)
    return res


def classify_difference(stripped_on: str, off: str) -> str:
    """'truncation_only' when the sole difference is ON text cut earlier and ended with '...'.

    The client truncates some blocks to a character budget (for example the 900-character
    LOCAL BRANCH CONTEXT in the style prompt); the marker uses part of that budget.
    """
    import difflib
    ops = [op for op in difflib.SequenceMatcher(None, stripped_on, off, autojunk=False).get_opcodes()
           if op[0] != 'equal']
    if len(ops) == 1:
        tag, i1, i2, j1, j2 = ops[0]
        removed, added = stripped_on[i1:i2], off[j1:j2]
        if removed.strip(' .') == '' and '...' in stripped_on[max(0, i1 - 3):i2 + 3] and len(added) > len(removed):
            return 'truncation_only'
    if 'PROACTIVE AFFECT' in stripped_on + off:
        return 'other_with_proactive_block'
    return 'other'


# ---------------------------------------------------------------------------
# PostgreSQL custom-format dump reader (the private host has no pg_restore).
# ---------------------------------------------------------------------------

class PgDump:
    """Minimal reader for pg_dump --format=custom archives (v1.14-1.16, gzip data)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = self.path.open('rb')
        if self.f.read(5) != b'PGDMP':
            raise ValueError(f'not a custom-format dump: {path}')
        self.version = tuple(self.f.read(3))
        self.intsize, self.offsize, self.fmt = self.f.read(3)
        if self.version >= (1, 15, 0):
            self.compression = self.f.read(1)[0]
        else:
            self.compression = self._int()
        sec, minute, hour, mday, mon, year, _isdst = (self._int() for _ in range(7))
        from zoneinfo import ZoneInfo
        self.created = datetime(year + 1900, mon + 1, mday, hour, minute, sec, tzinfo=ZoneInfo('Europe/Belgrade'))
        self.dbname, self.remote_version, self.pg_version = self._str(), self._str(), self._str()
        self.toc = self._toc()

    def _int(self):
        sign = self.f.read(1)[0]
        value = int.from_bytes(self.f.read(self.intsize), 'little')
        return -value if sign else value

    def _str(self):
        n = self._int()
        return None if n < 0 else self.f.read(n).decode('utf-8')

    def _toc(self):
        entries = []
        for _ in range(self._int()):
            e = {'dump_id': self._int(), 'had_dumper': self._int(), 'tableoid': self._str(), 'oid': self._str(),
                 'tag': self._str(), 'desc': self._str(), 'section': self._int(), 'defn': self._str(),
                 'drop': self._str(), 'copy': self._str(), 'namespace': self._str(), 'tablespace': self._str()}
            if self.version >= (1, 14, 0):
                e['tableam'] = self._str()
            if self.version >= (1, 16, 0):
                e['relkind'] = self._int()
            e['owner'] = self._str()
            if self.version < (1, 13, 0):
                self._str()
            while self._str() is not None:
                pass
            flag = self.f.read(1)[0]
            e['offset'] = (flag, int.from_bytes(self.f.read(self.offsize), 'little'))
            entries.append(e)
        return entries

    def table(self, tag: str) -> list:
        entry = [e for e in self.toc if e['desc'] == 'TABLE DATA' and e['tag'] == tag]
        if not entry:
            raise KeyError(f'{tag} not in {self.path}')
        entry = entry[0]
        stmt = entry['copy']
        cols = [c.strip().strip('"') for c in stmt[stmt.index('(') + 1:stmt.index(')')].split(',')]
        flag, offset = entry['offset']
        if flag != 2:
            raise ValueError(f'{tag}: data offset not set')
        self.f.seek(offset)
        block_type, dump_id = self.f.read(1)[0], self._int()
        if block_type != 1 or dump_id != entry['dump_id']:
            raise ValueError(f'{tag}: bad data block header')
        dec, chunks = zlib.decompressobj(), []
        while True:
            n = self._int()
            if n == 0:
                break
            chunks.append(dec.decompress(self.f.read(n)))
        chunks.append(dec.flush())
        rows = []
        for line in b''.join(chunks).decode('utf-8').split('\n'):
            if not line or line == '\\.':
                continue
            values = [_copy_unescape(v) for v in line.split('\t')]
            if len(values) != len(cols):
                raise ValueError(f'{tag}: {len(values)} fields for {len(cols)} columns')
            rows.append(dict(zip(cols, values)))
        return rows


def _copy_unescape(field: str):
    if field == '\\N':
        return None
    if '\\' not in field:
        return field
    mapping = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', 'b': '\b', 'f': '\f', 'v': '\v'}
    out, i = [], 0
    while i < len(field):
        if field[i] == '\\' and i + 1 < len(field):
            out.append(mapping.get(field[i + 1], field[i + 1]))
            i += 2
        else:
            out.append(field[i])
            i += 1
    return ''.join(out)


def _i(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Server emulation. Transcribed from YServerReddit routes at the run commit
# (content_management.py:86-182 agent-visible text, 2160-2302 get_thread_tree,
# 3439-3560 memory/get_context, 3630-3684 topics and sentiment).
# ---------------------------------------------------------------------------

def _agent_context_text(value, max_chars):
    text_value = re.sub(r'\s+', ' ', str(value or '')).strip()
    text_value = text_value.replace('**', '').replace('*', '').replace('__', '').replace('`', '')
    for phrase in ('Here is a description of the image:', "Here's a description of the image:",
                   'Here’s a description of the image:'):
        text_value = re.sub(re.escape(phrase), '', text_value, flags=re.IGNORECASE)
    text_value = re.sub(r'\s+', ' ', text_value).strip()
    if not text_value:
        return ''
    if len(text_value) > max_chars:
        text_value = text_value[: max_chars - 3].rstrip() + '...'
    return text_value


def agent_visible_text(post: dict, db: dict) -> str:
    text = str(post.get('tweet') or '')
    ip = _i(post.get('image_post_id'))
    if ip is not None:
        image_post = db['image_posts'].get(ip)
        if image_post and image_post.get('description'):
            text = f"[Image: {image_post['description']}] {text}"
    article_id = _i(post.get('news_id'))
    if article_id is not None:
        article = db['articles'].get(article_id)
        if article is not None:
            lines = ['[Shared link]']
            website = db['websites'].get(_i(article.get('website_id')))
            if website is not None:
                source = _agent_context_text(website.get('name'), 80)
                if source:
                    lines.append(f'Source: {source}')
            image = db['images_by_article'].get(article_id)
            title = _agent_context_text(article.get('title'), 180)
            if title:
                lines.append(f'Title: {title}')
            summary = _agent_context_text(article.get('summary'), 500)
            if summary:
                lines.append(f'Summary: {summary}')
            if image is not None:
                image_description = _agent_context_text(image.get('description'), 400)
                if image_description:
                    lines.append(f'Preview image: {image_description}')
            url = _agent_context_text(article.get('link'), 240)
            if url:
                lines.append(f'URL: {url}')
            if len(lines) > 1:
                shared = '\n'.join(lines)
                text = f'{text}\n{shared}' if text else shared
    return text


def reputation_hint_map(karma: dict, activity: dict) -> dict:
    """Server _build_reputation_hint_map with karma from user_mgmt."""
    established, hint = [], {}
    for uid, score in karma.items():
        authored = int(activity.get(uid, 0))
        is_established = authored >= 3
        hint[uid] = {'reputation_hint': 'new or low-activity', 'reputation_established': bool(is_established)}
        if is_established:
            established.append((uid, int(score or 0)))
    established.sort(key=lambda row: (-row[1], row[0]))
    total = len(established)
    for idx, (uid, _k) in enumerate(established):
        pct = ((idx + 0.5) / total) * 100.0 if total else 50.0
        if pct <= 10.0:
            label = 'top 10%'
        elif pct <= 25.0:
            label = 'top 25%'
        elif pct >= 90.0:
            label = 'bottom 10%'
        elif pct >= 75.0:
            label = 'bottom 25%'
        else:
            label = 'middle 50%'
        hint[uid]['reputation_hint'] = label
    return hint


class CommentCaptured(Exception):
    def __init__(self, payload):
        super().__init__('comment POST intercepted')
        self.payload = payload


class OfflineServer:
    """Answers the client's HTTP calls for one context from reconstructed state."""

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.calls = []
        self.decisions = []
        self.unexpected = []
        self.posts = {int(p['post_id']): p for p in ctx['thread_posts']}

    def _response(self, payload, status=200):
        import requests
        resp = requests.models.Response()
        resp.status_code = status
        resp._content = json.dumps(payload).encode('utf-8')
        resp.headers['Content-Type'] = 'application/json'
        return resp

    def handle(self, method, url, data=None, **_kw):
        path = url.split('a20-offline.invalid', 1)[-1].strip('/')
        try:
            body = json.loads(data) if isinstance(data, (str, bytes)) and data else {}
        except json.JSONDecodeError:
            body = {}
        self.calls.append(path)
        ctx = self.ctx
        persona = ctx['persona']
        if path == 'comment':
            raise CommentCaptured(body)
        if path == 'log/agent_decision':
            self.decisions.append(body)
            return self._response({'status': 200})
        if path == 'user_exists':
            return self._response({'status': 200, 'id': persona['id']})
        if path == 'get_user':
            return self._response({**persona, 'email': OFFLINE_EMAIL, 'password': ''})
        if path in ('get_user_interests', 'set_user_interests'):
            return self._response([])
        if path == 'update_user':
            return self._response({'status': 200})
        if path == 'current_time':
            return self._response({'id': ctx['tid'], 'day': ctx['day'], 'round': ctx['hour']})
        if path == 'get_thread_tree':
            return self._response(self._thread_tree(body))
        if path == 'get_thread_root':
            post = self.posts.get(_i(body.get('post_id')))
            return self._response(ctx['thread_root_id'] if post else {'status': 404})
        if path == 'get_username_from_post':
            post = self.posts.get(_i(body.get('post_id')))
            if post is None:
                return self._response({'status': 404})
            return self._response({'status': 200, 'user_id': post['user_id'], 'username': post['username']})
        if path == 'get_user_from_post':
            post = self.posts.get(_i(body.get('post_id')))
            return self._response(post['username'] if post else None)
        if path == 'get_user_id':
            ids = {p['username']: p['user_id'] for p in ctx['thread_posts']}
            return self._response({'id': ids.get(body.get('username'))})
        if path == 'get_post':
            post = self.posts.get(_i(body.get('post_id')))
            return self._response(post['tweet'] if post else None)
        if path == 'get_post_topics_name':
            return self._response(ctx['post_topics'].get(str(_i(body.get('post_id'))), []))
        if path == 'get_sentiment':
            return self._response(self._sentiment(body))
        if path == 'memory/get_context':
            return self._response(self._memory_context(body))
        if path == 'memory/search':
            # Design: no cross-thread memory search in either arm.
            return self._response({'status': 503, 'error': 'memory search disabled offline'})
        self.unexpected.append(path)
        return self._response({'status': 404, 'error': 'not emulated'}, status=404)

    def _thread_tree(self, body):
        ctx = self.ctx
        limit = _i(body.get('limit'), 200) or 200
        limit = 200 if limit <= 0 else min(limit, 2000)
        include_rep = bool(body.get('include_reputation', False))
        include_labels = bool(body.get('include_user_type_labels', False))
        if _i(body.get('post_id')) not in self.posts:
            return {'status': 404, 'error': 'Post not found'}
        hints = reputation_hint_map({int(k): v for k, v in ctx['karma'].items()},
                                    {int(k): v for k, v in ctx['activity'].items()}) if include_rep else {}
        out = []
        for p in sorted(ctx['thread_posts'], key=lambda r: int(r['post_id']))[:limit]:
            is_human = p['user_type'] == 'user'
            uid = int(p['user_id'])
            out.append({
                'post_id': int(p['post_id']), 'comment_to': p['comment_to'], 'user_id': uid,
                'username': p['username'], 'text': p['text'], 'round': p['round'],
                'reaction_count': int(p.get('reaction_count') or 0),
                'identity_hint': ('human' if is_human else None) if include_labels else None,
                'is_human': True if include_labels and is_human else None,
                'reputation_hint': hints.get(uid, {}).get('reputation_hint'),
                'reputation_established': hints.get(uid, {}).get('reputation_established'),
            })
        return {'status': 200, 'thread_root_id': int(ctx['thread_root_id']), 'posts': out}

    def _sentiment(self, body):
        latest = self.ctx['sentiment_latest']
        res = []
        for interest in body.get('interests') or []:
            row = latest.get(interest)
            if row is None:
                continue
            compound = float(row['compound'])
            label = 'positive' if compound > 0.05 else ('negative' if compound < -0.05 else 'neutral')
            res.append({'topic': interest, 'sentiment': label})
        return res

    def _memory_context(self, body):
        ctx = self.ctx
        other = _i(body.get('other_user_id'))
        thread_root = _i(body.get('thread_root_id'))
        limit = _i(body.get('pair_limit'), 5) or 5
        limit = 5 if limit <= 0 else min(limit, 20)
        mem = ctx['memory']
        users = {int(k): v for k, v in ctx['usernames'].items()}
        pair = mem['pair_events'][-limit:] if other == ctx['other_user_id'] else []
        social = mem['social_card'] if other == ctx['other_user_id'] else None
        thread_card = mem['thread_card'] if thread_root == ctx['thread_root_id'] else None
        if social is not None:
            social = {**social, 'other_username': users.get(other)}
        events = [{**ev, 'actor_username': users.get(ev['actor_user_id']),
                   'target_username': users.get(ev['target_user_id'])} for ev in pair]
        return {'status': 200, 'run_id': ctx['memory_run_id'], 'other_username': users.get(other),
                'user_map': {str(k): v for k, v in users.items()}, 'social_card': social,
                'thread_card': thread_card, 'community_digest': mem['digest'], 'recent_pair_events': events}


# ---------------------------------------------------------------------------
# Building the frame and the per-context inputs (private host).
# ---------------------------------------------------------------------------

def load_frame():
    """Admitted G1 AI comments replying directly to a participant (451)."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.data import _valid_contents, build_account_universe
    import pandas as pd
    config = load_config(REPO / 'analysis.yaml', profile='release',
                         run_id='rq2_m2_directional_contact_release_20260903')
    contents = _valid_contents(config).reset_index(drop=True)
    universe = build_account_universe(config)
    types = {(r.group, int(r.account_id)): r.account_type for r in universe.itertuples()}
    contents['account_type'] = [types[(g, int(a))] for g, a in zip(contents.group, contents.author_id)]
    assert contents.account_type.eq(contents.author_type).all()
    parents = contents[['group', 'content_id', 'account_type']].rename(
        columns={'content_id': 'parent_id', 'account_type': 'parent_type'})
    contents = contents.merge(parents, left_on=['group', 'comment_to'], right_on=['group', 'parent_id'], how='left')
    is_reply = contents.comment_to.notna() & contents.comment_to.ne(-1)
    frame = contents.loc[contents.group.eq('G1') & contents.account_type.eq('AI') & is_reply
                         & contents.parent_type.eq('H')].copy()
    snap = pd.read_parquet(SNAPSHOT_CONTENTS)
    snap = snap.loc[snap.group.eq('G1')]
    snap_type = snap[['content_id', 'author_type']].rename(columns={'content_id': 'pid', 'author_type': 'ptype'})
    s = snap.merge(snap_type, left_on='comment_to', right_on='pid', how='left')
    s_ai_h = s.loc[s.author_type.eq('AI') & s.ptype.eq('H')]
    s_valid = set(s_ai_h.loc[s_ai_h.invalid_reason.eq('valid'), 'content_id'].astype(int))
    in_frame = set(frame.content_id.astype(int))
    reasons = s.set_index('content_id').invalid_reason
    valid_days = set(contents.experiment_day.dropna().astype(int))
    outside = s_ai_h.loc[s_ai_h.content_id.isin(sorted(s_valid - in_frame))]
    reconcile = {
        'frame_admitted_cohort': len(in_frame),
        'snapshot_ai_to_participant_all_days': int(len(s_ai_h)),
        'snapshot_invalid_reason_valid': len(s_valid),
        'both': len(in_frame & s_valid),
        'frame_only_by_snapshot_reason': {str(k): int(v) for k, v in
                                          reasons.reindex(sorted(in_frame - s_valid)).value_counts(dropna=False).items()},
        'snapshot_valid_only': len(s_valid - in_frame),
        'snapshot_valid_only_experiment_day_in_21_windows': int(outside.experiment_day.astype('Int64').isin(valid_days).sum()),
        'snapshot_valid_only_parent_in_cohort': int(outside.comment_to.isin(contents.loc[contents.group.eq('G1'), 'content_id']).sum()),
    }
    return frame, reconcile


def load_tables():
    dump = PgDump(FINAL_DUMP)
    t = {name: dump.table(name) for name in (
        'post', 'user_mgmt', 'post_topics', 'interests', 'post_sentiment', 'memory_interaction_events',
        'mentions', 'articles', 'websites', 'images', 'image_posts', 'rounds')}
    t['dump_created'] = dump.created.isoformat()
    t['memory_run_ids'] = sorted({r['run_id'] for r in t['memory_interaction_events']})
    return t


def daily_dumps():
    out = []
    paths = sorted(SNAPSHOTS.glob(f'*/db_dumps/experiments_{G1_FOLDER}.dump')) + [FINAL_DUMP]
    for p in paths:
        d = PgDump(p)
        out.append((d.created, p))
    return sorted(out)


def server_decision_index():
    """Decision records from every archived G1 server log, keyed by (agent, parent, tid)."""
    idx, files = {}, []
    roots = [p for p in SNAPSHOTS.glob(f'*/artifacts/y_web_experiments/{G1_FOLDER}')]
    roots.append(FINAL_SNAPSHOT / f'artifacts/y_web_experiments/{G1_FOLDER}')
    for root in roots:
        files.extend(sorted(root.glob('_server.log*')))
    seen = set()
    for path in files:
        with path.open(errors='replace') as handle:
            for line in handle:
                if '"decision_type"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('decision_type') not in ('comment_style_selection', 'reply_memory_mode',
                                                    'length_enforcement', 'reply_edge_check'):
                    continue
                sig = hashlib.sha1(line.encode('utf-8')).hexdigest()
                if sig in seen:
                    continue
                seen.add(sig)
                key = (_i(rec.get('agent_user_id')), _i(rec.get('post_id')), _i(rec.get('tid')))
                idx.setdefault(key, []).append(rec)
    return idx, len(files)


def card_as_of(state_before, state_after, table, match, round_field, tid):
    """Memory card as the server held it at round ``tid``.

    Cards are upserted after writes and never deleted. If the first dump after the comment
    holds the card with ``round_field`` < tid, the card was not touched since before the
    round, so that state is exact. If that dump has no card, none existed. Otherwise fall
    back to the last dump before the comment (stale by up to a day).
    """
    after = [r for r in (state_after or {}).get(table, []) if match(r)] if state_after is not None else None
    before = [r for r in (state_before or {}).get(table, []) if match(r)] if state_before is not None else None
    if after is not None:
        if not after:
            return None, 'exact_absent_in_next_dump'
        if _i(after[0][round_field], 10**9) < tid:
            return after[0], 'exact_unchanged_in_next_dump'
    if before:
        return before[0], 'approx_earlier_dump'
    if before is not None:
        return None, 'approx_absent_in_earlier_dump'
    return None, 'missing_no_earlier_dump'


def build_contexts(out: Path, n_contexts: int, seed: int, gist_from_log: bool = False) -> dict:
    import pandas as pd
    frame, reconcile = load_frame()
    tables = load_tables()
    posts = {int(p['id']): p for p in tables['post']}
    users = {int(u['id']): u for u in tables['user_mgmt']}
    frame['disposition'] = [users[int(a)]['toxicity'] for a in frame.author_id]
    rows = [{'content_id': int(r.content_id), 'disposition': r.disposition,
             'created_at': r.created_at.isoformat()} for r in frame.itertuples()]
    sample, alloc = stratified_sample(rows, n_contexts, seed)
    dumps = daily_dumps()
    dump_cache = {}
    decisions, n_logs = server_decision_index()
    detox = pd.read_csv(DETOX_SCORES)
    detox = detox.loc[detox.group.eq('G1')].set_index('text_id').detoxify_toxicity
    interests = {int(r['iid']): r['interest'] for r in tables['interests']}
    topics_by_post = {}
    for r in tables['post_topics']:
        topics_by_post.setdefault(int(r['post_id']), []).append(int(r['topic_id']))
    db = {'articles': {int(a['id']): a for a in tables['articles']},
          'websites': {int(w['id']): w for w in tables['websites']},
          'image_posts': {int(i['id']): i for i in tables['image_posts']},
          'images_by_article': {}}
    for img in tables['images']:
        aid = _i(img.get('article_id'))
        if aid is not None and aid not in db['images_by_article']:
            db['images_by_article'][aid] = img
    rounds = {int(r['id']): r for r in tables['rounds']}
    agents_file = json.loads((G1_EXP_DIR / 'RedditExperiment.json').read_text())['agents']
    recsys = {a['name']: (a['rec_sys'], a['frec_sys']) for a in agents_file}
    run_ids = tables['memory_run_ids']
    config = json.loads((G1_EXP_DIR / 'client_G1-Reddit Experiment.json').read_text())
    run_id = config['agents']['memory_run_id']
    events_all = [e for e in tables['memory_interaction_events'] if e['run_id'] == run_id]
    mentions = {(int(m['user_id']), int(m['post_id'])): m for m in tables['mentions']}
    (out / 'contexts').mkdir(parents=True, exist_ok=True)
    fidelity_rows = []
    for row in sample:
        ai_id = row['content_id']
        ai = posts[ai_id]
        agent_id = int(ai['user_id'])
        parent_id = int(ai['comment_to'])
        parent = posts[parent_id]
        root_id = int(parent['thread_id'])
        tid = int(ai['round'])
        created = datetime.fromisoformat(row['created_at'])
        thread = [p for p in tables['post'] if _i(p['thread_id']) == root_id and int(p['id']) < ai_id]
        thread_posts = []
        for p in thread:
            u = users[int(p['user_id'])]
            ct = _i(p.get('comment_to'))
            thread_posts.append({'post_id': int(p['id']), 'comment_to': None if ct in (-1, None) else ct,
                                 'user_id': int(p['user_id']), 'username': u['username'], 'user_type': u['user_type'],
                                 'tweet': p['tweet'], 'text': agent_visible_text(p, db), 'round': _i(p['round']),
                                 'reaction_count': _i(p['reaction_count'], 0)})
        activity = {}
        for p in tables['post']:
            if int(p['id']) < ai_id:
                activity[int(p['user_id'])] = activity.get(int(p['user_id']), 0) + 1
        writing = sum(1 for p in tables['post'] if int(p['user_id']) == agent_id and _i(p['round']) == tid
                      and int(p['id']) < ai_id)
        earlier = [(t, p) for t, p in dumps if t <= created]
        later = [(t, p) for t, p in dumps if t > created]
        daily = earlier[-1] if earlier else None
        nxt = later[0] if later else None
        for item in (daily, nxt):
            if item is not None and item[1] not in dump_cache:
                d = PgDump(item[1])
                dump_cache[item[1]] = {name: d.table(name) for name in (
                    'memory_social_cards', 'memory_thread_cards', 'memory_community_digests', 'user_mgmt')}
        state = dump_cache.get(daily[1]) if daily else None
        state_next = dump_cache.get(nxt[1]) if nxt else None
        other_id = int(parent['user_id'])
        digest = None
        karma_state = state or state_next
        karma = {int(u['id']): _i(u.get('karma'), 0) for u in karma_state['user_mgmt']}
        karma_source = 'earlier_daily_dump' if state is not None else 'next_daily_dump'

        def is_social(sc):
            return sc['run_id'] == run_id and int(sc['agent_user_id']) == agent_id and int(sc['other_user_id']) == other_id

        def is_thread(tc):
            return tc['run_id'] == run_id and int(tc['agent_user_id']) == agent_id and int(tc['thread_root_id']) == root_id

        sc, social_source = card_as_of(state, state_next, 'memory_social_cards', is_social, 'last_updated_round', tid)
        social = None if sc is None else {
            'affinity': _f(sc['affinity']), 'conflict': _f(sc['conflict']), 'humor': _f(sc['humor']),
            'trust': _f(sc['trust']), 'last_relation_label': sc['last_relation_label'],
            'last_round_id': _i(sc['last_round_id']), 'last_thread_root_id': _i(sc['last_thread_root_id']),
            'last_updated_round': _i(sc['last_updated_round']), 'event_count': _i(sc['event_count']),
            'summary_text': sc['summary_text'], 'evidence_tail': sc['evidence_tail_json']}
        tc, thread_source = card_as_of(state, state_next, 'memory_thread_cards', is_thread, 'last_seen_round_id', tid)
        thread_card = None if tc is None else {
            'gist_text': tc['gist_text'], 'my_role': tc['my_role'], 'participants_top': tc['participants_top_json'],
            'entry_points': tc['entry_points_json'], 'last_seen_round_id': _i(tc['last_seen_round_id'])}
        logged_keys = decisions.get((agent_id, parent_id, tid), [])
        logged_gists = [seg[len('this thread: '):] for r in logged_keys if r.get('decision_type') == 'reply_memory_mode'
                        for seg in str(r.get('continuity_text') or '').split(' | ') if seg.startswith('this thread: ')]
        if gist_from_log and thread_card is None and logged_gists:
            # Thread cards keep no history; the original decision log holds the gist the prompt used.
            thread_card = {'gist_text': logged_gists[0], 'my_role': None, 'participants_top': None,
                           'entry_points': None, 'last_seen_round_id': None}
            thread_source = 'logged_continuity'
        if state is not None:
            digests = [g for g in state['memory_community_digests'] if g['run_id'] == run_id]
            if digests:
                g = max(digests, key=lambda r: int(r['id']))
                digest = {'round_id': _i(g['round_id']), 'digest_text': g['digest_text'], 'top_topics': g['top_topics_json'],
                          'norms': g['norms_json'], 'memes': g['memes_json'], 'polarizing_issues': g['polarizing_issues_json']}
        # Events are appended after a write; the event this comment produced bounds the as-of set.
        own = [e for e in events_all if _i(e['actor_user_id']) == agent_id and _i(e['target_post_id']) == parent_id
               and _i(e['round_id']) == tid and e['event_type'] == 'comment']
        if len(own) == 1:
            cutoff = int(own[0]['id'])
            event_rule = 'before_event_logged_for_this_comment'
            keep = [e for e in events_all if int(e['id']) < cutoff]
        else:
            event_rule = f'earlier_rounds_only_{len(own)}_own_events'
            keep = [e for e in events_all if _i(e['round_id'], 10**9) < tid]
        pair = [e for e in keep if {_i(e['actor_user_id']), _i(e['target_user_id'])} == {agent_id, other_id}
                and (_i(e['actor_user_id']), _i(e['target_user_id'])) in ((agent_id, other_id), (other_id, agent_id))]
        pair = sorted(pair, key=lambda e: int(e['id']))[-20:]
        pair_events = [{'round_id': _i(e['round_id']), 'actor_user_id': _i(e['actor_user_id']),
                        'target_user_id': _i(e['target_user_id']), 'event_type': e['event_type'],
                        'relation_label': e['relation_label'], 'tone_label': e['tone_label'],
                        'thread_root_id': _i(e['thread_root_id']), 'target_post_id': _i(e['target_post_id']),
                        'salient_claim': e['salient_claim']} for e in pair]
        topic_names = [interests[t] for t in topics_by_post.get(parent_id, []) if t in interests]
        topic_ids = {t for t in topics_by_post.get(parent_id, [])}
        latest = {}
        for s in tables['post_sentiment']:
            if _i(s['user_id']) == agent_id and _i(s['topic_id']) in topic_ids and int(s['post_id']) < ai_id:
                name = interests.get(_i(s['topic_id']))
                if name and (name not in latest or int(s['id']) > int(latest[name]['id'])):
                    latest[name] = {'id': s['id'], 'compound': s['compound']}
        u = users[agent_id]
        persona = {'id': agent_id, 'username': u['username'], 'leaning': u['leaning'], 'age': int(_f(u['age'], 0)),
                   'user_type': u['user_type'], 'ag': u['ag'], 'ne': u['ne'], 'ex': u['ex'], 'co': u['co'], 'oe': u['oe'],
                   'rec_sys': u['recsys_type'], 'language': u['language'], 'education_level': u['education_level'],
                   'joined_on': _i(u['joined_on']), 'owner': u['owner'], 'round_actions': _i(u['round_actions']),
                   'frec_sys': u['frecsys_type'], 'gender': u['gender'], 'nationality': u['nationality'],
                   'toxicity': u['toxicity'], 'is_page': _i(u['is_page'], 0),
                   'daily_activity_level': _i(u['daily_activity_level'], 1), 'profession': u.get('profession') or ''}
        mention = mentions.get((agent_id, parent_id))
        path = 'mention' if mention is not None and str(mention.get('answered')).lower() in ('1', 't', 'true') else 'browse'
        logged = decisions.get((agent_id, parent_id, tid), [])
        style_logged = [r.get('selected_style') for r in logged if r.get('decision_type') == 'comment_style_selection']
        memory_logged = [r for r in logged if r.get('decision_type') == 'reply_memory_mode']
        rnd = rounds.get(tid, {})
        usernames = {int(k): v['username'] for k, v in users.items()}
        ctx = {
            'context_id': row['context_id'], 'stratum': row['disposition'], 'ai_post_id': ai_id,
            'agent_user_id': agent_id, 'parent_post_id': parent_id, 'other_user_id': other_id,
            'thread_root_id': root_id, 'tid': tid, 'day': _i(rnd.get('day')), 'hour': _i(rnd.get('hour')),
            'created_at': row['created_at'], 'persona': persona,
            'recsys': list(recsys.get(u['username'], ('HotNW', 'PreferentialAttachment'))),
            'thread_posts': thread_posts, 'karma': karma, 'activity': activity,
            'usernames': usernames, 'post_topics': {str(parent_id): topic_names}, 'sentiment_latest': latest,
            'memory_run_id': run_id,
            'memory': {'social_card': social, 'thread_card': thread_card, 'digest': digest, 'pair_events': pair_events,
                       'pair_event_rule': event_rule, 'social_card_source': social_source,
                       'thread_card_source': thread_source},
            'daily_state': {'dump': str(daily[1]) if daily else None, 'created': daily[0].isoformat() if daily else None,
                            'staleness_hours': round((created - daily[0]).total_seconds() / 3600, 2) if daily else None,
                            'next_dump': str(nxt[1]) if nxt else None, 'karma_source': karma_source},
            'writing_actions_this_round': writing,
            'write_temperature': min(1.0 + 0.05 * writing, 1.5),
            'path': path,
            'original': {'text': ai['tweet'], 'detoxify_toxicity': _f(detox.get(ai_id)),
                         'style_logged': style_logged, 'memory_logged': [
                             {k: r.get(k) for k in ('continuity_text', 'memory_scope', 'memory_retrieved_item_count',
                                                    'cross_thread_callback_candidate', 'cross_thread_callback_used')}
                             for r in memory_logged]},
        }
        assert parent_id in {p['post_id'] for p in thread_posts}, ctx['context_id']
        (out / 'contexts' / f"{row['context_id']}.json").write_text(json.dumps(ctx, indent=1, default=str))
        fidelity_rows.append({
            'context_id': row['context_id'], 'stratum': row['disposition'], 'path': path,
            'thread_nodes': len(thread_posts), 'daily_state_found': daily is not None,
            'staleness_hours': ctx['daily_state']['staleness_hours'], 'social_card': social is not None,
            'thread_card': thread_card is not None, 'social_card_source': social_source,
            'thread_card_source': thread_source, 'karma_source': karma_source,
            'pair_events': len(pair_events), 'pair_event_rule': event_rule,
            'topics': len(topic_names), 'sentiment_topics': len(latest), 'writing_actions_this_round': writing,
            'original_style_logged': bool(style_logged), 'original_memory_logged': bool(memory_logged),
            'root_has_shared_link': _i(posts[root_id].get('news_id')) is not None,
        })
    build = {'run_id': RUN_ID, 'seed': seed, 'contexts': n_contexts, 'allocation': alloc,
             'memory_gist_from_log': bool(gist_from_log),
             'frame_by_disposition': {k: int(v) for k, v in frame.disposition.value_counts().items()},
             'reconcile': reconcile, 'final_dump_created': tables['dump_created'],
             'daily_dumps': [{'created': t.isoformat(), 'path': str(p)} for t, p in dumps],
             'server_logs_indexed': n_logs, 'memory_run_id': run_id, 'memory_run_ids_in_dump': run_ids,
             'fidelity': fidelity_rows,
             'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    (out / 'build.json').write_text(json.dumps(build, indent=1, default=str))
    return build


# ---------------------------------------------------------------------------
# Client setup and one draw (private host).
# ---------------------------------------------------------------------------

def setup_client_copy():
    """Copy the client part of src_run into WORK and add the missing untracked module."""
    dest = WORK / 'src_run'
    client = dest / 'external/YClientReddit'
    if not client.exists():
        (dest / 'external').mkdir(parents=True, exist_ok=True)
        shutil.copytree(SRC_RUN / 'external/YClientReddit', client, symlinks=True)
        shutil.copytree(SRC_RUN / 'data_schema', dest / 'data_schema', symlinks=True)
    stub = client / 'y_client/url_privacy.py'
    if not stub.exists():
        # url_privacy.py was untracked at the run commit and is absent from src_run. It only
        # sanitises shared-link URLs in the feed readers, never on the comment() path.
        stub.write_text('def sanitize_shared_url(value):\n    return value\n')
    return client


_CLIENT = {}


def client_modules():
    if _CLIENT:
        return _CLIENT
    os.environ.pop('DATABASE_URL', None)
    sys.dont_write_bytecode = True
    client = setup_client_copy()
    sys.path.insert(0, str(client))
    import y_client.classes.base_agent as ba
    import y_client.classes.time as yt
    import y_client.recsys as recsys
    from y_client.clients.client_web import YClientWeb
    prompts = YClientWeb._load_prompts_with_defaults(None, str(G1_EXP_DIR))
    config = json.loads((G1_EXP_DIR / 'client_G1-Reddit Experiment.json').read_text())
    _CLIENT.update(ba=ba, yt=yt, recsys=recsys, prompts=prompts, config=config)
    return _CLIENT


class LLMRecorder:
    """Wraps openai chat completions: records parameters, timings and messages."""

    def __init__(self, stub=False):
        self.stub = stub
        self.calls = []
        self.phase = ['write']

    def install(self):
        from openai.resources.chat.completions import Completions
        if getattr(Completions.create, '_a20', False):
            Completions.create._a20_owner[0] = self
            return
        original = Completions.create
        owner = [self]

        def create(inner, *args, **kwargs):
            rec = owner[0]
            messages = kwargs.get('messages') or []
            entry = {'phase': rec.phase[-1], 'model': kwargs.get('model'), 'temperature': kwargs.get('temperature'),
                     'max_tokens': kwargs.get('max_tokens'), 'seed': kwargs.get('seed'), 'top_p': kwargs.get('top_p'),
                     'messages': [{'role': m.get('role'), 'content': m.get('content')} for m in messages]}
            started = time.monotonic()
            if rec.stub:
                result = _stub_completion(kwargs)
            else:
                result = original(inner, *args, **kwargs)
            entry['seconds'] = round(time.monotonic() - started, 3)
            usage = getattr(result, 'usage', None)
            entry['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
            entry['completion_tokens'] = getattr(usage, 'completion_tokens', None)
            entry['response'] = result.choices[0].message.content if result.choices else None
            rec.calls.append(entry)
            return result

        create._a20 = True
        create._a20_owner = owner
        Completions.create = create


def _stub_completion(kwargs):
    from openai.types.chat import ChatCompletion
    text = json.dumps(kwargs.get('messages'))
    content = '1' if 'Select ONE response style' in text else 'That claim skips the actual numbers in the post, so show them.'
    return ChatCompletion.model_validate({
        'id': 'stub', 'object': 'chat.completion', 'created': 0, 'model': kwargs.get('model') or 'stub',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})


def run_one(ctx: dict, arm: str, draw: int, base_seed: int, recorder: LLMRecorder) -> dict:
    import numpy as np
    mods = client_modules()
    ba, yt = mods['ba'], mods['yt']
    server = OfflineServer(ctx)

    def fake_get(url, *a, **kw):
        return server.handle('GET', url, kw.get('data'))

    def fake_post(url, *a, **kw):
        return server.handle('POST', url, kw.get('data'))

    ba.get, ba.post, yt.get, yt.post = fake_get, fake_post, fake_get, fake_post
    cfg = copy.deepcopy(mods['config'])
    cfg['servers']['api'] = OFFLINE_BASE
    cfg['simulation']['show_user_type_labels'] = arm == 'on'
    persona = ctx['persona']
    content_rs = getattr(mods['recsys'], ctx['recsys'][0])()
    follow_rs = getattr(mods['recsys'], ctx['recsys'][1])(leaning_bias=1.5)
    agent = ba.Agent(name=persona['username'], email=OFFLINE_EMAIL, load=True, config=cfg, web=True,
                     recsys=content_rs, frecsys=follow_rs,
                     daily_activity_level=persona['daily_activity_level'])
    agent.set_prompts(copy.deepcopy(mods['prompts']))
    assert agent.show_user_type_labels is (arm == 'on')
    assert agent.type == MODEL, agent.type
    seeds = draw_seeds(base_seed, ctx['context_id'], draw)
    writing = int(ctx['writing_actions_this_round'])
    if writing >= int(agent.max_writing_actions_per_round):
        writing = int(agent.max_writing_actions_per_round) - 1
    agent.writing_actions_this_round = writing
    agent._proactive_affect_this_round = 0
    agent.llm_config['seed'] = seeds['style_llm']
    agent.llm_config['temperature'] = agent._base_temperature
    trace = {'filters': {}, 'length': None, 'alignment': None, 'alignment_rewrite_called': False}

    def wrap(name, before=None, after=None, phase=None):
        original = getattr(agent, name)

        def inner(*args, **kwargs):
            if before:
                before(*args, **kwargs)
            if phase:
                recorder.phase.append(phase)
            try:
                result = original(*args, **kwargs)
            finally:
                if phase:
                    recorder.phase.pop()
            if after:
                after(result, *args, **kwargs)
            return result
        setattr(agent, name, inner)

    wrap('_select_comment_style', before=lambda *a, **k: random.seed(seeds['shuffle']), phase='style')
    wrap('_adversarial_style_override', before=lambda *a, **k: random.seed(seeds['override']))
    wrap('_rewrite_comment_for_parent_alignment', before=lambda *a, **k: trace.update(alignment_rewrite_called=True),
         phase='alignment_rewrite')
    wrap('_rewrite_text_to_limits', phase='length_rewrite')
    wrap('_enforce_text_limits', after=lambda r, *a, **k: trace.update(length={'text': r[0], 'meta': r[1]}))
    wrap('_validate_forum_reply_alignment',
         after=lambda r, *a, **k: trace.update(alignment={'text': a[0] if a else k.get('draft_text'),
                                                          'passes': bool((r or {}).get('passes'))}))
    # Post-generation filters in comment() (last call wins; the filter stage runs last).
    def record_filter(n):
        def after(r, *a, **k):
            text = k.get('text', k.get('draft_text', a[0] if a else None))
            trace['filters'][n] = {'text': text, 'flag': bool(r[0] if isinstance(r, tuple) else r)}
        return after

    for name in FILTER_ORDER:
        wrap(name, after=record_filter(name))

    # Browse path: the target's quality metadata comes from the client's own candidate filter.
    reply_meta, reply_depth = None, None
    target = next(p for p in ctx['thread_posts'] if p['post_id'] == ctx['parent_post_id'])
    if ctx['path'] == 'browse':
        tree = agent._Agent__get_thread_tree(int(ctx['thread_root_id']))
        maps = agent._build_thread_analysis_maps(tree['posts'], tree['thread_root_id'])
        node = maps['node_by_id'][int(ctx['parent_post_id'])]
        reply_depth = maps['depth_by_post_id'].get(int(ctx['parent_post_id']))
        cand = [{'node': node, 'depth': reply_depth, 'score': 0.0}]
        agent._filter_forum_browse_candidates(
            cand, thread_root_id=tree['thread_root_id'], node_by_id=maps['node_by_id'],
            children_by_parent=maps['children_by_parent'], ordered_nodes=maps['ordered_nodes'],
            ordered_index=maps['ordered_index'])
        reply_meta = cand[0].get('quality_meta')
        reply_text = node.get('text')
    else:
        reply_text = target['tweet']
    random.seed(seeds['python'])
    np.random.seed(seeds['numpy'])
    recorder.calls = []
    started = time.monotonic()
    status, payload, error = None, None, None
    try:
        result = agent.comment(int(ctx['parent_post_id']), int(ctx['tid']), max_length_threads=MAX_THREAD_READING,
                               reply_to_username=target['username'], reply_to_text=reply_text,
                               reply_target_depth=reply_depth, reply_target_meta=reply_meta)
        status = 'not_posted' if result is False else 'returned_true_without_post'
    except CommentCaptured as cap:
        status, payload = 'posted', cap.payload
    except Exception as exc:  # recorded and retried on --resume
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    seconds = round(time.monotonic() - started, 3)
    decisions = {d.get('decision_type'): d for d in server.decisions}
    style = decisions.get('comment_style_selection', {})
    reached_filters = FILTER_ORDER[0] in trace['filters']
    misaligned = trace['alignment'] is not None and not trace['alignment']['passes']
    reason = None
    if status == 'not_posted':
        if style.get('selected_style') == 'SKIP':
            reason = 'style_skip'
        elif misaligned:
            # comment() drops drafts that still fail the parent-alignment check after one rewrite.
            reason = 'parent_misaligned'
        elif not reached_filters:
            reason = 'too_short_or_empty'
        else:
            flagged = [n for n in FILTER_ORDER if trace['filters'].get(n, {}).get('flag')]
            reason = flagged[0] if flagged else 'too_short_or_empty'
    # Text the agent posted, or for dropped drafts the last text the client produced.
    final_text, text_source = None, None
    if payload is not None:
        final_text, text_source = payload.get('text'), 'posted_payload'
    elif reached_filters:
        final_text, text_source = trace['filters'][FILTER_ORDER[0]]['text'], 'filtered_payload'
    elif trace['length'] is not None:
        final_text, text_source = trace['length']['text'], 'after_length_limits'
    elif trace['alignment'] is not None:
        final_text, text_source = trace['alignment']['text'], 'misaligned_draft'
    memory_mode = decisions.get('reply_memory_mode', {})
    return {
        'key': record_key(ctx['context_id'], arm, draw), 'run_id': RUN_ID, 'context_id': ctx['context_id'],
        'stratum': ctx['stratum'], 'arm': arm, 'draw': int(draw), 'seeds': seeds, 'status': status,
        'not_posted_reason': reason, 'error': error, 'seconds': seconds,
        'llm_seconds': round(sum(c['seconds'] for c in recorder.calls), 3),
        'selected_style': style.get('selected_style'), 'available_styles': style.get('available_styles'),
        'alignment_rewrite_called': trace['alignment_rewrite_called'],
        'length_rewrite_used': bool((trace['length'] or {}).get('meta', {}).get('rewrite_used')),
        'writing_actions_this_round': writing, 'path': ctx['path'],
        'llm_calls': [{k: c[k] for k in ('phase', 'model', 'temperature', 'max_tokens', 'seed', 'top_p', 'seconds',
                                         'prompt_tokens', 'completion_tokens')} for c in recorder.calls],
        'prompt_sha256': [hashlib.sha256(json.dumps(c['messages']).encode()).hexdigest() for c in recorder.calls],
        'memory_continuity_text': memory_mode.get('continuity_text'), 'memory_scope': memory_mode.get('memory_scope'),
        'unexpected_endpoints': sorted(set(server.unexpected)), 'final_text': final_text,
        'final_text_source': text_source,
        'would_post': status == 'posted',
    }


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------

class Scorer:
    def __init__(self):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        import torch
        torch.set_num_threads(1)
        from detoxify import Detoxify
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.detox = Detoxify('unbiased', device='cpu')
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text):
        if not text:
            return {'detoxify_toxicity': None, 'length_chars': 0 if text == '' else None, 'vader_compound': None}
        return {'detoxify_toxicity': float(self.detox.predict([text])['toxicity'][0]),
                'length_chars': len(text), 'vader_compound': float(self.vader.polarity_scores(text)['compound'])}


# ---------------------------------------------------------------------------
# GPU guard and Ollama housekeeping.
# ---------------------------------------------------------------------------

def gpu_guard():
    me = os.getuid()
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    foreign = []
    for line in out.strip().splitlines():
        pid, mib = [x.strip() for x in line.split(',')]
        try:
            uid = os.stat(f'/proc/{pid}').st_uid
        except FileNotFoundError:
            uid = None
        if uid != me and int(mib) > POLITE_GPU_LIMIT_MIB:
            foreign.append({'pid': int(pid), 'used_mib': int(mib)})
    return foreign


def ollama_unload():
    import urllib.request
    req = urllib.request.Request(f'{OLLAMA}/api/generate', data=json.dumps({'model': MODEL, 'keep_alive': 0}).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


# ---------------------------------------------------------------------------
# Pilot prompt check, aggregation, CLI.
# ---------------------------------------------------------------------------

def prompt_check(contexts, out: Path, base_seed: int, live_records=None):
    """Dry run with a stub model: every call's prompt, ON against OFF."""
    recorder = LLMRecorder(stub=True)
    recorder.install()
    results = []
    pdir = out / 'pilot'
    pdir.mkdir(parents=True, exist_ok=True)
    for ctx in contexts:
        per_arm = {}
        for arm in ARMS:
            rec = run_one(ctx, arm, 0, base_seed, recorder)
            per_arm[arm] = ([c['messages'] for c in recorder.calls], [c['phase'] for c in recorder.calls], rec)
        on_msgs, on_phases, on_rec = per_arm['on']
        off_msgs, off_phases, off_rec = per_arm['off']
        logged = [m.get('continuity_text') for m in ctx['original']['memory_logged']]
        row = {'context_id': ctx['context_id'], 'calls_on': on_phases, 'calls_off': off_phases,
               'unexpected_endpoints': sorted(set(on_rec['unexpected_endpoints']) | set(off_rec['unexpected_endpoints'])),
               'status_on': on_rec['status'], 'status_off': off_rec['status'],
               'error_on': on_rec['error'], 'error_off': off_rec['error'],
               'memory_logged': bool(logged),
               'memory_matches_log': (on_rec['memory_continuity_text'] or '') in [x or '' for x in logged] if logged else None,
               'memory_reconstructed_empty': not on_rec['memory_continuity_text'],
               'memory_logged_empty': all(not x for x in logged) if logged else None,
               'calls': []}
        for i, (mo, mf) in enumerate(zip(on_msgs, off_msgs)):
            for j, (a, b) in enumerate(zip(mo, mf)):
                cmp = compare_prompts(a['content'] or '', b['content'] or '')
                row['calls'].append({'call': i, 'phase': on_phases[i], 'message': j, 'role': a['role'], **cmp})
        results.append(row)
    (pdir / 'prompt_check_dry.json').write_text(json.dumps(results, indent=1))
    return results


def save_live_prompts(ctx_id, calls_by_arm, out: Path):
    pdir = out / 'pilot/prompts'
    pdir.mkdir(parents=True, exist_ok=True)
    checks = []
    for arm, calls in calls_by_arm.items():
        with (pdir / f'{ctx_id}_{arm}.txt').open('w') as fh:
            for i, c in enumerate(calls):
                for m in c['messages']:
                    fh.write(f'===== call {i} phase={c["phase"]} role={m["role"]} =====\n{m["content"]}\n')
                fh.write(f'===== call {i} response =====\n{c.get("response")}\n')
    on, off = calls_by_arm['on'], calls_by_arm['off']
    import difflib
    diff_lines = []
    for i, (a, b) in enumerate(zip(on, off)):
        for j, (ma, mb) in enumerate(zip(a['messages'], b['messages'])):
            checks.append({'call': i, 'phase': a['phase'], 'message': j, 'role': ma['role'],
                           **compare_prompts(ma['content'] or '', mb['content'] or '')})
            diff_lines.extend(difflib.unified_diff((mb['content'] or '').splitlines(), (ma['content'] or '').splitlines(),
                                                   f'off call{i} msg{j}', f'on call{i} msg{j}', lineterm='', n=1))
    (pdir / f'{ctx_id}_off_vs_on.diff').write_text('\n'.join(diff_lines) + '\n')
    return checks


def summarise(out: Path):
    import pandas as pd
    path = out / 'draws.jsonl'
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('final_text', 'llm_calls', 'seeds', 'prompt_sha256',
                                                               'available_styles')} for r in recs])
    df = df.loc[df.status.ne('error')]
    if df.empty:
        return None
    rows = []
    for subset, frame in (('all_draws_with_text', df.loc[df.length_chars.fillna(0).gt(0)]),
                          ('posted', df.loc[df.would_post.astype(bool)])):
        for arm, g in frame.groupby('arm'):
            rows.append({'subset': subset, 'arm': arm, 'draws': len(g), 'contexts': g.context_id.nunique(),
                         'mean_detoxify': g.detoxify_toxicity.mean(),
                         'share_detoxify_gt_0_5': g.detoxify_toxicity.gt(0.5).mean(),
                         'mean_length_chars': g.length_chars.mean(), 'mean_vader_compound': g.vader_compound.mean(),
                         'mean_seconds_per_draw': g.seconds.mean()})
    for arm, g in df.groupby('arm'):
        rows.append({'subset': 'all_draws', 'arm': arm, 'draws': len(g), 'contexts': g.context_id.nunique(),
                     'posted': int(g.would_post.sum()), 'mean_seconds_per_draw': g.seconds.mean(),
                     **{f'not_posted_{k}': int(v) for k, v in g.not_posted_reason.value_counts().items()}})
    table = pd.DataFrame(rows)
    table.insert(0, 'run_id', RUN_ID)
    table.to_csv(out / 'summary_by_arm.csv', index=False)
    styles = df.groupby(['arm', 'selected_style']).size().rename('draws').reset_index()
    styles.insert(0, 'run_id', RUN_ID)
    styles.to_csv(out / 'summary_styles_by_arm.csv', index=False)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--contexts', type=int, default=60, help='sample size (stratified, proportional)')
    parser.add_argument('--draws', type=int, default=4, help='draws per arm and context')
    parser.add_argument('--pilot', type=int, default=None, help='run only the first N contexts and the prompt check')
    parser.add_argument('--resume', action='store_true', help='append to an existing draws.jsonl, skipping done keys')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--memory-gist-from-log', action='store_true',
                        help='fill same-thread gists missing from the dumps with the gist in the original decision log')
    parser.add_argument('--dry-only', action='store_true',
                        help='run the stub-model prompt check on the selected contexts and stop (no GPU)')
    parser.add_argument('--keep-loaded', action='store_true', help='do not unload the Ollama model at the end')
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # the client's execution logger writes agent_execution.log to the working directory
    draws_path = out / 'draws.jsonl'
    if draws_path.exists() and not args.resume:
        sys.exit(f'{draws_path} exists; pass --resume to continue it')
    build_path = out / 'build.json'
    if build_path.exists():
        build = json.loads(build_path.read_text())
        if (build['contexts'] != args.contexts or build['seed'] != args.seed
                or build.get('memory_gist_from_log', False) != args.memory_gist_from_log):
            sys.exit('build.json was made with a different --contexts, --seed or --memory-gist-from-log')
    else:
        t0 = time.monotonic()
        build = build_contexts(out, args.contexts, args.seed, args.memory_gist_from_log)
        print(f'built {args.contexts} contexts in {time.monotonic() - t0:.1f} s', flush=True)
    if args.prepare_only:
        return
    ids = [f'c{i:02d}' for i in range(args.contexts)]
    if args.pilot:
        ids = ids[:args.pilot]
    contexts = {c: json.loads((out / 'contexts' / f'{c}.json').read_text()) for c in ids}
    log = (out / 'run.log').open('a')

    def say(msg):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {msg}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    if args.pilot or args.dry_only:
        dry = prompt_check([contexts[c] for c in ids], out, args.seed)
        say('dry prompt check: ' + json.dumps([{
            'context': r['context_id'], 'calls_on': r['calls_on'], 'calls_off': r['calls_off'],
            'unexpected': r['unexpected_endpoints'], 'errors': [r['error_on'], r['error_off']],
            'all_on_marker_ok': all(c['on_has_marker_token'] and c['on_has_note'] for c in r['calls']
                                    if c['role'] == 'user' and c['phase'] in ('style', 'write')),
            'off_clean': not any(c['off_has_marker_token'] or c['off_has_note'] for c in r['calls']),
            'memory_matches_log': r['memory_matches_log'],
            'identical_after_strip': [c['identical_after_strip'] for c in r['calls']]} for r in dry]))
        if args.dry_only:
            return
    foreign = gpu_guard()
    if foreign:
        say(f'GPU in use by another user above {POLITE_GPU_LIMIT_MIB} MiB: {foreign}; not loading the model')
        sys.exit(3)
    recorder = LLMRecorder(stub=False)
    recorder.install()
    scorer = Scorer()
    done = completed_keys(draws_path) if args.resume else set()
    jobs = plan_jobs(ids, args.draws, done)
    say(f'{len(jobs)} jobs ({len(done)} already done)')
    live_checks = {}
    pending_prompts = {}
    with draws_path.open('a') as sink:
        for n, (cid, draw, arm) in enumerate(jobs):
            if n and n % 10 == 0 and gpu_guard():
                say('another user started a large GPU job; stopping (resume later)')
                break
            rec = run_one(contexts[cid], arm, draw, args.seed, recorder)
            rec.update(scorer.score(rec['final_text']) if rec['status'] != 'error' else {})
            rec['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            sink.write(json.dumps(rec, default=str) + '\n')
            sink.flush()
            os.fsync(sink.fileno())
            say(f"{rec['key']} {rec['status']} style={rec['selected_style']} {rec['seconds']}s "
                f"llm={rec['llm_seconds']}s calls={len(rec['llm_calls'])}")
            if args.pilot and draw == 0:
                pending_prompts.setdefault(cid, {})[arm] = [dict(c) for c in recorder.calls]
                if len(pending_prompts[cid]) == 2:
                    live_checks[cid] = save_live_prompts(cid, pending_prompts[cid], out)
    if args.pilot:
        (out / 'pilot/prompt_check_live.json').write_text(json.dumps(live_checks, indent=1))
    table = summarise(out)
    if table is not None:
        say('\n' + table.to_string(index=False))
    if not args.keep_loaded:
        ollama_unload()
        say('model unloaded')


if __name__ == '__main__':
    main()
