"""CHB pre-submission item A13: direct thread entry with feed-position and conduct covariates.

Run ``chb_prereview_20261110`` (item ``a13_entry_covariates``). Extends the
admitted RQ1b model (run ``rq1b_direct_thread_entry_release_20260904``; 5,668
participant-feed-visit-root-post opportunities, 438 direct entries, 32
participants) with

* ``z_log_feed_rank``: the card's order of first appearance within its feed
  visit (1 = first feed card to appear), log then standardized over the model
  rows. A feed-card appearance is a raw Moveo row with action ``appear``, type
  ``post`` and semantic group ``feed_container``; distinct cards are ranked by
  their first such row strictly after the visit's opening feed page view and
  before the visit's closing page view (the panel builder's
  ``feed_entry_event_seq`` and ``feed_exit_event_seq``; an open visit runs to
  the end of the session file). The rank counts every feed card, not only
  eligible opportunities.
* ``toxic_persona``: the root post was written by an AI account whose
  configured persona disposition is "moderately" or "extremely" toxic
  (0 for participant posts), from the public ``01_accounts/accounts.csv``.
* ``z_root_toxicity`` (secondary sensitivity only): Detoxify 'unbiased'
  toxicity of the root post, logit(clip(p, 1e-4, 1 - 1e-4)) standardized. AI
  root posts are long RSS-news posts and score implausibly high, so this term
  is not a primary covariate.

Feed rank is partly determined by the outcome: a visit closes when the
participant opens a thread, so the entered card is usually among the visit's
last cards to appear, and given visit size the rank nearly encodes the entry.
Two outcome-robust sensitivities (fix-up 2026-11-11) therefore refit

* ``position_disposition_no_visit_size``: the reported specification plus
  persona and ``z_log_feed_rank`` without the visit-size covariate
  ``z_log_feed_spell_cards``;
* ``prefinal_position_disposition``: the reported specification (visit size
  kept) plus persona and ``z_log_feed_rank_prefinal``, the rank among the
  cards that appeared before the visit's last two distinct feed cards. Cards
  in the final two positions are coded at the cap (number of earlier cards
  plus one, i.e. ``min(rank, max(cards - 1, 1))``), not excluded, because 397
  of the 438 entries are in those two positions.

Covariate priors are Normal(0, 0.5), the reported adjustment scale; every other
prior, the varying-effect structure and the sampler settings (4 chains, 1,500
tuning, 2,000 draws, target acceptance 0.95, seed 20261715) are those of the
admitted fit. Coefficient intervals are equal-tailed 95% posterior quantiles,
as are the estimand intervals.

Estimand: hidden-minus-visible (G2 minus G1) change in the AI-minus-participant
direct-entry probability. Predictions keep each community's observed
opportunities (post age, visit size and feed rank at observed values), set the
authorship indicator to the cell value, average within participant and then
equally across participants. Post-type covariates (persona, root toxicity) are
integrated over atoms of their distribution among the cell's author type:
``community`` uses that community's opportunities, ``pooled`` uses both
communities' opportunities (removing between-community differences in AI
persona or toxicity mix). Participant cells have persona 0. Root toxicity is
integrated over ``N_TOXICITY_ATOMS`` (40) mid-quantile atoms within each
persona level. Without post-type covariates both schemes equal the reported
standardization.

Stages:

``extract``    (private host, one thread): raw Moveo session CSVs + admitted panel ->
    per-opportunity feed rank keyed by the public opportunity id, and Detoxify
    scores for root posts that lack a public score.
``crosscheck`` (public record): recompute the feed rank from the public
    ``session_events.parquet`` and compare it with the extracted ranks.
``fit``        (public record): one specification on the public full-record panel.
``estimands``  (public record): recompute one specification's estimands from its saved
    posterior (used to move the toxicity integration to 40 atoms).
``tables``     : assemble runs/chb_prereview_20261110/tables/chb_prereview_a13_entry_covariates_*.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'BLIS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN_ID = 'chb_prereview_20261110'
ITEM = 'a13_entry_covariates'
SOURCE_RUN = 'rq1b_direct_thread_entry_release_20260904'
ADMITTED_PANEL_SHA256 = 'c600d00aa3acdc8c659f4f9bf99249c757569fb4380233b713b3916f2e301615'
ADMITTED_FOCAL = (0.044142629110372554, 0.008138925676092694, 0.08103535397864467)
DRAWS, TUNE, CHAINS, TARGET_ACCEPT = 2000, 1500, 4, 0.95
SEED = 20260715 + 1000  # admitted: config.seed + round(1000 * prior_multiplier)
TOXIC_DISPOSITIONS = {'moderately', 'extremely'}
GROUPS = ('G1', 'G2')
AUTHORS = ('H', 'AI')
SIGN = ('focal: (G2 AI - G2 H) - (G1 AI - G1 H), i.e. hidden-label (G2) minus visible-label (G1) change in '
        'the AI-minus-participant direct-entry probability; G2_minus_G1 rows are hidden minus visible; '
        'AI_minus_H rows are AI minus participant within a community')
SPECS: dict[str, tuple[str, ...]] = {
    'reported': (),
    'position': ('z_log_feed_rank',),
    'disposition': ('toxic_persona',),
    'position_disposition': ('z_log_feed_rank', 'toxic_persona'),
    'position_disposition_detox': ('z_log_feed_rank', 'toxic_persona', 'z_root_toxicity'),
    'position_disposition_no_visit_size': ('z_log_feed_rank', 'toxic_persona'),
    'prefinal_position_disposition': ('z_log_feed_rank_prefinal', 'toxic_persona'),
}
NO_VISIT_SIZE_SPECS = frozenset({'position_disposition_no_visit_size'})
N_TOXICITY_ATOMS = 40
PREFINAL_TAIL = 2  # cards in the visit's last PREFINAL_TAIL positions are coded at the cap
POST_TYPE_TERMS = ('toxic_persona', 'z_root_toxicity')
FEED_CARD = {'action': 'appear', 'type': 'post', 'semantic_group': 'feed_container'}


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


class RunLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        print(line, flush=True)
        with self.path.open('a') as stream:
            stream.write(line + '\n')


def save_json(path: Path | str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=False, default=str) + '\n')


# ---------------------------------------------------------------------------
# Feed position (pure functions)
# ---------------------------------------------------------------------------

def feed_card_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Feed-card appearance rows of one raw session CSV, with 1-based ``event_seq``.

    ``event_seq`` is the row position in the CSV as read by pandas, exactly as
    in ``scripts/session_tracking/normalize_session_tracking.py``.
    """
    seq = np.arange(1, len(raw) + 1)
    mask = np.ones(len(raw), dtype=bool)
    for column, value in FEED_CARD.items():
        mask &= raw[column].astype(str).str.lower().eq(value).to_numpy()
    mask &= raw['id'].notna().to_numpy()
    return pd.DataFrame({'event_seq': seq[mask], 'id': raw['id'].to_numpy()[mask].astype(str)})


def visit_feed_ranks(raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Rank of each opportunity's card among all feed cards of its visit.

    ``visits`` has one row per opportunity with ``opportunity_id``,
    ``feed_entry_event_seq``, ``feed_exit_event_seq`` (missing for an open
    visit) and ``post_id``. Returns ``feed_rank`` (1 = first distinct feed card
    to appear in the visit), ``feed_visit_cards`` (distinct feed cards in the
    visit), ``raw_first_appear_seq`` and the number of raw page views inside the
    window.
    """
    cards = feed_card_rows(raw)
    page_seq = np.flatnonzero(raw['action'].astype(str).str.lower().eq('page_view').to_numpy()) + 1
    n_rows = len(raw)
    out = []
    for (entry, exit_), part in visits.groupby(['feed_entry_event_seq', 'feed_exit_event_seq'], dropna=False,
                                               sort=False):
        entry = int(entry)
        upper = n_rows + 1 if pd.isna(exit_) else int(exit_)
        window = cards[(cards.event_seq > entry) & (cards.event_seq < upper)]
        first = window.groupby('id', sort=False).event_seq.min().sort_values(kind='mergesort')
        rank = pd.Series(np.arange(1, len(first) + 1), index=first.index)
        inside_pages = int(((page_seq > entry) & (page_seq < upper)).sum())
        later_pages = int((page_seq > entry).sum()) if pd.isna(exit_) else 0
        for row in part.itertuples(index=False):
            card_id = f'post_{int(row.post_id)}'
            out.append({
                'opportunity_id': row.opportunity_id,
                'feed_rank': int(rank[card_id]) if card_id in rank.index else pd.NA,
                'feed_visit_cards': int(len(first)),
                'raw_first_appear_seq': int(first[card_id]) if card_id in first.index else pd.NA,
                'raw_page_views_inside_visit': inside_pages,
                'raw_page_views_after_open_visit': later_pages,
                'raw_entry_is_feed_page_view': bool(
                    1 <= entry <= n_rows
                    and str(raw['action'].iloc[entry - 1]).lower() == 'page_view'
                    and '/rfeed' in str(raw['section'].iloc[entry - 1])),
                'raw_exit_is_page_view': bool(
                    pd.isna(exit_) or (1 <= upper <= n_rows
                                       and str(raw['action'].iloc[upper - 1]).lower() == 'page_view')),
            })
    return pd.DataFrame(out)


def eligible_feed_rank(panel: pd.DataFrame) -> pd.Series:
    """Rank among the visit's eligible opportunities by first appearance."""
    return (panel.groupby('feed_spell_key')['first_appear_event_seq']
            .rank(method='first').astype(int))


def prefinal_feed_rank(rank: Iterable[float], cards: Iterable[float], tail: int = PREFINAL_TAIL) -> np.ndarray:
    """Rank among the cards that appeared before the visit's last ``tail`` cards.

    Cards at positions ``1 .. cards - tail`` keep their rank; cards in the final
    ``tail`` positions are coded at the cap ``cards - tail + 1`` (at least 1).
    """
    rank = np.asarray(rank, dtype=float)
    cards = np.asarray(cards, dtype=float)
    if (rank < 1).any() or (rank > cards).any():
        raise ValueError('feed rank outside 1..visit cards')
    return np.minimum(rank, np.maximum(cards - tail + 1, 1.0))


def public_feed_ranks(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Feed rank recomputed from the public ``session_events`` table.

    Same rule as ``visit_feed_ranks`` but on the released ``event_seq``: feed-card
    appearances strictly after the visit's opening page view and before its
    closing page view (to the end of the session for an open visit), distinct
    ids ranked by first appearance.
    """
    events = events.assign(event_seq=pd.to_numeric(events.event_seq))
    is_card = np.ones(len(events), dtype=bool)
    for column, value in FEED_CARD.items():
        is_card &= events[column].astype(str).str.lower().eq(value).to_numpy()
    is_card &= events['id'].notna().to_numpy()
    cards = events.loc[is_card, ['session_id', 'event_seq', 'id']]
    cards_by_session = {sid: part for sid, part in cards.groupby('session_id', sort=False)}
    last_seq = events.groupby('session_id').event_seq.max()
    empty = cards.iloc[:0]
    out = []
    for (sid, entry, exit_), part in panel.groupby(['session_id', 'feed_entry_event_seq', 'feed_exit_event_seq'],
                                                   dropna=False, sort=False):
        upper = int(last_seq[sid]) + 1 if pd.isna(exit_) else int(exit_)
        session_cards = cards_by_session.get(sid, empty)
        window = session_cards[(session_cards.event_seq > int(entry)) & (session_cards.event_seq < upper)]
        first = window.groupby('id', sort=False).event_seq.min().sort_values(kind='mergesort')
        rank = pd.Series(np.arange(1, len(first) + 1), index=first.index)
        for row in part.itertuples(index=False):
            card_id = f'post_{int(row.post_id)}'
            out.append({'opportunity_id': row.opportunity_id,
                        'public_feed_rank': int(rank[card_id]) if card_id in rank.index else pd.NA,
                        'public_feed_visit_cards': int(len(first)),
                        'public_first_appear_seq': int(first[card_id]) if card_id in first.index else pd.NA})
    return pd.DataFrame(out)


def rank_crosscheck(public: pd.DataFrame, extracted: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    """Compare public-event ranks with the raw-CSV extraction (aggregate counts only)."""
    m = extracted.merge(public, on='opportunity_id', how='outer', indicator=True, validate='one_to_one')
    m = m.merge(panel[['opportunity_id', 'first_appear_event_seq']], on='opportunity_id', how='left')
    both = m['_merge'].eq('both')
    diff = (m.public_feed_rank.astype(float) - m.feed_rank.astype(float)).abs()
    return {
        'opportunities': int(len(panel)),
        'matched_rows': int(both.sum()),
        'rank_equal': int((m.public_feed_rank.astype('Int64') == m.feed_rank.astype('Int64')).fillna(False).sum()),
        'cards_equal': int((m.public_feed_visit_cards.astype('Int64')
                            == m.feed_visit_cards.astype('Int64')).fillna(False).sum()),
        'first_appear_seq_equal_panel': int((m.public_first_appear_seq.astype('Int64')
                                             == m.first_appear_event_seq.astype('Int64')).fillna(False).sum()),
        'public_rank_missing': int(m.public_feed_rank.isna().sum()),
        'max_abs_rank_diff': float(diff.max()) if diff.notna().any() else float('nan'),
    }


# ---------------------------------------------------------------------------
# Extraction stage (private host)
# ---------------------------------------------------------------------------

def run_extract(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / 'extract.log')
    started = time.monotonic()
    log(f'{RUN_ID}/{ITEM} extract script_sha256={sha256(__file__)}')
    panel = pd.read_parquet(args.panel)
    log(f'panel {args.panel}: {len(panel)} opportunities, {int(panel.direct_thread_entry.sum())} entries')
    sessions_dir = Path(args.sessions_dir)
    frames = []
    columns = ['opportunity_id', 'feed_entry_event_seq', 'feed_exit_event_seq', 'post_id']
    for session_id, part in panel.groupby('session_id', sort=True):
        raw = pd.read_csv(sessions_dir / f'{session_id}.csv')
        frames.append(visit_feed_ranks(raw, part[columns]))
    position = pd.concat(frames, ignore_index=True)
    merged = panel[['opportunity_id', 'feed_spell_key', 'first_appear_event_seq', 'group', 'author_type',
                    'feed_spell_closed_by_page_view']].merge(position, on='opportunity_id', how='left',
                                                              validate='one_to_one')
    merged['feed_rank_eligible'] = eligible_feed_rank(panel).to_numpy()
    checks = {
        'opportunities': int(len(panel)),
        'sessions_read': int(panel.session_id.nunique()),
        'visits': int(panel.feed_spell_key.nunique()),
        'rank_missing': int(merged.feed_rank.isna().sum()),
        'first_appear_seq_mismatch': int((merged.raw_first_appear_seq.astype('Int64')
                                          != merged.first_appear_event_seq.astype('Int64')).fillna(True).sum()),
        'entry_not_feed_page_view': int((~merged.raw_entry_is_feed_page_view).sum()),
        'exit_not_page_view': int((~merged.raw_exit_is_page_view).sum()),
        'opportunities_with_raw_page_view_inside_visit': int(merged.raw_page_views_inside_visit.gt(0).sum()),
        'open_visit_opportunities_with_later_raw_page_view': int(
            merged.raw_page_views_after_open_visit.gt(0).sum()),
        'rank_ge_eligible_rank': bool((merged.feed_rank.astype(float)
                                       >= merged.feed_rank_eligible.astype(float)).all()),
        'spearman_rank_vs_eligible_rank': float(merged.feed_rank.astype(float).rank().corr(
            merged.feed_rank_eligible.astype(float).rank())),
    }
    log(f'position checks {checks}')
    keep = ['opportunity_id', 'feed_rank', 'feed_visit_cards', 'feed_rank_eligible']
    merged[keep].to_csv(out / 'feed_position.csv', index=False)

    # Detoxify scores for root posts without a public score
    scores = pd.read_csv(args.public_scores)
    scores['post_key'] = scores.group.astype(str) + ':' + scores.content_id.astype(int).astype(str)
    roots = panel.drop_duplicates('post_key')[['post_key', 'group', 'post_id', 'author_type']].copy()
    roots = roots.merge(scores[['post_key', 'detoxify_toxicity']].drop_duplicates('post_key'), on='post_key',
                        how='left')
    roots['score_source'] = np.where(roots.detoxify_toxicity.notna(), 'public_scores', 'fresh_unbiased')
    missing = roots[roots.detoxify_toxicity.isna()]
    rng = np.random.default_rng(SEED)
    check_keys = []
    for author_type in AUTHORS:
        pool = roots.loc[roots.detoxify_toxicity.notna() & roots.author_type.eq(author_type), 'post_key']
        check_keys += list(rng.choice(pool.to_numpy(), size=min(10, len(pool)), replace=False))
    contents = pd.read_parquet(args.contents, columns=['group', 'content_id', 'tweet'])
    contents['post_key'] = contents.group.astype(str) + ':' + contents.content_id.astype(int).astype(str)
    texts = contents.drop_duplicates('post_key').set_index('post_key').tweet
    wanted = list(missing.post_key) + check_keys
    absent = sorted(set(wanted) - set(texts.index))
    if absent:
        raise RuntimeError(f'{len(absent)} root posts lack text')
    import torch
    torch.set_num_threads(1)
    from detoxify import Detoxify
    model = Detoxify('unbiased', device='cpu')
    batch = [str(value) if pd.notna(value) else '' for value in texts.loc[wanted]]
    values = []
    for start in range(0, len(batch), 16):
        values.extend(float(v) for v in model.predict(batch[start:start + 16])['toxicity'])
    fresh = pd.Series(values, index=wanted)
    roots.loc[roots.detoxify_toxicity.isna(), 'detoxify_toxicity'] = roots.loc[
        roots.detoxify_toxicity.isna(), 'post_key'].map(fresh)
    comparison = roots.set_index('post_key').loc[check_keys, 'detoxify_toxicity']
    detox_checks = {
        'roots': int(len(roots)),
        'roots_scored_fresh': int(len(missing)),
        'fresh_by_group_author': missing.groupby(['group', 'author_type']).size().rename('n').reset_index()
        .to_dict('records'),
        'rescored_public_check_n': int(len(check_keys)),
        'rescored_public_max_abs_difference': float((fresh.loc[check_keys] - comparison).abs().max()),
        'median_by_author_type': roots.groupby('author_type').detoxify_toxicity.median().to_dict(),
    }
    log(f'detoxify checks {detox_checks}')
    roots[['group', 'post_id', 'author_type', 'detoxify_toxicity', 'score_source']].to_csv(
        out / 'root_detoxify.csv', index=False)
    save_json(out / 'extract_checks.json', {
        'position': checks, 'detoxify': detox_checks,
        'inputs': {'panel': {'path': str(args.panel), 'sha256': sha256(args.panel)},
                   'sessions_dir': str(sessions_dir),
                   'public_scores': {'path': str(args.public_scores), 'sha256': sha256(args.public_scores)},
                   'contents': {'path': str(args.contents), 'sha256': sha256(args.contents)}},
        'session_csv_sha256': {sid: sha256(sessions_dir / f'{sid}.csv') for sid in sorted(panel.session_id.unique())},
        'elapsed_seconds': time.monotonic() - started, 'peak_rss_gib': peak_rss_gib(),
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
                     'torch': torch.__version__},
        'host': socket.gethostname(),
    })
    log(f'extract done in {time.monotonic() - started:.1f} s, peak RSS {peak_rss_gib():.2f} GiB')


# ---------------------------------------------------------------------------
# Model frame
# ---------------------------------------------------------------------------

def persona_lookup(accounts: pd.DataFrame) -> pd.Series:
    """1.0 for AI accounts with a moderately/extremely toxic disposition, else 0.0; keyed group:user_id."""
    keys = accounts.group.astype(str) + ':' + accounts.user_id.astype(int).astype(str)
    ai = accounts.actor_class.astype(str).str.lower().eq('ai')
    disposition = accounts.toxicity.astype(str).str.strip().str.lower()
    return pd.Series((ai & disposition.isin(TOXIC_DISPOSITIONS)).astype(float).to_numpy(), index=keys)


def standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('non-finite covariate values')
    mean, sd = float(values.mean()), float(values.std(ddof=0))
    return (values - mean) / (sd if sd > 0 else 1.0), mean, sd


def add_covariates(frame: pd.DataFrame, position: pd.DataFrame, persona: pd.Series,
                   root_scores: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach feed rank, toxic persona and root toxicity to a ``build_model_frame`` frame."""
    out = frame.merge(position[['opportunity_id', 'feed_rank', 'feed_visit_cards', 'feed_rank_eligible']],
                      on='opportunity_id', how='left', validate='one_to_one')
    if out.feed_rank.isna().any():
        raise ValueError(f'{int(out.feed_rank.isna().sum())} opportunities lack a feed rank')
    scalers = []
    z, mean, sd = standardize(np.log(out.feed_rank.astype(float).to_numpy()))
    out['z_log_feed_rank'] = z
    scalers.append(dict(model_term='z_log_feed_rank', source='feed_rank', transformation='log_then_standardize',
                        training_mean=mean, training_sd=sd))
    out['feed_rank_prefinal'] = prefinal_feed_rank(out.feed_rank, out.feed_visit_cards)
    z, mean, sd = standardize(np.log(out.feed_rank_prefinal.to_numpy()))
    out['z_log_feed_rank_prefinal'] = z
    scalers.append(dict(model_term='z_log_feed_rank_prefinal', source=f'feed_rank_capped_last_{PREFINAL_TAIL}',
                        transformation=f'min(rank, max(cards - {PREFINAL_TAIL - 1}, 1)) then log_then_standardize',
                        training_mean=mean, training_sd=sd))
    author_key = out.group.astype(str) + ':' + out.author_id.astype(int).astype(str)
    toxic = author_key.map(persona)
    is_ai = out.ai_author.astype(int).eq(1)
    if toxic[is_ai].isna().any():
        raise ValueError('AI root post without a persona disposition')
    out['toxic_persona'] = np.where(is_ai, toxic.fillna(0.0), 0.0)
    scores = root_scores.copy()
    scores['post_key'] = scores.group.astype(str) + ':' + scores.post_id.astype(int).astype(str)
    p = out.post_key.map(scores.drop_duplicates('post_key').set_index('post_key').detoxify_toxicity)
    if p.isna().any():
        raise ValueError(f'{int(p.isna().sum())} opportunities lack a root toxicity score')
    q = p.astype(float).clip(1e-4, 1 - 1e-4).to_numpy()
    z, mean, sd = standardize(np.log(q / (1 - q)))
    out['root_toxicity_probability'] = p.astype(float).to_numpy()
    out['z_root_toxicity'] = z
    scalers.append(dict(model_term='z_root_toxicity', source='detoxify_unbiased_root',
                        transformation='logit_clip_1e-4_then_standardize', training_mean=mean, training_sd=sd))
    return out, scalers


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(frame: pd.DataFrame, sampling: Any, extra_terms: Sequence[str] = (),
                visit_size: bool = True) -> Any:
    """The admitted RQ1b model (``rq1b.build_pymc_model``) plus ``beta_<term>`` covariates.

    ``visit_size=False`` drops the visit-size term ``beta_log_feed_spell_cards``.
    """
    import pymc as pm
    from scripts.publication_analysis.rq1b import PRIMARY_OUTCOME, _model_coordinates

    coords = _model_coordinates(frame)
    focal = float(sampling.prior_focal_sigma)
    adjustment = float(sampling.prior_adjustment_sigma)
    scale = float(sampling.prior_scale_sigma)
    with pm.Model(coords=coords) as model:
        def data(name, dtype):
            return pm.Data(name, frame[name].to_numpy(dtype=dtype), dims='opportunity')
        g2, ai = data('g2', int), data('ai_author', int)
        participant, session = data('participant_index', int), data('session_index', int)
        post, valid_day = data('post_index', int), data('day_index', int)
        z_age, z_cards = data('z_log_post_age', float), data('z_log_feed_spell_cards', float)
        extras = {term: data(term, float) for term in extra_terms}

        alpha = pm.Normal('alpha', mu=-2.0, sigma=1.5)
        beta_group = pm.Normal('beta_group', mu=0.0, sigma=adjustment)
        beta_ai = pm.Normal('beta_ai', mu=0.0, sigma=focal)
        beta_group_ai = pm.Normal('beta_group_ai', mu=0.0, sigma=focal)
        beta_log_age = pm.Normal('beta_log_age', mu=0.0, sigma=adjustment)
        beta_log_feed_spell_cards = (pm.Normal('beta_log_feed_spell_cards', mu=0.0, sigma=adjustment)
                                     if visit_size else 0.0)
        betas ={term: pm.Normal(f'beta_{term}', mu=0.0, sigma=adjustment) for term in extra_terms}

        sigma_participant = pm.HalfNormal('sigma_participant', sigma=scale)
        sigma_participant_ai = pm.HalfNormal('sigma_participant_ai', sigma=scale)
        sigma_session = pm.HalfNormal('sigma_session', sigma=scale)
        sigma_post = pm.HalfNormal('sigma_post', sigma=scale)
        sigma_day = pm.HalfNormal('sigma_day', sigma=scale)
        z_participant = pm.Normal('z_participant', mu=0.0, sigma=1.0, dims='participant')
        z_participant_ai = pm.Normal('z_participant_ai', mu=0.0, sigma=1.0, dims='participant')
        z_session = pm.Normal('z_session', mu=0.0, sigma=1.0, dims='session')
        z_post = pm.Normal('z_post', mu=0.0, sigma=1.0, dims='post')
        z_day = pm.Normal('z_day', mu=0.0, sigma=1.0, dims='valid_day')
        participant_effect = pm.Deterministic('participant_effect', z_participant * sigma_participant,
                                              dims='participant')
        participant_ai_effect = pm.Deterministic('participant_ai_effect', z_participant_ai * sigma_participant_ai,
                                                 dims='participant')
        session_effect = pm.Deterministic('session_effect', z_session * sigma_session, dims='session')
        post_effect = pm.Deterministic('post_effect', z_post * sigma_post, dims='post')
        day_effect = pm.Deterministic('day_effect', z_day * sigma_day, dims='valid_day')
        eta = (alpha + beta_group * g2 + beta_ai * ai + beta_group_ai * g2 * ai
               + beta_log_age * z_age + beta_log_feed_spell_cards * z_cards
               + participant_effect[participant] + participant_ai_effect[participant] * ai
               + session_effect[session] + post_effect[post] + day_effect[valid_day])
        for term in extra_terms:
            eta = eta + betas[term] * extras[term]
        pm.Bernoulli('opened', logit_p=eta, observed=frame[PRIMARY_OUTCOME].to_numpy(dtype=int),
                     dims='opportunity')
    return model


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------

def post_type_atoms(frame: pd.DataFrame, mask: np.ndarray, terms: Sequence[str],
                    n_atoms: int = N_TOXICITY_ATOMS) -> list[tuple[dict[str, float], float]]:
    """Atoms (values, weight) of the post-type covariates among the masked rows.

    Persona is integrated exactly over its observed levels; root toxicity by
    ``n_atoms`` mid-quantiles within each persona level (as in
    ``review_choice_models_conduct.atoms``). Weights sum to one.
    """
    terms = [t for t in terms if t in POST_TYPE_TERMS]
    if not terms:
        return [({}, 1.0)]
    sub = frame.loc[mask]
    if sub.empty:
        raise ValueError('empty atom support')
    levels = sub.groupby('toxic_persona') if 'toxic_persona' in terms else [(None, sub)]
    atoms: list[tuple[dict[str, float], float]] = []
    for level, part in levels:
        weight = len(part) / len(sub)
        base = {} if level is None else {'toxic_persona': float(level)}
        if 'z_root_toxicity' in terms:
            quantiles = np.quantile(part.z_root_toxicity.to_numpy(), (np.arange(n_atoms) + 0.5) / n_atoms)
            atoms += [({**base, 'z_root_toxicity': float(q)}, weight / n_atoms) for q in quantiles]
        else:
            atoms.append((base, weight))
    return atoms


def _draws(idata: Any, name: str, dim: str | None = None) -> np.ndarray:
    values = idata.posterior[name].stack(sample=('chain', 'draw'))
    return values.transpose('sample').values if dim is None else values.transpose('sample', dim).values


def cell_draws(idata: Any, frame: pd.DataFrame, extra_terms: Sequence[str], scheme: str,
               chunk: int = 256, n_atoms: int = N_TOXICITY_ATOMS) -> dict[str, dict[str, np.ndarray]]:
    """Posterior draws of the four cell probabilities under both standardization scopes.

    A posterior without ``beta_log_feed_spell_cards`` (no-visit-size fit) has that term set to 0.
    """
    from scipy.special import expit

    if scheme not in ('community', 'pooled'):
        raise ValueError(scheme)
    scalars = {n: _draws(idata, n) for n in ('alpha', 'beta_group', 'beta_ai', 'beta_group_ai', 'beta_log_age')}
    scalars['beta_log_feed_spell_cards'] = (_draws(idata, 'beta_log_feed_spell_cards')
                                            if 'beta_log_feed_spell_cards' in idata.posterior
                                            else np.zeros_like(scalars['alpha']))
    betas = {t: _draws(idata, f'beta_{t}') for t in extra_terms}
    pe = _draws(idata, 'participant_effect', 'participant')
    pae = _draws(idata, 'participant_ai_effect', 'participant')
    se = _draws(idata, 'session_effect', 'session')
    poe = _draws(idata, 'post_effect', 'post')
    de = _draws(idata, 'day_effect', 'valid_day')
    n_draws = len(scalars['alpha'])
    row_terms = [t for t in extra_terms if t not in POST_TYPE_TERMS]
    type_terms = [t for t in extra_terms if t in POST_TYPE_TERMS]
    out: dict[str, dict[str, np.ndarray]] = {'equal_participant_observed_opportunities': {},
                                             'card_weighted_observed_opportunities': {}}
    group = frame.group.astype(str).to_numpy()
    author = frame.author_type.astype(str).to_numpy()
    for g in GROUPS:
        rows = np.flatnonzero(group == g)
        part_idx = frame.participant_index.to_numpy()[rows]
        levels, inverse = np.unique(part_idx, return_inverse=True)
        member = np.zeros((len(rows), len(levels)))
        member[np.arange(len(rows)), inverse] = 1.0
        member /= member.sum(axis=0, keepdims=True)
        sub = frame.iloc[rows]
        g2 = float(g == 'G2')
        for a in AUTHORS:
            ai = float(a == 'AI')
            support = author == a
            if scheme == 'community':
                support = support & (group == g)
            atoms = post_type_atoms(frame, support, type_terms, n_atoms=n_atoms)
            if a == 'H':
                atoms = [({**values, 'toxic_persona': 0.0} if 'toxic_persona' in values else values, w)
                         for values, w in atoms]
            equal = np.empty(n_draws)
            weighted = np.empty(n_draws)
            for start in range(0, n_draws, chunk):
                sl = slice(start, min(start + chunk, n_draws))
                eta = (scalars['alpha'][sl, None] + scalars['beta_group'][sl, None] * g2
                       + scalars['beta_ai'][sl, None] * ai + scalars['beta_group_ai'][sl, None] * g2 * ai
                       + scalars['beta_log_age'][sl, None] * sub.z_log_post_age.to_numpy()[None, :]
                       + scalars['beta_log_feed_spell_cards'][sl, None]
                       * sub.z_log_feed_spell_cards.to_numpy()[None, :]
                       + pe[sl][:, sub.participant_index.to_numpy()]
                       + pae[sl][:, sub.participant_index.to_numpy()] * ai
                       + se[sl][:, sub.session_index.to_numpy()]
                       + poe[sl][:, sub.post_index.to_numpy()]
                       + de[sl][:, sub.day_index.to_numpy()])
                for t in row_terms:
                    eta = eta + betas[t][sl, None] * sub[t].to_numpy()[None, :]
                probability = np.zeros_like(eta)
                for values, w in atoms:
                    shift = sum(betas[t][sl] * values[t] for t in type_terms) if type_terms else 0.0
                    probability += w * expit(eta + (shift[:, None] if type_terms else 0.0))
                weighted[sl] = probability.mean(axis=1)
                equal[sl] = (probability @ member).mean(axis=1)
            out['equal_participant_observed_opportunities'][f'{g}_{a}'] = equal
            out['card_weighted_observed_opportunities'][f'{g}_{a}'] = weighted
    return out


def summarize(values: np.ndarray, **extra: Any) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {**extra, 'posterior_mean': float(v.mean()), 'posterior_median': float(np.median(v)),
            'posterior_sd': float(v.std(ddof=1)), 'ci_low': float(np.quantile(v, 0.025)),
            'ci_high': float(np.quantile(v, 0.975)), 'posterior_probability_gt_zero': float(np.mean(v > 0)),
            'posterior_draws': int(v.size)}


def estimand_rows(cells: Mapping[str, Mapping[str, np.ndarray]], **extra: Any) -> list[dict[str, Any]]:
    rows = []
    for scope, values in cells.items():
        base = dict(extra, standardization_scope=scope)
        for g in GROUPS:
            for a in AUTHORS:
                rows.append(summarize(values[f'{g}_{a}'], estimand='direct_thread_entry_probability', group=g,
                                      author_type=a, **base))
        for g in GROUPS:
            rows.append(summarize(values[f'{g}_AI'] - values[f'{g}_H'],
                                  estimand='AI_minus_H_direct_thread_entry_probability', group=g,
                                  author_type='AI-H', **base))
        for a in AUTHORS:
            rows.append(summarize(values[f'G2_{a}'] - values[f'G1_{a}'],
                                  estimand='G2_minus_G1_direct_thread_entry_probability', group='G2-G1',
                                  author_type=a, **base))
        rows.append(summarize(values['G2_AI'] - values['G2_H'] - values['G1_AI'] + values['G1_H'],
                              estimand='G2_minus_G1_change_in_AI_minus_H_gap', group='G2-G1',
                              author_type='AI-H', **base))
    return rows


# ---------------------------------------------------------------------------
# Fit stage (public record)
# ---------------------------------------------------------------------------

def load_frame(args: argparse.Namespace, log: RunLog) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from scripts.publication_analysis.rq1b import build_model_frame, panel_signature

    panel = pd.read_parquet(args.panel)
    signature = panel_signature(panel)
    if signature != ADMITTED_PANEL_SHA256:
        raise ValueError(f'panel signature {signature} differs from the admitted panel')
    frame, _ = build_model_frame(panel)
    position = pd.read_csv(args.position)
    persona = persona_lookup(pd.read_csv(args.accounts))
    roots = pd.read_csv(args.root_scores)
    frame, scalers = add_covariates(frame, position, persona, roots)
    log(f'frame {len(frame)} opportunities, {int(frame.direct_thread_entry.sum())} entries, '
        f'{frame.participant_key.nunique()} participants; panel signature matches admitted')
    return frame, scalers


def parameter_names(spec: str) -> list[str]:
    visit = [] if spec in NO_VISIT_SIZE_SPECS else ['beta_log_feed_spell_cards']
    return ['alpha', 'beta_group', 'beta_ai', 'beta_group_ai', 'beta_log_age', *visit,
            *[f'beta_{t}' for t in SPECS[spec]], 'sigma_participant', 'sigma_participant_ai', 'sigma_session',
            'sigma_post', 'sigma_day']


def parameter_table(idata: Any, spec: str) -> pd.DataFrame:
    """Posterior mean, sd, median and equal-tailed 95% interval (2.5%/97.5% quantiles) plus diagnostics."""
    import arviz as az

    names = parameter_names(spec)
    diagnostics = az.summary(idata, var_names=names, kind='diagnostics', round_to='none')
    rows = []
    for name in names:
        v = _draws(idata, name)
        rows.append({'spec': spec, 'parameter': name, 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                     'median': float(np.median(v)), 'eti_2.5%': float(np.quantile(v, 0.025)),
                     'eti_97.5%': float(np.quantile(v, 0.975)), 'posterior_probability_gt_zero': float(np.mean(v > 0)),
                     **{k: float(diagnostics.loc[name, k]) for k in ('mcse_mean', 'mcse_sd', 'ess_bulk', 'ess_tail',
                                                                     'r_hat')},
                     'interval': 'equal-tailed 95% (posterior 2.5% and 97.5% quantiles)'})
    return pd.DataFrame(rows)


def spec_estimands(idata: Any, frame: pd.DataFrame, spec: str, n_atoms: int = N_TOXICITY_ATOMS) -> pd.DataFrame:
    rows = []
    for scheme in ('community', 'pooled'):
        rows += estimand_rows(cell_draws(idata, frame, SPECS[spec], scheme, n_atoms=n_atoms), spec=spec,
                              post_type_scheme=scheme)
    out = pd.DataFrame(rows)
    out['toxicity_atoms'] = n_atoms if 'z_root_toxicity' in SPECS[spec] else 0
    return out


def log_focal(log: RunLog, estimands: pd.DataFrame, label: str) -> None:
    focal = estimands[(estimands.estimand == 'G2_minus_G1_change_in_AI_minus_H_gap')
                      & (estimands.standardization_scope == 'equal_participant_observed_opportunities')]
    log(f'{label} focal ' + '; '.join(f'{r.post_type_scheme}: {100 * r.posterior_mean:.2f} '
                                      f'[{100 * r.ci_low:.2f}, {100 * r.ci_high:.2f}]' for r in focal.itertuples()))


def run_fit(args: argparse.Namespace) -> None:
    import arviz as az
    import pymc as pm
    from scripts.publication_analysis.config import load_config
    from scripts.publication_analysis.rq1b import mcmc_diagnostics

    out = Path(args.out) / args.spec
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(Path(args.out) / 'fit.log')
    started = time.monotonic()
    t0 = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    config = load_config(Path(args.config), run_id=RUN_ID)
    terms = SPECS[args.spec]
    visit_size = args.spec not in NO_VISIT_SIZE_SPECS
    log(f'{RUN_ID}/{ITEM} fit spec={args.spec} terms={terms} visit_size={visit_size} tune={args.tune} '
        f'draws={DRAWS} chains={CHAINS} target_accept={TARGET_ACCEPT} seed={SEED} script_sha256={sha256(__file__)}')
    frame, scalers = load_frame(args, log)
    model = build_model(frame, config.sampling, terms, visit_size=visit_size)
    with model:
        idata = pm.sample(draws=DRAWS, tune=args.tune, chains=CHAINS, cores=CHAINS, target_accept=TARGET_ACCEPT,
                          random_seed=SEED, progressbar=False, return_inferencedata=True)
    sampling_seconds = time.monotonic() - started
    diagnostics = mcmc_diagnostics(idata, role=args.spec).iloc[0].to_dict()
    diagnostics.update(spec=args.spec, tune=args.tune, draws=DRAWS, chains=CHAINS, target_accept=TARGET_ACCEPT,
                       seed=SEED, sampling_seconds=sampling_seconds)
    log(f'diagnostics {diagnostics}')
    idata.to_netcdf(out / 'posterior.nc')
    estimands = spec_estimands(idata, frame, args.spec)
    estimands.to_csv(out / 'estimands.csv', index=False)
    parameter_table(idata, args.spec).to_csv(out / 'parameters.csv', index=False)
    pd.DataFrame([diagnostics]).to_csv(out / 'diagnostics.csv', index=False)
    pd.DataFrame(scalers).to_csv(out / 'scalers.csv', index=False)
    log_focal(log, estimands, args.spec)
    save_json(out / 'fit_record.json', {
        'spec': args.spec, 'terms': list(terms), 'visit_size_term': visit_size, 'started': t0,
        'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'host': socket.gethostname(), 'command': ' '.join(sys.argv),
        'inputs': {name: {'path': str(getattr(args, name)), 'sha256': sha256(getattr(args, name))}
                   for name in ('panel', 'position', 'accounts', 'root_scores', 'config')},
        'diagnostics': diagnostics, 'posterior_sha256': sha256(out / 'posterior.nc'),
        'toxicity_atoms': N_TOXICITY_ATOMS, 'script_sha256': sha256(__file__),
        'elapsed_seconds': time.monotonic() - started, 'peak_rss_gib_main_process': peak_rss_gib(),
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
                     'pymc': pm.__version__, 'arviz': az.__version__, 'platform': platform.platform()},
    })
    log(f'spec {args.spec} done in {time.monotonic() - started:.1f} s')


def run_estimands(args: argparse.Namespace) -> None:
    """Recompute one specification's estimands and parameter table from its saved posterior."""
    import arviz as az

    out = Path(args.out) / args.spec
    log = RunLog(Path(args.out) / 'fit.log')
    started = time.monotonic()
    log(f'{RUN_ID}/{ITEM} estimands spec={args.spec} toxicity_atoms={args.atoms} script_sha256={sha256(__file__)}')
    frame, _ = load_frame(args, log)
    idata = az.from_netcdf(out / 'posterior.nc')
    estimands = spec_estimands(idata, frame, args.spec, n_atoms=args.atoms)
    estimands.to_csv(out / 'estimands.csv', index=False)
    parameter_table(idata, args.spec).to_csv(out / 'parameters.csv', index=False)
    log_focal(log, estimands, f'{args.spec} ({args.atoms} toxicity atoms)')
    save_json(out / 'estimands_record.json', {
        'spec': args.spec, 'toxicity_atoms': args.atoms, 'posterior_sha256': sha256(out / 'posterior.nc'),
        'estimands_sha256': sha256(out / 'estimands.csv'), 'script_sha256': sha256(__file__),
        'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'elapsed_seconds': time.monotonic() - started})


def run_crosscheck(args: argparse.Namespace) -> None:
    """Recompute feed rank from the public session events and compare with the raw-CSV extraction."""
    log = RunLog(Path(args.out) / 'crosscheck.log')
    started = time.monotonic()
    log(f'{RUN_ID}/{ITEM} crosscheck script_sha256={sha256(__file__)}')
    panel = pd.read_parquet(args.panel)
    columns = ['session_id', 'event_seq', 'action', 'id', 'type', 'semantic_group', 'section']
    events = pd.read_parquet(args.events, columns=columns)
    events = events[events.session_id.isin(set(panel.session_id))]
    public = public_feed_ranks(events, panel)
    checks = rank_crosscheck(public, pd.read_csv(args.position), panel)
    checks.update(public_events_in_panel_sessions=int(len(events)),
                  panel_sessions=int(panel.session_id.nunique()),
                  sessions_with_public_events=int(events.session_id.nunique()))
    log(f'crosscheck {checks}')
    record = {'checks': checks, 'passed': checks['rank_equal'] == checks['cards_equal'] == checks['opportunities'],
              'inputs': {name: {'path': str(getattr(args, name)), 'sha256': sha256(getattr(args, name))}
                         for name in ('panel', 'events', 'position')},
              'script_sha256': sha256(__file__), 'elapsed_seconds': time.monotonic() - started,
              'peak_rss_gib': peak_rss_gib(), 'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save_json(Path(args.out) / 'public_rank_crosscheck.json', record)
    if not record['passed']:
        raise RuntimeError(f'public-event ranks differ from the extraction: {checks}')


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def denominator_text(frame: pd.DataFrame) -> str:
    parts = []
    for g in GROUPS:
        sub = frame[frame.group.eq(g)]
        parts.append(f'{g} {len(sub)} opportunities / {int(sub.direct_thread_entry.sum())} entries / '
                     f'{sub.participant_key.nunique()} participants')
    return (f'{len(frame)} participant-feed-visit-root-post opportunities, {int(frame.direct_thread_entry.sum())} '
            f'direct thread entries, {frame.participant_key.nunique()} participants ({"; ".join(parts)}); '
            f'{frame.post_key.nunique()} root posts; admitted panel {SOURCE_RUN}')


def support_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, a), sub in frame.groupby(['group', 'author_type']):
        rows.append(dict(group=g, author_type=a, opportunities=len(sub), entries=int(sub.direct_thread_entry.sum()),
                         observed_entry_rate=float(sub.direct_thread_entry.mean()),
                         participants=sub.participant_key.nunique(), root_posts=sub.post_key.nunique(),
                         feed_rank_median=float(sub.feed_rank.median()),
                         feed_rank_mean=float(sub.feed_rank.mean()),
                         feed_rank_p90=float(sub.feed_rank.quantile(0.9)),
                         share_feed_rank_1=float(sub.feed_rank.eq(1).mean()),
                         feed_visit_cards_median=float(sub.feed_visit_cards.median()),
                         feed_rank_prefinal_median=float(sub.feed_rank_prefinal.median()),
                         eligible_rank_median=float(sub.feed_rank_eligible.median()),
                         share_toxic_persona=float(sub.toxic_persona.mean()),
                         root_toxicity_median=float(sub.root_toxicity_probability.median()),
                         share_root_toxicity_above_0_5=float(sub.root_toxicity_probability.gt(0.5).mean())))
    for (g, a, persona), sub in frame[frame.author_type.eq('AI')].groupby(['group', 'author_type',
                                                                          'toxic_persona']):
        rows.append(dict(group=g, author_type=f'AI_toxic_persona_{int(persona)}', opportunities=len(sub),
                         entries=int(sub.direct_thread_entry.sum()),
                         observed_entry_rate=float(sub.direct_thread_entry.mean()),
                         participants=sub.participant_key.nunique(), root_posts=sub.post_key.nunique(),
                         feed_rank_median=float(sub.feed_rank.median()), feed_rank_mean=float(sub.feed_rank.mean()),
                         share_toxic_persona=float(persona)))
    entered = frame.groupby(pd.cut(frame.feed_rank, [0, 1, 3, 10, 30, np.inf],
                                   labels=['1', '2-3', '4-10', '11-30', '>30']), observed=True)
    for label, sub in entered:
        rows.append(dict(group='all', author_type=f'feed_rank_{label}', opportunities=len(sub),
                         entries=int(sub.direct_thread_entry.sum()),
                         observed_entry_rate=float(sub.direct_thread_entry.mean()),
                         participants=sub.participant_key.nunique(), root_posts=sub.post_key.nunique()))
    from_end = (frame.feed_visit_cards - frame.feed_rank).astype(int)
    labels = np.select([from_end.eq(0), from_end.eq(1)], ['last', 'second_to_last'], 'earlier')
    for label in ('last', 'second_to_last', 'earlier'):
        sub = frame[labels == label]
        rows.append(dict(group='all', author_type=f'visit_position_{label}', opportunities=len(sub),
                         entries=int(sub.direct_thread_entry.sum()),
                         observed_entry_rate=float(sub.direct_thread_entry.mean()),
                         participants=sub.participant_key.nunique(), root_posts=sub.post_key.nunique()))
    return pd.DataFrame(rows)


def run_tables(args: argparse.Namespace) -> None:
    import arviz as az

    log = RunLog(Path(args.out) / 'fit.log')
    frame, _ = load_frame(args, log)
    denominator = denominator_text(frame)
    out_dir = Path(args.results)
    estimands, params, diagnostics = [], [], []
    for spec in SPECS:
        directory = Path(args.out) / spec
        if not (directory / 'estimands.csv').exists():
            log(f'spec {spec} missing; skipped')
            continue
        spec_est = pd.read_csv(directory / 'estimands.csv')
        if 'toxicity_atoms' not in spec_est:
            spec_est['toxicity_atoms'] = 0
        if 'z_root_toxicity' in SPECS[spec] and not spec_est.toxicity_atoms.eq(N_TOXICITY_ATOMS).all():
            raise ValueError(f'{spec}: estimands not computed with {N_TOXICITY_ATOMS} toxicity atoms; run estimands')
        estimands.append(spec_est)
        params.append(parameter_table(az.from_netcdf(directory / 'posterior.nc'), spec))
        diagnostics.append(pd.read_csv(directory / 'diagnostics.csv'))
    est = pd.concat(estimands, ignore_index=True)
    est['terms_added'] = est.spec.map(lambda s: ';'.join(SPECS[s]) or 'none')
    est['visit_size_term'] = ~est.spec.isin(NO_VISIT_SIZE_SPECS)
    est['interval'] = 'equal-tailed 95% (posterior 2.5% and 97.5% quantiles)'
    est['sign_convention'] = SIGN
    est['admitted_focal_mean'], est['admitted_focal_ci_low'], est['admitted_focal_ci_high'] = ADMITTED_FOCAL
    focal_order = ['run_id', 'spec', 'terms_added', 'post_type_scheme', 'standardization_scope', 'estimand', 'group',
                   'author_type']
    est.insert(0, 'run_id', RUN_ID)
    est['opportunities'] = len(frame)
    est['entries'] = int(frame.direct_thread_entry.sum())
    est['participants'] = frame.participant_key.nunique()
    est['denominator'] = denominator
    est = est[focal_order + [c for c in est.columns if c not in focal_order]]
    par = pd.concat(params, ignore_index=True)
    par.insert(0, 'run_id', RUN_ID)
    par['sign_convention'] = ('coefficients on the logit scale; positive values raise the direct-entry '
                              'probability; beta_group is G2 (hidden) minus G1 (visible) for participant posts')
    par['denominator'] = denominator
    diag = pd.concat(diagnostics, ignore_index=True)
    diag.insert(0, 'run_id', RUN_ID)
    diag['denominator'] = denominator
    sup = support_table(frame)
    sup.insert(0, 'run_id', RUN_ID)
    sup['denominator'] = denominator
    prefix = f'chb_prereview_{ITEM}'
    written = {}
    for name, table in (('estimands', est), ('parameters', par), ('diagnostics', diag), ('support', sup)):
        path = out_dir / f'{prefix}_{name}.csv'
        table.to_csv(path, index=False)
        written[str(path)] = sha256(path)
        log(f'wrote {path} ({len(table)} rows)')
    save_json(Path(args.out) / 'tables_sha256.json', written)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='stage', required=True)
    ex = sub.add_parser('extract')
    ex.add_argument('--panel', required=True)
    ex.add_argument('--sessions-dir', required=True)
    ex.add_argument('--public-scores', required=True)
    ex.add_argument('--contents', required=True)
    ex.add_argument('--out', required=True)
    cc = sub.add_parser('crosscheck')
    cc.add_argument('--panel', required=True)
    cc.add_argument('--events', required=True)
    cc.add_argument('--position', required=True)
    cc.add_argument('--out', required=True)
    for name in ('fit', 'estimands', 'tables'):
        p = sub.add_parser(name)
        p.add_argument('--panel', required=True)
        p.add_argument('--position', required=True)
        p.add_argument('--accounts', required=True)
        p.add_argument('--root-scores', required=True)
        p.add_argument('--config', default='analysis.yaml')
        p.add_argument('--out', required=True)
        if name == 'fit':
            p.add_argument('--spec', choices=list(SPECS), required=True)
            p.add_argument('--tune', type=int, default=TUNE)
        elif name == 'estimands':
            p.add_argument('--spec', choices=list(SPECS), required=True)
            p.add_argument('--atoms', type=int, default=N_TOXICITY_ATOMS)
        else:
            p.add_argument('--results', default='runs/chb_prereview_20261110/tables')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    {'extract': run_extract, 'crosscheck': run_crosscheck, 'fit': run_fit, 'estimands': run_estimands,
     'tables': run_tables}[args.stage](args)


if __name__ == '__main__':
    main()
