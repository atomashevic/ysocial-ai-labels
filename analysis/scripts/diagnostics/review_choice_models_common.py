"""Shared helpers for the review_choice_models diagnostics and sensitivity fits.

The helpers evaluate the published conditional-choice linear predictor on the
valid (unpadded) candidate rows of a ``PreparedModelData`` object, so posterior
projections can run in bounded memory. ``check_against_original`` compares
them with ``scripts.publication_analysis.model`` before any result is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def limit_threads(threads: int = 1) -> None:
    """Set BLAS thread variables; call before importing numpy."""
    for name in THREAD_VARIABLES:
        os.environ[name] = str(int(threads))


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class RunLog:
    """Append timestamped lines to a run log and echo them."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        print(line, flush=True)
        with self.path.open('a') as stream:
            stream.write(line + '\n')


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def summarize(values: Any, **extra: Any) -> dict[str, Any]:
    import numpy as np
    clean = np.asarray(values, dtype=float).ravel()
    row = {
        'posterior_mean': float(clean.mean()),
        'posterior_median': float(np.median(clean)),
        'posterior_sd': float(clean.std(ddof=1)) if clean.size > 1 else float('nan'),
        'ci_low': float(np.quantile(clean, 0.025)),
        'ci_high': float(np.quantile(clean, 0.975)),
        'posterior_probability_gt_zero': float(np.mean(clean > 0)),
        'posterior_draws': int(clean.size),
    }
    row.update(extra)
    return row


class Posterior:
    """Flattened (chain*draw) posterior arrays aligned to a prepared data object."""

    def __init__(self, idata: Any, data: Any):
        import numpy as np
        post = idata.posterior
        self.chains = int(post.sizes['chain'])
        self.draws = int(post.sizes['draw'])
        terms = tuple(str(v) for v in post['beta'].coords['term'].values)
        missing = sorted(set(terms) - set(data.terms))
        if missing:
            raise ValueError(f'Posterior terms absent from prepared data: {missing}')
        raw = post['beta'].stack(sample=('chain', 'draw')).transpose('sample', 'term').values
        self.beta = np.zeros((raw.shape[0], len(data.terms)))
        for index, term in enumerate(terms):
            self.beta[:, data.terms.index(term)] = raw[:, index]
        self.posterior_terms = terms
        self.n = raw.shape[0]

        def flat(name: str, dims: Sequence[str]):
            if name not in post:
                return None
            return post[name].stack(sample=('chain', 'draw')).transpose('sample', *dims).values

        self.receiver = flat('receiver_effect', ('receiver',))
        self.actor = flat('actor_ai_effect', ('actor',))
        self.topic = flat('topic_ai_effect', ('topic',))
        if self.receiver is not None and self.receiver.shape[1] != len(data.receiver_labels):
            raise ValueError('Receiver effects do not match the prepared receivers')
        if self.actor is not None and self.actor.shape[1] != len(data.actor_labels):
            raise ValueError('Participant AI slopes do not match the prepared participants')
        if self.topic is not None and self.topic.shape[1] != len(data.topic_labels):
            raise ValueError('Topic AI slopes do not match the prepared topics')
        if 'receiver_block' in post:
            raise ValueError('Latent-block posteriors are not supported here')

    def chunks(self, size: int, limit: int | None = None) -> Iterable[slice]:
        total = self.n if limit is None else min(self.n, limit)
        for start in range(0, total, size):
            yield slice(start, min(total, start + size))


class Ragged:
    """Valid candidate rows of a padded PreparedModelData object."""

    def __init__(self, data: Any):
        import numpy as np
        self.data = data
        self.choice, self.position = np.nonzero(data.mask)
        self.n_rows = len(self.choice)
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.choice) != 0])
        if len(self.starts) != data.n_choices:
            raise ValueError('Every choice set needs at least one valid candidate')
        self.ai = data.candidate_is_ai[self.choice, self.position].astype(float)
        self.receiver = data.receiver_index[self.choice, self.position]
        self.actor = data.actor_index[self.choice]
        self.topic = data.topic_index[self.choice]
        self.chosen = (self.position == data.chosen_position[self.choice])
        self.group = np.asarray(data.choice_groups, dtype=object)

    def rows(self, frame: Any) -> Any:
        return frame[self.choice, self.position]

    def random_part(self, posterior: Posterior, draws: slice) -> Any:
        import numpy as np
        out = np.zeros((draws.stop - draws.start, self.n_rows))
        if posterior.receiver is not None:
            out += posterior.receiver[draws][:, self.receiver]
        if posterior.actor is not None:
            out += posterior.actor[draws][:, self.actor] * self.ai
        if posterior.topic is not None:
            out += posterior.topic[draws][:, self.topic] * self.ai
        return out

    def utility(self, posterior: Posterior, draws: slice, x_rows: Any, random: Any | None = None) -> Any:
        base = random if random is not None else self.random_part(posterior, draws)
        return posterior.beta[draws] @ x_rows.T + base

    def log_normalizer(self, utility: Any) -> Any:
        import numpy as np
        top = np.maximum.reduceat(utility, self.starts, axis=1)
        weights = np.exp(utility - top[:, self.choice])
        return np.log(np.add.reduceat(weights, self.starts, axis=1)) + top

    def probabilities(self, utility: Any) -> Any:
        import numpy as np
        return np.exp(utility - self.log_normalizer(utility)[:, self.choice])

    def ai_probability(self, utility: Any) -> Any:
        import numpy as np
        return np.add.reduceat(self.probabilities(utility) * self.ai, self.starts, axis=1)


def check_against_original(idata: Any, data: Any, frames: Mapping[str, Any], *, draws: int = 40,
                           tolerance: float = 1e-10) -> dict[str, float]:
    """Compare ragged AI probabilities with model._posterior_utilities on a draw subset."""
    import numpy as np
    from scripts.publication_analysis.model import _ai_choice_probability, _posterior_utilities
    subset = idata.isel(chain=slice(0, 1), draw=slice(0, draws))
    posterior = Posterior(subset, data)
    ragged = Ragged(data)
    result = {}
    for name, frame in frames.items():
        original = _ai_choice_probability(_posterior_utilities(subset, data, x_override=frame), data)
        mine = ragged.ai_probability(ragged.utility(posterior, slice(0, posterior.n), ragged.rows(frame)))
        difference = float(np.max(np.abs(original - mine)))
        if difference > tolerance:
            raise RuntimeError(f'Ragged projection differs from the published code for {name}: {difference}')
        result[name] = difference
    return result


def arm_frames(data: Any, toggled: Iterable[str]) -> tuple[Any, Any]:
    """G1/G2 frames that switch only the named G2 interaction terms.

    Toggling every ``g2_x_*`` term reproduces ``model._common_arm_frames`` (the
    published Primary projection); toggling only ``g2_x_candidate_ai``
    reproduces ``model._direct_ai_arm_frames`` (Authorship term only).
    """
    g1 = data.x.copy()
    g2 = data.x.copy()
    for term in toggled:
        index = data.terms.index(term)
        g1[:, :, index] = 0.0
        if term == 'g2_x_candidate_ai':
            g2[:, :, index] = data.candidate_is_ai
        else:
            base = term.removeprefix('g2_x_')
            g2[:, :, index] = data.x[:, :, data.terms.index(base)]
    return g1, g2


def account_layout(data: Any, ragged: Ragged, author_type: str = 'AI') -> dict[str, Any]:
    """Choice-by-account opportunities with at least one competing account.

    Mirrors rq2._projection_from_utility: every distinct account of the
    requested type in a choice set is one opportunity when the set contains a
    candidate from another account. All of that account's candidates are focal.
    """
    import numpy as np
    want = 1.0 if author_type == 'AI' else 0.0
    focal, other, focal_starts, other_starts = [], [], [], []
    meta = []
    for choice in range(data.n_choices):
        rows = np.arange(ragged.starts[choice],
                         ragged.starts[choice + 1] if choice + 1 < data.n_choices else ragged.n_rows)
        receivers = ragged.receiver[rows]
        typed = rows[ragged.ai[rows] == want]
        for receiver in np.unique(ragged.receiver[typed]):
            focal_rows = rows[receivers == receiver]
            other_rows = rows[receivers != receiver]
            if not len(other_rows):
                continue
            focal_starts.append(len(focal))
            other_starts.append(len(other))
            focal.extend(focal_rows)
            other.extend(other_rows)
            meta.append((choice, int(receiver), int(ragged.actor[rows[0]]), ragged.group[choice],
                         bool(ragged.chosen[focal_rows].any()), len(focal_rows)))
    return {
        'focal': np.asarray(focal, dtype=np.intp), 'other': np.asarray(other, dtype=np.intp),
        'focal_starts': np.asarray(focal_starts, dtype=np.intp),
        'other_starts': np.asarray(other_starts, dtype=np.intp),
        'choice': np.asarray([m[0] for m in meta]), 'receiver': np.asarray([m[1] for m in meta]),
        'actor': np.asarray([m[2] for m in meta]), 'group': np.asarray([m[3] for m in meta], dtype=object),
        'selected': np.asarray([m[4] for m in meta]), 'focal_count': np.asarray([m[5] for m in meta]),
    }


def account_changes(utility: Any, beta: Any, x_rows: Any, layout: Mapping[str, Any],
                    term_indexes: Sequence[int], low: Sequence[float], high: Sequence[float]) -> Any:
    """Per-opportunity change in account selection probability (draws x opportunities).

    The account probability is the sum of its candidates' softmax probabilities,
    expit(logsumexp(focal) - logsumexp(other)), as in rq2._projection_from_utility.
    """
    import numpy as np
    from scipy.special import expit
    focal = layout['focal']
    observed = x_rows[focal][:, term_indexes]
    coef = beta[:, term_indexes]
    low_delta = (np.asarray(low)[None, :] - observed) @ coef.T
    high_delta = (np.asarray(high)[None, :] - observed) @ coef.T
    log_other = np.logaddexp.reduceat(utility[:, layout['other']], layout['other_starts'], axis=1)
    base = utility[:, focal]
    low_mass = np.logaddexp.reduceat(base + low_delta.T, layout['focal_starts'], axis=1)
    high_mass = np.logaddexp.reduceat(base + high_delta.T, layout['focal_starts'], axis=1)
    return expit(high_mass - log_other) - expit(low_mass - log_other)


def weighted_mean(changes: Any, keys: Any | None) -> Any:
    """Average opportunities equally (keys None) or within keys, then across keys."""
    import numpy as np
    import pandas as pd
    if keys is None:
        return changes.mean(axis=1)
    codes, levels = pd.factorize(pd.Series(keys), sort=True)
    counts = np.bincount(codes, minlength=len(levels)).astype(float)
    weights = 1.0 / (counts[codes] * len(levels))
    return changes @ weights


def save_json(path: Path | str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


# ---------------------------------------------------------------------------
# Inputs and fitting shared by the sensitivity scripts
# ---------------------------------------------------------------------------

def load_analysis_config(path: Path | str, run_id: str, profile: str = 'release') -> Any:
    from scripts.publication_analysis.config import load_config
    return load_config(Path(path), profile=profile, run_id=run_id)


def build_network_from_config(config: Any, log: RunLog) -> tuple[Any, Any]:
    """Rebuild the canonical choice rows and strictly prior network history.

    Uses the same functions as ``prepare-choice``/``prepare-rq2`` but skips the
    provenance tables, so it runs on the public-input layout without the
    private source manifest. Returns (event_history, network_history).
    """
    from scripts.publication_analysis.data import (build_event_history, build_network_history,
                                                   build_thread_choice_sets)
    events = build_event_history(config)
    choice_sets = build_thread_choice_sets(config, event_history=events)
    choice_sets.attrs.pop('attrition_events', None)
    network = build_network_history(choice_sets, events)
    log(f'rebuilt {network.choice_set_id.nunique()} replies and {len(network)} candidate rows '
        f'from {config.processed_dir}')
    return events, network


def check_cohort(frame: Any, replies: int = 2143, candidates: int = 17489) -> None:
    if frame.choice_set_id.nunique() != replies or len(frame) != candidates:
        raise RuntimeError(f'Expected {replies} replies and {candidates} candidates; got '
                           f'{frame.choice_set_id.nunique()} and {len(frame)}')


def compare_reference_frame(frame: Any, reference: Any, columns: Sequence[str]) -> dict[str, float]:
    """Maximum absolute difference of model columns against an admitted frame."""
    import numpy as np
    keys = ['choice_set_id', 'candidate_content_id']
    left = frame[keys + list(columns)].copy()
    right = reference[keys + list(columns)].copy()
    for part in (left, right):
        part['candidate_content_id'] = part['candidate_content_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    merged = left.merge(right, on=keys, suffixes=('', '_ref'), validate='one_to_one')
    if len(merged) != len(reference):
        raise RuntimeError('Rebuilt frame does not cover the reference frame')
    return {c: float(np.max(np.abs(merged[c].astype(float) - merged[c + '_ref'].astype(float)))) for c in columns}


def fit_spec(data: Any, config: Any, spec: Any, *, seed: int, draws: int, tune: int, chains: int,
             cores: int, target_accept: float | None = None) -> Any:
    from scripts.publication_analysis.model import _build_model, fit_model
    model = _build_model(data, config, spec)
    return fit_model(model, config, draws=draws, tune=tune, chains=chains, cores=min(cores, chains),
                     target_accept=target_accept, random_seed=seed, progressbar=False)


def mcmc_check(idata: Any) -> dict[str, Any]:
    """Unrounded R-hat/ESS/divergence check over sampled parameters."""
    import arviz as az
    import numpy as np
    names = [n for n in ('beta', 'receiver_effect', 'actor_ai_effect', 'topic_ai_effect', 'sigma_receiver_effect',
                         'sigma_actor_ai_effect', 'sigma_topic_ai_effect') if n in idata.posterior]
    summary = az.summary(idata, var_names=names, kind='diagnostics', round_to='none')
    divergences = int(idata.sample_stats['diverging'].sum())
    result = dict(max_rhat=float(np.nanmax(summary.r_hat)), min_ess_bulk=float(np.nanmin(summary.ess_bulk)),
                  min_ess_tail=float(np.nanmin(summary.ess_tail)), divergences=divergences,
                  max_tree_depth=int(idata.sample_stats['tree_depth'].max()),
                  sampling_time_seconds=float(idata.posterior.attrs.get('sampling_time', float('nan'))))
    result['passed'] = bool(result['max_rhat'] < 1.01 and result['min_ess_bulk'] > 400
                            and result['min_ess_tail'] > 400 and divergences == 0)
    return result


def rq1_projection_rows(idata: Any, data: Any, *, chunk: int = 200, label: str = '') -> list[dict[str, Any]]:
    """Primary (all G2 terms) and Authorship-term-only projections on pooled sets."""
    import numpy as np
    g2_terms = [t for t in data.terms if t.startswith('g2_x_') and
                (t == 'g2_x_candidate_ai' or t.removeprefix('g2_x_') in data.terms)]
    scenarios = {'primary_all_G2_terms': g2_terms, 'authorship_term_only': ['g2_x_candidate_ai']}
    posterior = Posterior(idata, data)
    ragged = Ragged(data)
    frames = {k: tuple(ragged.rows(f) for f in arm_frames(data, v)) for k, v in scenarios.items()}
    store = {k: (np.empty(posterior.n), np.empty(posterior.n)) for k in scenarios}
    for draws in posterior.chunks(chunk):
        random = ragged.random_part(posterior, draws)
        for key, (g1, g2) in frames.items():
            store[key][0][draws] = ragged.ai_probability(ragged.utility(posterior, draws, g1, random)).mean(axis=1)
            store[key][1][draws] = ragged.ai_probability(ragged.utility(posterior, draws, g2, random)).mean(axis=1)
    rows = []
    for key, (g1, g2) in store.items():
        for group, values in (('G1', g1), ('G2', g2), ('G2-G1', g2 - g1)):
            rows.append(summarize(values, model=label, projection=key, group=group,
                                  toggled_terms=';'.join(scenarios[key]), replies=data.n_choices,
                                  candidates=int(data.mask.sum())))
    return rows


def beta_rows(idata: Any, label: str = '') -> list[dict[str, Any]]:
    beta = idata.posterior['beta']
    return [summarize(beta.sel(term=t).values, model=label, parameter=f'beta[{t}]')
            for t in beta.coords['term'].values]


def guard_output(path: Path) -> None:
    if (path / 'posterior.nc').exists() or any(path.glob('*/posterior.nc')):
        raise FileExistsError(f'{path} already holds a posterior; choose a new --out')


def fit_with_extension(data: Any, config: Any, spec: Any, *, seed: int, draws: int, tune: int, chains: int,
                       cores: int, log: RunLog, extend: bool = True) -> tuple[Any, dict[str, Any]]:
    """Fit once; if checks fail, refit once with doubled warm-up and draws (reported policy)."""
    started = time.monotonic()
    idata = fit_spec(data, config, spec, seed=seed, draws=draws, tune=tune, chains=chains, cores=cores)
    checks = mcmc_check(idata)
    checks['extended'] = False
    if not checks['passed'] and extend:
        log(f'checks failed {checks}; refitting with doubled warm-up and draws')
        idata = fit_spec(data, config, spec, seed=seed, draws=2 * draws, tune=2 * tune, chains=chains, cores=cores)
        checks = mcmc_check(idata)
        checks['extended'] = True
    checks['fit_seconds'] = time.monotonic() - started
    return idata, checks
