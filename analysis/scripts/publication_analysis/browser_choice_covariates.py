"""Browser-derived covariates for matched RQ1 conditional-choice models.

The functions in this module operate on already reconstructed DOM episodes.
They use the backend reply timestamp as a strict cutoff and preserve every
candidate in each retained M1 choice set.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ("choice_set_id", "group", "event_id", "candidate_content_id")
RAW_BROWSER_COLUMNS = (
    "candidate_visit_count",
    "candidate_dom_presence_seconds",
    "same_author_other_visit_count",
    "same_author_other_dom_presence_seconds",
)
TERMINAL_SENSITIVITY_COLUMNS = (
    "candidate_dom_presence_seconds_excluding_terminal",
    "same_author_other_dom_presence_seconds_excluding_terminal",
)
BROWSER_TERM_MAP = {
    "candidate_visit_count": "z_log1p_candidate_visit_count",
    "candidate_dom_presence_seconds": "z_log1p_candidate_dom_presence_seconds",
    "same_author_other_visit_count": "z_log1p_same_author_other_visit_count",
    "same_author_other_dom_presence_seconds": "z_log1p_same_author_other_dom_presence_seconds",
}
TERMINAL_SENSITIVITY_TERM_MAP = {
    "candidate_dom_presence_seconds_excluding_terminal": (
        "z_log1p_candidate_dom_presence_seconds_excluding_terminal"
    ),
    "same_author_other_dom_presence_seconds_excluding_terminal": (
        "z_log1p_same_author_other_dom_presence_seconds_excluding_terminal"
    ),
}
OBSERVED_INTERVAL_QUALITIES = {
    "observed_zero",
    "closed_exact",
    "closed_audited",
    "closed_unverified",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _id_key(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else format(numeric, ".17g")
    text = str(value).strip()
    return text or None


def _truth(value: Any, *, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def union_half_open_intervals(
    intervals: Iterable[tuple[Any, Any]],
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Return sorted union components, merging overlapping or touching intervals."""

    parsed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw_start, raw_end in intervals:
        start = _utc_timestamp(raw_start)
        end = _utc_timestamp(raw_end)
        if start is None or end is None:
            raise ValueError("DOM intervals require finite timestamps")
        if end < start:
            raise ValueError("DOM interval end precedes its start")
        parsed.append((start, end))
    parsed.sort(key=lambda value: (value[0], value[1]))
    merged: list[list[pd.Timestamp]] = []
    for start, end in parsed:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _episode_quality(episodes: Sequence[Mapping[str, Any]]) -> str:
    if not episodes:
        return "observed_zero"
    exact = [_truth(episode.get("interval_exact")) for episode in episodes]
    audited = [_truth(episode.get("interval_audit_approved")) for episode in episodes]
    if all(exact):
        return "closed_exact"
    if all(is_exact or is_audited for is_exact, is_audited in zip(exact, audited)):
        return "closed_audited"
    return "closed_unverified"


def summarize_closed_episodes(
    episodes: Sequence[Mapping[str, Any]],
    cutoff: Any,
) -> dict[str, Any]:
    """Summarize closed episodes beginning strictly before a backend cutoff."""

    cutoff_time = _utc_timestamp(cutoff)
    if cutoff_time is None:
        return {
            "visit_count": pd.NA,
            "seconds": pd.NA,
            "seconds_excluding_terminal": pd.NA,
            "quality": "invalid_reply_timestamp",
        }
    prior: list[Mapping[str, Any]] = []
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for episode in episodes:
        start = _utc_timestamp(episode.get("impression_start_utc"))
        if start is None:
            return {
                "visit_count": pd.NA,
                "seconds": pd.NA,
                "seconds_excluding_terminal": pd.NA,
                "quality": "unresolved_interval",
            }
        if start >= cutoff_time:
            continue
        prior.append(episode)
        if _truth(episode.get("right_censored"), default=True):
            return {
                "visit_count": pd.NA,
                "seconds": pd.NA,
                "seconds_excluding_terminal": pd.NA,
                "quality": "unresolved_interval",
            }
        end = _utc_timestamp(episode.get("impression_end_utc"))
        if end is None or end < start:
            return {
                "visit_count": pd.NA,
                "seconds": pd.NA,
                "seconds_excluding_terminal": pd.NA,
                "quality": "unresolved_interval",
            }
        intervals.append((start, min(end, cutoff_time)))
    if not prior:
        return {
            "visit_count": 0.0,
            "seconds": 0.0,
            "seconds_excluding_terminal": 0.0,
            "quality": "observed_zero",
        }
    components = union_half_open_intervals(intervals)
    seconds = sum((end - start).total_seconds() for start, end in components)
    seconds_excluding_terminal = sum(
        (end - start).total_seconds() for start, end in components if end < cutoff_time
    )
    return {
        "visit_count": float(len(components)),
        "seconds": float(seconds),
        "seconds_excluding_terminal": float(seconds_excluding_terminal),
        "quality": _episode_quality(prior),
    }


def _action_lookup(linked_actions: pd.DataFrame) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    if linked_actions is None or linked_actions.empty:
        return lookup
    for row in linked_actions.to_dict("records"):
        group = _id_key(row.get("group"))
        event = _id_key(row.get("event_id"))
        if group is None or event is None:
            continue
        key = (group, event)
        if key in lookup:
            raise ValueError(f"Browser action linkage is not unique for {key}")
        lookup[key] = row
    return lookup


def _session_episode_index(
    episodes: pd.DataFrame,
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    if episodes is None or episodes.empty:
        return index
    for row in episodes.to_dict("records"):
        session = _id_key(row.get("session_id"))
        group = _id_key(row.get("group"))
        if session is not None and group is not None:
            index[(session, group)].append(row)
    return index


def _session_status(action: Mapping[str, Any] | None) -> str:
    if action is None or str(action.get("observed_status")) != "observed":
        return "unresolved_session"
    if _truth(action.get("backend_duplicate_conflict"), default=True):
        return "unresolved_session"
    if _truth(action.get("session_ambiguous"), default=True):
        return "unresolved_session"
    if _id_key(action.get("session_id")) is None:
        return "unresolved_session"
    return "observed"


def build_browser_choice_covariates(
    choice_sets: pd.DataFrame,
    episodes: pd.DataFrame,
    linked_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Build nullable browser covariates for every original candidate row."""

    required = set(KEY_COLUMNS) | {
        "event_timestamp",
        "candidate_author_id",
        "chosen",
    }
    missing = sorted(required.difference(choice_sets.columns))
    if missing:
        raise ValueError(f"Choice rows are missing browser-covariate fields: {missing}")
    if choice_sets.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Choice rows require unique candidate keys")

    actions = _action_lookup(linked_actions)
    session_episodes = _session_episode_index(episodes)
    rows: list[dict[str, Any]] = []

    for choice in choice_sets.to_dict("records"):
        group = _id_key(choice.get("group"))
        event = _id_key(choice.get("event_id"))
        candidate = _id_key(choice.get("candidate_content_id"))
        author = _id_key(choice.get("candidate_author_id"))
        action = actions.get((group, event)) if group is not None and event is not None else None
        session_status = _session_status(action)
        session_id = _id_key(action.get("session_id")) if action is not None else None
        cutoff = _utc_timestamp(choice.get("event_timestamp"))
        base = {column: choice.get(column) for column in KEY_COLUMNS}
        base.update(
            {
                "browser_session_id": action.get("session_id", pd.NA) if action is not None else pd.NA,
                "backend_reply_timestamp_utc": cutoff if cutoff is not None else pd.NaT,
                "browser_covariate_session_status": session_status,
                "browser_link_ambiguous": (
                    _truth(action.get("link_ambiguous")) if action is not None else pd.NA
                ),
            }
        )
        if cutoff is None:
            base.update(
                {
                    "browser_covariate_session_status": "invalid_reply_timestamp",
                    "candidate_visit_count": pd.NA,
                    "candidate_dom_presence_seconds": pd.NA,
                    "candidate_dom_presence_seconds_excluding_terminal": pd.NA,
                    "candidate_interval_quality": "invalid_reply_timestamp",
                    "same_author_other_visit_count": pd.NA,
                    "same_author_other_dom_presence_seconds": pd.NA,
                    "same_author_other_dom_presence_seconds_excluding_terminal": pd.NA,
                    "same_author_interval_quality": "invalid_reply_timestamp",
                }
            )
            rows.append(base)
            continue
        if session_status != "observed" or session_id is None or group is None:
            base.update(
                {
                    "candidate_visit_count": pd.NA,
                    "candidate_dom_presence_seconds": pd.NA,
                    "candidate_dom_presence_seconds_excluding_terminal": pd.NA,
                    "candidate_interval_quality": "unresolved_session",
                    "same_author_other_visit_count": pd.NA,
                    "same_author_other_dom_presence_seconds": pd.NA,
                    "same_author_other_dom_presence_seconds_excluding_terminal": pd.NA,
                    "same_author_interval_quality": "unresolved_session",
                }
            )
            rows.append(base)
            continue

        available = session_episodes.get((session_id, group), [])
        candidate_rows = [
            episode for episode in available if _id_key(episode.get("item_id")) == candidate
        ]
        candidate_summary = summarize_closed_episodes(candidate_rows, cutoff)

        missing_episode_author = any(
            _utc_timestamp(episode.get("impression_start_utc")) is not None
            and _utc_timestamp(episode.get("impression_start_utc")) < cutoff
            and _id_key(episode.get("author_id")) is None
            for episode in available
        )
        if author is None or missing_episode_author:
            author_summary = {
                "visit_count": pd.NA,
                "seconds": pd.NA,
                "seconds_excluding_terminal": pd.NA,
                "quality": "missing_author_metadata",
            }
        else:
            author_rows = [
                episode
                for episode in available
                if _id_key(episode.get("author_id")) == author
                and _id_key(episode.get("item_id")) != candidate
            ]
            author_summary = summarize_closed_episodes(author_rows, cutoff)

        base.update(
            {
                "candidate_visit_count": candidate_summary["visit_count"],
                "candidate_dom_presence_seconds": candidate_summary["seconds"],
                "candidate_dom_presence_seconds_excluding_terminal": candidate_summary[
                    "seconds_excluding_terminal"
                ],
                "candidate_interval_quality": candidate_summary["quality"],
                "same_author_other_visit_count": author_summary["visit_count"],
                "same_author_other_dom_presence_seconds": author_summary["seconds"],
                "same_author_other_dom_presence_seconds_excluding_terminal": author_summary[
                    "seconds_excluding_terminal"
                ],
                "same_author_interval_quality": author_summary["quality"],
            }
        )
        rows.append(base)

    result = pd.DataFrame(rows)
    raw_complete = result[list(RAW_BROWSER_COLUMNS)].notna().all(axis=1)
    quality_complete = result["candidate_interval_quality"].isin(OBSERVED_INTERVAL_QUALITIES) & result[
        "same_author_interval_quality"
    ].isin(OBSERVED_INTERVAL_QUALITIES)
    result["browser_covariate_row_complete"] = (
        result["browser_covariate_session_status"].eq("observed")
        & raw_complete
        & quality_complete
    )
    complete_choices = result.groupby("choice_set_id", sort=False)[
        "browser_covariate_row_complete"
    ].transform("all")
    result["browser_choice_set_complete"] = complete_choices.astype(bool)
    verified_qualities = {"observed_zero", "closed_exact", "closed_audited"}
    result["browser_verified_row_complete"] = (
        result["browser_covariate_row_complete"]
        & result["candidate_interval_quality"].isin(verified_qualities)
        & result["same_author_interval_quality"].isin(verified_qualities)
    )
    result["browser_verified_choice_set_complete"] = result.groupby(
        "choice_set_id", sort=False
    )["browser_verified_row_complete"].transform("all").astype(bool)
    return result


def matched_choice_set_ids(covariates: pd.DataFrame) -> tuple[str, ...]:
    """Return complete choice-set IDs while rejecting row-level filtering."""

    required = {"choice_set_id", "browser_choice_set_complete"}
    missing = sorted(required.difference(covariates.columns))
    if missing:
        raise ValueError(f"Browser covariates are missing cohort fields: {missing}")
    mixed = covariates.groupby("choice_set_id")["browser_choice_set_complete"].nunique()
    if mixed.gt(1).any():
        raise ValueError("Browser eligibility must be constant within each choice set")
    selected = covariates.loc[
        covariates["browser_choice_set_complete"].astype(bool), "choice_set_id"
    ]
    return tuple(str(value) for value in pd.unique(selected))


def add_standardized_browser_terms(
    matched_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply pooled matched-cohort log1p standardization to browser terms."""

    out = matched_frame.copy()
    manifest_rows: list[dict[str, Any]] = []
    for raw_column, term in BROWSER_TERM_MAP.items():
        if raw_column not in out.columns:
            raise ValueError(f"Matched frame is missing {raw_column}")
        raw = pd.to_numeric(out[raw_column], errors="coerce")
        if raw.isna().any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
            raise ValueError(f"{raw_column} must be finite on the matched cohort")
        if raw.lt(0).any():
            raise ValueError(f"{raw_column} must be non-negative")
        transformed = np.log1p(raw.astype(float))
        mean = float(transformed.mean())
        standard_deviation = float(transformed.std(ddof=0))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"{raw_column} has no variation in the matched cohort")
        out[term] = (transformed - mean) / standard_deviation
        manifest_rows.append(
            {
                "term": term,
                "family": "opportunity_content",
                "source": raw_column,
                "description": f"Matched-cohort z score of log1p({raw_column}).",
                "focal": False,
                "standardized": True,
                "raw_mean": mean,
                "raw_sd": standard_deviation,
                "status": "included_exploratory_browser",
                "cutoff": "backend_reply_timestamp_half_open",
                "interval_policy": "closed_non_right_censored",
            }
        )
    return out, pd.DataFrame(manifest_rows)


def add_terminal_sensitivity_terms(
    matched_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize durations after removing any interval component ending at reply time."""

    out = matched_frame.copy()
    manifest_rows: list[dict[str, Any]] = []
    for raw_column, term in TERMINAL_SENSITIVITY_TERM_MAP.items():
        raw = pd.to_numeric(out.get(raw_column), errors="coerce")
        if raw.isna().any() or raw.lt(0).any() or not np.isfinite(raw.to_numpy(dtype=float)).all():
            raise ValueError(f"{raw_column} must be finite and non-negative")
        transformed = np.log1p(raw.astype(float))
        mean = float(transformed.mean())
        standard_deviation = float(transformed.std(ddof=0))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"{raw_column} has no variation in the matched cohort")
        out[term] = (transformed - mean) / standard_deviation
        manifest_rows.append(
            {
                "term": term,
                "family": "opportunity_content",
                "source": raw_column,
                "description": f"Matched-cohort z score of log1p({raw_column}).",
                "focal": False,
                "standardized": True,
                "raw_mean": mean,
                "raw_sd": standard_deviation,
                "status": "included_exploratory_terminal_interval_sensitivity",
                "cutoff": "backend_reply_timestamp_half_open",
                "interval_policy": "closed_non_right_censored_exclude_terminal_component",
            }
        )
    return out, pd.DataFrame(manifest_rows)


def _canonical_scalar(value: Any) -> str:
    if _is_missing(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat() if value.tzinfo else value.isoformat()
    key = _id_key(value)
    return key if key is not None else "<NA>"


def stable_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash ordered rows and named columns with stable scalar normalization."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Hash columns are missing: {missing}")
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode("utf-8"))
    for values in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        digest.update(b"\x1e")
        digest.update("\x1f".join(_canonical_scalar(value) for value in values).encode("utf-8"))
    return digest.hexdigest()


def candidate_universe_sha256(frame: pd.DataFrame) -> str:
    """Hash the complete ordered candidate universe for paired model checks."""

    columns = (
        "choice_set_id",
        "group",
        "event_id",
        "candidate_content_id",
        "candidate_author_id",
        "chosen",
    )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Candidate universe contains duplicate keys")
    return stable_frame_sha256(frame, columns)


__all__ = [
    "BROWSER_TERM_MAP",
    "KEY_COLUMNS",
    "OBSERVED_INTERVAL_QUALITIES",
    "RAW_BROWSER_COLUMNS",
    "TERMINAL_SENSITIVITY_COLUMNS",
    "TERMINAL_SENSITIVITY_TERM_MAP",
    "add_terminal_sensitivity_terms",
    "add_standardized_browser_terms",
    "build_browser_choice_covariates",
    "candidate_universe_sha256",
    "matched_choice_set_ids",
    "stable_frame_sha256",
    "summarize_closed_episodes",
    "union_half_open_intervals",
]
