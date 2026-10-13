"""Browser opportunity reconstruction, action linkage, and tracking gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .study import analysis_set_manifest, experiment_day_series, parse_time

from .config import AnalysisConfig, BrowserGateConfig
from .io import read_table


ACTION_TYPES = ("comment", "upvote", "downvote")
AUTHOR_TYPES = ("H", "AI")
GROUPS = ("G1", "G2")

EPISODE_COLUMNS = [
    "browser_episode_id",
    "session_id",
    "group",
    "y_user_id",
    "item_id",
    "item_type",
    "author_id",
    "author_type_ground_truth",
    "root_post_id",
    "route_kind",
    "route_path",
    "impression_start_utc",
    "impression_end_utc",
    "start_event_seq",
    "end_event_seq",
    "close_reason",
    "right_censored",
    "interval_closed",
    "interval_exact",
    "interval_audit_approved",
    "hazard_eligible",
    "observed_elapsed_seconds",
    "verified_visible_seconds",
    "visible_duration_s",
    "max_simultaneous_elements",
    "source_event_rows",
    "duplicate_appear_events",
    "unmatched_disappear_events",
    "observed_status",
]

LINK_COLUMNS = [
    "backend_action_id",
    "event_id",
    "action_type",
    "group",
    "actor_id",
    "actor_type",
    "target_content_id",
    "target_author_id",
    "target_author_type",
    "action_timestamp_utc",
    "session_id",
    "browser_episode_id",
    "session_candidate_count",
    "candidate_episode_count",
    "action_episode_distance_seconds",
    "action_within_episode",
    "pre_action_linked",
    "link_ambiguous",
    "session_ambiguous",
    "episode_right_censored",
    "episode_interval_exact",
    "episode_interval_verified",
    "backend_duplicate_rows",
    "backend_duplicate_conflict",
    "link_status",
    "observed_status",
]

AUDIT_SAMPLE_COLUMNS = [
    "audit_sample_id",
    "audit_sample_seed",
    "audit_sample_rank",
    "audit_stratum",
    "audit_record_signature",
    "audit_author_type",
    "automated_linkage_status",
    "automated_interval_status",
    "audit_stratum_population_n",
    "audit_stratum_sample_n",
    "audit_selection_probability",
    *LINK_COLUMNS,
    "manual_correct",
    "audit_notes",
]

INTERVAL_AUDIT_SAMPLE_COLUMNS = [
    "interval_audit_sample_id",
    "interval_audit_sample_seed",
    "interval_audit_sample_rank",
    "interval_audit_stratum",
    "interval_audit_record_signature",
    "interval_audit_stratum_population_n",
    "interval_audit_stratum_sample_n",
    "interval_audit_selection_probability",
    "browser_episode_id",
    "session_id",
    "group",
    "y_user_id",
    "item_id",
    "item_type",
    "impression_start_utc",
    "impression_end_utc",
    "close_reason",
    "right_censored",
    "interval_exact",
    "duplicate_appear_events",
    "unmatched_disappear_events",
    "interval_approved",
    "interval_audit_notes",
]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _first_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return pd.NA


def _first_series(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _timestamp_series(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    return pd.to_datetime(_first_series(frame, names), errors="coerce", utc=True)


def _id_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _display_id(value: Any) -> Any:
    key = _id_key(value)
    if key is None:
        return pd.NA
    try:
        return int(key)
    except ValueError:
        return key


def _author_type(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"h", "human"}:
        return "H"
    if text in {"ai", "agent", "bot"}:
        return "AI"
    return str(value)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _canonical_action(row: Mapping[str, Any]) -> str | None:
    value = _first_value(row, ("action_type", "event_type", "action"))
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "downvote" in text:
        return "downvote"
    if "upvote" in text:
        return "upvote"
    if "comment" in text or "reply" in text:
        return "comment"
    vote = _first_value(row, ("vote_value", "reaction_value", "value"))
    try:
        vote_number = float(vote)
    except (TypeError, ValueError):
        return None
    if vote_number > 0:
        return "upvote"
    if vote_number < 0:
        return "downvote"
    return None


def _is_human(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in {"h", "human"}


def _close_episode(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    close_time: pd.Timestamp,
    close_seq: Any,
    reason: str,
    *,
    right_censored: bool,
) -> None:
    start = state["impression_start_utc"]
    end = max(start, close_time)
    elapsed = max(0.0, (end - start).total_seconds())
    exact = bool(
        not right_censored
        and state["duplicate_appear_events"] == 0
        and state["unmatched_disappear_events"] == 0
    )
    verified = elapsed if exact else 0.0
    rows.append(
        {
            **state["metadata"],
            "impression_start_utc": start,
            "impression_end_utc": end,
            "start_event_seq": state["start_event_seq"],
            "end_event_seq": close_seq,
            "close_reason": reason,
            "right_censored": right_censored,
            "interval_closed": not right_censored,
            "interval_exact": exact,
            "interval_audit_approved": False,
            "hazard_eligible": exact,
            "observed_elapsed_seconds": elapsed,
            "verified_visible_seconds": verified,
            "visible_duration_s": verified,
            "max_simultaneous_elements": state["max_simultaneous_elements"],
            "source_event_rows": state["source_event_rows"],
            "duplicate_appear_events": state["duplicate_appear_events"],
            "unmatched_disappear_events": state["unmatched_disappear_events"],
            "observed_status": "observed",
        }
    )


def _episode_metadata(row: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_id": row.get("session_id"),
        "group": row.get("group"),
        "y_user_id": _display_id(row.get("y_user_id")),
        "item_id": _display_id(_first_value(row, ("item_id", "target_item_id"))),
        "item_type": _first_value(row, ("item_type", "target_item_type")),
        "author_id": _display_id(row.get("author_id")),
        "author_type_ground_truth": _author_type(
            _first_value(row, ("author_type_ground_truth", "target_author_type"))
        ),
        "root_post_id": _display_id(row.get("root_post_id")),
        "route_kind": _first_value(row, ("route_kind",))
        if not pd.isna(_first_value(row, ("route_kind",)))
        else route.get("route_kind", pd.NA),
        "route_path": _first_value(row, ("route_path",))
        if not pd.isna(_first_value(row, ("route_path",)))
        else route.get("route_path", pd.NA),
    }


def _new_dom_state(row: Mapping[str, Any], route: Mapping[str, Any], element_id: str) -> dict[str, Any]:
    return {
        "impression_start_utc": row["_event_time"],
        "start_event_seq": row.get("event_seq", pd.NA),
        "elements": {element_id},
        "max_simultaneous_elements": 1,
        "source_event_rows": 1,
        "duplicate_appear_events": 0,
        "unmatched_disappear_events": 0,
        "metadata": _episode_metadata(row, route),
    }


def reconstruct_dom_episodes(session_events: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct separate content visibility cycles from DOM telemetry."""

    if session_events is None or session_events.empty:
        return _empty(EPISODE_COLUMNS)
    events = session_events.copy()
    if "session_id" not in events.columns:
        raise ValueError("session_events requires session_id")
    events["_event_time"] = _timestamp_series(events, ("event_timestamp_utc", "event_timestamp", "t"))
    if "event_seq" not in events.columns:
        events["event_seq"] = events.groupby("session_id").cumcount() + 1
    events = events.sort_values(["session_id", "_event_time", "event_seq"], kind="stable")
    rows: list[dict[str, Any]] = []

    for _, session in events.groupby("session_id", sort=False, dropna=False):
        active: dict[tuple[str | None, str | None, str | None], dict[str, Any]] = {}
        route: dict[str, Any] = {"route_kind": pd.NA, "route_path": pd.NA}

        def close_all(row: Mapping[str, Any], reason: str) -> None:
            for key in list(active):
                state = active.pop(key)
                _close_episode(
                    rows,
                    state,
                    row["_event_time"],
                    row.get("event_seq", pd.NA),
                    reason,
                    right_censored=False,
                )

        for _, event in session.iterrows():
            if pd.isna(event["_event_time"]):
                continue
            action = _text(event.get("action"))
            target_action = _text(event.get("target_action"))
            value = _text(event.get("value"))
            if action == "page_view" or target_action == "route_enter":
                close_all(event, "route_change")
                route = {"route_kind": event.get("route_kind", pd.NA), "route_path": event.get("route_path", pd.NA)}
                continue
            if (action == "visibility_change" and value == "page_hidden") or target_action == "page_hidden":
                close_all(event, "page_hidden")
                continue
            if action not in {"appear", "disappear"}:
                continue
            item_value = _first_value(event, ("item_id", "target_item_id"))
            item_key = _id_key(item_value)
            if item_key is None:
                continue
            group_key = _id_key(event.get("group"))
            item_type_key = _id_key(_first_value(event, ("item_type", "target_item_type")))
            key = (group_key, item_key, item_type_key)
            element_value = _first_value(event, ("id", "element_id"))
            element_id = _id_key(element_value) or f"content:{item_key}"

            if action == "appear":
                state = active.get(key)
                if state is None:
                    active[key] = _new_dom_state(event, route, element_id)
                    continue
                state["source_event_rows"] += 1
                if element_id in state["elements"]:
                    state["duplicate_appear_events"] += 1
                else:
                    state["elements"].add(element_id)
                    state["max_simultaneous_elements"] = max(
                        state["max_simultaneous_elements"], len(state["elements"])
                    )
                continue

            state = active.get(key)
            if state is None:
                continue
            state["source_event_rows"] += 1
            if element_id not in state["elements"]:
                state["unmatched_disappear_events"] += 1
                continue
            state["elements"].remove(element_id)
            if state["elements"]:
                continue
            active.pop(key)
            _close_episode(
                rows,
                state,
                event["_event_time"],
                event.get("event_seq", pd.NA),
                "all_elements_disappeared",
                right_censored=False,
            )

        if active:
            session_end = pd.to_datetime(
                _first_series(session, ("session_end_utc", "session_end")), errors="coerce", utc=True
            ).dropna()
            close_time = session_end.iloc[0] if not session_end.empty else session["_event_time"].dropna().max()
            last_seq = session["event_seq"].max()
            for key in list(active):
                state = active.pop(key)
                _close_episode(
                    rows,
                    state,
                    close_time,
                    last_seq,
                    "session_end_open",
                    right_censored=True,
                )

    if not rows:
        return _empty(EPISODE_COLUMNS)
    episodes = pd.DataFrame(rows).sort_values(
        ["session_id", "impression_start_utc", "start_event_seq", "item_id"], kind="stable"
    )
    sequence = episodes.groupby("session_id", sort=False).cumcount() + 1
    episodes.insert(
        0,
        "browser_episode_id",
        episodes["session_id"].astype(str) + ":DOM" + sequence.astype(str).str.zfill(6),
    )
    return episodes.reindex(columns=EPISODE_COLUMNS).reset_index(drop=True)


def _session_windows(session_events: pd.DataFrame) -> pd.DataFrame:
    columns = ["session_id", "group", "y_user_id", "session_start_utc", "session_end_utc"]
    if session_events is None or session_events.empty or "session_id" not in session_events.columns:
        return _empty(columns)
    frame = session_events.copy()
    frame["_event_time"] = _timestamp_series(frame, ("event_timestamp_utc", "event_timestamp", "t"))
    rows: list[dict[str, Any]] = []
    for session_id, part in frame.groupby("session_id", sort=False, dropna=False):
        starts = pd.to_datetime(
            _first_series(part, ("session_start_utc", "session_start")), errors="coerce", utc=True
        ).dropna()
        ends = pd.to_datetime(
            _first_series(part, ("session_end_utc", "session_end")), errors="coerce", utc=True
        ).dropna()
        start = starts.iloc[0] if not starts.empty else part["_event_time"].min()
        end = ends.iloc[0] if not ends.empty else part["_event_time"].max()
        if pd.isna(start) or pd.isna(end):
            continue
        rows.append(
            {
                "session_id": session_id,
                "group": _first_value(part.iloc[0], ("group",)),
                "y_user_id": _display_id(_first_value(part.iloc[0], ("y_user_id", "actor_id"))),
                "session_start_utc": start,
                "session_end_utc": max(start, end),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _prepare_backend_actions(backend_events: pd.DataFrame) -> pd.DataFrame:
    if backend_events is None or backend_events.empty:
        return pd.DataFrame()
    frame = backend_events.copy().reset_index(drop=False).rename(columns={"index": "_input_row"})
    frame["action_type"] = [_canonical_action(row) for _, row in frame.iterrows()]
    frame = frame[frame["action_type"].isin(ACTION_TYPES)].copy()
    if "actor_type" in frame.columns:
        frame = frame[frame["actor_type"].map(_is_human)].copy()
    if frame.empty:
        return frame
    frame["action_timestamp_utc"] = _timestamp_series(
        frame, ("event_timestamp_utc", "event_timestamp", "timestamp", "created_at")
    )
    frame["actor_id"] = _first_series(frame, ("actor_id", "y_user_id", "user_id")).map(_display_id)
    frame["target_content_id"] = _first_series(
        frame, ("target_content_id", "item_id", "parent_item_id")
    ).map(_display_id)
    if "event_id" not in frame.columns:
        frame["event_id"] = pd.NA
    raw_ids = []
    for _, row in frame.iterrows():
        raw = _id_key(row.get("event_id")) or f"row:{row['_input_row']}"
        group = str(row.get("group", "unknown"))
        raw_ids.append(raw if raw.startswith(f"{group}:") else f"{group}:{raw}")
    frame["backend_action_id"] = raw_ids
    duplicate_counts = frame.groupby("backend_action_id")["backend_action_id"].transform("size")
    conflict_columns = [
        column
        for column in ["action_type", "group", "actor_id", "target_content_id", "action_timestamp_utc"]
        if column in frame.columns
    ]
    conflicts = frame.groupby("backend_action_id")[conflict_columns].nunique(dropna=False).gt(1).any(axis=1)
    frame["backend_duplicate_rows"] = duplicate_counts
    frame["backend_duplicate_conflict"] = frame["backend_action_id"].map(conflicts).fillna(False)
    return frame.sort_values("_input_row", kind="stable").drop_duplicates("backend_action_id", keep="first")


def _episode_index(episodes: pd.DataFrame) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    if episodes is None or episodes.empty:
        return index
    for row in episodes.to_dict("records"):
        session_key = _id_key(row.get("session_id"))
        group_key = _id_key(row.get("group"))
        item_key = _id_key(row.get("item_id"))
        if session_key is None or group_key is None or item_key is None:
            continue
        index[(session_key, group_key, item_key)].append(row)
    return index


def _candidate_distance(action_time: pd.Timestamp, episode: Mapping[str, Any]) -> tuple[float, bool]:
    start = episode["impression_start_utc"]
    end = episode["impression_end_utc"]
    if start <= action_time <= end:
        return 0.0, True
    if action_time > end:
        return (action_time - end).total_seconds(), False
    return (start - action_time).total_seconds(), False


def _best_episode(
    candidates: list[dict[str, Any]], action_time: pd.Timestamp, tolerance_seconds: float
) -> tuple[dict[str, Any] | None, int, float | None, bool]:
    ranked: list[tuple[tuple[Any, ...], dict[str, Any], float, bool]] = []
    for episode in candidates:
        if episode["impression_start_utc"] > action_time:
            continue
        distance, contained = _candidate_distance(action_time, episode)
        if distance > tolerance_seconds:
            continue
        rank = (
            0 if contained else 1,
            distance,
            -episode["impression_start_utc"].value,
            str(episode["browser_episode_id"]),
        )
        ranked.append((rank, episode, distance, contained))
    if not ranked:
        return None, 0, None, False
    ranked.sort(key=lambda value: value[0])
    _, selected, distance, contained = ranked[0]
    return selected, len(ranked), distance, contained


def link_backend_actions(
    episodes: pd.DataFrame,
    session_events: pd.DataFrame,
    backend_events: pd.DataFrame,
    tolerance_seconds: float,
) -> pd.DataFrame:
    """Link each human backend action to at most one target-content episode."""

    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    actions = _prepare_backend_actions(backend_events)
    if actions.empty:
        return _empty(LINK_COLUMNS)
    windows = _session_windows(session_events)
    window_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in windows.to_dict("records"):
        group_key = _id_key(row.get("group"))
        user_key = _id_key(row.get("y_user_id"))
        if group_key is not None and user_key is not None:
            window_index[(group_key, user_key)].append(row)
    episode_index = _episode_index(episodes)
    rows: list[dict[str, Any]] = []

    for action in actions.to_dict("records"):
        group_key = _id_key(action.get("group"))
        user_key = _id_key(action.get("actor_id"))
        target_key = _id_key(action.get("target_content_id"))
        action_time = action.get("action_timestamp_utc")
        eligible_sessions = []
        if group_key is not None and user_key is not None and not pd.isna(action_time):
            eligible_sessions = [
                session
                for session in window_index.get((group_key, user_key), [])
                if session["session_start_utc"] <= action_time <= session["session_end_utc"]
            ]
        base = {
            "backend_action_id": action.get("backend_action_id"),
            "event_id": action.get("event_id", pd.NA),
            "action_type": action.get("action_type"),
            "group": action.get("group"),
            "actor_id": action.get("actor_id"),
            "actor_type": _first_value(action, ("actor_type",)),
            "target_content_id": action.get("target_content_id"),
            "target_author_id": _display_id(_first_value(action, ("target_author_id", "author_id"))),
            "target_author_type": _author_type(
                _first_value(action, ("target_author_type", "target_author_type_ground_truth"))
            ),
            "action_timestamp_utc": action_time,
            "session_id": pd.NA,
            "browser_episode_id": pd.NA,
            "session_candidate_count": pd.NA,
            "candidate_episode_count": pd.NA,
            "action_episode_distance_seconds": pd.NA,
            "action_within_episode": pd.NA,
            "pre_action_linked": pd.NA,
            "link_ambiguous": pd.NA,
            "session_ambiguous": pd.NA,
            "episode_right_censored": pd.NA,
            "episode_interval_exact": pd.NA,
            "episode_interval_verified": pd.NA,
            "backend_duplicate_rows": int(action.get("backend_duplicate_rows", 1)),
            "backend_duplicate_conflict": bool(action.get("backend_duplicate_conflict", False)),
            "link_status": "unobserved_no_session",
            "observed_status": "unobserved",
        }
        if target_key is None or pd.isna(action_time) or group_key is None or user_key is None:
            base["link_status"] = "invalid_action_keys"
            rows.append(base)
            continue
        if not eligible_sessions:
            rows.append(base)
            continue
        base["observed_status"] = "observed"
        base["session_candidate_count"] = len(eligible_sessions)
        base["session_ambiguous"] = len(eligible_sessions) > 1
        candidates: list[dict[str, Any]] = []
        for session in eligible_sessions:
            session_key = _id_key(session["session_id"])
            candidates.extend(episode_index.get((session_key, group_key, target_key), []))
        selected, count, distance, contained = _best_episode(candidates, action_time, tolerance_seconds)
        base["candidate_episode_count"] = count
        base["link_ambiguous"] = count > 1
        if selected is None:
            base["session_id"] = eligible_sessions[0]["session_id"] if len(eligible_sessions) == 1 else pd.NA
            base["link_status"] = "observed_no_target_episode"
            rows.append(base)
            continue
        base.update(
            {
                "session_id": selected["session_id"],
                "browser_episode_id": selected["browser_episode_id"],
                "action_episode_distance_seconds": distance,
                "action_within_episode": contained,
                "pre_action_linked": selected["impression_start_utc"] <= action_time,
                "episode_right_censored": bool(selected.get("right_censored", False)),
                "episode_interval_exact": bool(selected.get("interval_exact", False)),
                "episode_interval_verified": bool(selected.get("hazard_eligible", False)),
                "link_status": "linked_ambiguous" if count > 1 else "linked_unique",
            }
        )
        if pd.isna(base["target_author_type"]):
            base["target_author_type"] = _author_type(selected.get("author_type_ground_truth"))
        if pd.isna(base["target_author_id"]):
            base["target_author_id"] = _display_id(selected.get("author_id"))
        rows.append(base)

    out = pd.DataFrame(rows).reindex(columns=LINK_COLUMNS)
    if out["backend_action_id"].duplicated().any():
        raise RuntimeError("Backend action linkage produced duplicate action rows")
    return out.reset_index(drop=True)


def _stable_audit_hash(seed: int, *values: Any) -> str:
    payload = "\x1f".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audit_bool(row: Mapping[str, Any], column: str, default: bool = False) -> bool:
    value = row.get(column, default)
    return default if value is None or pd.isna(value) else bool(value)


def _audit_linkage_status(row: Mapping[str, Any]) -> str:
    if _audit_bool(row, "backend_duplicate_conflict"):
        return "duplicate_conflict"
    if _audit_bool(row, "session_ambiguous"):
        return "session_ambiguous"
    if _audit_bool(row, "link_ambiguous"):
        return "episode_ambiguous"
    if not _audit_bool(row, "pre_action_linked"):
        return "not_pre_action"
    if not _audit_bool(row, "action_within_episode"):
        return "outside_interval"
    return "unique_pre_action_within_interval"


def _audit_interval_status(row: Mapping[str, Any]) -> str:
    if _audit_bool(row, "episode_right_censored"):
        return "right_censored"
    if _audit_bool(row, "episode_interval_exact"):
        return "exact"
    return "nonexact_requires_audit"


def _stratified_sample_quotas(
    population_sizes: Mapping[str, int], target: int, seed: int, group: str
) -> dict[str, int]:
    strata = [stratum for stratum, count in population_sizes.items() if count > 0]
    quotas = {stratum: 0 for stratum in strata}
    if not strata or target <= 0:
        return quotas
    target = min(target, sum(population_sizes.values()))
    order = sorted(
        strata,
        key=lambda stratum: (
            -population_sizes[stratum],
            _stable_audit_hash(seed, group, stratum),
        ),
    )
    for stratum in order[: min(target, len(order))]:
        quotas[stratum] = 1
    while sum(quotas.values()) < target:
        eligible = [
            stratum
            for stratum in strata
            if quotas[stratum] < population_sizes[stratum]
        ]
        selected = min(
            eligible,
            key=lambda stratum: (
                -population_sizes[stratum] / (quotas[stratum] + 1),
                _stable_audit_hash(seed, group, stratum, quotas[stratum]),
            ),
        )
        quotas[selected] += 1
    return quotas


def build_manual_audit_sample(
    linked_actions: pd.DataFrame,
    *,
    per_arm: int,
    seed: int,
) -> pd.DataFrame:
    """Create a deterministic, stratified template for manual linkage review.

    Only actions attributed to an episode enter the precision-audit population.
    Sampling strata cross target author type, automated linkage status, and the
    raw interval status. The latter uses ``interval_exact`` rather than a later
    manual approval, keeping the generated sample stable across audit reruns.
    """

    if per_arm <= 0:
        raise ValueError("per_arm must be positive")
    if linked_actions is None or linked_actions.empty:
        return _empty(AUDIT_SAMPLE_COLUMNS)
    required = {"backend_action_id", "group", "browser_episode_id"}
    missing = sorted(required.difference(linked_actions.columns))
    if missing:
        raise ValueError(f"Linked actions are missing audit fields: {missing}")
    population = linked_actions[
        linked_actions["group"].isin(GROUPS)
        & linked_actions["browser_episode_id"].notna()
    ].copy()
    if population.empty:
        return _empty(AUDIT_SAMPLE_COLUMNS)
    if population["backend_action_id"].isna().any():
        raise ValueError("Linked audit actions require non-missing backend_action_id")
    if population["backend_action_id"].astype(str).duplicated().any():
        raise ValueError("Linked audit actions require unique backend_action_id")

    population["audit_author_type"] = population["target_author_type"].map(_author_type).fillna("UNKNOWN")
    population["automated_linkage_status"] = [
        _audit_linkage_status(row) for row in population.to_dict("records")
    ]
    population["automated_interval_status"] = [
        _audit_interval_status(row) for row in population.to_dict("records")
    ]
    population["audit_stratum"] = (
        population["audit_author_type"].astype(str)
        + "::"
        + population["automated_linkage_status"].astype(str)
        + "::"
        + population["automated_interval_status"].astype(str)
    )
    population["_audit_sort"] = [
        _stable_audit_hash(seed, group, stratum, action_id)
        for group, stratum, action_id in zip(
            population["group"],
            population["audit_stratum"],
            population["backend_action_id"],
        )
    ]

    selected_parts: list[pd.DataFrame] = []
    for group in GROUPS:
        arm = population[population["group"].eq(group)].copy()
        if arm.empty:
            continue
        counts = arm["audit_stratum"].value_counts().astype(int).to_dict()
        quotas = _stratified_sample_quotas(counts, per_arm, seed, group)
        for stratum, quota in quotas.items():
            if quota <= 0:
                continue
            part = arm[arm["audit_stratum"].eq(stratum)].sort_values(
                "_audit_sort", kind="stable"
            ).head(quota).copy()
            part["audit_stratum_population_n"] = counts[stratum]
            part["audit_stratum_sample_n"] = quota
            part["audit_selection_probability"] = quota / counts[stratum]
            selected_parts.append(part)
    if not selected_parts:
        return _empty(AUDIT_SAMPLE_COLUMNS)

    sample = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["group", "audit_stratum", "_audit_sort"], kind="stable"
    )
    sample["audit_sample_rank"] = sample.groupby("group", sort=False).cumcount() + 1
    sample["audit_sample_id"] = (
        sample["group"].astype(str)
        + ":AUDIT:"
        + sample["audit_sample_rank"].astype(str).str.zfill(3)
    )
    sample["audit_sample_seed"] = int(seed)
    sample["audit_record_signature"] = [
        _stable_audit_hash(seed, group, action_id, stratum)
        for group, action_id, stratum in zip(
            sample["group"], sample["backend_action_id"], sample["audit_stratum"]
        )
    ]
    sample["manual_correct"] = pd.NA
    sample["audit_notes"] = pd.NA
    if sample.groupby("group")["backend_action_id"].nunique().gt(per_arm).any():
        raise RuntimeError("Manual audit sample exceeded its per-arm target")
    return sample.reindex(columns=AUDIT_SAMPLE_COLUMNS).reset_index(drop=True)


def build_interval_audit_sample(
    episodes: pd.DataFrame,
    *,
    per_arm: int,
    seed: int,
) -> pd.DataFrame:
    """Create a deterministic audit sample of closed, nonexact DOM episodes."""

    if per_arm <= 0:
        raise ValueError("per_arm must be positive")
    if episodes is None or episodes.empty:
        return _empty(INTERVAL_AUDIT_SAMPLE_COLUMNS)
    required = {
        "browser_episode_id",
        "group",
        "right_censored",
        "interval_exact",
        "close_reason",
        "duplicate_appear_events",
        "unmatched_disappear_events",
    }
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"Browser episodes are missing interval-audit fields: {missing}")
    population = episodes[
        episodes["group"].isin(GROUPS)
        & ~episodes["right_censored"].astype("boolean").fillna(True).astype(bool)
        & ~episodes["interval_exact"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    if population.empty:
        return _empty(INTERVAL_AUDIT_SAMPLE_COLUMNS)
    if population["browser_episode_id"].isna().any():
        raise ValueError("Interval-audit episodes require browser_episode_id")
    if population["browser_episode_id"].astype(str).duplicated().any():
        raise ValueError("Interval-audit episodes require unique browser_episode_id")

    population["interval_audit_stratum"] = (
        population["close_reason"].astype(str)
        + "::duplicate="
        + population["duplicate_appear_events"].gt(0).astype(int).astype(str)
        + "::unmatched="
        + population["unmatched_disappear_events"].gt(0).astype(int).astype(str)
    )
    population["_interval_audit_sort"] = [
        _stable_audit_hash(seed, "interval", group, stratum, episode_id)
        for group, stratum, episode_id in zip(
            population["group"],
            population["interval_audit_stratum"],
            population["browser_episode_id"],
        )
    ]
    selected_parts: list[pd.DataFrame] = []
    for group in GROUPS:
        arm = population[population["group"].eq(group)].copy()
        if arm.empty:
            continue
        counts = arm["interval_audit_stratum"].value_counts().astype(int).to_dict()
        quotas = _stratified_sample_quotas(counts, per_arm, seed, f"interval::{group}")
        for stratum, quota in quotas.items():
            if quota <= 0:
                continue
            part = arm[arm["interval_audit_stratum"].eq(stratum)].sort_values(
                "_interval_audit_sort", kind="stable"
            ).head(quota).copy()
            part["interval_audit_stratum_population_n"] = counts[stratum]
            part["interval_audit_stratum_sample_n"] = quota
            part["interval_audit_selection_probability"] = quota / counts[stratum]
            selected_parts.append(part)
    if not selected_parts:
        return _empty(INTERVAL_AUDIT_SAMPLE_COLUMNS)

    sample = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["group", "interval_audit_stratum", "_interval_audit_sort"], kind="stable"
    )
    sample["interval_audit_sample_rank"] = sample.groupby("group", sort=False).cumcount() + 1
    sample["interval_audit_sample_id"] = (
        sample["group"].astype(str)
        + ":INTERVAL_AUDIT:"
        + sample["interval_audit_sample_rank"].astype(str).str.zfill(3)
    )
    sample["interval_audit_sample_seed"] = int(seed)
    sample["interval_audit_record_signature"] = [
        _stable_audit_hash(
            seed,
            "interval",
            group,
            episode_id,
            stratum,
            start,
            end,
            reason,
            duplicate_count,
            unmatched_count,
        )
        for group, episode_id, stratum, start, end, reason, duplicate_count, unmatched_count in zip(
            sample["group"],
            sample["browser_episode_id"],
            sample["interval_audit_stratum"],
            sample["impression_start_utc"],
            sample["impression_end_utc"],
            sample["close_reason"],
            sample["duplicate_appear_events"],
            sample["unmatched_disappear_events"],
        )
    ]
    sample["interval_approved"] = pd.NA
    sample["interval_audit_notes"] = pd.NA
    if sample.groupby("group")["browser_episode_id"].nunique().gt(per_arm).any():
        raise RuntimeError("Interval audit sample exceeded its per-arm target")
    return sample.reindex(columns=INTERVAL_AUDIT_SAMPLE_COLUMNS).reset_index(drop=True)


def _action_lookup(linked_actions: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if linked_actions is None or linked_actions.empty:
        return lookup
    for row in linked_actions.to_dict("records"):
        group = _id_key(row.get("group"))
        event = _id_key(row.get("event_id"))
        if group is not None and event is not None:
            lookup[(group, event)] = row
    return lookup


def _prior_candidate_exposure(
    candidate_episodes: list[dict[str, Any]], action_time: pd.Timestamp
) -> dict[str, Any]:
    prior = [episode for episode in candidate_episodes if episode["impression_start_utc"] <= action_time]
    observed_seconds = 0.0
    verified_seconds = 0.0
    for episode in prior:
        if bool(episode.get("right_censored", False)):
            continue
        endpoint = min(episode["impression_end_utc"], action_time)
        duration = max(0.0, (endpoint - episode["impression_start_utc"]).total_seconds())
        observed_seconds += duration
        if bool(
            episode.get(
                "hazard_eligible",
                episode.get("episode_interval_verified", False),
            )
        ):
            verified_seconds += duration
    return {
        "candidate_seen_before_action": bool(prior),
        "candidate_visible_at_action": any(
            episode["impression_start_utc"] <= action_time <= episode["impression_end_utc"]
            for episode in prior
        ),
        "candidate_episode_count_prior": len(prior),
        "candidate_right_censored_episode_count_prior": sum(
            bool(episode.get("right_censored", False)) for episode in prior
        ),
        "candidate_verified_visible_seconds_prior": verified_seconds,
        "candidate_visible_seconds_prior": observed_seconds,
    }


def build_browser_choice_sets(
    thread_choice_sets: pd.DataFrame,
    episodes: pd.DataFrame,
    linked_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach nullable candidate exposure to same-thread choice sets."""

    if thread_choice_sets is None or thread_choice_sets.empty:
        return pd.DataFrame()
    out = thread_choice_sets.copy()
    actions = _action_lookup(linked_actions)
    episode_index = _episode_index(episodes)
    exposure_rows: list[dict[str, Any]] = []

    for row in out.to_dict("records"):
        group_key = _id_key(row.get("group"))
        event_key = _id_key(row.get("event_id"))
        action = actions.get((group_key, event_key)) if group_key is not None and event_key is not None else None
        missing = {
            "observed_status": "unobserved",
            "browser_session_id": pd.NA,
            "linked_browser_episode_id": pd.NA,
            "candidate_seen_before_action": pd.NA,
            "candidate_visible_at_action": pd.NA,
            "candidate_episode_count_prior": pd.NA,
            "candidate_right_censored_episode_count_prior": pd.NA,
            "candidate_verified_visible_seconds_prior": pd.NA,
            "candidate_visible_seconds_prior": pd.NA,
            "chosen_episode_linked": pd.NA,
            "browser_link_ambiguous": pd.NA,
            "browser_session_ambiguous": pd.NA,
            "chosen_episode_interval_verified": pd.NA,
            "chosen_action_within_episode": pd.NA,
        }
        if action is None or action.get("observed_status") != "observed" or pd.isna(action.get("session_id")):
            exposure_rows.append(missing)
            continue
        action_time = action.get("action_timestamp_utc")
        candidate_key = _id_key(_first_value(row, ("candidate_content_id", "content_id", "item_id")))
        session_key = _id_key(action.get("session_id"))
        if pd.isna(action_time) or candidate_key is None or session_key is None or group_key is None:
            exposure_rows.append(missing)
            continue
        metrics = _prior_candidate_exposure(
            episode_index.get((session_key, group_key, candidate_key), []), action_time
        )
        chosen = bool(row.get("chosen", False)) if not pd.isna(row.get("chosen", pd.NA)) else False
        metrics.update(
            {
                "observed_status": "observed",
                "browser_session_id": action.get("session_id"),
                "linked_browser_episode_id": action.get("browser_episode_id") if chosen else pd.NA,
                "chosen_episode_linked": bool(chosen and not pd.isna(action.get("browser_episode_id"))),
                "browser_link_ambiguous": bool(action.get("link_ambiguous", False)),
                "browser_session_ambiguous": bool(action.get("session_ambiguous", False)),
                "chosen_episode_interval_verified": bool(
                    chosen and action.get("episode_interval_verified", False)
                ),
                "chosen_action_within_episode": bool(
                    chosen and action.get("action_within_episode", False)
                ),
            }
        )
        exposure_rows.append(metrics)

    exposure = pd.DataFrame(exposure_rows, index=out.index)
    for column in exposure.columns:
        out[column] = exposure[column]
    for column in [
        "candidate_seen_before_action",
        "candidate_visible_at_action",
        "chosen_episode_linked",
        "chosen_episode_interval_verified",
        "chosen_action_within_episode",
    ]:
        out[column] = out[column].astype("boolean")
    return out


def _gate_settings(value: Any) -> BrowserGateConfig:
    if value is None:
        return BrowserGateConfig()
    if isinstance(value, AnalysisConfig):
        return value.browser_gates
    if isinstance(value, BrowserGateConfig):
        return value
    if isinstance(value, Mapping):
        return BrowserGateConfig(**dict(value))
    raise TypeError("gates must be BrowserGateConfig, AnalysisConfig, a mapping, or None")


def _metric_row(
    metric: str,
    group: str,
    numerator: int | None,
    denominator: int | None,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    observed = denominator is not None and denominator > 0 and numerator is not None
    value = numerator / denominator if observed else pd.NA
    if not observed:
        gate_pass: Any = pd.NA
    elif comparison == ">=":
        gate_pass = value >= threshold
    elif comparison == "<=":
        gate_pass = value <= threshold
    else:
        raise ValueError(f"Unsupported comparison {comparison}")
    return {
        "metric": metric,
        "group": group,
        "numerator": numerator if numerator is not None else pd.NA,
        "denominator": denominator if denominator is not None else pd.NA,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "gate_pass": gate_pass,
        "observed_status": "observed" if observed else "not_audited" if metric.startswith("manual_") else "unobserved",
    }


def _manual_correctness(manual_audit: pd.DataFrame) -> pd.Series | None:
    for column in ("manual_correct", "is_correct", "link_correct", "match_correct", "correct"):
        if column in manual_audit.columns:
            values = manual_audit[column]
            if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
                return values.astype("boolean")
            return values.astype(str).str.strip().str.lower().map(
                {
                    "true": True,
                    "yes": True,
                    "correct": True,
                    "approved": True,
                    "approve": True,
                    "1": True,
                    "false": False,
                    "no": False,
                    "incorrect": False,
                    "rejected": False,
                    "reject": False,
                    "0": False,
                }
            ).astype("boolean")
    if "verdict" in manual_audit.columns:
        return manual_audit["verdict"].astype(str).str.strip().str.lower().map(
            {"correct": True, "incorrect": False}
        ).astype("boolean")
    return None


def _action_uniqueness_row(frame: pd.DataFrame, group: str, metric: str) -> dict[str, Any]:
    part = frame if group == "all" else frame[frame.get("group", pd.Series(dtype=object)).eq(group)]
    if "backend_action_id" not in part.columns or part.empty:
        return _metric_row(metric, group, None, None, 1.0, ">=")
    identifiers = part["backend_action_id"].dropna().astype(str)
    return _metric_row(metric, group, int(identifiers.nunique()), len(part), 1.0, ">=")


def _audit_record_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    required = {"backend_action_id", "audit_record_signature", "audit_stratum"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    valid = frame.dropna(subset=list(required))
    return set(
        zip(
            valid["backend_action_id"].astype(str),
            valid["audit_record_signature"].astype(str),
            valid["audit_stratum"].astype(str),
        )
    )


def _manual_audit_gate_rows(
    audit_sample: pd.DataFrame,
    manual_audit: pd.DataFrame,
    settings: BrowserGateConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = manual_audit.copy()
    correctness = _manual_correctness(audit) if not audit.empty else None
    if correctness is not None:
        audit["_manual_correct"] = correctness

    for group in GROUPS:
        sample_part = (
            audit_sample[audit_sample.get("group", pd.Series(dtype=object)).eq(group)]
            if isinstance(audit_sample, pd.DataFrame)
            else pd.DataFrame()
        )
        generated_size = len(sample_part)
        generated_size_row = _metric_row(
            "generated_audit_sample_size",
            group,
            generated_size,
            1,
            float(settings.manual_audit_per_arm),
            ">=",
        )
        rows.append(generated_size_row)
        generated_unique = _action_uniqueness_row(
            sample_part, "all", "generated_audit_sample_uniqueness"
        )
        generated_unique["group"] = group
        rows.append(generated_unique)

        if correctness is None or "group" not in audit.columns:
            for metric, threshold in (
                ("manual_audit_sample_size", float(settings.manual_audit_per_arm)),
                ("manual_audit_action_uniqueness", 1.0),
                ("manual_audit_sample_stratum_match", 1.0),
                ("manual_match_precision", settings.min_manual_precision),
                ("manual_match_precision_weighted", settings.min_manual_precision),
            ):
                rows.append(_metric_row(metric, group, None, None, threshold, ">="))
            continue

        group_audit = audit[audit["group"].eq(group)].copy()
        labeled = group_audit[group_audit["_manual_correct"].notna()].copy()
        sample_size = len(labeled)
        size_row = _metric_row(
            "manual_audit_sample_size",
            group,
            sample_size,
            1,
            float(settings.manual_audit_per_arm),
            ">=",
        )
        rows.append(size_row)
        audit_unique = _action_uniqueness_row(
            labeled, "all", "manual_audit_action_uniqueness"
        )
        audit_unique["group"] = group
        rows.append(audit_unique)

        sample_keys = _audit_record_keys(sample_part)
        audit_keys = _audit_record_keys(labeled)
        union = sample_keys | audit_keys
        intersection = sample_keys & audit_keys
        match_row = _metric_row(
            "manual_audit_sample_stratum_match",
            group,
            len(intersection) if union else None,
            len(union) if union else None,
            1.0,
            ">=",
        )
        rows.append(match_row)

        if sample_keys and audit_keys:
            keyed = list(
                zip(
                    labeled["backend_action_id"].astype(str),
                    labeled["audit_record_signature"].astype(str),
                    labeled["audit_stratum"].astype(str),
                )
            )
            matched = labeled[[key in sample_keys for key in keyed]]
        else:
            matched = labeled.iloc[0:0]
        precision_row = _metric_row(
            "manual_match_precision",
            group,
            int(matched["_manual_correct"].astype(bool).sum()) if len(matched) else None,
            len(matched) if len(matched) else None,
            settings.min_manual_precision,
            ">=",
        )
        prerequisite_rows = (
            generated_size_row,
            generated_unique,
            size_row,
            audit_unique,
            match_row,
        )
        prerequisites = all(
            not pd.isna(row["gate_pass"]) and bool(row["gate_pass"])
            for row in prerequisite_rows
        )
        if len(labeled) and not prerequisites:
            precision_row["gate_pass"] = False
            precision_row["observed_status"] = "invalid_audit_sample"
        rows.append(precision_row)

        weight_lookup: dict[tuple[str, str, str], float] = {}
        weight_columns = {"audit_stratum_population_n", "audit_stratum_sample_n"}
        if sample_keys and weight_columns.issubset(sample_part.columns):
            for sample_row in sample_part.itertuples(index=False):
                key = (
                    str(sample_row.backend_action_id),
                    str(sample_row.audit_record_signature),
                    str(sample_row.audit_stratum),
                )
                population = float(sample_row.audit_stratum_population_n)
                sampled = float(sample_row.audit_stratum_sample_n)
                if sampled > 0.0 and population >= sampled:
                    weight_lookup[key] = population / sampled
        matched_keys = list(
            zip(
                matched.get("backend_action_id", pd.Series(dtype=object)).astype(str),
                matched.get("audit_record_signature", pd.Series(dtype=object)).astype(str),
                matched.get("audit_stratum", pd.Series(dtype=object)).astype(str),
            )
        )
        weights = np.asarray(
            [weight_lookup.get(key, np.nan) for key in matched_keys], dtype=float
        )
        valid_weights = np.isfinite(weights) & (weights > 0.0)
        if len(matched) and valid_weights.all():
            correct = matched["_manual_correct"].astype(bool).to_numpy(dtype=float)
            numerator = float(np.sum(weights * correct))
            denominator = float(np.sum(weights))
            value = numerator / denominator
            weighted_row = {
                "metric": "manual_match_precision_weighted",
                "group": group,
                "numerator": numerator,
                "denominator": denominator,
                "value": value,
                "threshold": settings.min_manual_precision,
                "comparison": ">=",
                "gate_pass": value >= settings.min_manual_precision,
                "observed_status": "observed",
            }
        else:
            weighted_row = _metric_row(
                "manual_match_precision_weighted",
                group,
                None,
                None,
                settings.min_manual_precision,
                ">=",
            )
        if len(labeled) and not prerequisites:
            weighted_row["gate_pass"] = False
            weighted_row["observed_status"] = "invalid_audit_sample"
        rows.append(weighted_row)
    return rows


def _episode_reconstruction_qc(
    session_events: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Audit whether every content DOM source row entered episode state."""

    events = session_events.copy()
    action = events.get("action", pd.Series(index=events.index, dtype=object)).map(_text)
    item = _first_series(events, ("item_id", "target_item_id")).map(_id_key)
    content_dom = events[action.isin({"appear", "disappear"}) & item.notna()].copy()
    content_dom["_dom_action"] = action.loc[content_dom.index]
    rows: list[dict[str, Any]] = []
    for group in (*GROUPS, "all"):
        source = (
            content_dom
            if group == "all"
            else content_dom[content_dom.get("group", pd.Series(dtype=object)).astype(str).eq(group)]
        )
        episode_part = (
            episodes
            if group == "all"
            else episodes[episodes.get("group", pd.Series(dtype=object)).astype(str).eq(group)]
        )
        source_rows = len(source)
        represented_rows = int(
            pd.to_numeric(
                episode_part.get("source_event_rows", pd.Series(dtype=float)), errors="coerce"
            ).fillna(0).sum()
        )
        unrepresented = max(0, source_rows - represented_rows)
        accounting = _metric_row(
            "dom_content_event_source_accounting_rate",
            group,
            represented_rows,
            source_rows if source_rows else None,
            1.0,
            ">=",
        )
        if represented_rows > source_rows:
            accounting["gate_pass"] = False
            accounting["observed_status"] = "invalid_source_row_overcount"
        rows.append(accounting)

        disappear_rows = int(source["_dom_action"].eq("disappear").sum())
        rows.append(
            _metric_row(
                "unmatched_disappear_source_rate",
                group,
                unrepresented,
                disappear_rows if disappear_rows else None,
                0.0,
                "<=",
            )
        )
        anomaly = (
            pd.to_numeric(
                episode_part.get("duplicate_appear_events", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).gt(0)
            | pd.to_numeric(
                episode_part.get("unmatched_disappear_events", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).gt(0)
        )
        anomalous = episode_part[anomaly]
        exact_anomalous = int(
            anomalous.get("interval_exact", pd.Series(False, index=anomalous.index))
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if anomalous.empty:
            exactness_row = {
                "metric": "anomalous_episode_exactness_rate",
                "group": group,
                "numerator": 0,
                "denominator": 0,
                "value": 0.0,
                "threshold": 0.0,
                "comparison": "<=",
                "gate_pass": True,
                "observed_status": "observed_no_anomalous_episodes",
            }
        else:
            exactness_row = _metric_row(
                "anomalous_episode_exactness_rate",
                group,
                exact_anomalous,
                len(anomalous),
                0.0,
                "<=",
            )
        rows.append(exactness_row)
    return pd.DataFrame(rows)


def _interval_audit_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    required = {
        "group",
        "browser_episode_id",
        "interval_audit_record_signature",
        "interval_audit_stratum",
    }
    if frame.empty or not required.issubset(frame.columns):
        return set()
    valid = frame.dropna(subset=list(required))
    return set(
        zip(
            valid["group"].astype(str),
            valid["browser_episode_id"].astype(str),
            valid["interval_audit_record_signature"].astype(str),
            valid["interval_audit_stratum"].astype(str),
        )
    )


def _authenticate_interval_audit(
    episodes: pd.DataFrame,
    audit_sample: pd.DataFrame,
    interval_audit: pd.DataFrame,
    settings: BrowserGateConfig,
) -> tuple[set[str], pd.DataFrame]:
    """Authenticate current-sample interval judgments before applying any approval."""

    audit = interval_audit.copy()
    approval: pd.Series | None = None
    if "interval_approved" in audit.columns:
        approval = _manual_correctness(
            pd.DataFrame({"manual_correct": audit["interval_approved"]})
        )
        if approval is not None:
            audit["_interval_approved"] = approval
    interval_audit_per_arm = (
        settings.interval_audit_per_arm
        if settings.interval_audit_per_arm is not None
        else settings.manual_audit_per_arm
    )
    rows: list[dict[str, Any]] = []
    approved_ids: set[str] = set()
    arm_passes: list[bool] = []
    for group in GROUPS:
        sample_part = audit_sample[
            audit_sample.get("group", pd.Series(dtype=object)).astype(str).eq(group)
        ]
        generated_size = len(sample_part)
        generated_size_row = _metric_row(
            "generated_interval_audit_sample_size",
            group,
            generated_size,
            1,
            float(interval_audit_per_arm),
            ">=",
        )
        rows.append(generated_size_row)
        generated_ids = sample_part.get(
            "browser_episode_id", pd.Series(index=sample_part.index, dtype=object)
        ).dropna().astype(str)
        generated_unique_row = _metric_row(
            "generated_interval_audit_sample_uniqueness",
            group,
            int(generated_ids.nunique()) if len(sample_part) else None,
            len(sample_part) if len(sample_part) else None,
            1.0,
            ">=",
        )
        rows.append(generated_unique_row)

        if approval is None or "group" not in audit.columns:
            for metric, threshold in (
                ("interval_audit_sample_size", float(interval_audit_per_arm)),
                ("interval_audit_sample_uniqueness", 1.0),
                ("interval_audit_sample_membership", 1.0),
                ("interval_audit_completed_judgments", 1.0),
            ):
                rows.append(_metric_row(metric, group, None, None, threshold, ">="))
            rows.append(
                {
                    "metric": "interval_audit_authentication_pass",
                    "group": group,
                    "numerator": pd.NA,
                    "denominator": pd.NA,
                    "value": False,
                    "threshold": True,
                    "comparison": "==",
                    "gate_pass": False,
                    "observed_status": "not_audited",
                }
            )
            arm_passes.append(False)
            continue

        submitted = audit[audit["group"].astype(str).eq(group)].copy()
        submitted_ids = submitted.get(
            "browser_episode_id", pd.Series(index=submitted.index, dtype=object)
        ).dropna().astype(str)
        size_row = _metric_row(
            "interval_audit_sample_size",
            group,
            len(submitted),
            1,
            float(interval_audit_per_arm),
            ">=",
        )
        rows.append(size_row)
        uniqueness_row = _metric_row(
            "interval_audit_sample_uniqueness",
            group,
            int(submitted_ids.nunique()) if len(submitted) else None,
            len(submitted) if len(submitted) else None,
            1.0,
            ">=",
        )
        rows.append(uniqueness_row)
        sample_keys = _interval_audit_keys(sample_part)
        submitted_keys = _interval_audit_keys(submitted)
        union = sample_keys | submitted_keys
        membership_row = _metric_row(
            "interval_audit_sample_membership",
            group,
            len(sample_keys & submitted_keys) if union else None,
            len(union) if union else None,
            1.0,
            ">=",
        )
        rows.append(membership_row)
        completed = int(submitted.get("_interval_approved", pd.Series(dtype="boolean")).notna().sum())
        completed_row = _metric_row(
            "interval_audit_completed_judgments",
            group,
            completed if len(submitted) else None,
            len(submitted) if len(submitted) else None,
            1.0,
            ">=",
        )
        rows.append(completed_row)
        prerequisites = (
            generated_size_row,
            generated_unique_row,
            size_row,
            uniqueness_row,
            membership_row,
            completed_row,
        )
        authenticated = all(
            not pd.isna(row["gate_pass"]) and bool(row["gate_pass"])
            for row in prerequisites
        )
        rows.append(
            {
                "metric": "interval_audit_authentication_pass",
                "group": group,
                "numerator": completed,
                "denominator": len(submitted),
                "value": authenticated,
                "threshold": True,
                "comparison": "==",
                "gate_pass": authenticated,
                "observed_status": "observed" if authenticated else "invalid_interval_audit",
            }
        )
        arm_passes.append(authenticated)
        if authenticated:
            approved_ids.update(
                submitted.loc[
                    submitted["_interval_approved"].astype(bool), "browser_episode_id"
                ].astype(str)
            )

    if not all(arm_passes):
        approved_ids.clear()
    for group in GROUPS:
        arm = episodes[episodes.get("group", pd.Series(dtype=object)).astype(str).eq(group)]
        raw_exact = int(
            arm.get("interval_exact", pd.Series(False, index=arm.index))
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            .sum()
        )
        exact_rate = raw_exact / len(arm) if len(arm) else np.nan
        group_auth = next(
            row for row in rows
            if row["metric"] == "interval_audit_authentication_pass" and row["group"] == group
        )
        requirement_pass = bool(
            (not pd.isna(exact_rate) and exact_rate >= settings.min_interval_closure)
            or group_auth["gate_pass"]
        )
        rows.append(
            {
                "metric": "interval_audit_requirement_pass",
                "group": group,
                "numerator": raw_exact,
                "denominator": len(arm),
                "value": requirement_pass,
                "threshold": True,
                "comparison": "==",
                "gate_pass": requirement_pass,
                "observed_status": (
                    "not_required_exact_closure_sufficient"
                    if not pd.isna(exact_rate) and exact_rate >= settings.min_interval_closure
                    else group_auth["observed_status"]
                ),
            }
        )
    return approved_ids, pd.DataFrame(rows)


def build_tracking_qc(
    episodes: pd.DataFrame,
    linked_actions: pd.DataFrame,
    browser_choice_sets: pd.DataFrame | None = None,
    manual_audit: pd.DataFrame | BrowserGateConfig | AnalysisConfig | Mapping[str, Any] | None = None,
    gates: BrowserGateConfig | AnalysisConfig | Mapping[str, Any] | None = None,
    *,
    audit_sample: pd.DataFrame | None = None,
    reconstruction_qc: pd.DataFrame | None = None,
    interval_audit_qc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate closure, linkage, arm-balance, and manual-audit gates."""

    if browser_choice_sets is not None and not isinstance(browser_choice_sets, pd.DataFrame):
        if gates is not None or manual_audit is not None:
            raise TypeError("Ambiguous positional gate arguments")
        gates = browser_choice_sets
        browser_choice_sets = None
    if gates is None and manual_audit is not None and not isinstance(manual_audit, pd.DataFrame):
        gates = manual_audit
        manual_audit = None
    settings = _gate_settings(gates)
    rows: list[dict[str, Any]] = []

    for group in (*GROUPS, "all"):
        rows.append(
            _action_uniqueness_row(
                linked_actions, group, "linked_action_record_uniqueness"
            )
        )

    for group in (*GROUPS, "all"):
        part = episodes if group == "all" else episodes[episodes.get("group", pd.Series(dtype=object)).eq(group)]
        denominator = len(part)
        numerator = int(part["hazard_eligible"].astype(bool).sum()) if denominator else None
        rows.append(
            _metric_row(
                "interval_closure_rate",
                group,
                numerator,
                denominator if denominator else None,
                settings.min_interval_closure,
                ">=",
            )
        )

    linkage_values: dict[str, float | None] = {}
    for group in (*GROUPS, "all"):
        part = linked_actions if group == "all" else linked_actions[linked_actions.get("group", pd.Series(dtype=object)).eq(group)]
        denominator = len(part)
        linked = (
            part.get("browser_episode_id", pd.Series(index=part.index, dtype=object)).notna()
            & part.get("pre_action_linked", pd.Series(index=part.index, dtype="boolean"))
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            & part.get("action_within_episode", pd.Series(index=part.index, dtype="boolean"))
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            & ~part.get("link_ambiguous", pd.Series(True, index=part.index))
            .astype("boolean")
            .fillna(True)
            .astype(bool)
            & ~part.get("session_ambiguous", pd.Series(True, index=part.index))
            .astype("boolean")
            .fillna(True)
            .astype(bool)
            & ~part.get("backend_duplicate_conflict", pd.Series(True, index=part.index))
            .astype("boolean")
            .fillna(True)
            .astype(bool)
        )
        numerator = int(linked.sum()) if denominator else None
        metric = _metric_row(
            "pre_action_linkage_rate",
            group,
            numerator,
            denominator if denominator else None,
            settings.min_pre_action_linkage,
            ">=",
        )
        rows.append(metric)
        linkage_values[group] = float(metric["value"]) if not pd.isna(metric["value"]) else None

    if linkage_values["G1"] is None or linkage_values["G2"] is None:
        gap_row = _metric_row("arm_pre_action_linkage_gap", "all", None, None, settings.max_arm_linkage_gap, "<=")
    else:
        gap = abs(linkage_values["G1"] - linkage_values["G2"])
        gap_row = {
            "metric": "arm_pre_action_linkage_gap",
            "group": "all",
            "numerator": pd.NA,
            "denominator": pd.NA,
            "value": gap,
            "threshold": settings.max_arm_linkage_gap,
            "comparison": "<=",
            "gate_pass": gap <= settings.max_arm_linkage_gap,
            "observed_status": "observed",
        }
    rows.append(gap_row)

    audit = manual_audit if isinstance(manual_audit, pd.DataFrame) else pd.DataFrame()
    sample = audit_sample if isinstance(audit_sample, pd.DataFrame) else pd.DataFrame()
    rows.extend(_manual_audit_gate_rows(sample, audit, settings))
    if isinstance(reconstruction_qc, pd.DataFrame) and not reconstruction_qc.empty:
        rows.extend(reconstruction_qc.to_dict("records"))
    if isinstance(interval_audit_qc, pd.DataFrame) and not interval_audit_qc.empty:
        rows.extend(interval_audit_qc.to_dict("records"))

    if browser_choice_sets is not None:
        for group in (*GROUPS, "all"):
            part = browser_choice_sets if group == "all" else browser_choice_sets[browser_choice_sets.get("group", pd.Series(dtype=object)).eq(group)]
            denominator = len(part)
            numerator = int(part.get("observed_status", pd.Series(index=part.index, dtype=object)).eq("observed").sum()) if denominator else None
            coverage = _metric_row("choice_row_telemetry_coverage", group, numerator, denominator if denominator else None, 0.0, ">=")
            coverage["gate_pass"] = pd.NA
            rows.append(coverage)

    qc = pd.DataFrame(rows)
    target_required = qc[qc["metric"].isin({
        "linked_action_record_uniqueness",
        "pre_action_linkage_rate",
        "arm_pre_action_linkage_gap",
        "generated_audit_sample_size",
        "generated_audit_sample_uniqueness",
        "manual_audit_action_uniqueness",
        "manual_audit_sample_stratum_match",
        "manual_match_precision",
        "manual_match_precision_weighted",
        "manual_audit_sample_size",
    })]
    target_passes = target_required["gate_pass"].astype("boolean")
    target_pass = bool(
        target_passes.notna().all() and target_passes.fillna(False).all()
    )
    closure = qc[
        qc["metric"].eq("interval_closure_rate") & qc["group"].isin(GROUPS)
    ]["gate_pass"].astype("boolean")
    auxiliary_metrics: set[str] = set()
    if isinstance(reconstruction_qc, pd.DataFrame) and not reconstruction_qc.empty:
        auxiliary_metrics.update(
            {
                "dom_content_event_source_accounting_rate",
                "unmatched_disappear_source_rate",
                "anomalous_episode_exactness_rate",
            }
        )
    if isinstance(interval_audit_qc, pd.DataFrame) and not interval_audit_qc.empty:
        auxiliary_metrics.add("interval_audit_requirement_pass")
    auxiliary = qc[
        qc["metric"].isin(auxiliary_metrics) & qc["group"].isin(GROUPS)
    ]["gate_pass"].astype("boolean")
    expected_auxiliary_rows = 2 * len(auxiliary_metrics)
    auxiliary_complete = len(auxiliary) == expected_auxiliary_rows
    source_integrity_pass = bool(
        auxiliary_complete
        and auxiliary.notna().all()
        and auxiliary.fillna(False).all()
    ) if auxiliary_metrics else True
    full_observed = bool(
        target_passes.notna().all()
        and closure.notna().all()
        and (not auxiliary_metrics or (auxiliary_complete and auxiliary.notna().all()))
    )
    full_pass = bool(
        target_pass
        and closure.notna().all()
        and closure.fillna(False).all()
        and source_integrity_pass
    )
    qc = pd.concat(
        [
            qc,
            pd.DataFrame(
                [
                    {
                        "metric": "target_choice_tracking_gates_pass",
                        "group": "all",
                        "numerator": pd.NA,
                        "denominator": pd.NA,
                        "value": target_pass,
                        "threshold": True,
                        "comparison": "==",
                        "gate_pass": target_pass,
                        "observed_status": (
                            "observed" if target_passes.notna().all() else "unobserved"
                        ),
                    },
                    {
                        "metric": "interval_source_integrity_gates_pass",
                        "group": "all",
                        "numerator": pd.NA,
                        "denominator": pd.NA,
                        "value": source_integrity_pass,
                        "threshold": True,
                        "comparison": "==",
                        "gate_pass": source_integrity_pass,
                        "observed_status": (
                            "observed"
                            if not auxiliary_metrics
                            or (auxiliary_complete and auxiliary.notna().all())
                            else "unobserved"
                        ),
                    },
                    {
                        "metric": "full_interval_hazard_gates_pass",
                        "group": "all",
                        "numerator": pd.NA,
                        "denominator": pd.NA,
                        "value": full_pass,
                        "threshold": True,
                        "comparison": "==",
                        "gate_pass": full_pass,
                        "observed_status": "observed" if full_observed else "unobserved",
                    },
                    {
                        "metric": "all_tracking_gates_pass",
                        "group": "all",
                        "numerator": pd.NA,
                        "denominator": pd.NA,
                        "value": full_pass,
                        "threshold": True,
                        "comparison": "==",
                        "gate_pass": full_pass,
                        "observed_status": "observed" if full_observed else "unobserved",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    qc["gate_pass"] = qc["gate_pass"].astype("boolean")
    return qc


def _conversion_grid(episodes: pd.DataFrame, linked_actions: pd.DataFrame) -> pd.MultiIndex:
    groups = sorted(set(GROUPS) | set(episodes.get("group", pd.Series(dtype=object)).dropna().astype(str)))
    author_types = sorted(
        set(AUTHOR_TYPES)
        | set(episodes.get("author_type_ground_truth", pd.Series(dtype=object)).dropna().map(_author_type).astype(str))
    )
    actions = sorted(set(ACTION_TYPES) | set(linked_actions.get("action_type", pd.Series(dtype=object)).dropna().astype(str)))
    return pd.MultiIndex.from_product(
        [groups, actions, author_types], names=["group", "action_type", "target_author_type"]
    )


def _build_conversion_summary(episodes: pd.DataFrame, linked_actions: pd.DataFrame) -> pd.DataFrame:
    grid = _conversion_grid(episodes, linked_actions)
    base = pd.DataFrame(index=grid).reset_index()
    if episodes is None or episodes.empty:
        for column in (
            "verified_visible_seconds",
            "verified_episodes_n",
            "linked_actions_n",
            "episodes_with_action_n",
            "episode_conversion_rate",
            "action_rate_per_1000_verified_visible_seconds",
            "action_rate_ci_low",
            "action_rate_ci_high",
        ):
            base[column] = pd.NA
        base["observed_status"] = "unobserved"
        return base

    verified = episodes[episodes["hazard_eligible"].astype(bool)].copy()
    verified["target_author_type"] = verified["author_type_ground_truth"].map(_author_type)
    exposure = (
        verified.groupby(["group", "target_author_type"], dropna=False)
        .agg(
            verified_visible_seconds=("verified_visible_seconds", "sum"),
            verified_episodes_n=("browser_episode_id", "nunique"),
        )
        .reset_index()
    )
    linked = linked_actions[
        linked_actions["browser_episode_id"].notna()
        & linked_actions["pre_action_linked"].astype("boolean").fillna(False).astype(bool)
        & linked_actions["action_within_episode"].astype("boolean").fillna(False).astype(bool)
        & ~linked_actions["link_ambiguous"].astype("boolean").fillna(True).astype(bool)
        & ~linked_actions["session_ambiguous"].astype("boolean").fillna(True).astype(bool)
        & ~linked_actions["episode_right_censored"].astype("boolean").fillna(True).astype(bool)
        & linked_actions["episode_interval_verified"].astype("boolean").fillna(False).astype(bool)
        & ~linked_actions["backend_duplicate_conflict"].astype("boolean").fillna(True).astype(bool)
    ].copy()
    linked["target_author_type"] = linked["target_author_type"].map(_author_type)
    action_counts = (
        linked.groupby(["group", "action_type", "target_author_type"], dropna=False)
        .agg(
            linked_actions_n=("backend_action_id", "nunique"),
            episodes_with_action_n=("browser_episode_id", "nunique"),
        )
        .reset_index()
    )
    out = base.merge(exposure, on=["group", "target_author_type"], how="left").merge(
        action_counts, on=["group", "action_type", "target_author_type"], how="left"
    )
    observed = out["verified_episodes_n"].notna()
    out.loc[observed, "linked_actions_n"] = out.loc[observed, "linked_actions_n"].fillna(0)
    out.loc[observed, "episodes_with_action_n"] = out.loc[observed, "episodes_with_action_n"].fillna(0)
    out["episode_conversion_rate"] = out["episodes_with_action_n"] / out["verified_episodes_n"].replace(0, pd.NA)
    out["action_rate_per_1000_verified_visible_seconds"] = (
        1000.0 * out["linked_actions_n"] / out["verified_visible_seconds"].replace(0, pd.NA)
    )
    from scipy.stats import chi2

    counts = pd.to_numeric(out["linked_actions_n"], errors="coerce")
    exposure_seconds = pd.to_numeric(out["verified_visible_seconds"], errors="coerce")
    low_counts = pd.Series(0.0, index=out.index, dtype=float)
    positive = counts.gt(0)
    low_counts.loc[positive] = 0.5 * chi2.ppf(0.025, 2.0 * counts.loc[positive])
    high_counts = 0.5 * chi2.ppf(0.975, 2.0 * (counts + 1.0))
    rate_scale = 1_000.0 / exposure_seconds.where(exposure_seconds.gt(0))
    out["action_rate_ci_low"] = low_counts * rate_scale
    out["action_rate_ci_high"] = high_counts * rate_scale
    out["observed_status"] = observed.map({True: "observed", False: "unobserved"})
    return out


def _read_session_events(browser_dir: Path) -> pd.DataFrame:
    stem = browser_dir / "session_events"
    parquet = stem.with_suffix(".parquet")
    columns = [
        "session_id",
        "event_seq",
        "action",
        "target_action",
        "value",
        "id",
        "item_id",
        "item_type",
        "group",
        "y_user_id",
        "event_timestamp_utc",
        "session_start_utc",
        "session_end_utc",
        "author_id",
        "author_type_ground_truth",
        "target_author_type",
        "is_content_item",
        "backend_content_created_at",
        "root_thread_created_at",
        "experiment_day",
        "valid_primary_event",
        "root_post_id",
        "route_kind",
        "route_path",
    ]
    if parquet.exists():
        try:
            return pd.read_parquet(parquet, columns=columns)
        except (KeyError, ValueError):
            return pd.read_parquet(parquet)
    if not stem.with_suffix(".csv").exists():
        return pd.DataFrame()
    return read_table(stem)


def _read_session_qc(browser_dir: Path) -> pd.DataFrame:
    stem = browser_dir / "session_qc"
    if stem.with_suffix(".parquet").exists() or stem.with_suffix(".csv").exists():
        return read_table(stem)
    return pd.DataFrame()


def _filter_analysis_session_events(
    config: AnalysisConfig,
    session_events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the publication window and usable-session cohort before reconstruction."""

    events = session_events.copy()
    raw_rows = len(events)
    raw_sessions = int(events["session_id"].nunique())
    event_time = _timestamp_series(events, ("event_timestamp_utc", "event_timestamp", "t"))
    event_day = experiment_day_series(event_time)
    expected_days = set(analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"])
    start = parse_time(config.start_time).tz_convert("UTC")
    keep = event_time.notna() & event_time.ge(start) & event_day.isin(expected_days)
    if config.end_time is not None:
        end = parse_time(config.end_time).tz_convert("UTC")
        keep &= event_time.lt(end)

    if "is_content_item" in events.columns:
        is_content = events["is_content_item"].fillna(False).astype(bool)
    else:
        is_content = events["action"].astype(str).str.lower().isin(["appear", "disappear"])
    if {"backend_content_created_at", "root_thread_created_at"}.issubset(events.columns):
        content_time = pd.to_datetime(events["backend_content_created_at"], utc=True, errors="coerce")
        root_time = pd.to_datetime(events["root_thread_created_at"], utc=True, errors="coerce")
        content_valid = (
            content_time.notna()
            & root_time.notna()
            & content_time.ge(start)
            & root_time.ge(start)
            & experiment_day_series(content_time).isin(expected_days)
            & experiment_day_series(root_time).isin(expected_days)
        )
        keep &= ~is_content | content_valid

    session_qc = _read_session_qc(config.browser_dir)
    usable_sessions: set[str] | None = None
    if not session_qc.empty:
        if not {"session_id", "usable_session"}.issubset(session_qc.columns):
            raise ValueError("session_qc requires session_id and usable_session")
        usable_sessions = set(
            session_qc.loc[
                session_qc["usable_session"].fillna(False).astype(bool), "session_id"
            ].astype(str)
        )
        keep &= events["session_id"].astype(str).isin(usable_sessions)
    filtered = events.loc[keep].copy()
    # A surviving route event is useful only when its session contains at least
    # one valid rendered content event.
    content_sessions = set(filtered.loc[is_content.loc[filtered.index], "session_id"].astype(str))
    filtered = filtered[filtered["session_id"].astype(str).isin(content_sessions)].copy()
    filtered_sessions = int(filtered["session_id"].nunique())
    rows = [
        {
            "metric": "analysis_window_event_rows_retained",
            "group": "all",
            "numerator": int(len(filtered)),
            "denominator": raw_rows,
            "value": len(filtered) / raw_rows if raw_rows else pd.NA,
            "threshold": pd.NA,
            "comparison": "descriptive",
            "gate_pass": True,
            "observed_status": "observed",
        },
        {
            "metric": "usable_valid_sessions_retained",
            "group": "all",
            "numerator": filtered_sessions,
            "denominator": raw_sessions,
            "value": filtered_sessions / raw_sessions if raw_sessions else pd.NA,
            "threshold": pd.NA,
            "comparison": "descriptive",
            "gate_pass": True,
            "observed_status": "observed" if usable_sessions is not None else "qc_file_unavailable",
        },
        {
            "metric": "analysis_filter_applied",
            "group": "all",
            "numerator": 1,
            "denominator": 1,
            "value": True,
            "threshold": True,
            "comparison": "==",
            "gate_pass": True,
            "observed_status": "observed",
        },
    ]
    return filtered, rows


def _read_optional_manual_audit(browser_dir: Path) -> pd.DataFrame:
    for name in ("browser_manual_audit", "manual_tracking_audit", "session_manual_audit", "manual_audit"):
        stem = browser_dir / name
        if stem.with_suffix(".parquet").exists() or stem.with_suffix(".csv").exists():
            return read_table(stem)
    return pd.DataFrame()


def _read_optional_interval_audit(browser_dir: Path) -> pd.DataFrame:
    stem = browser_dir / "browser_interval_audit"
    if stem.with_suffix(".parquet").exists() or stem.with_suffix(".csv").exists():
        return read_table(stem)
    return pd.DataFrame()


def _apply_interval_audit(
    episodes: pd.DataFrame,
    audit_sample: pd.DataFrame,
    interval_audit: pd.DataFrame,
    settings: BrowserGateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only complete, current-sample, authenticated interval approvals."""

    approved_ids, audit_qc = _authenticate_interval_audit(
        episodes,
        audit_sample,
        interval_audit,
        settings,
    )
    out = episodes.copy()
    selected = (
        out["browser_episode_id"].astype(str).isin(approved_ids)
        & ~out["right_censored"].astype("boolean").fillna(True).astype(bool)
        & ~out["interval_exact"].astype("boolean").fillna(False).astype(bool)
    )
    out.loc[selected, "interval_audit_approved"] = True
    out.loc[selected, "hazard_eligible"] = True
    out.loc[selected, "verified_visible_seconds"] = out.loc[
        selected, "observed_elapsed_seconds"
    ]
    out.loc[selected, "visible_duration_s"] = out.loc[
        selected, "observed_elapsed_seconds"
    ]
    return out, audit_qc


def prepare_browser(
    config: AnalysisConfig,
    event_history: pd.DataFrame,
    thread_choice_sets: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Prepare deterministic browser opportunity artifacts for publication models."""

    session_events = _read_session_events(config.browser_dir)
    session_events, filter_qc = _filter_analysis_session_events(config, session_events)
    manual_audit = _read_optional_manual_audit(config.browser_dir)
    raw_episodes = reconstruct_dom_episodes(session_events)
    reconstruction_qc = _episode_reconstruction_qc(session_events, raw_episodes)
    interval_audit_sample = build_interval_audit_sample(
        raw_episodes,
        per_arm=(
            config.browser_gates.interval_audit_per_arm
            if config.browser_gates.interval_audit_per_arm is not None
            else config.browser_gates.manual_audit_per_arm
        ),
        seed=config.seed,
    )
    episodes, interval_audit_qc = _apply_interval_audit(
        raw_episodes,
        interval_audit_sample,
        _read_optional_interval_audit(config.browser_dir),
        config.browser_gates,
    )
    links = link_backend_actions(
        episodes,
        session_events,
        event_history,
        config.browser_gates.action_tolerance_seconds,
    )
    audit_sample = build_manual_audit_sample(
        links,
        per_arm=config.browser_gates.manual_audit_per_arm,
        seed=config.seed,
    )
    choice_sets = build_browser_choice_sets(thread_choice_sets, episodes, links)
    qc = build_tracking_qc(
        episodes,
        links,
        choice_sets,
        manual_audit,
        config.browser_gates,
        audit_sample=audit_sample,
        reconstruction_qc=reconstruction_qc,
        interval_audit_qc=interval_audit_qc,
    )
    qc = pd.concat([qc, pd.DataFrame(filter_qc)], ignore_index=True)
    conversion = _build_conversion_summary(episodes, links)
    return {
        "browser_opportunity_episodes": episodes,
        "browser_linked_actions": links,
        "browser_manual_audit_sample": audit_sample,
        "browser_interval_audit_sample": interval_audit_sample,
        "browser_choice_sets": choice_sets,
        "tracking_linkage_qc": qc,
        "browser_conversion_summary": conversion,
    }


def browser_gate_metadata(gates: BrowserGateConfig) -> dict[str, Any]:
    """Expose gate settings for provenance manifests."""

    return asdict(gates)
