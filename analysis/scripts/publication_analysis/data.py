"""Canonical publication data and same-thread risk-set construction.

The functions in this module rebuild the valid analysis cohort from normalized
snapshot tables.  All dynamic covariates use information strictly earlier than
the focal event.  Invalid experiment days are removed and elapsed history time
is measured on a clock that joins consecutive valid analysis days.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .study import (
    GROUPS,
    experiment_day_series,
    filter_main_contents,
    filter_main_events,
    load_contents,
    load_events,
    normalize_timestamp_series,
    parse_time,
    read_table,
    sequential_valid_day_index,
    compute_depths,
)

from .config import AnalysisConfig
from .io import read_table as read_publication_table


COMMENT_EVENT = "comments_replies"
REACTION_DELTAS = {"upvote": 1.0, "downvote": -1.0}
SECONDS_PER_DAY = 86_400.0

__all__ = [
    "build_event_history",
    "build_source_availability",
    "build_thread_choice_sets",
    "build_network_history",
    "build_account_universe",
    "build_actor_label_qc",
    "build_group_database_provenance",
    "prepare_data",
]


def build_source_availability(config: AnalysisConfig) -> pd.DataFrame:
    """Summarize whether planned time-varying adjustment sources are usable."""

    raw_events = load_events(config.processed_dir)
    rows: list[dict[str, Any]] = []
    for group in sorted(raw_events["group"].astype(str).dropna().unique()):
        arm = raw_events[raw_events["group"].astype(str).eq(group)]
        reactions = arm[arm["event_type"].isin(REACTION_DELTAS)]
        timestamped = int(reactions["event_timestamp"].notna().sum())
        rows.append(
            {
                "group": group,
                "source": "backend_reaction_timestamps",
                "available_rows": timestamped,
                "total_rows": int(len(reactions)),
                "coverage": timestamped / len(reactions) if len(reactions) else np.nan,
                "status": "available" if timestamped else "unavailable_missing_event_timestamp",
            }
        )
        for event_type in REACTION_DELTAS:
            action = reactions[reactions["event_type"].eq(event_type)]
            action_timestamped = int(action["event_timestamp"].notna().sum())
            rows.append(
                {
                    "group": group,
                    "source": f"backend_{event_type}_timestamps",
                    "available_rows": action_timestamped,
                    "total_rows": int(len(action)),
                    "coverage": action_timestamped / len(action) if len(action) else np.nan,
                    "status": (
                        "available"
                        if action_timestamped
                        else "unavailable_missing_event_timestamp"
                    ),
                }
            )
    reputation = _load_reputation_history(config)
    rows.append(
        {
            "group": "all",
            "source": "snapshot_user_reputation_history",
            "available_rows": int(len(reputation)),
            "total_rows": int(len(reputation)),
            "coverage": 1.0 if len(reputation) else np.nan,
            "status": "available" if len(reputation) else "unavailable_missing_table_or_timestamp",
        }
    )
    return pd.DataFrame(rows)


def _authoritative_accounts(config: AnalysisConfig) -> pd.DataFrame:
    contents = load_contents(config.processed_dir)
    events = load_events(config.processed_dir)
    observations = [
        contents[["group", "author_id", "author_type"]].rename(
            columns={"author_id": "account_id", "author_type": "account_type"}
        ),
        events[["group", "actor_id", "actor_type"]].rename(
            columns={"actor_id": "account_id", "actor_type": "account_type"}
        ),
        events[["group", "target_author_id", "target_author_type"]].rename(
            columns={"target_author_id": "account_id", "target_author_type": "account_type"}
        ),
    ]
    accounts = pd.concat(observations, ignore_index=True).dropna(
        subset=["group", "account_id", "account_type"]
    )
    accounts["group"] = accounts["group"].astype(str)
    accounts["account_key"] = accounts["account_id"].map(_id_key)
    accounts["account_type"] = accounts["account_type"].astype(str).str.upper()
    accounts = accounts[accounts["account_type"].isin(["H", "AI"])].dropna(
        subset=["account_key"]
    )
    return accounts


def build_account_universe(config: AnalysisConfig) -> pd.DataFrame:
    """Return every authoritative arm-scoped account, including structural zeros."""

    accounts = _authoritative_accounts(config)
    type_counts = accounts.groupby(["group", "account_key"])["account_type"].nunique()
    if (type_counts > 1).any():
        raise ValueError("Authoritative account universe contains H/AI label conflicts")
    universe = accounts.drop_duplicates(["group", "account_key"], keep="first").copy()
    universe["account_id"] = universe["account_key"]
    return (
        universe[["group", "account_id", "account_type"]]
        .sort_values(["group", "account_type", "account_id"], kind="stable")
        .reset_index(drop=True)
        .assign(source="normalized_contents_events_authoritative_type_fields")
    )


def build_actor_label_qc(config: AnalysisConfig) -> pd.DataFrame:
    """Reconcile account labels across normalized content and event sources."""

    accounts = _authoritative_accounts(config)
    type_counts = accounts.groupby(["group", "account_key"])["account_type"].nunique()
    conflicts = type_counts[type_counts > 1]
    rows: list[dict[str, Any]] = []
    for group in sorted(GROUPS):
        group_accounts = accounts[accounts["group"].eq(group)]
        group_conflicts = int(
            sum(index[0] == group for index in conflicts.index)
        )
        for account_type, expected in (
            ("H", GROUPS[group].human_count),
            ("AI", GROUPS[group].ai_count),
        ):
            observed = int(
                group_accounts.loc[
                    group_accounts["account_type"].eq(account_type), "account_key"
                ].nunique()
            )
            rows.append(
                {
                    "group": group,
                    "account_type": account_type,
                    "observed_accounts": observed,
                    "expected_accounts": expected,
                    "label_conflicts": group_conflicts,
                    "passed": observed == expected and group_conflicts == 0,
                    "source": "normalized_contents_events_authoritative_type_fields",
                }
            )
    return pd.DataFrame(rows)


def _source_manifest_path(config: AnalysisConfig) -> Path:
    if config.source_manifest is not None:
        return config.source_manifest
    return (
        config.project_root
        / "data"
        / "raw"
        / "datasets"
        / config.processed_dir.name
        / "metadata"
        / "manifest.json"
    )


def _canonical_manifest_group(record: Mapping[str, Any]) -> str | None:
    for field_name in ("group", "label"):
        value = record.get(field_name)
        if value is None:
            continue
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"g1", "grupa1", "grupa_1"}:
            return "G1"
        if normalized in {"g2", "grupa2", "grupa_2"}:
            return "G2"
    return None


def build_group_database_provenance(config: AnalysisConfig) -> pd.DataFrame:
    """Read arm/database identities from the immutable dataset source manifest."""

    manifest_path = _source_manifest_path(config)
    base = {
        "provenance_path": str(manifest_path),
        "source_dataset": pd.NA,
        "source_group": pd.NA,
        "source_label": pd.NA,
        "source_db_name": pd.NA,
        "source_experiment_id": pd.NA,
    }

    def unavailable(status: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"group": group, **base, "provenance_status": status} for group in sorted(GROUPS)]
        )

    if not manifest_path.exists() or not manifest_path.is_file():
        return unavailable("missing_source_manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unavailable("unreadable_source_manifest")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("groups"), list):
        return unavailable("invalid_source_manifest_groups")

    records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw_record in manifest["groups"]:
        if not isinstance(raw_record, Mapping):
            continue
        group = _canonical_manifest_group(raw_record)
        if group is not None:
            records[group].append(raw_record)

    rows: list[dict[str, Any]] = []
    for group in sorted(GROUPS):
        matches = records[group]
        if len(matches) != 1:
            status = "missing_group_record" if not matches else "ambiguous_group_records"
            rows.append({"group": group, **base, "provenance_status": status})
            continue
        record = matches[0]
        rows.append(
            {
                "group": group,
                "provenance_path": str(manifest_path),
                "source_dataset": manifest.get("dataset_name", pd.NA),
                "source_group": record.get("group", pd.NA),
                "source_label": record.get("label", pd.NA),
                "source_db_name": record.get("db", record.get("db_name", pd.NA)),
                "source_experiment_id": record.get(
                    "idexp", record.get("experiment_id", pd.NA)
                ),
                "provenance_status": "established_from_source_manifest",
            }
        )
    return pd.DataFrame(rows)


def _id_key(value: Any) -> str | None:
    """Return a stable group-local identifier across int/float/string inputs."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _valid_clock_seconds(timestamps: pd.Series, day_index: pd.Series) -> pd.Series:
    """Compress excluded days while preserving within-day event time."""

    parsed = normalize_timestamp_series(timestamps)
    day = pd.to_numeric(day_index, errors="coerce")
    seconds_in_day = (parsed - parsed.dt.normalize()).dt.total_seconds()
    clock = (day - 1.0) * SECONDS_PER_DAY + seconds_in_day
    return clock.where(parsed.notna() & day.notna()).astype(float)


def _ensure_event_clock(frame: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    out = frame.copy()
    out["event_timestamp"] = normalize_timestamp_series(out["event_timestamp"])
    if "experiment_day" not in out.columns:
        out["experiment_day"] = experiment_day_series(out["event_timestamp"])
    if "valid_day_index" not in out.columns:
        out["valid_day_index"] = sequential_valid_day_index(out["experiment_day"], config.analysis_set)
    out["valid_clock_seconds"] = _valid_clock_seconds(out["event_timestamp"], out["valid_day_index"])
    out["valid_clock_hours"] = out["valid_clock_seconds"] / 3_600.0
    return out


def build_event_history(config: AnalysisConfig) -> pd.DataFrame:
    """Load and recompute the configured valid event history.

    The returned table preserves the normalized human -> content -> author
    fields and adds a sequential valid-day clock.  Events from excluded days do
    not appear in the table or consume time on that clock.
    """

    events = load_events(config.processed_dir)
    events = filter_main_events(
        events,
        start_time=config.start_time,
        end_time=config.end_time,
        analysis_set=config.analysis_set,
    )
    events = _ensure_event_clock(events, config)
    events["group"] = events["group"].astype(str)
    events["human_originated"] = events["actor_type"].astype(str).str.upper().eq("H")
    events["target_is_ai"] = events["target_author_type"].astype(str).str.upper().eq("AI")
    return events.sort_values(
        ["group", "event_timestamp", "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _content_depth(contents: pd.DataFrame) -> pd.Series:
    if "comment_to" in contents.columns:
        return compute_depths(contents)
    is_root = contents["content_id"].map(_id_key).eq(contents["root_id"].map(_id_key))
    return pd.Series(np.where(is_root, 0, 1), index=contents.index, dtype=int)


def _topic_table(config: AnalysisConfig) -> pd.DataFrame:
    if config.legacy_tables_dir is None:
        return pd.DataFrame()
    stems = [
        "22_interest_event_panel",
        "70_interest_topic_affordance_addon_event_panel",
    ]
    topic_columns = [
        "primary_root_topic",
        "root_topics",
        "topic_affordance",
        "topic_assignment_source",
    ]
    for stem in stems:
        try:
            frame = read_publication_table(config.legacy_tables_dir / stem)
        except FileNotFoundError:
            continue
        available = [column for column in topic_columns if column in frame.columns]
        if not {"group", "root_id"}.issubset(frame.columns) or not available:
            continue
        topics = frame[["group", "root_id", *available]].copy()
        topics["group"] = topics["group"].astype(str)
        return topics.drop_duplicates(["group", "root_id"], keep="first")
    return pd.DataFrame()


def _attach_topics(contents: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    out = contents.copy()
    topics = _topic_table(config)
    if not topics.empty:
        missing = [
            column
            for column in topics.columns
            if column not in {"group", "root_id"} and column not in out.columns
        ]
        if missing:
            out = out.merge(
                topics[["group", "root_id", *missing]],
                on=["group", "root_id"],
                how="left",
                validate="many_to_one",
            )
    for column in [
        "primary_root_topic",
        "root_topics",
        "topic_affordance",
        "topic_assignment_source",
    ]:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _valid_contents(config: AnalysisConfig) -> pd.DataFrame:
    contents = load_contents(config.processed_dir)
    contents["candidate_depth"] = _content_depth(contents)
    contents = filter_main_contents(
        contents,
        start_time=config.start_time,
        end_time=config.end_time,
        analysis_set=config.analysis_set,
    )
    contents["group"] = contents["group"].astype(str)
    contents["content_valid_clock_seconds"] = _valid_clock_seconds(
        contents["created_at"], contents["valid_day_index"]
    )
    return _attach_topics(contents, config)


def _timestamp_column(frame: pd.DataFrame) -> str | None:
    for column in ["timestamp", "event_timestamp", "updated_at", "last_computed_ts", "round_timestamp"]:
        if column in frame.columns:
            return column
    return None


def _load_reputation_history(config: AnalysisConfig) -> pd.DataFrame:
    try:
        history = read_table(config.processed_dir / "snapshot_user_reputation_history")
    except FileNotFoundError:
        return pd.DataFrame()
    timestamp_column = _timestamp_column(history)
    if timestamp_column is None or not {"group", "user_id"}.issubset(history.columns):
        return pd.DataFrame()
    out = history.copy()
    out["timestamp"] = normalize_timestamp_series(out[timestamp_column])
    out["experiment_day"] = experiment_day_series(out["timestamp"])
    out["valid_day_index"] = sequential_valid_day_index(out["experiment_day"], config.analysis_set)
    out["valid_clock_seconds"] = _valid_clock_seconds(out["timestamp"], out["valid_day_index"])
    start = parse_time(config.start_time)
    end = parse_time(config.end_time)
    keep = out["timestamp"].notna() & out["valid_day_index"].notna() & out["timestamp"].ge(start)
    if end is not None:
        keep &= out["timestamp"].lt(end)
    out = out.loc[keep].copy()
    for column in ["reputation", "karma"]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["group"] = out["group"].astype(str)
    out["user_key"] = out["user_id"].map(_id_key)
    return out.dropna(subset=["user_key"]).sort_values(
        ["group", "user_key", "timestamp"], kind="mergesort"
    )


def _reputation_timelines(history: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    timelines: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    if history.empty:
        return timelines
    for key, part in history.groupby(["group", "user_key"], sort=False):
        ordered = part.sort_values("timestamp", kind="mergesort")
        timelines[(str(key[0]), str(key[1]))] = {
            "times": ordered["timestamp"].astype("int64").to_numpy(dtype=np.int64),
            "clock": ordered["valid_clock_seconds"].to_numpy(dtype=float),
            "reputation": ordered["reputation"].to_numpy(dtype=float),
            "karma": ordered["karma"].to_numpy(dtype=float),
        }
    return timelines


def _prior_reputation(
    timelines: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    group: str,
    user_id: Any,
    event_timestamp: pd.Timestamp,
    event_clock: float,
    history_available: bool,
) -> dict[str, Any]:
    user_key = _id_key(user_id)
    timeline = timelines.get((str(group), user_key)) if user_key is not None else None
    if timeline is None:
        return {
            "candidate_reputation_at_event": math.nan,
            "candidate_karma_at_event": math.nan,
            "candidate_reputation_timestamp": pd.NaT,
            "candidate_reputation_age_valid_hours": math.nan,
            "candidate_reputation_source": (
                "no_prior_record" if history_available else "unavailable_no_reputation_history_table"
            ),
        }
    index = int(np.searchsorted(timeline["times"], pd.Timestamp(event_timestamp).value, side="left") - 1)
    if index < 0:
        return {
            "candidate_reputation_at_event": math.nan,
            "candidate_karma_at_event": math.nan,
            "candidate_reputation_timestamp": pd.NaT,
            "candidate_reputation_age_valid_hours": math.nan,
            "candidate_reputation_source": "no_prior_record",
        }
    timestamp = pd.Timestamp(int(timeline["times"][index]), tz="UTC").tz_convert("Europe/Belgrade")
    prior_clock = float(timeline["clock"][index])
    return {
        "candidate_reputation_at_event": float(timeline["reputation"][index]),
        "candidate_karma_at_event": float(timeline["karma"][index]),
        "candidate_reputation_timestamp": timestamp,
        "candidate_reputation_age_valid_hours": (float(event_clock) - prior_clock) / 3_600.0,
        "candidate_reputation_source": "snapshot_user_reputation_history_latest_prior",
    }


def _candidate_rows_for_event(
    event: Any,
    candidates: pd.DataFrame,
    reputation: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    reputation_history_available: bool,
) -> list[dict[str, Any]]:
    group = str(event.group)
    event_timestamp = pd.Timestamp(event.event_timestamp)
    event_clock = float(event.valid_clock_seconds)
    target_key = _id_key(event.target_content_id)
    candidate_n = int(len(candidates))
    type_levels = int(candidates["author_type"].dropna().astype(str).nunique())
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        candidate_clock = float(candidate.content_valid_clock_seconds)
        candidate_key = _id_key(candidate.content_id)
        row = {
            "choice_set_id": f"{group}:{event.event_id}",
            "group": group,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "event_timestamp": event.event_timestamp,
            "event_valid_day_index": int(event.valid_day_index),
            "event_valid_clock_seconds": event_clock,
            "actor_id": event.actor_id,
            "actor_type": event.actor_type,
            "target_content_id": event.target_content_id,
            "target_author_id": event.target_author_id,
            "target_author_type": event.target_author_type,
            "root_id": event.root_id,
            "root_thread_created_at": event.root_thread_created_at,
            "candidate_content_id": candidate.content_id,
            "candidate_author_id": candidate.author_id,
            "candidate_author_type": candidate.author_type,
            "candidate_target_is_ai": str(candidate.author_type).upper() == "AI",
            "candidate_created_at": candidate.created_at,
            "candidate_valid_day_index": int(candidate.valid_day_index),
            "candidate_valid_clock_seconds": candidate_clock,
            "candidate_age_hours": (event_clock - candidate_clock) / 3_600.0,
            "candidate_elapsed_age_hours": (event_timestamp - pd.Timestamp(candidate.created_at)).total_seconds()
            / 3_600.0,
            "candidate_depth": int(candidate.candidate_depth),
            "candidate_is_root": _id_key(candidate.content_id) == _id_key(candidate.root_id),
            "chosen": candidate_key == target_key,
            "choice_set_candidate_n": candidate_n,
            "choice_set_author_type_levels": type_levels,
            "choice_set_has_author_type_variation": type_levels >= 2,
            "primary_root_topic": candidate.primary_root_topic,
            "root_topics": candidate.root_topics,
            "topic_affordance": candidate.topic_affordance,
            "topic_assignment_source": candidate.topic_assignment_source,
        }
        row.update(
            _prior_reputation(
                reputation,
                group,
                candidate.author_id,
                event_timestamp,
                event_clock,
                reputation_history_available,
            )
        )
        rows.append(row)
    return rows


def build_thread_choice_sets(
    config: AnalysisConfig,
    event_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build publication same-thread choice sets for human replies.

    Candidates are valid prior contents in the same root thread.  Self-authored
    candidates are excluded.  A set is retained when it contains at least two
    candidates and exactly one selected target; H/AI variation is descriptive,
    not an eligibility requirement.
    """

    events = build_event_history(config) if event_history is None else _ensure_event_clock(event_history, config)
    contents = _valid_contents(config)
    reputation_history = _load_reputation_history(config)
    reputation = _reputation_timelines(reputation_history)
    by_root = {
        (str(group), _id_key(root_id)): part.sort_values(["created_at", "content_id"], kind="mergesort")
        for (group, root_id), part in contents.groupby(["group", "root_id"], sort=False, dropna=False)
    }
    human_replies = events[
        events["event_type"].eq(COMMENT_EVENT)
        & events["actor_type"].astype(str).str.upper().eq("H")
    ].sort_values(["group", "event_timestamp", "event_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    attrition: list[dict[str, Any]] = []
    for event in human_replies.itertuples(index=False):
        candidates = by_root.get((str(event.group), _id_key(event.root_id)))
        if candidates is None:
            attrition.append(
                {"group": str(event.group), "event_id": event.event_id, "outcome": "no_valid_thread"}
            )
            continue
        prior = candidates[
            candidates["created_at"].notna() & candidates["created_at"].lt(event.event_timestamp)
        ].copy()
        actor_key = _id_key(event.actor_id)
        prior = prior[prior["author_id"].map(_id_key).ne(actor_key)].copy()
        if len(prior) < 2:
            attrition.append(
                {
                    "group": str(event.group),
                    "event_id": event.event_id,
                    "outcome": "fewer_than_two_eligible_candidates",
                }
            )
            continue
        chosen_count = int(prior["content_id"].map(_id_key).eq(_id_key(event.target_content_id)).sum())
        if chosen_count != 1:
            attrition.append(
                {
                    "group": str(event.group),
                    "event_id": event.event_id,
                    "outcome": (
                        "chosen_target_absent_after_filter"
                        if chosen_count == 0
                        else "chosen_target_duplicated"
                    ),
                }
            )
            continue
        attrition.append(
            {"group": str(event.group), "event_id": event.event_id, "outcome": "retained"}
        )
        rows.extend(
            _candidate_rows_for_event(
                event,
                prior,
                reputation,
                reputation_history_available=not reputation_history.empty,
            )
        )
    choice_sets = pd.DataFrame(rows)
    if choice_sets.empty:
        choice_sets.attrs["attrition_events"] = pd.DataFrame(attrition)
        return choice_sets
    choice_sets["chosen"] = choice_sets["chosen"].astype(bool)
    choice_sets = choice_sets.sort_values(
        ["group", "event_timestamp", "choice_set_id", "candidate_created_at", "candidate_content_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    choice_sets.attrs["attrition_events"] = pd.DataFrame(attrition)
    return choice_sets


def _choice_set_attrition_summary(attrition_events: pd.DataFrame) -> pd.DataFrame:
    if attrition_events.empty:
        return pd.DataFrame(
            columns=["group", "outcome", "human_reply_events", "all_human_replies", "share"]
        )
    totals = attrition_events.groupby("group")["event_id"].nunique()
    summary = (
        attrition_events.groupby(["group", "outcome"])["event_id"]
        .nunique()
        .rename("human_reply_events")
        .reset_index()
    )
    summary["all_human_replies"] = summary["group"].map(totals).astype(int)
    summary["share"] = summary["human_reply_events"] / summary["all_human_replies"]
    return summary.sort_values(["group", "outcome"], kind="stable").reset_index(drop=True)


def _risk_set_composition(choice_sets: pd.DataFrame) -> pd.DataFrame:
    if choice_sets.empty:
        return pd.DataFrame(columns=["group", "composition", "choice_sets", "candidate_rows"])
    set_rows = choice_sets.drop_duplicates("choice_set_id")
    composition = np.where(
        set_rows["choice_set_has_author_type_variation"].astype(bool),
        "mixed_H_AI",
        "single_author_type",
    )
    set_lookup = dict(zip(set_rows["choice_set_id"], composition))
    work = choice_sets.assign(composition=choice_sets["choice_set_id"].map(set_lookup))
    return (
        work.groupby(["group", "composition"], as_index=False)
        .agg(
            choice_sets=("choice_set_id", "nunique"),
            candidate_rows=("choice_set_id", "size"),
        )
        .sort_values(["group", "composition"], kind="stable")
        .reset_index(drop=True)
    )


def _unordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _time_since_hours(current: float, previous: float | None) -> float:
    if previous is None or not math.isfinite(previous):
        return math.nan
    return max(0.0, (current - previous) / 3_600.0)


@dataclass
class _NetworkState:
    edge_count: defaultdict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    edge_last_clock: dict[tuple[str, str], float] = field(default_factory=dict)
    receiver_count: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    receiver_human_count: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    receiver_senders: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    out_neighbors: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    in_neighbors: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    root_participants: defaultdict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    shared_thread_count: defaultdict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def add(self, event: Any) -> None:
        sender = _id_key(event.actor_id)
        receiver = _id_key(event.target_author_id)
        if sender is None or receiver is None or sender == receiver:
            return
        edge = (sender, receiver)
        self.edge_count[edge] += 1
        self.edge_last_clock[edge] = float(event.valid_clock_seconds)
        self.receiver_count[receiver] += 1
        if str(event.actor_type).upper() == "H":
            self.receiver_human_count[receiver] += 1
        self.receiver_senders[receiver].add(sender)
        self.out_neighbors[sender].add(receiver)
        self.in_neighbors[receiver].add(sender)
        root = _id_key(event.root_id)
        if root is None:
            return
        participants = self.root_participants[root]
        for participant in (sender, receiver):
            if participant in participants:
                continue
            for other in participants:
                self.shared_thread_count[_unordered_pair(participant, other)] += 1
            participants.add(participant)

    def features(self, actor_id: Any, receiver_id: Any, current_clock: float) -> dict[str, Any]:
        actor = _id_key(actor_id)
        receiver = _id_key(receiver_id)
        if actor is None or receiver is None:
            return _empty_network_features()
        forward = (actor, receiver)
        reverse = (receiver, actor)
        intermediaries = self.out_neighbors[actor].intersection(self.in_neighbors[receiver])
        return {
            "prior_human_to_author_count": int(self.edge_count[forward]),
            "prior_human_to_author_time_since_hours": _time_since_hours(
                current_clock, self.edge_last_clock.get(forward)
            ),
            "prior_author_to_human_count": int(self.edge_count[reverse]),
            "prior_author_to_human_time_since_hours": _time_since_hours(
                current_clock, self.edge_last_clock.get(reverse)
            ),
            "receiver_popularity_prior_count": int(self.receiver_count[receiver]),
            "receiver_popularity_prior_human_count": int(self.receiver_human_count[receiver]),
            "receiver_popularity_prior_unique_senders": int(len(self.receiver_senders[receiver])),
            "prior_shared_thread_count": int(self.shared_thread_count[_unordered_pair(actor, receiver)]),
            "prior_two_path_count": int(len(intermediaries - {actor, receiver})),
        }


def _empty_network_features() -> dict[str, Any]:
    return {
        "prior_human_to_author_count": 0,
        "prior_human_to_author_time_since_hours": math.nan,
        "prior_author_to_human_count": 0,
        "prior_author_to_human_time_since_hours": math.nan,
        "receiver_popularity_prior_count": 0,
        "receiver_popularity_prior_human_count": 0,
        "receiver_popularity_prior_unique_senders": 0,
        "prior_shared_thread_count": 0,
        "prior_two_path_count": 0,
    }


def _reaction_timelines(
    events: pd.DataFrame,
    key_column: str,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    reactions = events[events["event_type"].isin(REACTION_DELTAS)].copy()
    reactions["item_key"] = reactions[key_column].map(_id_key)
    reactions = reactions.dropna(subset=["group", "item_key", "event_timestamp"])
    if reactions.empty:
        return {}
    reactions["delta"] = reactions["event_type"].map(REACTION_DELTAS).astype(float)
    timelines: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, part in reactions.groupby(["group", "item_key"], sort=False):
        ordered = part.sort_values("event_timestamp", kind="mergesort")
        delta = ordered["delta"].to_numpy(dtype=float)
        timelines[(str(key[0]), str(key[1]))] = {
            "times": ordered["event_timestamp"].astype("int64").to_numpy(dtype=np.int64),
            "score": np.cumsum(delta),
            "upvotes": np.cumsum(delta > 0),
            "downvotes": np.cumsum(delta < 0),
        }
    return timelines


def _prior_reactions(
    timelines: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    group: str,
    item_id: Any,
    event_timestamp: pd.Timestamp,
) -> tuple[float, int, int]:
    item_key = _id_key(item_id)
    timeline = timelines.get((str(group), item_key)) if item_key is not None else None
    if timeline is None:
        return 0.0, 0, 0
    index = int(np.searchsorted(timeline["times"], pd.Timestamp(event_timestamp).value, side="left") - 1)
    if index < 0:
        return 0.0, 0, 0
    return (
        float(timeline["score"][index]),
        int(timeline["upvotes"][index]),
        int(timeline["downvotes"][index]),
    )


def _add_reaction_history(choice_sets: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = choice_sets.copy()
    timestamped_reactions = events[
        events["event_type"].isin(REACTION_DELTAS) & events["event_timestamp"].notna()
    ]
    if timestamped_reactions.empty:
        for column in [
            "candidate_content_prior_reaction_score",
            "candidate_content_prior_upvotes",
            "candidate_content_prior_downvotes",
            "candidate_author_prior_reaction_score",
            "candidate_author_prior_upvotes",
            "candidate_author_prior_downvotes",
        ]:
            out[column] = np.nan
        out["prior_reaction_history_source"] = "unavailable_no_timestamped_reactions"
        return out
    content = _reaction_timelines(events, "target_content_id")
    author = _reaction_timelines(events, "target_author_id")
    values: list[dict[str, Any]] = []
    for row in out.itertuples(index=False):
        content_score, content_up, content_down = _prior_reactions(
            content, row.group, row.candidate_content_id, row.event_timestamp
        )
        author_score, author_up, author_down = _prior_reactions(
            author, row.group, row.candidate_author_id, row.event_timestamp
        )
        values.append(
            {
                "candidate_content_prior_reaction_score": content_score,
                "candidate_content_prior_upvotes": content_up,
                "candidate_content_prior_downvotes": content_down,
                "candidate_author_prior_reaction_score": author_score,
                "candidate_author_prior_upvotes": author_up,
                "candidate_author_prior_downvotes": author_down,
            }
        )
    enriched = pd.concat([out.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    enriched["prior_reaction_history_source"] = "timestamped_backend_reactions"
    return enriched


def _network_features_for_group(
    choice_sets: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    state = _NetworkState()
    history_rows = list(history.itertuples(index=False))
    pointer = 0
    output: list[pd.DataFrame] = []
    ordered = choice_sets.sort_values(["event_timestamp", "choice_set_id"], kind="mergesort")
    for _, focal in ordered.groupby("choice_set_id", sort=False):
        event_timestamp = pd.Timestamp(focal["event_timestamp"].iloc[0])
        while pointer < len(history_rows) and pd.Timestamp(history_rows[pointer].event_timestamp) < event_timestamp:
            state.add(history_rows[pointer])
            pointer += 1
        current_clock = float(focal["event_valid_clock_seconds"].iloc[0])
        feature_rows = [
            state.features(row.actor_id, row.candidate_author_id, current_clock)
            for row in focal.itertuples(index=False)
        ]
        output.append(
            pd.concat([focal.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
        )
    return pd.concat(output, ignore_index=True) if output else ordered.iloc[0:0].copy()


def build_network_history(
    choice_sets: pd.DataFrame,
    event_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach strictly prior network and reaction history to choice candidates.

    State is updated once per ordered event rather than re-filtering the event
    log for each candidate.  Same-timestamp events are excluded from each focal
    history by construction.
    """

    if choice_sets.empty:
        return choice_sets.copy()
    events = event_history.copy()
    events["event_timestamp"] = normalize_timestamp_series(events["event_timestamp"])
    comments = events[events["event_type"].eq(COMMENT_EVENT)].sort_values(
        ["group", "event_timestamp", "event_id"], kind="mergesort"
    )
    enriched: list[pd.DataFrame] = []
    for group, candidates in choice_sets.groupby("group", sort=False):
        history = comments[comments["group"].astype(str).eq(str(group))]
        enriched.append(_network_features_for_group(candidates, history))
    network = pd.concat(enriched, ignore_index=True)
    network = _add_reaction_history(network, events)
    return network.sort_values(
        ["group", "event_timestamp", "choice_set_id", "candidate_created_at", "candidate_content_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _base_publication_tables(
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    event_history = build_event_history(config)
    choice_sets = build_thread_choice_sets(config, event_history=event_history)
    attrition_events = choice_sets.attrs.pop("attrition_events", pd.DataFrame())
    supporting = {
        "choice_set_attrition": _choice_set_attrition_summary(attrition_events),
        "risk_set_composition": _risk_set_composition(choice_sets),
        "source_availability": build_source_availability(config),
        "actor_label_qc": build_actor_label_qc(config),
        "account_universe": build_account_universe(config),
        "group_database_provenance": build_group_database_provenance(config),
    }
    return event_history, choice_sets, supporting


def prepare_choice_data(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Build canonical RQ1 events and same-thread conditional-choice rows."""

    event_history, choice_sets, supporting = _base_publication_tables(config)
    return {
        "event_history": event_history,
        "thread_choice_sets": choice_sets,
        **supporting,
    }


def prepare_data(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Build the canonical publication event, choice, and network tables."""

    event_history, choice_sets, supporting = _base_publication_tables(config)
    network_history = build_network_history(choice_sets, event_history)
    return {
        "event_history": event_history,
        "thread_choice_sets": choice_sets,
        "network_history": network_history,
        **supporting,
    }
