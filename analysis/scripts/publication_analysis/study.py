#!/usr/bin/env python3
"""Shared utilities for the YSocial two-arm snapshot script pack.

The scripts in this package intentionally consume normalized tables instead of
querying live services. Build those tables with `normalize_snapshot.py` from a
read-only restored snapshot database or from equivalent CSV exports.
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

START_TIME = "2026-05-11 08:00:00+02:00"
DEFAULT_ANALYSIS_WINDOW = "2026-05-11 08:00 Europe/Belgrade to snapshot extraction"
ANALYSIS_REVISION = "primary_valid_21day_revision_v3_day18_30_valid"
DEFAULT_ANALYSIS_SET = "primary_valid"
EXPERIMENT_DAY_ORIGIN = "2026-05-09"
EXPERIMENT_TZ = "Europe/Belgrade"
HUMAN_START_CUTOFF = START_TIME
WARMUP_DAYS = [0, 1]
BUG_CONFOUNDED_DAYS = list(range(10, 18))
PRIMARY_VALID_DAYS = [*range(2, 10), *range(18, 31)]
PRE_BUG_VALID_DAYS = list(range(2, 10))
POST_FIX_VALID_DAYS = list(range(18, 31))
NARROW_BUG_EXCLUSION_DAYS = list(range(14, 18))
NARROW_VALID_DAYS = [*range(2, 14), *range(18, 31)]
FULL_REALIZED_DAYS = list(range(2, 31))
ANALYSIS_SETS = {
    "primary_valid": {
        "valid_days": PRIMARY_VALID_DAYS,
        "strict_thread_validity": True,
        "label": "21 valid human-observation days",
    },
    "pre_bug_only": {
        "valid_days": PRE_BUG_VALID_DAYS,
        "strict_thread_validity": True,
        "label": "pre-bug valid human-observation days",
    },
    "post_fix_only": {
        "valid_days": POST_FIX_VALID_DAYS,
        "strict_thread_validity": True,
        "label": "post-fix valid human-observation days",
    },
    "narrow_bug_exclusion": {
        "valid_days": NARROW_VALID_DAYS,
        "strict_thread_validity": True,
        "label": "narrow severe-window sensitivity valid days",
    },
    "full_realized_descriptive": {
        "valid_days": FULL_REALIZED_DAYS,
        "strict_thread_validity": False,
        "label": "full realized-platform descriptive days",
    },
}
DYADS = ["H->H", "H->AI", "AI->H", "AI->AI"]
EVENT_ORDER = ["comments_replies", "upvote", "downvote", "pooled"]
REPUTATION_TABLES = {
    "user_reputation": "snapshot_user_reputation",
    "daily_trust_ratings": "snapshot_daily_trust_ratings",
    "memory_social_cards": "snapshot_memory_social_cards",
    "memory_interaction_events": "snapshot_memory_interaction_events",
}


class AnalysisSetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        analysis_set = validate_analysis_set(values)
        os.environ["YSOCIAL_ANALYSIS_SET"] = analysis_set
        setattr(namespace, self.dest, analysis_set)


@dataclass(frozen=True)
class GroupConfig:
    group: str
    label: str
    db_name: str
    experiment_id: int
    client_id: int
    human_count: int
    ai_count: int


GROUPS: Mapping[str, GroupConfig] = {
    "G1": GroupConfig(
        group="G1",
        label="labels_visible",
        db_name="experiments_75c440fb_c89d_4d17_abbb_a70b935de62a",
        experiment_id=39,
        client_id=40,
        human_count=20,
        ai_count=40,
    ),
    "G2": GroupConfig(
        group="G2",
        label="labels_hidden",
        db_name="experiments_f5c68d78_89e4_492b_acc0_d1a9ce6c3cd1",
        experiment_id=40,
        client_id=41,
        human_count=16,
        ai_count=32,
    ),
}


def compute_depths(contents: pd.DataFrame) -> pd.Series:
    """Compute reply depth from parent pointers without crossing groups."""

    parent = contents.set_index(["group", "content_id"])["comment_to"].to_dict()

    def depth_for(row: pd.Series) -> int:
        group = row["group"]
        current = row["content_id"]
        seen: set[Tuple[str, Any]] = set()
        depth = 0
        while True:
            key = (group, current)
            if key in seen:
                return depth
            seen.add(key)
            parent_id = parent.get(key)
            if pd.isna(parent_id) or int(parent_id) == -1:
                return depth
            current = int(parent_id)
            depth += 1

    return contents.apply(depth_for, axis=1).astype(int)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/ysocial_grupa1_grupa2_snapshot_2026-05-12T19-58-09_0200"),
        help="Directory containing normalized snapshot_events and snapshot_contents tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables/snapshot_2026-05-12T19-58-09_0200"),
        help="Directory for module output tables.",
    )
    parser.add_argument(
        "--start-time",
        default=START_TIME,
        help="Human-entry cutoff. Default is 2026-05-11 08:00 Europe/Belgrade.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional end timestamp. If omitted, uses all rows available in the snapshot.",
    )
    parser.add_argument(
        "--analysis-set",
        choices=sorted(ANALYSIS_SETS),
        default=DEFAULT_ANALYSIS_SET,
        action=AnalysisSetAction,
        help="Analysis-day filter. Default excludes warm-up days 0-1 and AI-runner-bug days 10-17.",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_time(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Belgrade")
    else:
        ts = ts.tz_convert("Europe/Belgrade")
    return ts


def validate_analysis_set(value: Optional[str]) -> str:
    analysis_set = str(value or os.environ.get("YSOCIAL_ANALYSIS_SET") or DEFAULT_ANALYSIS_SET).strip()
    if analysis_set not in ANALYSIS_SETS:
        raise ValueError(f"Unknown analysis_set={analysis_set!r}. Valid values: {sorted(ANALYSIS_SETS)}")
    return analysis_set


def set_current_analysis_set(value: Optional[str]) -> str:
    analysis_set = validate_analysis_set(value)
    os.environ["YSOCIAL_ANALYSIS_SET"] = analysis_set
    return analysis_set


def current_analysis_set(value: Optional[str] = None) -> str:
    return validate_analysis_set(value)


def analysis_set_config(value: Optional[str] = None) -> Mapping[str, Any]:
    return ANALYSIS_SETS[current_analysis_set(value)]


def compact_day_list(days: Sequence[int]) -> str:
    clean = sorted({int(day) for day in days})
    if not clean:
        return ""
    ranges: List[str] = []
    start = prev = clean[0]
    for day in clean[1:]:
        if day == prev + 1:
            prev = day
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = day
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def analysis_set_manifest(analysis_set: Optional[str] = None) -> Dict[str, Any]:
    selected = current_analysis_set(analysis_set)
    cfg = ANALYSIS_SETS[selected]
    return {
        "analysis_revision": ANALYSIS_REVISION,
        "analysis_set": selected,
        "analysis_set_label": cfg["label"],
        "experiment_day_origin": EXPERIMENT_DAY_ORIGIN,
        "experiment_timezone": EXPERIMENT_TZ,
        "warmup_days": WARMUP_DAYS,
        "bug_confounded_days": BUG_CONFOUNDED_DAYS,
        "primary_valid_days": PRIMARY_VALID_DAYS,
        "pre_bug_valid_days": PRE_BUG_VALID_DAYS,
        "post_fix_valid_days": POST_FIX_VALID_DAYS,
        "narrow_bug_exclusion_days": NARROW_BUG_EXCLUSION_DAYS,
        "analysis_set_valid_days": list(cfg["valid_days"]),
        "human_start_cutoff": pd.Timestamp(HUMAN_START_CUTOFF).isoformat(),
        "strict_thread_validity": bool(cfg["strict_thread_validity"]),
        "raw_data_mutated": False,
        "canonical_phrase": "Primary analyses use 21 valid human-observation days under the intended AI-writing regime.",
    }


def phase_for_day(day: Any) -> Optional[str]:
    if pd.isna(day):
        return None
    day_int = int(day)
    if day_int in WARMUP_DAYS:
        return "ai_warmup"
    if day_int in PRE_BUG_VALID_DAYS:
        return "pre_bug_valid"
    if day_int in BUG_CONFOUNDED_DAYS:
        return "bug_confounded"
    if day_int in POST_FIX_VALID_DAYS:
        return "post_fix_valid"
    return "outside_revision_window"


def local_calendar_date(series: pd.Series) -> pd.Series:
    parsed = normalize_timestamp_series(series)
    return parsed.dt.date.astype("string")


def experiment_day_series(series: pd.Series) -> pd.Series:
    parsed = normalize_timestamp_series(series)
    origin = pd.Timestamp(EXPERIMENT_DAY_ORIGIN, tz=EXPERIMENT_TZ)
    local_midnight = parsed.dt.normalize()
    days = ((local_midnight - origin) / pd.Timedelta(days=1)).astype("float")
    out = np.floor(days).astype("float")
    result = pd.Series(out, index=series.index)
    result[parsed.isna()] = np.nan
    return result.astype("Int64")


def _day_bool(days: pd.Series, values: Sequence[int]) -> pd.Series:
    return days.isin([int(day) for day in values]).fillna(False)


def _timestamp_after_human_start(ts: pd.Series, start_time: str) -> pd.Series:
    start = parse_time(start_time)
    return ts.notna() & (ts >= start)


def add_content_analysis_flags(contents: pd.DataFrame, start_time: str = START_TIME) -> pd.DataFrame:
    out = contents.copy()
    out["created_at"] = normalize_timestamp_series(out["created_at"])
    if "root_thread_created_at" in out.columns:
        out["root_thread_created_at"] = normalize_timestamp_series(out["root_thread_created_at"])
    else:
        out["root_thread_created_at"] = pd.NaT
    out["experiment_day"] = experiment_day_series(out["created_at"])
    out["calendar_date"] = local_calendar_date(out["created_at"])
    out["root_thread_experiment_day"] = experiment_day_series(out["root_thread_created_at"])
    out["root_thread_calendar_date"] = local_calendar_date(out["root_thread_created_at"])
    out["analysis_phase"] = out["experiment_day"].map(phase_for_day)
    out["is_warmup_day"] = _day_bool(out["experiment_day"], WARMUP_DAYS)
    out["is_bug_confounded_day"] = _day_bool(out["experiment_day"], BUG_CONFOUNDED_DAYS)
    out["is_valid_human_observation_day"] = _day_bool(out["experiment_day"], PRIMARY_VALID_DAYS)
    out["event_day_valid"] = out["is_valid_human_observation_day"]
    out["root_thread_day_valid"] = _day_bool(out["root_thread_experiment_day"], PRIMARY_VALID_DAYS)
    out["target_content_day_valid"] = out["event_day_valid"]
    out["post_fix_period"] = _day_bool(out["experiment_day"], POST_FIX_VALID_DAYS).astype(int)
    out["root_thread_from_bug_window"] = _day_bool(out["root_thread_experiment_day"], BUG_CONFOUNDED_DAYS)
    out["target_content_from_bug_window"] = _day_bool(out["experiment_day"], BUG_CONFOUNDED_DAYS)
    after_start = _timestamp_after_human_start(out["created_at"], start_time)
    root_after_start = _timestamp_after_human_start(out["root_thread_created_at"], start_time)
    out["valid_primary_event"] = after_start & root_after_start & out["event_day_valid"] & out["root_thread_day_valid"]
    out["invalid_reason"] = invalid_reasons(out, content_level=True, start_time=start_time)
    return out


def add_event_analysis_flags(events: pd.DataFrame, start_time: str = START_TIME) -> pd.DataFrame:
    out = events.copy()
    for col in ["event_timestamp", "root_thread_created_at", "target_content_created_at"]:
        if col in out.columns:
            out[col] = normalize_timestamp_series(out[col])
        else:
            out[col] = pd.NaT
    out["experiment_day"] = experiment_day_series(out["event_timestamp"])
    out["calendar_date"] = local_calendar_date(out["event_timestamp"])
    out["root_thread_experiment_day"] = experiment_day_series(out["root_thread_created_at"])
    out["target_content_experiment_day"] = experiment_day_series(out["target_content_created_at"])
    out["root_thread_calendar_date"] = local_calendar_date(out["root_thread_created_at"])
    out["target_content_calendar_date"] = local_calendar_date(out["target_content_created_at"])
    out["analysis_phase"] = out["experiment_day"].map(phase_for_day)
    out["is_warmup_day"] = _day_bool(out["experiment_day"], WARMUP_DAYS)
    out["is_bug_confounded_day"] = _day_bool(out["experiment_day"], BUG_CONFOUNDED_DAYS)
    out["is_valid_human_observation_day"] = _day_bool(out["experiment_day"], PRIMARY_VALID_DAYS)
    out["event_day_valid"] = out["is_valid_human_observation_day"]
    out["root_thread_day_valid"] = _day_bool(out["root_thread_experiment_day"], PRIMARY_VALID_DAYS)
    out["target_content_day_valid"] = _day_bool(out["target_content_experiment_day"], PRIMARY_VALID_DAYS)
    out["post_fix_period"] = _day_bool(out["experiment_day"], POST_FIX_VALID_DAYS).astype(int)
    out["root_thread_from_bug_window"] = _day_bool(out["root_thread_experiment_day"], BUG_CONFOUNDED_DAYS)
    out["target_content_from_bug_window"] = _day_bool(out["target_content_experiment_day"], BUG_CONFOUNDED_DAYS)
    event_after_start = _timestamp_after_human_start(out["event_timestamp"], start_time)
    root_after_start = _timestamp_after_human_start(out["root_thread_created_at"], start_time)
    target_after_start = _timestamp_after_human_start(out["target_content_created_at"], start_time)
    out["valid_primary_event"] = (
        event_after_start
        & root_after_start
        & target_after_start
        & out["event_day_valid"]
        & out["root_thread_day_valid"]
        & out["target_content_day_valid"]
    )
    out["invalid_reason"] = invalid_reasons(out, content_level=False, start_time=start_time)
    return out


def invalid_reasons(frame: pd.DataFrame, content_level: bool, start_time: str) -> pd.Series:
    def day_in(day: Any, days: Sequence[int]) -> bool:
        return pd.notna(day) and int(day) in set(int(value) for value in days)

    reasons: List[str] = []
    start = parse_time(start_time)
    for _, row in frame.iterrows():
        row_reasons: List[str] = []
        event_ts = row.get("created_at") if content_level else row.get("event_timestamp")
        root_ts = row.get("root_thread_created_at")
        target_ts = row.get("created_at") if content_level else row.get("target_content_created_at")
        event_day = row.get("experiment_day")
        root_day = row.get("root_thread_experiment_day")
        target_day = event_day if content_level else row.get("target_content_experiment_day")
        if pd.isna(event_ts) or pd.isna(root_ts) or pd.isna(target_ts):
            row_reasons.append("missing_timestamp")
        if pd.notna(event_ts) and event_ts < start:
            row_reasons.append("pre_human_cutoff")
        if pd.notna(root_ts) and root_ts < start:
            row_reasons.append("pre_human_cutoff_root")
        if pd.notna(target_ts) and target_ts < start:
            row_reasons.append("pre_human_cutoff_target")
        if day_in(event_day, WARMUP_DAYS):
            row_reasons.append("warmup_event")
        if day_in(event_day, BUG_CONFOUNDED_DAYS):
            row_reasons.append("bug_event")
        if day_in(root_day, WARMUP_DAYS):
            row_reasons.append("warmup_root")
        if day_in(root_day, BUG_CONFOUNDED_DAYS):
            row_reasons.append("bug_root")
        if day_in(target_day, WARMUP_DAYS):
            row_reasons.append("warmup_target")
        if day_in(target_day, BUG_CONFOUNDED_DAYS):
            row_reasons.append("bug_target")
        if pd.notna(event_day) and int(event_day) not in PRIMARY_VALID_DAYS:
            row_reasons.append("invalid_event_day")
        if pd.notna(root_day) and int(root_day) not in PRIMARY_VALID_DAYS:
            row_reasons.append("invalid_root_day")
        if pd.notna(target_day) and int(target_day) not in PRIMARY_VALID_DAYS:
            row_reasons.append("invalid_target_day")
        reasons.append(";".join(dict.fromkeys(row_reasons)) if row_reasons else "valid")
    return pd.Series(reasons, index=frame.index)


def ensure_event_analysis_flags(events: pd.DataFrame, start_time: str = START_TIME) -> pd.DataFrame:
    events = events.copy()
    for col in ["event_timestamp", "root_thread_created_at", "target_content_created_at"]:
        if col in events.columns:
            events[col] = normalize_timestamp_series(events[col])
    # Recompute these flags every load so already-normalized snapshots inherit
    # analysis-plan amendments such as moving day 18 into the valid window.
    return add_event_analysis_flags(events, start_time)


def ensure_content_analysis_flags(contents: pd.DataFrame, start_time: str = START_TIME) -> pd.DataFrame:
    contents = contents.copy()
    for col in ["created_at", "root_thread_created_at"]:
        if col in contents.columns:
            contents[col] = normalize_timestamp_series(contents[col])
    # Recompute these flags every load so already-normalized snapshots inherit
    # analysis-plan amendments such as moving day 18 into the valid window.
    return add_content_analysis_flags(contents, start_time)


def sequential_valid_day_index(days: pd.Series, analysis_set: Optional[str] = None) -> pd.Series:
    valid_days = list(analysis_set_config(analysis_set)["valid_days"])
    mapping = {day: idx + 1 for idx, day in enumerate(valid_days)}
    return days.map(mapping).astype("Int64")


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize("Europe/Belgrade", nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert("Europe/Belgrade")


def read_table(path_stem: Path) -> pd.DataFrame:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing table: {parquet_path} or {csv_path}")


def write_table(df: pd.DataFrame, path_stem: Path) -> None:
    ensure_dir(path_stem.parent)
    csv_path = path_stem.with_suffix(".csv")
    parquet_path = path_stem.with_suffix(".parquet")
    if parquet_path.exists():
        parquet_path.unlink()
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except (ImportError, ValueError) as exc:
        # CSV is the canonical fallback when no parquet engine is available
        # or the dtype cannot be encoded; keep the parquet file absent.
        logging.getLogger(__name__).warning(
            "Skipping parquet write for %s: %s", csv_path.name, exc
        )


def load_events(processed_dir: Path) -> pd.DataFrame:
    df = read_table(processed_dir / "snapshot_events")
    for col in ["event_timestamp", "root_thread_created_at", "target_content_created_at"]:
        if col in df.columns:
            df[col] = normalize_timestamp_series(df[col])
    required = {
        "group",
        "event_type",
        "actor_id",
        "target_author_id",
        "actor_type",
        "target_author_type",
        "dyad",
        "event_timestamp",
        "root_thread_created_at",
        "target_content_created_at",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot_events is missing required columns: {missing}")
    return ensure_event_analysis_flags(df)


def load_contents(processed_dir: Path) -> pd.DataFrame:
    df = read_table(processed_dir / "snapshot_contents")
    for col in ["created_at", "root_thread_created_at"]:
        if col in df.columns:
            df[col] = normalize_timestamp_series(df[col])
    required = {"group", "content_id", "author_id", "author_type", "created_at", "root_id", "root_thread_created_at"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot_contents is missing required columns: {missing}")
    return ensure_content_analysis_flags(df)


def load_user_reputation(processed_dir: Path) -> pd.DataFrame:
    df = read_table(processed_dir / "snapshot_user_reputation")
    required = {"group", "user_id", "user_type", "reputation", "karma"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot_user_reputation is missing required columns: {missing}")
    for col in ["reputation", "karma"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_daily_trust_ratings(processed_dir: Path) -> pd.DataFrame:
    df = read_table(processed_dir / "snapshot_daily_trust_ratings")
    required = {
        "group",
        "rater_user_id",
        "target_user_id",
        "rater_type",
        "target_type",
        "interaction_count",
        "target_reputation",
        "target_karma",
        "reputation_exposed",
        "rating",
        "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot_daily_trust_ratings is missing required columns: {missing}")
    for col in ["interaction_count", "target_reputation", "target_karma", "rating", "day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "reputation_exposed" in df.columns:
        df["reputation_exposed"] = df["reputation_exposed"].map(normalize_bool)
    return df


def load_memory_social_cards(processed_dir: Path) -> pd.DataFrame:
    df = read_table(processed_dir / "snapshot_memory_social_cards")
    required = {
        "group",
        "agent_user_id",
        "other_user_id",
        "agent_type",
        "other_type",
        "dyad",
        "trust",
        "affinity",
        "conflict",
        "event_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"snapshot_memory_social_cards is missing required columns: {missing}")
    for col in ["trust", "affinity", "conflict", "humor", "event_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_bool(value: Any) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"t", "true", "1", "yes", "y"}:
        return True
    if text in {"f", "false", "0", "no", "n"}:
        return False
    return None


def filter_main_events(
    events: pd.DataFrame,
    start_time: str = START_TIME,
    end_time: Optional[str] = None,
    analysis_set: Optional[str] = None,
) -> pd.DataFrame:
    selected_set = current_analysis_set(analysis_set)
    cfg = analysis_set_config(selected_set)
    events = ensure_event_analysis_flags(events, start_time)
    end = parse_time(end_time)
    event_day_valid = _day_bool(events["experiment_day"], cfg["valid_days"])
    event_after_start = _timestamp_after_human_start(events["event_timestamp"], start_time)
    mask = events["event_timestamp"].notna() & event_after_start & event_day_valid
    if cfg["strict_thread_validity"]:
        mask &= (
            events["root_thread_created_at"].notna()
            & events["target_content_created_at"].notna()
            & _timestamp_after_human_start(events["root_thread_created_at"], start_time)
            & _timestamp_after_human_start(events["target_content_created_at"], start_time)
            & _day_bool(events["root_thread_experiment_day"], cfg["valid_days"])
            & _day_bool(events["target_content_experiment_day"], cfg["valid_days"])
        )
    else:
        mask &= events["root_thread_created_at"].notna() & events["target_content_created_at"].notna()
    if end is not None:
        mask &= events["event_timestamp"] < end
    out = events.loc[mask].copy()
    out["analysis_set"] = selected_set
    out["valid_day_index"] = sequential_valid_day_index(out["experiment_day"], selected_set)
    return out


def filter_postcutoff_excluded_artifacts(
    events: pd.DataFrame,
    start_time: str = START_TIME,
    end_time: Optional[str] = None,
    analysis_set: Optional[str] = None,
) -> pd.DataFrame:
    selected_set = current_analysis_set(analysis_set)
    events = ensure_event_analysis_flags(events, start_time)
    start = parse_time(start_time)
    end = parse_time(end_time)
    postcutoff = events["event_timestamp"].notna() & (events["event_timestamp"] >= start)
    if end is not None:
        postcutoff &= events["event_timestamp"] < end
    main_index = filter_main_events(events, start_time, end_time, selected_set).index
    excluded = postcutoff & ~events.index.isin(main_index)
    out = events.loc[excluded].copy()
    out["analysis_set"] = selected_set
    return out


def filter_main_contents(
    contents: pd.DataFrame,
    start_time: str = START_TIME,
    end_time: Optional[str] = None,
    analysis_set: Optional[str] = None,
) -> pd.DataFrame:
    selected_set = current_analysis_set(analysis_set)
    cfg = analysis_set_config(selected_set)
    contents = ensure_content_analysis_flags(contents, start_time)
    end = parse_time(end_time)
    created_valid = _day_bool(contents["experiment_day"], cfg["valid_days"])
    mask = contents["created_at"].notna() & _timestamp_after_human_start(contents["created_at"], start_time) & created_valid
    if cfg["strict_thread_validity"]:
        mask &= (
            contents["root_thread_created_at"].notna()
            & _timestamp_after_human_start(contents["root_thread_created_at"], start_time)
            & _day_bool(contents["root_thread_experiment_day"], cfg["valid_days"])
        )
    else:
        mask &= contents["root_thread_created_at"].notna()
    if end is not None:
        mask &= contents["created_at"] < end
    out = contents.loc[mask].copy()
    out["analysis_set"] = selected_set
    out["valid_day_index"] = sequential_valid_day_index(out["experiment_day"], selected_set)
    return out


def analysis_metadata(
    start_time: str,
    end_time: Optional[str],
    denominator: str,
    event_type: str,
    analysis_set: Optional[str] = None,
) -> Dict[str, str]:
    end_label = end_time or "snapshot extraction"
    manifest = analysis_set_manifest(analysis_set)
    return {
        "analysis_window": f"{start_time} to {end_label}",
        "pre_human_artifacts_excluded": "yes",
        "denominator": denominator,
        "event_type": event_type,
        "analysis_revision": manifest["analysis_revision"],
        "analysis_set": manifest["analysis_set"],
        "valid_days": compact_day_list(manifest["analysis_set_valid_days"]),
        "excluded_warmup_days": compact_day_list(manifest["warmup_days"]),
        "excluded_bug_days": compact_day_list(manifest["bug_confounded_days"]),
        "strict_thread_validity": str(bool(manifest["strict_thread_validity"])).lower(),
        "human_start_cutoff": str(manifest["human_start_cutoff"]),
        "raw_data_mutated": "false",
    }


def dyad_from_types(actor_type: Any, target_type: Any) -> str:
    left = "AI" if str(actor_type).upper() == "AI" else "H"
    right = "AI" if str(target_type).upper() == "AI" else "H"
    return f"{left}->{right}"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return (math.nan, math.nan)
    p1 = k1 / n1
    p2 = k2 / n2
    se = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    diff = p2 - p1
    return (diff - z * se, diff + z * se)


def two_prop_pvalue(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, str]:
    if n1 <= 0 or n2 <= 0:
        return (math.nan, "not_tested")
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=float)
    if (table < 5).any():
        try:
            from scipy.stats import fisher_exact

            _, p = fisher_exact(table)
            return (float(p), "fisher_exact")
        except Exception:
            return (math.nan, "fisher_exact_unavailable")
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * ((1 / n1) + (1 / n2)))
    if se == 0:
        return (math.nan, "two_proportion_z")
    z = (k2 / n2 - k1 / n1) / se
    try:
        from scipy.stats import norm

        p = 2 * (1 - norm.cdf(abs(z)))
    except Exception:
        p = math.erfc(abs(z) / math.sqrt(2))
    return (float(p), "two_proportion_z")


def bootstrap_ci(values: Sequence[float], statistic=np.mean, reps: int = 2000, seed: int = 20260512) -> Tuple[float, float]:
    clean = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if clean.size == 0:
        return (math.nan, math.nan)
    if clean.size == 1:
        val = float(statistic(clean))
        return (val, val)
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(reps, clean.size), replace=True)
    stats = np.apply_along_axis(statistic, 1, samples)
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def correlation_with_pvalue(x: Sequence[float], y: Sequence[float], method: str = "pearson") -> Tuple[float, float]:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if pd.notna(a) and pd.notna(b)]
    if len(pairs) < 3:
        return (math.nan, math.nan)
    x_arr = np.asarray([p[0] for p in pairs], dtype=float)
    y_arr = np.asarray([p[1] for p in pairs], dtype=float)
    if np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return (math.nan, math.nan)
    try:
        from scipy import stats

        if method == "spearman":
            result = stats.spearmanr(x_arr, y_arr)
        else:
            result = stats.pearsonr(x_arr, y_arr)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        if method == "spearman":
            x_arr = pd.Series(x_arr).rank(method="average").to_numpy()
            y_arr = pd.Series(y_arr).rank(method="average").to_numpy()
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        return (r, math.nan)


def weighted_corr(x: Sequence[float], y: Sequence[float], weights: Sequence[float]) -> float:
    triples = [
        (float(a), float(b), float(w))
        for a, b, w in zip(x, y, weights)
        if pd.notna(a) and pd.notna(b) and pd.notna(w) and float(w) > 0
    ]
    if len(triples) < 2:
        return math.nan
    x_arr = np.asarray([t[0] for t in triples], dtype=float)
    y_arr = np.asarray([t[1] for t in triples], dtype=float)
    w_arr = np.asarray([t[2] for t in triples], dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return math.nan
    w_sum = w_arr.sum()
    x_mean = float(np.sum(w_arr * x_arr) / w_sum)
    y_mean = float(np.sum(w_arr * y_arr) / w_sum)
    cov = float(np.sum(w_arr * (x_arr - x_mean) * (y_arr - y_mean)) / w_sum)
    vx = float(np.sum(w_arr * (x_arr - x_mean) ** 2) / w_sum)
    vy = float(np.sum(w_arr * (y_arr - y_mean) ** 2) / w_sum)
    if vx <= 0 or vy <= 0:
        return math.nan
    return cov / math.sqrt(vx * vy)


def compare_group_proportions(
    rows: List[Dict[str, Any]],
    measure: str,
    group_col: str = "group",
    k_col: str = "numerator",
    n_col: str = "denominator_n",
) -> List[Dict[str, Any]]:
    by_group = {row[group_col]: row for row in rows if row.get("measure") == measure}
    if "G1" not in by_group or "G2" not in by_group:
        return []
    g1 = by_group["G1"]
    g2 = by_group["G2"]
    k1, n1 = int(g1[k_col]), int(g1[n_col])
    k2, n2 = int(g2[k_col]), int(g2[n_col])
    p = two_prop_pvalue(k1, n1, k2, n2)
    lo, hi = proportion_diff_ci(k1, n1, k2, n2)
    return [
        {
            "measure": measure,
            "contrast": "G2 - G1",
            "estimate": (k2 / n2 if n2 else math.nan) - (k1 / n1 if n1 else math.nan),
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p[0],
            "test": p[1],
        }
    ]


def mixing_counts(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    if event_type == "pooled":
        subset = events
    elif event_type == "comments_replies":
        subset = events[events["event_type"] == "comments_replies"]
    else:
        subset = events[events["event_type"] == event_type]
    idx = pd.MultiIndex.from_product([sorted(subset["group"].dropna().unique()), DYADS], names=["group", "dyad"])
    counts = subset.groupby(["group", "dyad"]).size().reindex(idx, fill_value=0).reset_index(name="count")
    counts.insert(1, "event_type", event_type)
    return counts


def build_comment_graph(events: pd.DataFrame, group: str):
    import networkx as nx

    subset = events[(events["group"] == group) & (events["event_type"] == "comments_replies")]
    graph = nx.DiGraph()
    for _, row in subset.iterrows():
        actor = row["actor_id"]
        target = row["target_author_id"]
        if pd.isna(actor) or pd.isna(target):
            continue
        graph.add_node(actor, label=row["actor_type"])
        graph.add_node(target, label=row["target_author_type"])
        if graph.has_edge(actor, target):
            graph[actor][target]["weight"] += 1
        else:
            graph.add_edge(actor, target, weight=1)
    return graph


def graph_summary_metrics(graph) -> Dict[str, Any]:
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "weighted_edges": 0,
            "assortativity": math.nan,
            "ei_index": math.nan,
            "average_clustering": math.nan,
            "transitivity": math.nan,
            "reciprocity": math.nan,
            "density": math.nan,
            "weak_components": 0,
            "largest_weak_component": 0,
        }
    weighted_edges = sum(float(d.get("weight", 1)) for _, _, d in graph.edges(data=True))
    dyad_weights = {dyad: 0.0 for dyad in DYADS}
    for u, v, data in graph.edges(data=True):
        dyad = dyad_from_types(graph.nodes[u].get("label"), graph.nodes[v].get("label"))
        dyad_weights[dyad] += float(data.get("weight", 1))
    external = dyad_weights["H->AI"] + dyad_weights["AI->H"]
    internal = dyad_weights["H->H"] + dyad_weights["AI->AI"]
    ei = (external - internal) / (external + internal) if (external + internal) else math.nan
    undirected = graph.to_undirected()
    try:
        assort = nx.attribute_assortativity_coefficient(graph, "label")
    except Exception:
        assort = math.nan
    weak_components = list(nx.weakly_connected_components(graph)) if graph.number_of_nodes() else []
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weighted_edges": weighted_edges,
        "assortativity": assort,
        "ei_index": ei,
        "average_clustering": nx.average_clustering(undirected, weight="weight") if graph.number_of_nodes() else math.nan,
        "transitivity": nx.transitivity(undirected) if graph.number_of_nodes() else math.nan,
        "reciprocity": nx.reciprocity(graph) if graph.number_of_edges() else math.nan,
        "density": nx.density(graph),
        "weak_components": len(weak_components),
        "largest_weak_component": max((len(c) for c in weak_components), default=0),
        **{f"weight_{key}": val for key, val in dyad_weights.items()},
    }


def weighted_degree_by_label(graph) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for node, data in graph.nodes(data=True):
        label = data.get("label", "unknown")
        in_w = float(sum(edge_data.get("weight", 1) for _, _, edge_data in graph.in_edges(node, data=True)))
        out_w = float(sum(edge_data.get("weight", 1) for _, _, edge_data in graph.out_edges(node, data=True)))
        rows.append({"node_id": node, "label": label, "weighted_in_degree": in_w, "weighted_out_degree": out_w})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["label", "n", "mean_in", "median_in", "mean_out", "median_out"])
    return (
        df.groupby("label")
        .agg(
            n=("node_id", "count"),
            mean_in=("weighted_in_degree", "mean"),
            median_in=("weighted_in_degree", "median"),
            mean_out=("weighted_out_degree", "mean"),
            median_out=("weighted_out_degree", "median"),
        )
        .reset_index()
    )


def fixed_hai_modularity(graph) -> float:
    import networkx as nx

    if graph.number_of_edges() == 0:
        return math.nan
    undirected = graph.to_undirected()
    humans = {n for n, d in undirected.nodes(data=True) if d.get("label") == "H"}
    ais = {n for n, d in undirected.nodes(data=True) if d.get("label") == "AI"}
    communities = [c for c in [humans, ais] if c]
    if len(communities) < 2:
        return math.nan
    return float(nx.algorithms.community.quality.modularity(undirected, communities, weight="weight"))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False) + "```"


def output_metadata_defaults(analysis_set: Optional[str] = None) -> Dict[str, str]:
    manifest = analysis_set_manifest(analysis_set)
    return {
        "analysis_revision": manifest["analysis_revision"],
        "analysis_set": manifest["analysis_set"],
        "valid_days": compact_day_list(manifest["analysis_set_valid_days"]),
        "excluded_warmup_days": compact_day_list(manifest["warmup_days"]),
        "excluded_bug_days": compact_day_list(manifest["bug_confounded_days"]),
        "strict_thread_validity": str(bool(manifest["strict_thread_validity"])).lower(),
        "human_start_cutoff": str(manifest["human_start_cutoff"]),
        "raw_data_mutated": "false",
    }


def add_output_metadata_columns(df: pd.DataFrame, analysis_set: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    for col, value in output_metadata_defaults(analysis_set).items():
        if col not in out.columns:
            out[col] = value
    return out


def validate_table_analysis_set(
    df: pd.DataFrame,
    table_name: str,
    analysis_set: Optional[str] = None,
    *,
    require: bool = False,
) -> None:
    selected = current_analysis_set(analysis_set)
    if "analysis_set" not in df.columns:
        if require:
            raise ValueError(f"{table_name} lacks analysis_set metadata; rerun upstream modules with the valid-day revision.")
        return
    observed = sorted(df["analysis_set"].dropna().astype(str).unique().tolist())
    if observed and observed != [selected]:
        raise ValueError(f"{table_name} analysis_set mismatch: expected {selected!r}, observed {observed!r}.")


def write_outputs(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    ensure_dir(output_dir)
    out = add_output_metadata_columns(df)
    out.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(markdown_table(out) + "\n", encoding="utf-8")


def write_json(data: Any, output_dir: Path, stem: str) -> None:
    ensure_dir(output_dir)
    (output_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def format_ci(lo: float, hi: float) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{lo:.4f}, {hi:.4f}]"
