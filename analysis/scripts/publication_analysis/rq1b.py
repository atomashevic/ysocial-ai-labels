"""RQ1b browser telemetry: direct thread entry after a rendered feed card.

The primary unit is one participant-feed-spell-root-post opportunity.  A post
is eligible when its top-level feed card generated an ``appear`` event after
an explicit feed page view during a quality-passing session on a primary-valid
day.  The binary outcome is one when the next page view is the corresponding
thread.  The endpoint uses DOM and route events only.  It does not reconstruct
pixel visibility, visible duration, reading, or reply conversion.

This module is intentionally separate from the legacy browser summaries.  It
uses the authoritative participant and author roster, applies the analysis
window before constructing opportunities, and exposes pure panel-building
functions for tests and release validation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig, load_config
from .io import read_table, runtime_versions, sha256_file, write_json, write_table
from .study import GROUPS, analysis_set_manifest, experiment_day_series, parse_time


GROUP_ORDER = ("G1", "G2")
AUTHOR_ORDER = ("H", "AI")
PRIMARY_OUTCOME = "direct_thread_entry"
ANALYSIS_SCHEMA = "ysocial.rq1b-direct-thread-entry.v1"
PARETO_K_THRESHOLD = 0.70
MCMC_RHAT_THRESHOLD = 1.01
MCMC_ESS_THRESHOLD = 400.0
PRIOR_MEAN_SHIFT_TOLERANCE = 0.01
PRIOR_PROBABILITY_SHIFT_TOLERANCE = 0.05
VENDOR_THREAD_OPENS_RELATIVE = Path(
    "data/raw/moveo/feed_engagement_email_2026-07-15/full_dataset/thread_opens.csv"
)

EVENT_COLUMNS = (
    "session_id",
    "event_seq",
    "event_timestamp_utc",
    "action",
    "id",
    "type",
    "semantic_group",
    "section",
    "item_id",
    "item_type",
    "is_content_item",
    "group",
    "y_user_id",
    "route_kind",
    "route_item_id",
    "author_id",
    "author_type_ground_truth",
    "backend_content_created_at",
    "root_thread_created_at",
    "root_post_id",
)

PANEL_SIGNATURE_COLUMNS = (
    "opportunity_id",
    "group",
    "participant_id",
    "session_id",
    "feed_entry_event_seq",
    "feed_exit_event_seq",
    "feed_exit_route_kind",
    "feed_exit_route_item_id",
    "post_id",
    "author_id",
    "author_type",
    "experiment_day",
    "first_appear_event_seq",
    "first_appear_time_utc",
    "top_level_appear_events",
    "direct_thread_entry",
    "direct_open_event_seq",
    "direct_open_time_utc",
)


def configure_runtime(local_temp: Path | None = None) -> Path:
    """Use local temporary storage and one numerical thread per chain worker."""

    root = (
        Path("/tmp") / f"ysocial-rq1b-{os.getuid()}"
        if local_temp is None
        else Path(local_temp).expanduser()
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    for name in ("TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR"):
        os.environ[name] = str(root)
    tempfile.tempdir = str(root)
    compiledir = root / "pytensor"
    compiledir.mkdir(parents=True, exist_ok=True)
    existing = [
        value
        for value in os.environ.get("PYTENSOR_FLAGS", "").split(",")
        if value and not value.strip().startswith("base_compiledir=")
    ]
    existing.append(f"base_compiledir={compiledir}")
    os.environ["PYTENSOR_FLAGS"] = ",".join(existing)
    return root


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _int_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round().astype("Int64")


def _read_events(browser_dir: Path) -> pd.DataFrame:
    parquet = browser_dir / "session_events.parquet"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet, columns=list(EVENT_COLUMNS))
        except (KeyError, ValueError):
            return pd.read_parquet(parquet)
    return read_table(browser_dir / "session_events")


def _analysis_event_mask(
    events: pd.DataFrame,
    session_qc: pd.DataFrame,
    *,
    start_time: str,
    end_time: str | None,
    valid_days: Sequence[int],
) -> pd.Series:
    """Select valid-day events in quality-passing sessions."""

    _require_columns(
        events,
        ("session_id", "event_timestamp_utc", "group"),
        "session events",
    )
    _require_columns(session_qc, ("session_id", "usable_session"), "session QC")
    event_time = pd.to_datetime(events["event_timestamp_utc"], utc=True, errors="coerce")
    event_day = experiment_day_series(event_time)
    start = parse_time(start_time).tz_convert("UTC")
    keep = (
        event_time.notna()
        & event_time.ge(start)
        & event_day.isin([int(day) for day in valid_days])
        & events["group"].isin(GROUP_ORDER)
    )
    if end_time is not None:
        keep &= event_time.lt(parse_time(end_time).tz_convert("UTC"))
    usable = set(
        session_qc.loc[
            session_qc["usable_session"].fillna(False).astype(bool), "session_id"
        ].astype(str)
    )
    keep &= events["session_id"].astype(str).isin(usable)
    return keep


def _authoritative_rosters(
    reputation: pd.DataFrame,
) -> tuple[set[tuple[str, int]], dict[tuple[str, int], str]]:
    _require_columns(reputation, ("group", "user_id", "user_type"), "user reputation")
    roster = reputation[reputation["group"].isin(GROUP_ORDER)].copy()
    roster["user_id"] = _int_series(roster["user_id"])
    roster["user_type"] = roster["user_type"].astype(str).str.upper()
    roster = roster[
        roster["user_id"].notna() & roster["user_type"].isin(AUTHOR_ORDER)
    ]
    duplicated = roster.duplicated(["group", "user_id"], keep=False)
    if duplicated.any():
        conflict = roster.loc[duplicated].groupby(["group", "user_id"])["user_type"].nunique()
        if conflict.gt(1).any():
            raise ValueError("Authoritative reputation has conflicting user types")
        roster = roster.drop_duplicates(["group", "user_id"])
    humans = {
        (str(row.group), int(row.user_id))
        for row in roster[roster["user_type"].eq("H")].itertuples(index=False)
    }
    authors = {
        (str(row.group), int(row.user_id)): str(row.user_type)
        for row in roster.itertuples(index=False)
    }
    return humans, authors


def _stage_counts(frame: pd.DataFrame, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in (*GROUP_ORDER, "all"):
        part = frame if group == "all" else frame[frame["group"].eq(group)]
        if {"group", "y_user_id"}.issubset(part.columns):
            participant_count = int(
                part.assign(_participant_id=_int_series(part["y_user_id"]))
                .dropna(subset=["_participant_id"])
                .drop_duplicates(["group", "_participant_id"])
                .shape[0]
            )
        else:
            participant_count = 0
        rows.append(
            {
                "stage": stage,
                "group": group,
                "rows": int(len(part)),
                "sessions": int(part["session_id"].astype(str).nunique()) if "session_id" in part else 0,
                "participants": participant_count,
            }
        )
    return rows


def _attach_feed_route_spells(cards: pd.DataFrame, page: pd.DataFrame) -> pd.DataFrame:
    """Attach each card appearance to its current explicit feed-route spell."""

    route_columns = [
        "session_id",
        "event_seq",
        "event_time_utc",
        "route_kind",
        "route_item_id_int",
    ]
    routes = page[route_columns].dropna(subset=["event_seq"]).copy()
    routes = routes.sort_values(["session_id", "event_seq"], kind="mergesort")
    route_groups = {
        str(session_id): part.reset_index(drop=True)
        for session_id, part in routes.groupby("session_id", sort=False)
    }
    attached: list[pd.DataFrame] = []
    for session_id, session_cards in cards.groupby("session_id", sort=False):
        session_routes = route_groups.get(str(session_id))
        if session_routes is None or session_routes.empty:
            continue
        route_sequence = session_routes["event_seq"].astype(int).to_numpy()
        ordered_cards = session_cards.dropna(subset=["event_seq"]).sort_values(
            "event_seq", kind="mergesort"
        ).copy()
        card_sequence = ordered_cards["event_seq"].astype(int).to_numpy()
        entry_index = np.searchsorted(route_sequence, card_sequence, side="right") - 1
        has_entry = entry_index >= 0
        if not has_entry.any():
            continue
        ordered_cards = ordered_cards.loc[has_entry].copy()
        entry_index = entry_index[has_entry]
        entry_sequence = route_sequence[entry_index]
        strict_order = ordered_cards["event_seq"].astype(int).to_numpy() > entry_sequence
        ordered_cards = ordered_cards.loc[strict_order].copy()
        entry_index = entry_index[strict_order]
        if ordered_cards.empty:
            continue

        next_index = entry_index + 1
        has_exit = next_index < len(session_routes)
        ordered_cards["feed_entry_event_seq"] = route_sequence[entry_index]
        ordered_cards["feed_entry_time_utc"] = session_routes[
            "event_time_utc"
        ].to_numpy()[entry_index]
        ordered_cards["entry_route_kind"] = session_routes["route_kind"].astype(
            "string"
        ).to_numpy()[entry_index]
        ordered_cards["feed_spell_closed_by_page_view"] = has_exit
        ordered_cards["feed_exit_event_seq"] = pd.array(
            [route_sequence[index] if present else pd.NA for index, present in zip(next_index, has_exit)],
            dtype="Int64",
        )
        ordered_cards["feed_exit_time_utc"] = pd.to_datetime(
            [
                session_routes.iloc[index]["event_time_utc"] if present else pd.NaT
                for index, present in zip(next_index, has_exit)
            ],
            utc=True,
        )
        ordered_cards["feed_exit_route_kind"] = pd.array(
            [
                session_routes.iloc[index]["route_kind"] if present else pd.NA
                for index, present in zip(next_index, has_exit)
            ],
            dtype="string",
        )
        ordered_cards["feed_exit_route_item_id"] = pd.array(
            [
                session_routes.iloc[index]["route_item_id_int"] if present else pd.NA
                for index, present in zip(next_index, has_exit)
            ],
            dtype="Int64",
        )
        attached.append(ordered_cards)
    if not attached:
        return cards.iloc[0:0].copy()
    out = pd.concat(attached, ignore_index=True)
    return out[out["entry_route_kind"].astype(str).eq("rfeed")].copy()


def build_direct_thread_entry_panel(
    events: pd.DataFrame,
    session_qc: pd.DataFrame,
    reputation: pd.DataFrame,
    *,
    start_time: str,
    end_time: str | None,
    valid_days: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the strict feed-spell-post panel and exact route-link evidence."""

    _require_columns(events, EVENT_COLUMNS, "session events")
    humans, author_map = _authoritative_rosters(reputation)
    frame = events.copy()
    frame["session_id"] = frame["session_id"].astype(str)
    frame["event_seq"] = _int_series(frame["event_seq"])
    frame["event_time_utc"] = pd.to_datetime(
        frame["event_timestamp_utc"], utc=True, errors="coerce"
    )
    frame["experiment_day_derived"] = experiment_day_series(frame["event_time_utc"])
    frame["participant_id"] = _int_series(frame["y_user_id"])
    frame["item_id_int"] = _int_series(frame["item_id"])
    frame["root_post_id_int"] = _int_series(frame["root_post_id"])
    frame["route_item_id_int"] = _int_series(frame["route_item_id"])
    frame["author_id_int"] = _int_series(frame["author_id"])

    attrition_rows = _stage_counts(frame, "raw_event_rows")
    keep = _analysis_event_mask(
        frame,
        session_qc,
        start_time=start_time,
        end_time=end_time,
        valid_days=valid_days,
    )
    valid = frame.loc[keep].copy()
    attrition_rows.extend(_stage_counts(valid, "valid_day_quality_passing_event_rows"))

    participant_mask = [
        (str(group), int(user_id)) in humans if pd.notna(user_id) else False
        for group, user_id in valid[["group", "participant_id"]].itertuples(index=False, name=None)
    ]
    valid = valid.loc[participant_mask].copy()
    attrition_rows.extend(_stage_counts(valid, "authoritative_participant_event_rows"))

    cards = valid[
        valid["action"].astype(str).str.lower().eq("appear")
        & valid["type"].astype(str).str.lower().eq("post")
        & valid["route_kind"].astype(str).eq("rfeed")
        & valid["item_id_int"].notna()
    ].copy()
    attrition_rows.extend(_stage_counts(cards, "top_level_feed_card_appear_rows"))

    start = parse_time(start_time).tz_convert("UTC")
    content_time = pd.to_datetime(
        cards["backend_content_created_at"], utc=True, errors="coerce"
    )
    root_time = pd.to_datetime(
        cards["root_thread_created_at"], utc=True, errors="coerce"
    )
    content_valid = (
        content_time.notna()
        & root_time.notna()
        & content_time.ge(start)
        & root_time.ge(start)
        & experiment_day_series(content_time).isin([int(day) for day in valid_days])
        & experiment_day_series(root_time).isin([int(day) for day in valid_days])
    )
    cards = cards.loc[content_valid].copy()
    cards["backend_created_utc"] = content_time.loc[cards.index]
    attrition_rows.extend(_stage_counts(cards, "strict_valid_content_card_rows"))

    cards["expected_author_type"] = [
        author_map.get((str(group), int(author_id))) if pd.notna(author_id) else None
        for group, author_id in cards[["group", "author_id_int"]].itertuples(index=False, name=None)
    ]
    cards = cards[
        cards["author_type_ground_truth"].astype(str).isin(AUTHOR_ORDER)
        & cards["author_type_ground_truth"].astype(str).eq(cards["expected_author_type"])
    ].copy()
    attrition_rows.extend(_stage_counts(cards, "authoritative_author_card_rows"))

    expected_ids = "post_" + cards["item_id_int"].astype(str)
    structural = (
        cards["id"].astype(str).eq(expected_ids)
        & cards["semantic_group"].astype(str).eq("feed_container")
        & cards["item_id_int"].eq(cards["root_post_id_int"])
    )
    cards = cards.loc[structural].copy()
    attrition_rows.extend(_stage_counts(cards, "strict_root_feed_card_rows"))
    if cards.empty:
        raise ValueError("No eligible feed-card opportunities remain")

    page = valid[
        valid["action"].astype(str).str.lower().eq("page_view")
    ].sort_values(["session_id", "event_seq"], kind="mergesort").copy()
    cards = _attach_feed_route_spells(cards, page)
    attrition_rows.extend(_stage_counts(cards, "explicit_feed_route_card_rows"))
    if cards.empty:
        raise ValueError("No feed-route-bounded card opportunities remain")

    key = [
        "session_id",
        "group",
        "participant_id",
        "feed_entry_event_seq",
        "item_id_int",
    ]
    author_consistency = cards.groupby(key)["author_type_ground_truth"].nunique()
    if author_consistency.gt(1).any():
        raise ValueError("A feed-spell-post opportunity has conflicting author types")
    opportunities = (
        cards.sort_values(["session_id", "event_seq"], kind="mergesort")
        .groupby(key, as_index=False)
        .agg(
            author_id=("author_id_int", "first"),
            author_type=("author_type_ground_truth", "first"),
            experiment_day=("experiment_day_derived", "first"),
            backend_created_utc=("backend_created_utc", "first"),
            feed_entry_time_utc=("feed_entry_time_utc", "first"),
            feed_exit_event_seq=("feed_exit_event_seq", "first"),
            feed_exit_time_utc=("feed_exit_time_utc", "first"),
            feed_exit_route_kind=("feed_exit_route_kind", "first"),
            feed_exit_route_item_id=("feed_exit_route_item_id", "first"),
            feed_spell_closed_by_page_view=(
                "feed_spell_closed_by_page_view",
                "first",
            ),
            first_appear_event_seq=("event_seq", "min"),
            last_appear_event_seq=("event_seq", "max"),
            first_appear_time_utc=("event_time_utc", "min"),
            last_appear_time_utc=("event_time_utc", "max"),
            top_level_appear_events=("event_seq", "size"),
        )
    )
    opportunities = opportunities.rename(columns={"item_id_int": "post_id"})
    direct_open = (
        opportunities["feed_spell_closed_by_page_view"].astype(bool)
        & opportunities["feed_exit_route_kind"].astype(str).eq("rthread")
        & opportunities["feed_exit_route_item_id"].eq(opportunities["post_id"])
    ).fillna(False)
    panel = opportunities.copy()
    panel[PRIMARY_OUTCOME] = direct_open.astype(int)
    panel["direct_open_event_seq"] = panel["feed_exit_event_seq"].where(direct_open)
    panel["direct_open_time_utc"] = panel["feed_exit_time_utc"].where(direct_open)
    panel["direct_open_events"] = panel[PRIMARY_OUTCOME]
    panel["opportunity_id"] = (
        panel["group"].astype(str)
        + ":"
        + panel["session_id"].astype(str)
        + ":"
        + panel["feed_entry_event_seq"].astype(int).astype(str)
        + ":"
        + panel["post_id"].astype(int).astype(str)
    )
    panel["participant_key"] = (
        panel["group"].astype(str)
        + ":"
        + panel["participant_id"].astype(int).astype(str)
    )
    panel["post_key"] = (
        panel["group"].astype(str)
        + ":"
        + panel["post_id"].astype(int).astype(str)
    )
    panel["feed_spell_key"] = (
        panel["group"].astype(str)
        + ":"
        + panel["session_id"].astype(str)
        + ":"
        + panel["feed_entry_event_seq"].astype(int).astype(str)
    )
    panel["g2"] = panel["group"].eq("G2").astype(int)
    panel["ai_author"] = panel["author_type"].eq("AI").astype(int)
    panel["post_age_hours"] = (
        (panel["first_appear_time_utc"] - panel["backend_created_utc"])
        .dt.total_seconds()
        .div(3600.0)
        .clip(lower=0.0)
    )
    panel["post_fix_period"] = panel["experiment_day"].astype(int).ge(18)
    session_cards = panel.groupby("session_id")["opportunity_id"].transform("size")
    session_types = panel.groupby("session_id")["author_type"].transform("nunique")
    spell_cards = panel.groupby("feed_spell_key")["opportunity_id"].transform("size")
    spell_types = panel.groupby("feed_spell_key")["author_type"].transform("nunique")
    panel["session_card_opportunities"] = session_cards.astype(int)
    panel["session_contains_both_author_types"] = session_types.eq(2)
    panel["feed_spell_card_opportunities"] = spell_cards.astype(int)
    panel["feed_spell_contains_both_author_types"] = spell_types.eq(2)
    panel = panel.sort_values(
        [
            "group",
            "participant_id",
            "session_id",
            "feed_entry_event_seq",
            "post_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    evidence = (
        panel.loc[
            panel[PRIMARY_OUTCOME].eq(1),
            [
                "session_id",
                "group",
                "participant_id",
                "post_id",
                "feed_entry_event_seq",
                "feed_entry_time_utc",
                "last_appear_event_seq",
                "last_appear_time_utc",
                "direct_open_event_seq",
                "direct_open_time_utc",
            ],
        ]
        .rename(
            columns={
                "feed_entry_event_seq": "previous_route_event_seq",
                "feed_entry_time_utc": "previous_route_time_utc",
                "last_appear_event_seq": "linked_appear_event_seq",
                "last_appear_time_utc": "linked_appear_time_utc",
            }
        )
        .reset_index(drop=True)
        .reset_index(names="route_open_id")
    )
    attrition_rows.extend(
        _stage_counts(
            panel.rename(columns={"participant_id": "y_user_id"}),
            "unique_feed_spell_root_post_opportunities",
        )
    )
    return panel, evidence.reset_index(drop=True), pd.DataFrame(attrition_rows)


def panel_signature(panel: pd.DataFrame) -> str:
    """Hash the canonical opportunity rows independently of row order."""

    _require_columns(panel, PANEL_SIGNATURE_COLUMNS, "RQ1b panel")
    canonical = panel[list(PANEL_SIGNATURE_COLUMNS)].copy()
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("string")
        else:
            canonical[column] = canonical[column].astype("string")
        canonical[column] = canonical[column].fillna("<NA>")
    canonical = canonical.sort_values(list(PANEL_SIGNATURE_COLUMNS), kind="mergesort")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cell_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Return observed endpoint counts by condition and author type."""

    rows = (
        panel.groupby(["group", "author_type"], observed=True)
        .agg(
            direct_opens=(PRIMARY_OUTCOME, "sum"),
            rendered_cards=(PRIMARY_OUTCOME, "size"),
            participants=("participant_key", "nunique"),
            sessions=("session_id", "nunique"),
            posts=("post_key", "nunique"),
        )
        .reset_index()
    )
    rows["raw_open_rate"] = rows["direct_opens"] / rows["rendered_cards"]
    rows["condition"] = rows["group"].map(
        {"G1": "labels_visible", "G2": "labels_hidden"}
    )
    return rows.sort_values(["group", "author_type"]).reset_index(drop=True)


def observed_design_sensitivities(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize unadjusted endpoint contrasts under prespecified restrictions."""

    _require_columns(
        panel,
        (
            "group",
            "author_type",
            "session_id",
            "post_id",
            "feed_spell_closed_by_page_view",
            "feed_spell_contains_both_author_types",
            "session_contains_both_author_types",
            "post_fix_period",
            PRIMARY_OUTCOME,
        ),
        "RQ1b opportunity panel",
    )
    variants: list[tuple[str, pd.DataFrame, str]] = [
        ("primary_feed_spell", panel, "one post within one explicit feed-route spell"),
        (
            "closed_feed_spell",
            panel[panel["feed_spell_closed_by_page_view"].astype(bool)],
            "feed spells followed by another recorded page view",
        ),
        (
            "both_author_feed_spell",
            panel[panel["feed_spell_contains_both_author_types"].astype(bool)],
            "feed spells containing both AI-authored and participant-authored posts",
        ),
        (
            "both_author_session",
            panel[panel["session_contains_both_author_types"].astype(bool)],
            "sessions containing both AI-authored and participant-authored posts",
        ),
        (
            "pre_fix_valid_days",
            panel[~panel["post_fix_period"].astype(bool)],
            "primary-valid days 2 through 9",
        ),
        (
            "post_fix_valid_days",
            panel[panel["post_fix_period"].astype(bool)],
            "primary-valid days 18 through 30",
        ),
    ]
    collapsed = (
        panel.sort_values(
            ["session_id", "post_id", "feed_entry_event_seq"], kind="mergesort"
        )
        .groupby(["session_id", "group", "participant_id", "post_id"], as_index=False)
        .agg(
            author_type=("author_type", "first"),
            direct_thread_entry=(PRIMARY_OUTCOME, "max"),
        )
    )
    variants.append(
        (
            "collapsed_session_post",
            collapsed,
            "one post within one session among explicit feed-route spells",
        )
    )
    rows: list[dict[str, Any]] = []
    for name, frame, definition in variants:
        counts = frame.groupby(["group", "author_type"])[PRIMARY_OUTCOME].agg(
            ["sum", "size", "mean"]
        )
        expected = pd.MultiIndex.from_product(
            [GROUP_ORDER, AUTHOR_ORDER], names=["group", "author_type"]
        )
        counts = counts.reindex(expected)
        counts[["sum", "size"]] = counts[["sum", "size"]].fillna(0)
        rate = {
            (group, author): float(counts.loc[(group, author), "mean"])
            for group in GROUP_ORDER
            for author in AUTHOR_ORDER
        }
        rows.append(
            {
                "analysis_variant": name,
                "opportunity_definition": definition,
                "complete_group_author_cells": bool(counts["mean"].notna().all()),
                "opportunities": int(counts["size"].sum()),
                "direct_opens": int(counts["sum"].sum()),
                "g1_ai_opens": int(counts.loc[("G1", "AI"), "sum"]),
                "g1_ai_opportunities": int(counts.loc[("G1", "AI"), "size"]),
                "g1_ai_rate": rate[("G1", "AI")],
                "g1_h_opens": int(counts.loc[("G1", "H"), "sum"]),
                "g1_h_opportunities": int(counts.loc[("G1", "H"), "size"]),
                "g1_h_rate": rate[("G1", "H")],
                "g2_ai_opens": int(counts.loc[("G2", "AI"), "sum"]),
                "g2_ai_opportunities": int(counts.loc[("G2", "AI"), "size"]),
                "g2_ai_rate": rate[("G2", "AI")],
                "g2_h_opens": int(counts.loc[("G2", "H"), "sum"]),
                "g2_h_opportunities": int(counts.loc[("G2", "H"), "size"]),
                "g2_h_rate": rate[("G2", "H")],
                "g1_ai_minus_h": rate[("G1", "AI")] - rate[("G1", "H")],
                "g2_ai_minus_h": rate[("G2", "AI")] - rate[("G2", "H")],
                "difference_in_differences": (
                    rate[("G2", "AI")]
                    - rate[("G2", "H")]
                    - rate[("G1", "AI")]
                    + rate[("G1", "H")]
                ),
            }
        )
    return pd.DataFrame(rows)


def participant_deletion_descriptive(panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute the unadjusted focal contrast after deleting each participant."""

    full = observed_design_sensitivities(panel)
    full_did = float(
        full.loc[
            full["analysis_variant"].eq("primary_feed_spell"),
            "difference_in_differences",
        ].iloc[0]
    )
    rows: list[dict[str, Any]] = []
    for participant_key in sorted(panel["participant_key"].astype(str).unique()):
        deleted = panel[panel["participant_key"].astype(str).ne(participant_key)]
        counts = deleted.groupby(["group", "author_type"])[PRIMARY_OUTCOME].agg(
            ["sum", "size", "mean"]
        )
        did = (
            float(counts.loc[("G2", "AI"), "mean"])
            - float(counts.loc[("G2", "H"), "mean"])
            - float(counts.loc[("G1", "AI"), "mean"])
            + float(counts.loc[("G1", "H"), "mean"])
        )
        removed = panel[panel["participant_key"].astype(str).eq(participant_key)]
        rows.append(
            {
                "deleted_participant_key": participant_key,
                "deleted_group": str(removed["group"].iloc[0]),
                "deleted_opportunities": int(len(removed)),
                "deleted_direct_opens": int(removed[PRIMARY_OUTCOME].sum()),
                "difference_in_differences": did,
                "change_from_full": did - full_did,
                "direction_positive": bool(did > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("difference_in_differences").reset_index(
        drop=True
    )


def endpoint_source_audit(
    panel: pd.DataFrame,
    vendor_thread_opens: pd.DataFrame,
) -> pd.DataFrame:
    """Check every inferred positive against the vendor thread-open export."""

    _require_columns(
        vendor_thread_opens,
        ("session_id", "post_id", "thread_open_time"),
        "vendor thread opens",
    )
    positives = panel[panel[PRIMARY_OUTCOME].eq(1)][
        ["group", "session_id", "post_id", "direct_open_time_utc"]
    ].copy()
    vendor = vendor_thread_opens[
        ["session_id", "post_id", "thread_open_time"]
    ].copy()
    vendor["session_id"] = vendor["session_id"].astype(str)
    vendor["post_id"] = pd.to_numeric(
        vendor["post_id"].astype(str).str.extract(r"^post_(\d+)$", expand=False),
        errors="coerce",
    ).astype("Int64")
    vendor["thread_open_time"] = pd.to_datetime(
        vendor["thread_open_time"], utc=True, errors="coerce", format="mixed"
    )
    vendor_keys = set(
        zip(
            vendor["session_id"],
            vendor["post_id"],
            vendor["thread_open_time"],
        )
    )
    positives["exact_vendor_match"] = [
        (str(session_id), int(post_id), pd.Timestamp(open_time)) in vendor_keys
        for session_id, post_id, open_time in positives[
            ["session_id", "post_id", "direct_open_time_utc"]
        ].itertuples(index=False, name=None)
    ]
    rows: list[dict[str, Any]] = []
    for group in (*GROUP_ORDER, "all"):
        part = positives if group == "all" else positives[positives["group"].eq(group)]
        matched = int(part["exact_vendor_match"].sum())
        rows.append(
            {
                "group": group,
                "direct_opens": int(len(part)),
                "exact_vendor_matches": matched,
                "unmatched_direct_opens": int(len(part) - matched),
                "exact_match_rate": float(matched / len(part)) if len(part) else np.nan,
                "passed": bool(len(part) > 0 and matched == len(part)),
            }
        )
    return pd.DataFrame(rows)


def coverage_summary(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    session_qc: pd.DataFrame,
    reputation: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize overall telemetry and analytic endpoint coverage."""

    humans, _ = _authoritative_rosters(reputation)
    events = events.copy()
    events["participant_id"] = _int_series(events["y_user_id"])
    events["session_id"] = events["session_id"].astype(str)
    qc = session_qc[["session_id", "usable_session"]].copy()
    qc["session_id"] = qc["session_id"].astype(str)
    event_sessions = events[["session_id", "group", "participant_id"]].drop_duplicates()
    event_sessions = event_sessions[
        [
            (str(group), int(user_id)) in humans if pd.notna(user_id) else False
            for group, user_id in event_sessions[["group", "participant_id"]].itertuples(index=False, name=None)
        ]
    ].merge(qc, on="session_id", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        arm = panel[panel["group"].eq(group)]
        raw = event_sessions[event_sessions["group"].eq(group)]
        expected = int(GROUPS[group].human_count)
        contributions = arm.groupby("participant_key").size()
        rows.append(
            {
                "group": group,
                "condition": "labels_visible" if group == "G1" else "labels_hidden",
                "expected_participants": expected,
                "participants_with_any_telemetry": int(raw["participant_id"].nunique()),
                "participants_with_usable_sessions": int(
                    raw.loc[raw["usable_session"].fillna(False), "participant_id"].nunique()
                ),
                "participants_in_endpoint": int(arm["participant_key"].nunique()),
                "raw_participant_sessions": int(raw["session_id"].nunique()),
                "usable_participant_sessions": int(
                    raw.loc[raw["usable_session"].fillna(False), "session_id"].nunique()
                ),
                "endpoint_sessions": int(arm["session_id"].nunique()),
                "rendered_cards": int(len(arm)),
                "direct_opens": int(arm[PRIMARY_OUTCOME].sum()),
                "minimum_cards_per_endpoint_participant": int(contributions.min()),
                "maximum_cards_per_endpoint_participant": int(contributions.max()),
            }
        )
    return pd.DataFrame(rows)


def integrity_checks(
    panel: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    valid_days: Sequence[int],
) -> pd.DataFrame:
    """Evaluate deterministic endpoint invariants and fail closed."""

    expected_open_keys = set(
        zip(
            evidence["session_id"].astype(str),
            evidence["group"].astype(str),
            evidence["participant_id"].astype(int),
            evidence["previous_route_event_seq"].astype(int),
            evidence["post_id"].astype(int),
        )
    )
    observed_open = panel[panel[PRIMARY_OUTCOME].eq(1)]
    observed_open_keys = set(
        zip(
            observed_open["session_id"].astype(str),
            observed_open["group"].astype(str),
            observed_open["participant_id"].astype(int),
            observed_open["feed_entry_event_seq"].astype(int),
            observed_open["post_id"].astype(int),
        )
    )
    checks = [
        (
            "opportunity_ids_unique",
            bool(panel["opportunity_id"].is_unique),
            f"unique={panel['opportunity_id'].nunique()}; rows={len(panel)}",
        ),
        (
            "known_groups",
            bool(panel["group"].isin(GROUP_ORDER).all()),
            ",".join(sorted(panel["group"].astype(str).unique())),
        ),
        (
            "known_author_types",
            bool(panel["author_type"].isin(AUTHOR_ORDER).all()),
            ",".join(sorted(panel["author_type"].astype(str).unique())),
        ),
        (
            "primary_valid_days_only",
            bool(panel["experiment_day"].isin([int(day) for day in valid_days]).all()),
            ",".join(str(value) for value in sorted(panel["experiment_day"].astype(int).unique())),
        ),
        (
            "binary_outcome",
            bool(panel[PRIMARY_OUTCOME].isin([0, 1]).all()),
            str(panel[PRIMARY_OUTCOME].value_counts().sort_index().to_dict()),
        ),
        (
            "every_positive_has_exact_route_evidence",
            observed_open_keys == expected_open_keys,
            f"panel_positive_keys={len(observed_open_keys)}; evidence_keys={len(expected_open_keys)}",
        ),
        (
            "route_order_is_strict",
            bool(
                evidence["previous_route_event_seq"].lt(evidence["linked_appear_event_seq"]).all()
                and evidence["linked_appear_event_seq"].lt(evidence["direct_open_event_seq"]).all()
            ),
            f"linked_route_events={len(evidence)}",
        ),
        (
            "every_opportunity_has_explicit_feed_entry",
            bool(
                panel["feed_entry_event_seq"].notna().all()
                and panel["first_appear_event_seq"].gt(
                    panel["feed_entry_event_seq"]
                ).all()
            ),
            f"opportunities={len(panel)}",
        ),
        (
            "feed_exit_defines_every_positive",
            bool(
                observed_open["feed_spell_closed_by_page_view"].astype(bool).all()
                and observed_open["feed_exit_route_kind"].astype(str).eq("rthread").all()
                and observed_open["feed_exit_route_item_id"].eq(
                    observed_open["post_id"]
                ).all()
            ),
            f"direct_opens={len(observed_open)}",
        ),
        (
            "nonnegative_post_age",
            bool(panel["post_age_hours"].ge(0.0).all()),
            f"minimum_hours={panel['post_age_hours'].min():.6f}",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "passed": passed, "observed": observed}
            for name, passed, observed in checks
        ]
    )


def _source_paths(config: AnalysisConfig, config_path: Path) -> tuple[Path, ...]:
    return (
        config_path.resolve(),
        (config.browser_dir / "session_events.parquet").resolve(),
        (config.browser_dir / "session_qc.parquet").resolve(),
        (config.processed_dir / "snapshot_user_reputation.parquet").resolve(),
        (config.project_root / VENDOR_THREAD_OPENS_RELATIVE).resolve(),
        Path(__file__).resolve(),
    )


def _implementation_hash(functions: Sequence[Any], constants: Mapping[str, Any]) -> str:
    payload = json.dumps(constants, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload)
    for function in functions:
        digest.update(inspect.getsource(function).encode("utf-8"))
    return digest.hexdigest()


def panel_implementation_sha256() -> str:
    """Hash code and constants that define the endpoint panel."""

    return _implementation_hash(
        (
            _analysis_event_mask,
            _authoritative_rosters,
            _attach_feed_route_spells,
            build_direct_thread_entry_panel,
            panel_signature,
        ),
        {
            "schema": ANALYSIS_SCHEMA,
            "event_columns": EVENT_COLUMNS,
            "panel_signature_columns": PANEL_SIGNATURE_COLUMNS,
            "primary_outcome": PRIMARY_OUTCOME,
        },
    )


def model_implementation_sha256() -> str:
    """Hash code and constants that define fitting and posterior estimands."""

    return _implementation_hash(
        (
            build_model_frame,
            _model_coordinates,
            build_pymc_model,
            fit_model,
            _posterior_array,
            _summarize_draws,
            posterior_estimands,
            parameter_summary,
            mcmc_diagnostics,
            fitted_calibration,
        ),
        {
            "schema": ANALYSIS_SCHEMA,
            "primary_outcome": PRIMARY_OUTCOME,
            "mcmc_rhat_threshold": MCMC_RHAT_THRESHOLD,
            "mcmc_ess_threshold": MCMC_ESS_THRESHOLD,
        },
    )


def implementation_sha256() -> str:
    """Hash the complete module for the source manifest."""

    return sha256_file(Path(__file__).resolve())


def prepare_analysis(
    config: AnalysisConfig,
    config_path: Path,
) -> Mapping[str, pd.DataFrame]:
    """Build and write the deterministic RQ1b analytic surfaces."""

    events = _read_events(config.browser_dir)
    session_qc = read_table(config.browser_dir / "session_qc")
    reputation = read_table(config.processed_dir / "snapshot_user_reputation")
    vendor_thread_opens = pd.read_csv(
        config.project_root / VENDOR_THREAD_OPENS_RELATIVE
    )
    valid_days = analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"]
    panel, evidence, attrition = build_direct_thread_entry_panel(
        events,
        session_qc,
        reputation,
        start_time=config.start_time,
        end_time=config.end_time,
        valid_days=valid_days,
    )
    tables: dict[str, pd.DataFrame] = {
        "rq1b_opportunity_panel": panel,
        "rq1b_direct_open_evidence": evidence,
        "rq1b_attrition": attrition,
        "rq1b_cell_summary": cell_summary(panel),
        "rq1b_observed_design_sensitivities": observed_design_sensitivities(panel),
        "rq1b_participant_deletion_descriptive": participant_deletion_descriptive(
            panel
        ),
        "rq1b_endpoint_source_audit": endpoint_source_audit(
            panel, vendor_thread_opens
        ),
        "rq1b_coverage": coverage_summary(panel, events, session_qc, reputation),
        "rq1b_integrity_checks": integrity_checks(panel, evidence, valid_days=valid_days),
        "rq1b_panel_signature": pd.DataFrame(
            [
                {
                    "schema": ANALYSIS_SCHEMA,
                    "panel_sha256": panel_signature(panel),
                    "panel_implementation_sha256": panel_implementation_sha256(),
                    "module_sha256": implementation_sha256(),
                    "rows": len(panel),
                    "direct_opens": int(panel[PRIMARY_OUTCOME].sum()),
                    "analysis_set": config.analysis_set,
                }
            ]
        ),
    }
    for name, table in tables.items():
        write_table(table, config.tables_dir / name)
    sources = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _source_paths(config, config_path)
    ]
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "run_id": config.run_id,
        "profile": config.profile,
        "analysis_set": config.analysis_set,
        "panel_sha256": panel_signature(panel),
        "panel_implementation_sha256": panel_implementation_sha256(),
        "module_sha256": implementation_sha256(),
        "sources": sources,
        "runtime_versions": runtime_versions(),
    }
    write_json(manifest, config.run_dir / "rq1b_manifest.json")
    return tables


def build_model_frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode the opportunity panel and record continuous-predictor scalers."""

    required = {
        "opportunity_id",
        "group",
        "author_type",
        "participant_key",
        "session_id",
        "post_key",
        "experiment_day",
        "post_age_hours",
        "feed_spell_card_opportunities",
        PRIMARY_OUTCOME,
    }
    _require_columns(panel, required, "RQ1b opportunity panel")
    frame = panel.copy().reset_index(drop=True)
    frame["g2"] = frame["group"].eq("G2").astype(int)
    frame["ai_author"] = frame["author_type"].eq("AI").astype(int)
    frame["g2_x_ai_author"] = frame["g2"] * frame["ai_author"]
    scalers: list[dict[str, Any]] = []
    for source, target in (
        ("post_age_hours", "z_log_post_age"),
        ("feed_spell_card_opportunities", "z_log_feed_spell_cards"),
    ):
        raw = np.log1p(pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float))
        if not np.isfinite(raw).all():
            raise ValueError(f"RQ1b model predictor {source} contains non-finite values")
        mean = float(raw.mean())
        sd = float(raw.std(ddof=0))
        if not np.isfinite(sd):
            raise ValueError(f"RQ1b model predictor {source} has invalid variance")
        scaling_sd = sd if sd > 0.0 else 1.0
        frame[target] = (raw - mean) / scaling_sd
        scalers.append(
            {
                "source": source,
                "transformation": "log1p_then_standardize",
                "model_term": target,
                "training_mean": mean,
                "training_sd": sd,
            }
        )
    categorical = (
        ("participant_key", "participant_index", "participant"),
        ("session_id", "session_index", "session"),
        ("post_key", "post_index", "post"),
        ("experiment_day", "day_index", "valid_day"),
    )
    for source, target, label in categorical:
        labels = sorted(pd.unique(frame[source]), key=lambda value: str(value))
        mapping = {value: index for index, value in enumerate(labels)}
        frame[target] = frame[source].map(mapping).astype(int)
        for value, index in mapping.items():
            scalers.append(
                {
                    "source": source,
                    "transformation": "categorical_index",
                    "model_term": label,
                    "training_mean": np.nan,
                    "training_sd": np.nan,
                    "level": str(value),
                    "level_index": int(index),
                }
            )
    return frame, pd.DataFrame(scalers)


def _model_coordinates(frame: pd.DataFrame) -> dict[str, list[Any]]:
    def ordered_labels(column: str, index: str) -> list[Any]:
        return (
            frame[[column, index]]
            .drop_duplicates()
            .sort_values(index)[column]
            .astype(str)
            .tolist()
        )

    return {
        "opportunity": frame["opportunity_id"].astype(str).tolist(),
        "participant": ordered_labels("participant_key", "participant_index"),
        "session": ordered_labels("session_id", "session_index"),
        "post": ordered_labels("post_key", "post_index"),
        "valid_day": ordered_labels("experiment_day", "day_index"),
    }


def build_pymc_model(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    prior_multiplier: float = 1.0,
) -> Any:
    """Build the crossed multilevel Bernoulli model without sampling."""

    if prior_multiplier <= 0.0:
        raise ValueError("prior_multiplier must be positive")
    import pymc as pm

    coords = _model_coordinates(frame)
    focal = float(config.sampling.prior_focal_sigma) * prior_multiplier
    adjustment = float(config.sampling.prior_adjustment_sigma) * prior_multiplier
    scale = float(config.sampling.prior_scale_sigma) * prior_multiplier
    with pm.Model(coords=coords) as model:
        g2 = pm.Data("g2", frame["g2"].to_numpy(dtype=int), dims="opportunity")
        ai = pm.Data(
            "ai_author", frame["ai_author"].to_numpy(dtype=int), dims="opportunity"
        )
        participant = pm.Data(
            "participant_index",
            frame["participant_index"].to_numpy(dtype=int),
            dims="opportunity",
        )
        session = pm.Data(
            "session_index", frame["session_index"].to_numpy(dtype=int), dims="opportunity"
        )
        post = pm.Data(
            "post_index", frame["post_index"].to_numpy(dtype=int), dims="opportunity"
        )
        valid_day = pm.Data(
            "day_index", frame["day_index"].to_numpy(dtype=int), dims="opportunity"
        )
        z_age = pm.Data(
            "z_log_post_age",
            frame["z_log_post_age"].to_numpy(dtype=float),
            dims="opportunity",
        )
        z_cards = pm.Data(
            "z_log_feed_spell_cards",
            frame["z_log_feed_spell_cards"].to_numpy(dtype=float),
            dims="opportunity",
        )

        alpha = pm.Normal("alpha", mu=-2.0, sigma=1.5 * prior_multiplier)
        beta_group = pm.Normal("beta_group", mu=0.0, sigma=adjustment)
        beta_ai = pm.Normal("beta_ai", mu=0.0, sigma=focal)
        beta_group_ai = pm.Normal("beta_group_ai", mu=0.0, sigma=focal)
        beta_log_age = pm.Normal("beta_log_age", mu=0.0, sigma=adjustment)
        beta_log_feed_spell_cards = pm.Normal(
            "beta_log_feed_spell_cards", mu=0.0, sigma=adjustment
        )

        sigma_participant = pm.HalfNormal("sigma_participant", sigma=scale)
        sigma_participant_ai = pm.HalfNormal("sigma_participant_ai", sigma=scale)
        sigma_session = pm.HalfNormal("sigma_session", sigma=scale)
        sigma_post = pm.HalfNormal("sigma_post", sigma=scale)
        sigma_day = pm.HalfNormal("sigma_day", sigma=scale)
        z_participant = pm.Normal(
            "z_participant", mu=0.0, sigma=1.0, dims="participant"
        )
        z_participant_ai = pm.Normal(
            "z_participant_ai", mu=0.0, sigma=1.0, dims="participant"
        )
        z_session = pm.Normal("z_session", mu=0.0, sigma=1.0, dims="session")
        z_post = pm.Normal("z_post", mu=0.0, sigma=1.0, dims="post")
        z_day = pm.Normal("z_day", mu=0.0, sigma=1.0, dims="valid_day")
        participant_effect = pm.Deterministic(
            "participant_effect", z_participant * sigma_participant, dims="participant"
        )
        participant_ai_effect = pm.Deterministic(
            "participant_ai_effect",
            z_participant_ai * sigma_participant_ai,
            dims="participant",
        )
        session_effect = pm.Deterministic(
            "session_effect", z_session * sigma_session, dims="session"
        )
        post_effect = pm.Deterministic("post_effect", z_post * sigma_post, dims="post")
        day_effect = pm.Deterministic(
            "day_effect", z_day * sigma_day, dims="valid_day"
        )
        eta = (
            alpha
            + beta_group * g2
            + beta_ai * ai
            + beta_group_ai * g2 * ai
            + beta_log_age * z_age
            + beta_log_feed_spell_cards * z_cards
            + participant_effect[participant]
            + participant_ai_effect[participant] * ai
            + session_effect[session]
            + post_effect[post]
            + day_effect[valid_day]
        )
        pm.Bernoulli(
            "opened",
            logit_p=eta,
            observed=frame[PRIMARY_OUTCOME].to_numpy(dtype=int),
            dims="opportunity",
        )
    return model


def fit_model(
    frame: pd.DataFrame,
    config: AnalysisConfig,
    *,
    prior_multiplier: float = 1.0,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    progressbar: bool = True,
) -> Any:
    """Fit one RQ1b model and retain pointwise log likelihood."""

    import pymc as pm

    requested_chains = int(config.sampling.chains if chains is None else chains)
    model = build_pymc_model(frame, config, prior_multiplier=prior_multiplier)
    with model:
        return pm.sample(
            draws=int(config.sampling.draws if draws is None else draws),
            tune=int(config.sampling.tune if tune is None else tune),
            chains=requested_chains,
            cores=min(requested_chains, 4),
            target_accept=float(config.sampling.target_accept),
            random_seed=int(config.seed + round(1000 * prior_multiplier)),
            progressbar=progressbar,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )


def build_participant_aggregate_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse feed-spell outcomes to participant-by-author-type counts."""

    frame = (
        panel.groupby(
            ["participant_key", "group", "author_type"],
            observed=True,
            as_index=False,
        )[PRIMARY_OUTCOME]
        .agg(successes="sum", opportunities="size")
    )
    participant_levels = sorted(frame["participant_key"].astype(str).unique())
    mapping = {value: index for index, value in enumerate(participant_levels)}
    frame["participant_index"] = frame["participant_key"].astype(str).map(mapping)
    frame["g2"] = frame["group"].eq("G2").astype(int)
    frame["ai_author"] = frame["author_type"].eq("AI").astype(int)
    return frame


def build_participant_aggregate_model(
    frame: pd.DataFrame,
    config: AnalysisConfig,
) -> Any:
    """Build a participant-aggregated hierarchical binomial sensitivity model."""

    import pymc as pm

    participant_labels = (
        frame[["participant_key", "participant_index"]]
        .drop_duplicates()
        .sort_values("participant_index")["participant_key"]
        .astype(str)
        .tolist()
    )
    focal = float(config.sampling.prior_focal_sigma)
    adjustment = float(config.sampling.prior_adjustment_sigma)
    scale = float(config.sampling.prior_scale_sigma)
    with pm.Model(coords={"participant": participant_labels}) as model:
        alpha = pm.Normal("alpha", mu=-2.0, sigma=1.5)
        beta_group = pm.Normal("beta_group", mu=0.0, sigma=adjustment)
        beta_ai = pm.Normal("beta_ai", mu=0.0, sigma=focal)
        beta_group_ai = pm.Normal("beta_group_ai", mu=0.0, sigma=focal)
        sigma_participant = pm.HalfNormal("sigma_participant", sigma=scale)
        sigma_participant_ai = pm.HalfNormal(
            "sigma_participant_ai", sigma=scale
        )
        z_participant = pm.Normal(
            "z_participant", mu=0.0, sigma=1.0, dims="participant"
        )
        z_participant_ai = pm.Normal(
            "z_participant_ai", mu=0.0, sigma=1.0, dims="participant"
        )
        participant_effect = pm.Deterministic(
            "participant_effect",
            z_participant * sigma_participant,
            dims="participant",
        )
        participant_ai_effect = pm.Deterministic(
            "participant_ai_effect",
            z_participant_ai * sigma_participant_ai,
            dims="participant",
        )
        participant_index = frame["participant_index"].to_numpy(dtype=int)
        g2 = frame["g2"].to_numpy(dtype=int)
        ai = frame["ai_author"].to_numpy(dtype=int)
        eta = (
            alpha
            + beta_group * g2
            + beta_ai * ai
            + beta_group_ai * g2 * ai
            + participant_effect[participant_index]
            + participant_ai_effect[participant_index] * ai
        )
        pm.Binomial(
            "opened_count",
            n=frame["opportunities"].to_numpy(dtype=int),
            logit_p=eta,
            observed=frame["successes"].to_numpy(dtype=int),
        )
    return model


def participant_aggregate_estimands(idata: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-participant contrasts from the aggregate sensitivity model."""

    from scipy.special import expit

    alpha = _posterior_array(idata, "alpha")
    beta_group = _posterior_array(idata, "beta_group")
    beta_ai = _posterior_array(idata, "beta_ai")
    beta_group_ai = _posterior_array(idata, "beta_group_ai")
    participant_effect = _posterior_array(
        idata, "participant_effect", "participant"
    )
    participant_ai_effect = _posterior_array(
        idata, "participant_ai_effect", "participant"
    )
    arrays: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        participant_indices = np.sort(
            frame.loc[frame["group"].eq(group), "participant_index"].unique()
        )
        g2 = float(group == "G2")
        for author_type in AUTHOR_ORDER:
            ai = float(author_type == "AI")
            eta = (
                alpha[:, None]
                + beta_group[:, None] * g2
                + beta_ai[:, None] * ai
                + beta_group_ai[:, None] * g2 * ai
                + participant_effect[:, participant_indices]
                + participant_ai_effect[:, participant_indices] * ai
            )
            arrays[f"{group}_{author_type}"] = expit(eta).mean(axis=1)

    scope = "participant_aggregated_equal_participant"
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        for author_type in AUTHOR_ORDER:
            rows.append(
                _summarize_draws(
                    arrays[f"{group}_{author_type}"],
                    estimand="direct_thread_entry_probability",
                    group=group,
                    author_type=author_type,
                    standardization_scope=scope,
                )
            )
        rows.append(
            _summarize_draws(
                arrays[f"{group}_AI"] - arrays[f"{group}_H"],
                estimand="AI_minus_H_direct_thread_entry_probability",
                group=group,
                author_type="AI-H",
                standardization_scope=scope,
            )
        )
    for author_type in AUTHOR_ORDER:
        rows.append(
            _summarize_draws(
                arrays[f"G2_{author_type}"] - arrays[f"G1_{author_type}"],
                estimand="G2_minus_G1_direct_thread_entry_probability",
                group="G2-G1",
                author_type=author_type,
                standardization_scope=scope,
            )
        )
    rows.append(
        _summarize_draws(
            arrays["G2_AI"]
            - arrays["G2_H"]
            - arrays["G1_AI"]
            + arrays["G1_H"],
            estimand="G2_minus_G1_change_in_AI_minus_H_gap",
            group="G2-G1",
            author_type="AI-H",
            standardization_scope=scope,
        )
    )
    return pd.DataFrame(rows)


def _posterior_array(idata: Any, variable: str, dimension: str | None = None) -> np.ndarray:
    values = idata.posterior[variable].stack(sample=("chain", "draw"))
    if dimension is None:
        return values.transpose("sample").values
    return values.transpose("sample", dimension).values


def _summarize_draws(
    draws: np.ndarray,
    *,
    estimand: str,
    group: str,
    author_type: str,
    standardization_scope: str,
) -> dict[str, Any]:
    values = np.asarray(draws, dtype=float)
    return {
        "estimand": estimand,
        "group": group,
        "author_type": author_type,
        "posterior_mean": float(values.mean()),
        "posterior_median": float(np.median(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "posterior_probability_gt_zero": float(np.mean(values > 0.0)),
        "posterior_draws": int(values.size),
        "standardization_scope": standardization_scope,
        "inference_boundary": "one_community_per_condition_ecological_contrast",
    }


def posterior_estimands(
    idata: Any,
    frame: pd.DataFrame,
    *,
    draw_chunk_size: int = 64,
) -> pd.DataFrame:
    """Compute equal-participant and card-weighted probability estimands."""

    from scipy.special import expit

    if draw_chunk_size <= 0:
        raise ValueError("draw_chunk_size must be positive")
    scalar_names = (
        "alpha",
        "beta_group",
        "beta_ai",
        "beta_group_ai",
        "beta_log_age",
        "beta_log_feed_spell_cards",
    )
    scalars = {name: _posterior_array(idata, name) for name in scalar_names}
    participant_effect = _posterior_array(idata, "participant_effect", "participant")
    participant_ai_effect = _posterior_array(
        idata, "participant_ai_effect", "participant"
    )
    session_effect = _posterior_array(idata, "session_effect", "session")
    post_effect = _posterior_array(idata, "post_effect", "post")
    day_effect = _posterior_array(idata, "day_effect", "valid_day")
    sample_count = len(scalars["alpha"])
    arrays: dict[str, dict[str, np.ndarray]] = {
        "equal_participant_observed_opportunities": {},
        "card_weighted_observed_opportunities": {},
    }
    group_values = frame["group"].astype(str).to_numpy()
    participant_index = frame["participant_index"].to_numpy(dtype=int)
    session_index = frame["session_index"].to_numpy(dtype=int)
    post_index = frame["post_index"].to_numpy(dtype=int)
    day_index = frame["day_index"].to_numpy(dtype=int)
    z_age = frame["z_log_post_age"].to_numpy(dtype=float)
    z_cards = frame["z_log_feed_spell_cards"].to_numpy(dtype=float)
    for group in GROUP_ORDER:
        selected = np.flatnonzero(group_values == group)
        participant_levels = np.unique(participant_index[selected])
        g2 = float(group == "G2")
        for author_type in AUTHOR_ORDER:
            ai = float(author_type == "AI")
            equal = np.empty(sample_count, dtype=float)
            weighted = np.empty(sample_count, dtype=float)
            for start in range(0, sample_count, draw_chunk_size):
                stop = min(start + draw_chunk_size, sample_count)
                sl = slice(start, stop)
                eta = (
                    scalars["alpha"][sl, None]
                    + scalars["beta_group"][sl, None] * g2
                    + scalars["beta_ai"][sl, None] * ai
                    + scalars["beta_group_ai"][sl, None] * g2 * ai
                    + scalars["beta_log_age"][sl, None] * z_age[selected][None, :]
                    + scalars["beta_log_feed_spell_cards"][sl, None]
                    * z_cards[selected][None, :]
                    + participant_effect[sl][:, participant_index[selected]]
                    + participant_ai_effect[sl][:, participant_index[selected]] * ai
                    + session_effect[sl][:, session_index[selected]]
                    + post_effect[sl][:, post_index[selected]]
                    + day_effect[sl][:, day_index[selected]]
                )
                probability = expit(eta)
                weighted[sl] = probability.mean(axis=1)
                participant_means = np.column_stack(
                    [
                        probability[:, participant_index[selected] == level].mean(axis=1)
                        for level in participant_levels
                    ]
                )
                equal[sl] = participant_means.mean(axis=1)
            key = f"{group}_{author_type}"
            arrays["equal_participant_observed_opportunities"][key] = equal
            arrays["card_weighted_observed_opportunities"][key] = weighted

    rows: list[dict[str, Any]] = []
    for scope, values in arrays.items():
        for group in GROUP_ORDER:
            for author_type in AUTHOR_ORDER:
                rows.append(
                    _summarize_draws(
                        values[f"{group}_{author_type}"],
                        estimand="direct_thread_entry_probability",
                        group=group,
                        author_type=author_type,
                        standardization_scope=scope,
                    )
                )
        for group in GROUP_ORDER:
            gap = values[f"{group}_AI"] - values[f"{group}_H"]
            rows.append(
                _summarize_draws(
                    gap,
                    estimand="AI_minus_H_direct_thread_entry_probability",
                    group=group,
                    author_type="AI-H",
                    standardization_scope=scope,
                )
            )
        for author_type in AUTHOR_ORDER:
            contrast = values[f"G2_{author_type}"] - values[f"G1_{author_type}"]
            rows.append(
                _summarize_draws(
                    contrast,
                    estimand="G2_minus_G1_direct_thread_entry_probability",
                    group="G2-G1",
                    author_type=author_type,
                    standardization_scope=scope,
                )
            )
        difference = (
            values["G2_AI"]
            - values["G2_H"]
            - values["G1_AI"]
            + values["G1_H"]
        )
        rows.append(
            _summarize_draws(
                difference,
                estimand="G2_minus_G1_change_in_AI_minus_H_gap",
                group="G2-G1",
                author_type="AI-H",
                standardization_scope=scope,
            )
        )
    return pd.DataFrame(rows)


def parameter_summary(idata: Any) -> pd.DataFrame:
    """Summarize focal coefficients and hierarchical scales."""

    import arviz as az

    names = [
        "alpha",
        "beta_group",
        "beta_ai",
        "beta_group_ai",
        "beta_log_age",
        "beta_log_feed_spell_cards",
        "sigma_participant",
        "sigma_participant_ai",
        "sigma_session",
        "sigma_post",
        "sigma_day",
    ]
    return (
        az.summary(idata, var_names=names, hdi_prob=0.95, kind="all")
        .reset_index()
        .rename(columns={"index": "parameter"})
    )


def mcmc_diagnostics(idata: Any, *, role: str) -> pd.DataFrame:
    """Evaluate convergence over all sampled scalar and varying parameters."""

    import arviz as az

    variable_names = [
        name
        for name in idata.posterior.data_vars
        if name
        not in {
            "participant_effect",
            "participant_ai_effect",
            "session_effect",
            "post_effect",
            "day_effect",
        }
    ]
    summary = az.summary(
        idata,
        var_names=variable_names,
        kind="diagnostics",
        round_to="none",
    )
    max_rhat = float(summary["r_hat"].max())
    min_bulk = float(summary["ess_bulk"].min())
    min_tail = float(summary["ess_tail"].min())
    divergences = int(idata.sample_stats["diverging"].sum().values)
    passed = bool(
        np.isfinite([max_rhat, min_bulk, min_tail]).all()
        and max_rhat <= MCMC_RHAT_THRESHOLD
        and min_bulk >= MCMC_ESS_THRESHOLD
        and min_tail >= MCMC_ESS_THRESHOLD
        and divergences == 0
    )
    return pd.DataFrame(
        [
            {
                "role": role,
                "max_rhat": max_rhat,
                "min_ess_bulk": min_bulk,
                "min_ess_tail": min_tail,
                "divergences": divergences,
                "rhat_threshold": MCMC_RHAT_THRESHOLD,
                "ess_threshold": MCMC_ESS_THRESHOLD,
                "passed": passed,
            }
        ]
    )


def fitted_calibration(
    idata: Any,
    frame: pd.DataFrame,
    *,
    seed: int,
    maximum_draws: int = 500,
) -> pd.DataFrame:
    """Compare observed cell rates with fitted and replicated rate intervals."""

    from scipy.special import expit

    rng = np.random.default_rng(seed)
    scalars = {
        name: _posterior_array(idata, name)
        for name in (
            "alpha",
            "beta_group",
            "beta_ai",
            "beta_group_ai",
            "beta_log_age",
            "beta_log_feed_spell_cards",
        )
    }
    sample_count = len(scalars["alpha"])
    selected_draws = np.linspace(
        0, sample_count - 1, min(maximum_draws, sample_count), dtype=int
    )
    participant_effect = _posterior_array(idata, "participant_effect", "participant")[
        selected_draws
    ]
    participant_ai_effect = _posterior_array(
        idata, "participant_ai_effect", "participant"
    )[selected_draws]
    session_effect = _posterior_array(idata, "session_effect", "session")[selected_draws]
    post_effect = _posterior_array(idata, "post_effect", "post")[selected_draws]
    day_effect = _posterior_array(idata, "day_effect", "valid_day")[selected_draws]
    eta = (
        scalars["alpha"][selected_draws, None]
        + scalars["beta_group"][selected_draws, None] * frame["g2"].to_numpy()[None, :]
        + scalars["beta_ai"][selected_draws, None]
        * frame["ai_author"].to_numpy()[None, :]
        + scalars["beta_group_ai"][selected_draws, None]
        * frame["g2_x_ai_author"].to_numpy()[None, :]
        + scalars["beta_log_age"][selected_draws, None]
        * frame["z_log_post_age"].to_numpy()[None, :]
        + scalars["beta_log_feed_spell_cards"][selected_draws, None]
        * frame["z_log_feed_spell_cards"].to_numpy()[None, :]
        + participant_effect[:, frame["participant_index"].to_numpy(dtype=int)]
        + participant_ai_effect[:, frame["participant_index"].to_numpy(dtype=int)]
        * frame["ai_author"].to_numpy()[None, :]
        + session_effect[:, frame["session_index"].to_numpy(dtype=int)]
        + post_effect[:, frame["post_index"].to_numpy(dtype=int)]
        + day_effect[:, frame["day_index"].to_numpy(dtype=int)]
    )
    probability = expit(eta)
    replicated = rng.binomial(1, probability)
    rows: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        for author_type in AUTHOR_ORDER:
            cell = frame["group"].eq(group) & frame["author_type"].eq(author_type)
            indices = np.flatnonzero(cell.to_numpy())
            fitted_rate = probability[:, indices].mean(axis=1)
            replicated_rate = replicated[:, indices].mean(axis=1)
            observed_rate = float(frame.loc[cell, PRIMARY_OUTCOME].mean())
            low, high = np.quantile(replicated_rate, [0.025, 0.975])
            rows.append(
                {
                    "group": group,
                    "author_type": author_type,
                    "observed_rate": observed_rate,
                    "fitted_rate_mean": float(fitted_rate.mean()),
                    "fitted_rate_ci_low": float(np.quantile(fitted_rate, 0.025)),
                    "fitted_rate_ci_high": float(np.quantile(fitted_rate, 0.975)),
                    "replicated_rate_ci_low": float(low),
                    "replicated_rate_ci_high": float(high),
                    "observed_within_replicated_95_interval": bool(
                        low <= observed_rate <= high
                    ),
                    "posterior_draws_checked": int(len(selected_draws)),
                }
            )
    return pd.DataFrame(rows)


def _grouped_log_likelihood_efficiency(log_likelihood: Any, az: Any) -> float:
    values = np.asarray(log_likelihood.values, dtype=float)
    chain_count = values.shape[0]
    draw_count = values.shape[1]
    shifted = values - np.max(values, axis=1, keepdims=True)
    likelihood = np.exp(shifted)
    try:
        ess = np.asarray(az.ess(likelihood, method="mean"), dtype=float)
        finite = ess[np.isfinite(ess)]
        if finite.size:
            return float(np.clip(finite.min() / (chain_count * draw_count), 1.0e-3, 1.0))
    except (TypeError, ValueError):
        pass
    return 1.0


def grouped_psis(
    idata: Any,
    frame: pd.DataFrame,
    *,
    unit: str,
    pareto_k_threshold: float = PARETO_K_THRESHOLD,
) -> pd.DataFrame:
    """Compute grouped PSIS diagnostics by participant, session, or post."""

    import arviz as az
    from scipy.special import logsumexp

    if "log_likelihood" not in idata.groups() or "opened" not in idata.log_likelihood:
        raise ValueError("RQ1b posterior requires log_likelihood.opened")
    columns = {
        "participant": "participant_key",
        "session": "session_id",
        "post": "post_key",
    }
    if unit not in columns:
        raise ValueError("unit must be participant, session, or post")
    likelihood = idata.log_likelihood["opened"]
    dimension = "opportunity" if "opportunity" in likelihood.dims else likelihood.dims[-1]
    if likelihood.sizes[dimension] != len(frame):
        raise ValueError("Pointwise likelihood and RQ1b frame have different lengths")
    labels = frame[columns[unit]].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for label in pd.unique(labels):
        selected = np.flatnonzero(labels == label)
        grouped = likelihood.isel({dimension: selected}).sum(dim=dimension)
        reff = _grouped_log_likelihood_efficiency(grouped, az)
        values = grouped.stack(sample=("chain", "draw")).transpose("sample").values
        log_weights, pareto_k = az.psislw(-values, reff=reff)
        group = str(label).split(":", maxsplit=1)[0] if unit != "session" else str(
            frame.iloc[selected[0]]["group"]
        )
        rows.append(
            {
                "group": group,
                "influence_unit": unit,
                "unit_id": str(label),
                "opportunities": int(len(selected)),
                "pareto_k": float(pareto_k),
                "elpd_loo": float(logsumexp(values + log_weights)),
                "needs_refit": bool(pareto_k > pareto_k_threshold),
                "threshold": float(pareto_k_threshold),
                "relative_efficiency": reff,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["needs_refit", "pareto_k"], ascending=[False, False]
    ).reset_index(drop=True)


def _write_posterior(idata: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.nc")
    temporary.unlink(missing_ok=True)
    try:
        idata.to_netcdf(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def participant_aggregate_implementation_sha256() -> str:
    """Hash the participant-aggregated sensitivity specification."""

    return _implementation_hash(
        (
            build_participant_aggregate_frame,
            build_participant_aggregate_model,
            participant_aggregate_estimands,
            _posterior_array,
            _summarize_draws,
            mcmc_diagnostics,
        ),
        {
            "schema": ANALYSIS_SCHEMA,
            "primary_outcome": PRIMARY_OUTCOME,
            "mcmc_rhat_threshold": MCMC_RHAT_THRESHOLD,
            "mcmc_ess_threshold": MCMC_ESS_THRESHOLD,
        },
    )


def fit_participant_aggregate_analysis(
    config: AnalysisConfig,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    progressbar: bool = True,
) -> Mapping[str, pd.DataFrame]:
    """Fit and persist the participant-aggregated Bayesian sensitivity."""

    import arviz as az
    import pymc as pm

    panel = read_table(config.tables_dir / "rq1b_opportunity_panel")
    signature = read_table(config.tables_dir / "rq1b_panel_signature")
    if len(signature) != 1 or str(signature.iloc[0]["panel_sha256"]) != panel_signature(panel):
        raise ValueError("RQ1b panel does not match its prepared signature")
    frame = build_participant_aggregate_frame(panel)
    requested_chains = int(config.sampling.chains if chains is None else chains)
    model = build_participant_aggregate_model(frame, config)
    with model:
        idata = pm.sample(
            draws=int(config.sampling.draws if draws is None else draws),
            tune=int(config.sampling.tune if tune is None else tune),
            chains=requested_chains,
            cores=min(requested_chains, 4),
            target_accept=float(config.sampling.target_accept),
            random_seed=int(config.seed + 3000),
            progressbar=progressbar,
            return_inferencedata=True,
        )
    role = "participant_aggregate"
    posterior_path = config.models_dir / f"rq1b_{role}.nc"
    _write_posterior(idata, posterior_path)
    variable_names = [
        "alpha",
        "beta_group",
        "beta_ai",
        "beta_group_ai",
        "sigma_participant",
        "sigma_participant_ai",
    ]
    tables = {
        f"rq1b_{role}_model_frame": frame,
        f"rq1b_{role}_parameter_summary": (
            az.summary(idata, var_names=variable_names, hdi_prob=0.95, kind="all")
            .reset_index()
            .rename(columns={"index": "parameter"})
        ),
        f"rq1b_{role}_estimands": participant_aggregate_estimands(idata, frame),
        f"rq1b_{role}_mcmc_diagnostics": mcmc_diagnostics(idata, role=role),
    }
    actual_draws = int(idata.posterior.sizes.get("draw", 0))
    actual_chains = int(idata.posterior.sizes.get("chain", 0))
    tables[f"rq1b_{role}_fit_signature"] = pd.DataFrame(
        [
            {
                "schema": ANALYSIS_SCHEMA,
                "role": role,
                "panel_sha256": panel_signature(panel),
                "panel_implementation_sha256": panel_implementation_sha256(),
                "model_implementation_sha256": (
                    participant_aggregate_implementation_sha256()
                ),
                "config_sha256": config.contract_sha256,
                "posterior_sha256": sha256_file(posterior_path),
                "posterior_size_bytes": posterior_path.stat().st_size,
                "draws": actual_draws,
                "tune": int(config.sampling.tune if tune is None else tune),
                "chains": actual_chains,
                "target_accept": float(config.sampling.target_accept),
                "publication_length_request": bool(
                    actual_draws >= config.sampling.draws
                    and actual_chains >= config.sampling.chains
                ),
            }
        ]
    )
    for name, table in tables.items():
        write_table(table, config.tables_dir / name)
    return tables


def _select_design_subset(panel: pd.DataFrame, subset: str) -> pd.DataFrame:
    subsets = {
        "all": pd.Series(True, index=panel.index),
        "closed_feed_spells": panel["feed_spell_closed_by_page_view"].astype(bool),
        "both_author_feed_spells": panel[
            "feed_spell_contains_both_author_types"
        ].astype(bool),
    }
    if subset not in subsets:
        raise ValueError(f"Unknown RQ1b design subset: {subset}")
    selected = panel.loc[subsets[subset]].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"RQ1b design subset {subset} is empty")
    return selected


def fit_analysis(
    config: AnalysisConfig,
    *,
    prior_multiplier: float,
    role: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    progressbar: bool = True,
    subset: str = "all",
) -> Mapping[str, pd.DataFrame]:
    """Fit, summarize, and persist one authenticated RQ1b posterior."""

    panel = read_table(config.tables_dir / "rq1b_opportunity_panel")
    signature = read_table(config.tables_dir / "rq1b_panel_signature")
    if len(signature) != 1 or str(signature.iloc[0]["panel_sha256"]) != panel_signature(panel):
        raise ValueError("RQ1b panel does not match its prepared signature")
    if str(signature.iloc[0]["panel_implementation_sha256"]) != panel_implementation_sha256():
        raise ValueError("RQ1b panel was prepared with a different endpoint implementation")
    prepared_panel_hash = panel_signature(panel)
    panel = _select_design_subset(panel, subset)
    frame, scalers = build_model_frame(panel)
    idata = fit_model(
        frame,
        config,
        prior_multiplier=prior_multiplier,
        draws=draws,
        tune=tune,
        chains=chains,
        progressbar=progressbar,
    )
    posterior_path = config.models_dir / f"rq1b_{role}.nc"
    _write_posterior(idata, posterior_path)
    tables = {
        f"rq1b_{role}_model_frame_scalers": scalers,
        f"rq1b_{role}_parameter_summary": parameter_summary(idata),
        f"rq1b_{role}_estimands": posterior_estimands(idata, frame),
        f"rq1b_{role}_mcmc_diagnostics": mcmc_diagnostics(idata, role=role),
        f"rq1b_{role}_fitted_calibration": fitted_calibration(
            idata, frame, seed=config.seed + 50
        ),
    }
    if role == "main":
        for unit in ("participant", "session", "post"):
            tables[f"rq1b_main_influence_{unit}"] = grouped_psis(
                idata, frame, unit=unit
            )
    actual_draws = int(idata.posterior.sizes.get("draw", 0))
    actual_chains = int(idata.posterior.sizes.get("chain", 0))
    tables[f"rq1b_{role}_fit_signature"] = pd.DataFrame(
        [
            {
                "schema": ANALYSIS_SCHEMA,
                "role": role,
                "design_subset": subset,
                "prior_multiplier": float(prior_multiplier),
                "prepared_panel_sha256": prepared_panel_hash,
                "panel_sha256": panel_signature(panel),
                "panel_implementation_sha256": panel_implementation_sha256(),
                "model_implementation_sha256": model_implementation_sha256(),
                "module_sha256": implementation_sha256(),
                "config_sha256": config.contract_sha256,
                "posterior_sha256": sha256_file(posterior_path),
                "posterior_size_bytes": posterior_path.stat().st_size,
                "draws": actual_draws,
                "tune": int(config.sampling.tune if tune is None else tune),
                "chains": actual_chains,
                "target_accept": float(config.sampling.target_accept),
                "publication_length_request": bool(
                    actual_draws >= config.sampling.draws
                    and actual_chains >= config.sampling.chains
                ),
            }
        ]
    )
    for name, table in tables.items():
        write_table(table, config.tables_dir / name)
    return tables


def prior_sensitivity(main: pd.DataFrame, doubled: pd.DataFrame) -> pd.DataFrame:
    """Compare matching main and doubled-prior probability estimands."""

    keys = ["estimand", "group", "author_type", "standardization_scope"]
    required = set(
        keys
        + [
            "posterior_mean",
            "ci_low",
            "ci_high",
            "posterior_probability_gt_zero",
        ]
    )
    _require_columns(main, required, "main RQ1b estimands")
    _require_columns(doubled, required, "doubled-prior RQ1b estimands")
    left = main[list(required)].rename(
        columns={
            "posterior_mean": "main_posterior_mean",
            "ci_low": "main_ci_low",
            "ci_high": "main_ci_high",
            "posterior_probability_gt_zero": "main_probability_gt_zero",
        }
    )
    right = doubled[list(required)].rename(
        columns={
            "posterior_mean": "doubled_posterior_mean",
            "ci_low": "doubled_ci_low",
            "ci_high": "doubled_ci_high",
            "posterior_probability_gt_zero": "doubled_probability_gt_zero",
        }
    )
    out = left.merge(right, on=keys, validate="one_to_one")
    out["absolute_mean_shift"] = (
        out["doubled_posterior_mean"] - out["main_posterior_mean"]
    ).abs()
    out["absolute_probability_shift"] = (
        out["doubled_probability_gt_zero"] - out["main_probability_gt_zero"]
    ).abs()
    out["direction_stable"] = np.sign(out["main_posterior_mean"]).eq(
        np.sign(out["doubled_posterior_mean"])
    )
    out["main_interval_class"] = np.select(
        [out["main_ci_low"].gt(0), out["main_ci_high"].lt(0)],
        ["positive", "negative"],
        default="uncertain",
    )
    out["doubled_interval_class"] = np.select(
        [out["doubled_ci_low"].gt(0), out["doubled_ci_high"].lt(0)],
        ["positive", "negative"],
        default="uncertain",
    )
    out["interval_class_stable"] = out["main_interval_class"].eq(
        out["doubled_interval_class"]
    )
    out["focal_stability_passed"] = (
        out["direction_stable"]
        & out["absolute_mean_shift"].le(PRIOR_MEAN_SHIFT_TOLERANCE)
        & out["absolute_probability_shift"].le(
            PRIOR_PROBABILITY_SHIFT_TOLERANCE
        )
    )
    return out.sort_values(keys).reset_index(drop=True)


def validate_analysis(config: AnalysisConfig, config_path: Path) -> Mapping[str, pd.DataFrame]:
    """Rebuild deterministic inputs and evaluate core release gates."""

    stored_signature = read_table(config.tables_dir / "rq1b_panel_signature")
    events = _read_events(config.browser_dir)
    session_qc = read_table(config.browser_dir / "session_qc")
    reputation = read_table(config.processed_dir / "snapshot_user_reputation")
    vendor_thread_opens = pd.read_csv(
        config.project_root / VENDOR_THREAD_OPENS_RELATIVE
    )
    valid_days = analysis_set_manifest(config.analysis_set)["analysis_set_valid_days"]
    rebuilt, evidence, _ = build_direct_thread_entry_panel(
        events,
        session_qc,
        reputation,
        start_time=config.start_time,
        end_time=config.end_time,
        valid_days=valid_days,
    )
    rebuilt_hash = panel_signature(rebuilt)
    main_signature = read_table(config.tables_dir / "rq1b_main_fit_signature")
    main_diagnostics = read_table(config.tables_dir / "rq1b_main_mcmc_diagnostics")
    main_estimands = read_table(config.tables_dir / "rq1b_main_estimands")
    doubled_signature = read_table(config.tables_dir / "rq1b_doubled_fit_signature")
    doubled_diagnostics = read_table(config.tables_dir / "rq1b_doubled_mcmc_diagnostics")
    doubled_estimands = read_table(config.tables_dir / "rq1b_doubled_estimands")
    closed_signature = read_table(config.tables_dir / "rq1b_closed_fit_signature")
    closed_diagnostics = read_table(config.tables_dir / "rq1b_closed_mcmc_diagnostics")
    closed_estimands = read_table(config.tables_dir / "rq1b_closed_estimands")
    both_signature = read_table(config.tables_dir / "rq1b_both_author_fit_signature")
    both_diagnostics = read_table(
        config.tables_dir / "rq1b_both_author_mcmc_diagnostics"
    )
    both_estimands = read_table(config.tables_dir / "rq1b_both_author_estimands")
    aggregate_signature = read_table(
        config.tables_dir / "rq1b_participant_aggregate_fit_signature"
    )
    aggregate_diagnostics = read_table(
        config.tables_dir / "rq1b_participant_aggregate_mcmc_diagnostics"
    )
    aggregate_estimands = read_table(
        config.tables_dir / "rq1b_participant_aggregate_estimands"
    )
    sensitivity = prior_sensitivity(main_estimands, doubled_estimands)
    write_table(sensitivity, config.tables_dir / "rq1b_prior_sensitivity")
    integrity = integrity_checks(rebuilt, evidence, valid_days=valid_days)
    write_table(integrity, config.tables_dir / "rq1b_integrity_checks_rebuilt")
    source_audit = endpoint_source_audit(rebuilt, vendor_thread_opens)
    write_table(source_audit, config.tables_dir / "rq1b_endpoint_source_audit_rebuilt")

    def one_true(frame: pd.DataFrame, column: str) -> bool:
        return bool(len(frame) == 1 and frame[column].fillna(False).astype(bool).all())

    def authenticated_fit(
        signature: pd.DataFrame,
        role: str,
        expected_panel: pd.DataFrame,
        expected_model_hash: str | None = None,
    ) -> tuple[bool, str]:
        posterior_path = config.models_dir / f"rq1b_{role}.nc"
        required = {
            "panel_sha256",
            "panel_implementation_sha256",
            "model_implementation_sha256",
            "config_sha256",
            "posterior_sha256",
        }
        if len(signature) != 1 or not required.issubset(signature.columns):
            return False, "missing or incomplete fit signature"
        row = signature.iloc[0]
        posterior_hash = sha256_file(posterior_path) if posterior_path.exists() else None
        checks = {
            "panel": str(row["panel_sha256"]) == panel_signature(expected_panel),
            "panel_code": str(row["panel_implementation_sha256"])
            == panel_implementation_sha256(),
            "model_code": str(row["model_implementation_sha256"])
            == (expected_model_hash or model_implementation_sha256()),
            "config": str(row["config_sha256"]) == config.contract_sha256,
            "posterior": str(row["posterior_sha256"]) == str(posterior_hash),
        }
        return bool(all(checks.values())), json.dumps(checks, sort_keys=True)

    main_authenticated, main_auth_observed = authenticated_fit(
        main_signature, "main", rebuilt
    )
    doubled_authenticated, doubled_auth_observed = authenticated_fit(
        doubled_signature, "doubled", rebuilt
    )
    closed_panel = _select_design_subset(rebuilt, "closed_feed_spells")
    both_panel = _select_design_subset(rebuilt, "both_author_feed_spells")
    closed_authenticated, closed_auth_observed = authenticated_fit(
        closed_signature, "closed", closed_panel
    )
    both_authenticated, both_auth_observed = authenticated_fit(
        both_signature, "both_author", both_panel
    )
    aggregate_authenticated, aggregate_auth_observed = authenticated_fit(
        aggregate_signature,
        "participant_aggregate",
        rebuilt,
        participant_aggregate_implementation_sha256(),
    )

    focal = sensitivity[
        sensitivity["estimand"].eq("G2_minus_G1_change_in_AI_minus_H_gap")
        & sensitivity["standardization_scope"].eq(
            "equal_participant_observed_opportunities"
        )
    ]
    design_rows: list[dict[str, Any]] = []
    for role, subset, estimands, signature in (
        ("main", "all_feed_spells", main_estimands, main_signature),
        ("closed", "closed_feed_spells", closed_estimands, closed_signature),
        (
            "both_author",
            "both_author_feed_spells",
            both_estimands,
            both_signature,
        ),
        (
            "participant_aggregate",
            "participant_by_author_type_counts",
            aggregate_estimands,
            aggregate_signature,
        ),
    ):
        expected_scope = (
            "participant_aggregated_equal_participant"
            if role == "participant_aggregate"
            else "equal_participant_observed_opportunities"
        )
        selected = estimands[
            estimands["estimand"].eq("G2_minus_G1_change_in_AI_minus_H_gap")
            & estimands["standardization_scope"].eq(expected_scope)
        ]
        if len(selected) != 1:
            raise ValueError(f"RQ1b {role} fit lacks its focal estimand")
        row = selected.iloc[0]
        design_rows.append(
            {
                "role": role,
                "design_subset": subset,
                "panel_sha256": str(signature.iloc[0]["panel_sha256"]),
                "posterior_mean": float(row["posterior_mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "posterior_probability_gt_zero": float(
                    row["posterior_probability_gt_zero"]
                ),
            }
        )
    design_sensitivity = pd.DataFrame(design_rows)
    write_table(
        design_sensitivity, config.tables_dir / "rq1b_design_sensitivity_estimands"
    )
    influence_observed: list[str] = []
    influence_complete = True
    for unit in ("participant", "session", "post"):
        table = read_table(config.tables_dir / f"rq1b_main_influence_{unit}")
        finite = bool(not table.empty and np.isfinite(table["pareto_k"]).all())
        influence_complete &= finite
        influence_observed.append(
            f"{unit}={int(table['needs_refit'].fillna(True).sum())}/{len(table)} flagged"
        )
    gates = [
        {
            "gate": "deterministic_panel_reproduction",
            "passed": bool(
                len(stored_signature) == 1
                and rebuilt_hash == str(stored_signature.iloc[0]["panel_sha256"])
                and str(stored_signature.iloc[0]["panel_implementation_sha256"])
                == panel_implementation_sha256()
            ),
            "observed": rebuilt_hash,
        },
        {
            "gate": "endpoint_integrity",
            "passed": bool(integrity["passed"].all()),
            "observed": f"{int(integrity['passed'].sum())}/{len(integrity)} checks passed",
        },
        {
            "gate": "endpoint_vendor_source_census",
            "passed": bool(source_audit["passed"].all()),
            "observed": json.dumps(
                source_audit.to_dict("records"), sort_keys=True, default=str
            ),
        },
        {
            "gate": "main_authenticated_posterior",
            "passed": main_authenticated,
            "observed": main_auth_observed,
        },
        {
            "gate": "main_publication_length_posterior",
            "passed": one_true(main_signature, "publication_length_request"),
            "observed": f"rows={len(main_signature)}",
        },
        {
            "gate": "main_mcmc",
            "passed": one_true(main_diagnostics, "passed"),
            "observed": json.dumps(
                main_diagnostics.to_dict("records"), sort_keys=True, default=str
            ),
        },
        {
            "gate": "doubled_prior_authenticated_posterior",
            "passed": doubled_authenticated,
            "observed": doubled_auth_observed,
        },
        {
            "gate": "doubled_prior_publication_length_posterior",
            "passed": one_true(doubled_signature, "publication_length_request"),
            "observed": f"rows={len(doubled_signature)}",
        },
        {
            "gate": "doubled_prior_mcmc",
            "passed": one_true(doubled_diagnostics, "passed"),
            "observed": json.dumps(
                doubled_diagnostics.to_dict("records"), sort_keys=True, default=str
            ),
        },
        {
            "gate": "closed_feed_spell_authenticated_posterior",
            "passed": closed_authenticated,
            "observed": closed_auth_observed,
        },
        {
            "gate": "closed_feed_spell_mcmc",
            "passed": one_true(closed_diagnostics, "passed"),
            "observed": json.dumps(
                closed_diagnostics.to_dict("records"), sort_keys=True, default=str
            ),
        },
        {
            "gate": "both_author_feed_spell_authenticated_posterior",
            "passed": both_authenticated,
            "observed": both_auth_observed,
        },
        {
            "gate": "both_author_feed_spell_mcmc",
            "passed": one_true(both_diagnostics, "passed"),
            "observed": json.dumps(
                both_diagnostics.to_dict("records"), sort_keys=True, default=str
            ),
        },
        {
            "gate": "participant_aggregate_authenticated_posterior",
            "passed": aggregate_authenticated,
            "observed": aggregate_auth_observed,
        },
        {
            "gate": "participant_aggregate_mcmc",
            "passed": one_true(aggregate_diagnostics, "passed"),
            "observed": json.dumps(
                aggregate_diagnostics.to_dict("records"),
                sort_keys=True,
                default=str,
            ),
        },
        {
            "gate": "primary_estimand_complete",
            "passed": bool(
                len(focal) == 1
                and np.isfinite(
                    focal[
                        [
                            "main_posterior_mean",
                            "main_ci_low",
                            "main_ci_high",
                            "doubled_posterior_mean",
                            "doubled_ci_low",
                            "doubled_ci_high",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            ),
            "observed": f"focal_rows={len(focal)}",
        },
        {
            "gate": "prior_sensitivity_focal_stability",
            "passed": bool(len(focal) == 1 and focal["focal_stability_passed"].all()),
            "observed": json.dumps(
                focal[
                    [
                        "main_interval_class",
                        "doubled_interval_class",
                        "absolute_mean_shift",
                        "absolute_probability_shift",
                        "direction_stable",
                        "focal_stability_passed",
                    ]
                ].to_dict("records"),
                sort_keys=True,
                default=str,
            ),
        },
        {
            "gate": "grouped_influence_computed",
            "passed": influence_complete,
            "observed": "; ".join(influence_observed),
        },
    ]
    readiness = pd.DataFrame(gates)
    write_table(readiness, config.tables_dir / "rq1b_core_readiness")
    return {
        "rq1b_integrity_checks_rebuilt": integrity,
        "rq1b_endpoint_source_audit_rebuilt": source_audit,
        "rq1b_design_sensitivity_estimands": design_sensitivity,
        "rq1b_prior_sensitivity": sensitivity,
        "rq1b_core_readiness": readiness,
    }


def write_report(config: AnalysisConfig) -> Path:
    """Write a compact, artifact-grounded report for the RQ1b core fit."""

    counts = read_table(config.tables_dir / "rq1b_cell_summary")
    coverage = read_table(config.tables_dir / "rq1b_coverage")
    design_sensitivities = read_table(
        config.tables_dir / "rq1b_observed_design_sensitivities"
    )
    source_audit = read_table(config.tables_dir / "rq1b_endpoint_source_audit")
    deletion = read_table(
        config.tables_dir / "rq1b_participant_deletion_descriptive"
    )
    estimands = read_table(config.tables_dir / "rq1b_main_estimands")
    bayesian_design_sensitivities = read_table(
        config.tables_dir / "rq1b_design_sensitivity_estimands"
    )
    diagnostics = read_table(config.tables_dir / "rq1b_main_mcmc_diagnostics")
    readiness = read_table(config.tables_dir / "rq1b_core_readiness")
    primary = estimands[
        estimands["standardization_scope"].eq(
            "equal_participant_observed_opportunities"
        )
    ]
    influence = []
    for unit in ("participant", "session", "post"):
        frame = read_table(config.tables_dir / f"rq1b_main_influence_{unit}")
        influence.append(
            {
                "unit": unit,
                "units": len(frame),
                "flagged": int(frame["needs_refit"].fillna(True).sum()),
                "maximum_pareto_k": float(frame["pareto_k"].max()),
            }
        )
    text = f"""# RQ1b direct thread-entry core report

- Run ID: `{config.run_id}`
- Analysis set: `{config.analysis_set}`
- Core readiness: `{'passed' if readiness['passed'].all() else 'failed'}`
- Inference boundary: the between-condition comparison is ecological because each condition contains one community
- Outcome boundary: direct route entry after a DOM-recorded feed-card render; no claim of pixel visibility, reading, attention, or reply conversion

## Observed feed-spell endpoint

{counts.to_markdown(index=False)}

## Coverage

{coverage.to_markdown(index=False)}

## Endpoint source audit

{source_audit.to_markdown(index=False)}

## Equal-participant posterior estimands

{primary[['estimand', 'group', 'author_type', 'posterior_mean', 'ci_low', 'ci_high', 'posterior_probability_gt_zero']].to_markdown(index=False)}

## Unadjusted design sensitivities

{design_sensitivities[['analysis_variant', 'opportunities', 'direct_opens', 'difference_in_differences']].to_markdown(index=False)}

## Bayesian design sensitivities

{bayesian_design_sensitivities[['role', 'design_subset', 'posterior_mean', 'ci_low', 'ci_high', 'posterior_probability_gt_zero']].to_markdown(index=False)}

## MCMC diagnostics

{diagnostics.to_markdown(index=False)}

## Grouped influence screen

{pd.DataFrame(influence).to_markdown(index=False)}

The unadjusted difference-in-differences remained positive in all {len(deletion)} single-participant deletions. Its range was {deletion['difference_in_differences'].min():.4f} to {deletion['difference_in_differences'].max():.4f}.

## Core gates

{readiness[['gate', 'passed', 'observed']].to_markdown(index=False)}

## Remaining admission work

The inferential analysis is complete. Grouped PSIS flags remain because the crossed model contains unit-specific varying effects, so the PSIS output cannot support a grouped predictive model-comparison claim without exact deletion refits. No predictive-superiority claim is made here. Exact deletion refits and forward evaluation are needed only if grouped predictive validation is added.
"""
    path = config.reports_dir / "rq1b_core_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "fit",
            "sensitivity",
            "fit-closed",
            "fit-both-author",
            "fit-participant-aggregate",
            "validate",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("analysis.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("pilot", "release"), default="release")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--local-temp", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, profile=args.profile, run_id=args.run_id)
    if args.stage == "prepare":
        tables = prepare_analysis(config, config_path)
        signature = tables["rq1b_panel_signature"].iloc[0]
        passed = bool(tables["rq1b_integrity_checks"]["passed"].all())
        print(
            json.dumps(
                {
                    "run_id": config.run_id,
                    "rows": int(signature["rows"]),
                    "direct_opens": int(signature["direct_opens"]),
                    "panel_sha256": str(signature["panel_sha256"]),
                    "integrity_passed": passed,
                },
                sort_keys=True,
            )
        )
        return 0 if passed else 2
    if args.stage == "fit-participant-aggregate":
        configure_runtime(args.local_temp)
        tables = fit_participant_aggregate_analysis(
            config,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            progressbar=not args.no_progress,
        )
        role = "participant_aggregate"
        diagnostics = tables[f"rq1b_{role}_mcmc_diagnostics"].iloc[0]
        focal = tables[f"rq1b_{role}_estimands"]
        focal = focal[
            focal["estimand"].eq("G2_minus_G1_change_in_AI_minus_H_gap")
        ].iloc[0]
        print(
            json.dumps(
                {
                    "role": role,
                    "mcmc_passed": bool(diagnostics["passed"]),
                    "difference_in_differences": float(focal["posterior_mean"]),
                    "ci_low": float(focal["ci_low"]),
                    "ci_high": float(focal["ci_high"]),
                    "posterior_probability_gt_zero": float(
                        focal["posterior_probability_gt_zero"]
                    ),
                },
                sort_keys=True,
            )
        )
        return 0 if bool(diagnostics["passed"]) else 2
    if args.stage in {"fit", "sensitivity", "fit-closed", "fit-both-author"}:
        configure_runtime(args.local_temp)
        role_by_stage = {
            "fit": "main",
            "sensitivity": "doubled",
            "fit-closed": "closed",
            "fit-both-author": "both_author",
        }
        subset_by_stage = {
            "fit": "all",
            "sensitivity": "all",
            "fit-closed": "closed_feed_spells",
            "fit-both-author": "both_author_feed_spells",
        }
        role = role_by_stage[args.stage]
        multiplier = 2.0 if role == "doubled" else 1.0
        tables = fit_analysis(
            config,
            prior_multiplier=multiplier,
            role=role,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            progressbar=not args.no_progress,
            subset=subset_by_stage[args.stage],
        )
        diagnostics = tables[f"rq1b_{role}_mcmc_diagnostics"].iloc[0]
        focal = tables[f"rq1b_{role}_estimands"]
        focal = focal[
            focal["estimand"].eq("G2_minus_G1_change_in_AI_minus_H_gap")
            & focal["standardization_scope"].eq(
                "equal_participant_observed_opportunities"
            )
        ].iloc[0]
        print(
            json.dumps(
                {
                    "role": role,
                    "mcmc_passed": bool(diagnostics["passed"]),
                    "difference_in_differences": float(focal["posterior_mean"]),
                    "ci_low": float(focal["ci_low"]),
                    "ci_high": float(focal["ci_high"]),
                    "posterior_probability_gt_zero": float(
                        focal["posterior_probability_gt_zero"]
                    ),
                },
                sort_keys=True,
            )
        )
        return 0 if bool(diagnostics["passed"]) else 2
    if args.stage == "validate":
        tables = validate_analysis(config, config_path)
        readiness = tables["rq1b_core_readiness"]
        passed = bool(readiness["passed"].all())
        print(json.dumps({"core_readiness_passed": passed}, sort_keys=True))
        return 0 if passed else 2
    if args.stage == "report":
        path = write_report(config)
        print(str(path))
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "PRIMARY_OUTCOME",
    "build_direct_thread_entry_panel",
    "build_model_frame",
    "build_participant_aggregate_frame",
    "build_participant_aggregate_model",
    "build_pymc_model",
    "cell_summary",
    "configure_runtime",
    "coverage_summary",
    "endpoint_source_audit",
    "fit_analysis",
    "fit_model",
    "fit_participant_aggregate_analysis",
    "fitted_calibration",
    "grouped_psis",
    "implementation_sha256",
    "integrity_checks",
    "main",
    "mcmc_diagnostics",
    "model_implementation_sha256",
    "observed_design_sensitivities",
    "panel_implementation_sha256",
    "participant_aggregate_estimands",
    "participant_aggregate_implementation_sha256",
    "participant_deletion_descriptive",
    "parameter_summary",
    "panel_signature",
    "posterior_estimands",
    "prepare_analysis",
    "prior_sensitivity",
    "validate_analysis",
    "write_report",
]
