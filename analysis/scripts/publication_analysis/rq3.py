"""RQ3 sanctioning: human downvoting of selected reply targets within fixed windows.

Reactions carry a platform ``round`` but no timestamp.  Every round is one
wall-clock hour, so a reaction is known only to within its hour.  This module
derives the round-to-hour map from comment events (which carry both ``round``
and an exact ``created_at``), stamps reactions at the round start, builds a
panel of human replies with their selected same-thread target, and counts
human downvotes on that target within 24 h and 48 h.  All functions are pure;
I/O lives in :mod:`scripts.publication_analysis.pipeline`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .study import (
    EXPERIMENT_TZ,
    _day_bool,
    analysis_set_config,
    experiment_day_series,
    two_prop_pvalue,
    wilson_ci,
)

COMMENT_EVENT = "comments_replies"
REACTION_EVENTS = ("upvote", "downvote")
ROUND_HOURS = 1.0
PLACEMENTS = ("round_start", "round_end", "same_round_inclusive")
SUBSETS = ("rq1_analysis_set", "all_valid_targets", "fully_observed_window")
PRIMARY_SUBSET = "rq1_analysis_set"
PRIMARY_PLACEMENT = "round_start"
PRIMARY_DYAD = "H->AI"
ESTIMAND_NAME = "rq3_human_downvote_of_selected_ai_target_within_24h"


def _id_text(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return series.map(convert)


def derive_round_timestamps(
    events: pd.DataFrame,
    *,
    round_start_time: str,
    first_experiment_round: int,
    verify_from: str | None = None,
) -> pd.DataFrame:
    """Map platform rounds to hour boundaries and verify the map is exact.

    Comment events carry ``round`` and an exact timestamp.  Each observed round
    must fall inside one clock hour, and that hour must equal the closed form
    ``round_start_time + (round - first_experiment_round) hours``.  Rounds that
    contain only reactions are filled from the closed form.  Any violation
    raises, because a wrong map would silently move reactions across windows.
    Rounds whose comments all precede ``verify_from`` (the human start) are
    recorded but not enforced; every event in them is excluded downstream.
    """

    required = {"event_type", "event_timestamp", "round"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events are missing columns required for round mapping: {missing}")
    base = pd.Timestamp(round_start_time)
    if base.tzinfo is None:
        base = base.tz_localize(EXPERIMENT_TZ)
    base = base.tz_convert(EXPERIMENT_TZ)
    hour = pd.Timedelta(hours=ROUND_HOURS)
    verify_start = None
    if verify_from is not None:
        verify_start = pd.Timestamp(verify_from)
        verify_start = verify_start.tz_localize(EXPERIMENT_TZ) if verify_start.tzinfo is None else verify_start.tz_convert(EXPERIMENT_TZ)

    comments = events[
        events["event_type"].eq(COMMENT_EVENT)
        & events["event_timestamp"].notna()
        & events["round"].notna()
    ]
    observed = (
        comments.groupby(comments["round"].astype(int))["event_timestamp"]
        .agg(["min", "max", "size"])
        .rename(columns={"size": "n_comments"})
    )
    rows: list[dict[str, Any]] = []
    for round_id, part in observed.iterrows():
        floor = pd.Timestamp(part["min"]).tz_convert(EXPERIMENT_TZ).floor("h")
        closed = base + (int(round_id) - int(first_experiment_round)) * hour
        residual = (pd.Timestamp(part["max"]).tz_convert(EXPERIMENT_TZ) - floor).total_seconds()
        latest = pd.Timestamp(part["max"]).tz_convert(EXPERIMENT_TZ)
        enforced = verify_start is None or latest >= verify_start
        rows.append(
            {
                "round": int(round_id),
                "round_start": closed if not enforced else floor,
                "closed_form_start": closed,
                "source": "observed_comments" if enforced else "observed_prestart_unverified",
                "n_comments": int(part["n_comments"]),
                "residual_max_seconds": float(residual),
                "within_one_hour": bool(0.0 <= residual < hour.total_seconds()),
                "closed_form_match": bool(floor == closed),
                "enforced": bool(enforced),
            }
        )
    observed_rounds = {row["round"] for row in rows}
    reaction_rounds = (
        events.loc[events["event_type"].isin(REACTION_EVENTS) & events["round"].notna(), "round"]
        .astype(int)
        .unique()
    )
    for round_id in sorted(set(int(value) for value in reaction_rounds) - observed_rounds):
        closed = base + (round_id - int(first_experiment_round)) * hour
        rows.append(
            {
                "round": round_id,
                "round_start": closed,
                "closed_form_start": closed,
                "source": "closed_form",
                "n_comments": 0,
                "residual_max_seconds": np.nan,
                "within_one_hour": True,
                "closed_form_match": True,
                "enforced": True,
            }
        )
    round_map = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    if round_map.empty:
        raise ValueError("No rounds could be mapped: no timestamped comment events found")
    if not round_map["enforced"].any():
        raise ValueError("No rounds could be verified after the human start")
    bad = round_map[round_map["enforced"] & ~(round_map["within_one_hour"] & round_map["closed_form_match"])]
    if not bad.empty:
        preview = ", ".join(str(value) for value in bad["round"].head(5))
        raise ValueError(
            "Round-to-hour map is not exact; refusing to timestamp reactions. "
            f"Offending rounds: {preview}"
        )
    round_map["resolution_hours"] = ROUND_HOURS
    return round_map


def timestamp_reactions(
    events: pd.DataFrame,
    round_map: pd.DataFrame,
    *,
    placement: str = PRIMARY_PLACEMENT,
) -> pd.DataFrame:
    """Assign reaction timestamps from rounds without touching comment rows."""

    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown reaction placement {placement!r}")
    out = events.copy()
    starts = pd.Series(round_map["round_start"].values, index=round_map["round"].astype(int).values)
    offset = pd.Timedelta(hours=ROUND_HOURS) if placement == "round_end" else pd.Timedelta(0)
    is_reaction = out["event_type"].isin(REACTION_EVENTS)
    rounds = out.loc[is_reaction, "round"]
    if rounds.isna().any():
        raise ValueError("Reaction rows without a round cannot be timestamped")
    mapped = rounds.astype(int).map(starts)
    if mapped.isna().any():
        missing = sorted(rounds[mapped.isna()].astype(int).unique())[:5]
        raise ValueError(f"Reaction rounds absent from the round map: {missing}")
    stamped = pd.to_datetime(pd.Series(list(mapped), index=mapped.index), utc=True).dt.tz_convert(EXPERIMENT_TZ)
    out.loc[is_reaction, "event_timestamp"] = stamped + offset
    out["event_timestamp"] = pd.to_datetime(out["event_timestamp"], utc=True).dt.tz_convert(EXPERIMENT_TZ)
    if "event_timestamp_missing" in out.columns:
        out.loc[is_reaction, "event_timestamp_missing"] = False
    out["event_timestamp_resolution_hours"] = np.where(is_reaction, ROUND_HOURS, 0.0)
    out["event_timestamp_source"] = np.where(is_reaction, placement, "platform_created_at")
    return out


def reaction_coverage(raw_events: pd.DataFrame, stamped_events: pd.DataFrame) -> pd.DataFrame:
    """Count reaction rows before and after stamping, by arm, type, and actor type."""

    rows: list[dict[str, Any]] = []
    raw = raw_events[raw_events["event_type"].isin(REACTION_EVENTS)]
    stamped = stamped_events[stamped_events["event_type"].isin(REACTION_EVENTS)]
    keys = ["group", "event_type", "actor_type"]
    raw_counts = raw.groupby(keys).size()
    raw_timestamped = raw.groupby(keys)["event_timestamp"].apply(lambda s: int(s.notna().sum()))
    stamped_counts = stamped.groupby(keys)["event_timestamp"].apply(lambda s: int(s.notna().sum()))
    for key in sorted(raw_counts.index):
        total = int(raw_counts.get(key, 0))
        after = int(stamped_counts.get(key, 0))
        rows.append(
            {
                "group": key[0],
                "event_type": key[1],
                "actor_type": key[2],
                "total_rows": total,
                "timestamped_before": int(raw_timestamped.get(key, 0)),
                "timestamped_after": after,
                "coverage_after": after / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_sanction_panel(
    choice_sets: pd.DataFrame,
    event_history: pd.DataFrame,
) -> pd.DataFrame:
    """One row per human reply with a selected same-thread target.

    Rows from ``choice_sets`` (chosen candidates of RQ1 choice sets) carry
    ``in_rq1_analysis_set = True``.  Human replies in ``event_history`` whose
    target is valid but which lacked a second candidate are appended with
    ``in_rq1_analysis_set = False``.  Self-targeted replies are excluded.
    """

    replies = event_history[
        event_history["event_type"].eq(COMMENT_EVENT)
        & event_history["actor_type"].astype(str).str.upper().eq("H")
    ].copy()
    replies["_event_key"] = replies["group"].astype(str) + ":" + replies["event_id"].astype(str)
    reply_ids = replies.set_index("_event_key")["text_id"] if "text_id" in replies.columns else None

    if choice_sets.empty:
        chosen = pd.DataFrame(columns=["group", "event_id"])
    else:
        chosen = choice_sets[choice_sets["chosen"].astype(bool)].copy()
    chosen_keys = set(chosen["group"].astype(str) + ":" + chosen["event_id"].astype(str)) if not chosen.empty else set()

    rows: list[dict[str, Any]] = []
    for reply in replies.itertuples(index=False):
        key = f"{reply.group}:{reply.event_id}"
        if _id_text(pd.Series([reply.target_author_id])).iloc[0] == _id_text(pd.Series([reply.actor_id])).iloc[0]:
            continue
        target_type = str(reply.target_author_type).upper()
        rows.append(
            {
                "group": str(reply.group),
                "event_id": reply.event_id,
                "actor_id": reply.actor_id,
                "event_timestamp": reply.event_timestamp,
                "experiment_day": reply.experiment_day,
                "selected_target_content_id": reply.target_content_id,
                "target_author_id": reply.target_author_id,
                "target_author_type": target_type,
                "dyad": f"H->{'AI' if target_type == 'AI' else 'H'}",
                "reply_content_id": reply_ids.get(key) if reply_ids is not None else None,
                "in_rq1_analysis_set": key in chosen_keys,
            }
        )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel["event_timestamp"] = pd.to_datetime(panel["event_timestamp"], utc=True).dt.tz_convert(EXPERIMENT_TZ)
    return panel.sort_values(["group", "event_timestamp", "event_id"], kind="mergesort").reset_index(drop=True)


def _window_counts(
    panel: pd.DataFrame,
    reactions: pd.DataFrame,
    *,
    content_column: str,
    hours: int,
    include_same_round: bool = False,
    voter: str = "any",
) -> tuple[pd.Series, pd.Series]:
    """Return (any_downvote, upvotes - downvotes) per panel row within (t, t+h].

    ``voter`` restricts reactions to ``"replier"`` (the replying participant),
    ``"other"`` (any other participant) or ``"any"``.
    """

    if panel.empty or reactions.empty:
        zeros = pd.Series(0, index=panel.index)
        return zeros.astype(bool), zeros.astype(int)
    left = panel[["group", content_column, "event_timestamp", "actor_id"]].copy()
    left["_row"] = panel.index
    left["_content"] = _id_text(left[content_column])
    left["_replier"] = _id_text(left["actor_id"])
    right = reactions[["group", "target_content_id", "event_timestamp", "event_type", "actor_id"]].copy()
    right["_content"] = _id_text(right["target_content_id"])
    right["_voter"] = _id_text(right["actor_id"])
    right = right.rename(columns={"event_timestamp": "reaction_time"})
    merged = left.merge(
        right[["group", "_content", "reaction_time", "event_type", "_voter"]], on=["group", "_content"], how="inner"
    )
    if voter == "replier":
        merged = merged[merged["_voter"].eq(merged["_replier"])]
    elif voter == "other":
        merged = merged[merged["_voter"].ne(merged["_replier"])]
    elif voter != "any":
        raise ValueError(f"Unknown voter filter {voter!r}")
    end = merged["event_timestamp"] + pd.Timedelta(hours=hours)
    if include_same_round:
        # Reactions are stamped at their round start; a reaction in the reply's own
        # round cannot be ordered against the reply and is counted here.
        lower = merged["event_timestamp"].dt.floor("h")
        inside = merged["reaction_time"].ge(lower) & merged["reaction_time"].le(end)
    else:
        inside = merged["reaction_time"].gt(merged["event_timestamp"]) & merged["reaction_time"].le(end)
    merged = merged[inside]
    downs = merged[merged["event_type"].eq("downvote")].groupby("_row").size()
    ups = merged[merged["event_type"].eq("upvote")].groupby("_row").size()
    any_down = pd.Series(False, index=panel.index)
    any_down.loc[downs.index] = True
    delta = pd.Series(0, index=panel.index, dtype=int)
    delta = delta.add(ups.reindex(panel.index, fill_value=0), fill_value=0).sub(
        downs.reindex(panel.index, fill_value=0), fill_value=0
    )
    return any_down.astype(bool), delta.astype(int)


def add_vote_windows(
    panel: pd.DataFrame,
    reactions: pd.DataFrame,
    *,
    horizons: Sequence[int],
    observation_end: Mapping[str, pd.Timestamp],
    valid_days: Iterable[int],
    placement: str,
) -> pd.DataFrame:
    """Attach windowed human-vote outcomes to the panel.

    ``reactions`` must already be restricted to human actors and to the valid
    analysis window.  ``observation_end`` gives the last observed event time per
    arm so right-censored windows can be flagged; ``valid_days`` flags windows
    that cross into an excluded day.
    """

    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown reaction placement {placement!r}")
    if not reactions.empty and not reactions["actor_type"].astype(str).str.upper().eq("H").all():
        raise ValueError("Vote windows require human-actor reactions only")
    out = panel.copy()
    out["reaction_placement"] = placement
    same_round = placement == "same_round_inclusive"
    valid = list(valid_days)
    for hours in horizons:
        any_target, delta_target = _window_counts(
            out, reactions, content_column="selected_target_content_id", hours=int(hours), include_same_round=same_round
        )
        out[f"target_received_human_downvote_{hours}h"] = any_target.values
        out[f"target_net_human_vote_delta_{hours}h"] = delta_target.values
        if "reply_content_id" in out.columns and out["reply_content_id"].notna().any():
            any_reply, _ = _window_counts(
                out, reactions, content_column="reply_content_id", hours=int(hours), include_same_round=same_round
            )
            out[f"reply_received_human_downvote_{hours}h"] = any_reply.values
        else:
            out[f"reply_received_human_downvote_{hours}h"] = False
        # Voter decomposition, always counting the reply's own round: a replier's
        # downvote in that hour belongs to the reply act; another participant's
        # downvote is a community reaction.
        for who in ("replier", "other"):
            any_who, _ = _window_counts(
                out,
                reactions,
                content_column="selected_target_content_id",
                hours=int(hours),
                include_same_round=True,
                voter=who,
            )
            out[f"target_{who}_downvote_{hours}h"] = any_who.values
        window_end = out["event_timestamp"] + pd.Timedelta(hours=int(hours))
        end_day = experiment_day_series(window_end)
        ends = out["group"].map(lambda g: observation_end.get(str(g), pd.NaT))
        ends = pd.to_datetime(ends, utc=True)
        before_end = window_end.dt.tz_convert("UTC").le(ends)
        end_day_valid = _day_bool(end_day, valid)
        out[f"window_{hours}h_fully_observed"] = (before_end & end_day_valid).fillna(False).astype(bool)
    return out


def _subset_mask(panel: pd.DataFrame, subset: str, hours: int) -> pd.Series:
    if subset == "rq1_analysis_set":
        return panel["in_rq1_analysis_set"].astype(bool)
    if subset == "all_valid_targets":
        return pd.Series(True, index=panel.index)
    if subset == "fully_observed_window":
        return panel["in_rq1_analysis_set"].astype(bool) & panel[f"window_{hours}h_fully_observed"].astype(bool)
    raise ValueError(f"Unknown subset {subset!r}")


def _metric_hours(metric: str) -> int:
    return int(metric.rsplit("_", 1)[-1].rstrip("h"))


def summarize_groups(
    panel: pd.DataFrame,
    *,
    metrics: Sequence[str],
    subsets: Sequence[str] = SUBSETS,
) -> pd.DataFrame:
    """Proportion (or mean) per subset x arm x dyad x metric with Wilson intervals."""

    rows: list[dict[str, Any]] = []
    placement = str(panel["reaction_placement"].iloc[0]) if not panel.empty else PRIMARY_PLACEMENT
    for subset in subsets:
        for metric in metrics:
            hours = _metric_hours(metric)
            selected = panel[_subset_mask(panel, subset, hours)] if not panel.empty else panel
            for (group, dyad), part in selected.groupby(["group", "dyad"], sort=True):
                values = part[metric]
                n = int(len(values))
                binary = values.dtype == bool
                if binary:
                    k = int(values.sum())
                    low, high = wilson_ci(k, n)
                    estimate = k / n if n else np.nan
                    numerator: float | int = k
                else:
                    estimate = float(values.mean()) if n else np.nan
                    low, high = (np.nan, np.nan)
                    numerator = float(values.sum())
                rows.append(
                    {
                        "subset": subset,
                        "reaction_placement": placement,
                        "group": group,
                        "dyad": dyad,
                        "metric": metric,
                        "measure_type": "proportion" if binary else "mean",
                        "estimate": estimate,
                        "numerator": numerator,
                        "denominator_n": n,
                        "actors_n": int(part["actor_id"].nunique()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def _cluster_bootstrap_difference(
    g1: pd.DataFrame,
    g2: pd.DataFrame,
    metric: str,
    *,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Pooled G2 - G1 difference with participants resampled as blocks."""

    blocks1 = [part[metric].to_numpy(dtype=float) for _, part in g1.groupby("actor_id", sort=True)]
    blocks2 = [part[metric].to_numpy(dtype=float) for _, part in g2.groupby("actor_id", sort=True)]
    if not blocks1 or not blocks2:
        return (np.nan, np.nan, np.nan)
    estimate = float(g2[metric].astype(float).mean() - g1[metric].astype(float).mean())
    draws = np.empty(int(reps))
    for index in range(int(reps)):
        sample1 = np.concatenate([blocks1[i] for i in rng.integers(0, len(blocks1), len(blocks1))])
        sample2 = np.concatenate([blocks2[i] for i in rng.integers(0, len(blocks2), len(blocks2))])
        draws[index] = sample2.mean() - sample1.mean()
    return (estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def bootstrap_contrasts(
    panel: pd.DataFrame,
    *,
    metrics: Sequence[str],
    reps: int,
    seed: int,
    subsets: Sequence[str] = SUBSETS,
) -> pd.DataFrame:
    """G2 - G1 difference in pooled proportions with a participant-block bootstrap.

    The point estimate equals the difference between the arm proportions in the
    group summary.  Uncertainty resamples participants with replacement inside
    each arm and carries all of a participant's rows together, so replies from
    one participant are never treated as independent.
    """

    rows: list[dict[str, Any]] = []
    placement = str(panel["reaction_placement"].iloc[0]) if not panel.empty else PRIMARY_PLACEMENT
    rng = np.random.default_rng(seed)
    for subset in subsets:
        for metric in metrics:
            hours = _metric_hours(metric)
            selected = panel[_subset_mask(panel, subset, hours)] if not panel.empty else panel
            for dyad, part in selected.groupby("dyad", sort=True):
                g1 = part[part["group"].eq("G1")]
                g2 = part[part["group"].eq("G2")]
                estimate, low, high = _cluster_bootstrap_difference(g1, g2, metric, reps=reps, rng=rng)
                m1 = g1.groupby("actor_id")[metric].mean().astype(float)
                m2 = g2.groupby("actor_id")[metric].mean().astype(float)
                if part[metric].dtype == bool:
                    p_value, test = two_prop_pvalue(int(g1[metric].sum()), len(g1), int(g2[metric].sum()), len(g2))
                else:
                    p_value, test = (np.nan, "not_tested")
                rows.append(
                    {
                        "subset": subset,
                        "reaction_placement": placement,
                        "contrast": "G2_minus_G1",
                        "dyad": dyad,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_reps": int(reps),
                        "cluster_unit": "participant",
                        "actors_g1": int(len(m1)),
                        "actors_g2": int(len(m2)),
                        "events_g1": int(len(g1)),
                        "events_g2": int(len(g2)),
                        "participant_mean_difference": (
                            float(m2.mean() - m1.mean()) if len(m1) and len(m2) else np.nan
                        ),
                        "p_value": p_value,
                        "p_value_test": test,
                    }
                )
    return pd.DataFrame(rows)


def resolution_sensitivity(
    panel_start: pd.DataFrame,
    panel_end: pd.DataFrame,
    *,
    metrics: Sequence[str],
    subsets: Sequence[str] = SUBSETS,
) -> pd.DataFrame:
    """Compare outcomes under the reference placement and an alternative placement."""

    alternative = str(panel_end["reaction_placement"].iloc[0]) if not panel_end.empty else "round_end"

    key = ["group", "event_id"]
    merged = panel_start.merge(
        panel_end[key + list(metrics)],
        on=key,
        suffixes=("_start", "_end"),
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        for metric in metrics:
            hours = _metric_hours(metric)
            selected = merged[_subset_mask(merged, subset, hours)] if not merged.empty else merged
            for dyad, part in selected.groupby("dyad", sort=True):
                start = part[f"{metric}_start"].astype(float)
                end = part[f"{metric}_end"].astype(float)
                changed = int((start != end).sum())
                by_group_start = part.groupby("group")[f"{metric}_start"].mean()
                by_group_end = part.groupby("group")[f"{metric}_end"].mean()
                contrast_start = float(by_group_start.get("G2", np.nan) - by_group_start.get("G1", np.nan))
                contrast_end = float(by_group_end.get("G2", np.nan) - by_group_end.get("G1", np.nan))
                stable = bool(
                    np.isnan(contrast_start)
                    or np.isnan(contrast_end)
                    or np.sign(contrast_start) == np.sign(contrast_end)
                )
                rows.append(
                    {
                        "comparison": f"round_start_vs_{alternative}",
                        "subset": subset,
                        "dyad": dyad,
                        "metric": metric,
                        "rows": int(len(part)),
                        "rows_reclassified": changed,
                        "contrast_round_start": contrast_start,
                        "contrast_round_end": contrast_end,
                        "direction_stable": stable,
                    }
                )
    return pd.DataFrame(rows)


def downvote_timing(
    panel: pd.DataFrame,
    reactions: pd.DataFrame,
    *,
    subsets: Sequence[str] = ("rq1_analysis_set", "all_valid_targets"),
) -> pd.DataFrame:
    """Where human downvotes on selected targets fall relative to the reply.

    Each downvote on a selected target is classed as before the reply's round,
    in the same round, or after it; ``hours_after_reply`` uses the round start.
    """

    downs = reactions[reactions["event_type"].eq("downvote")].copy()
    if panel.empty or downs.empty:
        return pd.DataFrame(
            columns=["subset", "group", "dyad", "downvotes", "before_round", "same_round", "after_round", "by_replier",
                     "distinct_targets", "distinct_voters", "median_hours_after_reply", "q25_hours", "q75_hours"]
        )
    left = panel.copy()
    left["_content"] = _id_text(left["selected_target_content_id"])
    downs["_content"] = _id_text(downs["target_content_id"])
    merged = left.merge(
        downs[["group", "_content", "event_timestamp", "actor_id"]].rename(
            columns={"event_timestamp": "reaction_time", "actor_id": "voter"}
        ),
        on=["group", "_content"],
        how="inner",
    )
    merged["hours_after_reply"] = (merged["reaction_time"] - merged["event_timestamp"]).dt.total_seconds() / 3600.0
    reply_round = merged["event_timestamp"].dt.floor("h")
    merged["position"] = np.where(
        merged["reaction_time"].lt(reply_round), "before_round",
        np.where(merged["reaction_time"].eq(reply_round), "same_round", "after_round"),
    )
    merged["by_replier"] = _id_text(merged["voter"]).eq(_id_text(merged["actor_id"]))
    rows: list[dict[str, Any]] = []
    for subset in subsets:
        selected = merged[_subset_mask(merged, subset, 24)] if subset != "all_valid_targets" else merged
        for (group, dyad), part in selected.groupby(["group", "dyad"], sort=True):
            rows.append(
                {
                    "subset": subset,
                    "group": group,
                    "dyad": dyad,
                    "downvotes": int(len(part)),
                    "before_round": int(part["position"].eq("before_round").sum()),
                    "same_round": int(part["position"].eq("same_round").sum()),
                    "after_round": int(part["position"].eq("after_round").sum()),
                    "by_replier": int(part["by_replier"].sum()),
                    "distinct_targets": int(part["_content"].nunique()),
                    "distinct_voters": int(part["voter"].nunique()),
                    "median_hours_after_reply": float(part["hours_after_reply"].median()),
                    "q25_hours": float(part["hours_after_reply"].quantile(0.25)),
                    "q75_hours": float(part["hours_after_reply"].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def unconditional_downvote_rates(
    contents: pd.DataFrame,
    reactions: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Share of valid contributions with at least one human downvote, by arm and author type.

    Rows are split by whether the contribution was ever selected as a reply
    target in the sanction panel, so the conditional and unconditional views
    can be read side by side.  No time window is applied.
    """

    items = contents.copy()
    items["_content"] = _id_text(items["content_id"])
    items["group"] = items["group"].astype(str)
    items["author_type"] = items["author_type"].astype(str).str.upper()
    downs = reactions[reactions["event_type"].eq("downvote")].copy()
    downs["_content"] = _id_text(downs["target_content_id"])
    downs["group"] = downs["group"].astype(str)
    counts = downs.groupby(["group", "_content"]).size().rename("human_downvotes").reset_index()
    items = items.merge(counts, on=["group", "_content"], how="left")
    items["human_downvotes"] = items["human_downvotes"].fillna(0).astype(int)
    replied = set()
    if not panel.empty:
        replied = set(zip(panel["group"].astype(str), _id_text(panel["selected_target_content_id"])))
    items["replied_to"] = [(g, c) in replied for g, c in zip(items["group"], items["_content"])]
    rows: list[dict[str, Any]] = []
    for split_name, frame in (("all", items), ("replied_to", items[items["replied_to"]]), ("not_replied_to", items[~items["replied_to"]])):
        for (group, author_type), part in frame.groupby(["group", "author_type"], sort=True):
            n = int(len(part))
            k = int(part["human_downvotes"].gt(0).sum())
            low, high = wilson_ci(k, n)
            rows.append(
                {
                    "split": split_name,
                    "group": group,
                    "author_type": author_type,
                    "contents_n": n,
                    "downvoted_n": k,
                    "rate": k / n if n else np.nan,
                    "ci_low": low,
                    "ci_high": high,
                    "human_downvotes_total": int(part["human_downvotes"].sum()),
                }
            )
    return pd.DataFrame(rows)


def build_primary_estimand(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    start_time: str,
    end_time: str | None,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Three-row primary table (G1, G2, G2-G1) mirroring the RQ1 estimand schema."""

    metric = f"target_received_human_downvote_{horizon_hours}h"
    arms = summary[
        summary["subset"].eq(PRIMARY_SUBSET)
        & summary["reaction_placement"].eq(PRIMARY_PLACEMENT)
        & summary["dyad"].eq(PRIMARY_DYAD)
        & summary["metric"].eq(metric)
        & summary["group"].isin(["G1", "G2"])
    ]
    contrast = contrasts[
        contrasts["subset"].eq(PRIMARY_SUBSET)
        & contrasts["reaction_placement"].eq(PRIMARY_PLACEMENT)
        & contrasts["dyad"].eq(PRIMARY_DYAD)
        & contrasts["metric"].eq(metric)
    ]
    if len(arms) != 2 or len(contrast) != 1:
        raise ValueError("RQ3 primary estimand requires G1, G2, and one G2-G1 contrast row")
    by_group = arms.set_index("group")
    rows = []
    for group in ("G1", "G2"):
        row = by_group.loc[group]
        rows.append(
            {
                "group": group,
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "interval_type": "wilson_95",
                "numerator": int(row["numerator"]),
                "denominator_n": int(row["denominator_n"]),
                "actors_n": int(row["actors_n"]),
                "p_value": np.nan,
            }
        )
    c = contrast.iloc[0]
    rows.append(
        {
            "group": "G2-G1",
            "estimate": float(c["estimate"]),
            "ci_low": float(c["ci_low"]),
            "ci_high": float(c["ci_high"]),
            "interval_type": f"participant_cluster_bootstrap_95_{int(c['bootstrap_reps'])}_reps",
            "numerator": np.nan,
            "denominator_n": int(c["events_g1"] + c["events_g2"]),
            "actors_n": int(c["actors_g1"] + c["actors_g2"]),
            "p_value": float(c["p_value"]) if pd.notna(c["p_value"]) else np.nan,
        }
    )
    result = pd.DataFrame(rows)
    n1 = int(by_group.loc["G1", "denominator_n"])
    n2 = int(by_group.loc["G2", "denominator_n"])
    result.insert(0, "estimand", ESTIMAND_NAME)
    result.insert(0, "model", "descriptive_windowed_proportion")
    result["condition"] = result["group"].map(
        {"G1": "labels_visible", "G2": "labels_hidden", "G2-G1": "labels_hidden_minus_visible"}
    )
    result["analysis_window"] = f"{start_time} to {end_time or 'snapshot extraction'}"
    result["pre_human_artifacts_excluded"] = True
    result["denominator"] = (
        f"{n1 + n2} selected AI-authored targets of human replies in the RQ1 analysis set "
        f"(G1 {n1}; G2 {n2}); human downvotes within {horizon_hours} h of the reply"
    )
    result["event_type"] = "human_reactions_on_selected_targets"
    result["time_resolution"] = "reply_time_exact; reaction_time_round_1h"
    result["reaction_placement"] = PRIMARY_PLACEMENT
    result["inference_boundary"] = "one_community_per_condition_ecological_contrast"
    result["outcome_boundary"] = "platform_downvote_not_observed_intent_or_hostility"
    return result.sort_values("group", key=lambda v: v.map({"G1": 0, "G2": 1, "G2-G1": 2})).reset_index(drop=True)


def panel_signature(panel: pd.DataFrame) -> str:
    """Stable hash of the sanction panel for validation-time reproduction checks."""

    columns = [
        "group",
        "event_id",
        "actor_id",
        "event_timestamp",
        "selected_target_content_id",
        "target_author_type",
        "in_rq1_analysis_set",
    ]
    if panel.empty:
        return hashlib.sha256(b"empty").hexdigest()
    frame = panel[columns].copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True).astype("int64")
    for column in ("actor_id", "selected_target_content_id"):
        frame[column] = _id_text(frame[column])
    frame = frame.sort_values(["group", "event_id"], kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def observation_end_by_group(events: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Last exact comment timestamp per arm, used to flag right-censored windows."""

    comments = events[events["event_type"].eq(COMMENT_EVENT) & events["event_timestamp"].notna()]
    return {
        str(group): pd.Timestamp(value)
        for group, value in comments.groupby("group")["event_timestamp"].max().items()
    }


def default_metrics(horizons: Sequence[int]) -> list[str]:
    metrics: list[str] = []
    for hours in horizons:
        metrics.append(f"target_received_human_downvote_{hours}h")
    for hours in horizons:
        metrics.append(f"target_other_downvote_{hours}h")
    for hours in horizons:
        metrics.append(f"target_replier_downvote_{hours}h")
    for hours in horizons:
        metrics.append(f"reply_received_human_downvote_{hours}h")
    for hours in horizons:
        metrics.append(f"target_net_human_vote_delta_{hours}h")
    return metrics


def valid_days_for(analysis_set: str | None) -> list[int]:
    return list(analysis_set_config(analysis_set)["valid_days"])


__all__ = [
    "COMMENT_EVENT",
    "ESTIMAND_NAME",
    "PLACEMENTS",
    "PRIMARY_PLACEMENT",
    "PRIMARY_SUBSET",
    "REACTION_EVENTS",
    "SUBSETS",
    "add_vote_windows",
    "bootstrap_contrasts",
    "build_primary_estimand",
    "build_sanction_panel",
    "default_metrics",
    "derive_round_timestamps",
    "downvote_timing",
    "unconditional_downvote_rates",
    "observation_end_by_group",
    "panel_signature",
    "reaction_coverage",
    "resolution_sensitivity",
    "summarize_groups",
    "timestamp_reactions",
    "valid_days_for",
]
