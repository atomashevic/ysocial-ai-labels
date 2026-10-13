"""Piecewise exposure-time models for verified browser intervals."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


ACTION_TYPES = ("comment", "upvote", "downvote")
DEFAULT_BREAKS = (0.0, 5.0, 15.0, 30.0, 60.0, np.inf)


def mark_unobserved_action_channels(
    conversion_summary: pd.DataFrame,
    source_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Prevent missing action telemetry from appearing as zero conversion."""

    out = conversion_summary.copy()
    if "target_author_type" in out.columns and "author_type" not in out.columns:
        out["author_type"] = out["target_author_type"]
    if "action_rate_per_1000_verified_visible_seconds" in out.columns:
        out["rate_per_1000_verified_seconds"] = out[
            "action_rate_per_1000_verified_visible_seconds"
        ]
    required = {"group", "source", "available_rows", "total_rows"}
    if not required.issubset(source_availability.columns):
        raise ValueError("Source availability is missing action-channel fields")
    rate_columns = [
        column
        for column in [
            "linked_actions_n",
            "episodes_with_action_n",
            "episode_conversion_rate",
            "action_rate_per_1000_verified_visible_seconds",
            "rate_per_1000_verified_seconds",
            "action_rate_ci_low",
            "action_rate_ci_high",
            "standardized_rate_per_1000_verified_seconds",
            "standardized_rate_ci_low",
            "standardized_rate_ci_high",
        ]
        if column in out.columns
    ]
    for group in ("G1", "G2"):
        for action_type in ("upvote", "downvote"):
            source = f"backend_{action_type}_timestamps"
            availability = source_availability[
                source_availability["group"].astype(str).eq(group)
                & source_availability["source"].eq(source)
            ]
            if len(availability) != 1:
                continue
            total_rows = int(availability["total_rows"].iloc[0])
            available_rows = int(availability["available_rows"].iloc[0])
            unavailable = total_rows > 0 and available_rows < total_rows
            if unavailable:
                selected = out["group"].astype(str).eq(group) & out["action_type"].eq(action_type)
                out.loc[selected, rate_columns] = np.nan
                status = (
                    "unobserved_missing_action_timestamps"
                    if available_rows == 0
                    else "unobserved_incomplete_action_timestamps"
                )
                out.loc[selected, "observed_status"] = status
                if "standardization_status" in out.columns:
                    out.loc[selected, "standardization_status"] = status
    out["inference_status"] = np.where(
        out["observed_status"].eq("observed"),
        "descriptive_pending_tracking_gate",
        out["observed_status"],
    )
    return out


def strict_browser_choice_sets(browser_choice_sets: pd.DataFrame) -> pd.DataFrame:
    """Keep choice alternatives observed before an unambiguous linked reply."""

    required = {
        "choice_set_id",
        "chosen",
        "observed_status",
        "candidate_seen_before_action",
        "chosen_episode_linked",
        "browser_link_ambiguous",
        "browser_session_ambiguous",
        "chosen_action_within_episode",
    }
    missing = sorted(required.difference(browser_choice_sets.columns))
    if missing:
        raise ValueError(f"Browser choice rows are missing strict-risk-set columns: {missing}")
    if browser_choice_sets.empty:
        return browser_choice_sets.copy()
    work = browser_choice_sets[
        browser_choice_sets["observed_status"].eq("observed")
        & ~browser_choice_sets["browser_link_ambiguous"].astype("boolean").fillna(True).astype(bool)
        & ~browser_choice_sets["browser_session_ambiguous"].astype("boolean").fillna(True).astype(bool)
        & browser_choice_sets["candidate_seen_before_action"].astype("boolean").fillna(False).astype(bool)
    ].copy()
    eligible_ids: list[Any] = []
    for choice_id, part in work.groupby("choice_set_id", sort=False):
        chosen = part[part["chosen"].astype(bool)]
        if (
            len(part) >= 2
            and len(chosen) == 1
            and chosen["chosen_episode_linked"].astype("boolean").fillna(False).astype(bool).all()
            and chosen["chosen_action_within_episode"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            .all()
        ):
            eligible_ids.append(choice_id)
    strict = work[work["choice_set_id"].isin(eligible_ids)].copy()
    if strict.empty:
        return strict
    counts = strict.groupby("choice_set_id")["choice_set_id"].transform("size")
    strict["browser_risk_set_candidate_n"] = counts.astype(int)
    strict["browser_risk_set_status"] = "strict_pre_action_rendered"
    return strict.reset_index(drop=True)


def _author_type(value: Any) -> str | None:
    text = str(value).strip().upper()
    if text in {"AI", "A"}:
        return "AI"
    if text in {"H", "HUMAN"}:
        return "H"
    return None


def _bin_label(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:g}s" if np.isfinite(upper) else f"{lower:g}s+"


def _eligible_actions(linked_actions: pd.DataFrame) -> pd.DataFrame:
    if linked_actions.empty:
        return linked_actions.copy()
    required = {
        "browser_episode_id",
        "action_type",
        "action_timestamp_utc",
        "action_within_episode",
        "pre_action_linked",
    }
    missing = sorted(required.difference(linked_actions.columns))
    if missing:
        raise ValueError(f"Linked actions are missing hazard columns: {missing}")
    keep = (
        linked_actions["browser_episode_id"].notna()
        & linked_actions["action_type"].isin(ACTION_TYPES)
        & linked_actions["action_within_episode"].astype("boolean").fillna(False).astype(bool)
        & linked_actions["pre_action_linked"].astype("boolean").fillna(False).astype(bool)
    )
    for column in [
        "link_ambiguous",
        "session_ambiguous",
        "episode_right_censored",
        "backend_duplicate_conflict",
    ]:
        if column in linked_actions.columns:
            keep &= ~linked_actions[column].astype("boolean").fillna(True).astype(bool)
    if "episode_interval_verified" in linked_actions.columns:
        keep &= (
            linked_actions["episode_interval_verified"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    actions = linked_actions[keep].copy()
    actions["action_timestamp_utc"] = pd.to_datetime(
        actions["action_timestamp_utc"], utc=True, errors="coerce"
    )
    return actions.dropna(subset=["action_timestamp_utc"]).sort_values(
        ["browser_episode_id", "action_type", "action_timestamp_utc"], kind="stable"
    )


def _action_channel_statuses(
    source_availability: pd.DataFrame | None,
) -> dict[tuple[str, str], str]:
    statuses: dict[tuple[str, str], str] = {}
    if source_availability is None or source_availability.empty:
        return statuses
    required = {"group", "source", "available_rows", "total_rows"}
    missing = sorted(required.difference(source_availability.columns))
    if missing:
        raise ValueError(f"Source availability is missing hazard fields: {missing}")
    for group in ("G1", "G2"):
        for action_type in ("upvote", "downvote"):
            selected = source_availability[
                source_availability["group"].astype(str).eq(group)
                & source_availability["source"].eq(f"backend_{action_type}_timestamps")
            ]
            if len(selected) != 1:
                statuses[(group, action_type)] = "unobserved_source_availability_missing"
                continue
            total_rows = int(selected["total_rows"].iloc[0])
            available_rows = int(selected["available_rows"].iloc[0])
            if total_rows > 0 and available_rows == 0:
                statuses[(group, action_type)] = "unobserved_missing_action_timestamps"
            elif total_rows > 0 and available_rows < total_rows:
                statuses[(group, action_type)] = "unobserved_incomplete_action_timestamps"
    return statuses


def build_piecewise_exposure(
    episodes: pd.DataFrame,
    linked_actions: pd.DataFrame,
    *,
    breaks: Sequence[float] = DEFAULT_BREAKS,
    source_availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Split hazard-eligible DOM episodes into action-specific intervals.

    Each action type receives a separate risk process. Right-censored DOM
    episodes never contribute verified time. The interval is truncated at the
    first linked action of that type. Channels with incomplete action timestamps
    retain observed episode-time with a missing event state and never enter a
    fitted hazard model.
    """

    required = {
        "browser_episode_id",
        "session_id",
        "group",
        "y_user_id",
        "item_id",
        "author_type_ground_truth",
        "impression_start_utc",
        "impression_end_utc",
        "right_censored",
    }
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"Browser episodes are missing hazard columns: {missing}")
    boundaries = np.asarray(breaks, dtype=float)
    if len(boundaries) < 2 or boundaries[0] != 0 or not np.all(np.diff(boundaries) > 0):
        raise ValueError("breaks must start at zero and increase strictly")

    eligibility = (
        episodes["hazard_eligible"].fillna(False).astype(bool)
        if "hazard_eligible" in episodes.columns
        else ~episodes["right_censored"].fillna(True).astype(bool)
    )
    verified = episodes[eligibility].copy()
    verified["impression_start_utc"] = pd.to_datetime(
        verified["impression_start_utc"], utc=True, errors="coerce"
    )
    verified["impression_end_utc"] = pd.to_datetime(
        verified["impression_end_utc"], utc=True, errors="coerce"
    )
    verified = verified.dropna(subset=["impression_start_utc", "impression_end_utc"])
    verified = verified[verified["impression_end_utc"].gt(verified["impression_start_utc"])]
    actions = _eligible_actions(linked_actions)
    channel_statuses = _action_channel_statuses(source_availability)
    first_action = (
        actions.groupby(["browser_episode_id", "action_type"], sort=False)["action_timestamp_utc"]
        .min()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for episode in verified.itertuples(index=False):
        start = pd.Timestamp(episode.impression_start_utc)
        end = pd.Timestamp(episode.impression_end_utc)
        duration = (end - start).total_seconds()
        author_type = _author_type(episode.author_type_ground_truth)
        for action_type in ACTION_TYPES:
            observation_status = channel_statuses.get(
                (str(episode.group), action_type), "verified_interval"
            )
            channel_observed = observation_status == "verified_interval"
            action_time = (
                first_action.get((episode.browser_episode_id, action_type))
                if channel_observed
                else None
            )
            event_second: float | None = None
            if action_time is not None and start <= action_time <= end:
                event_second = max(0.0, (pd.Timestamp(action_time) - start).total_seconds())
            risk_end = event_second if event_second is not None else duration
            for lower, upper in zip(boundaries[:-1], boundaries[1:]):
                if lower > risk_end or lower >= duration:
                    break
                interval_end = min(upper, risk_end, duration)
                exposure = interval_end - lower
                event = event_second is not None and lower <= event_second <= upper
                if event and exposure <= 0.0:
                    exposure = 1.0e-3
                if exposure <= 0.0:
                    continue
                rows.append(
                    {
                        "browser_episode_id": episode.browser_episode_id,
                        "session_id": episode.session_id,
                        "group": str(episode.group),
                        "y_user_id": episode.y_user_id,
                        "item_id": episode.item_id,
                        "author_type": author_type,
                        "action_type": action_type,
                        "time_bin": _bin_label(float(lower), float(upper)),
                        "interval_start_seconds": float(lower),
                        "interval_end_seconds": float(interval_end),
                        "exposure_seconds": float(exposure),
                        "event": int(event) if channel_observed else pd.NA,
                        "observed_status": observation_status,
                    }
                )
                if event:
                    break
    return pd.DataFrame(rows)


def _full_hazard_gate_passed(tracking_qc: pd.DataFrame) -> bool:
    if tracking_qc.empty:
        return False
    required = {"metric", "group", "gate_pass"}
    if not required.issubset(tracking_qc.columns):
        raise ValueError("Tracking QC needs metric, group, and gate_pass columns")
    metric = (
        "full_interval_hazard_gates_pass"
        if tracking_qc["metric"].eq("full_interval_hazard_gates_pass").any()
        else "all_tracking_gates_pass"
    )
    result = tracking_qc[
        tracking_qc["metric"].eq(metric) & tracking_qc["group"].eq("all")
    ]
    return bool(
        len(result) == 1
        and result["gate_pass"].notna().all()
        and result["gate_pass"].astype(bool).all()
    )


def _hazard_frame(intervals: pd.DataFrame, action_type: str) -> pd.DataFrame:
    frame = intervals[intervals["action_type"].eq(action_type)].copy()
    if "observed_status" in frame.columns:
        frame = frame[frame["observed_status"].eq("verified_interval")]
    frame = frame[frame.get("event", pd.Series(index=frame.index, dtype=float)).notna()]
    frame = frame[frame["author_type"].isin(["H", "AI"])]
    if frame.empty:
        return frame
    frame["is_ai"] = frame["author_type"].eq("AI").astype(float)
    frame["is_g2"] = frame["group"].eq("G2").astype(float)
    frame["g2_x_ai"] = frame["is_g2"] * frame["is_ai"]
    return frame


def _unobserved_action_status(intervals: pd.DataFrame, action_type: str) -> str | None:
    if "observed_status" not in intervals.columns:
        return None
    statuses = set(
        intervals.loc[intervals["action_type"].eq(action_type), "observed_status"]
        .dropna()
        .astype(str)
    )
    unobserved = sorted(status for status in statuses if status.startswith("unobserved_"))
    if not unobserved:
        return None
    if len(unobserved) == 1:
        return unobserved[0]
    return "unobserved_incomplete_action_timestamps"


def _fit_hazard_model(frame: pd.DataFrame) -> tuple[Any, pd.Series]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    clusters = frame["group"].astype(str) + "::" + frame["y_user_id"].astype(str)
    model = smf.glm(
        "event ~ is_g2 + is_ai + g2_x_ai + C(time_bin)",
        data=frame,
        family=sm.families.Poisson(),
        offset=np.log(frame["exposure_seconds"]),
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": clusters}), clusters


def standardized_piecewise_rates(
    intervals: pd.DataFrame,
    tracking_qc: pd.DataFrame,
    *,
    random_seed: int = 20260715,
    coefficient_draws: int = 4_000,
) -> pd.DataFrame:
    """Standardize action hazards over one pooled time-bin opportunity profile.

    Rates are produced only after every tracking and interval gate passes. The
    empirical exposure-time distribution across piecewise bins is held common
    across the four arm-by-author projections. Cluster-robust coefficient draws
    propagate model uncertainty to the standardized rate.
    """

    rows: list[dict[str, Any]] = []
    gate_passed = _full_hazard_gate_passed(tracking_qc)
    if coefficient_draws < 100:
        raise ValueError("coefficient_draws must be at least 100")
    rng = np.random.default_rng(random_seed)
    for action_type in ACTION_TYPES:
        frame = _hazard_frame(intervals, action_type)
        unavailable_status = _unobserved_action_status(intervals, action_type)
        if unavailable_status is not None:
            status = unavailable_status
        elif not gate_passed:
            status = "blocked_pending_tracking_audit"
        elif frame.empty or int(frame["event"].sum()) == 0:
            status = "no_linked_events"
        else:
            status = "gate_passed_common_time_bin_standardization"
        if status != "gate_passed_common_time_bin_standardization":
            for group in ("G1", "G2"):
                for author_type in ("H", "AI"):
                    rows.append(
                        {
                            "group": group,
                            "action_type": action_type,
                            "author_type": author_type,
                            "standardized_rate_per_1000_verified_seconds": np.nan,
                            "standardized_rate_ci_low": np.nan,
                            "standardized_rate_ci_high": np.nan,
                            "standardization_status": status,
                        }
                    )
            continue

        from patsy import build_design_matrices

        result, _ = _fit_hazard_model(frame)
        exposure = (
            frame.groupby("time_bin", observed=True)["exposure_seconds"]
            .sum()
            .sort_index()
        )
        time_bins = exposure.index.astype(str).tolist()
        weights = exposure.to_numpy(dtype=float)
        weights /= weights.sum()
        coefficients = rng.multivariate_normal(
            result.params.to_numpy(dtype=float),
            result.cov_params().to_numpy(dtype=float),
            size=coefficient_draws,
            check_valid="raise",
        )
        for group in ("G1", "G2"):
            for author_type in ("H", "AI"):
                is_g2 = float(group == "G2")
                is_ai = float(author_type == "AI")
                counterfactual = pd.DataFrame(
                    {
                        "is_g2": is_g2,
                        "is_ai": is_ai,
                        "g2_x_ai": is_g2 * is_ai,
                        "time_bin": time_bins,
                    }
                )
                design = np.asarray(
                    build_design_matrices(
                        [result.model.data.design_info],
                        counterfactual,
                        return_type="dataframe",
                    )[0],
                    dtype=float,
                )
                rate_draws = 1_000.0 * (np.exp(coefficients @ design.T) @ weights)
                point_rate = float(
                    1_000.0
                    * np.dot(np.exp(design @ result.params.to_numpy(dtype=float)), weights)
                )
                rows.append(
                    {
                        "group": group,
                        "action_type": action_type,
                        "author_type": author_type,
                        "standardized_rate_per_1000_verified_seconds": point_rate,
                        "standardized_rate_ci_low": float(np.quantile(rate_draws, 0.025)),
                        "standardized_rate_ci_high": float(np.quantile(rate_draws, 0.975)),
                        "standardization_status": status,
                    }
                )
    return pd.DataFrame(rows)


def fit_piecewise_hazards(
    intervals: pd.DataFrame,
    tracking_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Fit separate clustered piecewise-exponential models after all gates pass."""

    if intervals.empty:
        raise ValueError("No verified interval-time remains after tracking gates")

    rows: list[dict[str, Any]] = []
    gate_passed = _full_hazard_gate_passed(tracking_qc)
    for action_type in ACTION_TYPES:
        unavailable_status = _unobserved_action_status(intervals, action_type)
        if unavailable_status is not None or not gate_passed:
            rows.append(
                {
                    "action_type": action_type,
                    "term": "not_estimated",
                    "hazard_ratio": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "inference_status": (
                        unavailable_status
                        if unavailable_status is not None
                        else "blocked_pending_tracking_audit"
                    ),
                }
            )
            continue
        frame = _hazard_frame(intervals, action_type)
        if frame.empty or int(frame["event"].sum()) == 0:
            rows.append(
                {
                    "action_type": action_type,
                    "term": "not_estimable",
                    "hazard_ratio": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "inference_status": "no_linked_events",
                }
            )
            continue
        result, clusters = _fit_hazard_model(frame)
        confidence = result.conf_int()
        for term in ("is_ai", "g2_x_ai"):
            rows.append(
                {
                    "action_type": action_type,
                    "term": term,
                    "hazard_ratio": float(np.exp(result.params[term])),
                    "ci_low": float(np.exp(confidence.loc[term, 0])),
                    "ci_high": float(np.exp(confidence.loc[term, 1])),
                    "p_value": float(result.pvalues[term]),
                    "events": int(frame["event"].sum()),
                    "verified_seconds": float(frame["exposure_seconds"].sum()),
                    "participants": int(clusters.nunique()),
                    "inference_status": "gate_passed_clustered_piecewise_exponential",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "build_piecewise_exposure",
    "fit_piecewise_hazards",
    "mark_unobserved_action_channels",
    "standardized_piecewise_rates",
    "strict_browser_choice_sets",
]
