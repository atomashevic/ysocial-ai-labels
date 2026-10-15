"""Sequential posterior simulations for evolving reply-network diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .data import COMMENT_EVENT, _NetworkState
from .model import _posterior_utilities


DYNAMIC_TERMS = (
    "z_log_prior_contact_count",
    "z_recent_prior_contact",
    "z_log_reverse_contact_count",
    "z_recent_reverse_contact",
    "z_log_receiver_popularity",
    "z_log_two_path_closure",
)


def _raw_dynamic_terms(features: dict[str, Any]) -> dict[str, float]:
    prior_count = float(features["prior_human_to_author_count"])
    reverse_count = float(features["prior_author_to_human_count"])
    prior_hours = float(features["prior_human_to_author_time_since_hours"])
    reverse_hours = float(features["prior_author_to_human_time_since_hours"])
    return {
        "z_log_prior_contact_count": float(np.log1p(prior_count)),
        "z_recent_prior_contact": (
            float(np.exp(-prior_hours / 24.0)) if prior_count > 0 else 0.0
        ),
        "z_log_reverse_contact_count": float(np.log1p(reverse_count)),
        "z_recent_reverse_contact": (
            float(np.exp(-reverse_hours / 24.0)) if reverse_count > 0 else 0.0
        ),
        "z_log_receiver_popularity": float(
            np.log1p(features["receiver_popularity_prior_human_count"])
        ),
        "z_log_two_path_closure": float(np.log1p(features["prior_two_path_count"])),
    }


def _term_scalers(term_manifest: pd.DataFrame) -> dict[str, tuple[float, float]]:
    settings = term_manifest.set_index("term")
    result: dict[str, tuple[float, float]] = {}
    for term in DYNAMIC_TERMS:
        if term not in settings.index or settings.loc[term, "status"] != "included_standardized":
            raise ValueError(f"Recursive network simulation requires standardized term {term}")
        mean = float(settings.loc[term, "raw_mean"])
        standard_deviation = float(settings.loc[term, "raw_sd"])
        if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError(f"Recursive network simulation has invalid scaler for {term}")
        result[term] = (mean, standard_deviation)
    return result


def _choice_event_lookup(model_frame: pd.DataFrame, data: Any) -> dict[tuple[str, str], int]:
    metadata = model_frame.drop_duplicates("choice_set_id").set_index("choice_set_id")
    lookup: dict[tuple[str, str], int] = {}
    for index, choice_id in enumerate(data.choice_ids):
        if choice_id not in metadata.index:
            raise ValueError(f"Choice {choice_id!r} is absent from the model frame")
        row = metadata.loc[choice_id]
        key = (str(row["group"]), str(row["event_id"]))
        if key in lookup:
            raise ValueError(f"Multiple choice sets map to event key {key!r}")
        lookup[key] = index
    return lookup


def _event_key(event: Any) -> tuple[str, str]:
    return str(event.group), str(event.event_id)


def _compiled_choice_specs(
    data: Any,
    model_frame: pd.DataFrame,
    term_indexes: list[int],
) -> dict[tuple[str, str], tuple[int, np.ndarray, tuple[str, ...], np.ndarray]]:
    event_lookup = _choice_event_lookup(model_frame, data)
    specs: dict[
        tuple[str, str], tuple[int, np.ndarray, tuple[str, ...], np.ndarray]
    ] = {}
    for event_key, choice_index in event_lookup.items():
        positions = np.flatnonzero(data.mask[choice_index])
        receivers = tuple(
            str(
                data.receiver_labels[
                    data.receiver_index[choice_index, int(position)]
                ]
            ).split("::", maxsplit=1)[1]
            for position in positions
        )
        observed = data.x[choice_index, positions][:, term_indexes]
        specs[event_key] = (choice_index, positions, receivers, observed)
    return specs


def _compiled_event_batches(
    event_history: pd.DataFrame,
) -> dict[str, list[tuple[Any, ...]]]:
    columns = [
        "group",
        "event_id",
        "event_timestamp",
        "actor_id",
        "actor_type",
        "target_author_id",
        "valid_clock_seconds",
        "root_id",
    ]
    comments = event_history[event_history["event_type"].eq(COMMENT_EVENT)][columns].copy()
    comments["event_timestamp"] = pd.to_datetime(
        comments["event_timestamp"], utc=True, errors="raise"
    )
    comments = comments.sort_values(
        ["group", "event_timestamp", "event_id"], kind="stable"
    )
    batches: dict[str, list[tuple[Any, ...]]] = {}
    for group, arm in comments.groupby("group", sort=True):
        batches[str(group)] = [
            tuple(part.itertuples(index=False, name="NetworkEvent"))
            for _, part in arm.groupby("event_timestamp", sort=False)
        ]
    return batches


def _posterior_subset(idata: Any, selected_draws: np.ndarray) -> Any:
    """Materialize only the posterior samples used by recursive simulation."""

    variables = [
        name
        for name in (
            "beta",
            "receiver_effect",
            "actor_ai_effect",
            "topic_ai_effect",
            "receiver_block",
            "block_intercept",
            "block_network_slope",
        )
        if name in idata.posterior
    ]
    posterior = (
        idata.posterior[variables]
        .stack(sample=("chain", "draw"))
        .isel(sample=selected_draws)
        .reset_index("sample", drop=True)
        .rename({"sample": "draw"})
        .expand_dims(chain=[0])
    )
    return SimpleNamespace(posterior=posterior)


def recursive_posterior_predictive_choices(
    idata: Any,
    data: Any,
    model_frame: pd.DataFrame,
    event_history: pd.DataFrame,
    term_manifest: pd.DataFrame,
    *,
    draws: int = 1_000,
    random_seed: int = 20260715,
) -> pd.DataFrame:
    """Simulate selected receivers while recursively updating network history.

    Thread risk sets, content covariates, AI events, unmodeled human events, and
    event times remain observed. Modeled human target choices update the state
    used for every later modeled reply.
    """

    if draws < 1:
        raise ValueError("Recursive posterior prediction requires at least one draw")
    scalers = _term_scalers(term_manifest)
    term_indexes = [data.terms.index(term) for term in DYNAMIC_TERMS]
    posterior_draws = int(
        idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    )
    rng = np.random.default_rng(random_seed)
    selected_draws = (
        np.arange(posterior_draws, dtype=np.int64)
        if draws >= posterior_draws
        else np.sort(rng.choice(posterior_draws, size=draws, replace=False))
    )
    selected_idata = _posterior_subset(idata, selected_draws)
    beta_da = selected_idata.posterior["beta"]
    posterior_terms = tuple(str(value) for value in beta_da.coords["term"].values)
    beta_indexes = [posterior_terms.index(term) for term in DYNAMIC_TERMS]
    beta = beta_da.stack(sample=("chain", "draw")).transpose("sample", "term").values
    utility = _posterior_utilities(selected_idata, data)
    choice_specs = _compiled_choice_specs(data, model_frame, term_indexes)
    event_batches = _compiled_event_batches(event_history)
    event_keys = {
        _event_key(event)
        for batches in event_batches.values()
        for timestamp_events in batches
        for event in timestamp_events
    }
    missing_events = sorted(set(choice_specs).difference(event_keys))
    if missing_events:
        preview = missing_events[:5]
        raise ValueError(
            f"Recursive simulation cannot locate {len(missing_events)} modeled events; "
            f"examples={preview}"
        )

    rows: list[dict[str, Any]] = []
    for local_draw, posterior_draw in enumerate(selected_draws):
        for group, batches in event_batches.items():
            state = _NetworkState()
            for timestamp_events in batches:
                simulated: dict[
                    tuple[str, str], tuple[str, dict[str, Any], float]
                ] = {}
                for event in timestamp_events:
                    event_key = _event_key(event)
                    spec = choice_specs.get(event_key)
                    if spec is None:
                        continue
                    choice_index, positions, receivers, observed = spec
                    dynamic = np.empty((len(positions), len(DYNAMIC_TERMS)), dtype=float)
                    feature_rows: list[dict[str, Any]] = []
                    for row_index, receiver in enumerate(receivers):
                        features = state.features(
                            event.actor_id,
                            receiver,
                            float(event.valid_clock_seconds),
                        )
                        feature_rows.append(features)
                        raw = _raw_dynamic_terms(features)
                        dynamic[row_index] = [
                            (raw[term] - scalers[term][0]) / scalers[term][1]
                            for term in DYNAMIC_TERMS
                        ]
                    delta = dynamic - observed
                    row_utility = utility[local_draw, choice_index, positions] + (
                        delta @ beta[local_draw, beta_indexes]
                    )
                    centered = row_utility - np.max(row_utility)
                    probability = np.exp(centered)
                    probability /= probability.sum()
                    selected_offset = int(rng.choice(len(positions), p=probability))
                    selected_position = int(positions[selected_offset])
                    selected_receiver = data.receiver_labels[
                        data.receiver_index[choice_index, selected_position]
                    ]
                    selected_features = feature_rows[selected_offset]
                    simulated[event_key] = (
                        str(selected_receiver).split("::", maxsplit=1)[1],
                        selected_features,
                        float(probability[selected_offset]),
                    )
                    rows.append(
                        {
                            "posterior_draw_index": int(posterior_draw),
                            "choice_set_id": data.choice_ids[choice_index],
                            "group": str(group),
                            "chosen_position": selected_position,
                            "chosen_receiver": selected_receiver,
                            "chosen_is_ai": bool(
                                data.candidate_is_ai[choice_index, selected_position]
                            ),
                            "chosen_probability": float(probability[selected_offset]),
                            "simulated_prior_human_to_author_count": int(
                                selected_features["prior_human_to_author_count"]
                            ),
                            "simulated_prior_author_to_human_count": int(
                                selected_features["prior_author_to_human_count"]
                            ),
                            "simulated_prior_two_path_count": int(
                                selected_features["prior_two_path_count"]
                            ),
                            "simulation_scope": (
                                "recursively_updated_network_history_conditional_on_"
                                "observed_event_times_and_thread_risk_sets"
                            ),
                        }
                    )
                for event in timestamp_events:
                    selected = simulated.get(_event_key(event))
                    if selected is None:
                        state.add(event)
                        continue
                    receiver = selected[0]
                    state.add(
                        SimpleNamespace(
                            actor_id=event.actor_id,
                            target_author_id=receiver,
                            actor_type=event.actor_type,
                            valid_clock_seconds=event.valid_clock_seconds,
                            root_id=event.root_id,
                        )
                    )
    expected_rows = len(selected_draws) * data.n_choices
    if len(rows) != expected_rows:
        raise ValueError(
            f"Recursive simulation produced {len(rows)} rows; expected {expected_rows}"
        )
    return pd.DataFrame(rows)


__all__ = ["recursive_posterior_predictive_choices"]
