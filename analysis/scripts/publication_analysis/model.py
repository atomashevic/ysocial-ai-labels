"""Bayesian relational-event choice models for publication analyses.

The primary likelihood conditions on an observed human action and compares the
chosen content item with the other candidates in its choice set.  Receiver,
participant, and topic effects share information across the evolving reply
network without treating candidate rows as independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import AnalysisConfig


REQUIRED_COLUMNS = (
    "choice_set_id",
    "chosen",
    "candidate_author_id",
    "actor_id",
    "primary_root_topic",
)
AI_TERM_CANDIDATES = ("candidate_target_is_ai", "target_ai", "candidate_is_ai")
NETWORK_TERM_TOKENS = (
    "prior",
    "repeat",
    "reciproc",
    "degree",
    "popular",
    "two_path",
    "shared",
    "closure",
    "contact",
)
FOCAL_ARM_TOKENS = ("g2", "group2", "group_2", "hidden")
CLOSURE_TOKENS = ("two_path", "two-path", "shared_partner", "shared-partner", "closure", "triad")
POPULARITY_TOKENS = ("popularity", "popular", "in_degree", "indegree", "target_in")
REVERSE_TOKENS = (
    "reverse",
    "reciproc",
    "author_to_human",
    "author_to_actor",
    "incoming_from_candidate",
)
REPETITION_TOKENS = (
    "prior_contact",
    "repeat",
    "human_to_author",
    "actor_to_author",
    "actor_to_candidate",
    "outgoing_to_candidate",
    "network_history",
)


@dataclass(frozen=True)
class ModelSpec:
    """Nested random-effect specification for a publication model."""

    name: str
    description: str
    fixed_families: tuple[str, ...]
    receiver_frailty: bool
    participant_ai_slope: bool
    topic_ai_slope: bool


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "M0": ModelSpec(
        "M0",
        "opportunity and content controls",
        ("opportunity_content",),
        False,
        False,
        False,
    ),
    "M1": ModelSpec(
        "M1",
        "M0 plus AI, G2-by-AI, and all hierarchical frailties",
        ("opportunity_content", "focal"),
        True,
        True,
        True,
    ),
    "M2": ModelSpec(
        "M2",
        "M1 plus repetition and reverse-contact terms",
        ("opportunity_content", "focal", "repetition", "reverse"),
        True,
        True,
        True,
    ),
    "M3": ModelSpec(
        "M3",
        "M2 plus popularity and one prespecified closure term",
        ("opportunity_content", "focal", "repetition", "reverse", "popularity", "closure"),
        True,
        True,
        True,
    ),
    "RQ2": ModelSpec(
        "RQ2",
        "M1 plus general, AI-specific, and G2-specific directional contact history",
        ("opportunity_content", "focal", "repetition", "reverse"),
        True,
        True,
        True,
    ),
}


@dataclass(frozen=True)
class PreparedModelData:
    """Padded arrays and stable indexes for candidate-content softmax models."""

    x: np.ndarray
    candidate_is_ai: np.ndarray
    mask: np.ndarray
    chosen_position: np.ndarray
    receiver_index: np.ndarray
    receiver_is_ai: np.ndarray
    receiver_groups: tuple[str, ...]
    actor_index: np.ndarray
    actor_groups: tuple[str, ...]
    topic_index: np.ndarray
    terms: tuple[str, ...]
    term_families: tuple[str, ...]
    ai_term: str
    choice_ids: tuple[str, ...]
    receiver_labels: tuple[str, ...]
    actor_labels: tuple[str, ...]
    topic_labels: tuple[str, ...]
    choice_groups: tuple[str, ...]

    @property
    def n_choices(self) -> int:
        return int(self.x.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.x.shape[1])

    def __getitem__(self, name: str) -> Any:
        """Allow lightweight dictionary-style access in pipeline code."""

        return getattr(self, name)


@dataclass(frozen=True)
class LatentBlockRecoveryInput:
    """Synthetic conditional-choice fixture and its known receiver partition."""

    choice_rows: pd.DataFrame
    receiver_truth: pd.DataFrame
    true_parameters: Mapping[str, Any]
    simulation_scope: str = "conditional_candidate_sets_not_recursive_event_history"


def _resolve_ai_term(frame: pd.DataFrame, ai_term: str | None) -> str:
    if ai_term is not None:
        if ai_term not in frame.columns:
            raise ValueError(f"AI term {ai_term!r} is absent from the choice-set rows")
        return ai_term
    for candidate in AI_TERM_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Choice-set rows need one AI indicator: {AI_TERM_CANDIDATES}")


def _binary_values(series: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        values = series.astype(int).to_numpy()
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=float)
    else:
        mapped = series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
        )
        values = mapped.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must contain only non-missing binary values")
    return values.astype(np.int8)


def _entity_labels(frame: pd.DataFrame, id_column: str) -> pd.Series:
    ids = frame[id_column].astype(str)
    if "group" not in frame.columns:
        return ids
    return frame["group"].fillna("UNKNOWN").astype(str) + "::" + ids


def _stable_index(labels: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    levels = tuple(sorted(set(labels)))
    lookup = {label: index for index, label in enumerate(levels)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int64), levels


def _validate_choice_set(part: pd.DataFrame, choice_id: str) -> None:
    if len(part) < 2:
        raise ValueError(f"Choice set {choice_id!r} has fewer than two candidates")
    if int(part["_chosen"].sum()) != 1:
        raise ValueError(f"Choice set {choice_id!r} must have exactly one chosen candidate")
    for column in ("_actor_label", "_topic_label", "_choice_group"):
        if part[column].nunique(dropna=False) != 1:
            raise ValueError(f"{column.removeprefix('_')} must be constant within choice set {choice_id!r}")


def _is_arm_ai_interaction(term: str) -> bool:
    return term.lower() == "g2_x_candidate_ai"


def _classify_term(term: str, ai_term: str) -> str:
    normalized = term.lower()
    if term == ai_term or _is_arm_ai_interaction(term):
        return "focal"
    if any(token in normalized for token in CLOSURE_TOKENS):
        return "closure"
    if any(token in normalized for token in REVERSE_TOKENS):
        return "reverse"
    if any(token in normalized for token in REPETITION_TOKENS):
        return "repetition"
    if any(token in normalized for token in POPULARITY_TOKENS):
        return "popularity"
    return "opportunity_content"


def _term_families(
    terms: Sequence[str],
    ai_term: str,
    closure_term: str | None,
) -> tuple[str, ...]:
    families = [_classify_term(term, ai_term) for term in terms]
    closure_candidates = [term for term, family in zip(terms, families) if family == "closure"]
    if closure_term is not None:
        if closure_term not in terms:
            raise ValueError(f"Prespecified closure term {closure_term!r} is absent from numeric terms")
        if closure_term not in closure_candidates:
            raise ValueError(f"Prespecified closure term {closure_term!r} is not recognized as closure")
        return tuple(
            "closure" if term == closure_term else "excluded_closure" if family == "closure" else family
            for term, family in zip(terms, families)
        )
    if len(closure_candidates) > 1:
        raise ValueError(
            "M3 permits one prespecified closure term; choose closure_term from "
            f"{closure_candidates}"
        )
    return tuple(families)


def _receiver_types(work: pd.DataFrame, receiver_labels: Sequence[str]) -> np.ndarray:
    type_counts = work.groupby("_receiver_label", sort=False)["_candidate_ai"].nunique()
    conflicted = type_counts[type_counts > 1].index.tolist()
    if conflicted:
        raise ValueError(f"Receiver accounts have conflicting human/AI labels: {conflicted[:5]}")
    lookup = (
        work.drop_duplicates("_receiver_label", keep="first")
        .set_index("_receiver_label")["_candidate_ai"]
        .astype(int)
        .to_dict()
    )
    return np.asarray([lookup[label] for label in receiver_labels], dtype=np.int8)


def _groups_for_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Recover the arm prefix embedded in stable actor and receiver labels."""

    groups = tuple(str(label).split("::", maxsplit=1)[0] for label in labels)
    if any(group in {"", "UNKNOWN"} for group in groups):
        raise ValueError("Actor and receiver labels require an observed group")
    return groups


def prepare_model_data(
    frame: pd.DataFrame,
    numeric_terms: Sequence[str] = (),
    *,
    ai_term: str | None = None,
    closure_term: str | None = None,
) -> PreparedModelData:
    """Validate canonical choice rows and construct padded model arrays.

    Candidate rows remain content-level alternatives. Repeated content from one
    author therefore shares a receiver effect while retaining separate utility.
    Numeric terms are used as supplied; upstream panel construction owns their
    scaling and event-time validity.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Choice-set rows are missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Choice-set rows are empty")
    resolved_ai = _resolve_ai_term(frame, ai_term)
    requested_terms = tuple(dict.fromkeys((resolved_ai, *numeric_terms)))
    missing_terms = [term for term in requested_terms if term not in frame.columns]
    if missing_terms:
        raise ValueError(f"Choice-set rows are missing numeric terms: {missing_terms}")
    term_families = _term_families(requested_terms, resolved_ai, closure_term)

    work = frame.copy()
    work["_chosen"] = _binary_values(work["chosen"], "chosen")
    work["_candidate_ai"] = _binary_values(work[resolved_ai], resolved_ai)
    work["_actor_label"] = _entity_labels(work, "actor_id")
    work["_receiver_label"] = _entity_labels(work, "candidate_author_id")
    work["_topic_label"] = work["primary_root_topic"].fillna("UNTAGGED").astype(str)
    work["_choice_group"] = (
        work["group"].fillna("UNKNOWN").astype(str) if "group" in work.columns else "ALL"
    )
    for term in requested_terms:
        work[term] = pd.to_numeric(work[term], errors="coerce")
    if work[list(requested_terms)].isna().any(axis=None):
        bad = [term for term in requested_terms if work[term].isna().any()]
        raise ValueError(f"Numeric model terms contain missing or non-numeric values: {bad}")
    if not np.isfinite(work[list(requested_terms)].to_numpy(dtype=float)).all():
        raise ValueError("Numeric model terms must be finite")
    if work[["choice_set_id", "candidate_author_id", "actor_id"]].isna().any(axis=None):
        raise ValueError("Choice, actor, and candidate-author identifiers cannot be missing")

    choice_order = tuple(str(value) for value in pd.unique(work["choice_set_id"]))
    choice_lookup = {choice_id: index for index, choice_id in enumerate(choice_order)}
    work["_choice_label"] = work["choice_set_id"].astype(str)
    work["_choice_order"] = work["_choice_label"].map(choice_lookup)
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["_choice_order", "_row_order"], kind="stable")

    parts: list[pd.DataFrame] = []
    for choice_id, part in work.groupby("_choice_label", sort=False, dropna=False):
        _validate_choice_set(part, str(choice_id))
        parts.append(part)
    if not parts:
        raise ValueError("No valid choice sets remain")

    receiver_row_index, receiver_labels = _stable_index(work["_receiver_label"].tolist())
    work["_receiver_index"] = receiver_row_index
    receiver_is_ai = _receiver_types(work, receiver_labels)
    receiver_groups = _groups_for_labels(receiver_labels)
    actor_row_index, actor_labels = _stable_index(work["_actor_label"].tolist())
    work["_actor_index"] = actor_row_index
    actor_groups = _groups_for_labels(actor_labels)
    topic_row_index, topic_labels = _stable_index(work["_topic_label"].tolist())
    work["_topic_index"] = topic_row_index

    # Rebuild the groups after attaching integer indexes; groupby materializes
    # frames, so the validation copies above do not receive later columns.
    parts = [part for _, part in work.groupby("_choice_label", sort=False, dropna=False)]

    n_choices = len(parts)
    max_candidates = max(len(part) for part in parts)
    n_terms = len(requested_terms)
    x = np.zeros((n_choices, max_candidates, n_terms), dtype=np.float64)
    candidate_ai = np.zeros((n_choices, max_candidates), dtype=np.float64)
    mask = np.zeros((n_choices, max_candidates), dtype=bool)
    chosen_position = np.zeros(n_choices, dtype=np.int64)
    receiver_index = np.zeros((n_choices, max_candidates), dtype=np.int64)
    actor_index = np.zeros(n_choices, dtype=np.int64)
    topic_index = np.zeros(n_choices, dtype=np.int64)
    choice_groups: list[str] = []

    for choice_index, part in enumerate(parts):
        candidate_count = len(part)
        x[choice_index, :candidate_count] = part[list(requested_terms)].to_numpy(dtype=float)
        candidate_ai[choice_index, :candidate_count] = part["_candidate_ai"].to_numpy(dtype=float)
        mask[choice_index, :candidate_count] = True
        receiver_index[choice_index, :candidate_count] = part["_receiver_index"].to_numpy(dtype=np.int64)
        chosen_position[choice_index] = int(np.flatnonzero(part["_chosen"].to_numpy())[0])
        actor_index[choice_index] = int(part["_actor_index"].iloc[0])
        topic_index[choice_index] = int(part["_topic_index"].iloc[0])
        choice_groups.append(str(part["_choice_group"].iloc[0]))

    return PreparedModelData(
        x=x,
        candidate_is_ai=candidate_ai,
        mask=mask,
        chosen_position=chosen_position,
        receiver_index=receiver_index,
        receiver_is_ai=receiver_is_ai,
        receiver_groups=receiver_groups,
        actor_index=actor_index,
        actor_groups=actor_groups,
        topic_index=topic_index,
        terms=requested_terms,
        term_families=term_families,
        ai_term=resolved_ai,
        choice_ids=tuple(str(part["_choice_label"].iloc[0]) for part in parts),
        receiver_labels=receiver_labels,
        actor_labels=actor_labels,
        topic_labels=topic_labels,
        choice_groups=tuple(choice_groups),
    )


def choice_utilities(
    data: PreparedModelData,
    beta: np.ndarray,
    *,
    receiver_effect: np.ndarray | None = None,
    actor_ai_effect: np.ndarray | None = None,
    topic_ai_effect: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one set of model parameters with NumPy."""

    beta_values = np.asarray(beta, dtype=float)
    if beta_values.shape != (len(data.terms),):
        raise ValueError(f"beta must have shape {(len(data.terms),)}")
    utility = np.einsum("nkp,p->nk", data.x, beta_values)
    if receiver_effect is not None:
        values = np.asarray(receiver_effect, dtype=float)
        if values.shape != (len(data.receiver_labels),):
            raise ValueError("receiver_effect has the wrong shape")
        utility += values[data.receiver_index]
    if actor_ai_effect is not None:
        values = np.asarray(actor_ai_effect, dtype=float)
        if values.shape != (len(data.actor_labels),):
            raise ValueError("actor_ai_effect has the wrong shape")
        utility += values[data.actor_index, None] * data.candidate_is_ai
    if topic_ai_effect is not None:
        values = np.asarray(topic_ai_effect, dtype=float)
        if values.shape != (len(data.topic_labels),):
            raise ValueError("topic_ai_effect has the wrong shape")
        utility += values[data.topic_index, None] * data.candidate_is_ai
    return np.where(data.mask, utility, -np.inf)


def choice_log_likelihood(data: PreparedModelData, utility: np.ndarray) -> np.ndarray:
    """Return one conditional-softmax log likelihood per choice set."""

    values = np.asarray(utility, dtype=float)
    if values.shape != data.mask.shape:
        raise ValueError(f"utility must have shape {data.mask.shape}")
    if np.isfinite(values[~data.mask]).any():
        raise ValueError("Padded candidate utilities must be non-finite")
    row_max = np.max(values, axis=1)
    log_denominator = row_max + np.log(np.sum(np.exp(values - row_max[:, None]), axis=1))
    chosen = values[np.arange(data.n_choices), data.chosen_position]
    return chosen - log_denominator


def _model_spec(value: str | ModelSpec) -> ModelSpec:
    if isinstance(value, ModelSpec):
        return value
    try:
        return MODEL_SPECS[value]
    except KeyError as error:
        raise ValueError(f"Unknown model specification {value!r}; choose from {tuple(MODEL_SPECS)}") from error


def _centered_effect(pm: Any, pt: Any, name: str, sigma_prior: float, dimension: str) -> Any:
    scale = pm.HalfNormal(f"sigma_{name}", sigma=sigma_prior)
    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=1.0, dims=dimension)
    return pm.Deterministic(name, scale * (raw - pt.mean(raw)), dims=dimension)


def _centered_effect_within_groups(
    pm: Any,
    pt: Any,
    name: str,
    sigma_prior: float,
    dimension: str,
    entity_groups: Sequence[str],
) -> Any:
    """Create deviations constrained to average zero separately within each arm."""

    scale = pm.HalfNormal(f"sigma_{name}", sigma=sigma_prior)
    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=1.0, dims=dimension)
    centered = pt.zeros_like(raw)
    group_array = np.asarray(entity_groups, dtype=object)
    for group in sorted(set(entity_groups)):
        indexes = np.flatnonzero(group_array == group)
        values = raw[indexes]
        centered = pt.set_subtensor(centered[indexes], values - pt.mean(values))
    return pm.Deterministic(name, scale * centered, dims=dimension)


def _centered_receiver_effect(
    pm: Any,
    pt: Any,
    data: PreparedModelData,
    sigma_prior: float,
    receiver_type_index: Any,
) -> Any:
    scale = pm.HalfNormal("sigma_receiver_effect", sigma=sigma_prior, dims="author_type")
    raw = pm.Normal("receiver_effect_raw", mu=0.0, sigma=1.0, dims="receiver")
    centered = pt.zeros_like(raw)
    receiver_groups = np.asarray(data.receiver_groups, dtype=object)
    for group in sorted(set(data.receiver_groups)):
        for author_type in (0, 1):
            indexes = np.flatnonzero(
                (receiver_groups == group) & (data.receiver_is_ai == author_type)
            )
            if indexes.size:
                values = raw[indexes]
                centered = pt.set_subtensor(centered[indexes], values - pt.mean(values))
    return pm.Deterministic(
        "receiver_effect",
        scale[receiver_type_index] * centered,
        dims="receiver",
    )


def _selected_term_indices(data: PreparedModelData, spec: ModelSpec) -> tuple[int, ...]:
    selected = tuple(
        index
        for index, family in enumerate(data.term_families)
        if family in spec.fixed_families
    )
    if not selected:
        raise ValueError(f"{spec.name} has no terms from its prespecified fixed-effect families")
    return selected


def _add_latent_block_effects(
    utility: Any,
    data: PreparedModelData,
    receiver_index: Any,
    latent_k: int,
    network_term_indices: Sequence[int],
    prior_scale: float,
) -> Any:
    import pymc as pm
    import pytensor.tensor as pt

    block_weights = pm.Dirichlet("block_weights", a=np.ones(latent_k), dims="block")
    receiver_block = pm.Categorical("receiver_block", p=block_weights, dims="receiver")
    raw_intercept = pm.Normal("block_intercept_raw", mu=0.0, sigma=1.0, dims="block")
    ordered_intercept = pt.sort(raw_intercept)
    block_intercept = pm.Deterministic(
        "block_intercept",
        prior_scale * (ordered_intercept - pt.mean(ordered_intercept)),
        dims="block",
    )
    candidate_block = receiver_block[receiver_index]
    utility += block_intercept[candidate_block]
    if not network_term_indices:
        return utility

    network_values = data.x[:, :, list(network_term_indices)]
    network_x = pm.Data(
        "network_x",
        network_values,
        dims=("choice_set", "candidate_position", "network_term"),
    )
    block_network_scale = pm.HalfNormal(
        "sigma_block_network",
        sigma=prior_scale,
        dims="network_term",
    )
    block_network_raw = pm.Normal(
        "block_network_raw",
        mu=0.0,
        sigma=1.0,
        dims=("block", "network_term"),
    )
    block_network_slope = pm.Deterministic(
        "block_network_slope",
        block_network_scale[None, :]
        * (block_network_raw - pt.mean(block_network_raw, axis=0, keepdims=True)),
        dims=("block", "network_term"),
    )
    return utility + pt.sum(network_x * block_network_slope[candidate_block], axis=2)


def _build_model(
    data: PreparedModelData,
    config: AnalysisConfig,
    spec: ModelSpec,
    *,
    latent_k: int | None = None,
    network_term_indices: Sequence[int] = (),
) -> Any:
    import pymc as pm
    import pytensor.tensor as pt

    fixed_term_indices = _selected_term_indices(data, spec)
    fixed_terms = tuple(data.terms[index] for index in fixed_term_indices)
    fixed_families = tuple(data.term_families[index] for index in fixed_term_indices)
    coords: dict[str, Sequence[Any]] = {
        "choice_set": data.choice_ids,
        "candidate_position": tuple(range(data.max_candidates)),
        "term": fixed_terms,
        "receiver": data.receiver_labels,
        "actor": data.actor_labels,
        "topic": data.topic_labels,
        "author_type": ("H", "AI"),
    }
    if latent_k is not None:
        coords["block"] = tuple(range(latent_k))
        coords["network_term"] = tuple(data.terms[index] for index in network_term_indices)

    prior_scales = np.asarray(
        [
            config.sampling.prior_focal_sigma
            if family == "focal"
            else config.sampling.prior_adjustment_sigma
            for family in fixed_families
        ],
        dtype=float,
    )
    with pm.Model(coords=coords) as model:
        x = pm.Data(
            "x",
            data.x[:, :, list(fixed_term_indices)],
            dims=("choice_set", "candidate_position", "term"),
        )
        candidate_ai = pm.Data(
            "candidate_is_ai", data.candidate_is_ai, dims=("choice_set", "candidate_position")
        )
        mask = pm.Data("candidate_mask", data.mask, dims=("choice_set", "candidate_position"))
        receiver_index = pm.Data(
            "receiver_index", data.receiver_index, dims=("choice_set", "candidate_position")
        )
        receiver_type_index = pm.Data(
            "receiver_type_index", data.receiver_is_ai, dims="receiver"
        )
        actor_index = pm.Data("actor_index", data.actor_index, dims="choice_set")
        topic_index = pm.Data("topic_index", data.topic_index, dims="choice_set")
        chosen_position = pm.Data("chosen_position", data.chosen_position, dims="choice_set")

        beta = pm.Normal("beta", mu=0.0, sigma=prior_scales, dims="term")
        utility = pt.sum(x * beta[None, None, :], axis=2)

        if spec.receiver_frailty and len(data.receiver_labels) > 1:
            receiver_effect = _centered_receiver_effect(
                pm,
                pt,
                data,
                config.sampling.prior_scale_sigma,
                receiver_type_index,
            )
            utility += receiver_effect[receiver_index]
        if spec.participant_ai_slope and len(data.actor_labels) > 1:
            actor_ai_effect = _centered_effect_within_groups(
                pm,
                pt,
                "actor_ai_effect",
                config.sampling.prior_scale_sigma,
                "actor",
                data.actor_groups,
            )
            utility += actor_ai_effect[actor_index, None] * candidate_ai
        if spec.topic_ai_slope and len(data.topic_labels) > 1:
            topic_ai_effect = _centered_effect(
                pm, pt, "topic_ai_effect", config.sampling.prior_scale_sigma, "topic"
            )
            utility += topic_ai_effect[topic_index, None] * candidate_ai

        if latent_k is not None:
            utility = _add_latent_block_effects(
                utility,
                data,
                receiver_index,
                latent_k,
                network_term_indices,
                config.sampling.prior_scale_sigma,
            )

        masked_utility = pt.switch(mask, utility, -1.0e30)
        selected = masked_utility[pt.arange(data.n_choices), chosen_position]
        choice_logp = selected - pm.math.logsumexp(masked_utility, axis=1)
        pm.Deterministic("choice_log_likelihood", choice_logp, dims="choice_set")
        pm.Potential("choice_likelihood", pt.sum(choice_logp))
    return model


def build_primary_pymc_model(
    data: PreparedModelData,
    config: AnalysisConfig,
    *,
    model_spec: str | ModelSpec = "M3",
) -> Any:
    """Build an M0--M3 PyMC model without starting a sampler."""

    return _build_model(data, config, _model_spec(model_spec))


def _attach_log_likelihood(idata: Any) -> Any:
    import xarray as xr

    if "choice_log_likelihood" not in idata.posterior:
        raise ValueError("Posterior is missing the stored per-choice log likelihood")
    values = idata.posterior["choice_log_likelihood"].rename("choice")
    dataset = xr.Dataset({"choice": values})
    if "log_likelihood" in idata.groups():
        idata.log_likelihood["choice"] = values
    else:
        idata.add_groups({"log_likelihood": dataset})
    return idata


def fit_model(
    model: Any,
    config: AnalysisConfig,
    *,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
    progressbar: bool = True,
    **sample_kwargs: Any,
) -> Any:
    """Sample a prepared M0--M3 model and attach its pointwise likelihood."""

    import pymc as pm

    sampling = config.sampling
    with model:
        idata = pm.sample(
            draws=sampling.draws if draws is None else draws,
            tune=sampling.tune if tune is None else tune,
            chains=sampling.chains if chains is None else chains,
            target_accept=sampling.target_accept if target_accept is None else target_accept,
            random_seed=config.seed if random_seed is None else random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            **sample_kwargs,
        )
    return _attach_log_likelihood(idata)


def summarize_fit(
    idata: Any,
    *,
    var_names: Sequence[str] | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Return compact posterior and convergence summaries for model parameters."""

    import arviz as az

    available = set(idata.posterior.data_vars)
    requested = tuple(var_names) if var_names is not None else (
        "beta",
        "sigma_receiver_effect",
        "sigma_actor_ai_effect",
        "sigma_topic_ai_effect",
        "block_weights",
        "block_intercept",
        "block_network_slope",
    )
    selected = [name for name in requested if name in available]
    if not selected:
        raise ValueError("None of the requested parameters are present in the posterior")
    summary = az.summary(idata, var_names=selected, hdi_prob=hdi_prob, kind="all")
    return summary.reset_index().rename(columns={"index": "parameter"})


def _posterior_array(idata: Any, variable: str, dimensions: Sequence[str]) -> np.ndarray | None:
    if variable not in idata.posterior:
        return None
    return (
        idata.posterior[variable]
        .stack(sample=("chain", "draw"))
        .transpose("sample", *dimensions)
        .values
    )


def _posterior_summary(values: np.ndarray, name: str, group: str) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    return {
        "estimand": name,
        "group": group,
        "posterior_mean": float(np.mean(clean)),
        "posterior_median": float(np.median(clean)),
        "ci_low": float(np.quantile(clean, 0.025)),
        "ci_high": float(np.quantile(clean, 0.975)),
        "posterior_probability_gt_zero": float(np.mean(clean > 0.0)),
        "posterior_draws": int(clean.size),
    }


def _posterior_utilities(
    idata: Any,
    data: PreparedModelData,
    *,
    x_override: np.ndarray | None = None,
) -> np.ndarray:
    x_source = data.x if x_override is None else np.asarray(x_override, dtype=float)
    if x_source.shape != data.x.shape:
        raise ValueError(f"x_override must have shape {data.x.shape}")
    beta_da = idata.posterior["beta"]
    model_terms = tuple(str(value) for value in beta_da.coords["term"].values)
    missing = sorted(set(model_terms) - set(data.terms))
    if missing:
        raise ValueError(f"Posterior beta contains terms absent from prepared data: {missing}")
    term_indices = [data.terms.index(term) for term in model_terms]
    beta = beta_da.stack(sample=("chain", "draw")).transpose("sample", "term").values
    utility = np.einsum("nkp,sp->snk", x_source[:, :, term_indices], beta)

    receiver = _posterior_array(idata, "receiver_effect", ("receiver",))
    if receiver is not None:
        utility += receiver[:, data.receiver_index]
    actor_ai = _posterior_array(idata, "actor_ai_effect", ("actor",))
    if actor_ai is not None:
        utility += actor_ai[:, data.actor_index, None] * data.candidate_is_ai[None, :, :]
    topic_ai = _posterior_array(idata, "topic_ai_effect", ("topic",))
    if topic_ai is not None:
        utility += topic_ai[:, data.topic_index, None] * data.candidate_is_ai[None, :, :]

    receiver_block = _posterior_array(idata, "receiver_block", ("receiver",))
    block_intercept = _posterior_array(idata, "block_intercept", ("block",))
    if receiver_block is not None and block_intercept is not None:
        candidate_block = receiver_block[:, data.receiver_index].astype(np.int64)
        sample_index = np.arange(candidate_block.shape[0])[:, None, None]
        utility += block_intercept[sample_index, candidate_block]
        block_slope = _posterior_array(idata, "block_network_slope", ("block", "network_term"))
        if block_slope is not None:
            network_terms = tuple(str(value) for value in idata.posterior.coords["network_term"].values)
            term_indices = [data.terms.index(term) for term in network_terms]
            candidate_slope = block_slope[sample_index, candidate_block, :]
            network_x = x_source[:, :, term_indices]
            utility += np.sum(network_x[None, :, :, :] * candidate_slope, axis=3)
    return np.where(data.mask[None, :, :], utility, -np.inf)


def _choice_probabilities(utility: np.ndarray, data: PreparedModelData) -> np.ndarray:
    row_max = np.max(utility, axis=2, keepdims=True)
    weights = np.exp(utility - row_max) * data.mask[None, :, :]
    return weights / np.sum(weights, axis=2, keepdims=True)


def _ai_choice_probability(utility: np.ndarray, data: PreparedModelData) -> np.ndarray:
    probabilities = _choice_probabilities(utility, data)
    return np.sum(probabilities * data.candidate_is_ai[None, :, :], axis=2)


def _direct_ai_arm_frames(data: PreparedModelData) -> tuple[np.ndarray, np.ndarray]:
    g1 = data.x.copy()
    g2 = data.x.copy()
    for index, term in enumerate(data.terms):
        if _is_arm_ai_interaction(term):
            g1[:, :, index] = 0.0
            g2[:, :, index] = data.candidate_is_ai
    return g1, g2


def _common_arm_frames(data: PreparedModelData) -> tuple[np.ndarray, np.ndarray]:
    """Switch every declared G2 interaction on one pooled opportunity stream."""

    g1, g2 = _direct_ai_arm_frames(data)
    prefix = "g2_x_"
    for index, term in enumerate(data.terms):
        if not term.startswith(prefix) or _is_arm_ai_interaction(term):
            continue
        base_term = term.removeprefix(prefix)
        if base_term not in data.terms:
            raise ValueError(f"Arm interaction {term!r} has no model term {base_term!r}")
        base_index = data.terms.index(base_term)
        g1[:, :, index] = 0.0
        g2[:, :, index] = data.x[:, :, base_index]
    return g1, g2


def _average_choice_probability(
    values: np.ndarray,
    data: PreparedModelData,
    *,
    selected: np.ndarray | None = None,
    equal_participant_weight: bool = False,
) -> np.ndarray:
    include = np.ones(data.n_choices, dtype=bool) if selected is None else selected.astype(bool)
    if not include.any():
        raise ValueError("Standardized probability has no eligible choice sets")
    if not equal_participant_weight:
        return np.mean(values[:, include], axis=1)
    actor_indexes = np.unique(data.actor_index[include])
    actor_means = [
        np.mean(values[:, include & (data.actor_index == actor_index)], axis=1)
        for actor_index in actor_indexes
    ]
    return np.mean(np.stack(actor_means, axis=1), axis=1)


def _counterfactual_terms(
    idata: Any,
    data: PreparedModelData,
    family: str,
) -> list[tuple[int, str]]:
    posterior_terms = {str(value) for value in idata.posterior["beta"].coords["term"].values}
    return [
        (index, term)
        for index, (term, term_family) in enumerate(zip(data.terms, data.term_families))
        if term_family == family and term in posterior_terms
    ]


def _counterfactual_draws(
    idata: Any,
    data: PreparedModelData,
    arm_x: np.ndarray,
    term_values: Mapping[int, tuple[float, float]],
) -> np.ndarray:
    low = arm_x.copy()
    high = arm_x.copy()
    ai_candidates = data.candidate_is_ai.astype(bool) & data.mask
    for term_index, (low_value, high_value) in term_values.items():
        low[:, :, term_index][ai_candidates] = low_value
        high[:, :, term_index][ai_candidates] = high_value
    low_probability = _ai_choice_probability(
        _posterior_utilities(idata, data, x_override=low),
        data,
    )
    high_probability = _ai_choice_probability(
        _posterior_utilities(idata, data, x_override=high),
        data,
    )
    return np.mean(high_probability - low_probability, axis=1)


def _counterfactual_rows(
    idata: Any,
    data: PreparedModelData,
    g1_x: np.ndarray,
    g2_x: np.ndarray,
    counterfactual_values: Mapping[str, tuple[float, float]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = {
        "repetition": "AI_candidate_prior_contact_zero_to_one_probability_difference",
        "reverse": "AI_candidate_recent_reverse_zero_to_one_probability_difference",
    }
    for family, estimand in names.items():
        selected = _counterfactual_terms(idata, data, family)
        if not selected:
            continue
        values = {
            term_index: (
                counterfactual_values.get(term, (0.0, 1.0))
                if counterfactual_values is not None
                else (0.0, 1.0)
            )
            for term_index, term in selected
        }
        g1_draws = _counterfactual_draws(idata, data, g1_x, values)
        g2_draws = _counterfactual_draws(idata, data, g2_x, values)
        source_terms = [term for _, term in selected]
        projection = "; ".join(
            f"{term} {values[index][0]:.6g} to {values[index][1]:.6g}"
            for index, term in selected
        )
        for group, draws in (("G1", g1_draws), ("G2", g2_draws), ("G2-G1", g2_draws - g1_draws)):
            row = _posterior_summary(draws, estimand, group)
            row.update(
                {
                    "source_term": ",".join(source_terms),
                    "standardization_scope": "pooled_common_candidate_sets",
                    "counterfactual": (
                        f"Joint AI-candidate model projection ({projection}); "
                        "human candidates and other covariates held observed"
                    ),
                    "estimand_interpretation": "model_projection_association_not_causal_intervention",
                }
            )
            rows.append(row)
    return rows


def standardized_estimands(
    idata: Any,
    data: PreparedModelData,
    *,
    counterfactual_values: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Estimate arm and network contrasts on one pooled opportunity stream."""

    utility = _posterior_utilities(idata, data)
    ai_probability = _ai_choice_probability(utility, data)

    rows: list[dict[str, Any]] = []
    groups = np.asarray(data.choice_groups, dtype=object)
    for group in tuple(dict.fromkeys(data.choice_groups)):
        selected = groups == group
        if selected.any():
            draws = np.mean(ai_probability[:, selected], axis=1)
            row = _posterior_summary(draws, "observed_risk_set_ai_choice_probability", group)
            row["standardization_scope"] = "group_observed_candidate_sets"
            rows.append(row)

    g1_x, g2_x = _common_arm_frames(data)
    g1_choice_probability = _ai_choice_probability(
        _posterior_utilities(idata, data, x_override=g1_x), data
    )
    g2_choice_probability = _ai_choice_probability(
        _posterior_utilities(idata, data, x_override=g2_x), data
    )
    g1_probability = _average_choice_probability(g1_choice_probability, data)
    g2_probability = _average_choice_probability(g2_choice_probability, data)
    common_estimand = "common_opportunity_ai_choice_probability"
    for group, draws in (("G1", g1_probability), ("G2", g2_probability)):
        row = _posterior_summary(draws, common_estimand, group)
        row["standardization_scope"] = "pooled_common_candidate_sets"
        row["projection_scope"] = "full_arm_including_content_selection_interactions"
        row["weighting"] = "reply_event_weighted"
        rows.append(row)
    contrast = _posterior_summary(
        g2_probability - g1_probability,
        "G2_minus_G1_common_opportunity_ai_choice_probability",
        "G2-G1",
    )
    contrast["standardization_scope"] = "pooled_common_candidate_sets"
    contrast["projection_scope"] = "full_arm_including_content_selection_interactions"
    contrast["weighting"] = "reply_event_weighted"
    rows.append(contrast)

    direct_g1_x, direct_g2_x = _direct_ai_arm_frames(data)
    direct_g1 = _average_choice_probability(
        _ai_choice_probability(_posterior_utilities(idata, data, x_override=direct_g1_x), data),
        data,
    )
    direct_g2 = _average_choice_probability(
        _ai_choice_probability(_posterior_utilities(idata, data, x_override=direct_g2_x), data),
        data,
    )
    direct_name = "direct_G2xAI_common_opportunity_probability"
    for group, draws in (("G1", direct_g1), ("G2", direct_g2), ("G2-G1", direct_g2 - direct_g1)):
        row = _posterior_summary(draws, direct_name, group)
        row["standardization_scope"] = "pooled_common_candidate_sets"
        row["projection_scope"] = "toggle_G2xAI_only_hold_nuisance_interactions_observed"
        row["weighting"] = "reply_event_weighted"
        rows.append(row)

    has_ai = np.any(data.candidate_is_ai.astype(bool) & data.mask, axis=1)
    has_human = np.any((~data.candidate_is_ai.astype(bool)) & data.mask, axis=1)
    mixed = has_ai & has_human
    for weighting, equal_weight in (
        ("reply_event_weighted_mixed_sets", False),
        ("equal_participant_weight_mixed_sets", True),
    ):
        mixed_g1 = _average_choice_probability(
            g1_choice_probability,
            data,
            selected=mixed,
            equal_participant_weight=equal_weight,
        )
        mixed_g2 = _average_choice_probability(
            g2_choice_probability,
            data,
            selected=mixed,
            equal_participant_weight=equal_weight,
        )
        name = (
            "equal_participant_mixed_risk_set_ai_choice_probability"
            if equal_weight
            else "mixed_risk_set_common_opportunity_ai_choice_probability"
        )
        for group, draws in (("G1", mixed_g1), ("G2", mixed_g2), ("G2-G1", mixed_g2 - mixed_g1)):
            row = _posterior_summary(draws, name, group)
            row["standardization_scope"] = "pooled_mixed_author_type_candidate_sets"
            row["projection_scope"] = "full_arm_including_content_selection_interactions"
            row["weighting"] = weighting
            rows.append(row)
    rows.extend(_counterfactual_rows(idata, data, g1_x, g2_x, counterfactual_values))
    return pd.DataFrame(rows)


def conditional_posterior_predictive_choices(
    idata: Any,
    data: PreparedModelData,
    *,
    draws: int | None = None,
    random_seed: int = 20260715,
) -> pd.DataFrame:
    """Draw targets conditional on the observed risk sets and network history.

    These draws do not recursively update network statistics after simulated
    choices. They assess the fitted target-choice distribution on the observed
    candidate sets and must not be described as fully generative event streams.
    """

    utility = _posterior_utilities(idata, data)
    available_draws = utility.shape[0]
    if draws is not None and draws <= 0:
        raise ValueError("draws must be positive when supplied")
    rng = np.random.default_rng(random_seed)
    if draws is None or draws >= available_draws:
        selected_draws = np.arange(available_draws, dtype=np.int64)
    else:
        selected_draws = np.sort(rng.choice(available_draws, size=draws, replace=False))
        utility = utility[selected_draws]

    row_max = np.max(utility, axis=2, keepdims=True)
    weights = np.exp(utility - row_max) * data.mask[None, :, :]
    probabilities = weights / np.sum(weights, axis=2, keepdims=True)
    uniforms = rng.random((len(selected_draws), data.n_choices))
    cumulative = np.cumsum(probabilities, axis=2)
    chosen_position = np.sum(uniforms[:, :, None] > cumulative, axis=2)
    chosen_position = np.minimum(chosen_position, data.max_candidates - 1).astype(np.int64)

    draw_index = np.arange(len(selected_draws))[:, None]
    choice_index = np.arange(data.n_choices)[None, :]
    receiver_index = data.receiver_index[choice_index, chosen_position]
    chosen_ai = data.candidate_is_ai[choice_index, chosen_position]
    chosen_probability = probabilities[draw_index, choice_index, chosen_position]
    receiver_labels = np.asarray(data.receiver_labels, dtype=object)[receiver_index]

    return pd.DataFrame(
        {
            "posterior_draw_index": np.repeat(selected_draws, data.n_choices),
            "choice_set_id": np.tile(np.asarray(data.choice_ids, dtype=object), len(selected_draws)),
            "group": np.tile(np.asarray(data.choice_groups, dtype=object), len(selected_draws)),
            "chosen_position": chosen_position.ravel(),
            "chosen_receiver": receiver_labels.ravel(),
            "chosen_is_ai": chosen_ai.astype(bool).ravel(),
            "chosen_probability": chosen_probability.ravel(),
            "simulation_scope": "conditional_on_observed_risk_sets_and_history",
        }
    )


def pointwise_log_scores(idata: Any, data: PreparedModelData | None = None) -> Any:
    """Expose raw chain-by-draw-by-choice log likelihood for PSIS or validation."""

    if "log_likelihood" in idata.groups() and "choice" in idata.log_likelihood:
        values = idata.log_likelihood["choice"]
    elif "choice_log_likelihood" in idata.posterior:
        values = idata.posterior["choice_log_likelihood"].rename("choice")
    else:
        raise ValueError("InferenceData contains no per-choice log likelihood")
    if data is not None:
        if values.sizes.get("choice_set") != data.n_choices:
            raise ValueError("Stored pointwise likelihood does not match the prepared choice sets")
        values = values.assign_coords(choice_set=list(data.choice_ids))
    return values.copy()


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    weights = np.exp(centered)
    return weights / weights.sum()


def simulate_latent_block_recovery(
    *,
    K: int = 2,
    n_actors: int = 12,
    n_receivers: int = 24,
    n_choice_sets: int = 240,
    candidates_per_set: int = 5,
    ai_coefficient: float = 0.35,
    block_intercepts: Sequence[float] | None = None,
    block_network_slopes: Sequence[float] | None = None,
    random_seed: int = 20260715,
) -> LatentBlockRecoveryInput:
    """Create a fast recovery fixture with known, fixed receiver blocks.

    The generated prior-contact covariate is exogenous. This fixture tests
    whether the latent choice model can recover a partition at YSocial-like
    sizes; it is not a recursive simulation of a relational-event history.
    """

    if K not in (2, 3):
        raise ValueError("K must be 2 or 3")
    if n_actors < 2:
        raise ValueError("n_actors must be at least two")
    if n_receivers < K * 2:
        raise ValueError("n_receivers must provide at least two accounts per block")
    if n_choice_sets < 2:
        raise ValueError("n_choice_sets must be at least two")
    if candidates_per_set < 2 or candidates_per_set > n_receivers:
        raise ValueError("candidates_per_set must be between two and n_receivers")

    intercept = np.asarray(
        block_intercepts if block_intercepts is not None else np.linspace(-0.8, 0.8, K), dtype=float
    )
    slopes = np.asarray(
        block_network_slopes if block_network_slopes is not None else np.linspace(-0.7, 0.7, K),
        dtype=float,
    )
    if intercept.shape != (K,) or slopes.shape != (K,):
        raise ValueError("Block intercepts and network slopes must each have length K")

    rng = np.random.default_rng(random_seed)
    receiver_ids = np.arange(n_receivers, dtype=np.int64)
    receiver_block = receiver_ids % K
    receiver_is_ai = ((receiver_ids // K) % 2).astype(np.int8)
    receiver_frailty = rng.normal(0.0, 0.25, size=n_receivers)
    actor_ai_slope = rng.normal(0.0, 0.25, size=n_actors)
    topics = np.asarray(["fandom", "recommendation", "debate"], dtype=object)
    topic_ai_slope = {topic: value for topic, value in zip(topics, (-0.2, 0.3, 0.0))}

    rows: list[dict[str, Any]] = []
    for choice_index in range(n_choice_sets):
        actor = int(choice_index % n_actors)
        topic = str(topics[choice_index % len(topics)])
        candidates = rng.choice(receiver_ids, size=candidates_per_set, replace=False)
        prior_contact = rng.normal(0.0, 1.0, size=candidates_per_set)
        block = receiver_block[candidates]
        is_ai = receiver_is_ai[candidates]
        utility = (
            ai_coefficient * is_ai
            + receiver_frailty[candidates]
            + actor_ai_slope[actor] * is_ai
            + topic_ai_slope[topic] * is_ai
            + intercept[block]
            + slopes[block] * prior_contact
        )
        chosen = int(rng.choice(candidates_per_set, p=_softmax(utility)))
        for position, receiver in enumerate(candidates):
            rows.append(
                {
                    "choice_set_id": f"SIM:{choice_index}",
                    "group": "SIM",
                    "chosen": int(position == chosen),
                    "candidate_author_id": int(receiver),
                    "actor_id": actor,
                    "primary_root_topic": topic,
                    "candidate_target_is_ai": int(receiver_is_ai[receiver]),
                    "prior_contact": float(prior_contact[position]),
                }
            )

    truth = pd.DataFrame(
        {
            "candidate_author_id": receiver_ids,
            "receiver_label": [f"SIM::{receiver}" for receiver in receiver_ids],
            "true_block": receiver_block,
            "candidate_target_is_ai": receiver_is_ai,
            "receiver_frailty": receiver_frailty,
        }
    )
    parameters: dict[str, Any] = {
        "K": K,
        "ai_coefficient": float(ai_coefficient),
        "block_intercepts": intercept.tolist(),
        "block_network_slopes": slopes.tolist(),
        "network_term": "prior_contact",
        "random_seed": random_seed,
    }
    return LatentBlockRecoveryInput(pd.DataFrame(rows), truth, parameters)


def _adjusted_rand_index(truth: np.ndarray, estimated: np.ndarray) -> float:
    if truth.shape != estimated.shape or truth.ndim != 1:
        raise ValueError("Partition labels must be one-dimensional arrays with equal length")
    if truth.size < 2:
        raise ValueError("Partition recovery requires at least two receiver accounts")
    truth_levels, truth_inverse = np.unique(truth, return_inverse=True)
    estimated_levels, estimated_inverse = np.unique(estimated, return_inverse=True)
    contingency = np.zeros((len(truth_levels), len(estimated_levels)), dtype=np.int64)
    np.add.at(contingency, (truth_inverse, estimated_inverse), 1)

    def pairs(counts: np.ndarray) -> float:
        values = np.asarray(counts, dtype=float)
        return float(np.sum(values * (values - 1.0) / 2.0))

    cell_pairs = pairs(contingency)
    truth_pairs = pairs(contingency.sum(axis=1))
    estimated_pairs = pairs(contingency.sum(axis=0))
    total_pairs = truth.size * (truth.size - 1.0) / 2.0
    expected = truth_pairs * estimated_pairs / total_pairs
    maximum = 0.5 * (truth_pairs + estimated_pairs)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0 if np.array_equal(truth_inverse, estimated_inverse) else 0.0
    return float((cell_pairs - expected) / denominator)


def latent_block_recovery_gate(
    idata: Any,
    data: PreparedModelData,
    fixture: LatentBlockRecoveryInput,
    *,
    minimum_adjusted_rand: float = 0.80,
) -> Mapping[str, Any]:
    """Score modal receiver memberships against a known synthetic partition."""

    if not 0.0 <= minimum_adjusted_rand <= 1.0:
        raise ValueError("minimum_adjusted_rand must be between zero and one")
    if "receiver_block" not in idata.posterior:
        raise ValueError("Posterior is missing latent receiver-block memberships")
    block_da = idata.posterior["receiver_block"]
    if "receiver" in block_da.coords:
        block_da = block_da.sel(receiver=list(data.receiver_labels))
    memberships = (
        block_da.stack(sample=("chain", "draw"))
        .transpose("sample", "receiver")
        .values.astype(np.int64)
    )
    modal = np.asarray(
        [np.bincount(memberships[:, index]).argmax() for index in range(memberships.shape[1])],
        dtype=np.int64,
    )
    truth_table = fixture.receiver_truth
    if "receiver_label" not in truth_table:
        raise ValueError("Recovery fixture is missing receiver_label")
    truth_lookup = truth_table.set_index("receiver_label")["true_block"].to_dict()
    missing = [label for label in data.receiver_labels if label not in truth_lookup]
    if missing:
        raise ValueError(f"Recovery truth is missing receiver labels: {missing[:5]}")
    truth = np.asarray([truth_lookup[label] for label in data.receiver_labels], dtype=np.int64)
    score = _adjusted_rand_index(truth, modal)
    return {
        "simulation_scope": fixture.simulation_scope,
        "n_receivers": len(data.receiver_labels),
        "adjusted_rand_index": score,
        "minimum_adjusted_rand": float(minimum_adjusted_rand),
        "passed": bool(score >= minimum_adjusted_rand),
    }


def _network_term_indices(data: PreparedModelData, network_terms: Sequence[str] | None) -> tuple[int, ...]:
    if network_terms is not None:
        unknown = sorted(set(network_terms) - set(data.terms))
        if unknown:
            raise ValueError(f"Unknown latent-block network terms: {unknown}")
        selected = tuple(network_terms)
    else:
        selected = tuple(
            term
            for term in data.terms
            if term != data.ai_term and any(token in term.lower() for token in NETWORK_TERM_TOKENS)
        )
        if not selected:
            selected = tuple(term for term in data.terms if term != data.ai_term)
    return tuple(data.terms.index(term) for term in selected)


def fit_latent_receiver_blocks(
    data: PreparedModelData,
    config: AnalysisConfig,
    *,
    K: int,
    network_terms: Sequence[str] | None = None,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    target_accept: float | None = None,
    random_seed: int | None = None,
    progressbar: bool = True,
    **sample_kwargs: Any,
) -> Any:
    """Fit the optional finite receiver-block extension with compound sampling."""

    if K not in (2, 3):
        raise ValueError("K must be 2 or 3")
    if K > len(data.receiver_labels):
        raise ValueError("K cannot exceed the number of receiver accounts")

    import pymc as pm

    indices = _network_term_indices(data, network_terms)
    model = _build_model(data, config, MODEL_SPECS["M3"], latent_k=K, network_term_indices=indices)
    sampling = config.sampling
    with model:
        receiver_block = model["receiver_block"]
        continuous = [variable for variable in model.free_RVs if variable.name != "receiver_block"]
        steps = [
            pm.NUTS(
                vars=continuous,
                target_accept=sampling.target_accept if target_accept is None else target_accept,
            ),
            pm.CategoricalGibbsMetropolis(vars=[receiver_block]),
        ]
        idata = pm.sample(
            draws=sampling.draws if draws is None else draws,
            tune=sampling.tune if tune is None else tune,
            chains=sampling.chains if chains is None else chains,
            random_seed=config.seed if random_seed is None else random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            step=steps,
            **sample_kwargs,
        )
    return _attach_log_likelihood(idata)


__all__ = [
    "MODEL_SPECS",
    "LatentBlockRecoveryInput",
    "ModelSpec",
    "PreparedModelData",
    "build_primary_pymc_model",
    "choice_log_likelihood",
    "choice_utilities",
    "conditional_posterior_predictive_choices",
    "fit_latent_receiver_blocks",
    "fit_model",
    "latent_block_recovery_gate",
    "pointwise_log_scores",
    "prepare_model_data",
    "simulate_latent_block_recovery",
    "standardized_estimands",
    "summarize_fit",
]
