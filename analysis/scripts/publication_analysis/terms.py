"""Prespecified transformations for the relational-event choice models.

The raw history table remains the audit surface.  This module creates a small,
explicit set of finite candidate-level predictors and records each
transformation so fitted models never depend on an implicit preprocessing
formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TermDefinition:
    """Definition of one model-ready candidate utility term."""

    name: str
    family: str
    source: str
    description: str
    transform: Callable[[pd.DataFrame], pd.Series]
    focal: bool = False
    standardize: bool = True


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Network history is missing required column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce")


def _binary(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and numeric.isin([0, 1]).all():
        return numeric.astype(float)
    normalized = values.astype(str).str.upper().str.strip()
    mapped = normalized.map({"AI": 1.0, "H": 0.0, "TRUE": 1.0, "FALSE": 0.0})
    if mapped.isna().any():
        raise ValueError(f"{column!r} cannot be converted to a binary indicator")
    return mapped


def _signed_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def _log_candidate_age(frame: pd.DataFrame) -> pd.Series:
    return np.log1p(_numeric(frame, "candidate_age_hours").clip(lower=0.0))


def _log_candidate_depth(frame: pd.DataFrame) -> pd.Series:
    return np.log1p(_numeric(frame, "candidate_depth").clip(lower=0.0))


def _g2(frame: pd.DataFrame) -> pd.Series:
    return frame["group"].astype(str).eq("G2").astype(float)


def _recent_contact(frame: pd.DataFrame, count_column: str, time_column: str) -> pd.Series:
    counts = _numeric(frame, count_column).fillna(0.0)
    hours = _numeric(frame, time_column)
    if ((counts > 0) & hours.isna()).any():
        raise ValueError(f"{time_column!r} is missing where prior contact exists")
    # Absence of a prior network event is a true structural zero.  It differs
    # from missing browser observation, which is never processed here.
    return np.where(counts > 0, np.exp(-hours.clip(lower=0.0) / 24.0), 0.0)


TERM_DEFINITIONS: tuple[TermDefinition, ...] = (
    TermDefinition(
        "z_log_candidate_age",
        "opportunity_content",
        "candidate_age_hours",
        "Log age of the eligible content item on the valid-analysis clock.",
        _log_candidate_age,
    ),
    TermDefinition(
        "z_log_candidate_age_squared",
        "opportunity_content",
        "candidate_age_hours",
        "Squared log age, allowing a curved recency-selection relationship.",
        lambda frame: _numeric(frame, "z_log_candidate_age") ** 2,
    ),
    TermDefinition(
        "z_log_candidate_depth",
        "opportunity_content",
        "candidate_depth",
        "Log depth of the eligible item in its comment tree.",
        _log_candidate_depth,
    ),
    TermDefinition(
        "candidate_is_root",
        "opportunity_content",
        "candidate_is_root",
        "Eligible content is the root post rather than a comment.",
        lambda frame: _binary(frame, "candidate_is_root"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_z_log_candidate_age",
        "opportunity_content",
        "group x candidate_age_hours",
        "Hidden-label-arm deviation in the log-age selection slope.",
        lambda frame: _g2(frame) * _numeric(frame, "z_log_candidate_age"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_z_log_candidate_age_squared",
        "opportunity_content",
        "group x candidate_age_hours squared",
        "Hidden-label-arm deviation in the curved log-age selection slope.",
        lambda frame: _g2(frame) * _numeric(frame, "z_log_candidate_age_squared"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_z_log_candidate_depth",
        "opportunity_content",
        "group x candidate_depth",
        "Hidden-label-arm deviation in the log-depth selection slope.",
        lambda frame: _g2(frame) * _numeric(frame, "z_log_candidate_depth"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_is_root",
        "opportunity_content",
        "group x candidate_is_root",
        "Hidden-label-arm deviation for selecting a root post.",
        lambda frame: _g2(frame) * _binary(frame, "candidate_is_root"),
        standardize=False,
    ),
    TermDefinition(
        "z_content_reaction_status",
        "opportunity_content",
        "candidate_content_prior_reaction_score",
        "Signed log prior vote score of the content item.",
        lambda frame: _signed_log1p(_numeric(frame, "candidate_content_prior_reaction_score")),
    ),
    TermDefinition(
        "z_author_reaction_status",
        "opportunity_content",
        "candidate_author_prior_reaction_score",
        "Signed log cumulative prior vote score of the content author.",
        lambda frame: _signed_log1p(_numeric(frame, "candidate_author_prior_reaction_score")),
    ),
    TermDefinition(
        "candidate_target_is_ai",
        "rq1_focal",
        "candidate_author_type",
        "Eligible content was authored by an observed AI account.",
        lambda frame: _binary(frame, "candidate_author_type"),
        focal=True,
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_ai",
        "rq1_focal",
        "group x candidate_author_type",
        "AI-authored candidate in the hidden-label arm (G2).",
        lambda frame: _g2(frame) * _binary(frame, "candidate_author_type"),
        focal=True,
        standardize=False,
    ),
    TermDefinition(
        "z_log_prior_contact_count",
        "repetition_reciprocity",
        "prior_human_to_author_count",
        "Log number of earlier replies from the focal human to the candidate author.",
        lambda frame: np.log1p(_numeric(frame, "prior_human_to_author_count").clip(lower=0.0)),
    ),
    TermDefinition(
        "z_recent_prior_contact",
        "repetition_reciprocity",
        "prior_human_to_author_time_since_hours",
        "Twenty-four-hour decay of the most recent focal-human-to-author reply.",
        lambda frame: _recent_contact(
            frame,
            "prior_human_to_author_count",
            "prior_human_to_author_time_since_hours",
        ),
    ),
    TermDefinition(
        "z_log_reverse_contact_count",
        "repetition_reciprocity",
        "prior_author_to_human_count",
        "Log number of earlier replies from the candidate author to the focal human.",
        lambda frame: np.log1p(_numeric(frame, "prior_author_to_human_count").clip(lower=0.0)),
    ),
    TermDefinition(
        "z_recent_reverse_contact",
        "repetition_reciprocity",
        "prior_author_to_human_time_since_hours",
        "Twenty-four-hour decay of the most recent author-to-human reply.",
        lambda frame: _recent_contact(
            frame,
            "prior_author_to_human_count",
            "prior_author_to_human_time_since_hours",
        ),
    ),
    TermDefinition(
        "candidate_ai_x_z_log_prior_contact_count",
        "rq2_ai_history",
        "candidate_author_type x prior_human_to_author_count",
        "AI-candidate deviation in the log count of earlier focal-human-to-author replies.",
        lambda frame: _binary(frame, "candidate_author_type")
        * _numeric(frame, "z_log_prior_contact_count"),
        standardize=False,
    ),
    TermDefinition(
        "candidate_ai_x_z_recent_prior_contact",
        "rq2_ai_history",
        "candidate_author_type x prior_human_to_author_time_since_hours",
        "AI-candidate deviation in the recency of the focal human's latest reply to the author.",
        lambda frame: _binary(frame, "candidate_author_type")
        * _numeric(frame, "z_recent_prior_contact"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_ai_x_z_log_prior_contact_count",
        "rq2_ai_history",
        "group x candidate_author_type x prior_human_to_author_count",
        "G2 deviation in the AI-specific prior focal-human-to-author count association.",
        lambda frame: _g2(frame)
        * _numeric(frame, "candidate_ai_x_z_log_prior_contact_count"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_ai_x_z_recent_prior_contact",
        "rq2_ai_history",
        "group x candidate_author_type x prior_human_to_author_time_since_hours",
        "G2 deviation in the AI-specific prior focal-human-to-author recency association.",
        lambda frame: _g2(frame)
        * _numeric(frame, "candidate_ai_x_z_recent_prior_contact"),
        standardize=False,
    ),
    TermDefinition(
        "candidate_ai_x_z_log_reverse_contact_count",
        "rq2_ai_history",
        "candidate_author_type x prior_author_to_human_count",
        "AI-candidate deviation in the log count of earlier author-to-focal-human replies.",
        lambda frame: _binary(frame, "candidate_author_type")
        * _numeric(frame, "z_log_reverse_contact_count"),
        standardize=False,
    ),
    TermDefinition(
        "candidate_ai_x_z_recent_reverse_contact",
        "rq2_ai_history",
        "candidate_author_type x prior_author_to_human_time_since_hours",
        "AI-candidate deviation in the recency of the author's latest reply to the focal human.",
        lambda frame: _binary(frame, "candidate_author_type")
        * _numeric(frame, "z_recent_reverse_contact"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_ai_x_z_log_reverse_contact_count",
        "rq2_ai_history",
        "group x candidate_author_type x prior_author_to_human_count",
        "G2 deviation in the AI-specific prior author-to-focal-human count association.",
        lambda frame: _g2(frame)
        * _numeric(frame, "candidate_ai_x_z_log_reverse_contact_count"),
        standardize=False,
    ),
    TermDefinition(
        "g2_x_candidate_ai_x_z_recent_reverse_contact",
        "rq2_ai_history",
        "group x candidate_author_type x prior_author_to_human_time_since_hours",
        "G2 deviation in the AI-specific prior author-to-focal-human recency association.",
        lambda frame: _g2(frame)
        * _numeric(frame, "candidate_ai_x_z_recent_reverse_contact"),
        standardize=False,
    ),
    TermDefinition(
        "z_log_receiver_popularity",
        "popularity_closure",
        "receiver_popularity_prior_human_count",
        "Log number of earlier human replies received by the candidate author.",
        lambda frame: np.log1p(
            _numeric(frame, "receiver_popularity_prior_human_count").clip(lower=0.0)
        ),
    ),
    TermDefinition(
        "z_log_two_path_closure",
        "popularity_closure",
        "prior_two_path_count",
        "Log directed two-path count from the focal human to the candidate author.",
        lambda frame: np.log1p(_numeric(frame, "prior_two_path_count").clip(lower=0.0)),
    ),
)


MODEL_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "M0": ("opportunity_content",),
    "M1": ("opportunity_content", "rq1_focal"),
    "M2": ("opportunity_content", "rq1_focal", "repetition_reciprocity"),
    "RQ2": (
        "opportunity_content",
        "rq1_focal",
        "repetition_reciprocity",
        "rq2_ai_history",
    ),
    "M3": (
        "opportunity_content",
        "rq1_focal",
        "repetition_reciprocity",
        "popularity_closure",
    ),
}


def terms_for_model(model_name: str) -> tuple[str, ...]:
    """Return ordered model terms for one prespecified nested stage."""

    if model_name not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {tuple(MODEL_FAMILIES)}")
    families = set(MODEL_FAMILIES[model_name])
    return tuple(term.name for term in TERM_DEFINITIONS if term.family in families)


def included_terms_for_model(model_name: str, manifest: pd.DataFrame) -> tuple[str, ...]:
    """Return the prespecified terms whose pre-event sources are available."""

    required = {"term", "status"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Term manifest is missing columns: {missing}")
    available = set(
        manifest.loc[manifest["status"].astype(str).str.startswith("included"), "term"].astype(str)
    )
    return tuple(term for term in terms_for_model(model_name) if term in available)


def build_model_frame(
    network_history: pd.DataFrame,
    *,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create finite model terms and an auditable transformation manifest."""

    if network_history.empty:
        raise ValueError("Choice rows are empty")
    in_scope_terms = (
        set(terms_for_model(model_name)) if model_name is not None else None
    )
    frame = network_history.copy()
    manifest_rows: list[dict[str, object]] = []
    for definition in TERM_DEFINITIONS:
        if in_scope_terms is not None and definition.name not in in_scope_terms:
            frame[definition.name] = 0.0
            manifest_rows.append(
                {
                    "term": definition.name,
                    "family": definition.family,
                    "source": definition.source,
                    "description": definition.description,
                    "focal": definition.focal,
                    "standardized": definition.standardize,
                    "raw_mean": np.nan,
                    "raw_sd": np.nan,
                    "status": f"excluded_out_of_scope_{model_name}",
                }
            )
            continue
        try:
            transformed = definition.transform(frame)
        except ValueError as error:
            if "missing required column" not in str(error):
                raise
            transformed = pd.Series(np.nan, index=frame.index, dtype=float)
        raw = pd.Series(transformed, index=frame.index, dtype=float)
        missing = int(raw.isna().sum())
        if missing == len(raw):
            frame[definition.name] = 0.0
            manifest_rows.append(
                {
                    "term": definition.name,
                    "family": definition.family,
                    "source": definition.source,
                    "description": definition.description,
                    "focal": definition.focal,
                    "standardized": definition.standardize,
                    "raw_mean": np.nan,
                    "raw_sd": np.nan,
                    "status": "excluded_unavailable_source",
                }
            )
            continue
        if missing:
            raise ValueError(
                f"Model term {definition.name!r} has {missing} missing values after transformation"
            )
        if not np.isfinite(raw.to_numpy()).all():
            raise ValueError(f"Model term {definition.name!r} contains non-finite values")
        mean = float(raw.mean())
        standard_deviation = float(raw.std(ddof=0))
        if definition.standardize and standard_deviation > 0.0:
            frame[definition.name] = (raw - mean) / standard_deviation
            status = "included_standardized"
        elif definition.standardize:
            frame[definition.name] = 0.0
            status = "included_constant_as_zero"
        else:
            frame[definition.name] = raw
            status = "included_unscaled"
        manifest_rows.append(
            {
                "term": definition.name,
                "family": definition.family,
                "source": definition.source,
                "description": definition.description,
                "focal": definition.focal,
                "standardized": definition.standardize,
                "raw_mean": mean,
                "raw_sd": standard_deviation,
                "status": status,
            }
        )

    reputation_available = (
        "candidate_reputation_at_event" in frame.columns
        and pd.to_numeric(frame["candidate_reputation_at_event"], errors="coerce").notna().any()
    )
    manifest_rows.append(
        {
            "term": "candidate_reputation_at_event",
            "family": "opportunity_content",
            "source": "snapshot_user_reputation_history",
            "description": "Latest account reputation strictly before the focal event.",
            "focal": False,
            "standardized": True,
            "raw_mean": np.nan,
            "raw_sd": np.nan,
            "status": "available_not_prespecified" if reputation_available else "excluded_no_event_time_history",
        }
    )
    return frame, pd.DataFrame(manifest_rows)


def apply_model_term_manifest(
    network_history: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Apply training-set transformations to held-out candidate rows."""

    required = {"term", "status", "raw_mean", "raw_sd"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Term manifest is missing scaling columns: {missing}")
    settings = manifest.set_index("term", drop=False)
    frame = network_history.copy()
    for definition in TERM_DEFINITIONS:
        if definition.name not in settings.index:
            raise ValueError(f"Term manifest has no row for {definition.name!r}")
        row = settings.loc[definition.name]
        status = str(row["status"])
        if status.startswith("excluded") or status == "included_constant_as_zero":
            frame[definition.name] = 0.0
            continue
        raw = pd.Series(definition.transform(frame), index=frame.index, dtype=float)
        if raw.isna().any() or not np.isfinite(raw.to_numpy()).all():
            raise ValueError(f"Held-out term {definition.name!r} contains missing or non-finite values")
        if status == "included_standardized":
            mean = float(row["raw_mean"])
            standard_deviation = float(row["raw_sd"])
            if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
                raise ValueError(f"Invalid training scaler for {definition.name!r}")
            frame[definition.name] = (raw - mean) / standard_deviation
        elif status == "included_unscaled":
            frame[definition.name] = raw
        else:
            raise ValueError(f"Unknown term-manifest status {status!r}")
    return frame


__all__ = [
    "MODEL_FAMILIES",
    "TERM_DEFINITIONS",
    "TermDefinition",
    "apply_model_term_manifest",
    "build_model_frame",
    "included_terms_for_model",
    "terms_for_model",
]
