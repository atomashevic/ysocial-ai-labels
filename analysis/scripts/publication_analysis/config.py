"""Analysis paths, sampling, and run directories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SamplingConfig:
    draws: int = 1_000
    tune: int = 1_000
    chains: int = 4
    target_accept: float = 0.95
    prior_focal_sigma: float = 1.0
    prior_adjustment_sigma: float = 0.5
    prior_scale_sigma: float = 0.5


@dataclass(frozen=True)
class BrowserGateConfig:
    manual_audit_per_arm: int = 100
    interval_audit_per_arm: int | None = None
    min_manual_precision: float = 0.95
    min_pre_action_linkage: float = 0.90
    max_arm_linkage_gap: float = 0.05
    min_interval_closure: float = 0.90
    action_tolerance_seconds: float = 30.0


@dataclass(frozen=True)
class Rq3Config:
    horizons_hours: tuple[int, ...] = (24, 48)
    bootstrap_reps: int = 2_000
    round_start_time: str = "2026-05-09T16:00:00+02:00"
    first_experiment_round: int = 2


@dataclass(frozen=True)
class AnalysisConfig:
    project_root: Path
    processed_dir: Path
    browser_dir: Path
    legacy_tables_dir: Path | None
    output_root: Path
    run_id: str
    profile: str = "test"
    contract_sha256: str = ""
    source_manifest: Path | None = None
    analysis_stage: str = "rq1_m1"
    analysis_set: str = "primary_valid"
    start_time: str = "2026-05-11 08:00:00+02:00"
    end_time: str | None = None
    seed: int = 20260715
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    browser_gates: BrowserGateConfig = field(default_factory=BrowserGateConfig)
    latent_block_sizes: tuple[int, ...] = (2, 3)
    rq3: Rq3Config = field(default_factory=Rq3Config)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    name: str,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {name} keys: {unknown}")
    missing = sorted((required or set()).difference(value))
    if missing:
        raise ValueError(f"Missing {name} keys: {missing}")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_analysis(analysis: Mapping[str, Any]) -> None:
    _keys(
        analysis,
        name="analysis",
        allowed={
            "default_profile",
            "inputs",
            "output_root",
            "analysis_stage",
            "analysis_set",
            "window",
            "seed",
            "sampling",
            "browser_gates",
            "latent_block_sizes",
            "profiles",
            "rq3",
        },
        required={
            "default_profile",
            "inputs",
            "output_root",
            "analysis_stage",
            "analysis_set",
            "window",
            "seed",
            "sampling",
            "browser_gates",
            "latent_block_sizes",
            "profiles",
        },
    )
    inputs = _mapping(analysis["inputs"], "analysis.inputs")
    _keys(
        inputs,
        name="analysis.inputs",
        allowed={"processed_dir", "source_manifest", "browser_dir", "legacy_tables_dir"},
        required={"processed_dir", "source_manifest", "browser_dir", "legacy_tables_dir"},
    )
    window = _mapping(analysis["window"], "analysis.window")
    _keys(
        window,
        name="analysis.window",
        allowed={"start", "end", "valid_days"},
        required={"start", "end", "valid_days"},
    )
    sampling_keys = set(SamplingConfig.__dataclass_fields__)
    sampling = _mapping(analysis["sampling"], "analysis.sampling")
    _keys(sampling, name="analysis.sampling", allowed=sampling_keys, required=sampling_keys)
    gate_keys = set(BrowserGateConfig.__dataclass_fields__)
    gates = _mapping(analysis["browser_gates"], "analysis.browser_gates")
    _keys(gates, name="analysis.browser_gates", allowed=gate_keys, required=gate_keys)
    rq3_keys = set(Rq3Config.__dataclass_fields__)
    if "rq3" in analysis:
        _keys(_mapping(analysis["rq3"], "analysis.rq3"), name="analysis.rq3", allowed=rq3_keys)
    profiles = _mapping(analysis["profiles"], "analysis.profiles")
    if set(profiles) != {"pilot", "release"}:
        raise ValueError("analysis.profiles must contain exactly pilot and release")
    for name, profile_value in profiles.items():
        profile = _mapping(profile_value, f"analysis.profiles.{name}")
        _keys(
            profile,
            name=f"analysis.profiles.{name}",
            allowed={"sampling", "browser_gates", "latent_block_sizes", "rq3"},
        )
        if "rq3" in profile:
            _keys(
                _mapping(profile["rq3"], f"analysis.profiles.{name}.rq3"),
                name=f"analysis.profiles.{name}.rq3",
                allowed=rq3_keys,
            )
        if "sampling" in profile:
            _keys(
                _mapping(profile["sampling"], f"analysis.profiles.{name}.sampling"),
                name=f"analysis.profiles.{name}.sampling",
                allowed=sampling_keys,
            )
        if "browser_gates" in profile:
            _keys(
                _mapping(profile["browser_gates"], f"analysis.profiles.{name}.browser_gates"),
                name=f"analysis.profiles.{name}.browser_gates",
                allowed=gate_keys,
            )
    if analysis["default_profile"] not in profiles:
        raise ValueError("analysis.default_profile must name a declared profile")


def load_analysis_file(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("analysis config must contain a mapping")
    value = dict(raw)
    _validate_analysis(_mapping(value.get("analysis"), "analysis"))
    return value


def resolve_effective_config(path: Path, profile: str, run_id: str) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    loaded = load_analysis_file(config_path)
    analysis = loaded["analysis"]
    profiles = analysis["profiles"]
    if profile not in profiles:
        raise ValueError(f"Unknown analysis profile: {profile}")
    resolved_analysis = _merge(
        {key: value for key, value in analysis.items() if key not in {"profiles", "default_profile"}},
        profiles[profile],
    )
    return {
        "config_path": str(config_path),
        "contract_sha256": contract_sha256(config_path),
        "run_id": run_id,
        "profile": profile,
        "analysis": resolved_analysis,
    }


def load_config(path: Path, *, profile: str | None = None, run_id: str | None = None) -> AnalysisConfig:
    config_path = path.expanduser().resolve()
    loaded = load_analysis_file(config_path)
    selected_profile = profile or str(loaded["analysis"]["default_profile"])
    if not run_id:
        raise ValueError("run_id is required")
    effective = resolve_effective_config(config_path, selected_profile, run_id)
    analysis = effective["analysis"]
    root = config_path.parent
    inputs = analysis["inputs"]
    sampling = SamplingConfig(**dict(analysis["sampling"]))
    gates = BrowserGateConfig(**dict(analysis["browser_gates"]))
    latent_sizes = tuple(int(value) for value in analysis["latent_block_sizes"])
    if any(value < 2 for value in latent_sizes):
        raise ValueError("analysis.latent_block_sizes must contain integers greater than one")
    rq3_raw = dict(analysis.get("rq3") or {})
    if "horizons_hours" in rq3_raw:
        rq3_raw["horizons_hours"] = tuple(int(value) for value in rq3_raw["horizons_hours"])
    rq3 = Rq3Config(**rq3_raw)
    if not rq3.horizons_hours or any(value <= 0 for value in rq3.horizons_hours):
        raise ValueError("analysis.rq3.horizons_hours must contain positive integers")
    return AnalysisConfig(
        project_root=root,
        processed_dir=_resolve_path(root, inputs["processed_dir"]),
        browser_dir=_resolve_path(root, inputs["browser_dir"]),
        legacy_tables_dir=_resolve_path(root, inputs.get("legacy_tables_dir")),
        output_root=_resolve_path(root, analysis["output_root"]),
        run_id=run_id,
        profile=selected_profile,
        contract_sha256=effective["contract_sha256"],
        source_manifest=_resolve_path(root, inputs.get("source_manifest")),
        analysis_stage=str(analysis["analysis_stage"]),
        analysis_set=str(analysis["analysis_set"]),
        start_time=str(analysis["window"]["start"]),
        end_time=analysis["window"].get("end"),
        seed=int(analysis["seed"]),
        sampling=sampling,
        browser_gates=gates,
        latent_block_sizes=latent_sizes,
        rq3=rq3,
    )


def analysis_snapshot(config: AnalysisConfig) -> dict[str, Any]:
    return {
        "project_root": str(config.project_root),
        "processed_dir": str(config.processed_dir),
        "browser_dir": str(config.browser_dir),
        "legacy_tables_dir": str(config.legacy_tables_dir) if config.legacy_tables_dir else None,
        "output_root": str(config.output_root),
        "run_id": config.run_id,
        "profile": config.profile,
        "contract_sha256": config.contract_sha256,
        "source_manifest": str(config.source_manifest) if config.source_manifest else None,
        "analysis_stage": config.analysis_stage,
        "analysis_set": config.analysis_set,
        "start_time": config.start_time,
        "end_time": config.end_time,
        "seed": config.seed,
        "sampling": asdict(config.sampling),
        "browser_gates": asdict(config.browser_gates),
        "latent_block_sizes": list(config.latent_block_sizes),
        "rq3": asdict(config.rq3),
    }


__all__ = [
    "AnalysisConfig",
    "BrowserGateConfig",
    "Rq3Config",
    "SamplingConfig",
    "analysis_snapshot",
    "contract_sha256",
    "load_analysis_file",
    "load_config",
    "resolve_effective_config",
]
