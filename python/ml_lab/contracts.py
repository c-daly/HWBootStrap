"""Durable local run metadata shared by Python and the Unity ML Lab."""

from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import atomic_write_json, read_json
from .scenarios import ResolvedScenario


RUN_SCHEMA_VERSION = 1
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RUN_STATES = {"created", "running", "stopping", "stopped", "completed", "failed"}
PROGRESS_HEADER = ["timestamp", "timesteps", "episodes", "mean_reward", "steps_per_second"]
MONITOR_HEADER = [
    "worker_id",
    "episode_index",
    "episode_seed",
    "learner_seat",
    "episode_reward",
    "episode_length",
    "elapsed_seconds",
]
ADAPTIVE_MONITOR_HEADER = [
    "episode",
    "design_count",
    "distinct_custom_templates_deployed",
    "deployment_completed",
    "invalid_sequences",
    "pregame_decisions",
]
TRACKER_CREDENTIAL_PARTS = {"token", "secret", "password"}


class ContractMismatch(ValueError):
    """Raised when a model and environment use different semantic contracts."""


@dataclass(frozen=True)
class EnvironmentContract:
    version: str
    contract_hash: str
    encoding_hash: str
    observation_size: int
    action_size: int
    board: Mapping[str, Any]
    roster: list[str]
    reward: Mapping[str, Any]
    semantics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.encoding_hash):
            raise ValueError("encoding_hash must be a lowercase SHA-256 hex digest")

    @property
    def environment(self) -> str:
        return self.version

    def to_dict(self) -> dict[str, Any]:
        return {"environment": self.environment, **asdict(self)}


@dataclass(frozen=True)
class RunConfig:
    backend: str
    algorithm: str
    policy: str
    run_name: str
    seed: int
    total_timesteps: int
    checkpoint_interval: int
    workers: int
    device: str
    learner_seat: str
    opponent: Mapping[str, Any]
    trackers: list[Mapping[str, Any]]
    resume_source: str | None
    timestep_mode: str = "absolute"
    allow_unsafe_legacy_resume: bool = False
    environment: str = "tactical-v2"

    def to_dict(self) -> dict[str, Any]:
        validate_tracker_specs(self.trackers)
        if self.timestep_mode not in {"absolute", "additional"}:
            raise ValueError("timestep mode must be 'absolute' or 'additional'")
        if self.environment not in {"tactical-v1", "tactical-v2", "adaptive-v1"}:
            raise ValueError("environment must be tactical-v1, tactical-v2, or adaptive-v1")
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_run_name(name: str) -> str:
    if not RUN_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "run name must be 1-64 characters and use only letters, numbers, '.', '_' or '-', "
            "starting with a letter or number"
        )
    return name


def _is_tracker_credential_key(key: str) -> bool:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    parts = [part for part in re.sub(r"[^A-Za-z0-9]+", "_", separated).lower().split("_") if part]
    if TRACKER_CREDENTIAL_PARTS.intersection(parts):
        return True
    return any(left == "api" and right == "key" for left, right in zip(parts, parts[1:]))


def _validate_tracker_value(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if isinstance(key, str) and _is_tracker_credential_key(key):
                raise ValueError(
                    f"tracker configuration contains forbidden credential key at {child_path}"
                )
            _validate_tracker_value(nested, child_path)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_tracker_value(nested, f"{path}[{index}]")


def validate_tracker_specs(trackers: list[Mapping[str, Any]]) -> None:
    """Reject credentials before tracker configuration can enter durable run metadata."""
    for index, tracker in enumerate(trackers):
        _validate_tracker_value(tracker, f"trackers[{index}]")


def _write_csv_header(path: Path, header: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(header)


def create_run(
    runs_root: Path,
    config: RunConfig,
    contract: EnvironmentContract,
    scenario: ResolvedScenario,
    *,
    opponent_snapshot: Mapping[str, Any],
) -> Path:
    """Create a new experiment directory; existing runs are never overwritten."""
    validate_run_name(config.run_name)
    config_data = config.to_dict()
    if config.environment != contract.environment:
        raise ContractMismatch("run environment does not match the environment contract")
    if scenario.environment != config.environment:
        raise ContractMismatch("scenario environment does not match the run configuration")
    contract_data = contract.to_dict()
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / config.run_name
    # The CLI may have already created an empty run directory (e.g. to host a
    # stderr capture file opened before this manifest exists). A run is only
    # "existing" once it has a manifest, so that is the overwrite guard.
    if (run_dir / "run.json").exists():
        raise FileExistsError(run_dir / "run.json")
    run_dir.mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "replays").mkdir()
    scenario.write(run_dir / "scenario.json")

    created_at = utc_now()
    monitor_files = (
        ["monitor.csv"]
        if config.workers == 1
        else [f"monitor.worker_{index}.csv" for index in range(config.workers)]
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": created_at,
        "updated_at": created_at,
        "state": "created",
        "pid": None,
        "timesteps": 0,
        "latest_message": None,
        "latest_checkpoint": None,
        "latest_checkpoint_step": None,
        "monitor_files": monitor_files,
        "config": config_data,
        "contract": contract_data,
        "scenario": {
            "path": "scenario.json",
            "template_id": scenario.template_id,
            "schema_version": scenario.schema_version,
        },
        "opponent_snapshot": dict(opponent_snapshot),
    }
    atomic_write_json(run_dir / "run.json", manifest)
    atomic_write_json(run_dir / "params.json", {"config": config_data, "contract": contract_data})
    atomic_write_json(run_dir / "control.json", {"request": None})
    atomic_write_json(run_dir / "evaluation.json", {})
    _write_csv_header(run_dir / "progress.csv", PROGRESS_HEADER)
    _write_csv_header(run_dir / "monitor.csv", MONITOR_HEADER)
    for monitor_file in monitor_files:
        if monitor_file != "monitor.csv":
            _write_csv_header(run_dir / monitor_file, MONITOR_HEADER)
    if config.environment == "adaptive-v1":
        adaptive_monitor_files = (
            ["adaptive_episodes.csv"]
            if config.workers == 1
            else [f"adaptive_episodes.worker_{index}.csv" for index in range(config.workers)]
        )
        for monitor_file in adaptive_monitor_files:
            _write_csv_header(run_dir / monitor_file, ADAPTIVE_MONITOR_HEADER)
    (run_dir / "train.log").touch(exist_ok=False)
    return run_dir


def update_run_state(run_dir: Path, state: str, **fields: Any) -> dict[str, Any]:
    """Update mutable status while preserving the run's config and contract."""
    if state not in RUN_STATES:
        raise ValueError(f"unknown run state: {state}")
    forbidden = {
        "config",
        "contract",
        "scenario",
        "opponent_snapshot",
        "schema_version",
        "created_at",
    }.intersection(fields)
    if forbidden:
        raise ValueError(f"immutable run fields cannot be updated: {', '.join(sorted(forbidden))}")
    manifest_path = Path(run_dir) / "run.json"
    manifest = read_json(manifest_path)
    manifest.update(fields)
    manifest["state"] = state
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest


def request_stop(run_dir: Path, *, after_checkpoint: bool) -> dict[str, Any]:
    control = {
        "request": "stop_after_checkpoint" if after_checkpoint else "stop_now",
        "updated_at": utc_now(),
    }
    atomic_write_json(Path(run_dir) / "control.json", control)
    return control


def _validate_model_contract(model_info: Mapping[str, Any], expected: EnvironmentContract) -> None:
    if model_info.get("environment") != expected.environment:
        raise ContractMismatch("model environment does not match the environment contract")
    if model_info.get("contract_version") != expected.version:
        raise ContractMismatch("model contract version does not match the environment contract")
    if model_info.get("encoding_hash") != expected.encoding_hash:
        raise ContractMismatch(
            f"model encoding hash {model_info.get('encoding_hash')!r} does not match "
            f"environment encoding hash {expected.encoding_hash!r}"
        )
    if model_info.get("observation_size") != expected.observation_size:
        raise ContractMismatch("model observation size does not match the environment")
    if model_info.get("action_size") != expected.action_size:
        raise ContractMismatch("model action size does not match the environment")


def publish_checkpoint(
    *,
    source: Path,
    run_dir: Path,
    step: int,
    expected_contract: EnvironmentContract,
    inspector: Callable[[Path], Mapping[str, Any]],
) -> Path:
    """Validate and atomically move a pending model into a run's checkpoints."""
    source = Path(source)
    run_dir = Path(run_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")

    _validate_model_contract(inspector(source), expected_contract)
    destination = run_dir / "checkpoints" / f"step_{step:09d}.zip"
    if destination.exists():
        raise FileExistsError(destination)

    staged_path: Path | None = None
    try:
        with source.open("rb") as source_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as staged_stream:
            staged_path = Path(staged_stream.name)
            shutil.copyfileobj(source_stream, staged_stream)
            staged_stream.flush()
            os.fsync(staged_stream.fileno())
        os.replace(staged_path, destination)
        staged_path = None
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)

    manifest_path = run_dir / "run.json"
    manifest = read_json(manifest_path)
    manifest["latest_checkpoint"] = destination.relative_to(run_dir).as_posix()
    manifest["latest_checkpoint_step"] = step
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    source.unlink()
    return destination
