"""Durable local run metadata shared by Python and the Unity ML Lab."""

from __future__ import annotations

import csv
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import atomic_write_json, read_json


RUN_SCHEMA_VERSION = 1
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RUN_STATES = {"created", "running", "stopping", "stopped", "completed", "failed"}
PROGRESS_HEADER = ["timestamp", "timesteps", "episodes", "mean_reward", "steps_per_second"]
MONITOR_HEADER = ["episode_reward", "episode_length", "elapsed_seconds"]


class ContractMismatch(ValueError):
    """Raised when a model and environment use different semantic contracts."""


@dataclass(frozen=True)
class EnvironmentContract:
    version: str
    contract_hash: str
    observation_size: int
    action_size: int
    board: Mapping[str, Any]
    roster: list[str]
    reward: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
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


def _write_csv_header(path: Path, header: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(header)


def create_run(runs_root: Path, config: RunConfig, contract: EnvironmentContract) -> Path:
    """Create a new experiment directory; existing runs are never overwritten."""
    validate_run_name(config.run_name)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / config.run_name
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "replays").mkdir()

    created_at = utc_now()
    config_data = config.to_dict()
    contract_data = contract.to_dict()
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
        "config": config_data,
        "contract": contract_data,
    }
    atomic_write_json(run_dir / "run.json", manifest)
    atomic_write_json(run_dir / "params.json", {"config": config_data, "contract": contract_data})
    atomic_write_json(run_dir / "control.json", {"request": None})
    atomic_write_json(run_dir / "evaluation.json", {})
    _write_csv_header(run_dir / "progress.csv", PROGRESS_HEADER)
    _write_csv_header(run_dir / "monitor.csv", MONITOR_HEADER)
    (run_dir / "train.log").touch(exist_ok=False)
    return run_dir


def update_run_state(run_dir: Path, state: str, **fields: Any) -> dict[str, Any]:
    """Update mutable status while preserving the run's config and contract."""
    if state not in RUN_STATES:
        raise ValueError(f"unknown run state: {state}")
    forbidden = {"config", "contract", "schema_version", "created_at"}.intersection(fields)
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
    if model_info.get("contract_hash") != expected.contract_hash:
        raise ContractMismatch(
            f"model contract hash {model_info.get('contract_hash')!r} does not match "
            f"environment contract hash {expected.contract_hash!r}"
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
    os.replace(source, destination)

    manifest_path = run_dir / "run.json"
    manifest = read_json(manifest_path)
    manifest["latest_checkpoint"] = destination.relative_to(run_dir).as_posix()
    manifest["latest_checkpoint_step"] = step
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return destination
