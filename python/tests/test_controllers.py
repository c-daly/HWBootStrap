from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml_lab.controllers import (
    ControllerResolutionError,
    ControllerResolver,
    normalize_controller_spec,
)
from ml_lab.contracts import EnvironmentContract
from ml_lab.io import atomic_write_json


@dataclass
class _Space:
    shape: tuple[int, ...] = ()
    n: int = 0


@dataclass
class _Model:
    observation_space: _Space
    action_space: _Space


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="a" * 64,
        observation_size=12,
        action_size=7,
        board={"width": 2, "height": 2},
        roster=["scout"],
        reward={"win": 1.0},
    )


@pytest.fixture
def loader():
    def load(path: Path, algorithm: str) -> _Model:
        return _Model(_Space(shape=(12,)), _Space(n=7))

    return load


def _write_run(
    root: Path,
    contract: EnvironmentContract,
    *,
    checkpoint: str | None = "checkpoints/step_000000010.zip",
    step: int | None = 10,
    algorithm: str = "maskable_ppo",
) -> Path:
    run = root / "run-a"
    (run / "checkpoints").mkdir(parents=True)
    if checkpoint is not None:
        (run / checkpoint).write_bytes(b"model")
    atomic_write_json(
        run / "run.json",
        {
            "schema_version": 1,
            "config": {"algorithm": algorithm},
            "contract": contract.to_dict(),
            "latest_checkpoint": checkpoint,
            "latest_checkpoint_step": step,
        },
    )
    return run


@pytest.mark.parametrize("name", ["greedy", "random"])
def test_normalize_scripted_controller(name: str) -> None:
    spec = normalize_controller_spec(name)

    assert spec.kind == "scripted"
    assert spec.name == name


def test_normalize_legacy_checkpoint_spec_without_inferring_algorithm(tmp_path: Path) -> None:
    checkpoint = tmp_path / "anything-at-all.zip"
    checkpoint.write_bytes(b"model")

    spec = normalize_controller_spec(f"ppo:{checkpoint}")

    assert spec.kind == "checkpoint"
    assert spec.algorithm == "maskable_ppo"
    assert spec.path == checkpoint
    with pytest.raises(ControllerResolutionError, match="algorithm"):
        normalize_controller_spec({"kind": "checkpoint", "path": str(checkpoint)})


def test_resolves_fixed_checkpoint_as_legacy_when_algorithm_is_explicit(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    checkpoint = tmp_path / "old-experiment.zip"
    checkpoint.write_bytes(b"model")

    resolved = ControllerResolver(contract, model_loader=loader).resolve(
        {"kind": "checkpoint", "path": str(checkpoint), "algorithm": "masked_dqn"}
    )

    assert resolved.path == checkpoint
    assert resolved.algorithm == "masked_dqn"
    assert resolved.step is None
    assert resolved.contract is None
    assert resolved.legacy is True
    assert resolved.promotable is False


def test_resolves_run_manifest_checkpoint_and_contract(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)

    resolved = ControllerResolver(contract, model_loader=loader).resolve(
        {"kind": "run", "path": str(run), "mode": "fixed"}
    )

    assert resolved.path == run / "checkpoints" / "step_000000010.zip"
    assert resolved.algorithm == "maskable_ppo"
    assert resolved.step == 10
    assert resolved.contract == contract
    assert resolved.legacy is False
    assert resolved.promotable is True


def test_resolves_legacy_checkpoints_directory_with_explicit_algorithm(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    checkpoints = tmp_path / "legacy" / "checkpoints"
    checkpoints.mkdir(parents=True)
    old = checkpoints / "old.zip"
    latest = checkpoints / "latest.zip"
    old.write_bytes(b"old")
    latest.write_bytes(b"latest")
    os.utime(old, (1, 1))
    os.utime(latest, (2, 2))

    resolved = ControllerResolver(contract, model_loader=loader).resolve(f"dqn:{checkpoints}")

    assert resolved.path == latest
    assert resolved.algorithm == "masked_dqn"
    assert resolved.legacy is True


def test_rejects_run_without_published_checkpoint_metadata(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract, checkpoint=None, step=None)

    with pytest.raises(ControllerResolutionError, match="latest_checkpoint"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


def test_rejects_incompatible_model_geometry_even_when_contract_hash_differs(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_contract = EnvironmentContract(
        version=contract.version,
        contract_hash="b" * 64,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
        board=contract.board,
        roster=contract.roster,
        reward=contract.reward,
    )
    run = _write_run(tmp_path, run_contract)

    def incompatible_loader(path: Path, algorithm: str) -> _Model:
        return _Model(_Space(shape=(11,)), _Space(n=7))

    with pytest.raises(ControllerResolutionError, match="observation"):
        ControllerResolver(contract, model_loader=incompatible_loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


def test_live_run_only_advances_after_explicit_reload(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    resolver = ControllerResolver(contract, model_loader=loader)
    binding = resolver.bind({"kind": "run", "path": str(run), "mode": "live"})
    first = binding.resolved
    second = run / "checkpoints" / "step_000000020.zip"
    second.write_bytes(b"model")
    manifest = {
        "schema_version": 1,
        "config": {"algorithm": "maskable_ppo"},
        "contract": contract.to_dict(),
        "latest_checkpoint": "checkpoints/step_000000020.zip",
        "latest_checkpoint_step": 20,
    }
    atomic_write_json(run / "run.json", manifest)

    assert binding.resolved == first
    assert binding.reload() is True
    assert binding.resolved.path == second
    assert binding.resolved.step == 20


def test_fixed_run_does_not_advance_when_reload_is_requested(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind(
        {"kind": "run", "path": str(run), "mode": "fixed"}
    )

    assert binding.reload() is False
