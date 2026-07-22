from __future__ import annotations

import os
import sys
import time
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import selfplay_env as selfplay_module

from ml_lab.controllers import (
    ControllerResolutionError,
    ControllerResolver,
    normalize_controller_spec,
)
from ml_lab.contracts import EnvironmentContract
from ml_lab.io import atomic_write_json
from selfplay_env import SelfPlayEnv, bind_opponents


def _pid_is_running(pid: int) -> bool:
    if os.name == "nt":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_for_pid_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.05)
    return False


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


def test_resolver_accepts_gymnasium_numpy_integer_action_count(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    checkpoint = tmp_path / "numpy-space.zip"
    checkpoint.write_bytes(b"model")

    resolved = ControllerResolver(
        contract,
        model_loader=lambda _path, _algorithm: _Model(
            _Space(shape=(12,)), _Space(n=np.int64(7))
        ),
    ).resolve(
        {
            "kind": "checkpoint",
            "path": str(checkpoint),
            "algorithm": "maskable_ppo",
        }
    )

    assert resolved.action_size == 7


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


def test_legacy_checkpoint_directory_advances_only_after_explicit_reload(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    checkpoints = tmp_path / "legacy" / "checkpoints"
    checkpoints.mkdir(parents=True)
    first = checkpoints / "first.zip"
    first.write_bytes(b"first")
    binding = ControllerResolver(contract, model_loader=loader).bind(f"ppo:{checkpoints}")
    second = checkpoints / "second.zip"
    second.write_bytes(b"second")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    assert binding.resolved.path == first
    assert binding.resolved.legacy is True
    assert binding.reload() is True
    assert binding.resolved.path == second


def test_legacy_unversioned_run_directory_reloads_nested_checkpoints_only_explicitly(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = tmp_path / "old-run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    first = checkpoints / "first.zip"
    first.write_bytes(b"first")
    binding = ControllerResolver(contract, model_loader=loader).bind(f"dqn:{run}")
    second = checkpoints / "second.zip"
    second.write_bytes(b"second")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    assert binding.resolved.path == first
    assert binding.resolved.legacy is True
    assert binding.resolved.promotable is False
    assert binding.resolved.path == first
    assert binding.reload() is True
    assert binding.resolved.path == second


def test_legacy_run_checkpoints_directory_uses_published_manifest_metadata(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)

    binding = ControllerResolver(contract, model_loader=loader).bind(f"ppo:{run / 'checkpoints'}")

    assert binding.resolved.path == run / "checkpoints" / "step_000000010.zip"
    assert binding.resolved.step == 10
    assert binding.resolved.contract == contract
    assert binding.resolved.legacy is False


def test_selfplay_reloads_live_bindings_at_reset_boundary(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind(
        {"kind": "run", "path": str(run), "mode": "live"}
    )
    second = run / "checkpoints" / "step_000000020.zip"
    second.write_bytes(b"model")
    atomic_write_json(
        run / "run.json",
        {
            "schema_version": 1,
            "config": {"algorithm": "maskable_ppo"},
            "contract": contract.to_dict(),
            "latest_checkpoint": "checkpoints/step_000000020.zip",
            "latest_checkpoint_step": 20,
        },
    )
    env = object.__new__(SelfPlayEnv)
    env.opp_pool = [binding]
    env.opp = binding
    env.learner = 0
    env.opp_seat = 1
    env._next_seed = 0
    env.obs_len = contract.observation_size
    env.n_actions = contract.action_size
    env._mask = None
    env._rpc = lambda message: {
        "terminated": True,
        "truncated": False,
        "obs": [0.0] * contract.observation_size,
        "mask": [True] * contract.action_size,
    }

    SelfPlayEnv.reset(env, seed=1)

    assert binding.resolved.path == second


def test_selfplay_rejects_raw_models_without_resolver_metadata(
    contract: EnvironmentContract, loader
) -> None:
    with pytest.raises(ControllerResolutionError, match="controller specification"):
        bind_opponents([loader(Path("unused.zip"), "maskable_ppo")], ControllerResolver(contract, model_loader=loader))


def test_selfplay_pool_sampling_is_seeded_per_episode_not_global_random(
    contract: EnvironmentContract, loader, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver = ControllerResolver(contract, model_loader=loader)
    bindings = [resolver.bind("greedy"), resolver.bind("random")]
    global_choices = iter([0, 1])
    monkeypatch.setattr(
        selfplay_module.random,
        "choice",
        lambda values: values[next(global_choices)],
    )

    def make_env() -> SelfPlayEnv:
        env = object.__new__(SelfPlayEnv)
        env.opp_pool = bindings
        env.opp = bindings[0]
        env.learner = 0
        env.opp_seat = 1
        env._next_seed = 0
        env.obs_len = contract.observation_size
        env.n_actions = contract.action_size
        env._mask = None
        env._rpc = lambda message: {
            "terminated": True,
            "truncated": False,
            "obs": [0.0] * contract.observation_size,
            "mask": [True] * contract.action_size,
        }
        return env

    first = make_env()
    second = make_env()
    SelfPlayEnv.reset(first, seed=37)
    SelfPlayEnv.reset(second, seed=37)

    assert first.opp.resolved.server_controller == second.opp.resolved.server_controller


def test_selfplay_constructor_failure_closes_and_reaps_server_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path = tmp_path / "selfplay-child.pid"
    marker_path = tmp_path / "selfplay-child.closed"
    child_code = (
        "import os, pathlib, sys; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "sys.stdin.readline(); print('{}', flush=True); "
        "sys.stdin.read(); "
        f"pathlib.Path({str(marker_path)!r}).write_text('closed')"
    )

    real_popen = selfplay_module.subprocess.Popen
    children = []

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(selfplay_module.subprocess, "Popen", capture_popen)

    with pytest.raises(KeyError) as constructor_error:
        SelfPlayEnv([sys.executable, "-c", child_code], ["greedy"])

    process = children[0]
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        assert marker_path.read_text(encoding="utf-8") == "closed"
        assert process.poll() is not None
        assert _wait_for_pid_exit(child_pid)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
    del constructor_error
