from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import ctypes
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
import selfplay_env as selfplay_module
import ml_lab.controllers as controller_module

from ml_lab.controllers import (
    ControllerResolutionError,
    ControllerResolver,
    _validate_contract_compatibility,
    normalize_controller_spec,
    predict,
    snapshot_opponents,
)
from ml_lab.algorithms import ActorTransferSource, MaskablePPOAdapter
from ml_lab.contracts import ContractMismatch, EnvironmentContract
from ml_lab.envs import build_vector_env
from ml_lab.io import atomic_write_json, read_json
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
        encoding_hash="b" * 64,
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
    run = _write_run(tmp_path, contract)

    resolved = ControllerResolver(
        contract,
        model_loader=lambda _path, _algorithm: _Model(
            _Space(shape=(12,)), _Space(n=np.int64(7))
        ),
    ).resolve({"kind": "run", "path": str(run), "mode": "fixed"})

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


class _ActorTransferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, contract: EnvironmentContract) -> None:
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(contract.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(contract.action_size)
        self.contract = contract
        self.spaces_info = {
            "channels": 1,
            "board_h": 1,
            "board_w": 1,
            "globals": contract.observation_size - 1,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)


def _actor_transfer_contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="e" * 64,
        encoding_hash="f" * 64,
        observation_size=3,
        action_size=2,
        board={"width": 1, "height": 1},
        roster=["scout"],
        reward={"terminal_win": 1.0},
    )


def _actor_modules(policy):
    return {
        "features_extractor": policy.features_extractor,
        "policy_net": policy.mlp_extractor.policy_net,
        "action_net": policy.action_net,
    }


def _module_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _value_state(model):
    return {
        "mlp_extractor.value_net": _module_state(model.policy.mlp_extractor.value_net),
        "value_net": _module_state(model.policy.value_net),
    }


def _assert_state_equal(actual, expected) -> None:
    import torch

    assert actual.keys() == expected.keys()
    for group in actual:
        assert actual[group].keys() == expected[group].keys()
        for name in actual[group]:
            assert torch.equal(actual[group][name], expected[group][name])


def _fixture_logits(model, observations: np.ndarray, legal_masks: np.ndarray):
    import torch

    with torch.no_grad():
        distribution = model.policy.get_distribution(
            torch.as_tensor(observations, dtype=torch.float32, device=model.device),
            action_masks=torch.as_tensor(legal_masks, dtype=torch.bool, device=model.device),
        )
        return distribution.distribution.logits.detach().cpu()


def _write_actor_source_run(root: Path, contract: EnvironmentContract):
    import torch

    adapter = MaskablePPOAdapter()
    env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        source = adapter.create(
            env,
            spaces_info=env.spaces_info,
            seed=17,
            device="cpu",
            checkpoint_interval=32,
        )
        with torch.no_grad():
            for module in _actor_modules(source.policy).values():
                for parameter in module.parameters():
                    parameter.add_(0.05)
        run = root / "clone-run"
        checkpoint = adapter.save(
            source,
            run / "checkpoints" / "step_000000000.zip",
        )
    finally:
        env.close()

    observations = np.asarray(
        [[0.0, 0.25, -0.5], [0.75, -0.25, 0.5]],
        dtype=np.float32,
    )
    legal_masks = np.asarray([[True, True], [False, True]], dtype=bool)
    clone_config = {
        "model_seed": 17,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "max_epochs": 50,
        "patience": 5,
    }
    np.savez(
        run / "actor-fixtures.npz",
        observations=observations,
        legal_masks=legal_masks,
    )
    atomic_write_json(
        run / "bc.json",
        {
            "schema_version": 1,
            "algorithm": "maskable_ppo",
            "policy": "HexCNN",
            "dataset_manifest_sha256": "a" * 64,
            "config": clone_config,
            "model_seed": 17,
            "best_epoch": 3,
            "epochs_trained": 3,
        },
    )
    atomic_write_json(
        run / "run.json",
        {
            "schema_version": 1,
            "state": "completed",
            "timesteps": 0,
            "config": {
                "algorithm": "maskable_ppo",
                "policy": "HexCNN",
                "seed": 17,
                "model_seed": 17,
                "behavioral_cloning": clone_config,
            },
            "contract": contract.to_dict(),
            "latest_checkpoint": "checkpoints/step_000000000.zip",
            "latest_checkpoint_step": 0,
            "dataset_manifest_sha256": "a" * 64,
            "bc_config": clone_config,
            "model_seed": 17,
            "best_epoch": 3,
        },
    )
    return run, source, observations, legal_masks, checkpoint


def _actor_transfer_source(
    source_run: Path,
    checkpoint: Path,
) -> ActorTransferSource:
    source = ActorTransferSource(
        source_kind="snapshot",
        controller={
            "kind": "snapshot",
            "path": str(checkpoint.resolve()),
            "source_run": str(source_run.resolve()),
            "algorithm": "maskable_ppo",
            "step": 0,
            "inference_mode": "deterministic",
        },
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    return source


def test_actor_transfer_preserves_masked_logits_but_not_value_parameters(
    tmp_path: Path,
) -> None:
    import torch

    contract = _actor_transfer_contract()
    source_run, source, observations, legal_masks, checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    resolved = ControllerResolver(contract).resolve(f"run:{source_run}")
    assert resolved.path == checkpoint.resolve()
    assert resolved.contract == contract

    adapter = MaskablePPOAdapter()
    target_env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        source_logits = _fixture_logits(source, observations, legal_masks)
        assert not torch.equal(
            source_logits,
            _fixture_logits(target, observations, legal_masks),
        )
        source_actor = {
            name: _module_state(module)
            for name, module in _actor_modules(source.policy).items()
        }
        source_values = _value_state(source)
        target_values_before = _value_state(target)
        optimizer = target.policy.optimizer
        rollout_buffer = target.rollout_buffer
        lr_schedule = target.lr_schedule
        clip_range = target.clip_range
        progress_before = target._current_progress_remaining
        episodes_before = target._episode_num
        timesteps_before = target.num_timesteps

        provenance = adapter.initialize_actor(
            target,
            source_run=source_run,
            expected_contract=contract,
            device="cpu",
        )

        torch.testing.assert_close(
            _fixture_logits(target, observations, legal_masks),
            source_logits,
            rtol=0,
            atol=0,
        )
        target_actor = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        _assert_state_equal(target_actor, source_actor)
        _assert_state_equal(_value_state(target), target_values_before)
        assert any(
            not torch.equal(source_values[group][name], target_values_before[group][name])
            for group in source_values
            for name in source_values[group]
        )
        assert target.policy.optimizer is optimizer
        assert target.policy.optimizer.state == {}
        assert target.rollout_buffer is rollout_buffer
        assert target.lr_schedule is lr_schedule
        assert target.clip_range is clip_range
        assert target._current_progress_remaining == progress_before
        assert target._episode_num == episodes_before
        assert target.num_timesteps == timesteps_before
        assert provenance["source_checkpoint_sha256"] == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        assert provenance["source_actor_fixtures_sha256"] == hashlib.sha256(
            (source_run / "actor-fixtures.npz").read_bytes()
        ).hexdigest()
        assert provenance["maximum_absolute_logit_difference"] == 0.0
    finally:
        target_env.close()


def test_actor_transfer_restores_target_actor_when_copy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _actor_transfer_contract()
    source_run, _source, _observations, _masks, checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    adapter = MaskablePPOAdapter()
    transfer_source = _actor_transfer_source(
        source_run, checkpoint,
    )
    target_env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        actor_before = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }

        original_load = target.policy.action_net.load_state_dict
        attempts = 0

        def fail_first_copy(state_dict, *, strict=True, assign=False):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected actor-copy failure")
            return original_load(state_dict, strict=strict, assign=assign)

        monkeypatch.setattr(
            target.policy.action_net, "load_state_dict", fail_first_copy,
        )

        with pytest.raises(RuntimeError, match="injected actor-copy failure"):
            adapter.initialize_actor_from_source(
                target,
                transfer_source,
                contract,
                "cpu",
            )

        assert attempts == 2
        actor_after = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        _assert_state_equal(actor_after, actor_before)
    finally:
        target_env.close()


def test_actor_transfer_rejects_noncanonical_zero_step_checkpoint_before_copy(
    tmp_path: Path,
) -> None:
    contract = _actor_transfer_contract()
    source_run, _source, _observations, _masks, checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    alternate_checkpoint = source_run / "checkpoints" / "alternate.zip"
    alternate_checkpoint.write_bytes(checkpoint.read_bytes())
    manifest = read_json(source_run / "run.json")
    manifest["latest_checkpoint"] = "checkpoints/alternate.zip"
    atomic_write_json(source_run / "run.json", manifest)

    adapter = MaskablePPOAdapter()
    target_env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        actor_before = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }

        with pytest.raises(ValueError, match="metadata does not match"):
            adapter.initialize_actor(
                target,
                source_run=source_run,
                expected_contract=contract,
                device="cpu",
            )

        actor_after = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        _assert_state_equal(actor_after, actor_before)
    finally:
        target_env.close()


def test_actor_transfer_rejects_bc_metadata_not_bound_to_run(
    tmp_path: Path,
) -> None:
    contract = _actor_transfer_contract()
    source_run, _source, _observations, _masks, _checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    bc = read_json(source_run / "bc.json")
    bc["dataset_manifest_sha256"] = "b" * 64
    atomic_write_json(source_run / "bc.json", bc)

    adapter = MaskablePPOAdapter()
    target_env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        actor_before = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }

        with pytest.raises(ValueError, match="metadata does not match"):
            adapter.initialize_actor(
                target,
                source_run=source_run,
                expected_contract=contract,
                device="cpu",
            )

        actor_after = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        _assert_state_equal(actor_after, actor_before)
    finally:
        target_env.close()


def test_actor_transfer_accepts_compatible_source_with_different_contract_hash(
    tmp_path: Path,
) -> None:
    import torch

    contract = _actor_transfer_contract()
    source_run, _source, _observations, _masks, _checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    expected_contract = dataclass_replace(contract, contract_hash="d" * 64)
    adapter = MaskablePPOAdapter()
    target_env = build_vector_env(
        1,
        lambda _worker: _ActorTransferEnv(expected_contract),
    )
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        actor_before = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }

        provenance = adapter.initialize_actor(
            target,
            source_run=source_run,
            expected_contract=expected_contract,
            device="cpu",
        )

        actor_after = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        assert any(
            not torch.equal(actor_after[group][name], actor_before[group][name])
            for group in actor_after
            for name in actor_after[group]
        )
        assert provenance["source_contract_hash"] == contract.contract_hash
        assert provenance["source_contract_hash"] != expected_contract.contract_hash
        assert provenance["source_encoding_hash"] == expected_contract.encoding_hash
    finally:
        target_env.close()


def test_actor_transfer_rejects_dtype_mismatch_before_any_module_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    contract = _actor_transfer_contract()
    source_run, _source, _observations, _masks, checkpoint = _write_actor_source_run(
        tmp_path,
        contract,
    )
    adapter = MaskablePPOAdapter()
    transfer_source = _actor_transfer_source(
        source_run, checkpoint,
    )
    original_load = MaskablePPOAdapter.load

    def load_with_dtype_mismatch(self, checkpoint_buffer, *, env, device):
        source_model = original_load(
            self, checkpoint_buffer, env=env, device=device,
        )
        source_model.policy.action_net.to(dtype=torch.float64)
        return source_model

    monkeypatch.setattr(
        MaskablePPOAdapter, "load", load_with_dtype_mismatch,
    )
    target_env = build_vector_env(1, lambda _worker: _ActorTransferEnv(contract))
    try:
        target = adapter.create(
            target_env,
            spaces_info=target_env.spaces_info,
            seed=999,
            device="cpu",
            checkpoint_interval=32,
        )
        actor_before = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }

        with pytest.raises(ContractMismatch, match="dtype"):
            adapter.initialize_actor_from_source(
                target,
                transfer_source,
                contract,
                "cpu",
            )

        actor_after = {
            name: _module_state(module)
            for name, module in _actor_modules(target.policy).items()
        }
        _assert_state_equal(actor_after, actor_before)
    finally:
        target_env.close()


def test_load_model_forces_cpu_device_for_maskable_ppo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mirrors policy_server's documented rule: inference always runs on CPU so
    checkpoint-time evaluation never competes with training for the GPU."""
    calls: list[dict] = []

    class _FakeMaskablePPO:
        @classmethod
        def load(cls, path, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(path=path)

    monkeypatch.setattr("sb3_contrib.MaskablePPO", _FakeMaskablePPO)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"model")

    controller_module.load_model(checkpoint, "maskable_ppo")

    assert calls == [{"device": "cpu"}]


def test_load_model_forces_cpu_device_for_masked_dqn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []

    class _FakeDQN:
        @classmethod
        def load(cls, path, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(path=path)

    monkeypatch.setattr("stable_baselines3.DQN", _FakeDQN)
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"model")

    controller_module.load_model(checkpoint, "masked_dqn")

    assert calls == [{"device": "cpu"}]


def test_scripted_opponent_snapshot_retains_exact_name() -> None:
    assert snapshot_opponents({"kind": "scripted", "name": "random"}) == {
        "kind": "scripted",
        "name": "random",
    }


def test_fixed_run_snapshot_retains_exact_checkpoint_identity(
    tmp_path: Path,
    contract: EnvironmentContract,
    loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_run(tmp_path, contract)
    monkeypatch.setattr(controller_module, "load_model", loader)

    snapshot = snapshot_opponents(
        {
            "kind": "run",
            "path": str(run),
            "mode": "fixed",
            "inference_mode": "stochastic",
        }
    )

    assert snapshot == {
        "kind": "snapshot",
        "path": str((run / "checkpoints" / "step_000000010.zip").resolve()),
        "source_run": str(run.resolve()),
        "algorithm": "maskable_ppo",
        "step": 10,
        "inference_mode": "stochastic",
    }
    (run / "checkpoints" / "step_000000020.zip").write_bytes(b"newer-model")
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
    resolved = ControllerResolver(contract, model_loader=loader).resolve(snapshot)
    assert resolved.path == run / "checkpoints" / "step_000000010.zip"
    assert resolved.contract == contract


def test_snapshot_controller_loads_contract_from_its_recorded_source_run(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    checkpoint = run / "checkpoints" / "step_000000010.zip"

    resolved = ControllerResolver(model_loader=loader).resolve(
        {
            "kind": "snapshot",
            "path": str(checkpoint),
            "source_run": str(run),
            "algorithm": "maskable_ppo",
            "step": 10,
        }
    )

    assert resolved.path == checkpoint
    assert resolved.contract == contract
    assert resolved.algorithm == "maskable_ppo"
    assert resolved.step == 10


@pytest.mark.parametrize("location", ["standalone", "escaped", "nested"])
def test_snapshot_controller_rejects_checkpoint_outside_recorded_checkpoint_directory(
    tmp_path: Path,
    contract: EnvironmentContract,
    loader,
    location: str,
) -> None:
    run = _write_run(tmp_path, contract)
    if location == "standalone":
        checkpoint = tmp_path / "step_000000010.zip"
    elif location == "escaped":
        checkpoint = run / "checkpoints" / ".." / "step_000000010.zip"
    else:
        checkpoint = run / "checkpoints" / "nested" / "step_000000010.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"model")

    with pytest.raises(ControllerResolutionError, match="inside the source run"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {
                "kind": "snapshot",
                "path": str(checkpoint),
                "source_run": str(run),
                "algorithm": "maskable_ppo",
                "step": 10,
            }
        )


def test_snapshot_controller_rejects_algorithm_not_recorded_by_source_run(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)

    with pytest.raises(ControllerResolutionError, match="algorithm"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {
                "kind": "snapshot",
                "path": str(run / "checkpoints" / "step_000000010.zip"),
                "source_run": str(run),
                "algorithm": "masked_dqn",
                "step": 10,
            }
        )


def test_snapshot_controller_requires_canonical_recorded_algorithm(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)

    with pytest.raises(
        ControllerResolutionError, match="snapshot controller algorithm"
    ):
        ControllerResolver(contract, model_loader=loader).resolve(
            {
                "kind": "snapshot",
                "path": str(run / "checkpoints" / "step_000000010.zip"),
                "source_run": str(run),
                "algorithm": "ppo",
                "step": 10,
            }
        )


def test_snapshot_controller_rejects_step_not_recorded_by_checkpoint_name(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)

    with pytest.raises(ControllerResolutionError, match="step"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {
                "kind": "snapshot",
                "path": str(run / "checkpoints" / "step_000000010.zip"),
                "source_run": str(run),
                "algorithm": "maskable_ppo",
                "step": 11,
            }
        )


def test_live_run_snapshot_retains_run_path_and_live_mode(
    tmp_path: Path,
    contract: EnvironmentContract,
    loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_run(tmp_path, contract)
    monkeypatch.setattr(controller_module, "load_model", loader)

    snapshot = snapshot_opponents(
        {"kind": "run", "path": str(run), "mode": "live"}
    )

    assert snapshot == {
        "kind": "run",
        "path": str(run.resolve()),
        "mode": "live",
        "inference_mode": "deterministic",
    }


def test_pool_snapshot_preserves_every_entry_in_input_order(
    tmp_path: Path,
    contract: EnvironmentContract,
    loader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_run(tmp_path, contract)
    monkeypatch.setattr(controller_module, "load_model", loader)

    snapshot = snapshot_opponents(
        {
            "kind": "pool",
            "controllers": [
                {"kind": "scripted", "name": "random"},
                {"kind": "run", "path": str(run), "mode": "fixed"},
                {"kind": "scripted", "name": "greedy"},
            ],
        }
    )

    assert [entry["kind"] for entry in snapshot["controllers"]] == [
        "scripted",
        "snapshot",
        "scripted",
    ]
    assert [entry.get("name") for entry in snapshot["controllers"]] == [
        "random",
        None,
        "greedy",
    ]


@pytest.mark.parametrize("name", ["greedy", "random", "bounded-search"])
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


def test_run_inference_mode_defaults_to_deterministic() -> None:
    spec = normalize_controller_spec({"kind": "run", "path": "run-a", "mode": "live"})

    assert spec.inference_mode == "deterministic"


def test_stochastic_run_inference_mode_survives_resolution_and_reload(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind(
        {
            "kind": "run",
            "path": str(run),
            "mode": "live",
            "inference_mode": "stochastic",
        }
    )

    assert binding.resolved.spec.inference_mode == "stochastic"
    assert binding.resolved.metadata()["inference_mode"] == "stochastic"
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

    assert binding.reload() is True
    assert binding.resolved.path == second
    assert binding.resolved.spec.inference_mode == "stochastic"


def test_unknown_run_inference_mode_is_rejected() -> None:
    with pytest.raises(ControllerResolutionError, match="inference mode"):
        normalize_controller_spec(
            {"kind": "run", "path": "run-a", "inference_mode": "epsilon"}
        )


def test_maskable_ppo_prediction_defaults_to_deterministic_and_can_sample() -> None:
    calls: list[dict] = []

    class Model:
        def predict(self, observation, **kwargs):
            calls.append(kwargs)
            return np.int64(3), None

    observation = np.zeros(4, dtype=np.float32)
    mask = np.array([True, False, True, True])

    assert predict(Model(), "maskable_ppo", observation, mask) == 3
    assert predict(
        Model(), "maskable_ppo", observation, mask, deterministic=False
    ) == 3
    assert len(calls) == 2
    assert calls[0]["deterministic"] is True
    assert calls[1]["deterministic"] is False
    np.testing.assert_array_equal(calls[0]["action_masks"], mask)
    np.testing.assert_array_equal(calls[1]["action_masks"], mask)


def test_rejects_fixed_checkpoint_even_when_algorithm_is_explicit(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    checkpoint = tmp_path / "old-experiment.zip"
    checkpoint.write_bytes(b"model")

    with pytest.raises(ControllerResolutionError, match="contract metadata"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "checkpoint", "path": str(checkpoint), "algorithm": "masked_dqn"}
        )


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
    assert resolved.metadata()["contract_version"] == "tactical-v1"
    assert resolved.metadata()["environment"] == "tactical-v1"
    assert resolved.metadata()["encoding_hash"] == "b" * 64


@pytest.mark.parametrize("algorithm", ("maskable_ppo", "masked_dqn"))
def test_legacy_run_algorithms_require_zip_checkpoints(
    tmp_path: Path, contract: EnvironmentContract, loader, algorithm: str,
) -> None:
    run = _write_run(tmp_path, contract, algorithm=algorithm)
    manifest = __import__("json").loads((run / "run.json").read_text(encoding="utf-8"))
    manifest["latest_checkpoint"] = "checkpoints/model.pt"
    (run / "checkpoints" / "model.pt").write_bytes(b"not-an-sb3-model")
    atomic_write_json(run / "run.json", manifest)

    with pytest.raises(ControllerResolutionError, match=".zip"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


def test_controller_rejects_wrong_environment_for_adaptive_runtime(
    contract: EnvironmentContract,
) -> None:
    adaptive = dataclass_replace(
        contract,
        version="adaptive-v1",
        contract_hash="d" * 64,
        semantics={"environment_kind": "adaptive_tactical"},
    )
    with pytest.raises(ControllerResolutionError, match="environment"):
        _validate_contract_compatibility(contract, adaptive)


def test_controller_rejects_tactical_v1_model_for_tactical_v2_runtime(
    contract: EnvironmentContract,
) -> None:
    """Adding tactical-v2 support must not make tactical-v1 checkpoints cross-compatible:
    different contract version and encoding hash, so compatibility stays exact."""
    tactical_v2 = dataclass_replace(
        contract,
        version="tactical-v2",
        contract_hash="d" * 64,
        encoding_hash="c" * 64,
    )
    with pytest.raises(ControllerResolutionError, match="environment"):
        _validate_contract_compatibility(contract, tactical_v2)


def test_resolves_tactical_v2_run_manifest_checkpoint_and_contract(
    tmp_path: Path, loader
) -> None:
    tactical_v2 = EnvironmentContract(
        version="tactical-v2",
        contract_hash="e" * 64,
        encoding_hash="f" * 64,
        observation_size=12,
        action_size=7,
        board={"width": 2, "height": 2},
        roster=["brute-85597320:Brute:7,2,2,3,2,1,1,2,1"],
        reward={"terminal_win": 1.0},
    )
    run = _write_run(tmp_path, tactical_v2)

    resolved = ControllerResolver(tactical_v2, model_loader=loader).resolve(
        {"kind": "run", "path": str(run), "mode": "fixed"}
    )

    assert resolved.contract == tactical_v2
    assert resolved.metadata()["contract_version"] == "tactical-v2"
    assert resolved.metadata()["environment"] == "tactical-v2"


def test_contract_from_manifest_rejects_unknown_encoding_version(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    manifest = __import__("json").loads((run / "run.json").read_text(encoding="utf-8"))
    manifest["contract"]["version"] = "tactical-v3"
    manifest["contract"]["environment"] = "tactical-v3"
    atomic_write_json(run / "run.json", manifest)

    with pytest.raises(ControllerResolutionError, match="unsupported"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


def test_controller_accepts_adaptive_tactical_run_for_duel_with_shared_encoding_hash(
    contract: EnvironmentContract,
) -> None:
    adaptive = dataclass_replace(
        contract,
        version="adaptive-v1",
        contract_hash="d" * 64,
        board={**contract.board, "environment_kind": "adaptive_tactical"},
        semantics={"environment_kind": "adaptive_tactical"},
    )
    duel = dataclass_replace(
        adaptive,
        contract_hash="e" * 64,
        board={**adaptive.board, "environment_kind": "adaptive_duel"},
        semantics={"environment_kind": "adaptive_duel"},
    )

    _validate_contract_compatibility(adaptive, duel)


def test_controller_rejects_encoding_hash_mismatch_before_inference(
    contract: EnvironmentContract,
) -> None:
    incompatible = dataclass_replace(contract, encoding_hash="c" * 64)

    with pytest.raises(ControllerResolutionError, match="encoding hash"):
        _validate_contract_compatibility(contract, incompatible)


def test_run_manifest_without_encoding_hash_is_rejected(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    manifest = __import__("json").loads((run / "run.json").read_text(encoding="utf-8"))
    del manifest["contract"]["encoding_hash"]
    atomic_write_json(run / "run.json", manifest)

    with pytest.raises(ControllerResolutionError, match="encoding_hash"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


def test_adaptive_runtime_rejects_contractless_checkpoint_before_loading(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    checkpoint = tmp_path / "contractless.zip"
    checkpoint.write_bytes(b"model")
    adaptive = dataclass_replace(
        contract,
        version="adaptive-v1",
        contract_hash="d" * 64,
        board={**contract.board, "environment_kind": "adaptive_tactical"},
        semantics={"environment_kind": "adaptive_tactical"},
    )
    loaded = False

    def loader(_path, _algorithm):
        nonlocal loaded
        loaded = True
        return _Model(_Space(shape=(12,)), _Space(n=7))

    with pytest.raises(ControllerResolutionError, match="contract metadata"):
        ControllerResolver(adaptive, model_loader=loader).resolve(
            {"kind": "checkpoint", "path": str(checkpoint), "algorithm": "maskable_ppo"}
        )
    assert loaded is False


def test_rejects_legacy_checkpoints_directory_with_explicit_algorithm(
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

    with pytest.raises(ControllerResolutionError, match="contract metadata"):
        ControllerResolver(contract, model_loader=loader).resolve(f"dqn:{checkpoints}")


def test_rejects_run_without_published_checkpoint_metadata(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract, checkpoint=None, step=None)

    with pytest.raises(ControllerResolutionError, match="latest_checkpoint"):
        ControllerResolver(contract, model_loader=loader).resolve(
            {"kind": "run", "path": str(run), "mode": "fixed"}
        )


@pytest.mark.parametrize("algorithm", ("maskable_ppo", "masked_dqn"))
@pytest.mark.parametrize(
    ("mismatch", "observation_shape", "action_size"),
    (
        ("observation", (11,), 7),
        ("action", (12,), 6),
    ),
)
def test_legacy_run_algorithms_resolve_valid_zip_then_reject_each_geometry_mismatch(
    tmp_path: Path,
    contract: EnvironmentContract,
    algorithm: str,
    mismatch: str,
    observation_shape: tuple[int, ...],
    action_size: int,
) -> None:
    run_contract = EnvironmentContract(
        version=contract.version,
        contract_hash="b" * 64,
        encoding_hash=contract.encoding_hash,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
        board=contract.board,
        roster=contract.roster,
        reward=contract.reward,
    )
    run = _write_run(tmp_path, run_contract, algorithm=algorithm)
    controller = {"kind": "run", "path": str(run), "mode": "fixed"}

    resolved = ControllerResolver(
        contract,
        model_loader=lambda _path, _algorithm: _Model(
            _Space(shape=(12,)), _Space(n=7)
        ),
    ).resolve(controller)
    assert resolved.path == run / "checkpoints" / "step_000000010.zip"
    assert resolved.algorithm == algorithm
    assert (resolved.observation_size, resolved.action_size) == (12, 7)

    with pytest.raises(ControllerResolutionError, match=mismatch):
        ControllerResolver(
            contract,
            model_loader=lambda _path, _algorithm: _Model(
                _Space(shape=observation_shape), _Space(n=action_size)
            ),
        ).resolve(controller)


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


def test_rejected_live_reload_keeps_previous_resolved_model(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind(
        {"kind": "run", "path": str(run), "mode": "live"}
    )
    previous = binding.resolved
    incompatible = dataclass_replace(contract, encoding_hash="c" * 64)
    second = run / "checkpoints" / "step_000000020.zip"
    second.write_bytes(b"model")
    atomic_write_json(run / "run.json", {
        "schema_version": 1,
        "config": {"algorithm": "maskable_ppo"},
        "contract": incompatible.to_dict(),
        "latest_checkpoint": "checkpoints/step_000000020.zip",
        "latest_checkpoint_step": 20,
    })

    with pytest.raises(ControllerResolutionError, match="encoding hash"):
        binding.reload(lambda candidate: _validate_contract_compatibility(candidate.contract, contract))

    assert binding.resolved is previous


def test_fixed_run_does_not_advance_when_reload_is_requested(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind(
        {"kind": "run", "path": str(run), "mode": "fixed"}
    )

    assert binding.reload() is False


def test_legacy_checkpoint_directory_cannot_create_live_binding(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    checkpoints = tmp_path / "legacy" / "checkpoints"
    checkpoints.mkdir(parents=True)
    first = checkpoints / "first.zip"
    first.write_bytes(b"first")
    with pytest.raises(ControllerResolutionError, match="contract metadata"):
        ControllerResolver(contract, model_loader=loader).bind(f"ppo:{checkpoints}")


def test_legacy_unversioned_run_directory_cannot_create_live_binding(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = tmp_path / "old-run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    first = checkpoints / "first.zip"
    first.write_bytes(b"first")
    with pytest.raises(ControllerResolutionError, match="contract metadata"):
        ControllerResolver(contract, model_loader=loader).bind(f"dqn:{run}")


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
    env.contract = contract
    env._mask = None
    env._rpc = lambda message: {
        "reward": 0.0,
        "terminated": True,
        "truncated": False,
        "obs": [0.0] * contract.observation_size,
        "mask": [True] * contract.action_size,
    }

    SelfPlayEnv.reset(env, seed=1)

    assert binding.resolved.path == second


def test_selfplay_rejects_incompatible_live_candidate_before_replacing_active_binding(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(model_loader=loader).bind(
        {"kind": "run", "path": str(run), "mode": "live"}
    )
    previous = binding.resolved
    second = run / "checkpoints" / "step_000000020.zip"
    second.write_bytes(b"model")
    incompatible = dataclass_replace(contract, encoding_hash="c" * 64)
    atomic_write_json(run / "run.json", {
        "schema_version": 1,
        "config": {"algorithm": "maskable_ppo"},
        "contract": incompatible.to_dict(),
        "latest_checkpoint": "checkpoints/step_000000020.zip",
        "latest_checkpoint_step": 20,
    })
    env = object.__new__(SelfPlayEnv)
    env.opp_pool = [binding]
    env.contract = contract
    env.obs_len = contract.observation_size
    env.n_actions = contract.action_size

    with pytest.raises(ControllerResolutionError, match="encoding hash"):
        env._reload_live_opponents()

    assert binding.resolved is previous


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
            "reward": 0.0,
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


def test_selfplay_accepts_tactical_v2_and_sends_environment_flag(tmp_path: Path) -> None:
    from .test_gym_client import tactical_v2_spaces

    spaces = tactical_v2_spaces(environment_kind="duel")
    server = tmp_path / "fake_duel_server.py"
    spaces_path = tmp_path / "spaces.json"
    spaces_path.write_text(json.dumps(spaces), encoding="utf-8")
    server.write_text(
        """import json
import sys

response = json.load(open(sys.argv[1], encoding="utf-8"))
for line in sys.stdin:
    request = json.loads(line)
    if request["cmd"] == "duel_spaces":
        print(json.dumps(response), flush=True)
    elif request["cmd"] == "close":
        break
""",
        encoding="utf-8",
    )

    env = SelfPlayEnv(
        [sys.executable, str(server), str(spaces_path)],
        ["greedy"],
        environment="tactical-v2",
    )
    try:
        assert env.contract.version == "tactical-v2"
        assert env.proc.args[-2:] == ["--environment", "tactical-v2"]
    finally:
        env.close()


def test_selfplay_rejects_tactical_kind_handshake_for_tactical_v2_duel(tmp_path: Path) -> None:
    from .test_gym_client import tactical_v2_spaces

    spaces = tactical_v2_spaces(environment_kind="tactical")
    server = tmp_path / "fake_wrong_kind_server.py"
    spaces_path = tmp_path / "spaces.json"
    spaces_path.write_text(json.dumps(spaces), encoding="utf-8")
    server.write_text(
        """import json
import sys

response = json.load(open(sys.argv[1], encoding="utf-8"))
for line in sys.stdin:
    request = json.loads(line)
    if request["cmd"] == "duel_spaces":
        print(json.dumps(response), flush=True)
    elif request["cmd"] == "close":
        break
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="environment_kind"):
        SelfPlayEnv(
            [sys.executable, str(server), str(spaces_path)],
            ["greedy"],
            environment="tactical-v2",
        )


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

    scenario_path = tmp_path / "run" / "scenario.json"
    with pytest.raises(KeyError) as constructor_error:
        SelfPlayEnv(
            [sys.executable, "-c", child_code],
            ["greedy"],
            scenario_path=scenario_path,
        )

    process = children[0]
    assert process.args[-2:] == ["--scenario-file", str(scenario_path)]
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


class _FakeSelfPlayServerProcess:
    """A stand-in for subprocess.Popen that answers one queued handshake line."""

    def __init__(self, response_line: str) -> None:
        self._response_line = response_line
        self.stdin = SimpleNamespace(
            write=lambda _payload: None, flush=lambda: None, close=lambda: None,
        )
        self.stdout = SimpleNamespace(
            readline=lambda: self._response_line, close=lambda: None,
        )

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_selfplay_env_passes_no_window_creationflags_to_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as json_module

    captured: dict[str, object] = {}
    response_line = json_module.dumps({"n_actions": 3, "obs_len": 3}) + "\n"

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeSelfPlayServerProcess(response_line)

    def fake_parse_contract(_spaces, *, environment, required_kind):
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(selfplay_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(selfplay_module, "parse_contract", fake_parse_contract)

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        SelfPlayEnv(["dotnet", "server.dll"], ["greedy"])

    assert captured.get("creationflags") == selfplay_module.no_window_creationflags()
