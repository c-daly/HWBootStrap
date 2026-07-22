from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class _TinyEnv(gym.Env):
    observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}


def test_seat_models_is_structured_and_stably_ordered() -> None:
    from policy_server import seat_models

    class FakeSeat:
        def __init__(self, algorithm: str, step: int) -> None:
            self.algorithm = algorithm
            self.step = step

        def metadata(self) -> dict:
            return {
                "kind": "run",
                "inference_mode": "deterministic",
                "path": f"{self.algorithm}.zip",
                "algorithm": self.algorithm,
                "step": self.step,
                "contract_hash": "c" * 64,
                "contract_version": "adaptive-v1",
                "environment": "adaptive-v1",
                "encoding_hash": "d" * 64,
            }

    assert seat_models(
        {1: FakeSeat("masked_dqn", 96), 0: FakeSeat("maskable_ppo", 64)}
    ) == [
        {
            "seat": 0,
            "kind": "run",
            "inference_mode": "deterministic",
            "path": "maskable_ppo.zip",
            "algorithm": "maskable_ppo",
            "step": 64,
            "contract_hash": "c" * 64,
            "contract_version": "adaptive-v1",
            "environment": "adaptive-v1",
            "encoding_hash": "d" * 64,
        },
        {
            "seat": 1,
            "kind": "run",
            "inference_mode": "deterministic",
            "path": "masked_dqn.zip",
            "algorithm": "masked_dqn",
            "step": 96,
            "contract_hash": "c" * 64,
            "contract_version": "adaptive-v1",
            "environment": "adaptive-v1",
            "encoding_hash": "d" * 64,
        },
    ]


def test_predict_for_seat_uses_resolved_stochastic_mode(monkeypatch) -> None:
    import policy_server

    calls: list[bool] = []
    resolved = SimpleNamespace(
        model=object(),
        algorithm="maskable_ppo",
        spec=SimpleNamespace(inference_mode="stochastic"),
    )
    seat = SimpleNamespace(resolved=resolved)
    monkeypatch.setattr(
        policy_server,
        "predict",
        lambda model, algorithm, observation, mask, *, deterministic: (
            calls.append(deterministic) or 4
        ),
    )

    action = policy_server.predict_for_seat(
        seat,
        np.zeros(3, dtype=np.float32),
        np.array([True, False]),
    )

    assert action == 4
    assert calls == [False]


def test_policy_expectation_rejects_model_encoding_mismatch() -> None:
    from policy_server import PolicyExpectation, validate_resolved_contract

    class Contract:
        environment = "adaptive-v1"
        version = "adaptive-v1"
        encoding_hash = "e" * 64

    class Resolved:
        contract = Contract()

    expected = PolicyExpectation("adaptive-v1", "adaptive-v1", "d" * 64)

    import pytest
    with pytest.raises(ValueError, match="encoding hash"):
        validate_resolved_contract(Resolved(), expected)


def test_policy_expectation_allows_scripted_only_server_without_model_metadata() -> None:
    from policy_server import PolicyExpectation

    assert PolicyExpectation("tactical-v1", "tactical-v1", "a" * 64).encoding_hash == "a" * 64


def test_policy_server_subprocess_rejects_encoding_mismatch_before_ready(tmp_path: Path) -> None:
    from sb3_contrib import MaskablePPO

    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "model.zip"
    checkpoint.parent.mkdir(parents=True)
    MaskablePPO("MlpPolicy", _TinyEnv(), n_steps=2, batch_size=2, verbose=0).save(checkpoint)
    manifest = {
        "schema_version": 1,
        "config": {"algorithm": "maskable_ppo"},
        "latest_checkpoint": "checkpoints/model.zip",
        "latest_checkpoint_step": 0,
        "contract": {
            "environment": "tactical-v1",
            "version": "tactical-v1",
            "contract_hash": "c" * 64,
            "encoding_hash": "a" * 64,
            "observation_size": 3,
            "action_size": 2,
            "board": {"width": 1, "height": 1},
            "roster": ["scout"],
            "reward": {"terminal_win": 1.0},
            "semantics": {},
        },
    }
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "policy_server.py"

    completed = subprocess.run(
        [
            sys.executable, str(script), "--p0", f"run:{run}",
            "--expected-environment", "tactical-v1",
            "--expected-contract-version", "tactical-v1",
            "--expected-encoding-hash", "b" * 64,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "encoding hash" in completed.stderr
    assert completed.stdout == ""
