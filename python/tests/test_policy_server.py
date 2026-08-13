from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tests.test_tactical_v3_controller import make_structured_run_case


@dataclass(frozen=True, slots=True)
class PolicyServerCase:
    args: tuple[str, ...]
    seat: int
    view_payload: Mapping[str, object]
    legal_identities: frozenset[tuple[int, int]]


def make_policy_server_case(tmp_path: Path) -> PolicyServerCase:
    structured = make_structured_run_case(tmp_path)
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "tactical_v3" / "seed-41-decision.json").read_text(
            encoding="utf-8"
        )
    )
    script = Path(__file__).resolve().parents[1] / "policy_server.py"
    return PolicyServerCase(
        (
            sys.executable, str(script), "--p0", f"run:{structured.run_dir}",
            "--expected-environment", "tactical-v3",
            "--expected-contract-version", "tactical-v3",
            "--expected-encoding-hash", structured.identity.encoding_hash,
            "--expected-capacity-hash", structured.identity.capacity_hash,
        ),
        0,
        payload,
        frozenset(
            (candidate["decision_id"], candidate["candidate_id"])
            for candidate in payload["candidates"]
        ),
    )


def without_argument(args: tuple[str, ...], flag: str) -> tuple[str, ...]:
    index = args.index(flag)
    assert args.count(flag) == 1
    return args[:index] + args[index + 2:]


@contextmanager
def start_policy_server(args: tuple[str, ...]) -> Iterator[tuple[subprocess.Popen[str], Mapping[str, object]]]:
    process = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    assert process.stdout is not None
    ready = json.loads(process.stdout.readline())
    try:
        yield process, ready
    finally:
        if process.poll() is None:
            assert process.stdin is not None
            process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            process.stdin.flush()
            process.terminate()
        process.communicate(timeout=30)


def request(process: subprocess.Popen[str], payload: Mapping[str, object]) -> Mapping[str, object]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


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


def test_policy_expectation_accepts_tactical_v2() -> None:
    from policy_server import PolicyExpectation

    assert PolicyExpectation("tactical-v2", "tactical-v2", "a" * 64).encoding_hash == "a" * 64


def test_policy_expectation_requires_capacity_hash_for_tactical_v3() -> None:
    from policy_server import PolicyExpectation

    import pytest
    with pytest.raises(ValueError, match="expected-capacity-hash"):
        PolicyExpectation("tactical-v3", "tactical-v3", "a" * 64)


def test_legacy_expectation_rejects_capacity_hash() -> None:
    from policy_server import PolicyExpectation

    import pytest
    with pytest.raises(ValueError, match="capacity hash is valid only for tactical-v3"):
        PolicyExpectation("tactical-v1", "tactical-v1", "a" * 64, "b" * 64)


def test_tactical_v3_requires_expected_capacity_hash(tmp_path: Path) -> None:
    case = make_policy_server_case(tmp_path)
    result = subprocess.run(
        without_argument(case.args, "--expected-capacity-hash"),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--expected-capacity-hash is required for tactical-v3" in result.stderr


def test_tactical_v3_request_returns_exact_legal_candidate_identity(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    with start_policy_server(case.args) as (server, ready):
        assert ready["ready"] is True
        response = request(server, {"seat": case.seat, "decision": case.view_payload})

    assert set(response) == {"decision_id", "candidate_id"}
    assert response["decision_id"] == case.view_payload["decision_id"]
    assert (response["decision_id"], response["candidate_id"]) in case.legal_identities


def test_tactical_v3_request_rejects_flat_or_mixed_payloads(tmp_path: Path) -> None:
    case = make_policy_server_case(tmp_path)
    invalid = (
        {"seat": case.seat, "obs": [0.0], "mask": [True]},
        {"seat": case.seat, "decision": case.view_payload, "obs": [0.0], "mask": [True]},
        {"seat": case.seat, "decision": case.view_payload, "extra": 1},
    )
    with start_policy_server(case.args) as (server, _ready):
        for payload in invalid:
            response = request(server, payload)
            assert set(response) == {"error"}
            assert "structured policy request fields" in response["error"]


def test_policy_expectation_rejects_tactical_v1_model_for_tactical_v2_expectation() -> None:
    from policy_server import PolicyExpectation, validate_resolved_contract

    class Contract:
        environment = "tactical-v1"
        version = "tactical-v1"
        encoding_hash = "a" * 64

    class Resolved:
        contract = Contract()

    expected = PolicyExpectation("tactical-v2", "tactical-v2", "a" * 64)

    import pytest
    with pytest.raises(ValueError, match="environment"):
        validate_resolved_contract(Resolved(), expected)


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


def test_policy_server_subprocess_accepts_tactical_v2_expectation(tmp_path: Path) -> None:
    """tactical-v2 is a recognized expected environment/version — the subprocess still
    fails closed on the encoding-hash mismatch, not on 'unsupported' environment."""
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
            "environment": "tactical-v2",
            "version": "tactical-v2",
            "contract_hash": "c" * 64,
            "encoding_hash": "a" * 64,
            "observation_size": 3,
            "action_size": 2,
            "board": {"width": 1, "height": 1},
            "roster": ["brute-85597320:Brute:7,2,2,3,2,1,1,2,1"],
            "reward": {"terminal_win": 1.0},
            "semantics": {},
        },
    }
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "policy_server.py"

    completed = subprocess.run(
        [
            sys.executable, str(script), "--p0", f"run:{run}",
            "--expected-environment", "tactical-v2",
            "--expected-contract-version", "tactical-v2",
            "--expected-encoding-hash", "b" * 64,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "encoding hash" in completed.stderr
    assert completed.stdout == ""
