from __future__ import annotations

import json
import subprocess
import sys
from contextlib import AbstractContextManager
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
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


def replace_argument(
    args: tuple[str, ...], flag: str, value: str,
) -> tuple[str, ...]:
    index = args.index(flag)
    assert args.count(flag) == 1
    return args[:index + 1] + (value,) + args[index + 2:]


def run_policy_server_until_exit(
    args: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, input="", capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


def bounded_readline(
    process: subprocess.Popen[str],
    stream,
) -> str:
    lines: Queue[str] = Queue(maxsize=1)
    reader = Thread(target=lambda: lines.put(stream.readline()), daemon=True)
    reader.start()
    try:
        return lines.get(timeout=30)
    except Empty as error:
        process.kill()
        process.wait(timeout=30)
        raise AssertionError("policy server JSONL read timed out") from error


class PolicyServerProcess(AbstractContextManager["PolicyServerProcess"]):
    ready: Mapping[str, object]

    def __init__(self, args: tuple[str, ...]) -> None:
        self.process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        assert self.process.stdout is not None
        ready_line = bounded_readline(self.process, self.process.stdout)
        if not ready_line:
            _stdout, stderr = self.process.communicate(timeout=30)
            raise AssertionError(f"policy server exited before ready: {stderr[:4096]}")
        self.ready = json.loads(ready_line)

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return json.loads(bounded_readline(self.process, self.process.stdout))

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=30)
        assert self.process.stderr is not None
        stderr = self.process.stderr.read(65537)
        assert len(stderr) <= 65536, "policy server stderr exceeded bound"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self.process.poll() is None:
            self.close()


def start_policy_server(args: tuple[str, ...]) -> PolicyServerProcess:
    return PolicyServerProcess(args)


def request_once(
    args: tuple[str, ...],
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    with start_policy_server(args) as server:
        assert server.ready["ready"] is True
        return server.request(payload)


def run_request_lines(
    args: tuple[str, ...],
    payloads: tuple[Mapping[str, object], ...],
) -> tuple[subprocess.CompletedProcess[str], tuple[Mapping[str, object], ...]]:
    completed = subprocess.run(
        args,
        input="".join(json.dumps(payload) + "\n" for payload in payloads),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    replies = tuple(json.loads(line) for line in completed.stdout.splitlines())
    return completed, replies


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
    result = run_policy_server_until_exit(
        without_argument(case.args, "--expected-capacity-hash")
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--expected-capacity-hash is required for tactical-v3" in result.stderr


def test_tactical_v3_request_returns_exact_legal_candidate_identity(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    with start_policy_server(case.args) as server:
        assert server.ready["ready"] is True
        response = server.request(
            {"seat": case.seat, "decision": case.view_payload}
        )

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
    with start_policy_server(case.args) as server:
        for payload in invalid:
            response = server.request(payload)
            assert set(response) == {"error"}
            assert "structured policy request fields" in response["error"]


def test_structured_response_is_deterministic_across_server_restarts(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    payload = {"seat": case.seat, "decision": case.view_payload}
    first = request_once(case.args, payload)
    second = request_once(case.args, payload)
    assert first == second
    assert (first["decision_id"], first["candidate_id"]) in case.legal_identities


def test_wrong_encoding_or_capacity_fails_before_tensor_load_and_ready(
    tmp_path: Path,
) -> None:
    for flag in ("--expected-encoding-hash", "--expected-capacity-hash"):
        case_root = tmp_path / flag.removeprefix("--expected-")
        case_root.mkdir()
        case = make_policy_server_case(case_root)
        run_arg = case.args[case.args.index("--p0") + 1]
        run_dir = Path(run_arg.removeprefix("run:"))
        (run_dir / "checkpoints" / "best.pt").write_bytes(b"not a checkpoint")
        result = run_policy_server_until_exit(
            replace_argument(case.args, flag, "0" * 64)
        )
        assert result.returncode != 0
        assert result.stdout == ""
        assert "does not match expected" in result.stderr


def test_structured_request_seat_requires_exact_built_in_int(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    payloads = tuple(
        {"seat": invalid, "decision": case.view_payload}
        for invalid in (True, 0.0, "0")
    ) + ({"cmd": "close"},)
    completed, lines = run_request_lines(case.args, payloads)
    assert completed.returncode == 0
    assert lines[0]["ready"] is True
    assert len(lines) == 4
    for response in lines[1:]:
        assert set(response) == {"error"}
        assert "seat must be a built-in int" in response["error"]


def test_command_fields_are_exact_before_reload_or_close_routing(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    completed, lines = run_request_lines(case.args, (
        {"cmd": "reload", "extra": 1},
        {"cmd": "close", "extra": 1},
        {"cmd": "reload"},
        {"seat": case.seat, "decision": case.view_payload},
        {"cmd": "close"},
    ))
    assert completed.returncode == 0
    assert lines[0]["ready"] is True
    assert len(lines) == 5
    for response in lines[1:3]:
        assert set(response) == {"error"}
        assert "command fields" in response["error"]
    assert set(lines[3]) == {"reloaded", "seats", "seat_models"}
    assert set(lines[4]) == {"decision_id", "candidate_id"}


def test_live_structured_reload_keeps_old_model_then_swaps_valid_replacement(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    run_arg = case.args[case.args.index("--p0") + 1]
    original_run = Path(run_arg.removeprefix("run:"))
    invalid_parent = tmp_path / "invalid"
    invalid_parent.mkdir()
    invalid = make_structured_run_case(invalid_parent, best_epoch=1)
    (invalid.run_dir / "checkpoints" / "best.pt").write_bytes(
        b"not a checkpoint"
    )
    replacement_parent = tmp_path / "replacement"
    replacement_parent.mkdir()
    replacement = make_structured_run_case(replacement_parent, best_epoch=2)
    live_spec = json.dumps({
        "kind": "run",
        "path": str(original_run),
        "mode": "live",
    })
    live_args = replace_argument(case.args, "--p0", live_spec)
    payload = {"seat": case.seat, "decision": case.view_payload}

    with start_policy_server(live_args) as server:
        baseline = server.request(payload)
        original_run.rename(tmp_path / "original")
        invalid.run_dir.rename(original_run)
        rejected = server.request({"cmd": "reload"})
        assert set(rejected) == {"error"}
        assert server.request(payload) == baseline

        original_run.rename(tmp_path / "rejected")
        replacement.run_dir.rename(original_run)
        reloaded = server.request({"cmd": "reload"})
        assert reloaded["reloaded"] == [0]
        assert reloaded["seats"]["0"]["step"] == 2
        selected = server.request(payload)
        assert (
            selected["decision_id"], selected["candidate_id"]
        ) in case.legal_identities


def test_capacity_hash_without_complete_expectation_fails_before_ready(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    result = run_policy_server_until_exit(
        case.args[:2] + ("--expected-capacity-hash", case.args[-1])
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "expected-capacity-hash" in result.stderr


def test_legacy_exact_flat_request_and_action_protocol_remains_supported(
    tmp_path: Path,
) -> None:
    from sb3_contrib import MaskablePPO

    run = tmp_path / "legacy-run"
    checkpoint = run / "checkpoints" / "model.zip"
    checkpoint.parent.mkdir(parents=True)
    MaskablePPO(
        "MlpPolicy", _TinyEnv(), n_steps=2, batch_size=2, verbose=0
    ).save(checkpoint)
    (run / "run.json").write_text(json.dumps({
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
    }), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "policy_server.py"
    args = (
        sys.executable, str(script), "--p0", f"run:{run}",
        "--expected-environment", "tactical-v1",
        "--expected-contract-version", "tactical-v1",
        "--expected-encoding-hash", "a" * 64,
    )

    with start_policy_server(args) as server:
        response = server.request({
            "seat": 0,
            "obs": [0.0, 0.0, 0.0],
            "mask": [True, False],
        })
    assert response == {"action": 0}


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
