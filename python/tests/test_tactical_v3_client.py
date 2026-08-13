from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import FrozenInstanceError

from hexwars_gym.env import no_window_creationflags

import pytest


class _FakeServer:
    def __init__(self, command: list[str], requests_path: Path, replies_path: Path) -> None:
        self.command = command
        self._requests_path = requests_path
        self._replies_path = replies_path

    @property
    def requests(self) -> list[dict[str, object]]:
        return json.loads(self._requests_path.read_text(encoding="utf-8"))

    def reply_with(self, command: str, payload: object) -> None:
        replies = json.loads(self._replies_path.read_text(encoding="utf-8"))
        replies[command] = payload
        self._replies_path.write_text(json.dumps(replies), encoding="utf-8")


@pytest.fixture
def fake_server(tmp_path: Path) -> _FakeServer:
    requests_path = tmp_path / "requests.json"
    replies_path = tmp_path / "replies.json"
    replies_path.write_text("{}", encoding="utf-8")
    script_path = tmp_path / "fake_tactical_v3_server.py"
    script_path.write_text(
        """import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
sys.path.insert(0, sys.argv[4])
from test_tactical_v3_schema import minimal_spaces_payload, minimal_view_payload

requests_path = Path(sys.argv[1])
replies_path = Path(sys.argv[2])
requests = []
for line in sys.stdin:
    request = json.loads(line)
    requests.append(request)
    requests_path.write_text(json.dumps(requests), encoding="utf-8")
    if request["cmd"] == "close":
        break
    replies = json.loads(replies_path.read_text(encoding="utf-8"))
    if request["cmd"] in replies:
        response = replies[request["cmd"]]
    if request["cmd"] == "spaces":
        response = replies.get(request["cmd"], minimal_spaces_payload())
    elif request["cmd"] == "duel_spaces":
        duel_spaces = json.loads((Path(sys.argv[3]) / "fixtures" / "tactical_v3" /
            "seed-41-duel-spaces.json").read_text(encoding="utf-8"))
        response = replies.get(request["cmd"], duel_spaces)
    elif request["cmd"] in {"reset", "duel_reset"}:
        response = minimal_view_payload()
    elif request["cmd"] in {"step", "duel_step"}:
        response = minimal_view_payload()
        response["candidates"][0]["decision_id"] = 8
    elif request["cmd"] == "duel_oracle_step":
        response = replies.get(request["cmd"])
        if response is None:
            view = minimal_view_payload()
            view["decision_id"] = 8
            view["candidates"][0]["decision_id"] = 8
            response = {"selection": {
                "decision_id": request["decision_id"], "candidate_id": 0,
                "search_depth": 4, "expansion_budget": 512,
                "actual_expansions": 17,
                "heuristic_identity": "material-plus-pursuit-v1",
            }, "view": view}
    elif request["cmd"] == "duel_status":
        response = replies.get(request["cmd"], {"internal_fallback_count": 0})
    elif request["cmd"] == "duel_save":
        response = {"saved": request["path"]}
    else:
        raise RuntimeError(request["cmd"])
    print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    return _FakeServer(
        [sys.executable, str(script_path), str(requests_path), str(replies_path), str(Path(__file__).parent), str(Path(__file__).resolve().parents[1])],
        requests_path,
        replies_path,
    )


def test_tactical_v3_client_public_api_is_importable() -> None:
    """Catches removal or absence of the structured GymServer client API."""
    from ml_lab.tactical_v3_client import (
        CandidateSelection, OracleStepResult, TacticalV3GymClient, TeacherSelection,
    )

    assert CandidateSelection is not None
    assert TeacherSelection is not None
    assert OracleStepResult is not None
    assert TacticalV3GymClient is not None


def test_duel_oracle_step_sends_exact_request_returns_frozen_result_and_status(
    fake_server: _FakeServer,
) -> None:
    from ml_lab.tactical_v3_client import OracleStepResult, TacticalV3GymClient

    with TacticalV3GymClient(fake_server.command, environment_kind="duel") as client:
        initial = client.duel_reset(41, "external", "random", 0, "standard-3v3", 0)
        result = client.duel_oracle_step(initial.decision.decision_id)
        assert type(result) is OracleStepResult
        assert result.selection.decision_id == initial.decision.decision_id
        assert result.selection.candidate_id == initial.decision.candidates[0].candidate_id
        assert result.selection.actual_expansions == 17
        assert result.view.decision.decision_id == 8
        assert client.duel_status() == 0
        with pytest.raises(FrozenInstanceError):
            result.selection.candidate_id = 1  # type: ignore[misc]

    assert fake_server.requests[-3] == {
        "cmd": "duel_oracle_step", "decision_id": 7,
        "search_depth": 4, "expansion_budget": 512,
        "heuristic_identity": "material-plus-pursuit-v1",
    }
    assert fake_server.requests[-2] == {"cmd": "duel_status"}


def test_duel_oracle_step_exact_stale_error_preserves_view_and_allows_retry(
    fake_server: _FakeServer,
) -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    with TacticalV3GymClient(fake_server.command, environment_kind="duel") as client:
        initial = client.duel_reset(41, "external", "random", 0, "standard-3v3", 0)
        fake_server.reply_with(
            "duel_oracle_step", {"error": "tactical-v3 decision id is stale"},
        )
        with pytest.raises(ValueError, match="decision id is stale"):
            client.duel_oracle_step(initial.decision.decision_id)
        fake_server.reply_with("duel_oracle_step", None)
        result = client.duel_oracle_step(initial.decision.decision_id)
        assert result.view.decision.decision_id == 8


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_top", "extra_top", "missing_selection", "extra_selection",
        "bool_integer", "float_integer", "string_integer", "negative_expansions",
        "over_budget", "wrong_decision", "missing_candidate", "wrong_depth",
        "wrong_budget", "wrong_heuristic", "malformed_view", "other_error",
    ],
)
def test_duel_oracle_step_rejects_response_drift(
    fake_server: _FakeServer, mutation: str,
) -> None:
    from tests.test_tactical_v3_schema import minimal_view_payload
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    view = minimal_view_payload()
    view["decision_id"] = 8
    view["candidates"][0]["decision_id"] = 8
    selection: dict[str, object] = {
        "decision_id": 7, "candidate_id": 0, "search_depth": 4,
        "expansion_budget": 512, "actual_expansions": 17,
        "heuristic_identity": "material-plus-pursuit-v1",
    }
    payload: dict[str, object] = {"selection": selection, "view": view}
    if mutation == "missing_top": payload.pop("view")
    elif mutation == "extra_top": payload["extra"] = True
    elif mutation == "missing_selection": selection.pop("actual_expansions")
    elif mutation == "extra_selection": selection["extra"] = True
    elif mutation == "bool_integer": selection["candidate_id"] = True
    elif mutation == "float_integer": selection["search_depth"] = 4.0
    elif mutation == "string_integer": selection["decision_id"] = "7"
    elif mutation == "negative_expansions": selection["actual_expansions"] = -1
    elif mutation == "over_budget": selection["actual_expansions"] = 513
    elif mutation == "wrong_decision": selection["decision_id"] = 6
    elif mutation == "missing_candidate": selection["candidate_id"] = 9
    elif mutation == "wrong_depth": selection["search_depth"] = 3
    elif mutation == "wrong_budget": selection["expansion_budget"] = 511
    elif mutation == "wrong_heuristic": selection["heuristic_identity"] = "wrong"
    elif mutation == "malformed_view": view["extra"] = True
    elif mutation == "other_error": payload = {"error": "other"}
    else: raise AssertionError(mutation)
    fake_server.reply_with("duel_oracle_step", payload)

    with TacticalV3GymClient(fake_server.command, environment_kind="duel") as client:
        initial = client.duel_reset(41, "external", "random", 0, "standard-3v3", 0)
        with pytest.raises((TypeError, ValueError), match="."):
            client.duel_oracle_step(initial.decision.decision_id)


@pytest.mark.parametrize(
    "payload",
    [
        {}, {"internal_fallback_count": 0, "extra": 1},
        {"internal_fallback_count": -1}, {"internal_fallback_count": True},
        {"internal_fallback_count": 0.0},
    ],
)
def test_duel_status_rejects_response_drift(
    fake_server: _FakeServer, payload: dict[str, object],
) -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    fake_server.reply_with("duel_status", payload)
    with TacticalV3GymClient(fake_server.command, environment_kind="duel") as client:
        with pytest.raises((TypeError, ValueError), match="."):
            client.duel_status()


def test_client_sends_both_candidate_identity_fields_and_rejects_stale_reply(
    fake_server: _FakeServer,
) -> None:
    from ml_lab.tactical_v3_client import CandidateSelection, TacticalV3GymClient

    with TacticalV3GymClient(fake_server.command, environment_kind="tactical") as client:
        view = client.reset(41)
        selected = CandidateSelection(
            view.decision.decision_id,
            view.decision.candidates[0].candidate_id,
        )
        with pytest.raises(ValueError, match="candidate decision_id does not match"):
            client.step(selected)

    step_request = next(
        request for request in reversed(fake_server.requests)
        if request["cmd"] == "step"
    )
    assert step_request == {
        "cmd": "step",
        "decision_id": selected.decision_id,
        "candidate_id": selected.candidate_id,
    }


def test_close_is_idempotent_and_sends_protocol_close(fake_server: _FakeServer) -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    client = TacticalV3GymClient(fake_server.command, environment_kind="tactical")
    client.close()
    client.close()

    assert fake_server.requests[-1] == {"cmd": "close"}


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "tactical_v3"
CHECKED_IN_SCENARIO = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
SCENARIO_24X16 = FIXTURES / "scenario-24x16.json"
SPACES_FIXTURE = FIXTURES / "seed-41-spaces.json"
DUEL_SPACES_FIXTURE = FIXTURES / "seed-41-duel-spaces.json"
DECISION_FIXTURE = FIXTURES / "seed-41-decision.json"
SERVER_DLL = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
CAPTURE_ENV = "HEXWARS_CAPTURE_TACTICAL_V3_FIXTURES"


def _write_fixture(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _raw_request(process: subprocess.Popen[str], request: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert type(reply) is dict
    return reply


def _capture_fixtures() -> None:
    if os.environ.get(CAPTURE_ENV) != "1":
        return
    large = json.loads(CHECKED_IN_SCENARIO.read_text(encoding="utf-8"))
    standard = json.loads(CHECKED_IN_SCENARIO.read_text(encoding="utf-8"))
    large["id"] = "annihilation-structured-imitation-24x16"
    large["name"] = "Annihilation Structured Imitation 24x16"
    large["board"]["width"] = 24
    large["board"]["height"] = 16
    assert {key for key in large if large[key] != standard[key]} == {"id", "name", "board"}
    assert {key for key in large["board"] if large["board"][key] != standard["board"][key]} == {"width", "height"}
    _write_fixture(SCENARIO_24X16, large)
    process = subprocess.Popen(["dotnet", str(SERVER_DLL), "--scenario-file", str(CHECKED_IN_SCENARIO), "--environment", "tactical-v3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", creationflags=no_window_creationflags())
    try:
        _write_fixture(SPACES_FIXTURE, _raw_request(process, {"cmd": "spaces"}))
        _write_fixture(DUEL_SPACES_FIXTURE, _raw_request(process, {"cmd": "duel_spaces"}))
        _write_fixture(DECISION_FIXTURE, _raw_request(process, {"cmd": "reset", "seed": 41}))
        assert process.stdin is not None
        process.stdin.write('{"cmd":"close"}\n')
        process.stdin.flush()
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try: process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=2)


def _spaces_for(scenario: Path):
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    with TacticalV3GymClient(["dotnet", str(SERVER_DLL), "--scenario-file", str(scenario)], environment_kind="tactical") as client:
        return client.identity


def test_real_server_13x9_and_24x16_share_encoding_not_match_hash() -> None:
    _capture_fixtures()
    standard = _spaces_for(CHECKED_IN_SCENARIO)
    large = _spaces_for(SCENARIO_24X16)
    assert standard.encoding_hash == large.encoding_hash
    assert standard.capacity_hash == large.capacity_hash
    assert standard.contract_hash != large.contract_hash


def test_checked_in_fixtures_are_canonical_project_a_wire_values() -> None:
    _capture_fixtures()
    spaces = json.loads(SPACES_FIXTURE.read_text(encoding="utf-8"))
    duel_spaces = json.loads(DUEL_SPACES_FIXTURE.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_FIXTURE.read_text(encoding="utf-8"))
    process = subprocess.Popen(["dotnet", str(SERVER_DLL), "--scenario-file", str(CHECKED_IN_SCENARIO), "--environment", "tactical-v3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert _raw_request(process, {"cmd": "spaces"}) == spaces
        assert _raw_request(process, {"cmd": "duel_spaces"}) == duel_spaces
        assert _raw_request(process, {"cmd": "reset", "seed": 41}) == decision
        assert process.stdin is not None
        process.stdin.write('{"cmd":"close"}\n'); process.stdin.flush()
    finally:
        if process.stdin is not None: process.stdin.close()
        process.wait(timeout=2)
    assert len(decision["observation"]["cells"]) == 117
    assert {"obs", "mask", "obs_len", "n_actions"}.isdisjoint(spaces)
    assert duel_spaces["environment_kind"] == "duel"
    assert duel_spaces["contract_hash"] == "bac4af4d4b8e68466ffaf37c2721f98129edc93b90f529999ba45463cd921437"
    assert duel_spaces["encoding_hash"] == spaces["encoding_hash"]
    assert duel_spaces["capacity_hash"] == spaces["capacity_hash"]
    assert {"obs", "mask", "obs_len", "n_actions"}.isdisjoint(decision)

    process = subprocess.Popen(["dotnet", str(SERVER_DLL), "--scenario-file", str(SCENARIO_24X16), "--environment", "tactical-v3"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        large_spaces = _raw_request(process, {"cmd": "spaces"})
        large_view = _raw_request(process, {"cmd": "reset", "seed": 41})
        assert large_spaces["scenario_id"] == "annihilation-structured-imitation-24x16"
        assert large_spaces["match"]["board"]["width"] == 24 and large_spaces["match"]["board"]["height"] == 16
        assert len(large_view["observation"]["cells"]) == 384
        assert large_spaces["encoding_hash"] == spaces["encoding_hash"]
        assert large_spaces["capacity_hash"] == spaces["capacity_hash"]
        assert large_spaces["contract_hash"] != spaces["contract_hash"]
        assert {"obs", "mask", "obs_len", "n_actions"}.isdisjoint(large_view)
        assert process.stdin is not None
        process.stdin.write('{"cmd":"close"}\n'); process.stdin.flush()
    finally:
        if process.stdin is not None: process.stdin.close()
        process.wait(timeout=2)

@pytest.mark.parametrize("reply, expected", [("not-json\\n", "not valid JSON"), ("", "closed unexpectedly")])
def test_client_fails_closed_on_malformed_or_eof_handshake(tmp_path: Path, reply: str, expected: str) -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient

    script = tmp_path / "bad.py"
    script.write_text("import sys\nsys.stdin.readline()\nsys.stderr.write('x' * 20000)\nsys.stderr.flush()\nsys.stdout.write(" + repr(reply) + ")\nsys.stdout.flush()\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match=expected) as raised:
        TacticalV3GymClient([sys.executable, str(script)], environment_kind="tactical")
    assert len(str(raised.value).split("GymServer stderr tail: ")[-1]) <= 8192


def test_client_timeout_is_bounded_and_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml_lab.tactical_v3_client as module
    monkeypatch.setattr(module, "_REPLY_TIMEOUT_SECONDS", 0.05)
    script = tmp_path / "hang.py"
    script.write_text("import sys,time\nsys.stdin.readline()\ntime.sleep(30)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="timed out"):
        module.TacticalV3GymClient([sys.executable, str(script)], environment_kind="tactical")
