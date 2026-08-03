from pathlib import Path

import pytest

import duel


def test_duel_defaults_are_resolved_from_repository_root() -> None:
    repository_root = Path(duel.__file__).resolve().parents[1]

    assert duel.DEFAULT_SERVER == (
        repository_root
        / "engine"
        / "HexWars.GymServer"
        / "bin"
        / "Release"
        / "net8.0"
        / "HexWars.GymServer.dll"
    )


def test_duel_closes_server_when_handshake_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = object()
    launched: list[list[str]] = []
    closed: list[object] = []

    def fake_popen(command, **_kwargs):
        launched.append(command)
        return process

    monkeypatch.setattr(duel.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(duel, "rpc", lambda _proc, _message: (_ for _ in ()).throw(
        RuntimeError("bad handshake")
    ))
    monkeypatch.setattr(duel, "_close_process", closed.append)
    monkeypatch.setattr(
        duel.argparse.ArgumentParser,
        "parse_args",
        lambda _self: type("Args", (), {
            "p0": "greedy",
            "p1": "random",
            "server": str(duel.DEFAULT_SERVER),
            "seed": 0,
            "out": "duel.replay",
            "environment": "adaptive-v1",
        })(),
    )

    with pytest.raises(RuntimeError, match="bad handshake"):
        duel.main()

    assert launched == [["dotnet", str(duel.DEFAULT_SERVER), "--environment", "adaptive-v1"]]
    assert closed == [process]


def test_duel_sends_tactical_v2_and_requires_duel_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = object()
    launched: list[list[str]] = []
    closed: list[object] = []
    captured: dict[str, str] = {}

    def fake_popen(command, **_kwargs):
        launched.append(command)
        return process

    def fake_parse_contract(_spaces, *, environment, required_kind):
        captured["environment"] = environment
        captured["required_kind"] = required_kind
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(duel.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(duel, "rpc", lambda _proc, _message: {"cmd": "duel_spaces"})
    monkeypatch.setattr(duel, "parse_contract", fake_parse_contract)
    monkeypatch.setattr(duel, "_close_process", closed.append)
    monkeypatch.setattr(
        duel.argparse.ArgumentParser,
        "parse_args",
        lambda _self: type("Args", (), {
            "p0": "greedy",
            "p1": "random",
            "server": str(duel.DEFAULT_SERVER),
            "seed": 0,
            "out": "duel.replay",
            "environment": "tactical-v2",
        })(),
    )

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        duel.main()

    assert launched == [["dotnet", str(duel.DEFAULT_SERVER), "--environment", "tactical-v2"]]
    assert captured == {"environment": "tactical-v2", "required_kind": "duel"}
    assert closed == [process]


def test_duel_rejects_unknown_environment_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["duel.py", "--environment", "tactical-v3"])

    with pytest.raises(SystemExit):
        duel.main()


def test_duel_passes_no_window_creationflags_to_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = object()
    captured: dict[str, object] = {}
    closed: list[object] = []

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(duel.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(duel, "rpc", lambda _proc, _message: (_ for _ in ()).throw(
        RuntimeError("bad handshake")
    ))
    monkeypatch.setattr(duel, "_close_process", closed.append)
    monkeypatch.setattr(
        duel.argparse.ArgumentParser,
        "parse_args",
        lambda _self: type("Args", (), {
            "p0": "greedy",
            "p1": "random",
            "server": str(duel.DEFAULT_SERVER),
            "seed": 0,
            "out": "duel.replay",
            "environment": "tactical-v1",
        })(),
    )

    with pytest.raises(RuntimeError, match="bad handshake"):
        duel.main()

    assert captured.get("creationflags") == duel.no_window_creationflags()
