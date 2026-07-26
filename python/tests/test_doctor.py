"""Unit tests for python/ml_lab/doctor.py's direct subprocess call sites.

doctor_environment() itself takes injected handshake/dotnet_version callables (see
test_cli.py), so it never exercises the real subprocess.run calls. These tests target
the two private functions that actually spawn processes: _dotnet_version() and
_gymserver_handshake().
"""
from __future__ import annotations

import json

import pytest

import ml_lab.doctor as doctor_module


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_dotnet_version_passes_no_window_creationflags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        return _FakeCompleted("8.0.100\n")

    monkeypatch.setattr(doctor_module.subprocess, "run", fake_run)

    assert doctor_module._dotnet_version() == "8.0.100"
    assert captured["command"] == ["dotnet", "--version"]
    assert captured.get("creationflags") == doctor_module.no_window_creationflags()


def test_gymserver_handshake_passes_no_window_creationflags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    response = {
        "contract_version": "tactical-v1",
        "contract_hash": "a" * 64,
        "encoding_hash": "b" * 64,
    }

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        return _FakeCompleted(json.dumps(response) + "\n")

    monkeypatch.setattr(doctor_module.subprocess, "run", fake_run)

    result = doctor_module._gymserver_handshake(["dotnet", "server.dll"])

    assert result == response
    assert captured["command"] == ["dotnet", "server.dll"]
    assert captured.get("creationflags") == doctor_module.no_window_creationflags()
