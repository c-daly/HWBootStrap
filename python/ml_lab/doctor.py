"""Headless dependency and GymServer health checks for HexWars ML Lab."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

from hexwars_gym.env import parse_contract


PackageVersion = Callable[[str], str]
DotnetVersion = Callable[[], str]
Handshake = Callable[[Sequence[str]], dict[str, Any]]
CudaInfo = Callable[[], dict[str, Any]]
TrackerAvailable = Callable[[str], bool]
WriteProbe = Callable[[Path], bool]

REQUIRED_PACKAGES = (
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
    "numpy",
)


def _package_version(name: str) -> str:
    return metadata.version(name)


def _dotnet_version() -> str:
    completed = subprocess.run(
        ["dotnet", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def _gymserver_handshake(command: Sequence[str]) -> dict[str, Any]:
    payload = "\n".join(
        (json.dumps({"cmd": "spaces"}), json.dumps({"cmd": "close"}), "")
    )
    completed = subprocess.run(
        list(command),
        input=payload,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("GymServer returned no handshake")
    response = json.loads(lines[0])
    version = response.get("contract_version")
    contract_hash = response.get("contract_hash")
    encoding_hash = response.get("encoding_hash")
    if not isinstance(version, str) or not version:
        raise RuntimeError("GymServer handshake omitted contract_version")
    if not isinstance(contract_hash, str) or not contract_hash:
        raise RuntimeError("GymServer handshake omitted contract_hash")
    if not isinstance(encoding_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", encoding_hash):
        raise RuntimeError("GymServer handshake omitted a valid lowercase encoding_hash")
    return response


def _validate_selected_contract(response: dict[str, Any], environment: str) -> dict[str, Any]:
    required_kind = "adaptive_tactical" if environment == "adaptive-v1" else "tactical"
    parse_contract(response, environment=environment, required_kind=required_kind)
    return response


def _cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"available": False, "detail": f"torch unavailable: {error}"}
    available = bool(torch.cuda.is_available())
    detail = torch.cuda.get_device_name(0) if available else "CPU-only host"
    return {"available": available, "detail": detail}


def _tracker_available(name: str) -> bool:
    if name.startswith("custom="):
        name = name.removeprefix("custom=").split(":", 1)[0]
    if name in {"local", "tensorboard"}:
        module = "tensorboard" if name == "tensorboard" else "json"
    elif name == "wandb":
        module = "wandb"
    else:
        module = name.split(":", 1)[0]
    return importlib.util.find_spec(module) is not None


def _write_probe(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-", delete=True):
        return True


def _check(name: str, ok: bool, required: bool, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "detail": detail,
    }


def _attempt(
    name: str,
    required: bool,
    operation: Callable[[], Any],
    detail: Callable[[Any], str] = str,
) -> dict[str, Any]:
    try:
        value = operation()
        return _check(name, True, required, detail(value))
    except Exception as error:
        return _check(name, False, required, f"{type(error).__name__}: {error}")


def doctor_environment(
    *,
    server_cmd: Sequence[str],
    environment: str = "tactical-v1",
    runs_root: Path,
    trackers: Sequence[str] = (),
    package_version: PackageVersion = _package_version,
    dotnet_version: DotnetVersion = _dotnet_version,
    handshake: Handshake = _gymserver_handshake,
    cuda_info: CudaInfo = _cuda_info,
    tracker_available: TrackerAvailable = _tracker_available,
    write_probe: WriteProbe = _write_probe,
) -> dict[str, Any]:
    """Return structured health without depending on Unity or any remote tracker."""
    checks = [
        _attempt(
            f"python:{package}",
            True,
            lambda package=package: package_version(package),
        )
        for package in REQUIRED_PACKAGES
    ]
    checks.append(_attempt("dotnet", True, dotnet_version))
    checks.append(
        _attempt(
            "gymserver_handshake",
            True,
            lambda: _validate_selected_contract(
                handshake((*server_cmd, "--environment", environment)), environment
            ),
            lambda value: (
                f"{value.get('contract_version')} {value.get('contract_hash')}"
            ),
        )
    )
    checks.append(
        _attempt(
            "write_access",
            True,
            lambda: write_probe(Path(runs_root)),
            lambda _value: str(Path(runs_root).resolve()),
        )
    )
    try:
        cuda = cuda_info()
        checks.append(
            _check(
                "cuda",
                bool(cuda.get("available")),
                False,
                str(cuda.get("detail", "unknown")),
            )
        )
    except Exception as error:
        checks.append(
            _check("cuda", False, False, f"{type(error).__name__}: {error}")
        )
    for tracker in trackers:
        checks.append(
            _attempt(
                f"tracker:{tracker}",
                False,
                lambda tracker=tracker: tracker_available(tracker),
                lambda available: "available" if available else "unavailable",
            )
        )
        if checks[-1]["detail"] == "unavailable":
            checks[-1]["ok"] = False
    return {
        "ok": all(check["ok"] for check in checks if check["required"]),
        "python": sys.version.split()[0],
        "checks": checks,
    }
