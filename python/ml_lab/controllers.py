"""Normalize and resolve scripted and trained HexWars controllers.

Run manifests are authoritative for published model checkpoints.  Older standalone
checkpoints are intentionally supported only through an explicit algorithm choice;
they can be used for inference after inspecting their spaces, but are never treated
as promotable experiment artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .contracts import EnvironmentContract
from .io import read_json


Algorithm = Literal["maskable_ppo", "masked_dqn"]
SCRIPTED_NAMES = frozenset({"greedy", "random"})
ALGORITHM_ALIASES: dict[str, Algorithm] = {"ppo": "maskable_ppo", "dqn": "masked_dqn"}
SUPPORTED_ENCODING_VERSIONS = frozenset({"tactical-v1"})


class ControllerResolutionError(ValueError):
    """Raised when a controller cannot safely be used for inference."""


@dataclass(frozen=True)
class ControllerSpec:
    kind: Literal["scripted", "checkpoint", "run"]
    name: str | None = None
    path: Path | None = None
    algorithm: Algorithm | None = None
    mode: Literal["fixed", "live"] = "fixed"


@dataclass(frozen=True)
class ResolvedController:
    spec: ControllerSpec
    server_controller: str
    model: Any | None
    path: Path | None
    algorithm: Algorithm | None
    step: int | None
    contract: EnvironmentContract | None
    observation_size: int | None
    action_size: int | None
    legacy: bool
    promotable: bool

    def metadata(self) -> dict[str, Any]:
        """JSON-safe information for policy-server status replies."""
        return {
            "kind": self.spec.kind,
            "path": str(self.path) if self.path is not None else None,
            "algorithm": self.algorithm,
            "step": self.step,
            "contract_hash": self.contract.contract_hash if self.contract is not None else None,
            "contract": self.contract.to_dict() if self.contract is not None else None,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "legacy": self.legacy,
            "promotable": self.promotable,
        }


ModelLoader = Callable[[Path, Algorithm], Any]


def normalize_controller_spec(raw: str | Mapping[str, Any] | ControllerSpec) -> ControllerSpec:
    """Parse a JSON-compatible controller spec and the old ``ppo:PATH`` boundary form."""
    if isinstance(raw, ControllerSpec):
        return raw
    if isinstance(raw, str):
        raw = _parse_string_spec(raw)
    if not isinstance(raw, Mapping):
        raise ControllerResolutionError("controller specification must be a string or object")

    kind = raw.get("kind")
    if kind == "scripted":
        name = raw.get("name")
        if name not in SCRIPTED_NAMES:
            raise ControllerResolutionError("scripted controller name must be 'greedy' or 'random'")
        return ControllerSpec(kind="scripted", name=name)
    if kind == "checkpoint":
        mode = raw.get("mode", "fixed")
        if mode not in {"fixed", "live"}:
            raise ControllerResolutionError("checkpoint controller mode must be 'fixed' or 'live'")
        return ControllerSpec(
            kind="checkpoint",
            path=_path_field(raw),
            algorithm=_algorithm_field(raw),
            mode=mode,
        )
    if kind == "run":
        mode = raw.get("mode", "fixed")
        if mode not in {"fixed", "live"}:
            raise ControllerResolutionError("run controller mode must be 'fixed' or 'live'")
        return ControllerSpec(kind="run", path=_path_field(raw), mode=mode)
    raise ControllerResolutionError("controller kind must be 'scripted', 'checkpoint', or 'run'")


def _parse_string_spec(raw: str) -> Mapping[str, Any]:
    value = raw.strip()
    if value in SCRIPTED_NAMES:
        return {"kind": "scripted", "name": value}
    if value.startswith("ppo:") or value.startswith("dqn:"):
        alias, path = value.split(":", 1)
        # The old directory form was a live checkpoint source. Keep that behavior,
        # but only refresh it when the caller explicitly invokes ControllerBinding.reload().
        return {
            "kind": "checkpoint",
            "path": path,
            "algorithm": ALGORITHM_ALIASES[alias],
            "mode": "live" if Path(path).is_dir() else "fixed",
        }
    if value.startswith("run:"):
        return {"kind": "run", "path": value[4:], "mode": "fixed"}
    if value.startswith("@"):
        try:
            return read_json(Path(value[1:]))
        except (OSError, json.JSONDecodeError) as error:
            raise ControllerResolutionError(f"could not read controller spec file {value[1:]!r}") from error
    if value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ControllerResolutionError("controller JSON is invalid") from error

    path = Path(value)
    if path.is_dir() and (path / "run.json").is_file():
        return {"kind": "run", "path": value, "mode": "fixed"}
    raise ControllerResolutionError(
        "controller strings must be random, greedy, ppo:PATH, dqn:PATH, run:PATH, JSON, or @spec.json"
    )


def _path_field(raw: Mapping[str, Any]) -> Path:
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ControllerResolutionError("controller path must be a non-empty string")
    return Path(path)


def _algorithm_field(raw: Mapping[str, Any]) -> Algorithm:
    algorithm = raw.get("algorithm")
    if algorithm in ALGORITHM_ALIASES:
        algorithm = ALGORITHM_ALIASES[algorithm]
    if algorithm not in ALGORITHM_ALIASES.values():
        raise ControllerResolutionError("checkpoint controller requires algorithm 'maskable_ppo' or 'masked_dqn'")
    return algorithm


class ControllerResolver:
    """Resolve immutable controller specs against an optional runtime contract."""

    def __init__(self, runtime_contract: EnvironmentContract | None = None, *, model_loader: ModelLoader | None = None):
        self.runtime_contract = runtime_contract
        self.model_loader = model_loader or load_model

    def bind(self, raw: str | Mapping[str, Any] | ControllerSpec) -> "ControllerBinding":
        return ControllerBinding(self, normalize_controller_spec(raw))

    def resolve(self, raw: str | Mapping[str, Any] | ControllerSpec) -> ResolvedController:
        return self._resolve(normalize_controller_spec(raw))

    def _resolve(self, spec: ControllerSpec) -> ResolvedController:
        if spec.kind == "scripted":
            return ResolvedController(
                spec=spec,
                server_controller=spec.name or "random",
                model=None,
                path=None,
                algorithm=None,
                step=None,
                contract=None,
                observation_size=None,
                action_size=None,
                legacy=False,
                promotable=False,
            )
        assert spec.path is not None
        if spec.kind == "run":
            return self._resolve_run(spec)
        if spec.path.is_dir() and (spec.path / "run.json").is_file():
            return self._resolve_run(ControllerSpec(kind="run", path=spec.path, mode=spec.mode), spec.algorithm)
        if spec.path.is_dir() and spec.path.name == "checkpoints" and (spec.path.parent / "run.json").is_file():
            return self._resolve_run(
                ControllerSpec(kind="run", path=spec.path.parent, mode=spec.mode), spec.algorithm
            )
        return self._resolve_legacy_checkpoint(spec)

    def _resolve_run(self, spec: ControllerSpec, requested_algorithm: Algorithm | None = None) -> ResolvedController:
        assert spec.path is not None
        manifest_path = spec.path / "run.json"
        if not manifest_path.is_file():
            raise ControllerResolutionError(f"run controller requires manifest {manifest_path}")
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            raise ControllerResolutionError(f"could not read run manifest {manifest_path}") from error
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
            raise ControllerResolutionError("run manifest must have schema_version 1")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ControllerResolutionError("run manifest is missing config metadata")
        algorithm = _algorithm_field({"algorithm": config.get("algorithm")})
        if requested_algorithm is not None and requested_algorithm != algorithm:
            raise ControllerResolutionError("legacy algorithm does not match the run manifest algorithm")
        contract = _contract_from_manifest(manifest.get("contract"))
        checkpoint = manifest.get("latest_checkpoint")
        step = manifest.get("latest_checkpoint_step")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ControllerResolutionError("run manifest is missing latest_checkpoint metadata")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ControllerResolutionError("run manifest is missing latest_checkpoint_step metadata")
        checkpoint_path = (spec.path / checkpoint).resolve()
        run_path = spec.path.resolve()
        if run_path not in checkpoint_path.parents or checkpoint_path.suffix.lower() != ".zip":
            raise ControllerResolutionError("run manifest latest_checkpoint must name a .zip inside the run")
        if not checkpoint_path.is_file():
            raise ControllerResolutionError(f"published checkpoint does not exist: {checkpoint_path}")
        return self._load_model(spec, checkpoint_path, algorithm, step, contract, legacy=False)

    def _resolve_legacy_checkpoint(self, spec: ControllerSpec) -> ResolvedController:
        assert spec.path is not None and spec.algorithm is not None
        path = _latest_legacy_checkpoint(spec.path)
        return self._load_model(spec, path, spec.algorithm, None, None, legacy=True)

    def _load_model(
        self,
        spec: ControllerSpec,
        path: Path,
        algorithm: Algorithm,
        step: int | None,
        contract: EnvironmentContract | None,
        *,
        legacy: bool,
    ) -> ResolvedController:
        _validate_contract_compatibility(contract, self.runtime_contract)
        model = self.model_loader(path, algorithm)
        observation_size, action_size = _model_geometry(model)
        target = self.runtime_contract or contract
        if target is not None:
            if observation_size != target.observation_size:
                raise ControllerResolutionError("model observation size does not match the inference environment")
            if action_size != target.action_size:
                raise ControllerResolutionError("model action size does not match the inference environment")
        return ResolvedController(
            spec=spec,
            server_controller="external",
            model=model,
            path=path,
            algorithm=algorithm,
            step=step,
            contract=contract,
            observation_size=observation_size,
            action_size=action_size,
            legacy=legacy,
            promotable=not legacy,
        )


class ControllerBinding:
    """A controller frozen at construction unless its run spec explicitly opts into live reloads."""

    def __init__(self, resolver: ControllerResolver, spec: ControllerSpec):
        self._resolver = resolver
        self.spec = spec
        self.resolved = resolver._resolve(spec)

    def reload(self) -> bool:
        if self.spec.mode != "live":
            return False
        if self.spec.kind == "checkpoint" and (self.spec.path is None or not self.spec.path.is_dir()):
            return False
        updated = self._resolver._resolve(self.spec)
        if _resolution_key(updated) == _resolution_key(self.resolved):
            return False
        self.resolved = updated
        return True


def _resolution_key(resolved: ResolvedController) -> tuple[Any, ...]:
    return (resolved.path, resolved.algorithm, resolved.step, resolved.contract, resolved.legacy)


def _latest_legacy_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ControllerResolutionError("checkpoint controller path must name a .zip file")
        return path
    if not path.is_dir():
        raise ControllerResolutionError(f"checkpoint path does not exist: {path}")
    candidates = list(path.glob("*.zip"))
    if not candidates:
        raise ControllerResolutionError(f"no .zip checkpoints found in {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)


def _contract_from_manifest(value: Any) -> EnvironmentContract:
    if not isinstance(value, Mapping):
        raise ControllerResolutionError("run manifest is missing contract metadata")
    required = ("version", "contract_hash", "observation_size", "action_size", "board", "roster", "reward")
    missing = [field for field in required if field not in value]
    if missing:
        raise ControllerResolutionError(f"run manifest contract is missing {', '.join(missing)}")
    version = value["version"]
    if not isinstance(version, str) or version not in SUPPORTED_ENCODING_VERSIONS:
        raise ControllerResolutionError(f"unsupported model encoding version {version!r}")
    observation_size = value["observation_size"]
    action_size = value["action_size"]
    if isinstance(observation_size, bool) or not isinstance(observation_size, int) or observation_size <= 0:
        raise ControllerResolutionError("run manifest contract observation_size must be positive")
    if isinstance(action_size, bool) or not isinstance(action_size, int) or action_size <= 0:
        raise ControllerResolutionError("run manifest contract action_size must be positive")
    if not isinstance(value["contract_hash"], str) or not value["contract_hash"]:
        raise ControllerResolutionError("run manifest contract_hash must be a non-empty string")
    if not isinstance(value["board"], Mapping) or not isinstance(value["roster"], list) or not isinstance(value["reward"], Mapping):
        raise ControllerResolutionError("run manifest contract metadata has invalid semantic fields")
    return EnvironmentContract(
        version=version,
        contract_hash=value["contract_hash"],
        observation_size=observation_size,
        action_size=action_size,
        board=dict(value["board"]),
        roster=list(value["roster"]),
        reward=dict(value["reward"]),
    )


def _validate_contract_compatibility(
    model_contract: EnvironmentContract | None, runtime_contract: EnvironmentContract | None
) -> None:
    if model_contract is None or runtime_contract is None:
        return
    if runtime_contract.version not in SUPPORTED_ENCODING_VERSIONS:
        raise ControllerResolutionError(f"unsupported inference encoding version {runtime_contract.version!r}")
    if model_contract.version != runtime_contract.version:
        raise ControllerResolutionError("model encoding version does not match the inference environment")
    # Contract hashes include reward and horizon semantics. Tactical training and duel
    # inference intentionally have different hashes, so only representation geometry is shared here.
    if model_contract.observation_size != runtime_contract.observation_size:
        raise ControllerResolutionError("model contract observation size does not match the inference environment")
    if model_contract.action_size != runtime_contract.action_size:
        raise ControllerResolutionError("model contract action size does not match the inference environment")


def _model_geometry(model: Any) -> tuple[int, int]:
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    shape = getattr(observation_space, "shape", None)
    action_size = getattr(action_space, "n", None)
    if not isinstance(shape, tuple) or len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
        raise ControllerResolutionError("model does not expose a one-dimensional observation space")
    if isinstance(action_size, bool) or not isinstance(action_size, int) or action_size <= 0:
        raise ControllerResolutionError("model does not expose a discrete action space")
    return shape[0], action_size


def load_model(path: Path, algorithm: Algorithm) -> Any:
    """Load the declared SB3 model type. Algorithm selection never comes from a filename."""
    if algorithm == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, device="cpu")
    if algorithm == "masked_dqn":
        from stable_baselines3 import DQN

        return DQN.load(path, device="cpu")
    raise AssertionError(f"unreachable algorithm {algorithm!r}")


def predict(model: Any, algorithm: Algorithm, observation: np.ndarray, mask: np.ndarray) -> int:
    """Choose a legal deterministic action for either supported model family."""
    if algorithm == "maskable_ppo":
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        return int(action)
    import torch

    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation[None]).float().to(model.device)
        values = model.q_net(obs_tensor).cpu().numpy()[0]
    values[~mask] = -np.inf
    return int(np.argmax(values))


def validate_inference_input(resolved: ResolvedController, observation: np.ndarray, mask: np.ndarray) -> None:
    """Reject malformed bridge payloads before passing them into an SB3 policy."""
    if resolved.observation_size is not None and observation.shape != (resolved.observation_size,):
        raise ControllerResolutionError("inference observation size does not match the resolved model")
    if resolved.action_size is not None and mask.shape != (resolved.action_size,):
        raise ControllerResolutionError("inference action mask size does not match the resolved model")
