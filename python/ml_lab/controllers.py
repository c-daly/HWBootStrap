"""Normalize and resolve scripted and trained HexWars controllers.

Run manifests are authoritative for published model checkpoints. Standalone
checkpoints are rejected because they do not carry authoritative contract metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .contracts import EnvironmentContract
from .io import read_json
from .tactical_v3_schema import TacticalV3SemanticIdentity, parse_spaces


Algorithm = Literal["maskable_ppo", "masked_dqn", "structured_imitation"]
ControllerContract = EnvironmentContract | TacticalV3SemanticIdentity
InferenceMode = Literal["deterministic", "stochastic"]
SCRIPTED_NAMES = frozenset({"bounded-search", "greedy", "random"})
ALGORITHM_ALIASES: dict[str, Algorithm] = {"ppo": "maskable_ppo", "dqn": "masked_dqn"}
SUPPORTED_ENCODING_VERSIONS = frozenset({"tactical-v1", "tactical-v2", "adaptive-v1"})


class ControllerResolutionError(ValueError):
    """Raised when a controller cannot safely be used for inference."""


@dataclass(frozen=True)
class ControllerSpec:
    kind: Literal["scripted", "checkpoint", "run", "snapshot"]
    name: str | None = None
    path: Path | None = None
    source_run: Path | None = None
    algorithm: Algorithm | None = None
    step: int | None = None
    mode: Literal["fixed", "live"] = "fixed"
    inference_mode: InferenceMode = "deterministic"


@dataclass(frozen=True)
class ResolvedController:
    spec: ControllerSpec
    server_controller: str
    model: Any | None
    path: Path | None
    algorithm: Algorithm | None
    step: int | None
    contract: ControllerContract | None
    observation_size: int | None
    action_size: int | None
    legacy: bool
    promotable: bool

    def metadata(self) -> dict[str, Any]:
        """JSON-safe information for policy-server status replies."""
        return {
            "kind": self.spec.kind,
            "inference_mode": self.spec.inference_mode,
            "path": str(self.path) if self.path is not None else None,
            "algorithm": self.algorithm,
            "step": self.step,
            "contract_hash": self.contract.contract_hash if self.contract is not None else None,
            "contract_version": (
                self.contract.version if isinstance(self.contract, EnvironmentContract)
                else self.contract.contract_version if self.contract is not None else None
            ),
            "environment": (
                self.contract.environment if isinstance(self.contract, EnvironmentContract)
                else "tactical-v3" if self.contract is not None else None
            ),
            "encoding_hash": self.contract.encoding_hash if self.contract is not None else None,
            "capacity_hash": (
                self.contract.capacity_hash if isinstance(self.contract, TacticalV3SemanticIdentity)
                else None
            ),
            "contract": self.contract.to_dict() if isinstance(self.contract, EnvironmentContract) else None,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "legacy": self.legacy,
            "promotable": self.promotable,
        }


ModelLoader = Callable[[Path, Algorithm], Any]


def normalize_controller_spec(raw: str | Mapping[str, Any] | ControllerSpec) -> ControllerSpec:
    """Parse a controller spec; legacy path forms survive only for clear rejection."""
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
            raise ControllerResolutionError("unsupported scripted controller name")
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
            inference_mode=_inference_mode_field(raw),
        )
    if kind == "run":
        mode = raw.get("mode", "fixed")
        if mode not in {"fixed", "live"}:
            raise ControllerResolutionError("run controller mode must be 'fixed' or 'live'")
        return ControllerSpec(
            kind="run",
            path=_path_field(raw),
            mode=mode,
            inference_mode=_inference_mode_field(raw),
        )
    if kind == "snapshot":
        step = raw.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ControllerResolutionError("snapshot controller step must be a non-negative integer")
        source_run = raw.get("source_run")
        if not isinstance(source_run, str) or not source_run.strip():
            raise ControllerResolutionError("snapshot controller source_run must be a non-empty string")
        algorithm = raw.get("algorithm")
        if algorithm not in ALGORITHM_ALIASES.values():
            raise ControllerResolutionError(
                "snapshot controller algorithm must be 'maskable_ppo' or 'masked_dqn'"
            )
        return ControllerSpec(
            kind="snapshot",
            path=_path_field(raw),
            source_run=Path(source_run),
            algorithm=algorithm,
            step=step,
            inference_mode=_inference_mode_field(raw),
        )
    raise ControllerResolutionError(
        "controller kind must be 'scripted', 'checkpoint', 'run', or 'snapshot'"
    )


def _parse_string_spec(raw: str) -> Mapping[str, Any]:
    value = raw.strip()
    if value in SCRIPTED_NAMES:
        return {"kind": "scripted", "name": value}
    if value.startswith("ppo:") or value.startswith("dqn:"):
        alias, path = value.split(":", 1)
        # Preserve legacy syntax at the parsing boundary so resolution can explain
        # that contract metadata is required instead of reporting an unknown format.
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
        "controller string must name a supported scripted controller, run:PATH, JSON, or @spec.json"
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


def _inference_mode_field(raw: Mapping[str, Any]) -> InferenceMode:
    inference_mode = raw.get("inference_mode", "deterministic")
    if inference_mode not in {"deterministic", "stochastic"}:
        raise ControllerResolutionError(
            "controller inference mode must be 'deterministic' or 'stochastic'"
        )
    return inference_mode


class ControllerResolver:
    """Resolve immutable controller specs against an optional runtime contract."""

    def __init__(
        self,
        runtime_contract: EnvironmentContract | None = None,
        *,
        model_loader: ModelLoader | None = None,
        expected_structured_hashes: tuple[str, str] | None = None,
    ):
        self.runtime_contract = runtime_contract
        self.model_loader = model_loader or load_model
        self.expected_structured_hashes = expected_structured_hashes

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
        if spec.kind == "snapshot":
            return self._resolve_snapshot(spec)
        if spec.kind == "run":
            return self._resolve_run(spec)
        if spec.path.is_dir() and (spec.path / "run.json").is_file():
            return self._resolve_run(
                ControllerSpec(
                    kind="run",
                    path=spec.path,
                    mode=spec.mode,
                    inference_mode=spec.inference_mode,
                ),
                spec.algorithm,
            )
        if spec.path.is_dir() and spec.path.name == "checkpoints" and (spec.path.parent / "run.json").is_file():
            return self._resolve_run(
                ControllerSpec(
                    kind="run",
                    path=spec.path.parent,
                    mode=spec.mode,
                    inference_mode=spec.inference_mode,
                ),
                spec.algorithm,
            )
        return self._resolve_legacy_checkpoint(spec)

    def _resolve_snapshot(self, spec: ControllerSpec) -> ResolvedController:
        assert spec.path is not None
        assert spec.source_run is not None
        assert spec.algorithm is not None
        assert spec.step is not None
        source_run = spec.source_run.resolve()
        manifest_path = source_run / "run.json"
        if not manifest_path.is_file():
            raise ControllerResolutionError(
                f"snapshot controller requires source manifest {manifest_path}"
            )
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            raise ControllerResolutionError(
                f"could not read snapshot source manifest {manifest_path}"
            ) from error
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
            raise ControllerResolutionError("snapshot source manifest must have schema_version 1")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ControllerResolutionError("snapshot source manifest is missing config metadata")
        recorded_algorithm = _algorithm_field({"algorithm": config.get("algorithm")})
        if recorded_algorithm != spec.algorithm:
            raise ControllerResolutionError(
                "snapshot algorithm does not match the source run manifest"
            )
        checkpoint_path = spec.path.resolve()
        checkpoints_dir = (source_run / "checkpoints").resolve()
        if (
            checkpoint_path.parent != checkpoints_dir
            or checkpoint_path.suffix.lower() != ".zip"
        ):
            raise ControllerResolutionError(
                "snapshot path must name a checkpoint inside the source run"
            )
        if checkpoint_path.name != f"step_{spec.step:09d}.zip":
            raise ControllerResolutionError(
                "snapshot step does not match the recorded checkpoint path"
            )
        if not checkpoint_path.is_file():
            raise ControllerResolutionError(
                f"snapshot checkpoint does not exist: {checkpoint_path}"
            )
        contract = _contract_from_manifest(manifest.get("contract"))
        return self._load_model(
            spec,
            checkpoint_path,
            spec.algorithm,
            spec.step,
            contract,
            legacy=False,
        )

    def _resolve_run(self, spec: ControllerSpec, requested_algorithm: Algorithm | None = None) -> ResolvedController:
        assert spec.path is not None
        manifest_path = spec.path / "run.json"
        if not manifest_path.is_file():
            raise ControllerResolutionError(f"run controller requires manifest {manifest_path}")
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            raise ControllerResolutionError(f"could not read run manifest {manifest_path}") from error
        if not isinstance(manifest, Mapping):
            raise ControllerResolutionError("run manifest must be a JSON object")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ControllerResolutionError("run manifest is missing config metadata")
        declared_algorithm = config.get("algorithm")
        if declared_algorithm == "structured_imitation":
            if manifest.get("schema_version") != 2:
                raise ControllerResolutionError(
                    "structured run manifest must have schema_version 2"
                )
            if requested_algorithm is not None:
                raise ControllerResolutionError("legacy algorithm does not match the run manifest algorithm")
            return self._resolve_structured_run(spec, manifest)
        if manifest.get("schema_version") != 1:
            raise ControllerResolutionError("run manifest must have schema_version 1")
        algorithm = _algorithm_field({"algorithm": declared_algorithm})
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

    def _resolve_structured_run(
        self, spec: ControllerSpec, manifest: Mapping[str, Any],
    ) -> ResolvedController:
        assert spec.path is not None
        try:
            if manifest.get("policy_identity") != "policy-identity.json":
                raise ValueError(
                    "structured run must declare policy-identity.json"
                )
            policy_identity = read_json(spec.path / "policy-identity.json")
            identity = parse_spaces(policy_identity)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ControllerResolutionError(
                "structured run requires a valid policy identity"
            ) from error
        from .tactical_v3_controller import load_structured_controller

        try:
            expected_encoding_hash, expected_capacity_hash = (
                self.expected_structured_hashes
                if self.expected_structured_hashes is not None
                else (identity.encoding_hash, identity.capacity_hash)
            )
            structured = load_structured_controller(
                spec.path, expected_encoding_hash, expected_capacity_hash
            )
        except (OSError, TypeError, ValueError) as error:
            raise ControllerResolutionError(str(error)) from error
        step = manifest.get("latest_checkpoint_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ControllerResolutionError("run manifest is missing latest_checkpoint_step metadata")
        return ResolvedController(
            spec=spec,
            server_controller="external",
            model=structured,
            path=structured.checkpoint_path,
            algorithm="structured_imitation",
            step=step,
            contract=structured.identity,
            observation_size=None,
            action_size=None,
            legacy=False,
            promotable=False,
        )

    def _resolve_legacy_checkpoint(self, spec: ControllerSpec) -> ResolvedController:
        raise ControllerResolutionError(
            "standalone checkpoints lack contract metadata; use a metadata-backed run:PATH"
        )

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
        if (
            contract is None
            and self.runtime_contract is not None
            and self.runtime_contract.version == "adaptive-v1"
        ):
            raise ControllerResolutionError(
                "adaptive inference requires checkpoint contract metadata"
            )
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

    def reload(self, validator: Callable[[ResolvedController], None] | None = None) -> bool:
        if self.spec.mode != "live":
            return False
        if self.spec.kind == "checkpoint" and (self.spec.path is None or not self.spec.path.is_dir()):
            return False
        updated = self._resolver._resolve(self.spec)
        if validator is not None:
            validator(updated)
        if _resolution_key(updated) == _resolution_key(self.resolved):
            return False
        self.resolved = updated
        return True


def _resolution_key(resolved: ResolvedController) -> tuple[Any, ...]:
    return (resolved.path, resolved.algorithm, resolved.step, resolved.contract, resolved.legacy)


def _resolved_source_run(resolved: ResolvedController) -> Path:
    if resolved.path is None:
        raise ControllerResolutionError("resolved model controller is missing a checkpoint path")
    for candidate in resolved.path.parents:
        if (candidate / "run.json").is_file():
            return candidate.resolve()
    raise ControllerResolutionError(
        "resolved model checkpoint does not belong to a metadata-backed run"
    )


def _snapshot_controller(
    raw: str | Mapping[str, Any] | ControllerSpec,
) -> Mapping[str, Any]:
    binding = ControllerResolver().bind(raw)
    spec = binding.spec
    resolved = binding.resolved
    if spec.kind == "scripted":
        return {"kind": "scripted", "name": spec.name}
    source_run = _resolved_source_run(resolved)
    if spec.mode == "live":
        return {
            "kind": "run",
            "path": str(source_run),
            "mode": "live",
            "inference_mode": spec.inference_mode,
        }
    if (
        resolved.path is None
        or resolved.algorithm is None
        or resolved.step is None
        or resolved.contract is None
    ):
        raise ControllerResolutionError(
            "fixed opponent snapshot requires checkpoint and contract metadata"
        )
    return {
        "kind": "snapshot",
        "path": str(resolved.path.resolve()),
        "source_run": str(source_run),
        "algorithm": resolved.algorithm,
        "step": resolved.step,
        "inference_mode": spec.inference_mode,
    }


def snapshot_opponents(opponent: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze fixed opponents at exact checkpoints while preserving live bindings."""
    if opponent.get("kind") == "pool":
        controllers = opponent.get("controllers")
        if not isinstance(controllers, list) or not controllers:
            raise ControllerResolutionError(
                "opponent pool requires at least one controller"
            )
        return {
            "kind": "pool",
            "controllers": [_snapshot_controller(entry) for entry in controllers],
        }
    return _snapshot_controller(opponent)


def _latest_legacy_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ControllerResolutionError("checkpoint controller path must name a .zip file")
        return path
    if not path.is_dir():
        raise ControllerResolutionError(f"checkpoint path does not exist: {path}")
    candidates = list(path.glob("*.zip"))
    if not candidates:
        # Pre-manifest training runs commonly kept checkpoints under this nested
        # directory. The explicit legacy algorithm still governs loading.
        candidates = list((path / "checkpoints").glob("*.zip"))
    if not candidates:
        raise ControllerResolutionError(f"no .zip checkpoints found in {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)


def _contract_from_manifest(value: Any) -> EnvironmentContract:
    if not isinstance(value, Mapping):
        raise ControllerResolutionError("run manifest is missing contract metadata")
    required = ("environment", "version", "contract_hash", "encoding_hash", "observation_size", "action_size", "board", "roster", "reward")
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
    if value["environment"] != version:
        raise ControllerResolutionError("run manifest contract environment does not match version")
    if not isinstance(value["encoding_hash"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["encoding_hash"]
    ):
        raise ControllerResolutionError("run manifest encoding_hash must be a lowercase SHA-256 hex digest")
    if (
        not isinstance(value["board"], Mapping)
        or not isinstance(value["roster"], list)
        or not isinstance(value["reward"], Mapping)
        or ("semantics" in value and not isinstance(value["semantics"], Mapping))
    ):
        raise ControllerResolutionError("run manifest contract metadata has invalid semantic fields")
    return EnvironmentContract(
        version=version,
        contract_hash=value["contract_hash"],
        encoding_hash=value["encoding_hash"],
        observation_size=observation_size,
        action_size=action_size,
        board=dict(value["board"]),
        roster=list(value["roster"]),
        reward=dict(value["reward"]),
        semantics=dict(value.get("semantics", {})),
    )


def _validate_contract_compatibility(
    model_contract: EnvironmentContract | None, runtime_contract: EnvironmentContract | None
) -> None:
    if model_contract is None or runtime_contract is None:
        return
    if runtime_contract.version not in SUPPORTED_ENCODING_VERSIONS:
        raise ControllerResolutionError(f"unsupported inference encoding version {runtime_contract.version!r}")
    if model_contract.environment != runtime_contract.environment:
        raise ControllerResolutionError("model environment does not match the inference environment")
    if model_contract.version != runtime_contract.version:
        raise ControllerResolutionError("model encoding version does not match the inference environment")
    if model_contract.encoding_hash != runtime_contract.encoding_hash:
        raise ControllerResolutionError("model encoding hash does not match the inference environment")
    if model_contract.observation_size != runtime_contract.observation_size:
        raise ControllerResolutionError("model contract observation size does not match the inference environment")
    if model_contract.action_size != runtime_contract.action_size:
        raise ControllerResolutionError("model contract action size does not match the inference environment")

def _model_geometry(model: Any) -> tuple[int, int]:
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    shape = getattr(observation_space, "shape", None)
    action_size = getattr(action_space, "n", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], Integral)
        or shape[0] <= 0
    ):
        raise ControllerResolutionError("model does not expose a one-dimensional observation space")
    if isinstance(action_size, bool) or not isinstance(action_size, Integral) or action_size <= 0:
        raise ControllerResolutionError("model does not expose a discrete action space")
    return int(shape[0]), int(action_size)


def load_model(path: Path, algorithm: Algorithm) -> Any:
    """Load the declared SB3 model type. Algorithm selection never comes from a filename."""
    if algorithm == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, device="cpu")
    if algorithm == "masked_dqn":
        from stable_baselines3 import DQN

        return DQN.load(path, device="cpu")
    raise AssertionError(f"unreachable algorithm {algorithm!r}")


def predict(
    model: Any,
    algorithm: Algorithm,
    observation: np.ndarray,
    mask: np.ndarray,
    *,
    deterministic: bool = True,
) -> int:
    """Choose a legal action, optionally sampling a MaskablePPO policy."""
    if algorithm == "maskable_ppo":
        action, _ = model.predict(
            observation,
            action_masks=mask,
            deterministic=deterministic,
        )
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
