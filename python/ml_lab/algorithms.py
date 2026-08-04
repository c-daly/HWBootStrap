"""Central SB3 algorithm adapters for HexWars training and checkpoint validation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import read_json


ACTOR_MODULES = {
    "features_extractor": lambda policy: policy.features_extractor,
    "policy_net": lambda policy: policy.mlp_extractor.policy_net,
    "action_net": lambda policy: policy.action_net,
}

ACTOR_SOURCE_KINDS = frozenset({"snapshot", "dagger_actor"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ActorTransferSource:
    """Immutable physical identity for one controller-resolved actor source."""

    source_kind: str
    controller: Mapping[str, Any]
    checkpoint_sha256: str
    published_actor_sha256: str | None = None

    def __post_init__(self) -> None:
        from .controllers import normalize_controller_spec

        if self.source_kind not in ACTOR_SOURCE_KINDS:
            raise ValueError("actor transfer source kind is invalid")
        if (
            not isinstance(self.controller, Mapping)
            or set(self.controller) != {
                "kind", "path", "source_run", "algorithm", "step",
                "inference_mode",
            }
        ):
            raise ValueError("actor transfer controller fields are invalid")
        spec = normalize_controller_spec(self.controller)
        if (
            spec.kind != "snapshot"
            or spec.algorithm != "maskable_ppo"
            or spec.inference_mode != "deterministic"
            or spec.path is None
            or spec.source_run is None
            or spec.step is None
        ):
            raise ValueError(
                "actor transfer source must be a deterministic MaskablePPO snapshot"
            )
        if (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_sha256
            )
        ):
            raise ValueError(
                "actor transfer checkpoint SHA-256 must be lowercase hexadecimal"
            )
        if self.source_kind == "dagger_actor":
            if (
                not isinstance(self.published_actor_sha256, str)
                or len(self.published_actor_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.published_actor_sha256
                )
            ):
                raise ValueError(
                    "published DAgger actor SHA-256 must be lowercase hexadecimal"
                )
        elif self.published_actor_sha256 is not None:
            raise ValueError(
                "snapshot actor source must not claim a published actor SHA-256"
            )
        object.__setattr__(
            self, "controller", MappingProxyType(dict(self.controller)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": 1,
            "source_kind": self.source_kind,
            "controller": dict(self.controller),
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        if self.published_actor_sha256 is not None:
            result["published_actor_sha256"] = self.published_actor_sha256
        return result


@dataclass(frozen=True)
class _AuthenticatedActorTransfer:
    """A loaded actor bound to one immutable, authenticated checkpoint snapshot."""

    source: ActorTransferSource
    model: Any
    checkpoint_bytes: bytes
    source_run: str
    source_checkpoint: str
    checkpoint_sha256: str
    run_manifest_sha256: str
    source_bc_sha256: str | None
    contract: EnvironmentContract
    exact_contract: bool
    observation_size: int
    action_size: int
    algorithm: str
    policy_class: str
    step: int
    inference_mode: str
    source_actor_sha256: str


def _actor_modules(model: Any) -> dict[str, Any]:
    return {
        name: accessor(model.policy) for name, accessor in ACTOR_MODULES.items()
    }


def _state_hash(states: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for module_name in ACTOR_MODULES:
        state = states[module_name]
        for tensor_name, tensor in state.items():
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(module_name.encode("utf-8"))
            digest.update(tensor_name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _actor_states(model: Any, *, copy_states: bool = False) -> dict[str, Mapping[str, Any]]:
    states: dict[str, Mapping[str, Any]] = {}
    for name, module in _actor_modules(model).items():
        state = module.state_dict()
        states[name] = copy.deepcopy(state) if copy_states else state
    return states


def actor_state_sha256(model: Any) -> str:
    """Hash actor/shared-feature tensors by stable module, name, shape, and bytes."""

    return _state_hash(_actor_states(model))


def _restore_actor_states(model: Any, states: Mapping[str, Mapping[str, Any]]) -> None:
    for name, module in _actor_modules(model).items():
        module.load_state_dict(states[name], strict=True)


def _read_file_snapshot(path: Path) -> bytes:
    with Path(path).open("rb") as stream:
        return stream.read()


def _read_json_snapshot(path: Path) -> tuple[Mapping[str, Any], str]:
    payload = _read_file_snapshot(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"actor transfer metadata is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"actor transfer metadata must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _contained_source_file(source_run: Path, relative: str) -> Path:
    candidate = (source_run / relative).resolve(strict=True)
    if not candidate.is_relative_to(source_run) or not candidate.is_file():
        raise ValueError("actor transfer metadata must be contained by the source run")
    return candidate


def _contract_from_metadata(value: object) -> EnvironmentContract:
    required = {
        "environment", "version", "contract_hash", "encoding_hash",
        "observation_size", "action_size", "board", "roster", "reward",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ContractMismatch("actor transfer source contract metadata is invalid")
    if (
        not isinstance(value.get("version"), str)
        or value.get("environment") != value.get("version")
        or not isinstance(value.get("contract_hash"), str)
        or not value.get("contract_hash")
        or type(value.get("observation_size")) is not int
        or value["observation_size"] <= 0
        or type(value.get("action_size")) is not int
        or value["action_size"] <= 0
        or not isinstance(value.get("board"), Mapping)
        or not isinstance(value.get("roster"), list)
        or not isinstance(value.get("reward"), Mapping)
        or not isinstance(value.get("semantics", {}), Mapping)
    ):
        raise ContractMismatch("actor transfer source contract metadata is invalid")
    try:
        return EnvironmentContract(
            version=value["version"],
            contract_hash=value["contract_hash"],
            encoding_hash=value["encoding_hash"],
            observation_size=value["observation_size"],
            action_size=value["action_size"],
            board=dict(value["board"]),
            roster=list(value["roster"]),
            reward=dict(value["reward"]),
            semantics=dict(value.get("semantics", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractMismatch(
            "actor transfer source contract metadata is invalid"
        ) from exc


def _validate_compatible_contract(
    source: EnvironmentContract,
    expected: EnvironmentContract,
) -> None:
    if (
        source.environment != expected.environment
        or source.version != expected.version
        or source.encoding_hash != expected.encoding_hash
        or source.observation_size != expected.observation_size
        or source.action_size != expected.action_size
    ):
        raise ContractMismatch(
            "actor transfer source contract is not compatible"
        )


def preflight_actor_transfer_source(
    source: ActorTransferSource,
) -> tuple[Path, Path, str]:
    """Resolve the canonical contained path without opening checkpoint bytes."""

    if not isinstance(source, ActorTransferSource):
        raise TypeError("source must be an ActorTransferSource")
    source_run = Path(source.controller["source_run"]).resolve(strict=True)
    checkpoint = Path(source.controller["path"]).resolve(strict=True)
    step = source.controller["step"]
    if (
        not checkpoint.is_relative_to(source_run)
        or checkpoint.parent != (source_run / "checkpoints").resolve(strict=True)
        or checkpoint.name != f"step_{step:09d}.zip"
    ):
        raise ValueError(
            "actor transfer checkpoint must be contained by the resolved source run"
        )
    if source.source_kind == "dagger_actor" and (
        step != 0 or checkpoint.name != "step_000000000.zip"
    ):
        raise ValueError(
            "published DAgger actor must use the canonical step-zero checkpoint"
        )
    if not source_run.is_dir() or not checkpoint.is_file():
        raise ValueError("actor transfer source must name regular run and checkpoint files")
    return source_run, checkpoint, source.checkpoint_sha256


class AlgorithmAdapter(Protocol):
    name: str
    policy_name: str
    experimental: bool

    def create(self, env: Any, **kwargs: Any) -> Any: ...

    def load(self, path: Any, *, env: Any, device: str) -> Any: ...

    def initialize_actor(
        self,
        model: Any,
        source_run: Path,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]: ...

    def initialize_actor_from_source(
        self,
        model: Any,
        source: ActorTransferSource,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]: ...

    def validate_model(self, model: Any, expected_contract: EnvironmentContract) -> None: ...

    def predict(self, model: Any, observation: np.ndarray, mask: np.ndarray) -> int: ...

    def save(self, model: Any, path: Path) -> Path: ...

    def inspect(
        self, path: Path, expected_contract: EnvironmentContract
    ) -> dict[str, Any]: ...


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
    ):
        raise ContractMismatch("model does not expose a one-dimensional observation space")
    if isinstance(action_size, bool) or not isinstance(action_size, Integral):
        raise ContractMismatch("model does not expose a discrete action space")
    return int(shape[0]), int(action_size)


def _validate_geometry(model: Any, expected_contract: EnvironmentContract) -> None:
    observation_size, action_size = _model_geometry(model)
    if observation_size != expected_contract.observation_size:
        raise ContractMismatch("model observation size does not match the training contract")
    if action_size != expected_contract.action_size:
        raise ContractMismatch("model action size does not match the training contract")


def _save_sb3_model(model: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    written = path if path.is_file() else Path(f"{path}.zip")
    if not written.is_file():
        raise FileNotFoundError(f"model save did not create {path}")
    return written


@dataclass(frozen=True)
class MaskablePPOAdapter:
    name: str = "maskable_ppo"
    policy_name: str = "HexCNN"
    experimental: bool = False

    def create(
        self,
        env: Any,
        *,
        spaces_info: dict[str, Any],
        seed: int,
        device: str,
        checkpoint_interval: int,
        algorithm_options: Mapping[str, Any] | None = None,
    ) -> Any:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

        from hex_cnn import cnn_policy_kwargs

        options = dict(algorithm_options or {})
        allowed = {"learning_rate", "n_epochs", "target_kl"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"unsupported MaskablePPO option {sorted(unknown)[0]!r}")
        for key in ("learning_rate", "target_kl"):
            if key in options:
                value = options[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{key} must be finite and positive")
        if "n_epochs" in options and (
            type(options["n_epochs"]) is not int or options["n_epochs"] <= 0
        ):
            raise ValueError("n_epochs must be a positive integer")
        workers = max(1, int(getattr(env, "num_envs", 1)))
        n_steps = max(2, min(512, max(2, checkpoint_interval // workers)))
        rollout_size = n_steps * workers
        batch_size = max(2, min(64, rollout_size))
        return MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            verbose=1,
            policy_kwargs=cnn_policy_kwargs(spaces_info),
            **options,
        )

    def initialize_actor_from_source(
        self,
        model: Any,
        source: ActorTransferSource,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]:
        """Authenticate one checkpoint snapshot and transactionally copy its actor."""

        return self._initialize_actor_from_source(
            model,
            source,
            expected_contract,
            device,
            checkpoint_bytes=None,
            run_snapshot=None,
            require_exact_contract=True,
            authenticated_source_callback=None,
        )

    def _authenticate_actor_transfer(
        self,
        source: ActorTransferSource,
        expected_contract: EnvironmentContract,
        *,
        checkpoint_bytes: bytes | None,
        run_snapshot: tuple[Mapping[str, Any], str] | None,
        require_exact_contract: bool,
    ) -> _AuthenticatedActorTransfer:
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import (
            MaskableActorCriticPolicy,
        )

        source_run, checkpoint, _expected_sha256 = (
            preflight_actor_transfer_source(source)
        )
        controller = source.controller
        step = controller["step"]
        manifest_path = _contained_source_file(source_run, "run.json")
        if run_snapshot is None:
            manifest, run_manifest_sha256 = _read_json_snapshot(manifest_path)
        else:
            manifest, run_manifest_sha256 = run_snapshot
            if not isinstance(manifest, Mapping) or not _is_sha256(
                run_manifest_sha256
            ):
                raise TypeError("authenticated run metadata snapshot is invalid")
        manifest_config = manifest.get("config")
        relative_checkpoint = checkpoint.relative_to(source_run).as_posix()
        if (
            manifest.get("schema_version") != 1
            or manifest.get("latest_checkpoint") != relative_checkpoint
            or manifest.get("latest_checkpoint_step") != step
            or not isinstance(manifest_config, Mapping)
            or manifest_config.get("algorithm") != self.name
            or manifest_config.get("policy") != self.policy_name
        ):
            raise ValueError("actor transfer source run metadata does not match")

        source_contract = _contract_from_metadata(manifest.get("contract"))
        if require_exact_contract:
            if manifest.get("contract") != expected_contract.to_dict():
                raise ContractMismatch(
                    "actor transfer source contract does not exactly match"
                )
        else:
            _validate_compatible_contract(source_contract, expected_contract)

        source_bc_sha256 = None
        if source.source_kind == "dagger_actor":
            bc_path = _contained_source_file(source_run, "bc.json")
            bc, source_bc_sha256 = _read_json_snapshot(bc_path)
            published_actor_sha256 = source.published_actor_sha256
            distillation_iteration = manifest.get("distillation_iteration")
            actor_initialization = manifest.get("actor_initialization")
            publication_verification = manifest.get(
                "publication_verification"
            )
            if (
                manifest.get("state") != "completed"
                or manifest.get("production") is not True
                or manifest.get("training_kind")
                != "selective-dagger-distillation-v1"
                or type(distillation_iteration) is not int
                or distillation_iteration not in {1, 2}
                or manifest.get("checkpoint_sha256")
                != source.checkpoint_sha256
                or manifest.get("target_actor_sha256_final")
                != published_actor_sha256
                or not isinstance(actor_initialization, Mapping)
                or not isinstance(publication_verification, Mapping)
                or publication_verification.get("checkpoint_sha256")
                != source.checkpoint_sha256
                or publication_verification.get("actor_sha256")
                != published_actor_sha256
                or bc.get("schema_version") != 1
                or bc.get("training_kind")
                != "selective-dagger-distillation-v1"
                or bc.get("algorithm") != self.name
                or bc.get("policy") != self.policy_name
                or bc.get("production") is not True
                or bc.get("checkpoint_sha256")
                != source.checkpoint_sha256
                or bc.get("distillation_iteration") != distillation_iteration
                or bc.get("target_actor_sha256_final")
                != published_actor_sha256
                or bc.get("actor_initialization") != actor_initialization
                or bc.get("publication_verification")
                != publication_verification
            ):
                raise ValueError(
                    "published DAgger actor provenance does not agree across manifests"
                )

        if checkpoint_bytes is None:
            checkpoint_bytes = _read_file_snapshot(checkpoint)
        elif not isinstance(checkpoint_bytes, bytes):
            raise TypeError("checkpoint snapshot must be immutable bytes")
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if checkpoint_sha256 != source.checkpoint_sha256:
            raise ValueError("actor transfer checkpoint SHA-256 does not match")

        authenticated_checkpoint = io.BytesIO(checkpoint_bytes)
        try:
            if authenticated_checkpoint.tell() != 0:
                raise AssertionError("authenticated checkpoint buffer is not rewound")
            source_model = self.load(
                authenticated_checkpoint, env=None, device="cpu",
            )
        finally:
            authenticated_checkpoint.close()

        self.validate_model(source_model, expected_contract)
        if type(source_model) is not MaskablePPO:
            raise ValueError("actor transfer source model class is not MaskablePPO")
        if type(source_model.policy) is not MaskableActorCriticPolicy:
            raise ValueError("actor transfer source policy class does not match")
        if {parameter.device.type for parameter in source_model.policy.parameters()} != {
            "cpu"
        }:
            raise RuntimeError("authenticated actor transfer source is not on CPU")
        source_actor_sha256 = actor_state_sha256(source_model)
        if (
            source.source_kind == "dagger_actor"
            and source_actor_sha256 != source.published_actor_sha256
        ):
            raise ValueError(
                "published DAgger actor hash does not match the loaded physical actor"
            )
        policy_class = (
            f"{type(source_model.policy).__module__}."
            f"{type(source_model.policy).__qualname__}"
        )
        return _AuthenticatedActorTransfer(
            source=source,
            model=source_model,
            checkpoint_bytes=checkpoint_bytes,
            source_run=str(source_run),
            source_checkpoint=relative_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            run_manifest_sha256=run_manifest_sha256,
            source_bc_sha256=source_bc_sha256,
            contract=source_contract,
            exact_contract=require_exact_contract,
            observation_size=source_contract.observation_size,
            action_size=source_contract.action_size,
            algorithm=self.name,
            policy_class=policy_class,
            step=step,
            inference_mode=controller["inference_mode"],
            source_actor_sha256=source_actor_sha256,
        )

    def _initialize_actor_from_source(
        self,
        model: Any,
        source: ActorTransferSource,
        expected_contract: EnvironmentContract,
        device: str,
        *,
        checkpoint_bytes: bytes | None,
        run_snapshot: tuple[Mapping[str, Any], str] | None,
        require_exact_contract: bool,
        authenticated_source_callback: Callable[[Any], None] | None,
    ) -> Mapping[str, Any]:
        """Authenticate and copy without exposing the intermediate trusted record."""

        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import (
            MaskableActorCriticPolicy,
        )
        from stable_baselines3.common.utils import get_device

        authenticated = self._authenticate_actor_transfer(
            source,
            expected_contract,
            checkpoint_bytes=checkpoint_bytes,
            run_snapshot=run_snapshot,
            require_exact_contract=require_exact_contract,
        )
        if authenticated.exact_contract:
            if authenticated.contract != expected_contract:
                raise ContractMismatch(
                    "actor transfer source contract does not exactly match"
                )
        else:
            _validate_compatible_contract(
                authenticated.contract, expected_contract,
            )
        if authenticated.algorithm != self.name:
            raise ValueError("actor transfer source algorithm is not MaskablePPO")
        if authenticated.inference_mode != "deterministic":
            raise ValueError("actor transfer source must use deterministic inference")
        if authenticated.step != source.controller["step"]:
            raise ValueError("actor transfer source checkpoint step differs")
        if (
            authenticated.observation_size != expected_contract.observation_size
            or authenticated.action_size != expected_contract.action_size
        ):
            raise ContractMismatch("actor transfer source geometry does not match")
        if authenticated.checkpoint_sha256 != source.checkpoint_sha256:
            raise ValueError(
                "authenticated actor checkpoint SHA-256 differs from source pin"
            )
        if hashlib.sha256(authenticated.checkpoint_bytes).hexdigest() != (
            authenticated.checkpoint_sha256
        ):
            raise ValueError("authenticated actor checkpoint SHA-256 changed")

        source_model = authenticated.model
        self.validate_model(source_model, expected_contract)
        self.validate_model(model, expected_contract)
        if type(source_model) is not MaskablePPO or type(model) is not MaskablePPO:
            raise ValueError("actor transfer source model class is not MaskablePPO")
        if (
            type(source_model.policy) is not MaskableActorCriticPolicy
            or type(model.policy) is not MaskableActorCriticPolicy
            or type(source_model.policy) is not type(model.policy)
        ):
            raise ValueError("actor transfer source policy class does not match")

        requested_device = get_device(device)
        model_device = torch.device(model.device)
        if model_device != requested_device:
            raise ValueError(
                f"actor transfer device {requested_device} does not match "
                f"model device {model_device}"
            )
        if authenticated_source_callback is not None:
            authenticated_source_callback(source_model)
        source_modules = _actor_modules(source_model)
        target_modules = _actor_modules(model)
        source_states: dict[str, Mapping[str, Any]] = {}
        target_states: dict[str, Mapping[str, Any]] = {}
        for name in ACTOR_MODULES:
            source_state = source_modules[name].state_dict()
            target_state = target_modules[name].state_dict()
            if tuple(source_state) != tuple(target_state):
                raise ContractMismatch(
                    f"actor module {name!r} state keys do not match"
                )
            for key in source_state:
                source_tensor = source_state[key]
                target_tensor = target_state[key]
                if source_tensor.shape != target_tensor.shape:
                    raise ContractMismatch(
                        f"actor module {name!r} tensor {key!r} shape does not match"
                    )
                if source_tensor.dtype != target_tensor.dtype:
                    raise ContractMismatch(
                        f"actor module {name!r} tensor {key!r} dtype does not match"
                    )
            source_states[name] = source_state
            target_states[name] = copy.deepcopy(target_state)

        source_actor_sha256 = _state_hash(source_states)
        if source_actor_sha256 != authenticated.source_actor_sha256:
            raise ValueError("authenticated actor state changed before transfer")
        target_actor_before = _state_hash(target_states)
        try:
            for name in ACTOR_MODULES:
                target_modules[name].load_state_dict(
                    source_states[name], strict=True,
                )
            target_actor_after = actor_state_sha256(model)
            if target_actor_after != source_actor_sha256:
                raise RuntimeError("actor transfer post-copy hash does not match source")
        except BaseException as error:
            rollback_failures = []
            for name in ACTOR_MODULES:
                try:
                    target_modules[name].load_state_dict(
                        target_states[name], strict=True,
                    )
                except BaseException as rollback_error:
                    rollback_failures.append(f"{name}: {rollback_error!r}")
            if rollback_failures:
                error.add_note(
                    "actor rollback also failed: " + "; ".join(rollback_failures)
                )
            raise

        provenance = {
            "schema_version": 1,
            "kind": "actor_only",
            "source_kind": source.source_kind,
            "source_controller": dict(source.controller),
            "source_run": authenticated.source_run,
            "source_checkpoint": authenticated.source_checkpoint,
            "source_checkpoint_sha256": authenticated.checkpoint_sha256,
            "source_run_manifest_sha256": authenticated.run_manifest_sha256,
            "source_contract_hash": authenticated.contract.contract_hash,
            "source_encoding_hash": authenticated.contract.encoding_hash,
            "source_observation_size": authenticated.observation_size,
            "source_action_size": authenticated.action_size,
            "source_algorithm": authenticated.algorithm,
            "source_policy_class": authenticated.policy_class,
            "source_step": authenticated.step,
            "source_inference_mode": authenticated.inference_mode,
            "actor_modules": list(ACTOR_MODULES),
            "source_actor_sha256": source_actor_sha256,
            "source_published_actor_sha256": source.published_actor_sha256,
            "target_actor_sha256_before": target_actor_before,
            "target_actor_sha256_after": target_actor_after,
            "device": str(requested_device),
        }
        if authenticated.source_bc_sha256 is not None:
            provenance["source_bc_sha256"] = authenticated.source_bc_sha256
        return MappingProxyType(provenance)

    def initialize_actor(
        self,
        model: Any,
        source_run: Path,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]:
        import torch
        from stable_baselines3.common.utils import get_device

        source_run = Path(source_run).resolve(strict=True)
        bc_path = source_run / "bc.json"
        fixtures_path = source_run / "actor-fixtures.npz"
        if not bc_path.is_file():
            raise FileNotFoundError(bc_path)
        if not fixtures_path.is_file():
            raise FileNotFoundError(fixtures_path)
        bc_path = _contained_source_file(source_run, "bc.json")
        fixtures_path = _contained_source_file(
            source_run, "actor-fixtures.npz",
        )
        manifest_path = _contained_source_file(source_run, "run.json")
        bc, bc_sha256 = _read_json_snapshot(bc_path)
        dataset_hash = bc.get("dataset_manifest_sha256")
        if (
            bc.get("schema_version") != 1
            or bc.get("algorithm") != self.name
            or bc.get("policy") != self.policy_name
            or not isinstance(dataset_hash, str)
            or len(dataset_hash) != 64
            or any(character not in "0123456789abcdef" for character in dataset_hash)
        ):
            raise ValueError("actor initialization source is not a valid behavioral clone")

        manifest, manifest_sha256 = _read_json_snapshot(manifest_path)
        manifest_config = manifest.get("config") if isinstance(manifest, Mapping) else None
        clone_config = bc.get("config")
        model_seed = bc.get("model_seed")
        best_epoch = bc.get("best_epoch")
        epochs_trained = bc.get("epochs_trained")
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("state") != "completed"
            or type(manifest.get("timesteps")) is not int
            or manifest.get("timesteps") != 0
            or manifest.get("latest_checkpoint") != "checkpoints/step_000000000.zip"
            or type(manifest.get("latest_checkpoint_step")) is not int
            or manifest.get("latest_checkpoint_step") != 0
            or not isinstance(manifest_config, Mapping)
            or manifest_config.get("algorithm") != self.name
            or manifest_config.get("policy") != self.policy_name
            or not isinstance(clone_config, Mapping)
            or type(model_seed) is not int
            or model_seed < 0
            or type(best_epoch) is not int
            or best_epoch < 1
            or type(epochs_trained) is not int
            or epochs_trained < best_epoch
            or clone_config.get("model_seed") != model_seed
            or manifest.get("dataset_manifest_sha256") != dataset_hash
            or manifest.get("bc_config") != clone_config
            or manifest.get("model_seed") != model_seed
            or manifest.get("best_epoch") != best_epoch
            or manifest_config.get("seed") != model_seed
            or manifest_config.get("model_seed") != model_seed
            or manifest_config.get("behavioral_cloning") != clone_config
        ):
            raise ValueError("behavioral clone metadata does not match run")

        fixtures_bytes = _read_file_snapshot(fixtures_path)
        fixtures_sha256 = hashlib.sha256(fixtures_bytes).hexdigest()
        canonical_checkpoint = (
            source_run / "checkpoints" / "step_000000000.zip"
        ).resolve(strict=True)
        checkpoint_bytes = _read_file_snapshot(canonical_checkpoint)
        checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        source = ActorTransferSource(
            source_kind="snapshot",
            controller={
                "kind": "snapshot",
                "path": str(canonical_checkpoint),
                "source_run": str(source_run),
                "algorithm": self.name,
                "step": 0,
                "inference_mode": "deterministic",
            },
            checkpoint_sha256=checkpoint_sha256,
        )

        fixtures_buffer = io.BytesIO(fixtures_bytes)
        try:
            with np.load(fixtures_buffer, allow_pickle=False) as loaded:
                if set(loaded.files) not in (
                    {"observations", "legal_masks"},
                    {"observations", "legal_masks", "expected_logits"},
                ):
                    raise ValueError(
                        "actor fixtures must contain observations and legal_masks"
                    )
                observations = loaded["observations"].copy()
                legal_masks = loaded["legal_masks"].copy()
        finally:
            fixtures_buffer.close()
        if (
            observations.dtype != np.float32
            or legal_masks.dtype != np.bool_
            or observations.ndim != 2
            or legal_masks.ndim != 2
            or observations.shape[0] == 0
            or observations.shape
            != (legal_masks.shape[0], expected_contract.observation_size)
            or legal_masks.shape[1] != expected_contract.action_size
            or not np.isfinite(observations).all()
            or not legal_masks.any(axis=1).all()
        ):
            raise ValueError("actor fixtures have incompatible shape, dtype, or values")

        requested_device = get_device(device)
        model_device = torch.device(model.device)
        if model_device != requested_device:
            raise ValueError(
                f"actor initialization device {requested_device} does not match model device {model_device}"
            )

        def masked_logits(policy: Any) -> Any:
            with torch.no_grad():
                distribution = policy.get_distribution(
                    torch.as_tensor(
                        observations,
                        dtype=torch.float32,
                        device=requested_device,
                    ),
                    action_masks=torch.as_tensor(
                        legal_masks,
                        dtype=torch.bool,
                        device=requested_device,
                    ),
                )
                return distribution.distribution.logits.detach().cpu()

        source_logits: Any | None = None

        def capture_source_logits(source_model: Any) -> None:
            nonlocal source_logits
            source_model.policy.to(requested_device)
            source_logits = masked_logits(source_model.policy)

        target_states = _actor_states(model, copy_states=True)
        try:
            provenance = dict(
                self._initialize_actor_from_source(
                    model,
                    source,
                    expected_contract,
                    device,
                    checkpoint_bytes=checkpoint_bytes,
                    run_snapshot=(manifest, manifest_sha256),
                    require_exact_contract=False,
                    authenticated_source_callback=capture_source_logits,
                )
            )
            if source_logits is None:
                raise AssertionError("authenticated source logits were not captured")
            target_logits = masked_logits(model.policy)
            rtol = 0.0 if requested_device.type == "cpu" else 1e-6
            atol = 0.0 if requested_device.type == "cpu" else 1e-7
            torch.testing.assert_close(
                target_logits,
                source_logits,
                rtol=rtol,
                atol=atol,
            )
            maximum_difference = float(
                torch.max(torch.abs(target_logits - source_logits)).item()
            )
            provenance.update(
                {
                    "comparison_rtol": rtol,
                    "comparison_atol": atol,
                    "maximum_absolute_logit_difference": maximum_difference,
                    "source_actor_fixtures_sha256": fixtures_sha256,
                    "source_bc_sha256": bc_sha256,
                    "source_dataset_manifest_sha256": dataset_hash,
                }
            )
            return provenance
        except BaseException as error:
            rollback_failures = []
            for name, module in _actor_modules(model).items():
                try:
                    module.load_state_dict(
                        target_states[name],
                        strict=True,
                    )
                except BaseException as rollback_error:
                    rollback_failures.append(f"{name}: {rollback_error!r}")
            if rollback_failures:
                error.add_note(
                    "actor rollback also failed: " + "; ".join(rollback_failures)
                )
            raise

    def load(self, path: Any, *, env: Any, device: str) -> Any:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(path, env=env, device=device)

    def validate_model(self, model: Any, expected_contract: EnvironmentContract) -> None:
        _validate_geometry(model, expected_contract)

    def predict(self, model: Any, observation: np.ndarray, mask: np.ndarray) -> int:
        action, _ = model.predict(
            observation, action_masks=mask, deterministic=True
        )
        return int(action)

    def save(self, model: Any, path: Path) -> Path:
        return _save_sb3_model(model, path)

    def inspect(self, path: Path, expected_contract: EnvironmentContract) -> dict[str, Any]:
        model = self.load(path, env=None, device="cpu")
        self.validate_model(model, expected_contract)
        return _checkpoint_info(expected_contract)


def _masked_dqn_type():
    from stable_baselines3 import DQN

    class MaskedDQN(DQN):
        """Experimental DQN whose exploration and exploitation both honor legal actions."""

        def _action_masks(self) -> np.ndarray:
            assert self.env is not None
            return np.asarray(self.env.env_method("action_masks"), dtype=bool)

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            masks = self._action_masks()
            explore = self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate
            if explore:
                actions = np.asarray(
                    [np.random.choice(np.flatnonzero(mask)) for mask in masks], dtype=np.int64
                )
            else:
                import torch

                with torch.no_grad():
                    observations, _ = self.policy.obs_to_tensor(self._last_obs)
                    values = self.q_net(observations).cpu().numpy()
                values[~masks] = -np.inf
                actions = values.argmax(axis=1)
            return actions, actions

    MaskedDQN.__name__ = "MaskedDQN"
    MaskedDQN.__qualname__ = "MaskedDQN"
    return MaskedDQN


@dataclass(frozen=True)
class MaskedDQNAdapter:
    name: str = "masked_dqn"
    policy_name: str = "MlpPolicy"
    experimental: bool = True

    def create(
        self,
        env: Any,
        *,
        spaces_info: dict[str, Any],
        seed: int,
        device: str,
        checkpoint_interval: int,
    ) -> Any:
        del spaces_info, checkpoint_interval
        return _masked_dqn_type()(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            buffer_size=100_000,
            learning_starts=1_000,
        )

    def load(self, path: Path, *, env: Any, device: str) -> Any:
        return _masked_dqn_type().load(path, env=env, device=device)

    def validate_model(self, model: Any, expected_contract: EnvironmentContract) -> None:
        _validate_geometry(model, expected_contract)

    def predict(self, model: Any, observation: np.ndarray, mask: np.ndarray) -> int:
        import torch

        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation[None]).float().to(model.device)
            values = model.q_net(observation_tensor).cpu().numpy()[0]
        values[~mask] = -np.inf
        return int(np.argmax(values))

    def save(self, model: Any, path: Path) -> Path:
        return _save_sb3_model(model, path)

    def inspect(self, path: Path, expected_contract: EnvironmentContract) -> dict[str, Any]:
        model = self.load(path, env=None, device="cpu")
        self.validate_model(model, expected_contract)
        return _checkpoint_info(expected_contract)


def _checkpoint_info(contract: EnvironmentContract) -> dict[str, Any]:
    return {
        "environment": contract.environment,
        "contract_version": contract.version,
        "contract_hash": contract.contract_hash,
        "encoding_hash": contract.encoding_hash,
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
    }


def get_algorithm_adapter(name: str | None) -> AlgorithmAdapter:
    selected = name or "maskable_ppo"
    if selected == "maskable_ppo":
        return MaskablePPOAdapter()
    if selected == "masked_dqn":
        return MaskedDQNAdapter()
    raise ValueError(f"unsupported algorithm {selected!r}")


def _manifest_for_resume(source: Path) -> tuple[Path | None, Path]:
    source = Path(source)
    if source.is_dir() and (source / "run.json").is_file():
        return source / "run.json", source
    if source.is_file():
        for directory in (source.parent, source.parent.parent):
            manifest_path = directory / "run.json"
            if manifest_path.is_file():
                return manifest_path, directory
        return None, source
    raise FileNotFoundError(source)


def resolve_resume_checkpoint(
    source: Path,
    expected_algorithm: str,
    expected_contract: EnvironmentContract,
    *,
    allow_unsafe_legacy_resume: bool = False,
) -> Path:
    """Resolve a resume source and reject any authoritative metadata mismatch."""
    manifest_path, run_or_checkpoint = _manifest_for_resume(Path(source))
    if manifest_path is None:
        checkpoint = Path(run_or_checkpoint)
        if checkpoint.suffix.lower() != ".zip":
            raise ValueError("legacy resume source must be a .zip checkpoint")
        raise ValueError(
            "standalone checkpoint resume is unsupported; use a metadata-backed run directory"
        )

    manifest = read_json(manifest_path)
    config = manifest.get("config", {})
    actual_algorithm = config.get("algorithm")
    if actual_algorithm != expected_algorithm:
        raise ValueError(
            f"resume algorithm {actual_algorithm!r} does not match {expected_algorithm!r}"
        )
    if manifest.get("contract") != expected_contract.to_dict():
        raise ContractMismatch("resume training contract does not match the current environment")

    source_path = Path(source)
    if source_path.is_file():
        checkpoint = source_path.resolve()
    else:
        relative = manifest.get("latest_checkpoint")
        if not isinstance(relative, str) or not relative:
            raise ValueError("resume run has no published checkpoint")
        checkpoint = (Path(run_or_checkpoint) / relative).resolve()
    run_dir = Path(run_or_checkpoint).resolve()
    if run_dir not in checkpoint.parents or checkpoint.suffix.lower() != ".zip":
        raise ValueError("resume checkpoint must be a .zip inside its run directory")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def create_or_resume_model(
    adapter: AlgorithmAdapter,
    *,
    env: Any,
    expected_contract: EnvironmentContract,
    spaces_info: dict[str, Any],
    seed: int,
    device: str,
    checkpoint_interval: int,
    resume_source: Path | None,
    allow_unsafe_legacy_resume: bool = False,
    algorithm_options: Mapping[str, Any] | None = None,
) -> tuple[Any, bool]:
    if resume_source is None:
        create_options = {
            "spaces_info": spaces_info,
            "seed": seed,
            "device": device,
            "checkpoint_interval": checkpoint_interval,
        }
        if algorithm_options:
            create_options["algorithm_options"] = algorithm_options
        model = adapter.create(env, **create_options)
        resumed = False
    else:
        if adapter.name == "masked_dqn":
            raise ValueError(
                "masked_dqn resume is disabled until replay buffer sidecars are persisted"
            )
        checkpoint = resolve_resume_checkpoint(
            Path(resume_source),
            adapter.name,
            expected_contract,
            allow_unsafe_legacy_resume=allow_unsafe_legacy_resume,
        )
        model = adapter.load(checkpoint, env=env, device=device)
        resumed = True
    adapter.validate_model(model, expected_contract)
    return model, resumed
