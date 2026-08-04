"""Central SB3 algorithm adapters for HexWars training and checkpoint validation."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import read_json


ACTOR_MODULES = {
    "features_extractor": lambda policy: policy.features_extractor,
    "policy_net": lambda policy: policy.mlp_extractor.policy_net,
    "action_net": lambda policy: policy.action_net,
}

ACTOR_SOURCE_KINDS = frozenset({"snapshot", "dagger_actor"})


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_actor_transfer_source(
    source: ActorTransferSource,
) -> tuple[Path, Path, str]:
    """Authenticate physical source location and bytes before model deserialization."""

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
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != source.checkpoint_sha256:
        raise ValueError("actor transfer checkpoint SHA-256 does not match")
    return source_run, checkpoint, checkpoint_sha256


class AlgorithmAdapter(Protocol):
    name: str
    policy_name: str
    experimental: bool

    def create(self, env: Any, **kwargs: Any) -> Any: ...

    def load(self, path: Path, *, env: Any, device: str) -> Any: ...

    def initialize_actor(
        self,
        model: Any,
        source_run: Path,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]: ...

    def initialize_actor_from_resolved(
        self,
        model: Any,
        resolved: Any,
        *,
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

    def initialize_actor_from_resolved(
        self,
        model: Any,
        resolved: Any,
        *,
        source: ActorTransferSource,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]:
        """Preflight and transactionally copy a resolved actor without value state."""

        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import (
            MaskableActorCriticPolicy,
        )
        from stable_baselines3.common.utils import get_device

        from .controllers import ResolvedController

        source_run, checkpoint, _preflight_sha256 = (
            preflight_actor_transfer_source(source)
        )
        if not isinstance(resolved, ResolvedController):
            raise TypeError("resolved source must be a ResolvedController")
        controller = source.controller
        step = controller["step"]
        if (
            resolved.algorithm != self.name
            or resolved.spec.algorithm != self.name
            or controller["algorithm"] != self.name
        ):
            raise ValueError("actor transfer source algorithm is not MaskablePPO")
        if (
            resolved.spec.inference_mode != "deterministic"
            or controller["inference_mode"] != "deterministic"
        ):
            raise ValueError("actor transfer source must use deterministic inference")
        if (
            resolved.spec.kind != "snapshot"
            or resolved.spec.path is None
            or resolved.spec.source_run is None
            or resolved.path is None
            or resolved.step != step
            or resolved.spec.step != step
            or resolved.spec.path.resolve() != checkpoint
            or resolved.path.resolve() != checkpoint
            or resolved.spec.source_run.resolve() != source_run
        ):
            raise ValueError("actor transfer source checkpoint step or identity differs")
        if (
            not checkpoint.is_relative_to(source_run)
            or checkpoint.parent != (source_run / "checkpoints").resolve()
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
        checkpoint_sha256 = _sha256_file(checkpoint)
        if checkpoint_sha256 != source.checkpoint_sha256:
            raise ValueError("actor transfer checkpoint SHA-256 does not match")

        manifest_path = source_run / "run.json"
        manifest = read_json(manifest_path)
        manifest_config = (
            manifest.get("config") if isinstance(manifest, Mapping) else None
        )
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("latest_checkpoint")
            != checkpoint.relative_to(source_run).as_posix()
            or manifest.get("latest_checkpoint_step") != step
            or not isinstance(manifest_config, Mapping)
            or manifest_config.get("algorithm") != self.name
            or manifest_config.get("policy") != self.policy_name
        ):
            raise ValueError("actor transfer source run metadata does not match")
        if source.source_kind == "dagger_actor":
            bc = read_json(source_run / "bc.json")
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
                or manifest.get("checkpoint_sha256") != checkpoint_sha256
                or manifest.get("target_actor_sha256_final")
                != published_actor_sha256
                or not isinstance(actor_initialization, Mapping)
                or not isinstance(publication_verification, Mapping)
                or publication_verification.get("checkpoint_sha256")
                != checkpoint_sha256
                or publication_verification.get("actor_sha256")
                != published_actor_sha256
                or not isinstance(bc, Mapping)
                or bc.get("schema_version") != 1
                or bc.get("training_kind")
                != "selective-dagger-distillation-v1"
                or bc.get("algorithm") != self.name
                or bc.get("policy") != self.policy_name
                or bc.get("production") is not True
                or bc.get("checkpoint_sha256") != checkpoint_sha256
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

        if resolved.contract != expected_contract:
            raise ContractMismatch(
                "actor transfer source contract does not exactly match"
            )
        if (
            resolved.observation_size != expected_contract.observation_size
            or resolved.action_size != expected_contract.action_size
        ):
            raise ContractMismatch("actor transfer source geometry does not match")
        if resolved.model is None:
            raise ValueError("actor transfer source model is missing")
        source_model = resolved.model
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
        if (
            source.source_kind == "dagger_actor"
            and source_actor_sha256 != source.published_actor_sha256
        ):
            raise ValueError(
                "published DAgger actor hash does not match the loaded physical actor"
            )
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

        return {
            "schema_version": 1,
            "kind": "actor_only",
            "source_kind": source.source_kind,
            "source_controller": dict(controller),
            "source_run": str(source_run),
            "source_checkpoint": checkpoint.relative_to(source_run).as_posix(),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_run_manifest_sha256": _sha256_file(manifest_path),
            "source_contract_hash": resolved.contract.contract_hash,
            "source_encoding_hash": resolved.contract.encoding_hash,
            "source_observation_size": resolved.observation_size,
            "source_action_size": resolved.action_size,
            "source_algorithm": resolved.algorithm,
            "source_policy_class": (
                f"{type(source_model.policy).__module__}."
                f"{type(source_model.policy).__qualname__}"
            ),
            "source_step": resolved.step,
            "source_inference_mode": resolved.spec.inference_mode,
            "actor_modules": list(ACTOR_MODULES),
            "source_actor_sha256": source_actor_sha256,
            "source_published_actor_sha256": source.published_actor_sha256,
            "target_actor_sha256_before": target_actor_before,
            "target_actor_sha256_after": target_actor_after,
            "device": str(requested_device),
        }

    def initialize_actor(
        self,
        model: Any,
        source_run: Path,
        expected_contract: EnvironmentContract,
        device: str,
    ) -> Mapping[str, Any]:
        import torch
        from stable_baselines3.common.utils import get_device

        from .controllers import ControllerResolver

        source_run = Path(source_run).resolve()
        bc_path = source_run / "bc.json"
        fixtures_path = source_run / "actor-fixtures.npz"
        if not bc_path.is_file():
            raise FileNotFoundError(bc_path)
        if not fixtures_path.is_file():
            raise FileNotFoundError(fixtures_path)
        bc = read_json(bc_path)
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

        manifest = read_json(source_run / "run.json")
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

        canonical_checkpoint = (
            source_run / "checkpoints" / "step_000000000.zip"
        ).resolve()
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
            checkpoint_sha256=_sha256_file(canonical_checkpoint),
        )
        preflight_actor_transfer_source(source)
        resolved = ControllerResolver(expected_contract).resolve(source.controller)
        if resolved.contract is None:
            raise ContractMismatch("actor initialization source contract is missing")
        if (
            resolved.algorithm != self.name
            or resolved.model is None
            or resolved.step != 0
            or resolved.path is None
            or resolved.path.resolve() != canonical_checkpoint
        ):
            raise ValueError("actor initialization source is not a MaskablePPO run")
        source_model = resolved.model
        self.validate_model(source_model, expected_contract)
        self.validate_model(model, expected_contract)

        with np.load(fixtures_path, allow_pickle=False) as loaded:
            if set(loaded.files) not in (
                {"observations", "legal_masks"},
                {"observations", "legal_masks", "expected_logits"},
            ):
                raise ValueError("actor fixtures must contain observations and legal_masks")
            observations = loaded["observations"].copy()
            legal_masks = loaded["legal_masks"].copy()
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
        source_model.policy.to(requested_device)

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

        source_logits = masked_logits(source_model.policy)
        target_states = _actor_states(model, copy_states=True)
        try:
            provenance = dict(
                self.initialize_actor_from_resolved(
                    model,
                    resolved,
                    source=source,
                    expected_contract=expected_contract,
                    device=device,
                )
            )
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
                    "source_actor_fixtures_sha256": _sha256_file(fixtures_path),
                    "source_bc_sha256": _sha256_file(bc_path),
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

    def load(self, path: Path, *, env: Any, device: str) -> Any:
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
