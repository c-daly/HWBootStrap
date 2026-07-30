"""Central SB3 algorithm adapters for HexWars training and checkpoint validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import read_json


ACTOR_MODULES = {
    "features_extractor": lambda policy: policy.features_extractor,
    "policy_net": lambda policy: policy.mlp_extractor.policy_net,
    "action_net": lambda policy: policy.action_net,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

        resolved = ControllerResolver(expected_contract).resolve(f"run:{source_run}")
        if resolved.contract != expected_contract:
            raise ContractMismatch(
                "actor initialization contract does not match the training contract"
            )
        if (
            resolved.algorithm != self.name
            or resolved.model is None
            or resolved.path is None
        ):
            raise ValueError("actor initialization source is not a MaskablePPO run")
        source_model = resolved.model
        self.validate_model(source_model, expected_contract)
        self.validate_model(model, expected_contract)

        with np.load(fixtures_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {"observations", "legal_masks"}:
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

        source_modules = {
            name: accessor(source_model.policy)
            for name, accessor in ACTOR_MODULES.items()
        }
        target_modules = {
            name: accessor(model.policy)
            for name, accessor in ACTOR_MODULES.items()
        }
        source_states: dict[str, Mapping[str, Any]] = {}
        for name in ACTOR_MODULES:
            source_state = source_modules[name].state_dict()
            target_state = target_modules[name].state_dict()
            if tuple(source_state) != tuple(target_state):
                raise ContractMismatch(f"actor module {name!r} state keys do not match")
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
        for name in ACTOR_MODULES:
            target_modules[name].load_state_dict(source_states[name], strict=True)
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
        checkpoint = resolved.path.resolve()
        return {
            "schema_version": 1,
            "kind": "actor_only",
            "actor_modules": list(ACTOR_MODULES),
            "device": str(requested_device),
            "comparison_rtol": rtol,
            "comparison_atol": atol,
            "maximum_absolute_logit_difference": maximum_difference,
            "source_run": str(source_run),
            "source_checkpoint": checkpoint.relative_to(source_run).as_posix(),
            "source_checkpoint_sha256": _sha256_file(checkpoint),
            "source_actor_fixtures_sha256": _sha256_file(fixtures_path),
            "source_run_manifest_sha256": _sha256_file(source_run / "run.json"),
            "source_bc_sha256": _sha256_file(bc_path),
            "source_dataset_manifest_sha256": dataset_hash,
            "source_contract_hash": expected_contract.contract_hash,
            "source_encoding_hash": expected_contract.encoding_hash,
        }

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
