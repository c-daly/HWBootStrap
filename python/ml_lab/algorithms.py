"""Central SB3 algorithm adapters for HexWars training and checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import read_json


class AlgorithmAdapter(Protocol):
    name: str
    policy_name: str
    experimental: bool

    def create(self, env: Any, **kwargs: Any) -> Any: ...

    def load(self, path: Path, *, env: Any, device: str) -> Any: ...

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
    ) -> Any:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

        from hex_cnn import cnn_policy_kwargs

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
        )

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
        "contract_hash": contract.contract_hash,
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
) -> Path:
    """Resolve a resume source and reject any authoritative metadata mismatch."""
    manifest_path, run_or_checkpoint = _manifest_for_resume(Path(source))
    if manifest_path is None:
        checkpoint = Path(run_or_checkpoint)
        if checkpoint.suffix.lower() != ".zip":
            raise ValueError("legacy resume source must be a .zip checkpoint")
        return checkpoint.resolve()

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
) -> tuple[Any, bool]:
    if resume_source is None:
        model = adapter.create(
            env,
            spaces_info=spaces_info,
            seed=seed,
            device=device,
            checkpoint_interval=checkpoint_interval,
        )
        resumed = False
    else:
        checkpoint = resolve_resume_checkpoint(
            Path(resume_source), adapter.name, expected_contract
        )
        model = adapter.load(checkpoint, env=env, device=device)
        resumed = True
    adapter.validate_model(model, expected_contract)
    return model, resumed
