"""Unified headless HexWars SB3 training orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .algorithms import AlgorithmAdapter, create_or_resume_model, get_algorithm_adapter
from .callbacks import SB3RunCallback, TrainingLifecycle
from .contracts import RunConfig, create_run, update_run_state
from .envs import TrainingEnvironmentFactory
from .tracking import TrackerHub


EnvironmentFactory = Callable[[RunConfig, Path], Any]


def _log(run_dir: Path, message: str) -> None:
    with (Path(run_dir) / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{message}\n")
        stream.flush()


def run_training(
    config: RunConfig,
    *,
    runs_root: Path,
    server_cmd: list[str] | None = None,
    environment_factory: EnvironmentFactory | None = None,
    algorithm_adapter: AlgorithmAdapter | None = None,
    tracker_factory: Callable[[Path, list[dict[str, Any]]], TrackerHub] = TrackerHub,
) -> Path:
    """Run one local experiment and leave its manifest in a terminal state."""
    adapter = algorithm_adapter or get_algorithm_adapter(config.algorithm)
    if adapter.name != config.algorithm:
        raise ValueError("algorithm adapter does not match the run configuration")
    if adapter.policy_name != config.policy:
        raise ValueError("algorithm policy does not match the run configuration")
    if config.total_timesteps < 0:
        raise ValueError("total timesteps must be non-negative")
    if config.workers <= 0:
        raise ValueError("worker count must be positive")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")

    run_dir = Path(runs_root) / config.run_name
    if environment_factory is None:
        if not server_cmd:
            raise ValueError("server command is required for real training")
        environment_factory = TrainingEnvironmentFactory(server_cmd)

    env: Any | None = None
    trackers: TrackerHub | None = None
    run_created = False
    final_status = "failed"
    try:
        env = environment_factory(config, run_dir)
        contract = env.contract
        spaces_info = dict(env.spaces_info)
        create_run(Path(runs_root), config, contract)
        run_created = True
        trackers = tracker_factory(run_dir, list(config.trackers))
        trackers.start_run()
        update_run_state(
            run_dir,
            "running",
            pid=os.getpid(),
            timesteps=0,
            latest_message="training started",
        )
        _log(run_dir, f"starting {config.algorithm} training")

        resume_source = Path(config.resume_source) if config.resume_source else None
        model, resumed = create_or_resume_model(
            adapter,
            env=env,
            expected_contract=contract,
            spaces_info=spaces_info,
            seed=config.seed,
            device=config.device,
            checkpoint_interval=config.checkpoint_interval,
            resume_source=resume_source,
        )
        lifecycle = TrainingLifecycle(
            run_dir,
            contract,
            adapter,
            trackers,
            checkpoint_interval=config.checkpoint_interval,
        )
        current_step = int(getattr(model, "num_timesteps", 0))
        remaining = max(0, config.total_timesteps - current_step)
        if remaining:
            model.learn(
                total_timesteps=remaining,
                callback=SB3RunCallback(lifecycle),
                reset_num_timesteps=not resumed,
            )

        final_step = int(getattr(model, "num_timesteps", current_step))
        if lifecycle.stop_requested:
            final_status = "stopped"
            update_run_state(
                run_dir,
                "stopped",
                pid=None,
                timesteps=final_step,
                latest_message="training stopped by request",
            )
            _log(run_dir, "training stopped by request")
        else:
            lifecycle.publish_checkpoint(model, final_step)
            final_status = "completed"
            update_run_state(
                run_dir,
                "completed",
                pid=None,
                timesteps=final_step,
                latest_message="training completed",
            )
            _log(run_dir, "training completed")
        return run_dir
    except KeyboardInterrupt:
        final_status = "stopped"
        if run_created:
            update_run_state(
                run_dir,
                "stopped",
                pid=None,
                latest_message="training interrupted",
            )
            _log(run_dir, "training interrupted")
        return run_dir
    except BaseException as error:
        final_status = "failed"
        if run_created:
            update_run_state(
                run_dir,
                "failed",
                pid=None,
                latest_message=f"training failed: {type(error).__name__}: {error}",
            )
            _log(run_dir, f"training failed: {type(error).__name__}: {error}")
        raise
    finally:
        if trackers is not None:
            trackers.finish(final_status)
        if env is not None:
            env.close()
