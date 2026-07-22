"""Unified headless HexWars SB3 training orchestration."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .algorithms import AlgorithmAdapter, create_or_resume_model, get_algorithm_adapter
from .callbacks import SB3RunCallback, TrainingLifecycle, TrainingStopRequested
from .contracts import RunConfig, create_run, update_run_state
from .envs import TrainingEnvironmentFactory
from .io import read_json
from .tracking import TrackerHub


EnvironmentFactory = Callable[[RunConfig, Path], Any]


def _default_summary_writer(log_dir: Path) -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def build_sb3_logger(run_dir: Path, *, stdout: TextIO = sys.stdout) -> Any:
    """Mirror SB3's human-readable output to the console and durable run log."""
    from stable_baselines3.common.logger import HumanOutputFormat, Logger

    log_stream = (Path(run_dir) / "train.log").open("a", encoding="utf-8")
    file_output = HumanOutputFormat(log_stream)
    file_output.own_file = True
    return Logger(
        folder=str(run_dir),
        output_formats=[HumanOutputFormat(stdout), file_output],
    )


def _log(run_dir: Path, message: str) -> None:
    with (Path(run_dir) / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{message}\n")
        stream.flush()


def _resume_run_dir(source: Path | None) -> Path | None:
    if source is None:
        return None
    source = Path(source)
    if source.is_dir() and (source / "run.json").is_file():
        return source
    if source.is_file():
        for directory in (source.parent, source.parent.parent):
            if (directory / "run.json").is_file():
                return directory
    return None


def _resume_episode_count(source: Path | None, model: Any) -> int:
    counts = [int(getattr(model, "_episode_num", 0) or 0)]
    run_dir = _resume_run_dir(source)
    if run_dir is None:
        return max(counts)
    manifest = read_json(run_dir / "run.json")
    manifest_episodes = manifest.get("episodes")
    if isinstance(manifest_episodes, int) and not isinstance(manifest_episodes, bool):
        counts.append(manifest_episodes)
    progress_path = run_dir / "progress.csv"
    if progress_path.is_file():
        with progress_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    counts.append(int(row.get("episodes") or 0))
                except ValueError:
                    pass
    monitor_paths = [run_dir / "monitor.csv", *sorted(run_dir.glob("monitor.worker_*.csv"))]
    monitor_total = 0
    for path in monitor_paths:
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as stream:
                monitor_total += sum(1 for _ in csv.DictReader(stream))
    counts.append(monitor_total)
    return max(counts)


def run_training(
    config: RunConfig,
    *,
    runs_root: Path,
    server_cmd: list[str] | None = None,
    environment_factory: EnvironmentFactory | None = None,
    algorithm_adapter: AlgorithmAdapter | None = None,
    tracker_factory: Callable[..., TrackerHub] = TrackerHub,
    summary_writer_factory: Callable[[Path], Any | None] = _default_summary_writer,
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
    tensorboard_writer: Any | None = None
    sb3_logger: Any | None = None
    run_created = False
    final_status = "failed"
    try:
        env = environment_factory(config, run_dir)
        contract = env.contract
        spaces_info = dict(env.spaces_info)
        create_run(Path(runs_root), config, contract)
        run_created = True
        if any(str(spec.get("kind", "")) == "tensorboard" for spec in config.trackers):
            tensorboard_writer = summary_writer_factory(run_dir / "tensorboard")
        trackers = tracker_factory(
            run_dir,
            list(config.trackers),
            tensorboard_writer=tensorboard_writer,
        )
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
            allow_unsafe_legacy_resume=config.allow_unsafe_legacy_resume,
        )
        sb3_logger = build_sb3_logger(run_dir)
        set_logger = getattr(model, "set_logger", None)
        if callable(set_logger):
            set_logger(sb3_logger)
        current_step = int(getattr(model, "num_timesteps", 0))
        lifecycle = TrainingLifecycle(
            run_dir,
            contract,
            adapter,
            trackers,
            checkpoint_interval=config.checkpoint_interval,
            initial_step=current_step,
            episode_offset=_resume_episode_count(resume_source, model),
        )
        remaining = (
            config.total_timesteps
            if config.timestep_mode == "additional"
            else max(0, config.total_timesteps - current_step)
        )
        if remaining:
            try:
                model.learn(
                    total_timesteps=remaining,
                    callback=SB3RunCallback(lifecycle),
                    reset_num_timesteps=not resumed,
                )
            except TrainingStopRequested:
                pass

        final_step = int(getattr(model, "num_timesteps", current_step))
        if lifecycle.stop_mode == "stop_after_checkpoint" and not lifecycle.stop_requested:
            lifecycle.publish_checkpoint(model, final_step)
            lifecycle.stop_requested = True
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
        if tensorboard_writer is not None:
            close_writer = getattr(tensorboard_writer, "close", None)
            if callable(close_writer):
                close_writer()
        if sb3_logger is not None:
            sb3_logger.close()
        if env is not None:
            env.close()
