"""SB3 callback lifecycle backed by the durable HexWars run contract."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .algorithms import AlgorithmAdapter
from .contracts import EnvironmentContract, publish_checkpoint, update_run_state
from .io import read_json
from .tracking import SB3TrackingFacade, TrackerHub


class TrainingStopRequested(RuntimeError):
    """Internal control-flow signal raised only after a requested checkpoint is safe."""


class TrainingLifecycle:
    """Testable callback core for progress, controls, and checkpoint publication."""

    def __init__(
        self,
        run_dir: Path,
        contract: EnvironmentContract,
        adapter: AlgorithmAdapter,
        trackers: TrackerHub,
        *,
        checkpoint_interval: int,
        clock: Callable[[], float] = time.monotonic,
        initial_step: int = 0,
        episode_offset: int = 0,
    ) -> None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.run_dir = Path(run_dir)
        self.contract = contract
        self.adapter = adapter
        self.trackers = trackers
        self.tracking_facade = SB3TrackingFacade(trackers)
        self.checkpoint_interval = checkpoint_interval
        self._clock = clock
        manifest = read_json(self.run_dir / "run.json")
        latest = manifest.get("latest_checkpoint_step")
        base_step = latest if isinstance(latest, int) and not isinstance(latest, bool) else 0
        base_step = max(base_step, initial_step)
        self._next_checkpoint = (base_step // checkpoint_interval + 1) * checkpoint_interval
        self._pending_checkpoint_threshold: int | None = None
        self._last_progress_step: int | None = None
        self._last_metric_step = base_step
        self._last_metric_time = clock()
        self._episode_offset = max(0, episode_offset)
        self._episode_maxima: dict[int, int] = {}
        self._episodes_total = self._episode_offset
        self.stop_requested = False
        self.stop_mode: str | None = None

    def on_rollout_end(self, model: Any) -> None:
        step = int(model.num_timesteps)
        self._record_progress(model, step)
        update_run_state(
            self.run_dir,
            "stopping" if self.stop_mode else "running",
            timesteps=step,
            episodes=self._episodes_total,
            latest_message="rollout completed",
        )
        self._flush_logger(model)

    def _record_progress(self, model: Any, step: int) -> None:
        if self._last_progress_step == step:
            return
        values = dict(getattr(model.logger, "name_to_value", {}))
        now = self._clock()
        elapsed = now - self._last_metric_time
        episode_buffer = list(getattr(model, "ep_info_buffer", ()))
        anonymous_episodes = 0
        for item in episode_buffer:
            if not isinstance(item, Mapping):
                continue
            worker_id = item.get("worker_id")
            episode_number = item.get("episode_number")
            if (
                isinstance(worker_id, int)
                and not isinstance(worker_id, bool)
                and isinstance(episode_number, int)
                and not isinstance(episode_number, bool)
                and episode_number > 0
            ):
                self._episode_maxima[worker_id] = max(
                    self._episode_maxima.get(worker_id, 0), episode_number
                )
            elif "r" in item:
                anonymous_episodes += 1
        buffered_episodes = (
            self._episode_offset
            + sum(self._episode_maxima.values())
            + anonymous_episodes
        )
        reported_episodes = values.get(
            "time/episodes", getattr(model, "_episode_num", 0)
        )
        episodes = max(
            self._episodes_total,
            int(reported_episodes or 0),
            buffered_episodes,
        )
        self._episodes_total = episodes
        mean_reward = values.get("rollout/ep_rew_mean")
        if mean_reward is None:
            rewards = [
                float(item["r"])
                for item in episode_buffer
                if isinstance(item, Mapping) and "r" in item
            ]
            mean_reward = float(np.mean(rewards)) if rewards else ""
        steps_per_second = values.get("time/fps")
        if steps_per_second is None:
            steps_per_second = (
                (step - self._last_metric_step) / elapsed if elapsed > 0 else ""
            )
        normalized = dict(values)
        normalized.update(
            {
                "episodes": episodes,
                "mean_reward": mean_reward,
                "steps_per_second": steps_per_second,
            }
        )
        logger_view = type("LoggerView", (), {"name_to_value": normalized})()
        self.tracking_facade.log_from_logger(logger_view, step=step)
        self._last_progress_step = step
        self._last_metric_step = step
        self._last_metric_time = now

    def on_step(self, model: Any) -> bool:
        step = int(model.num_timesteps)
        request = read_json(self.run_dir / "control.json").get("request")
        if request == "stop_now":
            self.stop_requested = True
            self.stop_mode = request
            update_run_state(
                self.run_dir,
                "stopping",
                timesteps=step,
                latest_message="immediate stop requested",
            )
            self._flush_logger(model)
            return False
        if request == "stop_after_checkpoint" and self.stop_mode is None:
            self.stop_mode = request
            update_run_state(
                self.run_dir,
                "stopping",
                timesteps=step,
                latest_message="stop requested after next checkpoint",
            )

        if step >= self._next_checkpoint and self._pending_checkpoint_threshold is None:
            self._pending_checkpoint_threshold = self._next_checkpoint
        return True

    def on_rollout_start(self, model: Any) -> None:
        """Publish a due checkpoint only after the previous rollout's policy update."""
        if self._pending_checkpoint_threshold is None:
            return
        due_threshold = self._pending_checkpoint_threshold
        actual_step = int(model.num_timesteps)
        self.publish_checkpoint(model, actual_step)
        self._pending_checkpoint_threshold = None
        self._next_checkpoint = due_threshold + self.checkpoint_interval
        while self._next_checkpoint <= actual_step:
            self._next_checkpoint += self.checkpoint_interval
        if self.stop_mode == "stop_after_checkpoint":
            self.stop_requested = True
            raise TrainingStopRequested("stop after checkpoint completed")

    def publish_checkpoint(self, model: Any, step: int | None = None) -> Path:
        checkpoint_step = int(model.num_timesteps if step is None else step)
        manifest = read_json(self.run_dir / "run.json")
        if manifest.get("latest_checkpoint_step") == checkpoint_step:
            return self.run_dir / manifest["latest_checkpoint"]

        pending = self.run_dir / "checkpoints" / (
            f".pending-{os.getpid()}-{checkpoint_step}-{uuid.uuid4().hex}.zip"
        )
        written: Path | None = None
        try:
            written = self.adapter.save(model, pending)
            destination = publish_checkpoint(
                source=written,
                run_dir=self.run_dir,
                step=checkpoint_step,
                expected_contract=self.contract,
                inspector=lambda path: self.adapter.inspect(path, self.contract),
            )
        finally:
            pending.unlink(missing_ok=True)
            if written is not None and written != pending:
                written.unlink(missing_ok=True)

        self.trackers.log_artifact(
            destination, name=f"{self.run_dir.name}-step-{checkpoint_step}"
        )
        self._record_progress(model, checkpoint_step)
        update_run_state(
            self.run_dir,
            "stopping" if self.stop_mode else "running",
            timesteps=checkpoint_step,
            episodes=self._episodes_total,
            latest_message=f"published checkpoint at step {checkpoint_step}",
        )
        self._flush_logger(model)
        return destination

    @staticmethod
    def _flush_logger(model: Any) -> None:
        logger = getattr(model, "logger", None)
        for output in getattr(logger, "output_formats", []):
            flush = getattr(output, "flush", None)
            if callable(flush):
                flush()
                continue
            stream = getattr(output, "file", None)
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()


class SB3RunCallback(BaseCallback):
    """Thin Stable-Baselines callback wrapper around :class:`TrainingLifecycle`."""

    def __init__(self, lifecycle: TrainingLifecycle) -> None:
        super().__init__(verbose=0)
        self.lifecycle = lifecycle

    def _on_step(self) -> bool:
        return self.lifecycle.on_step(self.model)

    def _on_rollout_start(self) -> None:
        self.lifecycle.on_rollout_start(self.model)

    def _on_rollout_end(self) -> None:
        self.lifecycle.on_rollout_end(self.model)
