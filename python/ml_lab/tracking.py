"""Fault-isolated experiment tracking mirrors for locally durable ML runs."""

from __future__ import annotations

import csv
import importlib
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from .contracts import PROGRESS_HEADER, utc_now, validate_tracker_specs
from .io import atomic_write_json, read_json


MetricEvent = dict[str, Any]
TrackerCallback = Callable[[MetricEvent], None]


class TrackerHub:
    """Keep local run files authoritative while forwarding optional tracker events."""

    def __init__(
        self,
        run_dir: Path,
        tracker_specs: list[Mapping[str, Any]],
        *,
        tensorboard_writer: Any | None = None,
    ) -> None:
        validate_tracker_specs(tracker_specs)
        self.run_dir = Path(run_dir)
        self._tensorboard_writer = tensorboard_writer
        self._adapters: dict[str, TrackerCallback] = {}
        self._wandb_runs: dict[str, Any] = {}
        self._wandb_specs: dict[str, Mapping[str, Any]] = {}
        self._degraded: dict[str, str] = {}
        self._configure(trackers=tracker_specs)

    @property
    def degraded(self) -> dict[str, str]:
        """A copy of optional tracker failures keyed by their stable adapter name."""
        return dict(self._degraded)

    def start_run(self) -> None:
        """Start optional mirrors; the local CSV remains available regardless."""
        self._write_tracker_status()
        self._dispatch(self._event("start"))

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int) -> None:
        """Append authoritative local progress, then safely mirror a normalized event."""
        event = self._event("metrics", step=step, metrics=dict(metrics))
        self._append_local_progress(event)
        self._dispatch(event)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """Offer a completed local artifact to trackers that explicitly support it."""
        artifact_path = Path(path)
        self._dispatch(
            self._event(
                "artifact",
                path=str(artifact_path),
                name=name or artifact_path.name,
            )
        )

    def finish(self, status: str) -> None:
        """Finish every healthy optional tracker without affecting the local run."""
        self._dispatch(self._event("finish", status=status))

    def _configure(self, *, trackers: list[Mapping[str, Any]]) -> None:
        for index, spec in enumerate(trackers):
            kind = str(spec.get("kind", ""))
            if kind == "" or kind == "local":
                continue
            try:
                if kind == "tensorboard":
                    self._configure_tensorboard(index)
                elif kind == "wandb":
                    self._configure_wandb(index, spec)
                elif kind == "custom":
                    self._configure_custom(index, spec)
                else:
                    raise ValueError(f"unknown tracker kind {kind!r}")
            except Exception as error:  # Optional integrations must never stop training.
                name = self._tracker_name(kind, index, spec)
                self._mark_degraded(name, "configure", error)

    def _configure_tensorboard(self, index: int) -> None:
        name = self._tracker_name("tensorboard", index, {})
        if self._tensorboard_writer is None:
            raise RuntimeError("TensorBoard tracker requires an injected writer")

        def record(event: MetricEvent) -> None:
            if event["type"] == "metrics":
                for key, value in event["metrics"].items():
                    if isinstance(value, Real) and not isinstance(value, bool):
                        self._tensorboard_writer.add_scalar(key, float(value), event["step"])
            elif event["type"] == "finish":
                flush = getattr(self._tensorboard_writer, "flush", None)
                if callable(flush):
                    flush()

        self._adapters[name] = record

    def _configure_wandb(self, index: int, spec: Mapping[str, Any]) -> None:
        name = self._tracker_name("wandb", index, spec)
        # Import only for an explicitly configured W&B mirror. Authentication is W&B-owned.
        wandb = importlib.import_module("wandb")
        self._wandb_specs[name] = spec

        def record(event: MetricEvent) -> None:
            if event["type"] == "start":
                self._wandb_runs[name] = wandb.init(**self._wandb_start_options(spec))
            elif event["type"] == "metrics" and name in self._wandb_runs:
                self._wandb_runs[name].log(event["metrics"], step=event["step"])
            elif event["type"] == "artifact" and spec.get("upload_artifacts"):
                self._wandb_log_artifact(wandb, self._wandb_runs.get(name), event)
            elif event["type"] == "finish" and name in self._wandb_runs:
                exit_code = 0 if event["status"] == "completed" else 1
                self._wandb_runs[name].finish(exit_code=exit_code)

        self._adapters[name] = record

    def _configure_custom(self, index: int, spec: Mapping[str, Any]) -> None:
        target = spec.get("adapter")
        if not isinstance(target, str) or target.count(":") != 1:
            raise ValueError("custom tracker adapter must use 'module:function'")
        module_name, function_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        callback = getattr(module, function_name)
        if not callable(callback):
            raise TypeError(f"custom tracker adapter {target!r} is not callable")
        self._adapters[self._tracker_name("custom", index, spec)] = callback

    def _dispatch(self, event: MetricEvent) -> None:
        operation = "log_metrics" if event["type"] == "metrics" else f"{event['type']}_run"
        for name, callback in list(self._adapters.items()):
            if name in self._degraded:
                continue
            try:
                callback(event)
            except Exception as error:  # Optional integrations must never stop training.
                self._mark_degraded(name, operation, error)

    def _event(self, event_type: str, **fields: Any) -> MetricEvent:
        return {
            "type": event_type,
            "run_name": self.run_dir.name,
            "timestamp": utc_now(),
            **fields,
        }

    def _append_local_progress(self, event: MetricEvent) -> None:
        metrics = event["metrics"]
        progress_path = self.run_dir / "progress.csv"
        if not progress_path.exists():
            with progress_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=PROGRESS_HEADER).writeheader()
        row = {
            "timestamp": event["timestamp"],
            "timesteps": event["step"],
            "episodes": metrics.get("episodes", ""),
            "mean_reward": metrics.get("mean_reward", ""),
            "steps_per_second": metrics.get("steps_per_second", ""),
        }
        with progress_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=PROGRESS_HEADER).writerow(row)

    def _mark_degraded(self, name: str, operation: str, error: Exception) -> None:
        self._degraded[name] = f"{operation} failed: {error}"
        self._write_tracker_status()

    def _write_tracker_status(self) -> None:
        manifest_path = self.run_dir / "run.json"
        manifest = read_json(manifest_path)
        manifest["tracker_status"] = [
            {"name": name, "status": "degraded", "message": message}
            for name, message in sorted(self._degraded.items())
        ]
        atomic_write_json(manifest_path, manifest)

    def _wandb_start_options(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "project": spec.get("project"),
            "entity": spec.get("entity"),
            "mode": spec.get("mode"),
            "name": self.run_dir.name,
            "dir": str(self.run_dir),
            "config": {"run_name": self.run_dir.name},
        }
        for key in ("group", "tags"):
            if key in spec:
                options[key] = spec[key]
        return {key: value for key, value in options.items() if value is not None}

    @staticmethod
    def _wandb_log_artifact(wandb: Any, run: Any, event: MetricEvent) -> None:
        if run is None:
            return
        artifact = wandb.Artifact(event["name"], type="hexwars-artifact")
        artifact.add_file(event["path"])
        run.log_artifact(artifact)

    @staticmethod
    def _tracker_name(kind: str, index: int, spec: Mapping[str, Any]) -> str:
        if kind == "custom":
            return f"custom:{index}:{spec.get('adapter', index)}"
        return f"{kind}:{index}"


class SB3TrackingFacade:
    """Small SB3-facing bridge that needs only the logger shape, not an SB3 import."""

    def __init__(self, trackers: TrackerHub) -> None:
        self._trackers = trackers

    def log_from_logger(self, logger: Any, *, step: int) -> None:
        values = getattr(logger, "name_to_value", {})
        self._trackers.log_metrics(dict(values), step=step)

    def on_rollout_end(self, model: Any) -> None:
        """Callback-friendly entry point for a model exposing SB3's logger and timesteps."""
        self.log_from_logger(model.logger, step=int(model.num_timesteps))
