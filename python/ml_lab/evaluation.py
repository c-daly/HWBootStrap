"""Deterministic reciprocal evaluation and editor-only candidate publication."""

from __future__ import annotations

import os
import shutil
import json
import csv
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from statistics import NormalDist
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .contracts import EnvironmentContract, utc_now, validate_run_name
from .controllers import (
    ControllerResolver,
    ResolvedController,
    _validate_contract_compatibility,
    normalize_controller_spec,
    predict,
    validate_inference_input,
)
from hexwars_gym.env import parse_contract
from .contracts import ADAPTIVE_MONITOR_HEADER
from .protocol import validate_json_object, validate_step_payload
from .io import atomic_write_json, read_json


DEFAULT_HELD_OUT_SEED = 1_000_000
MAX_DECISIONS_PER_GAME = 10_000


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> dict[str, float]:
    """Return a Wilson score interval for one W/L/D proportion."""
    if total <= 0:
        raise ValueError("confidence interval total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("confidence interval successes must be inside total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "confidence": confidence,
    }


def controller_identity(resolved: ResolvedController) -> dict[str, Any]:
    """Return stable evidence identifying exactly what participated."""
    identity = resolved.metadata()
    if resolved.spec.name is not None:
        identity["name"] = resolved.spec.name
    if resolved.path is not None:
        identity["path"] = str(resolved.path.resolve())
    if resolved.spec.kind == "run" and resolved.spec.path is not None:
        identity["source_run"] = str(resolved.spec.path.resolve())
    return identity


class DuelClient:
    """One reusable JSONL GymServer process for evaluation games."""

    def __init__(
        self, server_cmd: Sequence[str], *, environment: str = "tactical-v1"
    ) -> None:
        self.proc = subprocess.Popen(
            list(server_cmd) + ["--environment", environment],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            spaces = self._rpc({"cmd": "duel_spaces"})
            required_kind = (
                "duel" if environment in {"tactical-v1", "tactical-v2"} else "adaptive_duel"
            )
            self.contract = parse_contract(
                spaces, environment=environment, required_kind=required_kind
            )
        except BaseException:
            self.close()
            raise

    def _rpc(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("GymServer pipes are unavailable")
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("GymServer closed unexpectedly")
        return dict(validate_json_object(json.loads(line), "GymServer response"))

    def reset(self, *, seed: int, p0: str, p1: str) -> dict[str, Any]:
        response = self._rpc(
            {"cmd": "duel_reset", "seed": seed, "p0": p0, "p1": p1, "learner": 0}
        )
        validate_step_payload(
            response,
            observation_size=self.contract.observation_size,
            action_size=self.contract.action_size,
        )
        return response

    def step(self, action: int) -> dict[str, Any]:
        response = self._rpc({"cmd": "duel_step", "action": action})
        validate_step_payload(
            response,
            observation_size=self.contract.observation_size,
            action_size=self.contract.action_size,
        )
        return response

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


def _contract_from_spaces(spaces: dict[str, Any]) -> EnvironmentContract:
    required = (
        "contract_version",
        "contract_hash",
        "encoding_hash",
        "obs_len",
        "n_actions",
        "board",
        "contract_roster",
        "reward",
    )
    missing = [field for field in required if field not in spaces]
    if missing:
        raise ValueError(f"duel handshake is missing {', '.join(missing)}")
    return EnvironmentContract(
        version=str(spaces["contract_version"]),
        contract_hash=str(spaces["contract_hash"]),
        encoding_hash=str(spaces["encoding_hash"]),
        observation_size=int(spaces["obs_len"]),
        action_size=int(spaces["n_actions"]),
        board=dict(spaces["board"]),
        roster=list(spaces["contract_roster"]),
        reward=dict(spaces["reward"]),
        semantics=dict(spaces.get("adaptive", {})),
    )


def _validate_against_client(
    controller: ResolvedController, client: Any
) -> None:
    contract = getattr(client, "contract", None)
    if not isinstance(contract, EnvironmentContract) or controller.model is None:
        return
    _validate_contract_compatibility(controller.contract, contract)
    if controller.observation_size != contract.observation_size:
        raise ValueError("controller observation size does not match duel server")
    if controller.action_size != contract.action_size:
        raise ValueError("controller action size does not match duel server")


def _play_game(
    client: Any,
    seats: tuple[ResolvedController, ResolvedController],
    seed: int,
    predict_action: Callable[[Any, str, np.ndarray, np.ndarray], int],
    prediction_locks: dict[int, Lock],
) -> int:
    state = client.reset(
        seed=seed,
        p0=seats[0].server_controller,
        p1=seats[1].server_controller,
    )
    decisions = 0
    while not bool(state.get("terminated")) and not bool(state.get("truncated")):
        if decisions >= MAX_DECISIONS_PER_GAME:
            return -1
        seat = state.get("seat")
        if isinstance(seat, bool) or not isinstance(seat, int) or seat not in {0, 1}:
            raise RuntimeError("duel server returned an invalid acting seat")
        controller = seats[seat]
        if controller.model is None or controller.algorithm is None:
            raise RuntimeError("duel server surfaced a scripted seat for external action")
        observation = np.asarray(state.get("obs"), dtype=np.float32)
        mask = np.asarray(state.get("mask"), dtype=bool)
        validate_inference_input(controller, observation, mask)
        with prediction_locks[id(controller.model)]:
            action = int(
                predict_action(controller.model, controller.algorithm, observation, mask)
            )
        if action < 0 or action >= mask.size or not bool(mask[action]):
            raise RuntimeError("controller selected an action excluded by the action mask")
        state = client.step(action)
        decisions += 1
    winner = state.get("winner", -1)
    return winner if isinstance(winner, int) and not isinstance(winner, bool) and winner in {0, 1} else -1


def evaluate_matchup(
    candidate: ResolvedController,
    opponent: ResolvedController,
    *,
    games: int,
    seed_start: int = DEFAULT_HELD_OUT_SEED,
    both_seats: bool = True,
    workers: int = 1,
    client_factory: Callable[[int], Any],
    predict_action: Callable[[Any, str, np.ndarray, np.ndarray], int] = predict,
    output_path: Path | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a fixed controller identity on deterministic held-out seeds."""
    if games <= 0:
        raise ValueError("evaluation games must be positive")
    if workers <= 0:
        raise ValueError("evaluation workers must be positive")
    schedule = [
        (seed, candidate_seat)
        for seed in range(seed_start, seed_start + games)
        for candidate_seat in ((0, 1) if both_seats else (0,))
    ]
    prediction_locks = {
        id(controller.model): Lock()
        for controller in (candidate, opponent)
        if controller.model is not None
    }

    def run_partition(
        worker_index: int,
    ) -> list[tuple[int, dict[str, Any]]]:
        client = client_factory(worker_index)
        try:
            _validate_against_client(candidate, client)
            _validate_against_client(opponent, client)
            partition: list[tuple[int, dict[str, Any]]] = []
            for index in range(worker_index, len(schedule), workers):
                seed, candidate_seat = schedule[index]
                seats = (
                    (candidate, opponent)
                    if candidate_seat == 0
                    else (opponent, candidate)
                )
                winner = _play_game(
                    client, seats, seed, predict_action, prediction_locks
                )
                if winner == candidate_seat:
                    outcome = "win"
                elif winner in {0, 1}:
                    outcome = "loss"
                else:
                    outcome = "draw"
                partition.append(
                    (
                        index,
                        {
                            "seed": seed,
                            "candidate_seat": candidate_seat,
                            "winner": winner,
                            "outcome": outcome,
                        },
                    )
                )
            return partition
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_partition, index) for index in range(workers)]
        indexed_matches = [
            indexed
            for future in futures
            for indexed in future.result()
        ]
    matches = [match for _, match in sorted(indexed_matches, key=lambda item: item[0])]
    totals = {"wins": 0, "losses": 0, "draws": 0}
    seat_results = {
        "candidate_as_p0": {"wins": 0, "losses": 0, "draws": 0},
        "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 0},
    }
    for match in matches:
        counter = f"{match['outcome']}s" if match["outcome"] != "loss" else "losses"
        totals[counter] += 1
        seat_key = (
            "candidate_as_p0"
            if match["candidate_seat"] == 0
            else "candidate_as_p1"
        )
        seat_results[seat_key][counter] += 1
    total_games = len(matches)
    result = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "candidate": controller_identity(candidate),
        "opponent": controller_identity(opponent),
        "seed_start": seed_start,
        "seeds": list(range(seed_start, seed_start + games)),
        "reciprocal": both_seats,
        "games": total_games,
        **totals,
        "rates": {
            "win": totals["wins"] / total_games,
            "loss": totals["losses"] / total_games,
            "draw": totals["draws"] / total_games,
        },
        "confidence_intervals": {
            "win": wilson_interval(totals["wins"], total_games, confidence),
            "loss": wilson_interval(totals["losses"], total_games, confidence),
            "draw": wilson_interval(totals["draws"], total_games, confidence),
        },
        "seat_results": seat_results,
        "matches": matches,
    }
    if candidate.contract is not None and candidate.contract.version == "adaptive-v1":
        source_run = candidate.spec.path if candidate.spec.kind == "run" else None
        result.update(
            _adaptive_diagnostic_aggregates(Path(source_run) if source_run is not None else None)
        )
    if output_path is not None:
        atomic_write_json(Path(output_path), result)
    return result


def _adaptive_sidecars(run_dir: Path | None) -> list[Path]:
    if run_dir is None:
        return []
    workers = list(run_dir.glob("adaptive_episodes.worker_*.csv"))
    if workers:
        def worker_index(path: Path) -> int:
            try:
                return int(path.stem.rsplit("_", 1)[1])
            except ValueError as error:
                raise ValueError(f"adaptive worker sidecar has invalid name: {path.name}") from error
        return sorted(workers, key=worker_index)
    central = run_dir / "adaptive_episodes.csv"
    return [central] if central.is_file() else []


def _adaptive_diagnostic_aggregates(run_dir: Path | None) -> dict[str, float | int]:
    rows: list[dict[str, str]] = []
    for path in _adaptive_sidecars(run_dir):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ADAPTIVE_MONITOR_HEADER:
                raise ValueError("adaptive episode sidecar header is invalid")
            rows.extend(reader)
    count = len(rows)

    def total(name: str) -> int:
        try:
            return sum(int(row[name]) for row in rows)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"adaptive episode sidecar field {name!r} is invalid") from error

    completed = 0
    for row in rows:
        value = row.get("deployment_completed", "").strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("adaptive episode sidecar deployment_completed is invalid")
        completed += int(value == "true")
    return {
        "design_count": total("design_count"),
        "distinct_custom_templates_deployed": total("distinct_custom_templates_deployed"),
        "deployment_completion_rate": completed / count if count else 0.0,
        "invalid_sequences": total("invalid_sequences"),
        "average_pregame_decisions": total("pregame_decisions") / count if count else 0.0,
    }


def _default_output_path(raw: str) -> Path | None:
    try:
        spec = normalize_controller_spec(raw)
    except Exception:
        return None
    if spec.kind == "run" and spec.path is not None:
        return spec.path.resolve() / "evaluation.json"
    return None


def evaluate_controllers(
    p0: str,
    p1: str,
    *,
    games: int,
    seed_start: int = DEFAULT_HELD_OUT_SEED,
    both_seats: bool = True,
    workers: int = 1,
    server_cmd: Sequence[str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve any two supported controller specs and evaluate them headlessly."""
    resolver = ControllerResolver()
    candidate = resolver.resolve(p0)
    opponent = resolver.resolve(p1)
    _validate_contract_compatibility(candidate.contract, opponent.contract)
    environment = next(
        (
            controller.contract.version
            for controller in (candidate, opponent)
            if controller.contract is not None
        ),
        "tactical-v1",
    )
    destination = Path(output_path) if output_path is not None else _default_output_path(p0)
    return evaluate_matchup(
        candidate,
        opponent,
        games=games,
        seed_start=seed_start,
        both_seats=both_seats,
        workers=workers,
        client_factory=lambda _index: DuelClient(server_cmd, environment=environment),
        output_path=destination,
    )


def _validate_evaluation(evaluation: Any, checkpoint: Path) -> dict[str, Any]:
    if not isinstance(evaluation, dict) or not evaluation:
        raise ValueError("candidate publication requires evaluation evidence")
    games = evaluation.get("games")
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("candidate publication requires a completed evaluation")
    identity = evaluation.get("candidate")
    if not isinstance(identity, dict):
        raise ValueError("evaluation is missing candidate checkpoint identity")
    evaluated_path = identity.get("path")
    if not isinstance(evaluated_path, str) or Path(evaluated_path).resolve() != checkpoint:
        raise ValueError("evaluation evidence does not match the published checkpoint")
    return evaluation


def publish_candidate(
    run_dir: Path,
    name: str,
    *,
    resolver: ControllerResolver | None = None,
) -> Path:
    """Copy a run checkpoint into a named, lab-only candidate artifact.

    Candidate directories always remain under the source run. Nothing here writes
    into Unity Assets or any player-build input.
    """
    validate_run_name(name)
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run.json").is_file():
        raise FileNotFoundError(run_dir / "run.json")
    resolved = (resolver or ControllerResolver()).resolve(
        {"kind": "run", "path": str(run_dir), "mode": "fixed"}
    )
    checkpoint = Path(resolved.path).resolve() if resolved.path is not None else None
    if not resolved.promotable or checkpoint is None:
        raise ValueError("only a metadata-backed run checkpoint can become a candidate")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    evaluation = _validate_evaluation(
        read_json(run_dir / "evaluation.json"), checkpoint
    )

    candidates_root = run_dir / "candidates"
    candidates_root.mkdir(exist_ok=True)
    candidate_dir = candidates_root / name
    if candidate_dir.exists():
        raise FileExistsError(candidate_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=candidates_root))
    try:
        shutil.copyfile(checkpoint, staging / "model.zip")
        candidate = {
            "schema_version": 1,
            "name": name,
            "created_at": utc_now(),
            "publication_scope": "editor_lab_only",
            "player_build_published": False,
            "candidate_dir": str(candidate_dir),
            "model": "model.zip",
            "source_run": str(run_dir),
            "source_checkpoint": str(checkpoint),
            "checkpoint_identity": resolved.metadata(),
            "evaluation": evaluation,
        }
        atomic_write_json(staging / "candidate.json", candidate)
        os.replace(staging, candidate_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return candidate_dir
