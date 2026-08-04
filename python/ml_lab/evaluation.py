"""Deterministic reciprocal evaluation and editor-only candidate publication."""

from __future__ import annotations

import os
import shutil
import json
import re
import csv
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

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
from hexwars_gym.env import (
    SUPPORTED_ENVIRONMENTS,
    no_window_creationflags,
    parse_contract,
)
from .contracts import ADAPTIVE_MONITOR_HEADER
from .draw_classification import (
    DrawClassification,
    EpisodeSummary,
    classify_draw,
    summarize_episode,
)
from .protocol import (
    validate_json_object,
    validate_replay_save_response,
    validate_step_payload,
    validate_trace_enable_response,
)
from .io import atomic_write_json, read_json
from .tactical_trace import EpisodeTrace


DEFAULT_HELD_OUT_SEED = 1_000_000
MAX_DECISIONS_PER_GAME = 10_000

EvidenceRetention = Literal["diagnostic", "all"]

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


def _validate_demonstration_command(
    raw: Any, *, seat: int, index: int
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"demonstration decision {index} command must be an object")
    required = {"Kind", "Issuer", "ActorId", "TargetId", "Q", "R"}
    if set(raw) != required:
        raise ValueError(f"demonstration decision {index} command fields are invalid")

    kind = raw["Kind"]
    issuer = raw["Issuer"]
    if not isinstance(kind, str) or kind not in {
        "end_turn", "move", "attack", "deploy",
    }:
        raise ValueError(f"demonstration decision {index} command kind is invalid")
    if type(issuer) is not int or issuer not in {0, 1} or issuer != seat:
        raise ValueError(f"demonstration decision {index} command issuer is invalid")

    nullable_fields = ("ActorId", "TargetId", "Q", "R")
    for field in nullable_fields:
        value = raw[field]
        if value is not None and type(value) is not int:
            raise ValueError(
                f"demonstration decision {index} command {field} is invalid"
            )

    actor = raw["ActorId"]
    target = raw["TargetId"]
    q = raw["Q"]
    r = raw["R"]
    shape_is_valid = (
        (kind == "end_turn" and actor is None and target is None and q is None and r is None)
        or (kind == "move" and actor is not None and target is None and q is not None and r is not None)
        or (kind == "attack" and actor is not None and target is not None and q is None and r is None)
        or (kind == "deploy" and actor is None and target is None and q is not None and r is not None)
    )
    if not shape_is_valid:
        raise ValueError(f"demonstration decision {index} command shape is invalid")
    return dict(raw)


def validate_demonstration_payload(
    payload: Any, contract: EnvironmentContract
) -> list[dict[str, Any]]:
    """Validate and return a version-1 tactical-v2 demonstration batch."""
    if not isinstance(payload, Mapping):
        raise ValueError("demonstration payload must be an object")
    if set(payload) != {"schema_version", "decisions"}:
        raise ValueError("demonstration payload fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported demonstration schema version")
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("demonstration decisions must be a list")

    required = {"Observation", "LegalMask", "Action", "Seat", "Command"}
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"demonstration decision {index} must be an object")
        if set(raw) != required:
            raise ValueError(f"demonstration decision {index} fields are invalid")

        observation = raw["Observation"]
        if not isinstance(observation, list) or len(observation) != contract.observation_size:
            raise ValueError(
                f"demonstration decision {index} observation length is invalid"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in observation
        ):
            raise ValueError(
                f"demonstration decision {index} observation values must be finite"
            )

        legal_mask = raw["LegalMask"]
        if (
            not isinstance(legal_mask, list)
            or len(legal_mask) != contract.action_size
            or any(type(value) is not bool for value in legal_mask)
        ):
            raise ValueError(
                f"demonstration decision {index} legal mask length or values are invalid"
            )
        action = raw["Action"]
        if type(action) is not int or action < 0 or action >= contract.action_size:
            raise ValueError(f"demonstration decision {index} action is invalid")
        if not legal_mask[action]:
            raise ValueError(f"demonstration decision {index} action is masked off")
        seat = raw["Seat"]
        if type(seat) is not int or seat not in {0, 1}:
            raise ValueError(f"demonstration decision {index} seat is invalid")

        decision = dict(raw)
        decision["Command"] = _validate_demonstration_command(
            raw["Command"], seat=seat, index=index
        )
        result.append(decision)
    return result


def _validate_dagger_command(
    raw: Any, *, seat: int, index: int, role: str
) -> dict[str, Any]:
    descriptor = f"DAgger decision {index} {role} command"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{descriptor} must be an object")
    required = {"Kind", "Issuer", "ActorId", "TargetId", "Q", "R"}
    if set(raw) != required:
        raise ValueError(f"{descriptor} fields are invalid")

    kind = raw["Kind"]
    issuer = raw["Issuer"]
    if not isinstance(kind, str) or kind not in {
        "end_turn", "move", "attack", "deploy",
    }:
        raise ValueError(f"{descriptor} kind is invalid")
    if type(issuer) is not int or issuer not in {0, 1} or issuer != seat:
        raise ValueError(f"{descriptor} issuer is invalid")

    for field in ("ActorId", "TargetId", "Q", "R"):
        value = raw[field]
        if value is not None and type(value) is not int:
            raise ValueError(f"{descriptor} {field} is invalid")

    actor = raw["ActorId"]
    target = raw["TargetId"]
    q = raw["Q"]
    r = raw["R"]
    shape_is_valid = (
        (kind == "end_turn" and actor is None and target is None and q is None and r is None)
        or (kind == "move" and actor is not None and target is None and q is not None and r is not None)
        or (kind == "attack" and actor is not None and target is not None and q is None and r is None)
        or (kind == "deploy" and actor is None and target is None and q is not None and r is not None)
    )
    if not shape_is_valid:
        raise ValueError(f"{descriptor} shape is invalid")
    return dict(raw)


def validate_dagger_payload(
    payload: Any, contract: EnvironmentContract
) -> list[dict[str, Any]]:
    """Validate and return a version-1 tactical-v2 selective-DAgger batch."""
    if not isinstance(payload, Mapping):
        raise ValueError("DAgger payload must be an object")
    if set(payload) != {"schema_version", "decisions"}:
        raise ValueError("DAgger payload fields are invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported DAgger schema version")
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("DAgger decisions must be a list")

    required = {
        "Observation",
        "LegalMask",
        "LearnerAction",
        "LearnerCommand",
        "TeacherAction",
        "TeacherCommand",
        "Reasons",
        "StateHash",
        "NormalizedAdvantage",
        "OpponentLivingUnitCount",
        "ProductiveLegalActionCount",
        "Seat",
        "Round",
        "DecisionIndex",
        "Disagreement",
        "OracleDepth",
        "OracleExpansionBudget",
        "OracleHeuristicIdentity",
        "OracleActualExpansionCount",
    }
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(decisions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"DAgger decision {index} must be an object")
        if set(raw) != required:
            raise ValueError(f"DAgger decision {index} fields are invalid")

        observation = raw["Observation"]
        if not isinstance(observation, list) or len(observation) != contract.observation_size:
            raise ValueError(f"DAgger decision {index} observation length is invalid")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in observation
        ):
            raise ValueError(
                f"DAgger decision {index} observation values must be finite"
            )

        legal_mask = raw["LegalMask"]
        if (
            not isinstance(legal_mask, list)
            or len(legal_mask) != contract.action_size
            or any(type(value) is not bool for value in legal_mask)
        ):
            raise ValueError(
                f"DAgger decision {index} legal mask length or values are invalid"
            )

        actions: dict[str, int] = {}
        for field, label in (
            ("LearnerAction", "learner action"),
            ("TeacherAction", "teacher action"),
        ):
            action = raw[field]
            if type(action) is not int or action < 0 or action >= contract.action_size:
                raise ValueError(f"DAgger decision {index} {label} is invalid")
            if not legal_mask[action]:
                raise ValueError(f"DAgger decision {index} {label} is masked off")
            actions[field] = action

        reasons = raw["Reasons"]
        if type(reasons) is not int or reasons <= 0 or reasons & ~15:
            raise ValueError(f"DAgger decision {index} reasons bitmask is invalid")

        state_hash = raw["StateHash"]
        if not isinstance(state_hash, str) or re.fullmatch(r"[0-9a-f]{64}", state_hash) is None:
            raise ValueError(f"DAgger decision {index} state hash is invalid")

        normalized_advantage = raw["NormalizedAdvantage"]
        if (
            isinstance(normalized_advantage, bool)
            or not isinstance(normalized_advantage, (int, float))
            or not isfinite(float(normalized_advantage))
        ):
            raise ValueError(
                f"DAgger decision {index} normalized advantage must be finite"
            )

        for field, label in (
            ("OpponentLivingUnitCount", "opponent living unit count"),
            ("ProductiveLegalActionCount", "productive legal action count"),
            ("Round", "round"),
            ("DecisionIndex", "decision index"),
        ):
            value = raw[field]
            if type(value) is not int or value < 0:
                raise ValueError(f"DAgger decision {index} {label} is invalid")

        seat = raw["Seat"]
        if type(seat) is not int or seat not in {0, 1}:
            raise ValueError(f"DAgger decision {index} seat is invalid")

        disagreement = raw["Disagreement"]
        if type(disagreement) is not bool or disagreement is not (
            actions["LearnerAction"] != actions["TeacherAction"]
        ):
            raise ValueError(f"DAgger decision {index} disagreement is invalid")

        oracle_depth = raw["OracleDepth"]
        if type(oracle_depth) is not int or oracle_depth < 1:
            raise ValueError(f"DAgger decision {index} oracle depth is invalid")
        oracle_budget = raw["OracleExpansionBudget"]
        if type(oracle_budget) is not int or oracle_budget < 1:
            raise ValueError(
                f"DAgger decision {index} oracle expansion budget is invalid"
            )
        if raw["OracleHeuristicIdentity"] != "material-plus-pursuit-v1":
            raise ValueError(
                f"DAgger decision {index} oracle heuristic identity is invalid"
            )
        actual_expansions = raw["OracleActualExpansionCount"]
        if (
            type(actual_expansions) is not int
            or actual_expansions < 0
            or actual_expansions > oracle_budget
        ):
            raise ValueError(
                f"DAgger decision {index} actual expansion count is invalid"
            )

        decision = dict(raw)
        decision["LearnerCommand"] = _validate_dagger_command(
            raw["LearnerCommand"], seat=seat, index=index, role="learner"
        )
        decision["TeacherCommand"] = _validate_dagger_command(
            raw["TeacherCommand"], seat=seat, index=index, role="teacher"
        )
        result.append(decision)
    return result



class DuelClient:
    """One reusable JSONL GymServer process for evaluation games."""

    def __init__(
        self, server_cmd: Sequence[str], *, environment: str = "tactical-v1"
    ) -> None:
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"unsupported environment {environment!r}")
        self.proc = subprocess.Popen(
            list(server_cmd) + ["--environment", environment],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=no_window_creationflags(),
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

    def reset(
        self,
        *,
        seed: int,
        p0: str,
        p1: str,
        start_profile: str | None = None,
        reference_seat: int | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "cmd": "duel_reset", "seed": seed, "p0": p0, "p1": p1, "learner": 0,
        }
        if start_profile is not None:
            if reference_seat not in {0, 1}:
                raise ValueError("reference_seat must be 0 or 1 for a forced start profile")
            declared = _declared_start_profiles(self.contract)
            if start_profile not in declared:
                raise ValueError(f"start profile {start_profile!r} is not declared by the duel contract")
            request.update(start_profile=start_profile, reference_seat=reference_seat)
        elif reference_seat is not None:
            raise ValueError("reference_seat requires a forced start profile")
        response = self._rpc(request)
        validate_step_payload(
            response,
            observation_size=self.contract.observation_size,
            action_size=self.contract.action_size,
        )
        if start_profile is not None:
            if response.get("start_profile") != start_profile:
                raise ValueError("duel reset did not return the forced start profile")
            if response.get("reference_seat") != reference_seat:
                raise ValueError("duel reset did not return the requested reference seat")
        return response

    def step(self, action: int) -> dict[str, Any]:
        response = self._rpc({"cmd": "duel_step", "action": action})
        validate_step_payload(
            response,
            observation_size=self.contract.observation_size,
            action_size=self.contract.action_size,
        )
        return response

    def enable_trace(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("duel trace enabled flag must be boolean")
        response = self._rpc({"cmd": "duel_trace_enable", "enabled": enabled})
        validate_trace_enable_response(response, expected=enabled)

    def drain_trace(self) -> EpisodeTrace:
        return EpisodeTrace.from_payload(self._rpc({"cmd": "duel_trace_drain"}))

    def enable_demonstrations(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("demonstration capture enabled flag must be boolean")
        response = self._rpc({"cmd": "duel_demo_enable", "enabled": enabled})
        if (
            set(response) != {"enabled"}
            or type(response.get("enabled")) is not bool
            or response["enabled"] is not enabled
        ):
            raise ValueError(
                "GymServer did not acknowledge demonstration capture"
            )

    def drain_demonstrations(self) -> list[dict[str, Any]]:
        payload = self._rpc({"cmd": "duel_demo_drain"})
        return validate_demonstration_payload(payload, self.contract)

    def configure_dagger(
        self,
        *,
        enabled: bool,
        depth: int,
        expansion_budget: int,
        use_heuristic: bool,
    ) -> None:
        if type(enabled) is not bool:
            raise ValueError("DAgger enabled flag must be boolean")
        if type(depth) is not int or depth < 1:
            raise ValueError("DAgger depth must be a positive integer")
        if type(expansion_budget) is not int or expansion_budget < 1:
            raise ValueError("DAgger expansion budget must be a positive integer")
        if type(use_heuristic) is not bool or not use_heuristic:
            raise ValueError("DAgger heuristic choice must be true")

        expected = {
            "enabled": enabled,
            "depth": depth,
            "expansion_budget": expansion_budget,
            "use_heuristic": use_heuristic,
        }
        response = self._rpc({"cmd": "duel_dagger_configure", **expected})
        if (
            set(response) != set(expected)
            or type(response.get("enabled")) is not bool
            or type(response.get("depth")) is not int
            or type(response.get("expansion_budget")) is not int
            or type(response.get("use_heuristic")) is not bool
            or response != expected
        ):
            raise ValueError("GymServer did not acknowledge DAgger configuration")

    def drain_dagger(self) -> list[dict[str, Any]]:
        payload = self._rpc({"cmd": "duel_dagger_drain"})
        return validate_dagger_payload(payload, self.contract)


    def save_replay(self, path: Path) -> Path:
        path = Path(path)
        response = self._rpc({"cmd": "duel_save", "path": str(path)})
        return validate_replay_save_response(response, expected=path)

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

def _declared_start_profiles(contract: EnvironmentContract) -> tuple[str, ...]:
    if contract.version != "tactical-v2":
        return ()
    raw_profiles = contract.semantics.get("start_profiles")
    if not isinstance(raw_profiles, list):
        return ()
    result: list[str] = []
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"duel contract start_profiles[{index}] must be an object")
        profile_id = raw_profile.get("id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"duel contract start_profiles[{index}].id is invalid")
        result.append(profile_id)

    return tuple(result)

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


@dataclass(frozen=True)
class PlayedGame:
    winner: int
    terminated: bool
    truncated: bool
    summary: dict[str, Any] | None
    classification: dict[str, Any] | None
    staged_trace_path: Path | None
    staged_replay_path: Path | None


def _play_game(
    client: Any,
    seats: tuple[ResolvedController, ResolvedController],
    seed: int,
    predict_action: Callable[[Any, str, np.ndarray, np.ndarray], int],
    prediction_locks: dict[int, Lock],
    *,
    candidate_seat: int,
    start_profile: str | None = None,
    reference_seat: int | None = None,
    capture_trace: bool = False,
    trace_path: Path | None = None,
    replay_path: Path | None = None,
) -> PlayedGame:
    if capture_trace:
        client.enable_trace(True)
    state = client.reset(
        seed=seed,
        p0=seats[0].server_controller,
        p1=seats[1].server_controller,
        **(
            {"start_profile": start_profile, "reference_seat": reference_seat}
            if start_profile is not None else {}
        ),
    )
    decisions = 0
    forced_truncation = False
    while not bool(state.get("terminated")) and not bool(state.get("truncated")):
        if decisions >= MAX_DECISIONS_PER_GAME:
            forced_truncation = True
            break
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
    winner = (
        winner
        if isinstance(winner, int)
        and not isinstance(winner, bool)
        and winner in {0, 1}
        else -1
    )
    terminated = bool(state.get("terminated"))
    truncated = bool(state.get("truncated")) or forced_truncation
    summary_payload = None
    classification_payload = None
    if capture_trace:
        trace = client.drain_trace()
        if not trace.transitions:
            raise RuntimeError("requested empty trace")
        summary_payload = _summary_payload(summarize_episode(trace, candidate_seat))
        if winner not in {0, 1}:
            classification_payload = _classification_payload(
                classify_draw(
                    trace,
                    candidate_seat=candidate_seat,
                    terminated=terminated,
                    truncated=truncated,
                    winner=None,
                )
            )
        if trace_path is not None:
            atomic_write_json(trace_path, trace.to_dict())
        if replay_path is not None:
            client.save_replay(replay_path)
    return PlayedGame(
        winner=winner,
        terminated=terminated,
        truncated=truncated,
        summary=summary_payload,
        classification=classification_payload,
        staged_trace_path=trace_path,
        staged_replay_path=replay_path,
    )


def _evidence_stem(index: int, seed: int, candidate_seat: int) -> str:
    return f"match-{index:06d}-seed-{seed}-candidate-seat-{candidate_seat}"


def _summary_payload(summary: EpisodeSummary) -> dict[str, Any]:
    return {
        "command_count": summary.command_count,
        "round_count": summary.round_count,
        "damage_by_seat": list(summary.damage_by_seat),
        "kills_by_seat": list(summary.kills_by_seat),
        "end_turns_by_seat": list(summary.end_turns_by_seat),
        "wasted_end_turns_by_seat": list(summary.wasted_end_turns_by_seat),
        "peak_normalized_advantage": summary.peak_normalized_advantage,
        "final_normalized_advantage": summary.final_normalized_advantage,
        "maximum_state_repetition": summary.maximum_state_repetition,
    }


def _classification_payload(
    classification: DrawClassification,
) -> dict[str, Any]:
    return {
        "primary": classification.primary.value,
        "flags": [flag.value for flag in classification.flags],
        "evidence": dict(classification.evidence),
    }


def _files_identical(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as first_file, second.open("rb") as second_file:
        while True:
            first_chunk = first_file.read(1024 * 1024)
            second_chunk = second_file.read(1024 * 1024)
            if first_chunk != second_chunk:
                return False
            if not first_chunk:
                return True


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with source.open("rb") as source_file, destination.open("xb") as target_file:
            created = True
            shutil.copyfileobj(source_file, target_file)
            target_file.flush()
            os.fsync(target_file.fileno())
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _copy_file_atomically_exclusive(source: Path, destination: Path) -> None:
    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        _copy_file_exclusive(source, temporary_path)
        os.link(temporary_path, destination)
        temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _rollback_artifacts(paths: Sequence[Path]) -> None:
    for path in reversed(paths):
        path.unlink(missing_ok=True)


def _publish_artifact_pair(
    staged_trace: Path,
    staged_replay: Path,
    trace_path: Path,
    replay_path: Path,
) -> tuple[Path, ...]:
    trace_exists = trace_path.exists()
    replay_exists = replay_path.exists()
    if trace_exists != replay_exists:
        raise FileExistsError(
            f"incomplete artifact pair: {trace_path} and {replay_path}"
        )
    if trace_exists:
        if _files_identical(staged_trace, trace_path) and _files_identical(
            staged_replay, replay_path
        ):
            return ()
        raise FileExistsError(
            f"artifact pair collision: {trace_path} and {replay_path}"
        )

    created: list[Path] = []
    try:
        _copy_file_atomically_exclusive(staged_trace, trace_path)
        created.append(trace_path)
        _copy_file_atomically_exclusive(staged_replay, replay_path)
        created.append(replay_path)
    except BaseException:
        _rollback_artifacts(created)
        raise
    return tuple(created)


def _publish_artifact_pairs(
    pairs: list[tuple[Path, Path, Path, Path]],
) -> tuple[Path, ...]:
    created: list[Path] = []
    try:
        for staged_trace, staged_replay, trace_path, replay_path in pairs:
            created.extend(
                _publish_artifact_pair(
                    staged_trace, staged_replay, trace_path, replay_path
                )
            )
    except BaseException:
        _rollback_artifacts(created)
        raise
    return tuple(created)


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
    start_profile: str | None = None,
    confidence: float = 0.95,
    evidence_dir: Path | None = None,
    capture_trace: bool = False,
    evidence_retention: EvidenceRetention = "diagnostic",
) -> dict[str, Any]:
    """Evaluate a fixed controller identity on deterministic held-out seeds."""
    if evidence_retention not in {"diagnostic", "all"}:
        raise ValueError("evidence_retention must be 'diagnostic' or 'all'")
    if evidence_retention == "all" and (not capture_trace or evidence_dir is None):
        raise ValueError(
            "evidence_retention='all' requires trace capture and an evidence directory"
        )
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
    evidence_root = Path(evidence_dir) if evidence_dir is not None else None
    staging_owner: Any | None = None
    staging_dir: Path | None = None
    if capture_trace and evidence_root is not None:
        evidence_root.mkdir(parents=True, exist_ok=True)
        staging_owner = tempfile.TemporaryDirectory(
            prefix=".evaluation-staging-",
            dir=evidence_root,
        )
        staging_dir = Path(staging_owner.name)

    def run_partition(
        worker_index: int,
    ) -> list[tuple[int, dict[str, Any], PlayedGame]]:
        client = client_factory(worker_index)
        try:
            _validate_against_client(candidate, client)
            _validate_against_client(opponent, client)
            if start_profile is not None and start_profile not in _declared_start_profiles(client.contract):
                raise ValueError(f"start profile {start_profile!r} is not declared by the duel contract")
            partition: list[tuple[int, dict[str, Any], PlayedGame]] = []
            for index in range(worker_index, len(schedule), workers):
                seed, candidate_seat = schedule[index]
                seats = (
                    (candidate, opponent)
                    if candidate_seat == 0
                    else (opponent, candidate)
                )
                stem = _evidence_stem(index, seed, candidate_seat)
                staged_trace = (
                    staging_dir / f"{stem}.json"
                    if staging_dir is not None
                    else None
                )
                staged_replay = (
                    staging_dir / f"{stem}.replay"
                    if staging_dir is not None
                    else None
                )
                played = _play_game(
                    client,
                    seats,
                    seed,
                    predict_action,
                    prediction_locks,
                    candidate_seat=candidate_seat,
                    start_profile=start_profile,
                    reference_seat=candidate_seat if start_profile is not None else None,
                    capture_trace=capture_trace,
                    trace_path=staged_trace,
                    replay_path=staged_replay,
                )
                if played.winner == candidate_seat:
                    outcome = "win"
                elif played.winner in {0, 1}:
                    outcome = "loss"
                else:
                    outcome = "draw"
                partition.append(
                    (
                        index,
                        {
                            "seed": seed,
                            "candidate_seat": candidate_seat,
                            "winner": played.winner,
                            "outcome": outcome,
                        },
                        played,
                    )
                )
                if start_profile is not None:
                    partition[-1][1].update({"start_profile": start_profile, "reference_seat": candidate_seat})
            return partition
        finally:
            client.close()

    evidence_summary: dict[str, Any] | None = None
    artifact_pairs: list[tuple[Path, Path, Path, Path]] = []
    published_artifacts: tuple[Path, ...] = ()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_partition, index)
                for index in range(workers)
            ]
            indexed_games = [
                indexed
                for future in futures
                for indexed in future.result()
            ]
        ordered_games = sorted(indexed_games, key=lambda item: item[0])
        matches: list[dict[str, Any]] = []
        draw_categories: Counter[str] = Counter()
        selected_controls: set[tuple[int, str]] = set()
        draw_trace_count = 0
        control_trace_count = 0
        for index, match, played in ordered_games:
            is_draw = match["outcome"] == "draw"
            retain = evidence_retention == "all" or is_draw
            if is_draw:
                draw_trace_count += 1
            elif evidence_retention == "all":
                control_trace_count += 1
            else:
                stratum = (match["candidate_seat"], match["outcome"])
                if stratum not in selected_controls:
                    selected_controls.add(stratum)
                    control_trace_count += 1
                    retain = True

            if capture_trace:
                if played.summary is None:
                    raise RuntimeError("requested duel trace summary is unavailable")
                classification_payload = played.classification
                if match["outcome"] == "draw":
                    if classification_payload is None:
                        raise RuntimeError(
                            "requested duel draw classification is unavailable"
                        )
                    draw_categories[
                        str(classification_payload["primary"])
                    ] += 1
                match.update(
                    {
                        "terminated": played.terminated,
                        "truncated": played.truncated,
                        "summary": played.summary,
                        "classification": classification_payload,
                    }
                )

                if evidence_root is not None:
                    match["trace_path"] = None
                    match["replay_path"] = None
                    seed, candidate_seat = schedule[index]
                    stem = _evidence_stem(index, seed, candidate_seat)
                    trace_path = evidence_root / "traces" / f"{stem}.json"
                    replay_path = evidence_root / "replays" / f"{stem}.replay"
                    if retain:
                        staged_trace = played.staged_trace_path
                        staged_replay = played.staged_replay_path
                        if staged_trace is None or not staged_trace.is_file():
                            raise RuntimeError(
                                "requested duel trace was not staged"
                            )
                        if staged_replay is None or not staged_replay.is_file():
                            raise RuntimeError(
                                "requested duel replay was not saved"
                            )
                        artifact_pairs.append(
                            (staged_trace, staged_replay, trace_path, replay_path)
                        )
                        match["trace_path"] = str(trace_path)
                        match["replay_path"] = str(replay_path)
            matches.append(match)


        if capture_trace:
            evidence_summary = {
                "retention": evidence_retention,
                "retained": draw_trace_count + control_trace_count,
                "draw_traces": draw_trace_count,
                "control_traces": control_trace_count,
                "draw_categories": dict(sorted(draw_categories.items())),
            }

        published_artifacts = _publish_artifact_pairs(artifact_pairs)
    finally:
        if staging_owner is not None:
            try:
                staging_owner.cleanup()
            except BaseException:
                _rollback_artifacts(published_artifacts)
                raise

    try:
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
            "schedule": {
                "start_profile": start_profile,
                "reference_seat_policy": "candidate-seat" if start_profile is not None else None,
            },
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
        if evidence_summary is not None:
            result["evidence"] = evidence_summary
        if candidate.contract is not None and candidate.contract.version == "adaptive-v1":
            source_run = candidate.spec.path if candidate.spec.kind == "run" else None
            result.update(
                _adaptive_diagnostic_aggregates(
                    Path(source_run) if source_run is not None else None
                )
            )
        if output_path is not None:
            atomic_write_json(Path(output_path), result)
        return result
    except BaseException:
        _rollback_artifacts(published_artifacts)
        raise


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
    environment: str | None = None,
    evidence_dir: Path | None = None,
    capture_trace: bool = False,
    start_profile: str | None = None,
    evidence_retention: EvidenceRetention = "diagnostic",
) -> dict[str, Any]:
    """Resolve any two supported controller specs and evaluate them headlessly."""
    if environment is not None and environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"unsupported environment {environment!r}")
    # ControllerResolver's default model_loader (ml_lab.controllers.load_model) always
    # loads checkpoints with device="cpu", mirroring policy_server's documented rule:
    # inference must never compete with training for the GPU.
    resolver = ControllerResolver()
    candidate = resolver.resolve(p0)
    opponent = resolver.resolve(p1)
    _validate_contract_compatibility(candidate.contract, opponent.contract)
    if environment is not None:
        for controller in (candidate, opponent):
            if (
                controller.model is not None
                and controller.contract is not None
                and controller.contract.version != environment
            ):
                raise ValueError(
                    "controller contract does not match the explicit environment"
                )
    inferred_environment = next(
        (
            controller.contract.version
            for controller in (candidate, opponent)
            if controller.contract is not None
        ),
        "tactical-v1",
    )
    selected_environment = environment or inferred_environment
    if capture_trace and selected_environment != "tactical-v2":
        raise ValueError("trace capture requires the tactical-v2 environment")
    destination = Path(output_path) if output_path is not None else _default_output_path(p0)
    return evaluate_matchup(
        candidate,
        opponent,
        games=games,
        seed_start=seed_start,
        both_seats=both_seats,
        workers=workers,
        client_factory=lambda _index: DuelClient(
            server_cmd,
            environment=selected_environment,
        ),
        output_path=destination,
        start_profile=start_profile,
        evidence_dir=evidence_dir,
        capture_trace=capture_trace,
        evidence_retention=evidence_retention,
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
