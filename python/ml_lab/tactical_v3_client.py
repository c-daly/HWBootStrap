"""Strict JSONL client for the tactical-v3 GymServer protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any, Literal

from hexwars_gym.env import no_window_creationflags
from .tactical_v3_schema import (
    TacticalV3SemanticIdentity,
    TacticalV3View,
    parse_spaces,
    parse_view,
)


_STDERR_TAIL_LIMIT = 8192
_REAP_TIMEOUT_SECONDS = 2
_REPLY_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    decision_id: int
    candidate_id: int


@dataclass(frozen=True, slots=True)
class TeacherSelection:
    decision_id: int
    candidate_id: int
    search_depth: int
    expansion_budget: int
    actual_expansions: int
    heuristic_identity: str


@dataclass(frozen=True, slots=True)
class OracleStepResult:
    selection: TeacherSelection
    view: TacticalV3View


@dataclass(frozen=True, slots=True)
class SelectiveDaggerInspection:
    decision_id: int
    learner_candidate_id: int
    reasons: tuple[str, ...]
    state_hash: str
    state_occurrence: int
    normalized_advantage: float
    opponent_living_unit_count: int
    productive_legal_action_count: int


class TacticalV3GymClient:
    """Own one tactical-v3 GymServer process and fail closed on wire drift."""

    def __init__(
        self,
        server_cmd: Sequence[str],
        *,
        environment_kind: Literal["tactical", "duel"],
    ) -> None:
        if environment_kind not in {"tactical", "duel"}:
            raise ValueError("environment_kind must be 'tactical' or 'duel'")
        if not server_cmd or not all(type(part) is str and part for part in server_cmd):
            raise ValueError("server_cmd must be a non-empty sequence of strings")
        self._environment_kind = environment_kind
        self._closed = False
        self._view: TacticalV3View | None = None
        self._stderr_tail = ""
        self._stdout_lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_chunks: deque[str] = deque()
        self.proc = subprocess.Popen(
            [*server_cmd, "--environment", "tactical-v3"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            creationflags=no_window_creationflags(),
        )
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._identity = parse_spaces(
                self._rpc({"cmd": "spaces" if environment_kind == "tactical" else "duel_spaces"})
            )
            if self._identity.environment_kind != environment_kind:
                raise ValueError("GymServer spaces environment_kind does not match client")
        except BaseException as error:
            if self._closed: raise
            self._raise_after_protocol_error(error)

    @property
    def identity(self) -> TacticalV3SemanticIdentity:
        return self._identity

    def reset(self, seed: int) -> TacticalV3View:
        self._require_kind("tactical")
        return self._parse_next_view(self._rpc({"cmd": "reset", "seed": self._int32(seed, "seed")}))

    def step(self, selection: CandidateSelection) -> TacticalV3View:
        self._require_kind("tactical")
        return self._step("step", selection)

    def duel_reset(
        self,
        seed: int,
        p0: str,
        p1: str,
        learner: int,
        start_profile: str,
        reference_seat: int,
    ) -> TacticalV3View:
        self._require_kind("duel")
        if type(p0) is not str or type(p1) is not str or type(start_profile) is not str:
            raise TypeError("duel controller and start profile values must be strings")
        if learner not in {0, 1} or type(learner) is not int:
            raise ValueError("learner must be 0 or 1")
        if reference_seat not in {0, 1} or type(reference_seat) is not int:
            raise ValueError("reference_seat must be 0 or 1")
        request = {
            "cmd": "duel_reset", "seed": self._int32(seed, "seed"), "p0": p0,
            "p1": p1, "learner": learner, "start_profile": start_profile,
            "reference_seat": reference_seat,
        }
        return self._parse_next_view(self._rpc(request))

    def duel_step(self, selection: CandidateSelection) -> TacticalV3View:
        self._require_kind("duel")
        return self._step("duel_step", selection)

    def duel_oracle_query(
        self,
        decision_id: int,
        *,
        search_depth: int = 4,
        expansion_budget: int = 512,
        heuristic_identity: str = "material-plus-pursuit-v1",
    ) -> TeacherSelection:
        (
            current, decision_id, search_depth, expansion_budget, heuristic_identity,
        ) = self._prepare_teacher_request(
            "duel_oracle_query", decision_id, search_depth,
            expansion_budget, heuristic_identity,
        )
        payload = self._rpc({
            "cmd": "duel_oracle_query",
            "decision_id": decision_id,
            "search_depth": search_depth,
            "expansion_budget": expansion_budget,
            "heuristic_identity": heuristic_identity,
        })
        if payload == {"error": "tactical-v3 decision id is stale"}:
            raise ValueError("tactical-v3 decision id is stale")
        try:
            if set(payload) != {"selection"}:
                raise ValueError("GymServer duel_oracle_query fields changed")
            return self._parse_teacher_selection(
                payload["selection"], current, decision_id, search_depth,
                expansion_budget, heuristic_identity,
            )
        except BaseException as error:
            self._raise_after_protocol_error(error)

    def duel_dagger_inspect(
        self, decision_id: int, learner_candidate_id: int,
    ) -> SelectiveDaggerInspection:
        self._require_kind("duel")
        decision_id = self._int64(decision_id, "decision_id")
        learner_candidate_id = self._int32(
            learner_candidate_id, "learner_candidate_id",
        )
        current = self._view
        if current is None:
            raise RuntimeError("duel_reset must precede duel_dagger_inspect")
        if current.terminated or current.truncated:
            raise RuntimeError("duel_dagger_inspect requires a nonterminal view")
        if decision_id != current.decision.decision_id:
            raise ValueError("decision_id is stale")
        if sum(candidate.candidate_id == learner_candidate_id
               for candidate in current.decision.candidates) != 1:
            raise ValueError("learner_candidate_id is not present exactly once")
        payload = self._rpc({
            "cmd": "duel_dagger_inspect", "decision_id": decision_id,
            "candidate_id": learner_candidate_id,
        })
        try:
            if set(payload) != {"inspection"}:
                raise ValueError("GymServer duel_dagger_inspect fields changed")
            raw = payload["inspection"]
            expected = {
                "decision_id", "learner_candidate_id", "reasons", "state_hash",
                "state_occurrence", "normalized_advantage",
                "opponent_living_unit_count", "productive_legal_action_count",
            }
            if type(raw) is not dict or set(raw) != expected:
                raise ValueError("GymServer selective DAgger inspection fields changed")
            parsed_decision = self._int64(raw["decision_id"], "inspection.decision_id")
            parsed_candidate = self._int32(
                raw["learner_candidate_id"], "inspection.learner_candidate_id",
            )
            reasons = raw["reasons"]
            allowed = ("conversion", "favorable", "cycle_warning", "wasted_end_turn")
            if type(reasons) is not list or any(type(reason) is not str for reason in reasons):
                raise TypeError("inspection.reasons must be a string array")
            parsed_reasons = tuple(reasons)
            if tuple(reason for reason in allowed if reason in parsed_reasons) != parsed_reasons:
                raise ValueError("inspection.reasons are unknown, duplicate, or unordered")
            state_hash = raw["state_hash"]
            if type(state_hash) is not str or len(state_hash) != 64 or any(
                character not in "0123456789abcdef" for character in state_hash
            ):
                raise ValueError("inspection.state_hash must be lowercase SHA-256")
            occurrence = self._int32(raw["state_occurrence"], "inspection.state_occurrence")
            advantage = raw["normalized_advantage"]
            if type(advantage) not in {int, float} or not math.isfinite(float(advantage)):
                raise TypeError("inspection.normalized_advantage must be finite")
            opponent = self._int32(
                raw["opponent_living_unit_count"],
                "inspection.opponent_living_unit_count",
            )
            productive = self._int32(
                raw["productive_legal_action_count"],
                "inspection.productive_legal_action_count",
            )
            if parsed_decision != decision_id or parsed_candidate != learner_candidate_id:
                raise ValueError("selective DAgger inspection identity changed")
            if occurrence < 1 or opponent < 0 or productive < 0:
                raise ValueError("selective DAgger inspection counts are out of range")
            return SelectiveDaggerInspection(
                parsed_decision, parsed_candidate, parsed_reasons, state_hash,
                occurrence, float(advantage), opponent, productive,
            )
        except BaseException as error:
            self._raise_after_protocol_error(error)

    def duel_oracle_step(
        self,
        decision_id: int,
        *,
        search_depth: int = 4,
        expansion_budget: int = 512,
        heuristic_identity: str = "material-plus-pursuit-v1",
    ) -> OracleStepResult:
        (
            current, decision_id, search_depth, expansion_budget, heuristic_identity,
        ) = self._prepare_teacher_request(
            "duel_oracle_step", decision_id, search_depth,
            expansion_budget, heuristic_identity,
        )
        payload = self._rpc({
            "cmd": "duel_oracle_step",
            "decision_id": decision_id,
            "search_depth": search_depth,
            "expansion_budget": expansion_budget,
            "heuristic_identity": heuristic_identity,
        })
        if payload == {"error": "tactical-v3 decision id is stale"}:
            raise ValueError("tactical-v3 decision id is stale")
        try:
            if set(payload) != {"selection", "view"}:
                raise ValueError("GymServer duel_oracle_step fields changed")
            selection = self._parse_teacher_selection(
                payload["selection"], current, decision_id, search_depth,
                expansion_budget, heuristic_identity,
            )
            view = parse_view(payload["view"], self._identity)
        except BaseException as error:
            self._raise_after_protocol_error(error)
        self._view = view
        return OracleStepResult(selection, view)

    def _prepare_teacher_request(
        self,
        command: str,
        decision_id: int,
        search_depth: int,
        expansion_budget: int,
        heuristic_identity: str,
    ) -> tuple[TacticalV3View, int, int, int, str]:
        self._require_kind("duel")
        decision_id = self._int64(decision_id, "decision_id")
        search_depth = self._int32(search_depth, "search_depth")
        expansion_budget = self._int32(expansion_budget, "expansion_budget")
        if type(heuristic_identity) is not str or not heuristic_identity:
            raise TypeError("heuristic_identity must be a non-empty string")
        if search_depth != 4 or expansion_budget != 512 or (
            heuristic_identity != "material-plus-pursuit-v1"
        ):
            raise ValueError("unsupported tactical-v3 teacher configuration")
        current = self._view
        if current is None:
            raise RuntimeError(f"duel_reset must precede {command}")
        if current.terminated or current.truncated:
            raise RuntimeError(f"{command} requires a nonterminal view")
        if decision_id != current.decision.decision_id:
            raise ValueError("decision_id is stale")
        return current, decision_id, search_depth, expansion_budget, heuristic_identity

    def _parse_teacher_selection(
        self,
        raw_selection: object,
        current: TacticalV3View,
        decision_id: int,
        search_depth: int,
        expansion_budget: int,
        heuristic_identity: str,
    ) -> TeacherSelection:
        if type(raw_selection) is not dict or set(raw_selection) != {
            "decision_id", "candidate_id", "search_depth", "expansion_budget",
            "actual_expansions", "heuristic_identity",
        }:
            raise ValueError("GymServer teacher selection fields changed")
        selected_decision = self._int64(
            raw_selection["decision_id"], "selection.decision_id"
        )
        candidate_id = self._int32(
            raw_selection["candidate_id"], "selection.candidate_id"
        )
        selected_depth = self._int32(
            raw_selection["search_depth"], "selection.search_depth"
        )
        selected_budget = self._int32(
            raw_selection["expansion_budget"], "selection.expansion_budget"
        )
        actual_expansions = self._int32(
            raw_selection["actual_expansions"], "selection.actual_expansions"
        )
        selected_heuristic = raw_selection["heuristic_identity"]
        if type(selected_heuristic) is not str or not selected_heuristic:
            raise TypeError("selection.heuristic_identity must be a non-empty string")
        if selected_decision != decision_id:
            raise ValueError("teacher selection decision_id does not match request")
        if selected_depth != search_depth or selected_budget != expansion_budget or (
            selected_heuristic != heuristic_identity
        ):
            raise ValueError("teacher selection configuration does not match request")
        if not 1 <= actual_expansions <= expansion_budget:
            raise ValueError("teacher selection actual_expansions is out of range")
        if sum(
            candidate.candidate_id == candidate_id
            for candidate in current.decision.candidates
        ) != 1:
            raise ValueError(
                "teacher selection candidate_id must occur exactly once in current decision"
            )
        return TeacherSelection(
            selected_decision, candidate_id, selected_depth, selected_budget,
            actual_expansions, selected_heuristic,
        )

    def duel_status(self) -> int:
        self._require_kind("duel")
        payload = self._rpc({"cmd": "duel_status"})
        if set(payload) != {"internal_fallback_count"}:
            self._raise_after_protocol_error(
                ValueError("GymServer duel_status fields changed")
            )
        try:
            count = self._int32(
                payload["internal_fallback_count"], "internal_fallback_count"
            )
            if count < 0:
                raise ValueError("internal_fallback_count must be nonnegative")
        except BaseException as error:
            self._raise_after_protocol_error(error)
        return count

    def save_replay(self, path: Path) -> Path:
        self._require_kind("duel")
        path = Path(path)
        response = self._rpc({"cmd": "duel_save", "path": str(path)})
        if set(response) != {"saved"} or type(response["saved"]) is not str:
            self._raise_after_protocol_error(ValueError("GymServer duel_save reply must be exactly {'saved': path}"))
        if response["saved"] != str(path):
            self._raise_after_protocol_error(ValueError("GymServer duel_save reply did not preserve path"))
        return path

    def close(self) -> None:
        self._shutdown(send_close=True)

    def __enter__(self) -> TacticalV3GymClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def _step(self, command: Literal["step", "duel_step"], selection: CandidateSelection) -> TacticalV3View:
        self._validate_selection(selection)
        return self._parse_next_view(self._rpc({
            "cmd": command,
            "decision_id": selection.decision_id,
            "candidate_id": selection.candidate_id,
        }))

    def _parse_next_view(self, payload: Mapping[str, object]) -> TacticalV3View:
        try:
            view = parse_view(payload, self._identity)
        except BaseException as error:
            self._raise_after_protocol_error(error)
        self._view = view
        return view

    def _validate_selection(self, selection: CandidateSelection) -> None:
        if type(selection) is not CandidateSelection:
            raise TypeError("selection must be CandidateSelection")
        decision_id = self._int64(selection.decision_id, "selection.decision_id")
        candidate_id = self._int32(selection.candidate_id, "selection.candidate_id")
        if self._view is None:
            raise RuntimeError("reset must precede step")
        if decision_id != self._view.decision.decision_id:
            raise ValueError("selection decision_id is stale")
        if candidate_id not in {candidate.candidate_id for candidate in self._view.decision.candidates}:
            raise ValueError("selection candidate_id is not present in the current decision")

    def _rpc(self, request: Mapping[str, object]) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("GymServer client is closed")
        if self.proc.stdin is None or self.proc.stdout is None:
            self._raise_after_protocol_error(RuntimeError("GymServer pipes are unavailable"))
        try:
            encoded = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            self.proc.stdin.write(encoded + "\n")
            self.proc.stdin.flush()
        except BaseException as error:
            self._raise_after_protocol_error(error)
        try:
            line = self._stdout_lines.get(timeout=_REPLY_TIMEOUT_SECONDS)
        except queue.Empty:
            self._raise_after_protocol_error(RuntimeError("GymServer reply timed out"))
        if not line:
            self._raise_after_protocol_error(RuntimeError("GymServer closed unexpectedly"))
        if not line.strip():
            self._raise_after_protocol_error(ValueError("GymServer reply must not be blank"))
        try:
            reply: Any = json.loads(line)
        except json.JSONDecodeError as error:
            self._raise_after_protocol_error(ValueError("GymServer reply is not valid JSON"), error)
        if type(reply) is not dict or not all(type(key) is str for key in reply):
            self._raise_after_protocol_error(ValueError("GymServer reply must be an object"))
        return reply

    def _raise_after_protocol_error(self, error: BaseException, cause: BaseException | None = None) -> None:
        self._shutdown(send_close=False)
        suffix = f"; GymServer stderr tail: {self._stderr_tail}" if self._stderr_tail else ""
        if isinstance(error, (ValueError, TypeError, RuntimeError)):
            raised = type(error)(str(error) + suffix)
        else:
            raised = RuntimeError(str(error) + suffix)
        raise raised from (cause or error)

    def _shutdown(self, *, send_close: bool) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        if send_close and proc.poll() is None and proc.stdin is not None:
            try:
                proc.stdin.write('{"cmd":"close"}\n')
                proc.stdin.flush()
            except OSError:
                pass
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=_REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_REAP_TIMEOUT_SECONDS)
        self._stdout_thread.join(timeout=_REAP_TIMEOUT_SECONDS)
        self._stderr_thread.join(timeout=_REAP_TIMEOUT_SECONDS)
        self._stderr_tail = "".join(self._stderr_chunks)[-_STDERR_TAIL_LIMIT:]

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._stdout_lines.put(line)
        self._stdout_lines.put(None)

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        while chunk := self.proc.stderr.read(1024):
            self._stderr_chunks.append(chunk)
            size = sum(map(len, self._stderr_chunks))
            while size > _STDERR_TAIL_LIMIT:
                size -= len(self._stderr_chunks.popleft())

    def _require_kind(self, expected: Literal["tactical", "duel"]) -> None:
        if self._environment_kind != expected:
            raise RuntimeError(f"{expected} operation requires a {expected} client")

    @staticmethod
    def _int32(value: object, name: str) -> int:
        if type(value) is not int or not -(2**31) <= value < 2**31:
            raise TypeError(f"{name} must be an int32")
        return value

    @staticmethod
    def _int64(value: object, name: str) -> int:
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise TypeError(f"{name} must be an int64")
        return value
