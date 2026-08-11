"""Strict JSONL client for the tactical-v3 GymServer protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from dataclasses import dataclass
import json
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
