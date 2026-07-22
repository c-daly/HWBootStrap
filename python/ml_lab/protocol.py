"""Strict validation for raw GymServer/policy JSON before NumPy coercion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def validate_json_object(value: Any, context: str = "protocol response") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _true_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"protocol {field} must be a boolean")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"protocol {field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"protocol {field} must be finite")
    return result


def _validate_diagnostics(value: Any) -> None:
    diagnostics = validate_json_object(value, "protocol diagnostics")
    for field in (
        "design_count", "distinct_custom_templates_deployed", "invalid_sequences",
        "pregame_decisions",
    ):
        item = diagnostics.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"protocol diagnostics.{field} must be a non-negative integer")
    _true_bool(diagnostics.get("deployment_completed"), "diagnostics.deployment_completed")


def validate_view_payload(
    payload: Mapping[str, Any],
    *,
    observation_size: int,
    action_size: int,
    terminal: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = validate_json_object(payload)
    raw_observation = payload.get("obs")
    if not isinstance(raw_observation, list) or len(raw_observation) != observation_size:
        raise ValueError("protocol observation must be a list matching the handshake size")
    observation_values: list[float] = []
    for value in raw_observation:
        number = _finite_number(value, "observation value")
        if number < 0.0 or number > 1.0:
            raise ValueError("protocol observation value is outside the [0, 1] range")
        observation_values.append(number)

    raw_mask = payload.get("mask")
    if not isinstance(raw_mask, list) or len(raw_mask) != action_size:
        raise ValueError("protocol mask must be a list matching the handshake size")
    if any(not isinstance(value, bool) for value in raw_mask):
        raise ValueError("protocol mask entries must be boolean")

    if terminal is None:
        terminated = payload.get("terminated", False)
        truncated = payload.get("truncated", False)
        if "terminated" in payload:
            terminated = _true_bool(terminated, "terminated")
        if "truncated" in payload:
            truncated = _true_bool(truncated, "truncated")
        terminal = bool(terminated or truncated)
    if not terminal and not any(raw_mask):
        raise ValueError("nonterminal protocol view must expose at least one legal action")

    if "deployment_complete" in payload:
        _true_bool(payload["deployment_complete"], "deployment_complete")
    if "diagnostics" in payload:
        _validate_diagnostics(payload["diagnostics"])
    if "seat" in payload:
        seat = payload["seat"]
        if isinstance(seat, bool) or not isinstance(seat, int) or seat not in {0, 1}:
            raise ValueError("protocol seat must be integer 0 or 1")
    if "winner" in payload:
        winner = payload["winner"]
        if isinstance(winner, bool) or not isinstance(winner, int) or winner not in {-1, 0, 1}:
            raise ValueError("protocol winner must be integer -1, 0, or 1")
    return np.asarray(observation_values, dtype=np.float32), np.asarray(raw_mask, dtype=bool)


def validate_step_payload(
    payload: Mapping[str, Any], *, observation_size: int, action_size: int
) -> tuple[np.ndarray, np.ndarray]:
    payload = validate_json_object(payload)
    _finite_number(payload.get("reward"), "reward")
    terminated = _true_bool(payload.get("terminated"), "terminated")
    truncated = _true_bool(payload.get("truncated"), "truncated")
    return validate_view_payload(
        payload,
        observation_size=observation_size,
        action_size=action_size,
        terminal=terminated or truncated,
    )
