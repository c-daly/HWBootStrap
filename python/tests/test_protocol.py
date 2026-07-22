from __future__ import annotations

import math

import pytest


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"obs": [0.0, "0.5"], "mask": [True, False]}, "observation"),
        ({"obs": [0.0, math.nan], "mask": [True, False]}, "finite"),
        ({"obs": [0.0, 1.1], "mask": [True, False]}, "range"),
        ({"obs": [0.0, 0.5], "mask": [1, 0]}, "boolean"),
        ({"obs": [0.0, 0.5], "mask": [False, False]}, "legal action"),
    ],
)
def test_view_payload_rejects_values_numpy_would_silently_coerce(payload, message) -> None:
    from ml_lab.protocol import validate_view_payload

    with pytest.raises(ValueError, match=message):
        validate_view_payload(payload, observation_size=2, action_size=2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward", "1.0", "reward"),
        ("reward", math.inf, "finite"),
        ("terminated", 1, "terminated"),
        ("truncated", 0, "truncated"),
    ],
)
def test_step_payload_rejects_coercible_scalar_types(field, value, message) -> None:
    from ml_lab.protocol import validate_step_payload

    payload = {
        "obs": [0.0, 0.5], "mask": [True, False], "reward": 0.0,
        "terminated": False, "truncated": False,
    }
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_step_payload(payload, observation_size=2, action_size=2)


def test_terminal_payload_may_have_no_legal_action() -> None:
    from ml_lab.protocol import validate_step_payload

    payload = {
        "obs": [0.0, 0.5], "mask": [False, False], "reward": 1.0,
        "terminated": True, "truncated": False,
    }
    observation, mask = validate_step_payload(payload, observation_size=2, action_size=2)
    assert observation.tolist() == [0.0, 0.5]
    assert mask.tolist() == [False, False]
