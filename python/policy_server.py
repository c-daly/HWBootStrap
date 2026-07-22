"""Stateless policy server for the Unity bridge.

Unity owns the engine + rendering. Each AI turn it computes the observation + legal-action mask with the
SHARED codec (HexWars.Engine.Rl.TacticalCoding, same C# the models trained against), sends them here over
stdin, and this returns the model's action over stdout. So the model sees exactly what it saw at training
time, and Unity stays in charge of the game.

A seat spec is a metadata-backed run path, a JSON run-controller object, or
"@controller.json". Run specs are fixed unless their JSON mode is "live". Standalone checkpoints
are rejected because they lack authoritative contract metadata. No source changes until an explicit
{"cmd":"reload"} re-resolves live seats. Inference runs on CPU on purpose: it's one tiny forward pass per
turn, faster than a GPU round-trip and it never contends with training for the GPU.

Protocol (one JSON object per line):
    spawn:  python policy_server.py --p0 run:runs/sp6_r1 --p1 run:runs/sp6_baseline
    ready:  -> {"ready": true, "model_seats": [0,1], "seats": {"0": {...}}}
    in:     {"seat": 0, "obs": [...float...], "mask": [...bool...]}   -> {"action": 123}
    in:     {"cmd": "reload"}   -> {"reloaded": [0], "seats": {"0": {...}}}
    in:     {"cmd": "close"}    -> exits

Greedy/Random seats are NOT served here — Unity drives those with its own C# agents.
"""
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

from ml_lab.controllers import (
    ControllerResolutionError,
    ControllerResolver,
    normalize_controller_spec,
    predict,
    validate_inference_input,
)
from ml_lab.protocol import validate_json_object, validate_view_payload

# So models that reference a custom feature extractor (hex_cnn.HexCNN) load no matter what cwd Unity
# spawns us from — SB3 imports the class by module path on load.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class PolicyExpectation:
    environment: str
    version: str
    encoding_hash: str

    def __post_init__(self):
        if self.environment not in {"tactical-v1", "adaptive-v1"}:
            raise ValueError(f"unsupported expected environment {self.environment!r}")
        if self.version not in {"tactical-v1", "adaptive-v1"}:
            raise ValueError(f"unsupported expected contract version {self.version!r}")
        if self.environment != self.version:
            raise ValueError("expected environment must match expected contract version")
        if not re.fullmatch(r"[0-9a-f]{64}", self.encoding_hash):
            raise ValueError("expected encoding hash must be a lowercase SHA-256 hex digest")


def validate_resolved_contract(resolved, expected: PolicyExpectation) -> None:
    contract = resolved.contract
    if contract is None:
        raise ControllerResolutionError("model is missing contract metadata")
    if contract.environment != expected.environment:
        raise ControllerResolutionError(
            f"model environment {contract.environment!r} does not match expected {expected.environment!r}"
        )
    if contract.version != expected.version:
        raise ControllerResolutionError(
            f"model contract version {contract.version!r} does not match expected {expected.version!r}"
        )
    if contract.encoding_hash != expected.encoding_hash:
        raise ControllerResolutionError(
            f"model encoding hash {contract.encoding_hash} does not match expected {expected.encoding_hash}"
        )


class Seat:
    def __init__(self, spec, expectation: PolicyExpectation):
        self.binding = ControllerResolver().bind(spec)
        if self.binding.resolved.model is None:
            raise ControllerResolutionError("policy_server only serves trained checkpoint or run controllers")
        self.expectation = expectation
        validate_resolved_contract(self.binding.resolved, expectation)

    @property
    def resolved(self):
        return self.binding.resolved

    def reload(self):
        """Reload an explicitly-live run only after a bridge reload command."""
        return self.binding.reload(lambda candidate: validate_resolved_contract(candidate, self.expectation))

    def metadata(self):
        return self.resolved.metadata()


def seat_models(seats):
    """Array-shaped metadata for Unity's structured JSON DTO parser."""
    return [
        {"seat": index, **seat.metadata()}
        for index, seat in sorted(seats.items())
    ]


def predict_for_seat(seat, observation, mask) -> int:
    """Choose an action using the inference mode carried by the resolved spec."""
    resolved = seat.resolved
    assert resolved.model is not None and resolved.algorithm is not None
    return predict(
        resolved.model,
        resolved.algorithm,
        observation,
        mask,
        deterministic=resolved.spec.inference_mode == "deterministic",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=None, help="legacy spec, JSON controller, run:PATH, or @controller.json")
    ap.add_argument("--p1", default=None, help="legacy spec, JSON controller, run:PATH, or @controller.json")
    ap.add_argument("--expected-environment", default=None)
    ap.add_argument("--expected-contract-version", default=None)
    ap.add_argument("--expected-encoding-hash", default=None)
    args = ap.parse_args()

    expectation_values = (
        args.expected_environment,
        args.expected_contract_version,
        args.expected_encoding_hash,
    )
    if any(value is not None for value in expectation_values) and not all(
        value is not None for value in expectation_values
    ):
        sys.exit(
            "policy_server: --expected-environment, --expected-contract-version, and "
            "--expected-encoding-hash must be supplied together"
        )
    try:
        expectation = (
            PolicyExpectation(*expectation_values)
            if all(value is not None for value in expectation_values)
            else None
        )
    except ValueError as error:
        sys.exit(f"policy_server: {error}")

    seats = {}
    for i, spec in ((0, args.p0), (1, args.p1)):
        if not spec:
            continue
        try:
            normalized = normalize_controller_spec(spec)
            if normalized.kind != "scripted":
                if expectation is None:
                    raise ControllerResolutionError(
                        "model seats require --expected-environment, --expected-contract-version, "
                        "and --expected-encoding-hash"
                    )
                seats[i] = Seat(normalized, expectation)
        except ControllerResolutionError as error:
            sys.exit(f"policy_server: {error}")

    def seat_metadata():
        return {str(index): seat.metadata() for index, seat in seats.items()}

    print(json.dumps({
        "ready": True,
        "model_seats": sorted(seats.keys()),
        "seats": seat_metadata(),
        "seat_models": seat_models(seats),
    }), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = validate_json_object(json.loads(line), "policy request")
        cmd = msg.get("cmd")
        if cmd == "close":
            break
        if cmd == "reload":
            try:
                changed = [i for i, s in seats.items() if s.reload()]
                print(json.dumps({
                    "reloaded": changed,
                    "seats": seat_metadata(),
                    "seat_models": seat_models(seats),
                }), flush=True)
            except Exception as error:
                print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
            continue
        try:
            seat = seats[int(msg["seat"])]
            if seat.resolved.observation_size is None or seat.resolved.action_size is None:
                raise ControllerResolutionError("resolved model is missing inference geometry")
            obs, mask = validate_view_payload(
                msg,
                observation_size=seat.resolved.observation_size,
                action_size=seat.resolved.action_size,
            )
            validate_inference_input(seat.resolved, obs, mask)
            print(json.dumps({"action": predict_for_seat(seat, obs, mask)}), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
