"""Stateless policy server for the Unity bridge.

Unity owns the engine + rendering. Each AI turn it computes the observation + legal-action mask with the
SHARED codec (HexWars.Engine.Rl.TacticalCoding, same C# the models trained against), sends them here over
stdin, and this returns the model's action over stdout. So the model sees exactly what it saw at training
time, and Unity stays in charge of the game.

A seat spec is a legacy "ppo:PATH"/"dqn:PATH" string, a JSON controller object, a run path, or
"@controller.json". Explicit run specs are fixed unless their JSON mode is "live". Legacy directory
inputs remain live sources, while legacy .zip paths remain fixed. No source changes until an explicit
{"cmd":"reload"} re-resolves live seats. Inference runs on CPU on purpose: it's one tiny forward pass per
turn, faster than a GPU round-trip and it never contends with training for the GPU.

Protocol (one JSON object per line):
    spawn:  python policy_server.py --p0 ppo:runs/sp6_r1/checkpoints --p1 ppo:sp6base.zip
    ready:  -> {"ready": true, "model_seats": [0,1], "seats": {"0": {...}}}
    in:     {"seat": 0, "obs": [...float...], "mask": [...bool...]}   -> {"action": 123}
    in:     {"cmd": "reload"}   -> {"reloaded": [0], "seats": {"0": {...}}}
    in:     {"cmd": "close"}    -> exits

Greedy/Random seats are NOT served here — Unity drives those with its own C# agents.
"""
import argparse
import json
import os
import sys

import numpy as np

from ml_lab.controllers import (
    ControllerResolutionError,
    ControllerResolver,
    normalize_controller_spec,
    predict,
    validate_inference_input,
)

# So models that reference a custom feature extractor (hex_cnn.HexCNN) load no matter what cwd Unity
# spawns us from — SB3 imports the class by module path on load.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Seat:
    def __init__(self, spec):
        self.binding = ControllerResolver().bind(spec)
        if self.binding.resolved.model is None:
            raise ControllerResolutionError("policy_server only serves trained checkpoint or run controllers")

    @property
    def resolved(self):
        return self.binding.resolved

    def reload(self):
        """Reload an explicitly-live run only after a bridge reload command."""
        return self.binding.reload()

    def metadata(self):
        return self.resolved.metadata()


def seat_models(seats):
    """Array-shaped metadata for Unity's structured JSON DTO parser."""
    return [
        {"seat": index, **seat.metadata()}
        for index, seat in sorted(seats.items())
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=None, help="legacy spec, JSON controller, run:PATH, or @controller.json")
    ap.add_argument("--p1", default=None, help="legacy spec, JSON controller, run:PATH, or @controller.json")
    args = ap.parse_args()

    seats = {}
    for i, spec in ((0, args.p0), (1, args.p1)):
        if not spec:
            continue
        try:
            normalized = normalize_controller_spec(spec)
            if normalized.kind != "scripted":
                seats[i] = Seat(normalized)
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
        msg = json.loads(line)
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
            obs = np.asarray(msg["obs"], dtype=np.float32)
            mask = np.asarray(msg["mask"], dtype=bool)
            validate_inference_input(seat.resolved, obs, mask)
            assert seat.resolved.model is not None and seat.resolved.algorithm is not None
            print(json.dumps({"action": predict(seat.resolved.model, seat.resolved.algorithm, obs, mask)}), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
