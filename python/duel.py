"""Play two scripted or metadata-backed controllers and write a replay."""

import argparse
import json
import subprocess
from pathlib import Path

from hexwars_gym.env import parse_contract
from ml_lab.controllers import ControllerResolver, predict, validate_inference_input
from ml_lab.protocol import validate_json_object, validate_step_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = (
    PROJECT_ROOT / "engine" / "HexWars.GymServer" / "bin" / "Release" / "net8.0"
    / "HexWars.GymServer.dll"
)


def rpc(proc, message: dict) -> dict:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed unexpectedly")
    return dict(validate_json_object(json.loads(line), "GymServer response"))


def _close_process(proc) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", default="greedy", help="random|greedy|run:PATH|JSON|@controller.json")
    parser.add_argument("--p1", default="random", help="random|greedy|run:PATH|JSON|@controller.json")
    parser.add_argument("--server", default=str(DEFAULT_SERVER))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "replays" / "duel.replay"))
    parser.add_argument(
        "--environment", choices=["tactical-v1", "adaptive-v1"], default="tactical-v1"
    )
    args = parser.parse_args()

    proc = subprocess.Popen(
        ["dotnet", args.server, "--environment", args.environment],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        spaces = rpc(proc, {"cmd": "duel_spaces"})
        contract = parse_contract(
            spaces,
            environment=args.environment,
            required_kind="duel" if args.environment == "tactical-v1" else "adaptive_duel",
        )
        resolver = ControllerResolver(contract)
        bindings = {0: resolver.bind(args.p0), 1: resolver.bind(args.p1)}
        controllers = {seat: binding.resolved.server_controller for seat, binding in bindings.items()}
        state = rpc(
            proc,
            {"cmd": "duel_reset", "seed": args.seed, "p0": controllers[0], "p1": controllers[1]},
        )
        observation, mask = validate_step_payload(
            state, observation_size=contract.observation_size, action_size=contract.action_size
        )

        steps = 0
        while not state["terminated"] and not state["truncated"] and steps < 5000:
            resolved = bindings[state["seat"]].resolved
            if resolved.model is None:
                break
            validate_inference_input(resolved, observation, mask)
            assert resolved.algorithm is not None
            state = rpc(proc, {"cmd": "duel_step", "action": predict(
                resolved.model, resolved.algorithm, observation, mask
            )})
            observation, mask = validate_step_payload(
                state, observation_size=contract.observation_size, action_size=contract.action_size
            )
            steps += 1

        saved = rpc(proc, {"cmd": "duel_save", "path": args.out})
        print(
            f"duel finished in {steps} steps -> {saved.get('saved')}   "
            f"(p0={args.p0} vs p1={args.p1})"
        )
    finally:
        _close_process(proc)


if __name__ == "__main__":
    main()
