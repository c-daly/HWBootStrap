"""Play any two controllers head-to-head and write a .replay you can watch in Unity.

Each seat is one of: random | greedy | legacy ppo:PATH/dqn:PATH | a JSON controller spec | @spec.json.
Scripted seats (random/greedy) are played by the server; model seats are driven here.

    python duel.py --p0 ppo:ppo_a.zip --p1 greedy --out ../replays/ppo_vs_greedy.replay
    python duel.py --p0 ppo:ppo_a.zip --p1 dqn:dqn_b.zip
    python duel.py --p0 greedy --p1 random          # baselines, server plays both

Then in Unity: HexWars -> Replay -> Open Replay File... -> pick the .replay.
"""
import argparse
import json
import subprocess

import numpy as np

from ml_lab.controllers import ControllerResolver, predict as predict_resolved, validate_inference_input


def rpc(proc, msg: dict) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed unexpectedly")
    return json.loads(line)


def load_controller(spec: str):
    """Backward-compatible boundary for callers which only need the server controller and model."""
    resolved = ControllerResolver().resolve(spec)
    return resolved.server_controller, resolved.model


def predict(model, obs, mask) -> int:
    """Compatibility helper for older callers that pass only a loaded model."""
    from sb3_contrib import MaskablePPO
    if isinstance(model, MaskablePPO):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    # value-based (DQN): mask illegal Q-values then take the argmax
    import torch
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None]).float().to(model.device)  # match model's device (CPU/GPU)
        q = model.q_net(obs_t).cpu().numpy()[0]
    q[~mask] = -1e9
    return int(np.argmax(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--p0",
        default="greedy",
        help="random|greedy|ppo:PATH|dqn:PATH|run:PATH|JSON|@controller.json",
    )
    ap.add_argument(
        "--p1",
        default="random",
        help="random|greedy|ppo:PATH|dqn:PATH|run:PATH|JSON|@controller.json",
    )
    ap.add_argument("--server",
                    default="dotnet ../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../replays/duel.replay")
    args = ap.parse_args()

    resolver = ControllerResolver()
    bindings = {0: resolver.bind(args.p0), 1: resolver.bind(args.p1)}
    controllers = {seat: binding.resolved.server_controller for seat, binding in bindings.items()}

    proc = subprocess.Popen(args.server.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    v = rpc(proc, {"cmd": "duel_reset", "seed": args.seed, "p0": controllers[0], "p1": controllers[1]})

    steps = 0
    while not v["terminated"] and not v["truncated"] and steps < 5000:
        resolved = bindings[int(v["seat"])].resolved
        if resolved.model is None:
            break  # scripted seats are auto-played by the server; nothing to supply
        obs = np.asarray(v["obs"], dtype=np.float32)
        mask = np.asarray(v["mask"], dtype=bool)
        validate_inference_input(resolved, obs, mask)
        assert resolved.algorithm is not None
        v = rpc(proc, {"cmd": "duel_step", "action": predict_resolved(resolved.model, resolved.algorithm, obs, mask)})
        steps += 1

    saved = rpc(proc, {"cmd": "duel_save", "path": args.out})
    # the server's "close" just exits (no reply), so fire-and-forget — don't wait for a response
    try:
        proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
        proc.stdin.flush()
    except Exception:
        pass
    print(f"duel finished in {steps} steps -> {saved.get('saved')}   (p0={args.p0} vs p1={args.p1})")


if __name__ == "__main__":
    main()
