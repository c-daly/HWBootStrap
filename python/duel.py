"""Play any two controllers head-to-head and write a .replay you can watch in Unity.

Each seat is one of:  random | greedy | ppo:PATH | dqn:PATH
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


def rpc(proc, msg: dict) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed unexpectedly")
    return json.loads(line)


def load_controller(spec: str):
    """spec -> (server_controller_string, model_or_None)."""
    if spec in ("random", "greedy"):
        return spec, None
    if ":" not in spec:
        raise ValueError(f"bad controller '{spec}' (use random|greedy|ppo:PATH|dqn:PATH)")
    algo, path = spec.split(":", 1)
    if algo == "ppo":
        from sb3_contrib import MaskablePPO
        return "external", MaskablePPO.load(path)
    if algo == "dqn":
        from stable_baselines3 import DQN
        return "external", DQN.load(path)
    raise ValueError(f"unknown algo '{algo}' (use ppo or dqn)")


def predict(model, obs, mask) -> int:
    from sb3_contrib import MaskablePPO
    if isinstance(model, MaskablePPO):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    # value-based (DQN): mask illegal Q-values then take the argmax
    import torch
    with torch.no_grad():
        q = model.q_net(torch.as_tensor(obs[None]).float()).cpu().numpy()[0]
    q[~mask] = -1e9
    return int(np.argmax(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default="greedy", help="random|greedy|ppo:PATH|dqn:PATH")
    ap.add_argument("--p1", default="random", help="random|greedy|ppo:PATH|dqn:PATH")
    ap.add_argument("--server",
                    default="dotnet ../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../replays/duel.replay")
    args = ap.parse_args()

    c0, m0 = load_controller(args.p0)
    c1, m1 = load_controller(args.p1)
    models = {0: m0, 1: m1}

    proc = subprocess.Popen(args.server.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    v = rpc(proc, {"cmd": "duel_reset", "seed": args.seed, "p0": c0, "p1": c1})

    steps = 0
    while not v["terminated"] and not v["truncated"] and steps < 5000:
        model = models[int(v["seat"])]
        if model is None:
            break  # scripted seats are auto-played by the server; nothing to supply
        obs = np.asarray(v["obs"], dtype=np.float32)
        mask = np.asarray(v["mask"], dtype=bool)
        v = rpc(proc, {"cmd": "duel_step", "action": predict(model, obs, mask)})
        steps += 1

    saved = rpc(proc, {"cmd": "duel_save", "path": args.out})
    rpc(proc, {"cmd": "close"})
    print(f"duel finished in {steps} steps -> {saved.get('saved')}   (p0={args.p0} vs p1={args.p1})")


if __name__ == "__main__":
    main()
