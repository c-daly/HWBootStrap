"""Stateless policy server for the Unity bridge.

Unity owns the engine + rendering. Each AI turn it computes the observation + legal-action mask with the
SHARED codec (HexWars.Engine.Rl.TacticalCoding, same C# the models trained against), sends them here over
stdin, and this returns the model's action over stdout. So the model sees exactly what it saw at training
time, and Unity stays in charge of the game.

Protocol (one JSON object per line):
    spawn:  python policy_server.py --p0 ppo:sp6_r4.zip --p1 ppo:sp6base.zip   (only model seats load)
    ready:  -> {"ready": true, "model_seats": [0,1]}
    in:     {"seat": 0, "obs": [...float...], "mask": [...bool...]}
    out:    {"action": 123}
    in:     {"cmd": "close"}   -> exits

Greedy/Random seats are NOT served here — Unity drives those with its own C# agents; this process only
loads the trained models (ppo:PATH / dqn:PATH). Runs in the Windows venv (CUDA torch), launched by Unity
as a local subprocess, so it's all localhost — no WSL networking.
"""
import argparse
import json
import sys

import numpy as np


def load_model(spec):
    """spec = 'ppo:PATH' or 'dqn:PATH' -> (kind, model). device='auto' uses the GPU when present."""
    kind, _, path = spec.partition(":")
    if kind == "ppo":
        from sb3_contrib import MaskablePPO
        return "ppo", MaskablePPO.load(path, device="auto")
    if kind == "dqn":
        from stable_baselines3 import DQN
        return "dqn", DQN.load(path, device="auto")
    raise ValueError(f"unknown model spec '{spec}' (expected ppo:PATH or dqn:PATH)")


def predict(kind, model, obs, mask):
    if kind == "ppo":
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    # masked DQN: zero out illegal actions in the Q-values, then argmax
    import torch
    with torch.no_grad():
        q = model.q_net(torch.as_tensor(obs, dtype=torch.float32)
                         .unsqueeze(0).to(model.device)).cpu().numpy()[0]
    q[~mask] = -np.inf
    return int(q.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=None, help="model spec for seat 0 (ppo:PATH / dqn:PATH), else omit")
    ap.add_argument("--p1", default=None, help="model spec for seat 1")
    args = ap.parse_args()

    models = {}
    for seat, spec in ((0, args.p0), (1, args.p1)):
        if spec and spec.split(":", 1)[0] in ("ppo", "dqn"):
            models[seat] = load_model(spec)

    print(json.dumps({"ready": True, "model_seats": sorted(models.keys())}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("cmd") == "close":
            break
        seat = int(msg["seat"])
        kind, model = models[seat]
        obs = np.asarray(msg["obs"], dtype=np.float32)
        mask = np.asarray(msg["mask"], dtype=bool)
        print(json.dumps({"action": predict(kind, model, obs, mask)}), flush=True)


if __name__ == "__main__":
    main()
