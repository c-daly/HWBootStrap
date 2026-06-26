"""Stateless policy server for the Unity bridge.

Unity owns the engine + rendering. Each AI turn it computes the observation + legal-action mask with the
SHARED codec (HexWars.Engine.Rl.TacticalCoding, same C# the models trained against), sends them here over
stdin, and this returns the model's action over stdout. So the model sees exactly what it saw at training
time, and Unity stays in charge of the game.

A seat spec is "ppo:PATH" or "dqn:PATH" where PATH is either a .zip OR a DIRECTORY — for a directory we
load the NEWEST .zip in it, and a {"cmd":"reload"} re-scans (so the live-training viewer can pick up fresh
checkpoints between games). Inference runs on CPU on purpose: it's one tiny forward pass per turn, faster
than a GPU round-trip and it never contends with training for the GPU.

Protocol (one JSON object per line):
    spawn:  python policy_server.py --p0 ppo:runs/sp6_r1/checkpoints --p1 ppo:sp6base.zip
    ready:  -> {"ready": true, "model_seats": [0,1]}
    in:     {"seat": 0, "obs": [...float...], "mask": [...bool...]}   -> {"action": 123}
    in:     {"cmd": "reload"}   -> {"reloaded": [0]}   (seats whose newest checkpoint changed)
    in:     {"cmd": "close"}    -> exits

Greedy/Random seats are NOT served here — Unity drives those with its own C# agents.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# So models that reference a custom feature extractor (hex_cnn.HexCNN) load no matter what cwd Unity
# spawns us from — SB3 imports the class by module path on load.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def newest_zip(path):
    """The model .zip to load. A file path is used as-is. A directory resolves to the newest .zip in it,
    or — if it has none — the newest .zip in its checkpoints/ subfolder, so you can point at a run dir
    (e.g. runs/sp9base) and get its latest checkpoint. None if nothing is found."""
    if not os.path.isdir(path):
        return path
    zips = glob.glob(os.path.join(path, "*.zip")) or glob.glob(os.path.join(path, "checkpoints", "*.zip"))
    return max(zips, key=os.path.getmtime) if zips else None


def load(kind, file):
    if kind == "ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(file, device="cpu")
    if kind == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(file, device="cpu")
    raise ValueError(f"unknown model kind '{kind}' (expected ppo/dqn)")


def predict(kind, model, obs, mask):
    if kind == "ppo":
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    import torch
    with torch.no_grad():
        q = model.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
    q[~mask] = -np.inf
    return int(q.argmax())


class Seat:
    def __init__(self, spec):
        self.kind, _, self.path = spec.partition(":")  # path may be a file or a directory
        self.loaded_from = None
        self.model = None
        self.reload()
        if self.model is None:
            sys.exit(f"policy_server: no model .zip found for '{spec}' (looked in '{self.path}' and its "
                     f"checkpoints/ subfolder). Point the seat at a model .zip or a run/checkpoints dir.")

    def reload(self):
        """Load the newest checkpoint if it changed. Returns True if a (re)load happened."""
        file = newest_zip(self.path)
        if file is None or file == self.loaded_from:
            return False
        self.model = load(self.kind, file)
        self.loaded_from = file
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default=None, help="ppo:PATH / dqn:PATH (PATH = .zip or checkpoint dir)")
    ap.add_argument("--p1", default=None)
    args = ap.parse_args()

    seats = {}
    for i, spec in ((0, args.p0), (1, args.p1)):
        if spec and spec.split(":", 1)[0] in ("ppo", "dqn"):
            seats[i] = Seat(spec)

    print(json.dumps({"ready": True, "model_seats": sorted(seats.keys())}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        cmd = msg.get("cmd")
        if cmd == "close":
            break
        if cmd == "reload":
            changed = [i for i, s in seats.items() if s.reload()]
            print(json.dumps({"reloaded": changed}), flush=True)
            continue
        seat = seats[int(msg["seat"])]
        obs = np.asarray(msg["obs"], dtype=np.float32)
        mask = np.asarray(msg["mask"], dtype=bool)
        print(json.dumps({"action": predict(seat.kind, seat.model, obs, mask)}), flush=True)


if __name__ == "__main__":
    main()
