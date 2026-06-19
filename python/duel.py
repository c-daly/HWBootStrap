"""Play two trained SB3 models head-to-head and write a .replay you can watch in Unity.

Both seats are driven by policies (no scripted opponent) via the server's duel mode. Run in WSL2:
    python duel.py path/to/model0.zip path/to/model1.zip --out ../replays/ppo_vs_ppo.replay
Then in Unity: HexWars -> Replay -> Open Replay File... -> pick the .replay.
"""
import argparse
import json
import subprocess

import numpy as np
from sb3_contrib import MaskablePPO


def rpc(proc, msg: dict) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed unexpectedly")
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model0", help="SB3 model for Player 1 (seat 0)")
    ap.add_argument("model1", help="SB3 model for Player 2 (seat 1)")
    ap.add_argument("--server",
                    default="dotnet ../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../replays/duel.replay")
    args = ap.parse_args()

    models = [MaskablePPO.load(args.model0), MaskablePPO.load(args.model1)]
    proc = subprocess.Popen(args.server.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)

    v = rpc(proc, {"cmd": "duel_reset", "seed": args.seed})
    steps = 0
    while not v["terminated"] and not v["truncated"] and steps < 5000:
        obs = np.asarray(v["obs"], dtype=np.float32)
        mask = np.asarray(v["mask"], dtype=bool)
        action, _ = models[int(v["seat"])].predict(obs, action_masks=mask, deterministic=True)
        v = rpc(proc, {"cmd": "duel_step", "action": int(action)})
        steps += 1

    saved = rpc(proc, {"cmd": "duel_save", "path": args.out})
    rpc(proc, {"cmd": "close"})
    print(f"duel finished in {steps} steps -> replay: {saved.get('saved')}")


if __name__ == "__main__":
    main()
