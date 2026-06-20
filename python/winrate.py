"""Head-to-head win rate of two controllers over N seeds (each seed = a different board, deterministic).

    python winrate.py --p0 ppo:sp_r2.zip --p1 ppo:ppo_a.zip --games 30

Each game's winner is read from the terminal reward (computed from P0's perspective): >0.5 => P0 won,
<-0.5 => P1 won, else draw/timeout. Reuses one server process across all games.
"""
import argparse
import json
import subprocess

import numpy as np

from duel import load_controller, predict


def rpc(proc, msg: dict) -> dict:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server closed unexpectedly")
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True, help="ppo:PATH or dqn:PATH")
    ap.add_argument("--p1", required=True, help="ppo:PATH or dqn:PATH")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--server",
                    default="dotnet ../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll")
    args = ap.parse_args()

    _, m0 = load_controller(args.p0)
    _, m1 = load_controller(args.p1)
    if m0 is None or m1 is None:
        raise SystemExit("both --p0 and --p1 must be trained models (ppo:PATH / dqn:PATH)")
    models = {0: m0, 1: m1}

    proc = subprocess.Popen(args.server.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    p0w = p1w = draw = 0
    for seed in range(args.games):
        v = rpc(proc, {"cmd": "duel_reset", "seed": seed, "learner": 0})
        steps = 0
        while not v["terminated"] and not v["truncated"] and steps < 5000:
            seat = int(v["seat"])
            a = predict(models[seat], np.asarray(v["obs"], dtype=np.float32), np.asarray(v["mask"], dtype=bool))
            v = rpc(proc, {"cmd": "duel_step", "action": int(a)})
            steps += 1
        w = int(v.get("winner", -1)) if v["terminated"] else -1
        if w == 0:
            p0w += 1
        elif w == 1:
            p1w += 1
        else:
            draw += 1

    try:
        proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
        proc.stdin.flush()
    except Exception:
        pass

    n = args.games
    print(f"{args.p0} (P0) vs {args.p1} (P1) over {n} games:")
    print(f"  P0 wins        : {p0w}  ({p0w / n:.0%})")
    print(f"  P1 wins        : {p1w}  ({p1w / n:.0%})")
    print(f"  draws/timeouts : {draw}  ({draw / n:.0%})")


if __name__ == "__main__":
    main()
