"""Train a HexWars tactical agent with MaskablePPO (sb3-contrib) against a scripted baseline.

Train two *different* agents by varying the opponent/seed, then duel them (see duel.py):
    python train_maskable_ppo.py --opponent greedy --seed 1 --out ppo_a --timesteps 200000
    python train_maskable_ppo.py --opponent random --seed 2 --out ppo_b --timesteps 200000
    python duel.py ppo_a.zip ppo_b.zip --out ../replays/ppo_a_vs_b.replay
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from hexwars_gym import HexWarsEnv

DEFAULT_DLL = "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"


def mask_fn(env):
    return env.action_masks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="hexwars_ppo")
    ap.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    args = ap.parse_args()

    server_cmd = ["dotnet", args.server]
    env = ActionMasker(HexWarsEnv(server_cmd, opponent=args.opponent, seat=args.seat, base_seed=args.seed),
                       mask_fn)

    model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps=512, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"done -> {args.out}.zip  (vs {args.opponent}, seed {args.seed})")
    env.close()


if __name__ == "__main__":
    main()
