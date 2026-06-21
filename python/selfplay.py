"""Self-play training: train a learner against a frozen opponent policy, then (optionally) iterate —
each round retrains against the previous round's winner.

    # one round: learn to beat a frozen ppo_a
    python selfplay.py --opponent ppo:ppo_a.zip --out sp --timesteps 200000

    # iterated self-play: r0 vs ppo_a, r1 vs r0, r2 vs r1 ...
    python selfplay.py --opponent ppo:ppo_a.zip --out sp --rounds 3 --timesteps 200000

Then duel the final policy against anything: python duel.py --p0 ppo:sp_r2.zip --p1 ppo:ppo_a.zip
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from selfplay_env import SelfPlayEnv
from duel import load_controller
from hex_cnn import cnn_policy_kwargs
from experiment import write_params

DEFAULT_DLL = "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"


def mask_fn(env):
    return env.unwrapped.action_masks()


def load_opponent(spec):
    _, model = load_controller(spec)  # "ppo:PATH" / "dqn:PATH" -> (.., model)
    if model is None:
        raise ValueError("opponent must be a trained model: ppo:PATH or dqn:PATH")
    return model


def train_round(opp_specs, out, timesteps, server, seed, logdir):
    os.makedirs(logdir, exist_ok=True)
    # a pool; one sampled per episode. "greedy"/"random" stay as strings (server-side); rest are models.
    opponents = [s if s in ("greedy", "random") else load_opponent(s) for s in opp_specs]
    base = SelfPlayEnv(["dotnet", server], opponents, learner_seat=0, base_seed=seed)
    env = ActionMasker(Monitor(base, filename=os.path.join(logdir, "monitor.csv")), mask_fn)

    write_params(logdir, base.spaces_info,
                 dict(out=out, seed=seed, timesteps=timesteps, n_steps=512, policy="CNN", pool=list(opp_specs)))

    ckpt = CheckpointCallback(save_freq=25_000, save_path=os.path.join(logdir, "checkpoints"), name_prefix=out)
    model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps=512, seed=seed, verbose=1,
                        policy_kwargs=cnn_policy_kwargs(base.spaces_info))
    model.set_logger(configure(logdir, ["stdout", "csv"]))
    model.learn(total_timesteps=timesteps, callback=ckpt)
    model.save(out)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", required=True, help="frozen opponent to start from: ppo:PATH or dqn:PATH")
    ap.add_argument("--out", default="sp")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=1, help="self-play rounds (each retrains vs the last winner)")
    ap.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # greedy is anchored in the pool (decisive -> punishes passivity, stops the draw collapse); plus the
    # base model and every past self, added each round. ~half of episodes face greedy.
    pool = ["greedy", args.opponent]
    final = None
    for rnd in range(args.rounds):
        out = f"{args.out}_r{rnd}"
        print(f"=== self-play round {rnd}: train {out}.zip vs pool of {len(pool)} ({', '.join(pool)}) ===")
        train_round(pool, out, args.timesteps, args.server, args.seed + rnd, os.path.join("runs", out))
        final = out
        pool = pool + [f"ppo:{out}.zip"]  # add this round's policy to the opponent pool
    print(f"done -> {final}.zip  ({args.rounds} round(s))")


if __name__ == "__main__":
    main()
