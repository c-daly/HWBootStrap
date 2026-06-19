"""Train a HexWars tactical agent with MaskablePPO (sb3-contrib) against a scripted baseline.

Writes logs/results to disk under --logdir:
  - progress.csv : training curve (ep_rew_mean, loss, etc.) over time
  - monitor.csv  : per-episode reward + length
  - <out>.zip    : the trained model

Train two *different* agents, then duel them (see duel.py):
    python train_maskable_ppo.py --opponent greedy --seed 1 --out ppo_a --timesteps 200000
    python train_maskable_ppo.py --opponent random --seed 2 --out ppo_b --timesteps 200000
    python duel.py ppo_a.zip ppo_b.zip --out ../replays/ppo_a_vs_b.replay
"""
import argparse
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from hexwars_gym import HexWarsEnv

DEFAULT_DLL = "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"


def mask_fn(env):
    # gymnasium 1.x wrappers don't auto-forward attributes; reach the base env's method
    return env.unwrapped.action_masks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="hexwars_ppo")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    ap.add_argument("--resume", default=None, help="path to a saved model/checkpoint .zip to continue from")
    ap.add_argument("--checkpoint-freq", type=int, default=25_000, help="save a checkpoint every N steps")
    args = ap.parse_args()

    logdir = args.logdir or os.path.join("runs", args.out)
    os.makedirs(logdir, exist_ok=True)

    server_cmd = ["dotnet", args.server]
    base = HexWarsEnv(server_cmd, opponent=args.opponent, seat=args.seat, base_seed=args.seed)
    env = ActionMasker(Monitor(base, filename=os.path.join(logdir, "monitor.csv")), mask_fn)

    # periodic checkpoints so an interrupted run is resumable (--resume runs/.../checkpoints/...zip)
    ckpt = CheckpointCallback(save_freq=args.checkpoint_freq,
                              save_path=os.path.join(logdir, "checkpoints"), name_prefix=args.out)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=env)
        print(f"resuming from {args.resume}")
    else:
        model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps=512, seed=args.seed, verbose=1)

    model.set_logger(configure(logdir, ["stdout", "csv"]))  # add "tensorboard" if installed
    model.learn(total_timesteps=args.timesteps, callback=ckpt, reset_num_timesteps=(args.resume is None))
    model.save(args.out)
    print(f"done -> {args.out}.zip  (vs {args.opponent}, seed {args.seed})  logs: {logdir}/")
    env.close()


if __name__ == "__main__":
    main()
