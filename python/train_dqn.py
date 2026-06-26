"""EXPERIMENTAL: a masked DQN agent for HexWars (a value-based contrast to MaskablePPO).

SB3's DQN has no native action masking, and most of the ~379 actions are illegal each turn, so this
subclass masks BOTH exploration and exploitation using the env's action_masks(). The Q-target is left
unmasked (a known approximation) — enough to train a legal-playing value-based agent. PPO
(train_maskable_ppo.py) is the verified path; this is here so the two duelists can use *different*
algorithms. If an SB3 API detail differs in your version, this is the file to tweak.

    python train_dqn.py --opponent greedy --seed 3 --out dqn_a --timesteps 200000
    python duel.py --p0 ppo:ppo_a.zip --p1 dqn:dqn_a.zip
"""
import argparse
import os

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from hexwars_gym import HexWarsEnv

DEFAULT_DLL = "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"


class MaskableDQN(DQN):
    def _action_masks(self):
        # gymnasium 1.x: reach each base env's method through the Monitor wrapper
        return np.asarray([e.unwrapped.action_masks() for e in self.env.envs])

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        masks = self._action_masks()
        explore = self.num_timesteps < learning_starts or np.random.rand() < self.exploration_rate
        if explore:
            actions = np.array([np.random.choice(np.flatnonzero(m)) for m in masks])
        else:
            import torch
            with torch.no_grad():
                obs_t, _ = self.policy.obs_to_tensor(self._last_obs)
                q = self.q_net(obs_t).cpu().numpy()
            q[~masks] = -1e9
            actions = q.argmax(axis=1)
        return actions, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--out", default="hexwars_dqn")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    ap.add_argument("--checkpoint-freq", type=int, default=10_000, help="save a checkpoint every N steps (for the live viewer)")
    args = ap.parse_args()

    logdir = args.logdir or os.path.join("runs", args.out)
    os.makedirs(logdir, exist_ok=True)

    base = HexWarsEnv(["dotnet", args.server], opponent=args.opponent, seat=args.seat, base_seed=args.seed)
    env = Monitor(base, filename=os.path.join(logdir, "monitor.csv"))

    # buffer_size capped so the replay buffer (761-dim obs) doesn't balloon to multiple GB
    model = MaskableDQN("MlpPolicy", env, seed=args.seed, verbose=1,
                        buffer_size=100_000, learning_starts=1_000)
    model.set_logger(configure(logdir, ["stdout", "csv"]))
    ckpt = CheckpointCallback(save_freq=args.checkpoint_freq,
                              save_path=os.path.join(logdir, "checkpoints"), name_prefix=args.out)
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save(args.out)
    print(f"done -> {args.out}.zip  (EXPERIMENTAL masked DQN, vs {args.opponent})  logs: {logdir}/")
    env.close()


if __name__ == "__main__":
    main()
