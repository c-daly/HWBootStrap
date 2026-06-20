"""Single-agent Gymnasium view of a 2-player game for self-play training.

The learner controls one seat; the other seat is played by a *frozen* model whose moves this wrapper
supplies automatically over the server's duel channel. SB3 only ever sees the learner's decision
points, and each step's reward (from the learner's perspective) sums the learner's move plus the
opponent's reply. Use with sb3-contrib MaskablePPO (exposes action_masks()).
"""
import json
import random
import subprocess

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from duel import predict  # shared model -> legal action (handles MaskablePPO + DQN, device-aware)


class SelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, server_cmd, opponent_models, learner_seat: int = 0, base_seed: int = 0):
        super().__init__()
        self.proc = subprocess.Popen(list(server_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        # a pool of frozen opponents; one is picked at random each episode (reduces self-play cycling)
        self.opp_pool = list(opponent_models) if isinstance(opponent_models, (list, tuple)) else [opponent_models]
        self.opp = self.opp_pool[0]
        self.learner = learner_seat
        self.opp_seat = 1 - learner_seat
        self._next_seed = base_seed

        sp = self._rpc({"cmd": "duel_spaces"})
        self.n_actions = int(sp["n_actions"])
        self.obs_len = int(sp["obs_len"])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32)
        self._mask = np.ones(self.n_actions, dtype=bool)

    def _rpc(self, msg: dict) -> dict:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed unexpectedly")
        return json.loads(line)

    def _play_opponent(self, v):
        """Play the frozen opponent until it's the learner's turn (or terminal). Returns (view, reward)."""
        acc = 0.0
        while not v["terminated"] and not v["truncated"] and int(v["seat"]) == self.opp_seat:
            a = predict(self.opp, np.asarray(v["obs"], dtype=np.float32), np.asarray(v["mask"], dtype=bool))
            v = self._rpc({"cmd": "duel_step", "action": int(a)})
            acc += float(v["reward"])
        return v, acc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        self.opp = random.choice(self.opp_pool)  # sample a frozen opponent from the pool for this episode
        v = self._rpc({"cmd": "duel_reset", "seed": int(seed), "learner": self.learner})
        v, _ = self._play_opponent(v)  # in case the opponent moves first
        self._mask = np.asarray(v["mask"], dtype=bool)
        return np.asarray(v["obs"], dtype=np.float32), {}

    def step(self, action):
        v = self._rpc({"cmd": "duel_step", "action": int(action)})
        reward = float(v["reward"])
        v, acc = self._play_opponent(v)
        reward += acc
        self._mask = np.asarray(v["mask"], dtype=bool)
        return (np.asarray(v["obs"], dtype=np.float32), reward,
                bool(v["terminated"]), bool(v["truncated"]), {})

    def action_masks(self):
        return self._mask

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
