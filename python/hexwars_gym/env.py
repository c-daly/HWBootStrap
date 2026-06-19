"""Gymnasium environment backed by the .NET HexWars TacticalEnv (run as a subprocess).

The opponent is played inside the server, so the agent only ever acts on its own turn. Communication
is one JSON object per line over the subprocess's stdin/stdout. Designed for sb3-contrib MaskablePPO:
`action_masks()` exposes the legal-action mask so illegal actions are never sampled.
"""
import json
import subprocess
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HexWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, server_cmd: List[str], opponent: str = "greedy", seat: int = 0, base_seed: int = 0):
        super().__init__()
        cmd = list(server_cmd) + ["--opponent", opponent, "--seat", str(seat)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._next_seed = base_seed

        spaces_info = self._rpc({"cmd": "spaces"})
        self.n_actions = int(spaces_info["n_actions"])
        self.obs_len = int(spaces_info["obs_len"])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32)
        self._mask = np.ones(self.n_actions, dtype=bool)

    def _rpc(self, msg: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("HexWars server closed unexpectedly")
        return json.loads(line)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        r = self._rpc({"cmd": "reset", "seed": int(seed)})
        self._mask = np.asarray(r["mask"], dtype=bool)
        return np.asarray(r["obs"], dtype=np.float32), {}

    def step(self, action):
        r = self._rpc({"cmd": "step", "action": int(action)})
        self._mask = np.asarray(r["mask"], dtype=bool)
        obs = np.asarray(r["obs"], dtype=np.float32)
        return obs, float(r["reward"]), bool(r["terminated"]), bool(r["truncated"]), {}

    def action_masks(self) -> np.ndarray:
        return self._mask

    def close(self):
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
