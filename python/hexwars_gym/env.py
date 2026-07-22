"""Gymnasium environment backed by the .NET HexWars TacticalEnv (run as a subprocess).

The opponent is played inside the server, so the agent only ever acts on its own turn. Communication
is one JSON object per line over the subprocess's stdin/stdout. Designed for sb3-contrib MaskablePPO:
`action_masks()` exposes the legal-action mask so illegal actions are never sampled.
"""
import json
import re
import subprocess
from collections.abc import Mapping
from typing import Any, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ml_lab.contracts import EnvironmentContract


def _parse_contract(spaces_info: Mapping[str, Any]) -> EnvironmentContract:
    required = (
        "contract_version",
        "contract_hash",
        "obs_len",
        "n_actions",
        "channels",
        "globals",
        "board",
        "roster",
        "contract_roster",
        "reward",
    )
    for field in required:
        if field not in spaces_info:
            raise ValueError(f"GymServer contract is missing required field {field!r}")

    version = spaces_info["contract_version"]
    contract_hash = spaces_info["contract_hash"]
    if not isinstance(version, str) or not version:
        raise ValueError("GymServer contract_version must be a non-empty string")
    if not isinstance(contract_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
        raise ValueError("GymServer contract_hash must be a lowercase SHA-256 hex digest")

    observation_size = _positive_int(spaces_info["obs_len"], "obs_len")
    action_size = _positive_int(spaces_info["n_actions"], "n_actions")
    channels = _positive_int(spaces_info["channels"], "channels")
    globals_count = _positive_int(spaces_info["globals"], "globals")
    board = spaces_info["board"]
    roster_count = _positive_int(spaces_info["roster"], "roster")
    roster = spaces_info["contract_roster"]
    reward = spaces_info["reward"]
    if not isinstance(board, Mapping):
        raise ValueError("GymServer contract board must be an object")
    if not isinstance(roster, list) or not all(isinstance(entry, str) and entry for entry in roster):
        raise ValueError("GymServer contract roster must be a non-empty list of stat strings")
    if not isinstance(reward, Mapping):
        raise ValueError("GymServer contract reward must be an object")
    if roster_count != len(roster):
        raise ValueError("GymServer contract roster count does not match contract_roster")

    board_width = _positive_int(board.get("width"), "board.width")
    board_height = _positive_int(board.get("height"), "board.height")
    if "board_w" in spaces_info and _positive_int(spaces_info["board_w"], "board_w") != board_width:
        raise ValueError("GymServer contract board.width does not match board_w")
    if "board_h" in spaces_info and _positive_int(spaces_info["board_h"], "board_h") != board_height:
        raise ValueError("GymServer contract board.height does not match board_h")
    if channels != 2 * len(roster) + 1:
        raise ValueError("GymServer contract channels does not match contract_roster")
    expected_observation_size = channels * board_width * board_height + globals_count
    if observation_size != expected_observation_size:
        raise ValueError("GymServer contract obs_len does not match board geometry")
    expected_action_size = 1 + 3 * len(roster) * board_width * board_height
    if action_size != expected_action_size:
        raise ValueError("GymServer contract n_actions does not match board geometry")

    return EnvironmentContract(
        version=version,
        contract_hash=contract_hash,
        observation_size=observation_size,
        action_size=action_size,
        board=dict(board),
        roster=list(roster),
        reward=dict(reward),
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"GymServer contract {field} must be a positive integer")
    return value


class HexWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, server_cmd: List[str], opponent: str = "greedy", seat: int = 0, base_seed: int = 0):
        super().__init__()
        cmd = list(server_cmd) + ["--opponent", opponent, "--seat", str(seat)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._next_seed = base_seed

        self.spaces_info = self._rpc({"cmd": "spaces"})  # full handshake: shapes + env config (for params)
        spaces_info = self.spaces_info
        self.contract = _parse_contract(spaces_info)
        self.n_actions = self.contract.action_size
        self.obs_len = self.contract.observation_size
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
