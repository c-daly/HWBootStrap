"""Gymnasium environment backed by the .NET HexWars TacticalEnv subprocess."""
import json
import re
import subprocess
from collections.abc import Mapping
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ml_lab.contracts import EnvironmentContract


_CONTRACT_VERSION = "tactical-v1"
_BOARD_INT_FIELDS = (
    "width", "height", "max_elevation", "max_steps", "zone_depth", "plains_weight", "forest_weight",
    "rough_weight", "water_weight", "starting_points", "generator_cost", "generator_output",
    "generator_health", "damage_floor", "dmg_high_ground_bonus", "range_high_ground_bonus", "round_cap",
    "design_fee", "actions_per_turn", "win_conditions", "capture_cost", "economy_win_threshold",
    "score_kills", "score_points", "score_army", "score_territory", "territory_income",
)
_BOARD_NUMBER_FIELDS = (
    "flat_chance", "bounty_rate", "deploy_cost_multiplier", "upkeep_factor", "capture_factor",
    "build_factor", "point_decay",
)
_BOARD_BOOL_FIELDS = (
    "biomes_enabled", "territory_mode", "claim_ends_turn", "build_anywhere", "generators_enabled",
    "fog_of_war",
)
_TERRAIN_FIELDS = ("move_cost", "concealment", "defense", "passable")
_REWARD_FIELDS = (
    "shape_scale", "step_penalty", "closing_weight", "draw_credit_weight", "points_weight",
    "terminal_win", "terminal_loss",
)


def _parse_contract(spaces_info: Mapping[str, Any]) -> EnvironmentContract:
    required = (
        "contract_version", "contract_hash", "environment_kind", "obs_len", "n_actions", "channels",
        "globals", "board", "roster", "contract_roster", "reward",
    )
    for field in required:
        if field not in spaces_info:
            raise ValueError(f"GymServer contract is missing required field {field!r}")

    version = spaces_info["contract_version"]
    contract_hash = spaces_info["contract_hash"]
    if version != _CONTRACT_VERSION:
        raise ValueError(f"GymServer contract_version must be {_CONTRACT_VERSION!r}")
    if not isinstance(contract_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
        raise ValueError("GymServer contract_hash must be a lowercase SHA-256 hex digest")

    environment_kind = spaces_info["environment_kind"]
    if environment_kind not in {"tactical", "duel"}:
        raise ValueError("GymServer environment_kind must be 'tactical' or 'duel'")
    if environment_kind != "tactical":
        raise ValueError("HexWarsEnv requires a tactical environment_kind")
    observation_size = _positive_int(spaces_info["obs_len"], "obs_len")
    action_size = _positive_int(spaces_info["n_actions"], "n_actions")
    channels = _positive_int(spaces_info["channels"], "channels")
    globals_count = _positive_int(spaces_info["globals"], "globals")
    board = _mapping(spaces_info["board"], "board")
    roster_count = _positive_int(spaces_info["roster"], "roster")
    roster = spaces_info["contract_roster"]
    reward = _mapping(spaces_info["reward"], "reward")

    _validate_board(board, environment_kind)
    _validate_reward(reward)
    _validate_roster(roster)
    if roster_count != len(roster):
        raise ValueError("GymServer contract roster count does not match contract_roster")

    board_width = _positive_int(board["width"], "board.width")
    board_height = _positive_int(board["height"], "board.height")
    if _positive_int(spaces_info["board_w"], "board_w") != board_width:
        raise ValueError("GymServer contract board.width does not match board_w")
    if _positive_int(spaces_info["board_h"], "board_h") != board_height:
        raise ValueError("GymServer contract board.height does not match board_h")
    if channels != 2 * len(roster) + 1:
        raise ValueError("GymServer contract channels does not match contract_roster")
    if observation_size != channels * board_width * board_height + globals_count:
        raise ValueError("GymServer contract obs_len does not match board geometry")
    if action_size != 1 + 3 * len(roster) * board_width * board_height:
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


def _validate_board(board: Mapping[str, Any], environment_kind: str) -> None:
    for field in _BOARD_INT_FIELDS:
        _integer(board.get(field), f"board.{field}")
    for field in _BOARD_NUMBER_FIELDS:
        _number(board.get(field), f"board.{field}")
    for field in _BOARD_BOOL_FIELDS:
        if not isinstance(board.get(field), bool):
            raise ValueError(f"GymServer contract board.{field} must be a boolean")
    if not isinstance(board.get("turn_policy"), str) or not board["turn_policy"]:
        raise ValueError("GymServer contract board.turn_policy must be a non-empty string")
    if board.get("environment_kind") != environment_kind:
        raise ValueError("GymServer contract board.environment_kind does not match environment_kind")
    for terrain_name in ("plains", "forest", "rough", "water"):
        terrain = _mapping(board.get(terrain_name), f"board.{terrain_name}")
        for field in _TERRAIN_FIELDS[:-1]:
            _integer(terrain.get(field), f"board.{terrain_name}.{field}")
        if not isinstance(terrain.get("passable"), bool):
            raise ValueError(f"GymServer contract board.{terrain_name}.passable must be a boolean")


def _validate_reward(reward: Mapping[str, Any]) -> None:
    for field in _REWARD_FIELDS:
        _number(reward.get(field), f"reward.{field}")


def _validate_roster(roster: Any) -> None:
    if not isinstance(roster, list) or not roster:
        raise ValueError("GymServer contract contract_roster must be a non-empty list")
    for entry in roster:
        if not isinstance(entry, str) or len(entry.split(",")) != 9:
            raise ValueError("GymServer contract contract_roster entries must contain nine integer stats")
        if any(not re.fullmatch(r"-?\d+", stat) for stat in entry.split(",")):
            raise ValueError("GymServer contract contract_roster entries must contain nine integer stats")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"GymServer contract {field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"GymServer contract {field} must be a positive integer")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GymServer contract {field} must be an integer")
    return value


def _number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GymServer contract {field} must be a number")
    return value


class HexWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, server_cmd: List[str], opponent: str = "greedy", seat: int = 0, base_seed: int = 0):
        super().__init__()
        cmd = list(server_cmd) + ["--opponent", opponent, "--seat", str(seat)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._next_seed = base_seed
        try:
            self.spaces_info = self._rpc({"cmd": "spaces"})
            self.contract = _parse_contract(self.spaces_info)
            self.n_actions = self.contract.action_size
            self.obs_len = self.contract.observation_size
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32)
            self._mask = np.ones(self.n_actions, dtype=bool)
        except BaseException:
            self._shutdown()
            raise

    def _rpc(self, msg: dict) -> dict:
        try:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("HexWars server closed unexpectedly")
            return json.loads(line)
        except BaseException:
            self._shutdown()
            raise

    def _shutdown(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception:
            pass

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        try:
            response = self._rpc({"cmd": "reset", "seed": int(seed)})
            self._mask = np.asarray(response["mask"], dtype=bool)
            return np.asarray(response["obs"], dtype=np.float32), {}
        except BaseException:
            self._shutdown()
            raise

    def step(self, action):
        try:
            response = self._rpc({"cmd": "step", "action": int(action)})
            self._mask = np.asarray(response["mask"], dtype=bool)
            return (np.asarray(response["obs"], dtype=np.float32), float(response["reward"]),
                    bool(response["terminated"]), bool(response["truncated"]), {})
        except BaseException:
            self._shutdown()
            raise

    def action_masks(self) -> np.ndarray:
        return self._mask

    def close(self):
        self._shutdown()
