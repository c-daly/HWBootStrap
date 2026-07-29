"""Gymnasium environment backed by the .NET HexWars TacticalEnv subprocess."""
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ml_lab.contracts import EnvironmentContract
from ml_lab.protocol import validate_json_object, validate_step_payload, validate_view_payload


def no_window_creationflags() -> int:
    """Return the subprocess creationflags that suppress a new console window.

    GymServer is a console-subsystem .NET app. Every process that spawns it here is
    itself windowless (launched by Unity/pythonw with no console of its own), so an
    unflagged Popen/run call allocates a brand-new console window that steals
    foreground focus from whatever the user is doing. Every production spawn of
    GymServer (or any other console-subsystem child) must pass this value as
    creationflags.
    """
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


SUPPORTED_ENVIRONMENTS = frozenset({"tactical-v1", "tactical-v2", "adaptive-v1"})
_TACTICAL_V2_STAT_KEYS = (
    "health", "damage", "defense", "movement", "vertical_movement",
    "range", "range_arc", "vision", "vision_arc",
)
_ADAPTIVE_PHASES = (
    "deployment_root", "deployment_template", "deployment_cell", "deployment_placed_unit",
    "deployment_move_cell", "gameplay_root", "gameplay_unit", "gameplay_unit_command",
    "gameplay_move_cell", "gameplay_attack_cell", "design_slot", "design_stat", "design_value",
    "design_confirm",
)
_ADAPTIVE_TEMPLATES = (
    ("Frontline", (7, 2, 3, 2, 2, 1, 1, 3, 1)),
    ("Assault", (3, 6, 0, 3, 2, 2, 1, 3, 1)),
    ("Marksman", (2, 3, 0, 2, 2, 6, 1, 5, 1)),
    ("Artillery", (3, 6, 0, 1, 1, 5, 2, 3, 1)),
    ("Recon", (2, 1, 0, 5, 3, 1, 0, 7, 2)),
    ("Support", (4, 3, 2, 3, 2, 3, 1, 4, 1)),
    ("Custom A", (4, 3, 1, 3, 2, 2, 1, 3, 1)),
    ("Custom B", (5, 2, 2, 2, 2, 3, 1, 3, 1)),
    ("Custom C", (3, 4, 1, 3, 2, 2, 1, 4, 1)),
)
_ADAPTIVE_STAT_VALUES = {
    "health": tuple(range(1, 9)),
    "damage": tuple(range(0, 9)),
    "defense": tuple(range(0, 9)),
    "movement": tuple(range(0, 7)),
    "vertical_movement": tuple(range(0, 5)),
    "range": tuple(range(0, 9)),
    "range_arc": tuple(range(0, 5)),
    "vision": tuple(range(0, 11)),
    "vision_arc": tuple(range(0, 5)),
}
_ADAPTIVE_FOG_RULE = (
    "hide_current_enemy_units_and_all_opponent_deployment_until_both_confirm;"
    "derive_action_masks_from_seat_visible_projection;"
    "authoritative_hidden_blocker_rejection_is_only_allowed_mask_rejection"
)
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


def parse_contract(
    spaces_info: Mapping[str, Any],
    *,
    environment: str = "tactical-v1",
    required_kind: str | None = None,
) -> EnvironmentContract:
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"unsupported environment {environment!r}")
    required = (
        "contract_version", "contract_hash", "encoding_hash", "environment_kind", "obs_len", "n_actions", "channels",
        "globals", "board", "roster", "contract_roster", "reward",
    )
    for field in required:
        if field not in spaces_info:
            raise ValueError(f"GymServer contract is missing required field {field!r}")

    version = spaces_info["contract_version"]
    contract_hash = spaces_info["contract_hash"]
    encoding_hash = spaces_info["encoding_hash"]
    if version != environment:
        raise ValueError(f"GymServer contract_version must be {environment!r}")
    if not isinstance(contract_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
        raise ValueError("GymServer contract_hash must be a lowercase SHA-256 hex digest")
    if not isinstance(encoding_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", encoding_hash):
        raise ValueError("GymServer encoding_hash must be a lowercase SHA-256 hex digest")

    environment_kind = spaces_info["environment_kind"]
    allowed_kinds = (
        {"tactical", "duel"}
        if version in {"tactical-v1", "tactical-v2"}
        else {"adaptive_tactical", "adaptive_duel"}
    )
    if environment_kind not in allowed_kinds:
        raise ValueError(f"GymServer environment_kind is invalid for {version}")
    if required_kind is not None and environment_kind != required_kind:
        raise ValueError(f"client requires environment_kind {required_kind!r}")
    observation_size = _positive_int(spaces_info["obs_len"], "obs_len")
    action_size = _positive_int(spaces_info["n_actions"], "n_actions")
    channels = _positive_int(spaces_info["channels"], "channels")
    globals_count = _positive_int(spaces_info["globals"], "globals")
    board = _mapping(spaces_info["board"], "board")
    roster_count = _positive_int(spaces_info["roster"], "roster")
    roster = spaces_info["contract_roster"]
    reward = _mapping(spaces_info["reward"], "reward")

    _validate_board(board, environment_kind)
    if version in {"tactical-v1", "tactical-v2"}:
        _validate_reward(reward)
    else:
        _validate_adaptive_reward(reward)
    if version == "tactical-v1":
        _validate_roster(roster)
    if roster_count != len(roster):
        raise ValueError("GymServer contract roster count does not match contract_roster")

    board_width = _positive_int(board["width"], "board.width")
    board_height = _positive_int(board["height"], "board.height")
    if _positive_int(spaces_info["board_w"], "board_w") != board_width:
        raise ValueError("GymServer contract board.width does not match board_w")
    if _positive_int(spaces_info["board_h"], "board_h") != board_height:
        raise ValueError("GymServer contract board.height does not match board_h")
    semantics: Mapping[str, Any] = {}
    if version == "tactical-v1":
        if channels != 2 * len(roster) + 1:
            raise ValueError("GymServer contract channels does not match contract_roster")
        if observation_size != channels * board_width * board_height + globals_count:
            raise ValueError("GymServer contract obs_len does not match board geometry")
        if action_size != 1 + 3 * len(roster) * board_width * board_height:
            raise ValueError("GymServer contract n_actions does not match board geometry")
    elif version == "tactical-v2":
        semantics = _validate_tactical_v2(
            spaces_info,
            board=board,
            roster=roster,
            observation_size=observation_size,
            action_size=action_size,
            channels=channels,
            globals_count=globals_count,
            environment_kind=environment_kind,
        )
    else:
        semantics = _validate_adaptive_v1(
            spaces_info,
            board=board,
            roster=roster,
            reward=reward,
            observation_size=observation_size,
            action_size=action_size,
            channels=channels,
            globals_count=globals_count,
            environment_kind=environment_kind,
        )

    return EnvironmentContract(
        version=version,
        contract_hash=contract_hash,
        encoding_hash=encoding_hash,
        observation_size=observation_size,
        action_size=action_size,
        board=dict(board),
        roster=list(roster),
        reward=dict(reward),
        semantics=dict(semantics),
    )


_parse_contract = parse_contract


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


def _validate_adaptive_reward(reward: Mapping[str, Any]) -> None:
    for field in (
        "intermediate_decision_penalty", "deployment_completion_bonus",
        "terminal_win", "terminal_loss",
    ):
        _number(reward.get(field), f"reward.{field}")
    expected = {
        "intermediate_decision_penalty": 0.001,
        "deployment_completion_bonus": 0.0,
        "terminal_win": 1.0,
        "terminal_loss": -1.0,
    }
    for field, value in expected.items():
        if reward.get(field) != value:
            raise ValueError(f"GymServer contract reward.{field} must be {value}")


def _validate_roster(roster: Any) -> None:
    if not isinstance(roster, list) or not roster:
        raise ValueError("GymServer contract contract_roster must be a non-empty list")
    for entry in roster:
        if not isinstance(entry, str) or len(entry.split(",")) != 9:
            raise ValueError("GymServer contract contract_roster entries must contain nine integer stats")
        if any(not re.fullmatch(r"-?\d+", stat) for stat in entry.split(",")):
            raise ValueError("GymServer contract contract_roster entries must contain nine integer stats")


def _validate_tactical_v2(
    spaces_info: Mapping[str, Any],
    *,
    board: Mapping[str, Any],
    roster: Any,
    observation_size: int,
    action_size: int,
    channels: int,
    globals_count: int,
    environment_kind: str,
) -> Mapping[str, Any]:
    if environment_kind not in {"tactical", "duel"}:
        raise ValueError("GymServer contract environment_kind is invalid for tactical-v2")
    semantics = _mapping(spaces_info.get("tactical_v2"), "tactical_v2")
    if semantics.get("contract_version") != "tactical-v2":
        raise ValueError("GymServer contract tactical_v2.contract_version must be 'tactical-v2'")
    if semantics.get("environment_kind") != environment_kind:
        raise ValueError("GymServer contract tactical_v2.environment_kind does not match environment_kind")

    starting_units = _positive_int(
        semantics.get("starting_unit_count"), "tactical_v2.starting_unit_count"
    )
    if not 1 <= starting_units <= 12:
        raise ValueError("GymServer contract tactical_v2.starting_unit_count must be between 1 and 12")
    max_controllable = _positive_int(
        semantics.get("max_controllable_units"), "tactical_v2.max_controllable_units"
    )
    if max_controllable != starting_units:
        raise ValueError(
            "GymServer contract tactical_v2.max_controllable_units must equal starting_unit_count"
        )
    if semantics.get("placement_policy") != "symmetric-random-v1":
        raise ValueError("GymServer contract tactical_v2.placement_policy is not canonical")

    templates = semantics.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError("GymServer contract tactical_v2.templates must be a non-empty list")
    expected_roster: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_template in enumerate(templates):
        template = _mapping(raw_template, f"tactical_v2.templates[{index}]")
        template_id = template.get("id")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"GymServer contract tactical_v2.templates[{index}].id must be a non-empty string")
        if template_id in seen_ids:
            raise ValueError(f"GymServer contract tactical_v2.templates[{index}].id is duplicated")
        seen_ids.add(template_id)
        name = template.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"GymServer contract tactical_v2.templates[{index}].name must be a non-empty string")
        stats = template.get("stats")
        if (
            not isinstance(stats, list)
            or len(stats) != len(_TACTICAL_V2_STAT_KEYS)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in stats)
        ):
            raise ValueError(
                f"GymServer contract tactical_v2.templates[{index}].stats must contain "
                f"{len(_TACTICAL_V2_STAT_KEYS)} integers"
            )
        _integer(template.get("cost"), f"tactical_v2.templates[{index}].cost")
        expected_roster.append(f"{template_id}:{name}:{','.join(str(value) for value in stats)}")
    if list(roster) != expected_roster:
        raise ValueError("GymServer contract contract_roster does not match tactical_v2 templates")

    template_count = len(templates)
    cell_count = _positive_int(board["width"], "board.width") * _positive_int(board["height"], "board.height")

    regions = _mapping(semantics.get("action_regions"), "tactical_v2.action_regions")
    if set(regions) != {"move", "attack", "deploy"}:
        raise ValueError("GymServer contract tactical_v2.action_regions must contain move, attack, and deploy")
    expected_regions = (
        ("move", 1, starting_units * cell_count),
        ("attack", 1 + starting_units * cell_count, starting_units * cell_count),
        ("deploy", 1 + 2 * starting_units * cell_count, template_count * cell_count),
    )
    for name, expected_offset, expected_count in expected_regions:
        region = _mapping(regions[name], f"tactical_v2.action_regions.{name}")
        if region.get("offset") != expected_offset or region.get("count") != expected_count:
            raise ValueError(f"GymServer contract tactical_v2.action_regions.{name} is invalid")
    if spaces_info.get("action_regions") != regions:
        raise ValueError("GymServer contract action_regions does not match tactical_v2.action_regions")
    expected_action_size = 1 + (2 * starting_units + template_count) * cell_count
    if action_size != expected_action_size:
        raise ValueError("GymServer contract n_actions does not match tactical-v2 geometry")

    expected_channels = [f"friendly_role_hp_{index}" for index in range(template_count)]
    expected_channels += [f"visible_enemy_role_hp_{index}" for index in range(template_count)]
    expected_channels.append("elevation")
    observation_channels = semantics.get("observation_channels")
    if observation_channels != expected_channels or spaces_info.get("observation_channels") != expected_channels:
        raise ValueError("GymServer contract tactical_v2 observation_channels are incomplete")
    if channels != len(expected_channels):
        raise ValueError("GymServer contract tactical_v2 observation geometry is invalid")
    if globals_count != 5:
        raise ValueError("GymServer contract tactical_v2 globals must be 5")
    expected_observation_size = channels * cell_count + globals_count
    if observation_size != expected_observation_size:
        raise ValueError("GymServer contract tactical_v2 obs_len does not match board geometry")

    if semantics.get("action_size") != action_size or semantics.get("observation_size") != observation_size:
        raise ValueError("GymServer contract tactical_v2 semantics geometry does not match n_actions/obs_len")
    if semantics.get("board") != board:
        raise ValueError("GymServer contract tactical_v2.board does not match board")
    return semantics


def _adaptive_channels() -> list[str]:
    return [
        "elevation", "terrain_plains", "terrain_forest", "terrain_rough", "terrain_water",
        "deployment_zone_self", "current_visibility", "previously_seen",
        *[f"friendly_role_hp_{index}" for index in range(9)],
        *[f"visible_enemy_role_hp_{index}" for index in range(9)],
        *[f"friendly_slot_occupancy_{index}" for index in range(24)],
    ]


def _validate_adaptive_v1(
    spaces_info: Mapping[str, Any],
    *,
    board: Mapping[str, Any],
    roster: Any,
    reward: Mapping[str, Any],
    observation_size: int,
    action_size: int,
    channels: int,
    globals_count: int,
    environment_kind: str,
) -> Mapping[str, Any]:
    semantics = _mapping(spaces_info.get("adaptive"), "adaptive")
    pinned_scalars = {
        "adaptive": True,
        "contract_version": "adaptive-v1",
        "environment_kind": environment_kind,
        "fixed_template_count": 6,
        "custom_template_count": 3,
        "max_controllable_units": 24,
        "starting_unit_count": 6,
        "starting_army_budget": 132,
        "max_design_point_cost": 24,
    }
    for name, expected in pinned_scalars.items():
        if semantics.get(name) != expected:
            raise ValueError(f"GymServer contract adaptive.{name} must be {expected!r}")
    if semantics.get("adaptive") is not True:
        raise ValueError("GymServer contract adaptive.adaptive must be true")
    for name in (
        "fixed_template_count", "custom_template_count", "max_controllable_units",
        "starting_unit_count", "starting_army_budget", "max_design_point_cost",
    ):
        _integer(semantics.get(name), f"adaptive.{name}")
    for name in ("intermediate_decision_penalty", "deployment_completion_bonus"):
        _number(semantics.get(name), f"adaptive.{name}")
    if semantics["intermediate_decision_penalty"] != 0.001:
        raise ValueError("GymServer contract adaptive.intermediate_decision_penalty must be 0.001")
    if semantics["deployment_completion_bonus"] != 0.0:
        raise ValueError("GymServer contract adaptive.deployment_completion_bonus must be 0")
    if semantics["intermediate_decision_penalty"] != reward["intermediate_decision_penalty"]:
        raise ValueError("GymServer contract adaptive penalty does not match reward")
    if semantics["deployment_completion_bonus"] != reward["deployment_completion_bonus"]:
        raise ValueError("GymServer contract adaptive.deployment_completion_bonus does not match reward")
    if _positive_int(semantics.get("effective_horizon"), "adaptive.effective_horizon") != board["max_steps"]:
        raise ValueError("GymServer contract adaptive effective_horizon does not match board")
    if semantics.get("fog_rule") != _ADAPTIVE_FOG_RULE:
        raise ValueError("GymServer contract adaptive.fog_rule is not canonical")
    if semantics.get("board") != board:
        raise ValueError("GymServer contract adaptive.board does not match board")

    templates = semantics.get("templates")
    if not isinstance(templates, list) or len(templates) != 9:
        raise ValueError("GymServer contract adaptive.templates must contain exactly 9 templates")
    expected_roster: list[str] = []
    for index, ((expected_name, expected_stats), raw_template) in enumerate(
        zip(_ADAPTIVE_TEMPLATES, templates, strict=True)
    ):
        template = _mapping(raw_template, f"adaptive.templates[{index}]")
        if template.get("slot") != index or template.get("name") != expected_name:
            raise ValueError("GymServer contract adaptive.templates slots or names are invalid")
        stats = template.get("stats")
        if not isinstance(stats, list) or tuple(stats) != expected_stats:
            raise ValueError("GymServer contract adaptive.templates stats do not match adaptive-v1")
        if template.get("fixed") is not (index < 6):
            raise ValueError("GymServer contract adaptive.templates fixed flags are invalid")
        cost = _integer(template.get("cost"), f"adaptive.templates[{index}].cost")
        if cost != sum(expected_stats):
            raise ValueError(f"GymServer contract adaptive.templates[{index}].cost is invalid")
        expected_roster.append(f"{expected_name}:{','.join(map(str, expected_stats))}")
    if roster != expected_roster:
        raise ValueError("GymServer contract contract_roster does not match adaptive templates")

    stat_values = _mapping(semantics.get("stat_values"), "adaptive.stat_values")
    if set(stat_values) != set(_ADAPTIVE_STAT_VALUES):
        raise ValueError("GymServer contract adaptive.stat_values must contain all nine catalogs")
    for name, expected_values in _ADAPTIVE_STAT_VALUES.items():
        if not isinstance(stat_values[name], list) or tuple(stat_values[name]) != expected_values:
            raise ValueError(f"GymServer contract adaptive.stat_values.{name} is invalid")

    phases = semantics.get("phases")
    if not isinstance(phases, list) or tuple(phases) != _ADAPTIVE_PHASES:
        raise ValueError("GymServer contract adaptive.phases must contain the 14 adaptive-v1 phases")
    if spaces_info.get("phases") != phases:
        raise ValueError("GymServer contract phases does not match adaptive.phases")

    cell_count = _positive_int(board["width"], "board.width") * _positive_int(board["height"], "board.height")
    expected_counts = (("command", 12), ("unit", 24), ("template", 9), ("cell", cell_count),
                       ("stat", 9), ("value", 11))
    regions = _mapping(semantics.get("action_regions"), "adaptive.action_regions")
    if set(regions) != {name for name, _ in expected_counts}:
        raise ValueError("GymServer contract adaptive.action_regions must contain all six regions")
    offset = 0
    for name, expected_count in expected_counts:
        region = _mapping(regions[name], f"adaptive.action_regions.{name}")
        if region.get("offset") != offset or region.get("count") != expected_count:
            raise ValueError(f"GymServer contract adaptive.action_regions.{name} is invalid")
        offset += expected_count
    if offset != action_size or semantics.get("action_size") != action_size:
        raise ValueError("GymServer contract adaptive action regions do not match n_actions")
    if spaces_info.get("action_regions") != regions:
        raise ValueError("GymServer contract action_regions does not match adaptive.action_regions")

    expected_channels = _adaptive_channels()
    observation_channels = semantics.get("observation_channels")
    if observation_channels != expected_channels or spaces_info.get("observation_channels") != expected_channels:
        raise ValueError("GymServer contract adaptive observation_channels are incomplete")
    if channels != len(expected_channels) or globals_count != 124:
        raise ValueError("GymServer contract adaptive observation geometry is invalid")
    if observation_size != channels * cell_count + globals_count:
        raise ValueError("GymServer contract adaptive obs_len does not match board geometry")
    if semantics.get("observation_size") != observation_size:
        raise ValueError("GymServer contract adaptive observation_size does not match obs_len")
    return semantics


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

    def __init__(
        self,
        server_cmd: List[str],
        opponent: str = "greedy",
        seat: int = 0,
        base_seed: int = 0,
        environment: str = "tactical-v1",
        scenario_path: Path | None = None,
    ):
        super().__init__()
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"unsupported environment {environment!r}")
        cmd = list(server_cmd) + [
            "--opponent", opponent, "--seat", str(seat), "--environment", environment,
        ]
        if scenario_path is not None:
            cmd.extend(["--scenario-file", str(scenario_path)])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=no_window_creationflags(),
        )
        self._next_seed = base_seed
        try:
            self.spaces_info = self._rpc({"cmd": "spaces"})
            expected_kind = (
                "tactical" if environment in {"tactical-v1", "tactical-v2"} else "adaptive_tactical"
            )
            self.contract = parse_contract(
                self.spaces_info, environment=environment, required_kind=expected_kind
            )
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
            return dict(validate_json_object(json.loads(line), "GymServer response"))
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
            observation, self._mask = validate_view_payload(
                response, observation_size=self.obs_len, action_size=self.n_actions
            )
            return observation, _response_info(response)
        except BaseException:
            self._shutdown()
            raise

    def step(self, action):
        try:
            response = self._rpc({"cmd": "step", "action": int(action)})
            observation, self._mask = validate_step_payload(
                response, observation_size=self.obs_len, action_size=self.n_actions
            )
            return (observation, float(response["reward"]),
                    bool(response["terminated"]), bool(response["truncated"]),
                    _response_info(response))
        except BaseException:
            self._shutdown()
            raise

    def action_masks(self) -> np.ndarray:
        return self._mask

    def close(self):
        self._shutdown()


def _response_info(response: Mapping[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    diagnostics = response.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        info["diagnostics"] = dict(diagnostics)
    if "deployment_complete" in response:
        info["deployment_complete"] = bool(response["deployment_complete"])
    start_profile = response.get("start_profile")
    if isinstance(start_profile, str) and start_profile:
        info["start_profile"] = start_profile
    return info


def _response_arrays(
    response: Mapping[str, Any], observation_size: int, action_size: int
) -> tuple[np.ndarray, np.ndarray]:
    return validate_view_payload(
        response, observation_size=observation_size, action_size=action_size
    )
