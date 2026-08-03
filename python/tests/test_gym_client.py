import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hexwars_gym import HexWarsEnv
from hexwars_gym.env import parse_contract


def _fake_server(
    tmp_path: Path,
    spaces_response: dict,
    close_marker: Path | None = None,
    episode_response: dict | None = None,
) -> list[str]:
    server = tmp_path / "fake_gym_server.py"
    spaces_path = tmp_path / "spaces.json"
    episode_path = tmp_path / "episode.json"
    spaces_path.write_text(json.dumps(spaces_response), encoding="utf-8")
    episode_path.write_text(json.dumps(episode_response or {}), encoding="utf-8")
    server.write_text(
        """import json
import sys

response = json.load(open(sys.argv[1], encoding="utf-8"))
close_marker = sys.argv[2]
episode = json.load(open(sys.argv[3], encoding="utf-8"))
for line in sys.stdin:
    request = json.loads(line)
    if request[\"cmd\"] == \"spaces\":
        print(json.dumps(response), flush=True)
    elif request[\"cmd\"] == \"close\":
        if close_marker:
            open(close_marker, \"w\", encoding=\"utf-8\").write(\"closed\")
        break
    elif request[\"cmd\"] == \"reset\":
        print(json.dumps(episode), flush=True)
    elif request[\"cmd\"] == \"step\":
        print(json.dumps(episode), flush=True)
""",
        encoding="utf-8",
    )
    return [
        sys.executable,
        str(server),
        str(spaces_path),
        str(close_marker or ""),
        str(episode_path),
    ]


def _terrain() -> dict:
    return {"move_cost": 1, "concealment": 0, "defense": 0, "passable": True}


def _board() -> dict:
    return {
        "width": 13, "height": 9, "max_elevation": 4, "max_steps": 600, "zone_depth": 3,
        "flat_chance": 0.6, "plains_weight": 70, "forest_weight": 15, "rough_weight": 10,
        "water_weight": 5, "plains": _terrain(), "forest": _terrain(), "rough": _terrain(),
        "water": _terrain(), "biomes_enabled": False, "starting_points": 12, "bounty_rate": 0.5,
        "generator_cost": 2, "generator_output": 1, "generator_health": 3, "damage_floor": 0,
        "dmg_high_ground_bonus": 1, "range_high_ground_bonus": 1, "round_cap": 100,
        "design_fee": 0, "deploy_cost_multiplier": 1.0, "turn_policy": "HexWars.Engine.AllUnitsPolicy",
        "actions_per_turn": -1, "win_conditions": 1, "capture_cost": 3, "economy_win_threshold": 200,
        "score_kills": 1, "score_points": 1, "score_army": 1, "score_territory": 1,
        "upkeep_factor": 0.25, "capture_factor": 4.0, "build_factor": 4.0,
        "territory_mode": False, "claim_ends_turn": True, "build_anywhere": False,
        "territory_income": 0, "generators_enabled": True, "point_decay": 0.0,
        "fog_of_war": False, "environment_kind": "tactical",
    }


def _valid_spaces() -> dict:
    return {
        "contract_version": "tactical-v1", "contract_hash": "a" * 64,
        "encoding_hash": "b" * 64,
        "environment_kind": "tactical", "obs_len": 824, "n_actions": 1054,
        "channels": 7, "globals": 5, "board_h": 9, "board_w": 13, "board": _board(),
        "roster": 3,
        "contract_roster": [
            "5,3,2,3,2,1,1,2,1", "3,5,0,3,2,2,1,3,1", "2,2,0,4,3,1,0,5,2",
        ],
        "reward": {
            "shape_scale": 0.01, "step_penalty": 0.005, "closing_weight": 0.02,
            "draw_credit_weight": 0.25, "points_weight": 0.5,
            "terminal_win": 1.0, "terminal_loss": -1.0,
        },
    }


def tactical_v2_templates(count: int) -> list[dict]:
    stats = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    return [
        {"id": f"template-{index}", "name": f"Template {index}", "stats": list(stats), "cost": sum(stats)}
        for index in range(count)
    ]


def tactical_v2_spaces(
    *,
    template_count: int = 5,
    unit_count: int = 3,
    width: int = 13,
    height: int = 9,
    environment_kind: str = "tactical",
) -> dict:
    cells = width * height
    templates = tactical_v2_templates(template_count)
    action_regions = {
        "move": {"offset": 1, "count": unit_count * cells},
        "attack": {"offset": 1 + unit_count * cells, "count": unit_count * cells},
        "deploy": {"offset": 1 + 2 * unit_count * cells, "count": template_count * cells},
    }
    action_size = 1 + (2 * unit_count + template_count) * cells
    observation_channels = (
        [f"friendly_role_hp_{index}" for index in range(template_count)]
        + [f"visible_enemy_role_hp_{index}" for index in range(template_count)]
        + ["elevation"]
    )
    observation_size = len(observation_channels) * cells + 5
    board = _board()
    board.update(width=width, height=height, environment_kind=environment_kind)
    roster = [
        f"{template['id']}:{template['name']}:{','.join(map(str, template['stats']))}"
        for template in templates
    ]
    semantics = {
        "contract_version": "tactical-v2",
        "environment_kind": environment_kind,
        "starting_unit_count": unit_count,
        "max_controllable_units": unit_count,
        "placement_policy": "symmetric-random-v1",
        "templates": templates,
        "action_regions": action_regions,
        "observation_channels": observation_channels,
        "action_size": action_size,
        "observation_size": observation_size,
        "board": board,
    }
    return {
        "scenario_id": "tactical-v2-test",
        "scenario_schema_version": 1,
        "contract_version": "tactical-v2",
        "contract_hash": "a" * 64,
        "encoding_hash": "b" * 64,
        "environment_kind": environment_kind,
        "obs_len": observation_size,
        "n_actions": action_size,
        "channels": len(observation_channels),
        "globals": 5,
        "board_h": height,
        "board_w": width,
        "board": board,
        "roster": template_count,
        "contract_roster": roster,
        "reward": {
            "shape_scale": 0.01, "step_penalty": 0.005, "closing_weight": 0.02,
            "draw_credit_weight": 0.25, "points_weight": 0.5,
            "terminal_win": 1.0, "terminal_loss": -1.0,
        },
        "tactical_v2": semantics,
        "action_regions": action_regions,
        "observation_channels": observation_channels,
    }


def _profiled_tactical_v2_start_semantics() -> tuple[list[dict], list[dict]]:
    profiles = [
        {"id": "standard-3v3", "learner_units": 3, "opponent_units": 3,
         "separation": "legacy-mirrored"},
        {"id": "conversion-3v1-near", "learner_units": 3, "opponent_units": 1, "separation": "near"},
        {"id": "conversion-3v1-medium", "learner_units": 3, "opponent_units": 1, "separation": "medium"},
        {"id": "conversion-3v1-far", "learner_units": 3, "opponent_units": 1, "separation": "far"},
        {"id": "conversion-2v1-near", "learner_units": 2, "opponent_units": 1, "separation": "near"},
        {"id": "conversion-2v1-medium", "learner_units": 2, "opponent_units": 1, "separation": "medium"},
        {"id": "conversion-2v1-far", "learner_units": 2, "opponent_units": 1, "separation": "far"},
        {"id": "conversion-1v1-near", "learner_units": 1, "opponent_units": 1, "separation": "near"},
        {"id": "conversion-1v1-medium", "learner_units": 1, "opponent_units": 1, "separation": "medium"},
        {"id": "conversion-1v1-far", "learner_units": 1, "opponent_units": 1, "separation": "far"},
    ]
    weights = [
        {"profile_id": profile["id"], "basis_points": 10000 if profile["id"] == "standard-3v3" else 0}
        for profile in profiles
    ]
    return profiles, weights


class _IntSubclass(int):
    pass

def _valid_adaptive_spaces() -> dict:
    spaces = _valid_spaces()
    spaces["contract_version"] = "adaptive-v1"
    spaces["environment_kind"] = "adaptive_tactical"
    spaces["board"]["environment_kind"] = "adaptive_tactical"
    spaces["board"]["fog_of_war"] = True
    spaces["board"]["max_steps"] = 900
    templates = [
        ("Frontline", [7, 2, 3, 2, 2, 1, 1, 3, 1]),
        ("Assault", [3, 6, 0, 3, 2, 2, 1, 3, 1]),
        ("Marksman", [2, 3, 0, 2, 2, 6, 1, 5, 1]),
        ("Artillery", [3, 6, 0, 1, 1, 5, 2, 3, 1]),
        ("Recon", [2, 1, 0, 5, 3, 1, 0, 7, 2]),
        ("Support", [4, 3, 2, 3, 2, 3, 1, 4, 1]),
        ("Custom A", [4, 3, 1, 3, 2, 2, 1, 3, 1]),
        ("Custom B", [5, 2, 2, 2, 2, 3, 1, 3, 1]),
        ("Custom C", [3, 4, 1, 3, 2, 2, 1, 4, 1]),
    ]
    phases = [
        "deployment_root", "deployment_template", "deployment_cell",
        "deployment_placed_unit", "deployment_move_cell", "gameplay_root",
        "gameplay_unit", "gameplay_unit_command", "gameplay_move_cell",
        "gameplay_attack_cell", "design_slot", "design_stat", "design_value",
        "design_confirm",
    ]
    regions = {
        "command": {"offset": 0, "count": 12},
        "unit": {"offset": 12, "count": 24},
        "template": {"offset": 36, "count": 9},
        "cell": {"offset": 45, "count": 117},
        "stat": {"offset": 162, "count": 9},
        "value": {"offset": 171, "count": 11},
    }
    channels = [
        "elevation", "terrain_plains", "terrain_forest", "terrain_rough",
        "terrain_water", "deployment_zone_self", "current_visibility", "previously_seen",
        *[f"friendly_role_hp_{index}" for index in range(9)],
        *[f"visible_enemy_role_hp_{index}" for index in range(9)],
        *[f"friendly_slot_occupancy_{index}" for index in range(24)],
    ]
    semantics = {
        "adaptive": True,
        "contract_version": "adaptive-v1",
        "environment_kind": "adaptive_tactical",
        "fixed_template_count": 6,
        "custom_template_count": 3,
        "max_controllable_units": 24,
        "starting_unit_count": 6,
        "starting_army_budget": 132,
        "max_design_point_cost": 24,
        "intermediate_decision_penalty": 0.001,
        "deployment_completion_bonus": 0.0,
        "effective_horizon": 900,
        "fog_rule": (
            "hide_current_enemy_units_and_all_opponent_deployment_until_both_confirm;"
            "derive_action_masks_from_seat_visible_projection;"
            "authoritative_hidden_blocker_rejection_is_only_allowed_mask_rejection"
        ),
        "templates": [
            {"slot": index, "name": name, "stats": stats, "cost": sum(stats), "fixed": index < 6}
            for index, (name, stats) in enumerate(templates)
        ],
        "stat_values": {
            "health": list(range(1, 9)), "damage": list(range(0, 9)),
            "defense": list(range(0, 9)), "movement": list(range(0, 7)),
            "vertical_movement": list(range(0, 5)), "range": list(range(0, 9)),
            "range_arc": list(range(0, 5)), "vision": list(range(0, 11)),
            "vision_arc": list(range(0, 5)),
        },
        "phases": phases,
        "action_regions": regions,
        "observation_channels": channels,
        "action_size": 182,
        "observation_size": 5974,
        "board": spaces["board"],
    }
    spaces.update({
        "obs_len": 5974, "n_actions": 182, "channels": 50, "globals": 124,
        "roster": 9,
        "contract_roster": [f"{name}:{','.join(map(str, stats))}" for name, stats in templates],
        "reward": {
            "intermediate_decision_penalty": 0.001,
            "deployment_completion_bonus": 0.0,
            "terminal_win": 1.0,
            "terminal_loss": -1.0,
        },
        "adaptive": semantics,
        "action_regions": regions,
        "observation_channels": channels,
        "phases": phases,
    })
    return spaces


def test_contract_requires_lowercase_sha256_encoding_hash(tmp_path: Path) -> None:
    missing = _valid_spaces()
    del missing["encoding_hash"]
    with pytest.raises(ValueError, match="encoding_hash"):
        HexWarsEnv(_fake_server(tmp_path, missing))

    malformed = _valid_spaces()
    malformed["encoding_hash"] = "A" * 64
    with pytest.raises(ValueError, match="encoding_hash"):
        HexWarsEnv(_fake_server(tmp_path, malformed))


def test_contract_exposes_environment_version_and_encoding_hash(tmp_path: Path) -> None:
    env = HexWarsEnv(_fake_server(tmp_path, _valid_spaces()))
    try:
        assert env.contract.environment == "tactical-v1"
        assert env.contract.version == "tactical-v1"
        assert env.contract.encoding_hash == "b" * 64
    finally:
        env.close()


def test_client_passes_explicit_scenario_file_to_server(tmp_path: Path) -> None:
    scenario_path = tmp_path / "run" / "scenario.json"
    env = HexWarsEnv(
        _fake_server(tmp_path, _valid_spaces()),
        scenario_path=scenario_path,
    )
    try:
        assert env.proc.args[-2:] == ["--scenario-file", str(scenario_path)]
    finally:
        env.close()


def test_adaptive_client_accepts_complete_contract_and_keeps_fixed_spaces(tmp_path: Path) -> None:
    spaces = _valid_adaptive_spaces()
    env = HexWarsEnv(_fake_server(tmp_path, spaces), environment="adaptive-v1")
    try:
        assert env.contract.version == "adaptive-v1"
        assert env.action_space.n == spaces["n_actions"]
        assert env.contract.semantics["max_controllable_units"] == 24
        assert env.proc.args[-2:] == ["--environment", "adaptive-v1"]
    finally:
        env.close()


def test_adaptive_client_rejects_incomplete_semantics(tmp_path: Path) -> None:
    spaces = _valid_adaptive_spaces()
    del spaces["adaptive"]["stat_values"]["vision"]
    with pytest.raises(ValueError, match="stat_values"):
        HexWarsEnv(_fake_server(tmp_path, spaces), environment="adaptive-v1")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda spaces: spaces["adaptive"]["templates"][0].update(cost=1), "cost"),
        (lambda spaces: spaces["adaptive"].update(fog_rule="almost hidden"), "fog_rule"),
        (lambda spaces: spaces["reward"].update(terminal_win=2.0), "terminal_win"),
        (
            lambda spaces: spaces["adaptive"].update(deployment_completion_bonus=0.25),
            "deployment_completion_bonus",
        ),
    ],
)
def test_adaptive_client_rejects_noncanonical_semantics(
    tmp_path: Path, mutate, message: str
) -> None:
    spaces = _valid_adaptive_spaces()
    mutate(spaces)
    with pytest.raises(ValueError, match=message):
        HexWarsEnv(_fake_server(tmp_path, spaces), environment="adaptive-v1")


def test_adaptive_client_forwards_diagnostics_without_changing_reward(tmp_path: Path) -> None:
    response = {
        "obs": [0.0] * 5974,
        "mask": [True] * 182,
        "reward": 0.25,
        "terminated": True,
        "truncated": False,
        "deployment_complete": True,
        "diagnostics": {
            "design_count": 2,
            "distinct_custom_templates_deployed": 1,
            "deployment_completed": True,
            "invalid_sequences": 3,
            "pregame_decisions": 12,
        },
    }
    env = HexWarsEnv(
        _fake_server(tmp_path, _valid_adaptive_spaces(), episode_response=response),
        environment="adaptive-v1",
    )
    try:
        _, reward, terminated, truncated, info = env.step(0)
        assert (reward, terminated, truncated) == (0.25, True, False)
        assert info["diagnostics"] == response["diagnostics"]
        assert info["deployment_complete"] is True
    finally:
        env.close()


def test_adaptive_client_rejects_mask_size_change_after_handshake(tmp_path: Path) -> None:
    response = {
        "obs": [0.0] * 5974,
        "mask": [True] * 181,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    env = HexWarsEnv(
        _fake_server(tmp_path, _valid_adaptive_spaces(), episode_response=response),
        environment="adaptive-v1",
    )
    try:
        with pytest.raises(ValueError, match="protocol mask"):
            env.step(0)
    finally:
        env.close()


def test_contract_rejects_missing_semantic_field(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    del spaces["contract_hash"]
    with pytest.raises(ValueError, match="contract_hash"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_contract_rejects_board_dimensions_inconsistent_with_spaces(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    spaces["board"]["width"] = 12
    with pytest.raises(ValueError, match="board.width"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_contract_rejects_observation_geometry_inconsistent_with_spaces(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    spaces["obs_len"] = 823
    with pytest.raises(ValueError, match="obs_len"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_contract_rejects_unknown_version(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    spaces["contract_version"] = "tactical-v3"
    with pytest.raises(ValueError, match="contract_version"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_parse_tactical_v2_contract_uses_slots_and_templates() -> None:
    spaces = tactical_v2_spaces(template_count=5, unit_count=7, width=13, height=9)
    contract = parse_contract(spaces, environment="tactical-v2")

    cells = 13 * 9
    assert contract.action_size == 1 + (2 * 7 + 5) * cells
    assert contract.observation_size == (2 * 5 + 1) * cells + 5
    assert contract.semantics["starting_unit_count"] == 7


def test_tactical_v2_client_accepts_complete_contract_and_sends_environment_flag(
    tmp_path: Path,
) -> None:
    spaces = tactical_v2_spaces(template_count=5, unit_count=3, width=13, height=9)
    env = HexWarsEnv(_fake_server(tmp_path, spaces), environment="tactical-v2")
    try:
        assert env.contract.version == "tactical-v2"
        assert env.action_space.n == spaces["n_actions"]
        assert env.contract.semantics["starting_unit_count"] == 3
        assert env.proc.args[-2:] == ["--environment", "tactical-v2"]
    finally:
        env.close()


def test_tactical_v2_client_accepts_profiled_contract_with_declared_semantics(
    tmp_path: Path,
) -> None:
    spaces = tactical_v2_spaces(template_count=5, unit_count=3, width=13, height=9)
    semantics = spaces["tactical_v2"]
    semantics["placement_policy"] = "profiled-seeded-v1"
    profiles, weights = _profiled_tactical_v2_start_semantics()
    semantics["start_profiles"] = profiles
    semantics["start_distribution"] = weights

    env = HexWarsEnv(_fake_server(tmp_path, spaces), environment="tactical-v2")
    try:
        assert env.contract.semantics["start_profiles"] == semantics["start_profiles"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_policy", "placement_policy"),
        ("missing_profile", "exact versioned start profile catalog"),
        ("extra_profile", "exact versioned start profile catalog"),
        ("wrong_weight", "invalid start_distribution"),
        ("wrong_total", "invalid start_distribution"),
        ("profile_true", "invalid profile unit counts"),
        ("profile_float", "invalid profile unit counts"),
        ("profile_string", "invalid profile unit counts"),
        ("profile_int_subclass", "invalid profile unit counts"),
        ("profile_extra_field", "invalid start_profiles"),
        ("profile_missing_field", "invalid start_profiles"),
        ("weight_true", "invalid start_distribution"),
        ("weight_float", "invalid start_distribution"),
        ("weight_string", "invalid start_distribution"),
        ("weight_int_subclass", "invalid start_distribution"),
        ("weight_extra_field", "invalid start_distribution"),
        ("weight_missing_field", "invalid start_distribution"),
        ("outer_extra_field", "exact fields"),
        ("outer_missing_field", "exact fields"),
    ],
)
def test_tactical_v2_client_rejects_noncanonical_profiled_contract(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    spaces = tactical_v2_spaces(template_count=5, unit_count=3, width=13, height=9)
    semantics = spaces["tactical_v2"]
    semantics["placement_policy"] = "unknown-policy" if mutation == "unknown_policy" else "profiled-seeded-v1"
    profiles, weights = _profiled_tactical_v2_start_semantics()
    semantics["start_profiles"] = profiles
    semantics["start_distribution"] = weights
    if mutation == "missing_profile":
        profiles.pop()
    elif mutation == "extra_profile":
        profiles.append({"id": "extra", "learner_units": 1, "opponent_units": 1, "separation": "near"})
    elif mutation == "wrong_weight":
        weights[0]["basis_points"] = 10001
    elif mutation == "wrong_total":
        weights[0]["basis_points"] = 9999
    elif mutation == "profile_true":
        profiles[0]["learner_units"] = True
    elif mutation == "profile_float":
        profiles[0]["learner_units"] = 3.0
    elif mutation == "profile_string":
        profiles[0]["learner_units"] = "3"
    elif mutation == "profile_int_subclass":
        profiles[0]["learner_units"] = _IntSubclass(3)
    elif mutation == "profile_extra_field":
        profiles[0]["extra"] = "unexpected"
    elif mutation == "profile_missing_field":
        del profiles[0]["separation"]
    elif mutation == "weight_float":
        weights[0]["basis_points"] = 10000.0
    elif mutation == "weight_true":
        weights[0]["basis_points"] = True
    elif mutation == "weight_string":
        weights[0]["basis_points"] = "10000"
    elif mutation == "weight_int_subclass":
        weights[0]["basis_points"] = _IntSubclass(10000)
    elif mutation == "weight_extra_field":
        weights[0]["extra"] = "unexpected"
    elif mutation == "weight_missing_field":
        del weights[0]["basis_points"]
    elif mutation == "outer_extra_field":
        semantics["extra"] = "unexpected"
    elif mutation == "outer_missing_field":
        del semantics["start_distribution"]

    with pytest.raises(ValueError, match=message):
        if mutation.endswith("int_subclass"):
            parse_contract(spaces, environment="tactical-v2")
        else:
            HexWarsEnv(_fake_server(tmp_path, spaces), environment="tactical-v2")

def test_tactical_v2_client_rejects_duel_handshake_and_closes_server_process(
    tmp_path: Path,
) -> None:
    spaces = tactical_v2_spaces(environment_kind="duel")
    close_marker = tmp_path / "tactical-v2-duel-server-closed.txt"

    with pytest.raises(ValueError, match="environment_kind"):
        HexWarsEnv(_fake_server(tmp_path, spaces, close_marker), environment="tactical-v2")

    assert close_marker.read_text(encoding="utf-8") == "closed"


def test_tactical_v2_client_rejects_roster_mismatched_with_templates(tmp_path: Path) -> None:
    spaces = tactical_v2_spaces()
    spaces["contract_roster"][0] = "mismatched-entry"
    with pytest.raises(ValueError, match="contract_roster"):
        HexWarsEnv(_fake_server(tmp_path, spaces), environment="tactical-v2")


def test_contract_rejects_incomplete_reward(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    del spaces["reward"]["points_weight"]
    with pytest.raises(ValueError, match="reward.points_weight"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_contract_rejects_incomplete_board(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    del spaces["board"]["max_steps"]
    with pytest.raises(ValueError, match="board.max_steps"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_contract_rejects_malformed_roster_stat_line(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    spaces["contract_roster"][0] = "5,3,2"
    with pytest.raises(ValueError, match="contract_roster"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


def test_invalid_handshake_closes_server_process(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    del spaces["contract_hash"]
    close_marker = tmp_path / "server-closed.txt"
    with pytest.raises(ValueError, match="contract_hash"):
        HexWarsEnv(_fake_server(tmp_path, spaces, close_marker))
    assert close_marker.read_text(encoding="utf-8") == "closed"


def test_tactical_client_rejects_duel_handshake_and_closes_server_process(tmp_path: Path) -> None:
    spaces = _valid_spaces()
    spaces["environment_kind"] = "duel"
    spaces["board"]["environment_kind"] = "duel"
    close_marker = tmp_path / "duel-server-closed.txt"

    with pytest.raises(ValueError, match="environment_kind"):
        HexWarsEnv(_fake_server(tmp_path, spaces, close_marker))

    assert close_marker.read_text(encoding="utf-8") == "closed"


def test_no_window_creationflags_suppresses_console_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hexwars_gym.env import no_window_creationflags

    monkeypatch.setattr(sys, "platform", "win32")
    assert no_window_creationflags() == 0x08000000


def test_no_window_creationflags_is_zero_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hexwars_gym.env import no_window_creationflags

    monkeypatch.setattr(sys, "platform", "linux")
    assert no_window_creationflags() == 0


class _FakeGymServerProcess:
    """A stand-in for subprocess.Popen that answers one queued handshake line."""

    def __init__(self, response_line: str) -> None:
        self._response_line = response_line
        self.stdin = SimpleNamespace(
            write=lambda _payload: None, flush=lambda: None, close=lambda: None,
        )
        self.stdout = SimpleNamespace(readline=lambda: self._response_line)

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def test_client_passes_no_window_creationflags_to_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hexwars_gym.env as env_module

    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeGymServerProcess(json.dumps({"cmd": "spaces"}) + "\n")

    def fake_parse_contract(_spaces, *, environment, required_kind):
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(env_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(env_module, "parse_contract", fake_parse_contract)

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        HexWarsEnv(["dotnet", "server.dll"])

    assert captured.get("creationflags") == env_module.no_window_creationflags()

def test_response_info_preserves_profiled_start_id() -> None:
    from hexwars_gym.env import _response_info

    assert _response_info({"start_profile": "conversion-1v1-far"}) == {
        "start_profile": "conversion-1v1-far"}
