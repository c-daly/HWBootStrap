import json
import sys
from pathlib import Path

import pytest

from hexwars_gym import HexWarsEnv


def _fake_server(tmp_path: Path, spaces_response: dict, close_marker: Path | None = None) -> list[str]:
    server = tmp_path / "fake_gym_server.py"
    server.write_text(
        """import json
import sys

response = json.loads(sys.argv[1])
close_marker = sys.argv[2]
for line in sys.stdin:
    request = json.loads(line)
    if request[\"cmd\"] == \"spaces\":
        print(json.dumps(response), flush=True)
    elif request[\"cmd\"] == \"close\":
        if close_marker:
            open(close_marker, \"w\", encoding=\"utf-8\").write(\"closed\")
        break
""",
        encoding="utf-8",
    )
    return [sys.executable, str(server), json.dumps(spaces_response), str(close_marker or "")]


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
    spaces["contract_version"] = "tactical-v2"
    with pytest.raises(ValueError, match="contract_version"):
        HexWarsEnv(_fake_server(tmp_path, spaces))


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
