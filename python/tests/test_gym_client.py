import json
import sys
from pathlib import Path

import pytest

from hexwars_gym import HexWarsEnv


def _fake_server(tmp_path: Path, spaces_response: dict) -> list[str]:
    server = tmp_path / "fake_gym_server.py"
    server.write_text(
        """import json
import sys

response = json.loads(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    if request[\"cmd\"] == \"spaces\":
        print(json.dumps(response), flush=True)
    elif request[\"cmd\"] == \"close\":
        break
""",
        encoding="utf-8",
    )
    return [sys.executable, str(server), json.dumps(spaces_response)]


def _valid_spaces() -> dict:
    return {
        "contract_version": "tactical-v1",
        "contract_hash": "a" * 64,
        "obs_len": 824,
        "n_actions": 1054,
        "channels": 7,
        "globals": 5,
        "board_h": 9,
        "board_w": 13,
        "board": {"width": 13, "height": 9},
        "roster": 3,
        "contract_roster": [
            "5,3,2,3,2,1,1,2,1",
            "3,5,0,3,2,2,1,3,1",
            "2,2,0,4,3,1,0,5,2",
        ],
        "reward": {"terminal_win": 1.0, "terminal_loss": -1.0},
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
