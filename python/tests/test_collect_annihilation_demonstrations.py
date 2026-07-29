from __future__ import annotations

from pathlib import Path

from collect_annihilation_demonstrations import CollectionSpec, collect_partition
from ml_lab.contracts import EnvironmentContract


def contract() -> EnvironmentContract:
    return EnvironmentContract("tactical-v2", "a" * 64, "b" * 64, 3, 5, {}, [], {})


class FakeClient:
    def __init__(self, _worker: int, commands: list[dict[str, object]]) -> None:
        self.contract = contract(); self.commands = commands; self.current: dict[str, object] = {}

    def enable_demonstrations(self, enabled: bool) -> None:
        assert enabled

    def reset(self, **request: object) -> dict[str, object]:
        self.current = request; self.commands.append(request); return {"winner": request["reference_seat"]}

    def drain_demonstrations(self) -> list[dict[str, object]]:
        teacher = int(self.current["reference_seat"])
        return [
            {"Observation": [0.0, 1.0, 2.0], "LegalMask": [True, False, True, False, False], "Action": 2, "Seat": teacher, "Command": {"Kind": "move"}},
            {"Observation": [3.0, 1.0, 2.0], "LegalMask": [True, False, True, False, False], "Action": 2, "Seat": 1 - teacher, "Command": {"Kind": "move"}},
        ]

    def save_replay(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(f"{self.current['seed']}:{self.current['reference_seat']}".encode()); return path

    def close(self) -> None:
        return None


def spec(root: Path, commands: list[dict[str, object]], workers: int = 1) -> CollectionSpec:
    return CollectionSpec(dataset=root, partition="train", scenario_hash="c" * 64, contract=contract(), client_factory=lambda worker: FakeClient(worker, commands), workers=workers, standard_pairs=1, conversion_pairs=1)


def test_collector_requests_reciprocal_teacher_seats_and_retains_only_teacher_rows(tmp_path: Path) -> None:
    commands: list[dict[str, object]] = []
    root = collect_partition(spec(tmp_path / "one", commands))
    assert commands[:2] == [
        {"seed": 11_000_000, "p0": "greedy", "p1": "random", "start_profile": "standard-3v3", "reference_seat": 0},
        {"seed": 11_000_000, "p0": "random", "p1": "greedy", "start_profile": "standard-3v3", "reference_seat": 1},
    ]
    games = (root / "games.jsonl").read_text().splitlines()
    assert len(games) == 4 and all('"row_count":1' in line for line in games)
    assert {item["start_profile"] for item in commands[2:]} == {"conversion-3v1-near"}
    assert all(item["reference_seat"] in {0, 1} for item in commands)


def test_one_and_four_workers_have_identical_logical_collection(tmp_path: Path) -> None:
    one_commands: list[dict[str, object]] = []; four_commands: list[dict[str, object]] = []
    one = collect_partition(spec(tmp_path / "one", one_commands, 1))
    four = collect_partition(spec(tmp_path / "four", four_commands, 4))
    assert sorted(one_commands, key=lambda item: (item["seed"], item["reference_seat"])) == sorted(four_commands, key=lambda item: (item["seed"], item["reference_seat"]))
    assert (one / "games.jsonl").read_text() == (four / "games.jsonl").read_text()
