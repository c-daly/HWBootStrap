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

def test_train_thresholds_stop_after_crossing_complete_pairs(tmp_path: Path) -> None:
    commands: list[dict[str, object]] = []
    collected = collect_partition(CollectionSpec(tmp_path / "thresholds", "train", "c" * 64, contract(), lambda worker: FakeClient(worker, commands), standard_threshold=3, conversion_threshold=0))
    games = collected.joinpath("games.jsonl").read_text().splitlines()
    assert len(games) == 4
    assert [item["seed"] for item in commands] == [11_000_000, 11_000_000, 11_000_001, 11_000_001]


def test_validation_schedule_is_exact_per_profile() -> None:
    from collect_annihilation_demonstrations import _pair_jobs
    jobs = _pair_jobs(CollectionSpec(Path("dataset"), "validation", "c" * 64, contract(), lambda _: None))
    assert len(jobs) == 220
    assert sum(teacher == "greedy" and profile == "standard-3v3" for teacher, profile, _ in jobs) == 100
    assert {profile: sum(item[1] == profile for item in jobs) for profile in {item[1] for item in jobs if item[0] == "bounded-search"}} == {"conversion-3v1-near": 20, "conversion-3v1-far": 20, "conversion-2v1-near": 20, "conversion-2v1-far": 20, "conversion-1v1-near": 20, "conversion-1v1-far": 20}


def test_cli_resolves_scenario_launches_default_client_and_collects(monkeypatch, tmp_path: Path) -> None:
    import collect_annihilation_demonstrations as module
    received: list[CollectionSpec] = []
    class Scenario: canonical_json = "{}"
    monkeypatch.setattr(module, "resolve_scenario", lambda **_: Scenario())
    monkeypatch.setattr(module, "DuelClient", lambda *_args, **_kwargs: FakeClient(0, []))
    monkeypatch.setattr(module, "collect_partition", lambda value: received.append(value) or value.dataset)
    module.main(["--dataset", str(tmp_path / "dataset"), "--partition", "train", "--scenario", "scenario.json"])
    assert received[0].dataset == tmp_path / "dataset" and received[0].partition == "train"

def test_cli_forwards_resolved_nondefault_scenario_to_every_server(monkeypatch, tmp_path: Path) -> None:
    import collect_annihilation_demonstrations as module
    launched: list[list[str]] = []; received: list[CollectionSpec] = []
    requested = tmp_path / "nested" / "non-default.json"; canonical = requested.resolve()
    class Scenario: canonical_json = "{\"id\":\"non-default\"}"
    class Client:
        contract = contract()
        def __init__(self, command, **_kwargs): launched.append(list(command))
        def close(self): return None
    def resolve(**kwargs):
        assert kwargs["scenario_file"] == canonical
        return Scenario()
    monkeypatch.setattr(module, "resolve_scenario", resolve)
    monkeypatch.setattr(module, "DuelClient", Client)
    monkeypatch.setattr(module, "collect_partition", lambda value: received.append(value) or value.dataset)
    module.main(["--dataset", str(tmp_path / "dataset"), "--partition", "train", "--scenario", str(requested)])
    received[0].client_factory(1)
    assert launched and all(["--scenario-file", str(canonical)] in [command[index:index + 2] for index in range(len(command) - 1)] for command in launched)
