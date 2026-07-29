from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from ml_lab.contracts import EnvironmentContract
from ml_lab.controllers import ControllerResolver, ControllerSpec, ResolvedController
from ml_lab.io import atomic_write_json, read_json
from ml_lab.tactical_trace import (
    CommandFrame,
    EpisodeTrace,
    SeatFrame,
    StateFrame,
    TransitionFrame,
)


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="d" * 64,
        encoding_hash="e" * 64,
        observation_size=3,
        action_size=3,
        board={"width": 1, "height": 1},
        roster=["scout"],
        reward={"terminal_win": 1.0},
    )


def _model_controller(
    tmp_path: Path,
    contract: EnvironmentContract,
    label: str,
    step: int,
) -> ResolvedController:
    checkpoint = tmp_path / label / "checkpoints" / f"step_{step:09d}.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(label.encode())
    return ResolvedController(
        spec=ControllerSpec(kind="run", path=checkpoint.parents[1]),
        server_controller="external",
        model=SimpleNamespace(label=label),
        path=checkpoint,
        algorithm="maskable_ppo",
        step=step,
        contract=contract,
        observation_size=3,
        action_size=3,
        legacy=False,
        promotable=True,
    )


class FakeDuelClient:
    def __init__(self, outcomes: Iterator[int], worker_index: int) -> None:
        self._outcomes = outcomes
        self.worker_index = worker_index
        self.resets: list[tuple[int, str, str]] = []
        self.actions: list[int] = []
        self.closed = False
        self._winner = -1
        self._seat = 0
        self._seed = -1
        self.trace_enabled = False
        self.trace_calls: list[tuple[str, object]] = []
        self.events: list[str] = []
        self.saved_replays: list[Path] = []

    def enable_trace(self, enabled: bool) -> None:
        self.trace_enabled = enabled
        self.trace_calls.append(("enable", enabled))
        self.events.append("enable")

    def reset(self, *, seed: int, p0: str, p1: str) -> dict:
        self.events.append("reset")
        self.resets.append((seed, p0, p1))
        self._winner = next(self._outcomes)
        self._seat = 0
        self._seed = seed
        return self._state()

    def step(self, action: int) -> dict:
        self.events.append("step")
        assert action == 1
        self.actions.append(action)
        if self._seat == 0:
            self._seat = 1
            return self._state()
        return {
            "terminated": self._winner in {0, 1},
            "truncated": self._winner == -1,
            "winner": self._winner,
            "seat": 0,
            "obs": [0.0, 0.0, 0.0],
            "mask": [False, True, False],
        }

    def _state(self) -> dict:
        return {
            "terminated": False,
            "truncated": False,
            "winner": -1,
            "seat": self._seat,
            "obs": [float(self.worker_index), float(self._seat), 0.0],
            "mask": [False, True, False],
        }

    def drain_trace(self) -> EpisodeTrace:
        self.trace_calls.append(("drain", self._seed))
        self.events.append("drain")
        return _episode_trace(self._winner)

    def save_replay(self, path: Path) -> Path:
        path = Path(path)
        self.trace_calls.append(("save", path))
        self.events.append("save")
        self.saved_replays.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"seed={self._seed}\n", encoding="utf-8")
        return path

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def _episode_trace(winner: int) -> EpisodeTrace:
    seats = tuple(
        SeatFrame(
            seat=seat,
            points=0,
            destroyed_value=0,
            alive_units=1,
            current_hit_points=10,
            maximum_hit_points=10,
            health_adjusted_material=10.0,
            can_damage_enemy=True,
            can_currently_attack_enemy=False,
            can_move=True,
            units=(),
        )
        for seat in (0, 1)
    )
    before = StateFrame(
        round=1,
        active_seat=0,
        is_game_over=False,
        winner=None,
        productive_legal_actions=1,
        seats=seats,
    )
    after = replace(
        before,
        active_seat=1,
        is_game_over=winner in {0, 1},
        winner=winner if winner in {0, 1} else None,
        productive_legal_actions=0,
    )
    return EpisodeTrace(
        schema_version=1,
        transitions=(
            TransitionFrame(
                before=before,
                command=CommandFrame(
                    kind="end_turn",
                    issuer=0,
                    actor_id=None,
                    target_id=None,
                    q=None,
                    r=None,
                ),
                after=after,
            ),
        ),
    )


def test_wilson_interval_has_stable_95_percent_bounds() -> None:
    from ml_lab.evaluation import wilson_interval

    interval = wilson_interval(5, 10)

    assert interval == pytest.approx(
        {"low": 0.236593090512564, "high": 0.7634069094874361, "confidence": 0.95}
    )


def test_controller_identity_preserves_scripted_name_and_checkpoint_metadata(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import controller_identity

    model = _model_controller(tmp_path, contract, "candidate", 64)
    scripted = ResolvedController(
        spec=ControllerSpec(kind="scripted", name="greedy"),
        server_controller="greedy",
        model=None,
        path=None,
        algorithm=None,
        step=None,
        contract=None,
        observation_size=None,
        action_size=None,
        legacy=False,
        promotable=False,
    )

    assert controller_identity(scripted)["name"] == "greedy"
    identity = controller_identity(model)
    assert identity["path"] == str(model.path.resolve())
    assert identity["algorithm"] == "maskable_ppo"
    assert identity["step"] == 64
    assert identity["contract_hash"] == contract.contract_hash


def test_evaluation_controller_resolution_loads_checkpoints_on_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contract: EnvironmentContract
) -> None:
    """evaluate_controllers resolves both seats through ControllerResolver, which must
    always load checkpoints on CPU: mirrors policy_server's documented rule that
    inference runs on CPU so it never competes with training for the GPU."""
    calls: list[dict] = []

    class _FakeMaskablePPO:
        @classmethod
        def load(cls, path, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                observation_space=SimpleNamespace(shape=(contract.observation_size,)),
                action_space=SimpleNamespace(n=contract.action_size),
            )

    monkeypatch.setattr("sb3_contrib.MaskablePPO", _FakeMaskablePPO)

    run = tmp_path / "eval-run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"model")
    atomic_write_json(
        run / "run.json",
        {
            "schema_version": 1,
            "config": {"algorithm": "maskable_ppo"},
            "contract": contract.to_dict(),
            "latest_checkpoint": "checkpoints/step_000000064.zip",
            "latest_checkpoint_step": 64,
        },
    )

    resolved = ControllerResolver().resolve(f"run:{run}")

    assert resolved.model is not None
    assert calls == [{"device": "cpu"}]


def test_evaluation_uses_held_out_seeds_reciprocal_seats_masks_and_identity(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    candidate = _model_controller(tmp_path, contract, "candidate", 64)
    opponent = _model_controller(tmp_path, contract, "opponent", 96)
    outcomes = iter([0, 0, 1, -1])
    clients: list[FakeDuelClient] = []
    predictions: list[tuple[str, str, list[float], list[bool]]] = []

    def client_factory(worker_index: int) -> FakeDuelClient:
        client = FakeDuelClient(outcomes, worker_index)
        clients.append(client)
        return client

    def predict_action(model, algorithm, observation, mask) -> int:
        predictions.append(
            (model.label, algorithm, observation.tolist(), mask.tolist())
        )
        assert mask.dtype == np.bool_
        return int(np.flatnonzero(mask)[0])

    result = evaluate_matchup(
        candidate,
        opponent,
        games=2,
        seed_start=10_000,
        both_seats=True,
        workers=1,
        client_factory=client_factory,
        predict_action=predict_action,
    )

    assert [seed for seed, _, _ in clients[0].resets] == [
        10_000,
        10_000,
        10_001,
        10_001,
    ]
    assert [game["candidate_seat"] for game in result["matches"]] == [0, 1, 0, 1]
    assert result["seeds"] == [10_000, 10_001]
    assert result["games"] == 4
    assert (result["wins"], result["losses"], result["draws"]) == (1, 2, 1)
    assert result["candidate"]["path"] == str(candidate.path.resolve())
    assert result["opponent"]["path"] == str(opponent.path.resolve())
    assert result["seat_results"] == {
        "candidate_as_p0": {"wins": 1, "losses": 1, "draws": 0},
        "candidate_as_p1": {"wins": 0, "losses": 1, "draws": 1},
    }
    assert set(result["confidence_intervals"]) == {"win", "loss", "draw"}
    assert len(predictions) == 8
    assert all(mask == [False, True, False] for _, _, _, mask in predictions)
    assert clients[0].closed is True


def test_adaptive_diagnostics_are_reported_without_changing_win_rate(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.contracts import ADAPTIVE_MONITOR_HEADER
    from ml_lab.evaluation import evaluate_matchup

    adaptive = replace(
        contract,
        version="adaptive-v1",
        contract_hash="e" * 64,
        semantics={"environment_kind": "adaptive_tactical"},
    )
    candidate = _model_controller(tmp_path, adaptive, "adaptive-candidate", 64)
    opponent = _model_controller(tmp_path, adaptive, "adaptive-opponent", 96)
    sidecar = candidate.spec.path / "adaptive_episodes.csv"

    def write_sidecar(rows: list[list[object]]) -> None:
        with sidecar.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(ADAPTIVE_MONITOR_HEADER)
            writer.writerows(rows)

    def evaluate() -> dict:
        return evaluate_matchup(
            candidate,
            opponent,
            games=2,
            both_seats=False,
            workers=1,
            client_factory=lambda worker: FakeDuelClient(iter([0, 1]), worker),
            predict_action=lambda _model, _algorithm, _obs, _mask: 1,
        )

    write_sidecar([[1, 2, 1, True, 0, 10], [2, 3, 2, False, 4, 20]])
    first = evaluate()
    write_sidecar([[1, 99, 8, False, 77, 100], [2, 101, 9, False, 88, 200]])
    second = evaluate()

    assert first["rates"] == second["rates"] == {"win": 0.5, "loss": 0.5, "draw": 0.0}
    assert first["design_count"] == 5
    assert first["distinct_custom_templates_deployed"] == 3
    assert first["deployment_completion_rate"] == 0.5
    assert first["invalid_sequences"] == 4
    assert first["average_pregame_decisions"] == 15.0


def test_adaptive_evaluation_aggregates_sorted_worker_sidecars_without_loss(
    tmp_path: Path,
) -> None:
    from ml_lab.contracts import ADAPTIVE_MONITOR_HEADER
    from ml_lab.evaluation import _adaptive_diagnostic_aggregates, _adaptive_sidecars

    for worker, design_count in ((10, 11), (2, 3), (0, 1), (1, 2)):
        with (tmp_path / f"adaptive_episodes.worker_{worker}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(ADAPTIVE_MONITOR_HEADER)
            writer.writerow([f"{worker}:1", design_count, 1, True, worker, 10])

    assert [path.name for path in _adaptive_sidecars(tmp_path)] == [
        "adaptive_episodes.worker_0.csv",
        "adaptive_episodes.worker_1.csv",
        "adaptive_episodes.worker_2.csv",
        "adaptive_episodes.worker_10.csv",
    ]
    result = _adaptive_diagnostic_aggregates(tmp_path)
    assert result == {
        "design_count": 17,
        "distinct_custom_templates_deployed": 4,
        "deployment_completion_rate": 1.0,
        "invalid_sequences": 13,
        "average_pregame_decisions": 10.0,
    }


def test_evaluation_reuses_exactly_one_client_per_worker(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    candidate = _model_controller(tmp_path, contract, "candidate", 64)
    opponent = _model_controller(tmp_path, contract, "opponent", 96)
    outcomes = iter([0] * 6)
    clients: list[FakeDuelClient] = []

    def client_factory(worker_index: int) -> FakeDuelClient:
        client = FakeDuelClient(outcomes, worker_index)
        clients.append(client)
        return client

    evaluate_matchup(
        candidate,
        opponent,
        games=3,
        seed_start=20_000,
        both_seats=True,
        workers=2,
        client_factory=client_factory,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
    )

    assert sorted(client.worker_index for client in clients) == [0, 1]
    assert sorted(len(client.resets) for client in clients) == [3, 3]
    assert all(client.closed for client in clients)


class CoordinatedDuelClient(FakeDuelClient):
    def __init__(
        self,
        worker_index: int,
        rendezvous: threading.Barrier,
        second_worker_finished: threading.Event,
    ) -> None:
        super().__init__(iter([0 if worker_index == 0 else 1]), worker_index)
        self._rendezvous = rendezvous
        self._second_worker_finished = second_worker_finished
        self.thread_ids: list[int] = []

    def reset(self, *, seed: int, p0: str, p1: str) -> dict:
        self.thread_ids.append(threading.get_ident())
        self._rendezvous.wait(timeout=2)
        if self.worker_index == 0:
            assert self._second_worker_finished.wait(timeout=2)
        return super().reset(seed=seed, p0=p0, p1=p1)

    def step(self, action: int) -> dict:
        self.thread_ids.append(threading.get_ident())
        state = super().step(action)
        if self.worker_index == 1 and (state["terminated"] or state["truncated"]):
            self._second_worker_finished.set()
        return state

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        super().close()


def test_evaluation_workers_overlap_own_one_client_and_merge_in_schedule_order(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    candidate = _model_controller(tmp_path, contract, "candidate", 64)
    opponent = _model_controller(tmp_path, contract, "opponent", 96)
    rendezvous = threading.Barrier(2)
    second_worker_finished = threading.Event()
    clients: dict[int, CoordinatedDuelClient] = {}
    clients_lock = threading.Lock()

    def client_factory(worker_index: int) -> CoordinatedDuelClient:
        client = CoordinatedDuelClient(
            worker_index, rendezvous, second_worker_finished
        )
        with clients_lock:
            clients[worker_index] = client
        return client

    result = evaluate_matchup(
        candidate,
        opponent,
        games=2,
        seed_start=25_000,
        both_seats=False,
        workers=2,
        client_factory=client_factory,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
    )

    assert [match["seed"] for match in result["matches"]] == [25_000, 25_001]
    assert [match["outcome"] for match in result["matches"]] == ["win", "loss"]
    assert set(clients) == {0, 1}
    assert all(len(set(client.thread_ids)) == 1 for client in clients.values())
    assert len({client.thread_ids[0] for client in clients.values()}) == 2
    assert all(client.closed for client in clients.values())


class RendezvousDuelClient(FakeDuelClient):
    def __init__(self, worker_index: int, rendezvous: threading.Barrier) -> None:
        super().__init__(iter([0]), worker_index)
        self._rendezvous = rendezvous
        self.thread_ids: list[int] = []

    def reset(self, *, seed: int, p0: str, p1: str) -> dict:
        self.thread_ids.append(threading.get_ident())
        self._rendezvous.wait(timeout=2)
        return super().reset(seed=seed, p0=p0, p1=p1)

    def step(self, action: int) -> dict:
        self.thread_ids.append(threading.get_ident())
        return super().step(action)

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        super().close()


def test_evaluation_serializes_shared_model_predictions_but_not_distinct_models(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    shared = _model_controller(tmp_path, contract, "shared", 64)
    shared_clients: dict[int, RendezvousDuelClient] = {}
    clients_lock = threading.Lock()
    shared_client_rendezvous = threading.Barrier(2)
    shared_predictions_started = 0
    shared_model_active = False
    shared_prediction_lock = threading.Lock()
    second_shared_prediction_started = threading.Event()

    def shared_client_factory(worker_index: int) -> RendezvousDuelClient:
        client = RendezvousDuelClient(worker_index, shared_client_rendezvous)
        with clients_lock:
            shared_clients[worker_index] = client
        return client

    def shared_predict_action(_model, _algorithm, _observation, _mask) -> int:
        nonlocal shared_predictions_started, shared_model_active
        with shared_prediction_lock:
            shared_predictions_started += 1
            if shared_predictions_started == 2:
                second_shared_prediction_started.set()
            assert not shared_model_active, "shared model predictions overlapped"
            shared_model_active = True
        try:
            second_shared_prediction_started.wait(timeout=1)
        finally:
            with shared_prediction_lock:
                shared_model_active = False
        return 1

    evaluate_matchup(
        shared,
        shared,
        games=2,
        seed_start=26_000,
        both_seats=False,
        workers=2,
        client_factory=shared_client_factory,
        predict_action=shared_predict_action,
    )

    assert set(shared_clients) == {0, 1}
    assert len({client.thread_ids[0] for client in shared_clients.values()}) == 2
    assert all(client.closed for client in shared_clients.values())

    candidate = _model_controller(tmp_path, contract, "candidate", 96)
    opponent = _model_controller(tmp_path, contract, "opponent", 128)
    distinct_clients: dict[int, RendezvousDuelClient] = {}
    distinct_client_rendezvous = threading.Barrier(2)
    distinct_models_started = threading.Barrier(2)

    def distinct_client_factory(worker_index: int) -> RendezvousDuelClient:
        client = RendezvousDuelClient(worker_index, distinct_client_rendezvous)
        with clients_lock:
            distinct_clients[worker_index] = client
        return client

    def distinct_predict_action(_model, _algorithm, observation, _mask) -> int:
        if observation[1] == 0.0:
            distinct_models_started.wait(timeout=1)
        return 1

    evaluate_matchup(
        candidate,
        opponent,
        games=1,
        seed_start=27_000,
        both_seats=True,
        workers=2,
        client_factory=distinct_client_factory,
        predict_action=distinct_predict_action,
    )

    assert set(distinct_clients) == {0, 1}
    assert not distinct_models_started.broken
    assert len({client.thread_ids[0] for client in distinct_clients.values()}) == 2
    assert all(client.closed for client in distinct_clients.values())


def test_evaluation_closes_all_clients_when_prediction_fails(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    candidate = _model_controller(tmp_path, contract, "candidate", 64)
    opponent = _model_controller(tmp_path, contract, "opponent", 96)
    clients: list[FakeDuelClient] = []

    def client_factory(worker_index: int) -> FakeDuelClient:
        client = FakeDuelClient(iter([0]), worker_index)
        clients.append(client)
        return client

    with pytest.raises(RuntimeError, match="prediction failed"):
        evaluate_matchup(
            candidate,
            opponent,
            games=1,
            seed_start=30_000,
            both_seats=False,
            workers=1,
            client_factory=client_factory,
            predict_action=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("prediction failed")
            ),
        )

    assert clients[0].closed is True


def test_evaluation_atomically_replaces_evaluation_json(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml_lab.evaluation import evaluate_matchup
    import ml_lab.io as io_module

    candidate = _model_controller(tmp_path, contract, "candidate", 64)
    opponent = _model_controller(tmp_path, contract, "opponent", 96)
    output = tmp_path / "run" / "evaluation.json"
    output.parent.mkdir()
    output.write_text('{"old":true}\n', encoding="utf-8")
    replacements: list[tuple[Path, Path]] = []
    real_replace = io_module.os.replace

    def record_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", record_replace)
    result = evaluate_matchup(
        candidate,
        opponent,
        games=1,
        seed_start=40_000,
        both_seats=False,
        workers=1,
        client_factory=lambda worker: FakeDuelClient(iter([0]), worker),
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
        output_path=output,
    )

    assert replacements[-1][1] == output
    assert replacements[-1][0].parent == output.parent
    assert read_json(output) == result
    assert not list(output.parent.glob(".evaluation.json.*.tmp"))


def test_evaluation_without_capture_preserves_match_schema_and_skips_trace_rpc(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    client = FakeDuelClient(iter([0]), 0)
    result = evaluate_matchup(
        _model_controller(tmp_path, contract, "candidate-ordinary", 64),
        _model_controller(tmp_path, contract, "opponent-ordinary", 96),
        games=1,
        both_seats=False,
        workers=1,
        client_factory=lambda _worker: client,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
    )

    assert "evidence" not in result
    assert result["matches"] == [
        {"seed": 1_000_000, "candidate_seat": 0, "winner": 0, "outcome": "win"}
    ]
    assert client.trace_calls == []


def test_evaluation_writes_all_draws_and_first_controls_per_candidate_seat(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    client = FakeDuelClient(
        iter([0, 1, 1, 0, 0, 1, 1, 0, -1, -1]),
        0,
    )
    evidence_dir = tmp_path / "evidence"
    result = evaluate_matchup(
        _model_controller(tmp_path, contract, "candidate-evidence", 64),
        _model_controller(tmp_path, contract, "opponent-evidence", 96),
        games=5,
        seed_start=80_000,
        both_seats=True,
        workers=1,
        client_factory=lambda _worker: client,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
        evidence_dir=evidence_dir,
        capture_trace=True,
    )

    assert result["evidence"]["draw_traces"] == 2
    assert result["evidence"]["control_traces"] == 4
    assert result["evidence"]["draw_categories"] == {"truncation": 2}
    expected_stems = [
        "match-000000-seed-80000-candidate-seat-0",
        "match-000001-seed-80000-candidate-seat-1",
        "match-000002-seed-80001-candidate-seat-0",
        "match-000003-seed-80001-candidate-seat-1",
        "match-000008-seed-80004-candidate-seat-0",
        "match-000009-seed-80004-candidate-seat-1",
    ]
    assert sorted(path.stem for path in (evidence_dir / "traces").glob("*.json")) == expected_stems
    assert sorted(path.stem for path in (evidence_dir / "replays").glob("*.replay")) == expected_stems
    assert all(
        {"trace_path", "replay_path"} <= set(match)
        for match in result["matches"]
    )
    retained = [
        index
        for index, match in enumerate(result["matches"])
        if match["trace_path"] is not None
    ]
    assert retained == [0, 1, 2, 3, 8, 9]
    assert all(
        {"terminated", "truncated", "summary", "classification"} <= set(match)
        for match in result["matches"]
    )
    assert all(
        result["matches"][index]["classification"] is None
        for index in (0, 1, 2, 3)
    )
    assert all(
        result["matches"][index]["classification"]["primary"] == "truncation"
        for index in (8, 9)
    )
    assert all(
        Path(result["matches"][index]["trace_path"])
        == evidence_dir / "traces" / f"{expected_stems[offset]}.json"
        for offset, index in enumerate(retained)
    )
    assert client.events == [
        event
        for _game in range(10)
        for event in ("enable", "reset", "step", "step", "drain", "save")
    ] + ["close"]
    assert len(client.saved_replays) == 10
    assert all(read_json(path)["schema_version"] == 1 for path in (evidence_dir / "traces").glob("*.json"))
    assert not list(evidence_dir.rglob("*.tmp"))
    assert sorted(path.name for path in evidence_dir.iterdir()) == ["replays", "traces"]


def test_trace_capture_without_directory_returns_evidence_without_replay_paths(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    client = FakeDuelClient(iter([-1, 0]), 0)
    result = evaluate_matchup(
        _model_controller(tmp_path, contract, "candidate-memory", 64),
        _model_controller(tmp_path, contract, "opponent-memory", 96),
        games=2,
        both_seats=False,
        workers=1,
        client_factory=lambda _worker: client,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
        capture_trace=True,
    )

    assert result["evidence"]["draw_traces"] == 1
    assert result["evidence"]["control_traces"] == 1
    assert all("trace_path" not in match and "replay_path" not in match for match in result["matches"])
    assert result["matches"][0]["classification"]["primary"] == "truncation"
    assert result["matches"][1]["classification"] is None
    assert client.saved_replays == []
    assert [call[0] for call in client.trace_calls] == ["enable", "drain", "enable", "drain"]


def test_requested_empty_trace_fails_and_closes_worker_client(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    class EmptyTraceClient(FakeDuelClient):
        def drain_trace(self) -> EpisodeTrace:
            self.trace_calls.append(("drain", self._seed))
            self.events.append("drain")
            return EpisodeTrace(schema_version=1, transitions=())

    client = EmptyTraceClient(iter([0]), 0)
    with pytest.raises(RuntimeError, match="empty trace"):
        evaluate_matchup(
            _model_controller(tmp_path, contract, "candidate-empty", 64),
            _model_controller(tmp_path, contract, "opponent-empty", 96),
            games=1,
            both_seats=False,
            workers=1,
            client_factory=lambda _worker: client,
            predict_action=lambda _model, _algorithm, _obs, _mask: 1,
            capture_trace=True,
        )

    assert client.closed is True
    assert client.saved_replays == []


def test_control_selection_uses_schedule_order_not_worker_completion(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import evaluate_matchup

    rendezvous = threading.Barrier(2)
    worker_one_finished = threading.Event()
    completion_order: list[int] = []
    completion_lock = threading.Lock()

    class CompletionOrderedClient(FakeDuelClient):
        def reset(self, *, seed: int, p0: str, p1: str) -> dict:
            if not self.resets:
                rendezvous.wait(timeout=2)
                if self.worker_index == 0:
                    assert worker_one_finished.wait(timeout=2)
            return super().reset(seed=seed, p0=p0, p1=p1)

        def close(self) -> None:
            if self.worker_index == 1:
                worker_one_finished.set()
            with completion_lock:
                completion_order.append(self.worker_index)
            super().close()

    clients: dict[int, CompletionOrderedClient] = {}

    def client_factory(worker_index: int) -> CompletionOrderedClient:
        client = CompletionOrderedClient(iter([0, 1, -1]), worker_index)
        clients[worker_index] = client
        return client

    evidence_dir = tmp_path / "worker-evidence"
    result = evaluate_matchup(
        _model_controller(tmp_path, contract, "candidate-workers", 64),
        _model_controller(tmp_path, contract, "opponent-workers", 96),
        games=6,
        seed_start=90_000,
        both_seats=False,
        workers=2,
        client_factory=client_factory,
        predict_action=lambda _model, _algorithm, _obs, _mask: 1,
        evidence_dir=evidence_dir,
        capture_trace=True,
    )

    assert completion_order == [1, 0]
    assert result["evidence"]["draw_traces"] == 2
    assert result["evidence"]["control_traces"] == 2
    assert all(
        {"trace_path", "replay_path"} <= set(match)
        for match in result["matches"]
    )
    assert [
        index
        for index, match in enumerate(result["matches"])
        if match["trace_path"] is not None
    ] == [0, 2, 4, 5]
    assert sorted(path.stem for path in (evidence_dir / "traces").glob("*.json")) == [
        "match-000000-seed-90000-candidate-seat-0",
        "match-000002-seed-90002-candidate-seat-0",
        "match-000004-seed-90004-candidate-seat-0",
        "match-000005-seed-90005-candidate-seat-0",
    ]
    assert all(client.closed for client in clients.values())


def test_evaluate_controllers_rejects_unknown_environment_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.evaluation as evaluation_module

    monkeypatch.setattr(
        evaluation_module,
        "ControllerResolver",
        lambda: pytest.fail("unsupported environment reached controller resolution"),
    )

    with pytest.raises(ValueError, match="unsupported environment"):
        evaluation_module.evaluate_controllers(
            "greedy",
            "random",
            games=1,
            server_cmd=["server"],
            environment="tactical-v3",
        )


def test_evaluate_controllers_rejects_explicit_model_contract_mismatch_before_server(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.evaluation as evaluation_module

    controllers = iter(
        [
            _model_controller(tmp_path, contract, "candidate-mismatch", 64),
            _model_controller(tmp_path, contract, "opponent-mismatch", 96),
        ]
    )
    monkeypatch.setattr(
        evaluation_module,
        "ControllerResolver",
        lambda: SimpleNamespace(resolve=lambda _raw: next(controllers)),
    )
    monkeypatch.setattr(
        evaluation_module,
        "DuelClient",
        lambda *_args, **_kwargs: pytest.fail("contract mismatch started a server"),
    )

    with pytest.raises(ValueError, match="explicit environment"):
        evaluation_module.evaluate_controllers(
            "candidate",
            "opponent",
            games=1,
            server_cmd=["server"],
            environment="tactical-v2",
        )


def test_evaluate_controllers_rejects_trace_capture_outside_tactical_v2_before_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.evaluation as evaluation_module

    monkeypatch.setattr(
        evaluation_module,
        "DuelClient",
        lambda *_args, **_kwargs: pytest.fail("invalid trace environment started a server"),
    )

    with pytest.raises(ValueError, match="tactical-v2"):
        evaluation_module.evaluate_controllers(
            "greedy",
            "random",
            games=1,
            server_cmd=["server"],
            environment="tactical-v1",
            capture_trace=True,
        )


class _FakeDuelProcess:
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


def test_duel_client_trace_methods_send_exact_rpc_sequence(tmp_path: Path) -> None:
    from ml_lab.evaluation import DuelClient

    replay_path = tmp_path / "replays" / "match.replay"
    responses = iter(
        [
            {"enabled": True},
            {"schema_version": 1, "transitions": []},
            {"saved": str(replay_path)},
        ]
    )
    requests: list[dict[str, object]] = []
    client = object.__new__(DuelClient)
    client._rpc = lambda request: (requests.append(request), next(responses))[1]

    client.enable_trace(True)
    trace = client.drain_trace()
    saved = client.save_replay(replay_path)

    assert requests == [
        {"cmd": "duel_trace_enable", "enabled": True},
        {"cmd": "duel_trace_drain"},
        {"cmd": "duel_save", "path": str(replay_path)},
    ]
    assert trace == EpisodeTrace(schema_version=1, transitions=())
    assert saved == replay_path


def test_duel_client_sends_tactical_v2_and_requires_duel_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as json_module
    import ml_lab.evaluation as evaluation_module

    launched: list[list[str]] = []
    captured: dict[str, str] = {}

    def fake_popen(command, **_kwargs):
        launched.append(list(command))
        return _FakeDuelProcess(json_module.dumps({"cmd": "duel_spaces"}) + "\n")

    def fake_parse_contract(_spaces, *, environment, required_kind):
        captured["environment"] = environment
        captured["required_kind"] = required_kind
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(evaluation_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluation_module, "parse_contract", fake_parse_contract)

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        evaluation_module.DuelClient(["dotnet", "server.dll"], environment="tactical-v2")

    assert launched == [["dotnet", "server.dll", "--environment", "tactical-v2"]]
    assert captured == {"environment": "tactical-v2", "required_kind": "duel"}


def test_duel_client_rejects_unknown_environment_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as json_module
    import ml_lab.evaluation as evaluation_module

    def fake_popen(_command, **_kwargs):
        return _FakeDuelProcess(json_module.dumps({"cmd": "duel_spaces"}) + "\n")

    monkeypatch.setattr(evaluation_module.subprocess, "Popen", fake_popen)

    with pytest.raises(ValueError, match="unsupported environment"):
        evaluation_module.DuelClient(["dotnet", "server.dll"], environment="tactical-v3")


def test_duel_client_passes_no_window_creationflags_to_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as json_module
    import ml_lab.evaluation as evaluation_module

    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeDuelProcess(json_module.dumps({"cmd": "duel_spaces"}) + "\n")

    def fake_parse_contract(_spaces, *, environment, required_kind):
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(evaluation_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(evaluation_module, "parse_contract", fake_parse_contract)

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        evaluation_module.DuelClient(["dotnet", "server.dll"], environment="tactical-v2")

    assert captured.get("creationflags") == evaluation_module.no_window_creationflags()


def test_benchmark_client_passes_no_window_creationflags_to_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json as json_module
    import ml_lab.benchmark as benchmark_module

    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeDuelProcess(json_module.dumps({"cmd": "spaces"}) + "\n")

    def fake_parse_contract(_spaces, *, environment, required_kind):
        raise RuntimeError("stop after handshake capture")

    monkeypatch.setattr(benchmark_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(benchmark_module, "parse_contract", fake_parse_contract)

    with pytest.raises(RuntimeError, match="stop after handshake capture"):
        benchmark_module.BenchmarkClient(["dotnet", "server.dll"])

    assert captured.get("creationflags") == benchmark_module.no_window_creationflags()


class FakeBenchmarkClient:
    def __init__(self, worker_index: int) -> None:
        self.worker_index = worker_index
        self.seeds: list[int] = []
        self.bytes_sent = 100 + worker_index
        self.bytes_received = 200 + worker_index
        self.request_count = 4
        self.response_count = 4
        self.closed = False

    def run_episode(self, seed: int) -> int:
        self.seeds.append(seed)
        return 2

    def close(self) -> None:
        self.closed = True


class ConcurrentBenchmarkClient(FakeBenchmarkClient):
    def __init__(self, worker_index: int, rendezvous: threading.Barrier) -> None:
        super().__init__(worker_index)
        self._rendezvous = rendezvous
        self.thread_ids: list[int] = []
        self._first_episode = True

    def run_episode(self, seed: int) -> int:
        self.thread_ids.append(threading.get_ident())
        if self._first_episode:
            self._first_episode = False
            self._rendezvous.wait(timeout=2)
        return super().run_episode(seed)

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        super().close()


def test_benchmark_reports_throughput_workers_cpu_and_protocol_payloads() -> None:
    from ml_lab.benchmark import benchmark_gymserver

    clients: list[FakeBenchmarkClient] = []

    def client_factory(worker_index: int) -> FakeBenchmarkClient:
        client = FakeBenchmarkClient(worker_index)
        clients.append(client)
        return client

    times = iter([10.0, 12.0])
    result = benchmark_gymserver(
        games=4,
        seed_start=50_000,
        workers=2,
        client_factory=client_factory,
        clock=lambda: next(times),
        cpu_count=lambda: 12,
    )

    assert result["elapsed_seconds"] == 2.0
    assert result["reset_count"] == 4
    assert result["decision_count"] == 8
    assert result["resets_per_second"] == 2.0
    assert result["decisions_per_second"] == 4.0
    assert result["cpu_count"] == 12
    assert result["worker_count"] == 2
    assert result["protocol"] == {
        "bytes_sent": 201,
        "bytes_received": 401,
        "total_bytes": 602,
        "request_count": 8,
        "response_count": 8,
        "mean_request_bytes": 25.125,
        "mean_response_bytes": 50.125,
    }
    assert [client.seeds for client in sorted(clients, key=lambda client: client.worker_index)] == [
        [50_000, 50_002],
        [50_001, 50_003],
    ]
    assert all(client.closed for client in clients)


def test_benchmark_workers_overlap_and_own_their_clients() -> None:
    from ml_lab.benchmark import benchmark_gymserver

    rendezvous = threading.Barrier(2)
    clients: dict[int, ConcurrentBenchmarkClient] = {}
    clients_lock = threading.Lock()

    def client_factory(worker_index: int) -> ConcurrentBenchmarkClient:
        client = ConcurrentBenchmarkClient(worker_index, rendezvous)
        with clients_lock:
            clients[worker_index] = client
        return client

    times = iter([10.0, 12.0])
    result = benchmark_gymserver(
        games=4,
        seed_start=60_000,
        workers=2,
        client_factory=client_factory,
        clock=lambda: next(times),
        cpu_count=lambda: 8,
    )

    assert result["decision_count"] == 8
    assert clients[0].seeds == [60_000, 60_002]
    assert clients[1].seeds == [60_001, 60_003]
    assert all(len(set(client.thread_ids)) == 1 for client in clients.values())
    assert len({client.thread_ids[0] for client in clients.values()}) == 2
    assert all(client.closed for client in clients.values())


def test_benchmark_closes_every_thread_owned_client_when_one_worker_fails() -> None:
    from ml_lab.benchmark import benchmark_gymserver

    rendezvous = threading.Barrier(2)
    clients: dict[int, ConcurrentBenchmarkClient] = {}
    clients_lock = threading.Lock()

    class FailingClient(ConcurrentBenchmarkClient):
        def run_episode(self, seed: int) -> int:
            result = super().run_episode(seed)
            if self.worker_index == 1:
                raise RuntimeError("worker failed")
            return result

    def client_factory(worker_index: int) -> ConcurrentBenchmarkClient:
        client = FailingClient(worker_index, rendezvous)
        with clients_lock:
            clients[worker_index] = client
        return client

    with pytest.raises(RuntimeError, match="worker failed"):
        benchmark_gymserver(
            games=2,
            seed_start=70_000,
            workers=2,
            client_factory=client_factory,
        )

    assert set(clients) == {0, 1}
    assert all(client.closed for client in clients.values())
    assert all(len(set(client.thread_ids)) == 1 for client in clients.values())
