from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from ml_lab.contracts import EnvironmentContract
from ml_lab.controllers import ControllerSpec, ResolvedController
from ml_lab.io import read_json


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="d" * 64,
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

    def reset(self, *, seed: int, p0: str, p1: str) -> dict:
        self.resets.append((seed, p0, p1))
        self._winner = next(self._outcomes)
        self._seat = 0
        return self._state()

    def step(self, action: int) -> dict:
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

    def close(self) -> None:
        self.closed = True


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
