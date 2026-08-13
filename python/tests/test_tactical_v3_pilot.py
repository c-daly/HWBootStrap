from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ml_lab.tactical_v3_client import OracleStepResult, TeacherSelection
from ml_lab.tactical_v3_schema import parse_spaces, parse_view
from tests.test_tactical_v3_schema import minimal_view_payload


ROOT = Path(__file__).resolve().parents[2]
DUEL_SPACES = ROOT / "python" / "tests" / "fixtures" / "tactical_v3" / "seed-41-duel-spaces.json"


def _identity():
    return parse_spaces(json.loads(DUEL_SPACES.read_text(encoding="utf-8")))


def _view(decision_id: int, *, seat: int = 0, profile: str = "standard-3v3",
          reference_seat: int = 0, terminal: bool = False, truncated: bool = False):
    payload = minimal_view_payload()
    payload["decision_id"] = decision_id
    payload["seat"] = seat
    payload["start_profile"] = profile
    payload["reference_seat"] = reference_seat
    payload["candidates"][0]["decision_id"] = decision_id
    if terminal or truncated:
        payload["candidates"] = []
        payload["terminated"] = terminal
        payload["truncated"] = truncated
        payload["winner"] = seat if terminal else -1
        payload["reward"]["finalized"] = True
    return parse_view(payload, _identity())


class _FakeClient:
    def __init__(self, views, *, fallback: int = 0, identity_drift: bool = False) -> None:
        self._views = tuple(views)
        self._index = 0
        self._identity = _identity()
        self._identity_reads = 0
        self._identity_drift = identity_drift
        self._fallback = fallback
        self.reset_args = None

    @property
    def identity(self):
        self._identity_reads += 1
        if self._identity_drift and self._identity_reads > 1:
            return replace(self._identity, scenario_id="drift")
        return self._identity

    def duel_reset(self, *args):
        self.reset_args = args
        return self._views[0]

    def duel_oracle_step(self, decision_id: int) -> OracleStepResult:
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        self._index += 1
        return OracleStepResult(
            TeacherSelection(decision_id, current.decision.candidates[0].candidate_id,
                             4, 512, 11 + self._index, "material-plus-pursuit-v1"),
            self._views[self._index],
        )

    def duel_status(self) -> int:
        return self._fallback


def _canonical_collection():
    from ml_lab.tactical_v3_pilot import PilotCollection, collect_game, collection_schedule

    partitions = []
    games = []
    for partition in ("train", "validation"):
        examples = []
        for item in collection_schedule(partition):
            rows, game = collect_game(_FakeClient([
                _view(7, seat=item.learner_seat, profile=item.profile_id,
                      reference_seat=item.reference_seat),
                _view(8, seat=item.learner_seat, profile=item.profile_id,
                      reference_seat=item.reference_seat,
                      terminal=partition == "train", truncated=partition == "validation"),
            ]), item)
            examples.extend(rows)
            games.append(game)
        partitions.append(tuple(examples))
    return PilotCollection(_identity(), partitions[0], partitions[1], tuple(games))


def test_pilot_schedules_are_frozen_balanced_disjoint_and_exact() -> None:
    from ml_lab.tactical_v3_pilot import (
        PILOT_PROFILES, collection_schedule, evaluation_schedule,
    )

    assert PILOT_PROFILES == (
        "standard-3v3", "conversion-3v1-near", "conversion-3v1-far",
        "conversion-2v1-near", "conversion-2v1-far",
        "conversion-1v1-near", "conversion-1v1-far",
    )
    train = collection_schedule("train")
    validation = collection_schedule("validation")
    evaluation = evaluation_schedule()
    assert (len(train), len(validation), len(evaluation)) == (28, 14, 28)
    assert [(row.profile_id, row.episode_seed, row.learner_seat) for row in train[:4]] == [
        (PILOT_PROFILES[0], 61_000_000, 0),
        (PILOT_PROFILES[0], 61_000_000, 1),
        (PILOT_PROFILES[0], 61_000_001, 0),
        (PILOT_PROFILES[0], 61_000_001, 1),
    ]
    assert {row.episode_seed for row in train}.isdisjoint(
        {row.episode_seed for row in validation} | {row.episode_seed for row in evaluation}
    )
    for schedule in (train, validation, evaluation):
        assert all(row.learner_seat == row.reference_seat for row in schedule)
        grouped = {(row.profile_id, row.episode_seed): [] for row in schedule}
        for row in schedule: grouped[(row.profile_id, row.episode_seed)].append(row.learner_seat)
        assert all(seats == [0, 1] for seats in grouped.values())


def test_collect_game_labels_every_teacher_decision_and_backfills_win() -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("train", "standard-3v3", 61_000_000, 0, 0)
    client = _FakeClient([_view(7), _view(8), _view(9, terminal=True)])
    examples, summary = collect_game(client, item)

    assert client.reset_args == (61_000_000, "external", "random", 0, "standard-3v3", 0)
    assert [row.decision.decision_id for row in examples] == [7, 8]
    assert [row.target.trajectory_index for row in examples] == [0, 1]
    assert [row.target.remaining_turns_to_victory for row in examples] == [2, 1]
    assert all(row.target.terminal_outcome == "win" and not row.target.truncated for row in examples)
    assert all(row.teacher.identity == "bounded-search-v1" for row in examples)
    assert [row.teacher.actual_expansions for row in examples] == [12, 13]
    assert all(row.teacher.search_depth == 4 and row.teacher.expansion_budget == 512
               and row.teacher.confidence is None for row in examples)
    assert summary.decisions == 2 and summary.winner == 0 and summary.internal_fallback_count == 0


def test_collect_game_backfills_truncation_without_remaining_turns() -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("validation", "standard-3v3", 62_000_000, 1, 1)
    client = _FakeClient([
        _view(10, seat=1, reference_seat=1),
        _view(11, seat=1, reference_seat=1),
        _view(12, seat=1, reference_seat=1),
        _view(13, seat=1, reference_seat=1, truncated=True),
    ])
    examples, summary = collect_game(client, item)
    assert len(examples) == 3
    assert all(row.target.terminal_outcome == "draw" for row in examples)
    assert all(row.target.remaining_turns_to_victory is None for row in examples)
    assert all(row.target.truncated for row in examples)
    assert summary.truncated and not summary.terminated


@pytest.mark.parametrize("drift", ["profile", "seat", "reference", "identity", "fallback"])
def test_collect_game_rejects_semantic_or_fallback_drift(drift: str) -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("train", "standard-3v3", 61_000_000, 0, 0)
    kwargs = {}
    if drift == "profile": kwargs["profile"] = "conversion-1v1-near"
    if drift == "seat": kwargs["seat"] = 1
    if drift == "reference": kwargs["reference_seat"] = 1
    client = _FakeClient(
        [_view(7, **kwargs), _view(8, terminal=True, **kwargs)],
        fallback=1 if drift == "fallback" else 0,
        identity_drift=drift == "identity",
    )
    with pytest.raises(ValueError, match="profile|seat|identity|fallback"):
        collect_game(client, item)


def test_collection_evidence_is_canonical_content_addressed_and_write_once(tmp_path: Path) -> None:
    from ml_lab.tactical_v3_pilot import write_collection_evidence

    collection = _canonical_collection()
    output = tmp_path / "pilot"

    evidence = write_collection_evidence(output, collection)

    assert set(path.name for path in output.iterdir()) == {
        "train.jsonl", "validation.jsonl", "collection.json",
    }
    assert evidence.train_sha256 == hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()
    assert evidence.validation_sha256 == hashlib.sha256(
        (output / "validation.jsonl").read_bytes()
    ).hexdigest()
    manifest = json.loads((output / "collection.json").read_text(encoding="utf-8"))
    assert manifest["label_source"] == "bounded-search-v1"
    assert manifest["teacher"] == {
        "search_depth": 4, "expansion_budget": 512,
        "heuristic_identity": "material-plus-pursuit-v1",
    }
    assert (output / "collection.json").read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_collection_evidence(output, collection)


def test_train_pilot_uses_exact_configs_reloads_cpu_and_writes_exact_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_training import EpochMetrics, TrainingResult

    collection = _canonical_collection()
    output = tmp_path / "pilot"
    evidence = module.write_collection_evidence(output, collection)
    captured = {}

    def fake_train(train, validation, model_config, objective_config, trainer_config):
        captured.update(train=train, validation=validation, model=model_config,
                        objective=objective_config, trainer=trainer_config)
        model = TacticalV3Policy(model_config).eval()
        metrics = MappingProxyType({
            "total": 1.0, "policy": 1.0, "outcome": 0.0,
            "horizon": 0.0, "remaining_turns": 0.0,
        })
        history = (EpochMetrics(0, metrics, metrics, 1.0, True),)
        return TrainingResult(model, model_config, objective_config, trainer_config,
                              0, 1.0, False, history)

    monkeypatch.setattr(module, "train_offline", fake_train)
    artifacts = module.train_pilot(collection, evidence, output, 227, "cpu")

    assert captured["train"] is collection.train
    assert captured["validation"] is collection.validation
    assert captured["model"] == TacticalV3ModelConfig(
        hidden_dim=32, categorical_dim=8, cell_message_rounds=1,
        relation_rounds=1, attention_heads=4, feed_forward_dim=64,
        candidate_hidden_dim=64, horizon_turns=(4, 8, 16),
    )
    assert captured["objective"].policy_coefficient == 1.0
    assert captured["objective"].outcome_coefficient == 0.0
    assert captured["trainer"].batch_size == 32
    assert captured["trainer"].learning_rate == 0.001
    assert captured["trainer"].max_epochs == 100
    assert captured["trainer"].patience_epochs == 12
    assert artifacts.loaded.metadata.corpus_sha256 == evidence.collection_sha256
    assert next(artifacts.loaded.model.parameters()).device.type == "cpu"
    assert artifacts.restored_train.valid_logit_count == len(collection.train)
    assert artifacts.restored_validation.valid_logit_count == len(collection.validation)
    assert (output / "metrics.jsonl").read_text(encoding="utf-8") == (
        '{"epoch":0,"improved":true,"train":{"horizon":0.0,"outcome":0.0,'
        '"policy":1.0,"remaining_turns":0.0,"total":1.0},'
        '"validation":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"validation_policy_nll":1.0}\n'
    )


class _EvaluationClient:
    def __init__(self) -> None:
        self.resets = []
        self.steps = []
        self.status_calls = 0
        self.current = None

    def duel_reset(self, seed, p0, p1, learner, profile, reference):
        self.resets.append((seed, p0, p1, learner, profile, reference))
        self.current = (learner, profile, reference)
        return _view(8 if p0 == p1 == "random" else 7, seat=learner, profile=profile,
                     reference_seat=reference, terminal=p0 == p1 == "random")

    def duel_step(self, selection):
        self.steps.append(selection)
        learner, profile, reference = self.current
        return _view(8, seat=learner, profile=profile, reference_seat=reference,
                     terminal=True)

    def duel_status(self):
        self.status_calls += 1
        return 0


class _EvaluationPolicy:
    config = SimpleNamespace(horizon_turns=(4, 8, 16))

    def select(self, batch):
        from ml_lab.tactical_v3_model import CandidateIdentity
        return tuple(CandidateIdentity(
            int(batch.candidates.decision_id[index, 0]),
            int(batch.candidates.candidate_id[index, 0]),
        ) for index in range(batch.candidates.decision_id.shape[0]))


def test_matched_evaluation_uses_one_schedule_random_baseline_and_legal_model_actions() -> None:
    from ml_lab.tactical_v3_pilot import evaluate_pilot, evaluation_schedule

    schedule = evaluation_schedule()
    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    model_client = _EvaluationClient()
    random_client = _EvaluationClient()
    model = evaluate_pilot(model_client, loaded, "model", schedule)
    baseline = evaluate_pilot(random_client, loaded, "random", schedule)

    assert tuple(game.schedule for game in model.games) == schedule
    assert tuple(game.schedule for game in baseline.games) == schedule
    assert model.aggregate.games == baseline.aggregate.games == 28
    assert model.aggregate.wins == baseline.aggregate.wins == 28
    assert model.aggregate.candidate_errors == 0
    assert model_client.status_calls == random_client.status_calls == 28
    assert len(model_client.steps) == 28
    assert not random_client.steps
    assert all(p0 == p1 == "random" for _, p0, p1, *_ in random_client.resets)
    assert all(selection.candidate_id == 0 for selection in model_client.steps)


def test_pilot_decision_requires_metric_improvement_zero_errors_and_ten_point_margin() -> None:
    from ml_lab.tactical_v3_pilot import (
        PilotEvaluation, PilotEvaluationSummary, PilotTrainingArtifacts,
        PolicyTargetMetrics, pilot_decision,
    )

    initial = PolicyTargetMetrics(1.0, 0.25, 4, 4)
    restored = PolicyTargetMetrics(0.5, 0.75, 4, 4)
    training = PilotTrainingArtifacts(
        Path("best.pt"), initial, initial, restored, restored, object(), 1.0,
    )
    def evaluation(controller, win_rate, errors=0):
        summary = PilotEvaluationSummary(28, 0, 0, 0, 0, 1.0, errors, 0, win_rate)
        return PilotEvaluation(controller, (), summary, ())

    promising, reasons = pilot_decision(
        training, evaluation("model", 0.60), evaluation("random", 0.50),
    )
    assert promising and reasons == ()
    assert pilot_decision(
        training, evaluation("model", 0.59), evaluation("random", 0.50),
    )[0] is False
    assert pilot_decision(
        training, evaluation("model", 0.60, errors=1), evaluation("random", 0.50),
    )[0] is False


def test_run_pilot_writes_machine_authority_and_report_from_one_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collection = _canonical_collection()
    initial = module.PolicyTargetMetrics(1.0, 0.25, 4, 4)
    restored = module.PolicyTargetMetrics(0.5, 0.75, 4, 4)
    training = module.PilotTrainingArtifacts(
        Path("checkpoints/best.pt"), initial, initial, restored, restored,
        SimpleNamespace(metadata=SimpleNamespace(identity=collection.identity)), 2.5,
    )
    summary = module.PilotEvaluationSummary(28, 17, 9, 2, 2, 11.5, 0, 0, 17 / 28)
    baseline = module.PilotEvaluationSummary(28, 12, 12, 4, 4, 13.0, 0, 0, 12 / 28)
    evaluations = {
        "model": module.PilotEvaluation("model", (), summary, ()),
        "random": module.PilotEvaluation("random", (), baseline, ()),
    }

    monkeypatch.setattr(module, "collect_pilot", lambda command: collection)
    monkeypatch.setattr(module, "train_pilot",
                        lambda *args: training)
    monkeypatch.setattr(module, "evaluate_pilot",
                        lambda client, loaded, controller, schedule: evaluations[controller])

    class Client:
        def __init__(self, command, *, environment_kind):
            assert tuple(command) == ("dotnet", "server.dll")
            assert environment_kind == "duel"
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(module, "TacticalV3GymClient", Client)
    output = tmp_path / "pilot"
    command = ("python", "-m", "ml_lab", "pilot")
    assert module.run_pilot(
        ("dotnet", "server.dll"), output, 227, "cpu", command,
    ) == output

    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["execution"]["command"] == list(command)
    assert evaluation["execution"]["device"] == "cpu"
    assert evaluation["identity"]["contract_hash"] == collection.identity.contract_hash
    assert evaluation["collection"] == {
        "train_examples": 28, "validation_examples": 14,
        "train_games": 28, "validation_games": 14,
    }
    assert evaluation["decision"]["promising"] is True
    assert evaluation["claim_limit"].startswith("directional fixed-schedule")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "python -m ml_lab pilot" in report
    assert "PROMISING" in report
    assert collection.identity.contract_hash in report


def test_pilot_cli_routes_only_fixed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    server.write_bytes(b"server")
    scenario.write_text("{}", encoding="utf-8")
    output = tmp_path / "pilot"
    captured = {}
    monkeypatch.setattr(cli, "run_pilot", lambda *args: captured.setdefault("args", args))
    argv = (
        "pilot", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--seed", "227", "--device", "cuda",
    )

    assert cli.main(argv) == 0
    server_cmd, actual_output, seed, device, command = captured["args"]
    assert server_cmd == ("dotnet", str(server), "--scenario-file", str(scenario))
    assert actual_output == output
    assert seed == 227 and device == "cuda"
    assert tuple(command[-len(argv):]) == argv


@pytest.mark.parametrize("failure", ["server", "scenario", "output", "seed", "extra"])
def test_pilot_cli_rejects_invalid_inputs_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    server.write_bytes(b"server")
    scenario.write_text("{}", encoding="utf-8")
    output = tmp_path / "pilot"
    if failure == "server": server.unlink()
    if failure == "scenario": scenario.unlink()
    if failure == "output": output.mkdir()
    called = False
    def fail_if_called(*args):
        nonlocal called
        called = True
    monkeypatch.setattr(cli, "run_pilot", fail_if_called)
    argv = [
        "pilot", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--seed", "226" if failure == "seed" else "227",
        "--device", "cpu",
    ]
    if failure == "extra": argv.extend(("--profiles", "all"))

    with pytest.raises((ValueError, FileExistsError, SystemExit)):
        cli.main(argv)
    assert not called


def test_real_server_collects_reciprocal_standard_games() -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient
    from ml_lab.tactical_v3_pilot import collect_game, collection_schedule

    server = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
    scenario = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
    assert server.is_file() and scenario.is_file()
    items = collection_schedule("train")[:2]
    with TacticalV3GymClient(
        ("dotnet", str(server), "--scenario-file", str(scenario)),
        environment_kind="duel",
    ) as client:
        results = tuple(collect_game(client, item) for item in items)

    assert [item.learner_seat for item in items] == [0, 1]
    assert all(examples for examples, _ in results)
    assert all(summary.terminated or summary.truncated for _, summary in results)
    assert all(summary.internal_fallback_count == 0 for _, summary in results)
    assert all(
        example.teacher.search_depth == 4 and
        example.teacher.expansion_budget == 512 and
        example.teacher.heuristic_identity == "material-plus-pursuit-v1"
        for examples, _ in results for example in examples
    )
