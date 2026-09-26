from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from ml_lab.scenarios import resolve_scenario
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_outcome import (
    OutcomeEvaluation,
    OutcomeGameResult,
    OutcomeTrainingConfig,
    OutcomeUpdateMetrics,
    _TRAIN_SEED_BASE,
    _VALIDATION_SEED_BASE,
    _Telemetry,
    _better,
    _checkpoint_candidate,
    _cpu_policy_snapshot,
    _evaluate,
    _freeze_opponent,
    _greedy_sample,
    _opponent_provenance,
    _stop_mode,
    _validation_due,
    optimize_outcome_rollout,
    run_outcome_training,
)
from ml_lab.tactical_v3_outcome_checkpoint import (
    load_outcome_checkpoint,
    outcome_model_state_sha256,
)
from ml_lab.tactical_v3_schema import TacticalV3Reward
from ml_lab.tactical_v3_trajectory import (
    ControllerProvenance,
    TacticalV3TrajectoryGame,
    TrajectoryDecisionRecord,
)
from tests.tactical_v3_fixture_support import load_duel_identity_fixture
from tests.test_tactical_v3_model import canonical_model_example


def _policy() -> TacticalV3Policy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return TacticalV3Policy(TacticalV3ModelConfig(
            hidden_dim=16,
            categorical_dim=4,
            cell_message_rounds=1,
            relation_rounds=1,
            attention_heads=4,
            feed_forward_dim=32,
            candidate_hidden_dim=32,
            horizon_turns=(4, 8, 16),
        )).cpu()


def _game(
    model: TacticalV3Policy,
    reward_total: float,
    winner: int,
    *,
    forced: bool = False,
) -> OutcomeGameResult:
    identity = load_duel_identity_fixture()
    decision = canonical_model_example().decision
    if forced:
        decision = replace(decision, candidates=decision.candidates[:1])
    reward = TacticalV3Reward(
        terminal_outcome=1.0 if winner == 0 else -1.0,
        known_health_adjusted_material_progress=0.0,
        public_resource_progress=0.0,
        time_pressure=0.0,
        total=reward_total,
        finalized=True,
    )
    provenance = ControllerProvenance(
        "model", "structured_policy_gradient", "/tmp/run",
        outcome_model_state_sha256(model),
    )
    record = TrajectoryDecisionRecord(
        trajectory_index=0,
        decision=decision,
        selected_candidate_id=decision.candidates[0].candidate_id,
        behavior_mode="categorical",
        log_probability=-1.0,
        entropy=1.0,
        successor_reward=reward,
        terminated_after_selection=True,
        truncated_after_selection=False,
    )
    game = TacticalV3TrajectoryGame(
        identity=identity,
        partition="train",
        game_index=0,
        episode_seed=40_000_000,
        profile_id="conversion-1v1-near",
        learner_seat=0,
        reference_seat=0,
        actor=provenance,
        opponent=ControllerProvenance(
            "scripted", "passive", "GymServer:passive", "b" * 64,
        ),
        records=(record,),
        replay=b"unused by optimizer",
        winner=winner,
        terminated=True,
        truncated=False,
        terminal_reward=reward,
        internal_fallback_count=0,
    )
    return OutcomeGameResult(game, reward)


def _evaluation(
    win_rate: float,
    mean_return: float,
    decisions: float,
    *,
    opponent_artifact_sha256: str = "a" * 64,
) -> OutcomeEvaluation:
    games = 4
    wins = int(win_rate * games)
    return OutcomeEvaluation(
        games=games,
        wins=wins,
        losses=games - wins,
        draws=0,
        truncations=0,
        mean_return=mean_return,
        mean_learner_decisions=decisions,
        choice_decisions=games,
        forced_decisions=0,
        seat_0_win_rate=win_rate,
        seat_1_win_rate=win_rate,
        opponent_artifact_sha256=opponent_artifact_sha256,
        fixture_decisions=(canonical_model_example().decision,),
    )


def _run_stop_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: str,
) -> tuple[Path, list[str], int, list[int]]:
    import ml_lab.tactical_v3_outcome as outcome_module

    identity = load_duel_identity_fixture()
    scenario = resolve_scenario(
        environment="tactical-v3",
        scenario_file=None,
        template_id="tactical-v3-close-static-v1",
    )
    scenario_path = tmp_path / "scenario.json"
    scenario.write(scenario_path)
    config = OutcomeTrainingConfig(
        run_name="stop-lifecycle",
        scenario_file=scenario_path,
        opponent="passive",
        total_decisions=6,
        seed=227,
        device="cpu",
        learner_seat="alternating",
        trackers=({"kind": "local"},),
        rollout_decisions=2,
        validation_games=2,
        validation_every_updates=8,
        micro_batch_size=2,
    )

    class Client:
        def __init__(self, command, *, environment_kind):
            assert tuple(command) == ("dotnet", "server.dll")
            assert environment_kind == "duel"
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    evaluations = 0

    def fake_evaluate(
        _client, _model, _opponent, opponent_provenance, actual_config,
        *_args, **_kwargs,
    ):
        nonlocal evaluations
        evaluations += 1
        wins = 0 if evaluations == 1 else 2
        evaluation = OutcomeEvaluation(
            games=2,
            wins=wins,
            losses=2 - wins,
            draws=0,
            truncations=0,
            mean_return=-1.0 if wins == 0 else 1.0,
            mean_learner_decisions=1.0,
            choice_decisions=2,
            forced_decisions=0,
            seat_0_win_rate=float(wins == 2),
            seat_1_win_rate=float(wins == 2),
            opponent_artifact_sha256=opponent_provenance.artifact_sha256,
            fixture_decisions=(canonical_model_example().decision,),
        )
        return evaluation, actual_config.validation_games

    collected = 0

    def fake_collect(_client, model, _opponent, **kwargs):
        nonlocal collected
        collected += 1
        original = _game(model, 1.0, 0)
        result = replace(
            original,
            game=replace(
                original.game,
                partition="train",
                game_index=kwargs["game_index"],
                episode_seed=kwargs["episode_seed"],
                profile_id=kwargs["profile_id"],
                learner_seat=kwargs["learner_seat"],
                reference_seat=kwargs["learner_seat"],
                opponent=kwargs["frozen_opponent_provenance"],
                winner=kwargs["learner_seat"],
            ),
        )
        if request == "stop_after_checkpoint" and collected == 2:
            (tmp_path / config.run_name / "control.json").write_text(
                '{"request":"stop_after_checkpoint"}\n', encoding="utf-8",
            )
        return result

    def fake_optimize(_model, _optimizer, games, *, micro_batch_size):
        assert micro_batch_size == 2
        if request == "stop_now":
            (tmp_path / config.run_name / "control.json").write_text(
                '{"request":"stop_now"}\n', encoding="utf-8",
            )
        decisions = sum(len(result.game.records) for result in games)
        return OutcomeUpdateMetrics(
            total_loss=0.1,
            policy_loss=0.1,
            outcome_loss=0.1,
            entropy=0.1,
            mean_return=1.0,
            mean_choice_return=1.0,
            mean_game_return=1.0,
            mean_baseline=0.0,
            mean_advantage=1.0,
            mean_choice_baseline=0.0,
            mean_choice_advantage=1.0,
            approximate_kl=0.0,
            gradient_norm=0.1,
            decisions=decisions,
            choice_decisions=decisions,
            forced_decisions=0,
            games=len(games),
            wins=len(games),
            losses=0,
            draws=0,
        )

    def fake_manifest(root: Path, _identity):
        path = Path(root) / "manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    states: list[str] = []
    checkpoint_updates: list[int] = []
    real_update_run_state = outcome_module.update_run_state
    real_checkpoint_candidate = outcome_module._checkpoint_candidate

    def recording_update_run_state(run_dir, state, **fields):
        states.append(state)
        return real_update_run_state(run_dir, state, **fields)

    def recording_checkpoint_candidate(
        run_dir, model, actual_identity, initialization, evaluation,
        update, validation_game_start,
    ):
        checkpoint_updates.append(update)
        return real_checkpoint_candidate(
            run_dir,
            model,
            actual_identity,
            initialization,
            evaluation,
            update,
            validation_game_start,
        )

    initial_policy = _policy()
    initialization = {
        "kind": "scratch",
        "seed": 227,
        "model_state_sha256": outcome_model_state_sha256(initial_policy),
    }

    monkeypatch.setattr(outcome_module, "TacticalV3GymClient", Client)
    monkeypatch.setattr(
        outcome_module, "_validate_target_scenario_identity",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        outcome_module, "_load_initial_policy",
        lambda *_args: (initial_policy, initialization),
    )
    monkeypatch.setattr(outcome_module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(outcome_module, "collect_outcome_game", fake_collect)
    monkeypatch.setattr(outcome_module, "optimize_outcome_rollout", fake_optimize)
    monkeypatch.setattr(outcome_module, "write_trajectory_manifest", fake_manifest)
    monkeypatch.setattr(
        outcome_module, "_checkpoint_candidate", recording_checkpoint_candidate,
    )
    monkeypatch.setattr(outcome_module, "update_run_state", recording_update_run_state)

    run = run_outcome_training(
        config,
        runs_root=tmp_path,
        server_cmd=("dotnet", "server.dll"),
    )
    return run, states, evaluations, checkpoint_updates


def test_outcome_config_keeps_source_optional_and_rejects_nonreciprocal_validation(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}\n", encoding="utf-8")
    config = OutcomeTrainingConfig(
        run_name="candidate",
        scenario_file=scenario,
        opponent="passive",
        total_decisions=64,
        seed=227,
        device="cpu",
        learner_seat="alternating",
        trackers=({"kind": "local"},),
    )

    config.validate()
    assert config.source_run is None
    assert config.rollout_decisions == 64
    assert config.validation_games == 32
    assert config.validation_every_updates == 8
    with pytest.raises(ValueError, match="reciprocal and even"):
        replace(config, validation_games=3).validate()
    with pytest.raises(ValueError, match="validation_every_updates"):
        replace(config, validation_every_updates=0).validate()


def test_one_rollout_update_consumes_every_decision_and_changes_actor() -> None:
    model = _policy()
    games = (_game(model, 1.0, 0), _game(model, -1.0, 1))
    before = outcome_model_state_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    metrics = optimize_outcome_rollout(
        model, optimizer, games, micro_batch_size=1,
    )

    assert metrics.decisions == sum(len(game.game.records) for game in games)
    assert metrics.games == 2
    assert metrics.total_loss == pytest.approx(metrics.total_loss)
    assert metrics.gradient_norm > 0.0
    assert outcome_model_state_sha256(model) != before


def test_cpu_validation_snapshot_is_weight_identical_and_independent() -> None:
    model = _policy()
    snapshot = _cpu_policy_snapshot(model)
    original_hash = outcome_model_state_sha256(model)

    assert next(snapshot.parameters()).device.type == "cpu"
    assert outcome_model_state_sha256(snapshot) == original_hash
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] += 1.0
    assert outcome_model_state_sha256(snapshot) == original_hash
    assert outcome_model_state_sha256(model) != original_hash


def test_rollout_is_rejected_after_actor_weights_change() -> None:
    model = _policy()
    games = (_game(model, 1.0, 0),)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] += 1.0

    with pytest.raises(ValueError, match="actor hash"):
        optimize_outcome_rollout(
            model,
            torch.optim.Adam(model.parameters(), lr=3e-4),
            games,
            micro_batch_size=1,
        )


@pytest.mark.parametrize("mutation", ("validation", "greedy"))
def test_optimizer_rejects_nontraining_or_nonstochastic_trajectories(
    mutation: str,
) -> None:
    model = _policy()
    original = _game(model, 1.0, 0)
    if mutation == "validation":
        invalid = replace(
            original,
            game=replace(original.game, partition="validation"),
        )
        message = "only training trajectories"
    else:
        invalid_record = replace(
            original.game.records[0],
            behavior_mode="greedy",
            log_probability=0.0,
            entropy=0.0,
        )
        invalid = replace(
            original,
            game=replace(original.game, records=(invalid_record,)),
        )
        message = "categorical behavior"

    with pytest.raises(ValueError, match=message):
        optimize_outcome_rollout(
            model,
            torch.optim.Adam(model.parameters(), lr=3e-4),
            (invalid,),
            micro_batch_size=1,
        )


def test_greedy_archive_metadata_describes_the_deterministic_behavior() -> None:
    model = _policy()
    decision = canonical_model_example().decision

    candidate_id, log_probability, entropy = _greedy_sample(model, decision)

    assert candidate_id in {
        candidate.candidate_id for candidate in decision.candidates
    }
    assert log_probability == 0.0
    assert entropy == 0.0


def test_best_selection_prioritizes_wins_then_return_without_rewarding_passivity() -> None:
    base = _evaluation(0.5, 0.1, 8.0)

    assert _better(_evaluation(0.75, -0.9, 20.0), base)
    assert _better(_evaluation(0.5, 0.2, 20.0), base)
    assert not _better(_evaluation(0.5, 0.1, 7.0), base)
    assert not _better(_evaluation(0.5, 0.1, 9.0), base)


def test_validation_cadence_includes_interval_final_and_checkpoint_stop() -> None:
    assert not _validation_due(
        update=1,
        optimized_decisions=64,
        total_decisions=1_024,
        validation_every_updates=8,
        checkpoint_stop=False,
    )
    assert _validation_due(
        update=8,
        optimized_decisions=512,
        total_decisions=1_024,
        validation_every_updates=8,
        checkpoint_stop=False,
    )
    assert _validation_due(
        update=3,
        optimized_decisions=1_024,
        total_decisions=1_024,
        validation_every_updates=8,
        checkpoint_stop=False,
    )
    assert _validation_due(
        update=3,
        optimized_decisions=192,
        total_decisions=1_024,
        validation_every_updates=8,
        checkpoint_stop=True,
    )


def test_rollout_metrics_distinguish_choice_and_forced_decisions() -> None:
    model = _policy()
    games = (
        _game(model, 1.0, 0),
        _game(model, -1.0, 1, forced=True),
    )

    metrics = optimize_outcome_rollout(
        model,
        torch.optim.Adam(model.parameters(), lr=3e-4),
        games,
        micro_batch_size=1,
    )

    assert metrics.decisions == 2
    assert metrics.choice_decisions == 1
    assert metrics.forced_decisions == 1
    assert metrics.wins == 1
    assert metrics.losses == 1
    assert metrics.draws == 0
    assert metrics.mean_game_return == pytest.approx(0.0)


def test_choice_normalized_update_is_micro_batch_invariant() -> None:
    full_batch_model = _policy()
    split_batch_model = _policy()
    split_batch_model.load_state_dict(full_batch_model.state_dict())
    full_games = (
        _game(full_batch_model, 1.0, 0),
        _game(full_batch_model, -1.0, 1, forced=True),
    )
    split_games = (
        _game(split_batch_model, 1.0, 0),
        _game(split_batch_model, -1.0, 1, forced=True),
    )

    full_metrics = optimize_outcome_rollout(
        full_batch_model,
        torch.optim.Adam(full_batch_model.parameters(), lr=3e-4),
        full_games,
        micro_batch_size=2,
    )
    split_metrics = optimize_outcome_rollout(
        split_batch_model,
        torch.optim.Adam(split_batch_model.parameters(), lr=3e-4),
        split_games,
        micro_batch_size=1,
    )

    assert split_metrics.policy_loss == pytest.approx(full_metrics.policy_loss)
    assert split_metrics.outcome_loss == pytest.approx(full_metrics.outcome_loss)
    assert split_metrics.entropy == pytest.approx(full_metrics.entropy)
    for full_parameter, split_parameter in zip(
        full_batch_model.parameters(), split_batch_model.parameters(), strict=True,
    ):
        torch.testing.assert_close(full_parameter, split_parameter)


def test_best_selection_rejects_a_different_validation_opponent_revision() -> None:
    with pytest.raises(ValueError, match="opponent changed"):
        _better(
            _evaluation(
                1.0, 1.0, 1.0, opponent_artifact_sha256="b" * 64,
            ),
            _evaluation(
                0.0, -1.0, 9.0, opponent_artifact_sha256="a" * 64,
            ),
        )


def test_checkpoint_reopen_failure_cannot_advance_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_outcome_checkpoint as checkpoint_module

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "trajectories" / "train").mkdir(parents=True)
    (run / "trajectories" / "validation").mkdir()
    manifest = {"latest_checkpoint": None, "latest_checkpoint_step": None}
    (run / "run.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8",
    )

    def reject_staged_checkpoint(*_args, **_kwargs):
        raise ValueError("simulated corrupt staged checkpoint")

    monkeypatch.setattr(
        checkpoint_module, "load_outcome_checkpoint", reject_staged_checkpoint,
    )
    model = _policy()
    with pytest.raises(ValueError, match="corrupt staged checkpoint"):
        _checkpoint_candidate(
            run,
            model,
            load_duel_identity_fixture(),
            {
                "kind": "scratch",
                "seed": 71,
                "model_state_sha256": outcome_model_state_sha256(model),
            },
            _evaluation(0.5, 0.0, 1.0),
            update=1,
            validation_game_start=4,
        )

    assert json.loads((run / "run.json").read_text(encoding="utf-8")) == manifest
    assert not (run / "checkpoints" / "policy-update-000001.pt").exists()


def test_freezing_resolves_one_revision_for_a_reciprocal_pair() -> None:
    class ChangingOpponent:
        def __init__(self) -> None:
            self.calls = 0

        def controller_for_game(self, _identity):
            value = ("random", "greedy")[self.calls]
            self.calls += 1
            return value

    opponent = ChangingOpponent()
    identity = load_duel_identity_fixture()

    controller, provenance = _freeze_opponent(opponent, identity)

    assert opponent.calls == 1
    assert controller == "random"
    assert provenance == _opponent_provenance("random")
    assert [(controller, provenance) for _seat in (0, 1)] == [
        ("random", provenance),
        ("random", provenance),
    ]
    next_controller, next_provenance = _freeze_opponent(opponent, identity)
    assert opponent.calls == 2
    assert next_controller == "greedy"
    assert next_provenance != provenance


def test_validation_sweep_reuses_one_frozen_opponent_and_records_its_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_outcome as outcome_module

    model = _policy()
    provenance = _opponent_provenance("random")
    calls: list[tuple[object, object, int]] = []

    def fake_collect(_client, current_model, controller, **kwargs):
        calls.append((
            controller,
            kwargs["frozen_opponent_provenance"],
            kwargs["learner_seat"],
        ))
        seat = kwargs["learner_seat"]
        original = _game(current_model, 1.0, 0)
        return replace(
            original,
            game=replace(
                original.game,
                partition="validation",
                game_index=kwargs["game_index"],
                learner_seat=seat,
                reference_seat=seat,
                opponent=provenance,
                winner=seat,
            ),
        )

    class Telemetry:
        def record(self, _values, _step) -> None:
            pass

    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}\n", encoding="utf-8")
    config = OutcomeTrainingConfig(
        run_name="candidate",
        scenario_file=scenario,
        opponent="random",
        total_decisions=2,
        seed=227,
        device="cpu",
        learner_seat="alternating",
        trackers=(),
        validation_games=4,
    )
    monkeypatch.setattr(outcome_module, "collect_outcome_game", fake_collect)
    monkeypatch.setattr(outcome_module, "_record_game", lambda *_args, **_kwargs: None)

    evaluation, games = _evaluate(
        object(),
        model,
        "random",
        provenance,
        config,
        tmp_path,
        Telemetry(),
        (("conversion-1v1-near", 10_000),),
        validation_game_start=0,
        global_episode_start=0,
        update=0,
        started=0.0,
    )

    assert games == 4
    assert calls == [
        ("random", provenance, 0),
        ("random", provenance, 1),
        ("random", provenance, 0),
        ("random", provenance, 1),
    ]
    assert evaluation.opponent_artifact_sha256 == provenance.artifact_sha256


def test_training_and_development_seed_namespaces_are_disjoint() -> None:
    assert _TRAIN_SEED_BASE != 10_000_000
    assert _VALIDATION_SEED_BASE != 10_000_000
    train = set(range(_TRAIN_SEED_BASE, _TRAIN_SEED_BASE + 20_000))
    validation = set(range(_VALIDATION_SEED_BASE, _VALIDATION_SEED_BASE + 20_000))
    assert train.isdisjoint(validation)


def test_outcome_control_explicit_null_continues(
    tmp_path: Path,
) -> None:
    (tmp_path / "control.json").write_text(
        '{"request":null}\n', encoding="utf-8",
    )
    assert _stop_mode(tmp_path) is None


@pytest.mark.parametrize(
    "control_request", ("stop_now", "stop_after_checkpoint"),
)
def test_outcome_control_accepts_known_requests_with_writer_metadata(
    tmp_path: Path,
    control_request: str,
) -> None:
    (tmp_path / "control.json").write_text(json.dumps({
        "request": control_request,
        "updated_at": "2026-09-03T00:00:00Z",
    }), encoding="utf-8")

    assert _stop_mode(tmp_path) == control_request


@pytest.mark.parametrize(
    "payload",
    (
        '{"request":"pause"}\n',
        '{"request":false}\n',
        '{}\n',
        '[]\n',
    ),
)
def test_outcome_control_rejects_unknown_request_values_and_shapes(
    tmp_path: Path,
    payload: str,
) -> None:
    (tmp_path / "control.json").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="control"):
        _stop_mode(tmp_path)


def test_outcome_control_fails_closed_on_malformed_or_unreadable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_outcome as outcome_module

    with pytest.raises(RuntimeError, match="unreadable"):
        _stop_mode(tmp_path)

    (tmp_path / "control.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        _stop_mode(tmp_path)

    def unreadable(_path: Path):
        raise PermissionError("denied")

    monkeypatch.setattr(outcome_module, "read_json", unreadable)
    with pytest.raises(RuntimeError, match="unreadable"):
        _stop_mode(tmp_path)


def test_checkpoint_stop_forces_off_cadence_loadable_checkpoint_then_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, states, evaluations, checkpoint_updates = _run_stop_lifecycle(
        tmp_path, monkeypatch, "stop_after_checkpoint",
    )

    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "stopping" in states
    assert states[-1] == "stopped"
    assert manifest["state"] == "stopped"
    assert manifest["timesteps"] == 2
    assert manifest["timesteps"] < manifest["target_step"]
    assert manifest["latest_checkpoint_step"] == 1
    assert manifest["latest_training_checkpoint_step"] == 1
    assert evaluations == 2
    assert checkpoint_updates == [0, 1]
    checkpoint = run / manifest["latest_training_checkpoint"]
    loaded = load_outcome_checkpoint(
        checkpoint,
        load_duel_identity_fixture().encoding_hash,
        load_duel_identity_fixture().capacity_hash,
    )
    assert loaded.metadata.update == 1
    assert metrics[-1]["checkpointed"] is True
    assert metrics[-1]["validation"]["wins"] == 2


def test_immediate_stop_after_optimization_keeps_last_validated_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, states, evaluations, checkpoint_updates = _run_stop_lifecycle(
        tmp_path, monkeypatch, "stop_now",
    )

    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "stopping" in states
    assert states[-1] == "stopped"
    assert manifest["state"] == "stopped"
    assert manifest["timesteps"] == 2
    assert manifest["timesteps"] < manifest["target_step"]
    assert manifest["latest_checkpoint_step"] == 0
    assert manifest["latest_training_checkpoint_step"] == 0
    assert evaluations == 1
    assert checkpoint_updates == [0]
    assert metrics[-1]["checkpointed"] is False
    assert metrics[-1]["validation"] is None
    assert "last validated checkpoint is unchanged" in manifest["latest_message"]


def test_tensorboard_failure_is_recorded_without_failing_training(
    tmp_path: Path,
) -> None:
    class BrokenWriter:
        def add_scalar(self, *_args) -> None:
            raise OSError("simulated TensorBoard write failure")

        def close(self) -> None:
            raise OSError("simulated TensorBoard close failure")

    (tmp_path / "run.json").write_text(
        '{"tracker_status":[]}\n', encoding="utf-8",
    )
    telemetry = object.__new__(_Telemetry)
    telemetry.run_dir = tmp_path
    telemetry.writer = BrokenWriter()

    telemetry.record({"train/loss": 1.0}, 1)

    assert telemetry.writer is None
    manifest = json.loads(
        (tmp_path / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["tracker_status"] == [{
        "name": "tensorboard:0",
        "status": "degraded",
        "message": "OSError: simulated TensorBoard write failure",
    }]
