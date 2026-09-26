from dataclasses import replace

import pytest
import torch

from ml_lab.tactical_v3_checkpoint import load_training_resume_checkpoint, save_training_resume_checkpoint
from ml_lab.tactical_v3_curriculum import CurriculumTask, GameplayScore, ScenarioMix, passes_retention_gate
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_training import TrainerConfig, train_offline
from tests.tactical_v3_fixture_support import load_duel_identity_fixture, load_tiny_corpus_fixture


def fixture():
    identity = load_duel_identity_fixture()
    other = replace(identity, scenario_id="other-test-scenario", contract_hash="b" * 64)
    mix = ScenarioMix((CurriculumTask("combat", identity, 3), CurriculumTask("beacon", other, 1)))
    corpus = load_tiny_corpus_fixture()

    def combine(rows):
        return rows + tuple(replace(row, scenario_id=other.scenario_id, contract_hash=other.contract_hash) for row in rows)

    return mix, combine(corpus.train), combine(corpus.validation)


def test_weighted_batches_are_homogeneous_reproducible_and_cycle_exactly():
    mix, train, _ = fixture()
    assert mix.cycle().count(0) == 3
    assert mix.cycle().count(1) == 1
    observed = []
    for epoch in range(4):
        batches = list(mix.batches(train, 4, 42, epoch))
        assert batches == list(mix.batches(train, 4, 42, epoch))
        for task, rows in batches:
            assert len(rows) == 4
            assert {row.contract_hash for row in rows} == {task.identity.contract_hash}
            observed.append(task.name)
    assert observed.count("combat") == 3 * observed.count("beacon")
    assert mix.sha256 != replace(mix, tasks=(replace(mix.tasks[0], weight=2), mix.tasks[1])).sha256


def test_task_provenance_and_encoding_fail_closed():
    mix, train, _ = fixture()
    with pytest.raises(ValueError, match="undeclared"):
        mix.partitions((replace(train[0], contract_hash="c" * 64),))
    with pytest.raises(ValueError, match="provenance"):
        mix.partitions((replace(train[0], encoding_hash="c" * 64),))
    with pytest.raises(ValueError, match="every curriculum task"):
        mix.partitions((train[0],))
    with pytest.raises(ValueError, match="encoding and capacity"):
        ScenarioMix((mix.tasks[0], replace(mix.tasks[1], identity=replace(mix.tasks[1].identity, capacity_hash="c" * 64))))


def test_mixed_training_reports_separate_metrics_and_resumes_exactly(tmp_path):
    torch.set_num_threads(1)
    mix, train, validation = fixture()
    model = TacticalV3ModelConfig(hidden_dim=16, categorical_dim=4, cell_message_rounds=1, relation_rounds=1)
    config = TrainerConfig(seed=227, batch_size=4, max_epochs=3, patience_epochs=3, device="cpu")
    objectives = ObjectiveConfig()
    kwargs = dict(scenario_mix=mix, micro_batch_size=2)
    task_metrics = {}
    steps = []
    full = train_offline(train, validation, model, objectives, config, **kwargs,
        task_step_callback=lambda name, metric: steps.append((name, metric)),
        task_validation_callback=lambda epoch, name, metrics: task_metrics.update({(epoch, name): metrics}))
    for epoch in range(3):
        expected = (3 * task_metrics[epoch, "combat"]["policy"] + task_metrics[epoch, "beacon"]["policy"]) / 4
        assert full.history[epoch].validation["policy"] == expected
    assert {name for name, metric in steps if metric.phase == "train"} == {"combat", "beacon"}
    for phase in ("train", "validation"):
        phase_steps = [metric.global_step for _, metric in steps if metric.phase == phase]
        assert phase_steps == list(range(1, len(phase_steps) + 1))

    class ExitAtCheckpoint(Exception):
        pass

    path = tmp_path / "last.pt"

    def save(state):
        save_training_resume_checkpoint(path, state, identity=mix.tasks[0].identity,
            corpus_sha256="c" * 64, source_model_state_sha256="d" * 64)
        raise ExitAtCheckpoint

    with pytest.raises(ExitAtCheckpoint):
        train_offline(train, validation, model, objectives, config, **kwargs, checkpoint_callback=save)
    load_kwargs = dict(expected_identity=mix.tasks[0].identity, expected_corpus_sha256="c" * 64,
        expected_source_model_state_sha256="d" * 64)
    with pytest.raises(ValueError, match="curriculum changed"):
        load_training_resume_checkpoint(path, **load_kwargs)
    with pytest.raises(ValueError, match="curriculum changed"):
        load_training_resume_checkpoint(path, **load_kwargs, expected_curriculum_sha256="e" * 64)
    state = load_training_resume_checkpoint(path, **load_kwargs, expected_curriculum_sha256=mix.sha256)
    resumed = train_offline(train, validation, model, objectives, config, **kwargs, resume_state=state)
    assert full.history == resumed.history
    for name, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name]), name
    with pytest.raises(ValueError, match="curriculum"):
        train_offline(train, validation, model, objectives, config, resume_state=state, micro_batch_size=2)


def test_gate_rejects_combat_collapse_despite_high_beacon_success():
    baseline = {"combat": GameplayScore((16, 15), (0, 0), (20, 20)),
                "beacon": GameplayScore((0, 0), (20, 20), (20, 20))}
    collapsed = {"combat": GameplayScore((0, 0), (0, 0), (20, 20)),
                 "beacon": GameplayScore((17, 17), (3, 3), (20, 20))}
    assert not passes_retention_gate(baseline, collapsed, primary_task="combat")
    retained = dict(collapsed, combat=baseline["combat"])
    assert passes_retention_gate(baseline, retained, primary_task="combat")
    assert not passes_retention_gate(baseline, baseline, primary_task="combat")
    worse_seat = dict(retained, combat=GameplayScore((20, 14), (0, 0), (20, 20)))
    assert not passes_retention_gate(baseline, worse_seat, primary_task="combat")
    wrong_panel = dict(retained, beacon=GameplayScore((1, 1), (0, 0), (2, 2)))
    with pytest.raises(ValueError, match="panel sizes"):
        passes_retention_gate(baseline, wrong_panel, primary_task="combat")
