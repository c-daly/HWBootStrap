import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ml_lab.tactical_v3_curriculum_run as runner
from ml_lab.io import atomic_write_bytes, atomic_write_json, read_json
from ml_lab.tactical_v3_checkpoint import StructuredCheckpointMetadata, _canonical_json, structured_model_state_sha256, validate_structured_run
from ml_lab.tactical_v3_curriculum import GameplayScore
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from tests.test_tactical_v3_curriculum import fixture


RECIPE = Path(__file__).resolve().parents[1] / "curricula" / "closing-beacon-replay-v1.json"


def test_recipe_is_data_driven_and_keeps_confirmation_separate(tmp_path):
    recipe, config = runner.load_recipe(RECIPE)
    assert config.max_epochs == 50
    assert [task["weight"] for task in recipe["tasks"]] == [3, 1]
    for task in recipe["tasks"]:
        assert len(runner.panel_schedule(task["screen"])) == 40
        assert len(runner.panel_schedule(task["confirmation"])) == 200
    recipe["tasks"][0]["confirmation"] = recipe["tasks"][0]["screen"]
    path = tmp_path / "bad.json"
    atomic_write_json(path, recipe)
    with pytest.raises(ValueError, match="disjoint"):
        runner.load_recipe(path)


@pytest.mark.parametrize("passes", [False, True])
def test_runner_checkpoints_and_only_publishes_after_both_gates(tmp_path, monkeypatch, passes):
    torch.set_num_threads(1)
    recipe, _ = runner.load_recipe(RECIPE)
    recipe["trainer"].update(device="cpu", batch_size=4, max_epochs=1, patience_epochs=1)
    recipe["micro_batch_size"] = 2
    for task in recipe["tasks"]:
        for panel in ("screen", "confirmation"):
            task[panel]["profiles"] = ["standard-3v3"]
            task[panel]["pairs"] = 1
    recipe_path = tmp_path / "recipe.json"
    atomic_write_json(recipe_path, recipe)
    mix, train, validation = fixture()
    model = TacticalV3Policy(TacticalV3ModelConfig(hidden_dim=16, categorical_dim=4, cell_message_rounds=1, relation_rounds=1))
    metadata = StructuredCheckpointMetadata(1, "structured_imitation", mix.tasks[0].identity,
        model.config, ObjectiveConfig(), runner.TrainerConfig(**recipe["trainer"]), "c" * 64,
        structured_model_state_sha256(model), 0, 1.0, "cpu")
    source = SimpleNamespace(model=model, metadata=metadata)
    # Publication keeps the authentic deployment scenario, not another task's identity.
    scenarios = {task.name: runner.semantic_identity_wire(task.identity) for task in mix.tasks}
    scenarios["combat"] = read_json(RECIPE.parents[1] / "config" / "annihilation-structured-imitation-v1.json")
    monkeypatch.setattr(runner, "load_inputs", lambda *a: (source, mix, scenarios, train, validation, []))
    headers = {task["collections"][0]: SimpleNamespace(identity=mix.tasks[index].identity,
        owner_run=tmp_path / task["collections"][0], start_distribution=(("standard-3v3", 10000),))
        for index, task in enumerate(recipe["tasks"])}
    monkeypatch.setattr(runner, "_inspect_reusable_collection", lambda path, *a, **k: headers[path.name])

    class Client:
        def __init__(self, command, **kwargs):
            self.identity = headers[Path(command[-1]).parent.name].identity
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(runner, "TacticalV3GymClient", Client)
    observed = []

    def evaluate(controller, task, identity, panel, *args):
        is_candidate = controller.run_dir.name == "mixed-test"
        observed.append((panel, task["name"], is_candidate))
        if task["name"] == "combat":
            wins = (1, 1) if passes or not is_candidate else (0, 0)
        else:
            wins = (1, 1) if is_candidate else (0, 0)
        return GameplayScore(wins, (0, 0), (1, 1)), []

    monkeypatch.setattr(runner, "evaluate_panel", evaluate)
    server = tmp_path / "server.dll"
    atomic_write_bytes(server, b"fake-evaluation-server")
    args = SimpleNamespace(recipe=recipe_path, runs_root=tmp_path, run_name="mixed-test",
        server=server, preflight=False, resume_from=None, publish=True)
    output = runner.run(args)
    state = read_json(output / "run.json")
    assert state["state"] == "completed"
    assert state["gameplay_gate_passed"] is passes
    assert (output / "training" / "last.pt").is_file()
    assert (output / "training" / "checkpoints" / "best.pt").is_file()
    assert (output / "training" / "task-metrics.jsonl").is_file()
    if passes:
        published = validate_structured_run(tmp_path / "mixed-test-model")
        assert published.metadata.identity == metadata.identity
        assert len(observed) == 8
    else:
        assert not (tmp_path / "mixed-test-model").exists()
        assert all(panel == "screen" for panel, _, _ in observed)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(output / "tensorboard")).Reload()
    assert "tasks/combat/steps/train/policy" in events.Tags()["scalars"]
    assert "tasks/beacon/epoch/validation/policy" in events.Tags()["scalars"]
    with pytest.raises(FileExistsError):
        runner.run(args)
    if not passes:
        source_bytes = (output / "training" / "last.pt").read_bytes()
        args.run_name, args.resume_from, args.publish = "mixed-resume", output, False
        resumed = runner.run(args)
        assert read_json(resumed / "run.json")["state"] == "completed"
        assert (output / "training" / "last.pt").read_bytes() == source_bytes
        assert runner.file_hash(resumed / "training" / "checkpoints" / "best.pt") == runner.file_hash(output / "training" / "checkpoints" / "best.pt")
        state["state"] = "running"
        atomic_write_json(output / "run.json", state)
        args.run_name = "blocked-active-resume"
        with pytest.raises(ValueError, match="active training"):
            runner.run(args)
        assert not (tmp_path / args.run_name).exists()


def test_historical_deep_teacher_archive_is_explicit_and_diagnostics_authenticated(tmp_path):
    import ml_lab.tactical_v3_pilot as pilot
    from tests.test_tactical_v3_pilot import _DaggerClient, _EvaluationPolicy, _identity

    actor = SimpleNamespace(model=_EvaluationPolicy(), metadata=SimpleNamespace(identity=_identity(),
        model_state_sha256="a" * 64, corpus_sha256="b" * 64, best_epoch=3, best_validation_policy_nll=0.125))
    episode = pilot.collect_dagger_game(_DaggerClient(), actor,
        pilot.PilotScheduleItem("train", "standard-3v3", 65_000_000, 0, 0))
    output = pilot.write_dagger_episode(tmp_path / "historical", episode)
    manifest = read_json(output / "episode.json")
    del manifest["actor"]["semantic_identity"]
    teacher = dict(identity="deep-closing-search-v2", search_depth=8, expansion_budget=4096,
                   heuristic_identity="material-plus-closing-v2")
    manifest["teacher"] = teacher
    rows = [json.loads(line) for line in (output / "decisions.jsonl").read_bytes().splitlines()]
    diagnostics = []
    for index, row in enumerate(rows):
        row["example"]["teacher"].update(teacher)
        diagnostics.append(dict(record_index=index, decision_id=row["example"]["decision"]["decision_id"],
            teacher_candidate_id=row["teacher_candidate_id"], completed_search_depth=3,
            actual_expansions=row["example"]["teacher"]["actual_expansions"]))
    record_bytes = b"".join(_canonical_json(row) for row in rows)
    diag_bytes = b"".join(_canonical_json(row) for row in diagnostics)
    manifest["records"]["sha256"] = hashlib.sha256(record_bytes).hexdigest()
    manifest["teacher_diagnostics"] = dict(path="teacher-diagnostics.jsonl", count=len(rows),
        sha256=hashlib.sha256(diag_bytes).hexdigest())
    atomic_write_bytes(output / "decisions.jsonl", record_bytes)
    atomic_write_bytes(output / "teacher-diagnostics.jsonl", diag_bytes)
    atomic_write_bytes(output / "episode.json", _canonical_json(manifest))
    with pytest.raises(ValueError):
        pilot.load_dagger_episode(output, _identity(), oracle_expansion_budget=4096)
    loaded = pilot.load_dagger_episode(output, _identity(), oracle_expansion_budget=4096, allow_historical_teacher=True)
    assert loaded.actor_identity is None  # Do not invent provenance absent from that old schema.
    assert loaded.records[0].example.teacher.identity == "deep-closing-search-v2"
    diagnostics[0]["completed_search_depth"] = 9
    diag_bytes = b"".join(_canonical_json(row) for row in diagnostics)
    manifest["teacher_diagnostics"]["sha256"] = hashlib.sha256(diag_bytes).hexdigest()
    atomic_write_bytes(output / "teacher-diagnostics.jsonl", diag_bytes)
    atomic_write_bytes(output / "episode.json", _canonical_json(manifest))
    with pytest.raises(ValueError, match="diagnostic"):
        pilot.load_dagger_episode(output, _identity(), oracle_expansion_budget=4096, allow_historical_teacher=True)
