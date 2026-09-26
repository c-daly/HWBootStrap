from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import ml_lab.tactical_v3_checkpoint as tactical_v3_checkpoint
from ml_lab.tactical_v3_batching import collate_examples
from ml_lab.tactical_v3_checkpoint import (
    StructuredCheckpointMetadata,
    load_training_resume_checkpoint,
    load_structured_checkpoint,
    publish_structured_run as publish_schema2_run,
    replace_structured_checkpoint,
    save_training_resume_checkpoint,
    save_structured_checkpoint,
    structured_model_state_sha256,
    validate_structured_run,
)
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_schema import parse_spaces, parse_view
from ml_lab.tactical_v3_training import (
    METRIC_KEYS,
    EpochMetrics,
    TrainerConfig,
    TrainingResult,
    TrainingCheckpointState,
    _batch_to_device,
    train_offline,
)
from ml_lab.tactical_v3_corpus import (
    StructuredExample, StructuredTarget, TeacherEvidence, load_corpus,
)
from tests.test_tactical_v3_schema import minimal_view_payload
from tests.tactical_v3_fixture_support import (
    DUEL_IDENTITY_FIXTURE,
    TINY_CORPUS_ROOT,
    load_duel_identity_fixture,
    load_tiny_corpus_fixture,
)

TRAINING_SCENARIO = Path(
    'python/config/annihilation-structured-imitation-v1.json'
)


def publish_case_run(
    run_dir: Path,
    result: TrainingResult,
    corpus: object,
    policy_spaces_path: Path | None = None,
    *,
    training_scenario_path: Path = TRAINING_SCENARIO,
    policy_identity: object | None = None,
) -> Path:
    identity = policy_identity
    if identity is None:
        if policy_spaces_path is None:
            raise AssertionError("policy spaces path is required")
        identity = parse_spaces(json.loads(policy_spaces_path.read_bytes()))
    return publish_schema2_run(
        run_dir,
        result,
        corpus,
        training_scenario_path=training_scenario_path,
        policy_identity=identity,
    )


@dataclass(frozen=True, slots=True)
class CheckpointCase:
    metadata: StructuredCheckpointMetadata
    model: TacticalV3Policy
    examples: tuple
    corpus: object
    scenario: Path
    result: TrainingResult


def model_state_sha256(model: TacticalV3Policy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(cpu.dtype), "shape": list(cpu.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _save_one_epoch_training_resume_checkpoint(
    tmp_path: Path,
) -> tuple[
    Path,
    object,
    object,
    TacticalV3ModelConfig,
    TrainerConfig,
    TrainingCheckpointState,
]:
    corpus = load_tiny_corpus_fixture()
    identity = load_duel_identity_fixture()
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
    )
    config = TrainerConfig(
        seed=227,
        batch_size=4,
        max_epochs=3,
        patience_epochs=3,
        device="cpu",
    )
    captured: list[TrainingCheckpointState] = []

    class SimulatedExit(RuntimeError):
        pass

    def capture(state: TrainingCheckpointState) -> None:
        captured.append(state)
        raise SimulatedExit

    with pytest.raises(SimulatedExit):
        train_offline(
            corpus.train,
            corpus.validation,
            model_config,
            ObjectiveConfig(),
            config,
            checkpoint_callback=capture,
        )

    checkpoint = tmp_path / "last.pt"
    save_training_resume_checkpoint(
        checkpoint,
        captured[0],
        identity=identity,
        corpus_sha256=corpus.identity,
        source_model_state_sha256="a" * 64,
    )
    loaded = load_training_resume_checkpoint(
        checkpoint,
        expected_identity=identity,
        expected_corpus_sha256=corpus.identity,
        expected_source_model_state_sha256="a" * 64,
    )
    return checkpoint, corpus, identity, model_config, config, loaded


def test_training_resume_checkpoint_roundtrip_resumes_exact_trajectory(
    tmp_path: Path,
) -> None:
    _, corpus, _, model_config, config, loaded = (
        _save_one_epoch_training_resume_checkpoint(tmp_path)
    )
    uninterrupted = train_offline(
        corpus.train,
        corpus.validation,
        model_config,
        ObjectiveConfig(),
        config,
    )
    resumed = train_offline(
        corpus.train,
        corpus.validation,
        model_config,
        ObjectiveConfig(),
        config,
        resume_state=loaded,
    )

    assert resumed.history == uninterrupted.history
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert (
        resumed.best_validation_policy_nll
        == uninterrupted.best_validation_policy_nll
    )
    assert resumed.stopped_early == uninterrupted.stopped_early
    assert model_state_sha256(resumed.model) == model_state_sha256(
        uninterrupted.model,
    )
    for name, value in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], value)


def test_training_resume_checkpoint_rejects_optimizer_and_rng_tampering(
    tmp_path: Path,
) -> None:
    checkpoint, corpus, identity, _, _, _ = (
        _save_one_epoch_training_resume_checkpoint(tmp_path)
    )

    for field in ("optimizer_state", "torch_random_state"):
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = raw["state"]
        if field == "optimizer_state":
            optimizer_states = state["optimizer_state"]["state"]
            assert optimizer_states
            parameter_state = next(iter(optimizer_states.values()))
            tensor_name = next(
                name
                for name, value in parameter_state.items()
                if isinstance(value, torch.Tensor)
            )
            changed = parameter_state[tensor_name].clone()
            changed.reshape(-1)[0].add_(1)
            parameter_state[tensor_name] = changed
        else:
            changed = state[field].clone()
            changed[0] = int(changed[0].item()) ^ 1
            state[field] = changed
        tampered = tmp_path / f"tampered-{field}.pt"
        torch.save(raw, tampered)

        with pytest.raises(ValueError, match="state digest changed"):
            load_training_resume_checkpoint(
                tampered,
                expected_identity=identity,
                expected_corpus_sha256=corpus.identity,
                expected_source_model_state_sha256="a" * 64,
            )


def test_live_structured_best_checkpoint_replaces_atomically(
    tmp_path: Path,
) -> None:
    case = case_from_result(train_offline(
        load_tiny_corpus_fixture().train,
        load_tiny_corpus_fixture().validation,
        TacticalV3ModelConfig(),
        ObjectiveConfig(),
        TrainerConfig(
            seed=227, batch_size=4, max_epochs=1,
            patience_epochs=1, device="cpu",
        ),
    ))
    path = tmp_path / "best.pt"

    replace_structured_checkpoint(
        path, case.model, case.metadata, case.examples,
    )
    first = path.read_bytes()
    with torch.no_grad():
        next(case.model.parameters()).add_(0.01)
    replacement_metadata = replace(
        case.metadata,
        model_state_sha256=structured_model_state_sha256(case.model),
        best_epoch=case.metadata.best_epoch + 1,
    )
    replace_structured_checkpoint(
        path, case.model, replacement_metadata, case.examples,
    )

    assert path.read_bytes() != first
    loaded = load_structured_checkpoint(
        path,
        case.metadata.identity.encoding_hash,
        case.metadata.identity.capacity_hash,
    )
    assert loaded.metadata.best_epoch == replacement_metadata.best_epoch
    assert loaded.metadata.model_state_sha256 == (
        replacement_metadata.model_state_sha256
    )


def fixture_logits_and_actions(
    model: TacticalV3Policy,
    examples: tuple,
) -> tuple[tuple[torch.Tensor, ...], tuple[CandidateIdentity, ...]]:
    device = next(model.parameters()).device
    batch = _batch_to_device(
        collate_examples(examples, model.config.horizon_turns),
        device,
    )
    model.eval()
    with torch.no_grad():
        output = model(batch)
    logits: list[torch.Tensor] = []
    actions: list[CandidateIdentity] = []
    for index, example in enumerate(examples):
        valid = batch.candidates.mask[index]
        row = output.candidate_logits[index, valid].detach().clone()
        logits.append(row)
        selected = example.decision.candidates[int(torch.argmax(row).item())]
        actions.append(
            CandidateIdentity(selected.decision_id, selected.candidate_id)
        )
    return tuple(logits), tuple(actions)


def case_from_result(result: TrainingResult) -> CheckpointCase:
    identity = load_duel_identity_fixture()
    corpus = load_tiny_corpus_fixture()
    metadata = StructuredCheckpointMetadata(
        1,
        "structured_imitation",
        identity,
        result.model_config,
        result.objective_config,
        result.trainer_config,
        corpus.identity,
        model_state_sha256(result.model),
        result.best_epoch,
        result.best_validation_policy_nll,
        "cpu",
    )
    return CheckpointCase(
        metadata,
        result.model,
        corpus.validation[:2],
        corpus,
        DUEL_IDENTITY_FIXTURE,
        result,
    )


@pytest.fixture(scope="module")
def trained_cpu_case() -> CheckpointCase:
    corpus = load_tiny_corpus_fixture()
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(4, 8, 16),
    )
    objective_config = ObjectiveConfig()
    trainer_config = TrainerConfig(
        seed=227,
        batch_size=4,
        learning_rate=3e-4,
        max_epochs=1,
        patience_epochs=1,
        gradient_clip_norm=1.0,
        device="cpu",
    )
    return case_from_result(train_offline(
        corpus.train,
        corpus.validation,
        model_config,
        objective_config,
        trainer_config,
    ))


def make_case() -> CheckpointCase:
    identity = load_duel_identity_fixture()
    corpus = load_tiny_corpus_fixture()
    model = TacticalV3Policy(TacticalV3ModelConfig()).eval()
    objective_config = ObjectiveConfig()
    trainer_config = TrainerConfig(seed=0, device="cpu")
    metric_values = MappingProxyType({name: 0.0 for name in METRIC_KEYS})
    history = (
        EpochMetrics(
            epoch=0,
            train=metric_values,
            validation=metric_values,
            validation_policy_nll=0.0,
            improved=True,
        ),
    )
    metadata = StructuredCheckpointMetadata(
        1, "structured_imitation", identity, model.config, objective_config,
        trainer_config, corpus.identity, model_state_sha256(model), 0, 0.0, "cpu",
    )
    result = TrainingResult(
        model=model,
        model_config=model.config,
        objective_config=objective_config,
        trainer_config=trainer_config,
        best_epoch=0,
        best_validation_policy_nll=0.0,
        stopped_early=False,
        history=history,
    )
    return CheckpointCase(metadata, model, corpus.validation[:2], corpus,
                          DUEL_IDENTITY_FIXTURE, result)


def copied_corpus_case(tmp_path: Path) -> CheckpointCase:
    case = make_case()
    copied = tmp_path / "corpus"
    shutil.copytree(TINY_CORPUS_ROOT, copied)
    return replace(
        case,
        corpus=load_corpus(copied, case.metadata.identity),
    )


def test_public_model_state_hash_round_trips_through_cpu_checkpoint(tmp_path: Path) -> None:
    identity = load_duel_identity_fixture()
    view = parse_view(minimal_view_payload(), identity)
    example = StructuredExample(
        1, view.decision, StructuredTarget(0, "win", 0, 1, False),
        TeacherEvidence("bounded-search-v1", 4, 512, 17,
                        "material-plus-pursuit-v1", None),
        identity.scenario_id, identity.contract_hash, identity.encoding_hash,
        identity.capacity_hash, "standard-3v3", 61_000_000, 0,
    )
    model = TacticalV3Policy(TacticalV3ModelConfig()).eval()
    state_hash = structured_model_state_sha256(model)
    metadata = StructuredCheckpointMetadata(
        1, "structured_imitation", identity, model.config, ObjectiveConfig(),
        TrainerConfig(seed=227, device="cpu"), "a" * 64, state_hash, 0, 0.0, "cpu",
    )
    checkpoint = save_structured_checkpoint(
        tmp_path / "best.pt", model, metadata, (example,),
    )
    loaded = load_structured_checkpoint(
        checkpoint, metadata.identity.encoding_hash, metadata.identity.capacity_hash,
    )

    assert metadata.model_state_sha256 == state_hash
    assert loaded.metadata.model_state_sha256 == state_hash
    assert structured_model_state_sha256(loaded.model) == state_hash
    assert next(loaded.model.parameters()).device.type == "cpu"


@pytest.fixture(scope="module")
def published_run_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CheckpointCase, Path]:
    case = make_case()
    root = tmp_path_factory.mktemp("published-run-template")
    return case, publish_case_run(
        root / "run",
        case.result,
        case.corpus,
        case.scenario,
    )


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def rewrite_json(path: Path, mutation) -> None:
    value = json.loads(path.read_bytes())
    mutation(value)
    path.write_bytes(canonical_json_bytes(value))


def rewrite_json_value(path: Path, keys: tuple[str, ...], value: object) -> None:
    payload = json.loads(path.read_bytes())
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_bytes(canonical_json_bytes(payload))


def rewrite_source_scenario(wrapper_path: Path, source_text: str) -> None:
    wrapper = json.loads(wrapper_path.read_bytes())
    wrapper["source_scenario_json"] = source_text
    wrapper["source"]["scenario_sha256"] = hashlib.sha256(
        source_text.encode("utf-8")
    ).hexdigest()
    wrapper_path.write_bytes(canonical_json_bytes(wrapper))


def tree_snapshot(root: Path) -> tuple[tuple[str, int, int, str], ...]:
    rows: list[tuple[str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        rows.append((relative, info.st_mode, info.st_size, digest))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class AdoptionSources:
    case: CheckpointCase
    root: Path
    checkpoint: Path
    collection: Path
    training: Path
    metrics: Path
    scenario: Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_adoption_sources(tmp_path: Path) -> AdoptionSources:
    case = make_case()
    trainer = replace(case.metadata.trainer_config, max_epochs=1, patience_epochs=1)
    case = replace(
        case,
        metadata=replace(case.metadata, trainer_config=trainer),
        result=replace(case.result, trainer_config=trainer),
    )
    root = tmp_path / "source"
    checkpoint = save_structured_checkpoint(
        root / "training" / "checkpoints" / "best.pt",
        case.model, case.metadata, case.examples,
    )

    identity = case.metadata.identity
    collection_value = {
        "schema_version": 1,
        "kind": "tactical-v3-custom-closing-dagger",
        "identity": {
            name: getattr(identity, name)
            for name in (
                "scenario_id", "contract_version", "contract_hash", "encoding_hash",
                "capacity_hash", "environment_kind",
            )
        },
        "schedule": {"train": [], "validation": []},
        "partitions": {"train": [], "validation": []},
        "summary": {
            "train": {"games": 0, "labels": 0, "disagreements": 0, "reasons": {}},
            "validation": {
                "games": 0, "labels": 0, "disagreements": 0, "reasons": {},
            },
        },
    }
    collection = root / "collection.json"
    collection.write_bytes(canonical_json_bytes(collection_value))

    training = root / "training" / "dagger-training.json"
    training.write_bytes(canonical_json_bytes({
        "schema_version": 2,
        "kind": "tactical-v3-custom-closing-dagger-training",
        "actor_checkpoint_sha256": "a" * 64,
        "actor_model_state_sha256": "b" * 64,
        "base_collection_sha256": "c" * 64,
        "prior_manifest_sha256": "d" * 64,
        "closing_manifest_sha256": sha256_file(collection),
        "corpus_sha256": case.metadata.corpus_sha256,
        "definition": {
            "name": "synthetic-adoption-fixture", "train_seed_start": 75_000_000,
            "train_pair_count": 1, "train_label_target": 1,
            "validation_seed_start": 76_000_000, "validation_pair_count": 1,
            "validation_label_target": 1,
        },
        "mixture": {
            "greedy_standard": 0.35, "search_conversion": 0.35,
            "dagger_targeted_seat_0": 0.15, "dagger_targeted_seat_1": 0.15,
        },
        "validation": {"strategy": "synthetic", "examples": 1},
        "execution": {
            "attempt_number": 0, "micro_batch_size": 1,
            "cuda_allocator_config": "",
        },
        "trainer_config": asdict(trainer),
        "result": {
            "checkpoint": "checkpoints/best.pt", "best_epoch": case.metadata.best_epoch,
            "model_state_sha256": case.metadata.model_state_sha256,
            "best_validation_policy_nll": case.metadata.best_validation_policy_nll,
        },
    }))
    metrics = root / "training" / "metrics.jsonl"
    metrics.write_bytes(tactical_v3_checkpoint._metrics_jsonl(case.result.history))
    scenario = root / "scenario.json"
    scenario.write_bytes(TRAINING_SCENARIO.read_bytes())
    return AdoptionSources(
        case, root, checkpoint, collection, training, metrics, scenario
    )


def adopt_sources(
    run_dir: Path,
    source: AdoptionSources,
    *,
    expected_overrides: dict[str, object] | None = None,
) -> Path:
    options: dict[str, object] = {
        "source_checkpoint_path": source.checkpoint,
        "source_collection_path": source.collection,
        "source_training_path": source.training,
        "source_metrics_path": source.metrics,
        "training_scenario_path": source.scenario,
        "expected_identity": source.case.metadata.identity,
        "expected_checkpoint_sha256": sha256_file(source.checkpoint),
        "expected_collection_sha256": sha256_file(source.collection),
        "expected_training_sha256": sha256_file(source.training),
        "expected_metrics_sha256": sha256_file(source.metrics),
        "expected_scenario_sha256": sha256_file(source.scenario),
    }
    options.update(expected_overrides or {})
    return tactical_v3_checkpoint.adopt_structured_run(run_dir, **options)


def clone_published_run(
    template: Path,
    destination: Path,
) -> Path:
    shutil.copytree(template, destination)
    return destination


def create_windows_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        pytest.skip(f"Windows junction creation unavailable: {completed.stderr}")


def test_adopt_structured_run_preserves_checkpoint_and_binds_source_evidence(
    tmp_path: Path,
) -> None:
    sources = create_adoption_sources(tmp_path)
    source_before = tree_snapshot(sources.root)
    run_dir = adopt_sources(tmp_path / "run", sources)

    assert run_dir == tmp_path / "run"
    assert (run_dir / "checkpoints" / "best.pt").read_bytes() == (
        sources.checkpoint.read_bytes()
    )
    assert (run_dir / "metrics.jsonl").read_bytes() == sources.metrics.read_bytes()
    assert tree_snapshot(sources.root) == source_before

    wrapper = json.loads((run_dir / "corpus-manifest.json").read_bytes())
    assert set(wrapper) == {
        "schema_version", "kind", "provenance_scope",
        "dataset_manifest_sha256", "source", "published_scenario_sha256",
        "source_scenario_json", "collection", "training",
    }
    assert wrapper["kind"] == "tactical-v3-adopted-dagger-evidence"
    assert wrapper["provenance_scope"] == "adopted-local-artifact"
    assert wrapper["dataset_manifest_sha256"] == sources.case.metadata.corpus_sha256
    assert wrapper["source"] == {
        f"{name}_sha256": sha256_file(getattr(sources, name))
        for name in ("checkpoint", "collection", "training", "metrics", "scenario")
    }
    assert wrapper["published_scenario_sha256"] == sha256_file(run_dir / "scenario.json")
    assert wrapper["source_scenario_json"] == sources.scenario.read_text(encoding="utf-8")
    assert wrapper["source"]["scenario_sha256"] != wrapper["published_scenario_sha256"]
    assert wrapper["collection"] == json.loads(sources.collection.read_bytes())
    assert wrapper["training"] == json.loads(sources.training.read_bytes())

    loaded = validate_structured_run(run_dir)
    assert loaded.metadata == sources.case.metadata
    assert loaded.fixture.examples == sources.case.examples


def test_adopt_structured_run_rejects_untrusted_or_mismatched_sources(
    tmp_path: Path,
) -> None:
    replay = create_adoption_sources(tmp_path / "replay")
    raw = torch.load(replay.checkpoint, map_location="cpu", weights_only=True)
    raw["inference_fixture"]["selected_identities"][0]["candidate_id"] += 10_000
    torch.save(raw, replay.checkpoint)
    with pytest.raises(ValueError, match="fixture|replay|checkpoint"):
        adopt_sources(tmp_path / "replay-run", replay)

    hashes = create_adoption_sources(tmp_path / "hashes")
    with pytest.raises(ValueError, match="SHA-256|hash"):
        adopt_sources(
            tmp_path / "hash-run",
            hashes,
            expected_overrides={"expected_collection_sha256": "0" * 64},
        )

    identity = create_adoption_sources(tmp_path / "identity")
    with pytest.raises(ValueError, match="identity|checkpoint"):
        adopt_sources(
            tmp_path / "identity-run",
            identity,
            expected_overrides={
                "expected_identity": replace(
                    identity.case.metadata.identity,
                    encoding_hash="0" * 64,
                )
            },
        )

    stale = create_adoption_sources(tmp_path / "stale")
    raw = torch.load(stale.checkpoint, map_location="cpu", weights_only=True)
    raw["metadata"]["trainer_config"]["max_epochs"] = 3
    raw["metadata"]["trainer_config"]["patience_epochs"] = 1
    torch.save(raw, stale.checkpoint)
    metric_values = {name: 0.0 for name in METRIC_KEYS}
    stale.metrics.write_bytes(b"".join(
        canonical_json_bytes({
            "epoch": epoch,
            "train": metric_values,
            "validation": {**metric_values, "policy": policy_nll},
            "validation_policy_nll": policy_nll,
            "improved": epoch == 0,
        })
        for epoch, policy_nll in enumerate((0.0, 1.0, 1.0))
    ))
    with pytest.raises(ValueError, match="patience|early stopping"):
        adopt_sources(tmp_path / "stale-run", stale)

    assert not tuple(tmp_path.glob("*-run"))


def test_adopt_structured_run_is_atomic_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    sources = create_adoption_sources(tmp_path)
    with pytest.raises(ValueError, match="source evidence directory"):
        adopt_sources(sources.root / "published-run", sources)
    assert not (sources.root / "published-run").exists()

    run_dir = adopt_sources(tmp_path / "run", sources)
    run_before = tree_snapshot(run_dir)
    source_before = tree_snapshot(sources.root)

    with pytest.raises(FileExistsError):
        adopt_sources(run_dir, sources)

    assert tree_snapshot(run_dir) == run_before
    assert tree_snapshot(sources.root) == source_before
    assert {path.name for path in tmp_path.iterdir()} == {"run", "source"}


def test_validate_structured_run_rejects_adopted_evidence_tamper(
    tmp_path: Path,
) -> None:
    sources = create_adoption_sources(tmp_path)
    source_before = tree_snapshot(sources.root)
    template = adopt_sources(tmp_path / "run", sources)

    hash_run = clone_published_run(template, tmp_path / "hash-tamper")
    rewrite_json_value(
        hash_run / "corpus-manifest.json",
        ("source", "scenario_sha256"),
        "0" * 64,
    )
    with pytest.raises(ValueError, match="scenario|hash"):
        validate_structured_run(hash_run)

    semantic_run = clone_published_run(template, tmp_path / "semantic-tamper")
    value = json.loads(sources.scenario.read_bytes())
    value["id"] = "tampered-scenario"
    rewrite_source_scenario(
        semantic_run / "corpus-manifest.json",
        canonical_json_bytes(value).decode("utf-8"),
    )
    with pytest.raises(ValueError, match="scenario|identity"):
        validate_structured_run(semantic_run)

    typed_run = clone_published_run(template, tmp_path / "typed-tamper")
    rewrite_json_value(typed_run / "scenario.json", ("board", "width"), 13.0)
    rewrite_json_value(
        typed_run / "corpus-manifest.json",
        ("published_scenario_sha256",),
        sha256_file(typed_run / "scenario.json"),
    )
    with pytest.raises(ValueError, match="scenario"):
        validate_structured_run(typed_run)

    metrics_run = clone_published_run(template, tmp_path / "metrics-tamper")
    path = metrics_run / "metrics.jsonl"
    path.write_bytes(path.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="metrics|hash|JSON"):
        validate_structured_run(metrics_run)

    assert tree_snapshot(sources.root) == source_before


def test_training_result_preserves_exact_nondefault_configs() -> None:
    corpus = load_tiny_corpus_fixture()
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(3, 6),
    )
    objective_config = ObjectiveConfig(
        outcome_coefficient=0.1,
        horizon_coefficient=0.1,
        remaining_turns_coefficient=0.05,
    )
    trainer_config = TrainerConfig(
        seed=0,
        batch_size=4,
        learning_rate=1e-3,
        max_epochs=1,
        patience_epochs=1,
        gradient_clip_norm=0.5,
        device="cpu",
    )

    result = train_offline(
        corpus.train,
        corpus.validation,
        model_config,
        objective_config,
        trainer_config,
    )

    assert result.model_config is model_config
    assert result.objective_config is objective_config
    assert result.trainer_config is trainer_config


def test_publish_uses_result_provenance_without_silent_defaults(tmp_path: Path) -> None:
    case = make_case()
    run_dir = publish_case_run(
        tmp_path / "run",
        case.result,
        case.corpus,
        case.scenario,
    )

    loaded = validate_structured_run(run_dir)

    assert loaded.metadata.model_config == case.result.model_config
    assert loaded.metadata.objective_config == case.result.objective_config
    assert loaded.metadata.trainer_config == case.result.trainer_config
    assert loaded.metadata.trainer_config.seed == 0
    assert loaded.metadata.published_device == "cpu"


def test_publish_splits_training_scenario_from_policy_identity(
    tmp_path: Path,
) -> None:
    case = make_case()
    run_dir = publish_case_run(
        tmp_path / 'run',
        case.result,
        case.corpus,
        training_scenario_path=TRAINING_SCENARIO,
        policy_identity=case.metadata.identity,
    )

    assert json.loads((run_dir / 'scenario.json').read_bytes()) == json.loads(
        TRAINING_SCENARIO.read_bytes()
    )
    policy_payload = json.loads(
        (run_dir / 'policy-identity.json').read_bytes()
    )
    assert parse_spaces(policy_payload) == case.metadata.identity
    manifest = json.loads((run_dir / 'run.json').read_bytes())
    assert manifest['schema_version'] == 2
    assert manifest['policy_identity'] == 'policy-identity.json'
    assert validate_structured_run(run_dir).metadata.identity == case.metadata.identity


@pytest.mark.parametrize(
    "mutation",
    ("missing", "repointed", "mismatched_identity"),
)
def test_validate_rejects_invalid_policy_identity_sidecar(
    tmp_path: Path, mutation: str,
) -> None:
    case = make_case()
    run_dir = publish_case_run(
        tmp_path / "run", case.result, case.corpus, case.scenario
    )
    policy_path = run_dir / "policy-identity.json"
    if mutation == "missing":
        policy_path.unlink()
    elif mutation == "repointed":
        manifest_path = run_dir / "run.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["policy_identity"] = "scenario.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    else:
        policy_path.write_bytes(
            (Path(__file__).parent / "fixtures" / "tactical_v3" /
             "seed-41-spaces.json").read_bytes()
        )

    with pytest.raises(
        ValueError,
        match="inventory|policy identity|policy_identity|checkpoint identity",
    ):
        validate_structured_run(run_dir)


@pytest.mark.parametrize(
    "mutation",
    ("extra_field", "duplicate_key", "noncanonical", "semantic_tamper"),
)
def test_publish_rejects_unauthenticated_corpus_manifest_bytes(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = copied_corpus_case(tmp_path)
    manifest_path = case.corpus.root / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    if mutation == "extra_field":
        manifest["review_tamper"] = True
        changed = canonical_json_bytes(manifest)
    elif mutation == "duplicate_key":
        changed = (
            b'{"identity":"'
            + manifest["identity"].encode("ascii")
            + b'",'
            + original[1:]
        )
    elif mutation == "noncanonical":
        changed = json.dumps(manifest, indent=2).encode("utf-8")
    else:
        manifest["scenario_id"] = "tampered-scenario"
        changed = canonical_json_bytes(manifest)
    manifest_path.write_bytes(changed)

    with pytest.raises(ValueError, match="manifest|corpus"):
        publish_case_run(
            tmp_path / "run",
            case.result,
            case.corpus,
            case.scenario,
        )


def test_publish_copies_exact_authenticated_manifest_without_mutating_corpus(
    tmp_path: Path,
) -> None:
    case = copied_corpus_case(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in case.corpus.root.iterdir()
    }

    run_dir = publish_case_run(
        tmp_path / "run",
        case.result,
        case.corpus,
        case.scenario,
    )

    assert (run_dir / "corpus-manifest.json").read_bytes() == before["manifest.json"]
    assert {
        path.name: path.read_bytes()
        for path in case.corpus.root.iterdir()
    } == before


def test_published_metrics_are_canonical_exact_and_cross_file_consistent(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    case, template = published_run_template
    run_dir = tmp_path / "run"
    shutil.copytree(template, run_dir)
    metrics_bytes = (run_dir / "metrics.jsonl").read_bytes()
    rows = tuple(
        json.loads(line)
        for line in metrics_bytes.splitlines(keepends=True)
    )

    assert metrics_bytes == b"".join(canonical_json_bytes(row) for row in rows)
    assert tuple(row["epoch"] for row in rows) == (0,)
    assert rows[0]["validation_policy_nll"] == case.result.best_validation_policy_nll
    assert rows[0]["validation"]["policy"] == case.result.best_validation_policy_nll
    assert rows[0]["improved"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("not_json", "metrics"),
        ("duplicate_key", "duplicate"),
        ("noncanonical", "canonical"),
        ("extra_key", "fields"),
        ("missing_key", "fields"),
        ("epoch_bool", "epoch"),
        ("epoch_float", "epoch"),
        ("epoch_gap", "contiguous"),
        ("improved_int", "improved"),
        ("metric_int", "finite"),
        ("metric_nan", "finite"),
        ("metric_keys", "fields"),
        ("validation_nll_disagrees", "validation policy"),
        ("best_disagrees", "best"),
    ),
)
def test_validate_rejects_corrupt_metrics_jsonl(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
    mutation: str,
    message: str,
) -> None:
    _, template = published_run_template
    run_dir = tmp_path / "run"
    shutil.copytree(template, run_dir)
    path = run_dir / "metrics.jsonl"
    row = json.loads(path.read_bytes())
    if mutation == "not_json":
        changed = b"not-json\n"
    elif mutation == "duplicate_key":
        changed = b'{"epoch":0,' + canonical_json_bytes(row)[1:]
    elif mutation == "noncanonical":
        changed = (
            json.dumps(row, sort_keys=True, separators=(", ", ": ")) + "\n"
        ).encode("utf-8")
    else:
        if mutation == "extra_key":
            row["extra"] = 1
        elif mutation == "missing_key":
            del row["train"]
        elif mutation == "epoch_bool":
            row["epoch"] = True
        elif mutation == "epoch_float":
            row["epoch"] = 0.0
        elif mutation == "epoch_gap":
            row["epoch"] = 1
        elif mutation == "improved_int":
            row["improved"] = 1
        elif mutation == "metric_int":
            row["train"]["policy"] = 0
        elif mutation == "metric_nan":
            changed = canonical_json_bytes(row).replace(
                b'"policy":0.0',
                b'"policy":NaN',
                1,
            )
        elif mutation == "metric_keys":
            row["validation"]["extra"] = 0.0
        elif mutation == "validation_nll_disagrees":
            row["validation_policy_nll"] = 1.0
        elif mutation == "best_disagrees":
            row["validation_policy_nll"] = 1.0
            row["validation"]["policy"] = 1.0
        if mutation != "metric_nan":
            changed = canonical_json_bytes(row)
    path.write_bytes(changed)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_structured_run(run_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate_key", "duplicate"),
        ("noncanonical", "canonical"),
        ("schema_bool", "schema_version"),
        ("best_epoch_float", "best_epoch"),
        ("latest_step_float", "latest_checkpoint_step"),
        ("best_nll_int", "best_validation_policy_nll"),
        ("extra_contract", "contract.*fields"),
        ("checkpoint_epoch_disagrees", "best epoch"),
        ("checkpoint_nll_disagrees", "best"),
    ),
)
def test_validate_rejects_noncanonical_wrong_typed_or_inconsistent_run_json(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
    mutation: str,
    message: str,
) -> None:
    _, template = published_run_template
    run_dir = tmp_path / "run"
    shutil.copytree(template, run_dir)
    path = run_dir / "run.json"
    value = json.loads(path.read_bytes())
    if mutation == "duplicate_key":
        changed = b'{"schema_version":1,' + path.read_bytes()[1:]
    elif mutation == "noncanonical":
        changed = json.dumps(value, indent=2).encode("utf-8")
    else:
        if mutation == "schema_bool":
            value["schema_version"] = True
        elif mutation == "best_epoch_float":
            value["best_epoch"] = 0.0
        elif mutation == "latest_step_float":
            value["latest_checkpoint_step"] = 0.0
        elif mutation == "best_nll_int":
            value["best_validation_policy_nll"] = 0
        elif mutation == "extra_contract":
            value["contract"]["extra"] = "tamper"
        elif mutation == "checkpoint_epoch_disagrees":
            value["best_epoch"] = 1
            value["latest_checkpoint_step"] = 1
        elif mutation == "checkpoint_nll_disagrees":
            value["best_validation_policy_nll"] = 1.0
        changed = canonical_json_bytes(value)
    path.write_bytes(changed)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_structured_run(run_dir)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_top",
        "missing_top",
        "extra_checkpoint",
        "top_file_is_directory",
        "checkpoint_dir_is_file",
        "checkpoint_is_directory",
    ),
)
def test_validate_rejects_recursive_inventory_and_entry_kind_mutations(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
    mutation: str,
) -> None:
    _, template = published_run_template
    run_dir = clone_published_run(template, tmp_path / "run")
    if mutation == "extra_top":
        (run_dir / "extra.json").write_bytes(b"{}\n")
    elif mutation == "missing_top":
        (run_dir / "scenario.json").rename(tmp_path / "removed-scenario.json")
    elif mutation == "extra_checkpoint":
        (run_dir / "checkpoints" / "extra.pt").write_bytes(b"extra")
    elif mutation == "top_file_is_directory":
        (run_dir / "scenario.json").rename(tmp_path / "scenario-file")
        (run_dir / "scenario.json").mkdir()
    elif mutation == "checkpoint_dir_is_file":
        (run_dir / "checkpoints").rename(tmp_path / "checkpoint-directory")
        (run_dir / "checkpoints").write_bytes(b"not-a-directory")
    elif mutation == "checkpoint_is_directory":
        (run_dir / "checkpoints" / "best.pt").rename(tmp_path / "checkpoint-file")
        (run_dir / "checkpoints" / "best.pt").mkdir()

    with pytest.raises(ValueError, match="inventory|plain|checkpoint"):
        validate_structured_run(run_dir)


@pytest.mark.parametrize("entry", ("root", "scenario", "checkpoints", "checkpoint"))
def test_validate_rejects_posix_symlinks_and_windows_reparse_links(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
    entry: str,
) -> None:
    _, template = published_run_template
    run_dir = clone_published_run(template, tmp_path / "real-run")
    if entry == "root":
        link = tmp_path / "run-link"
        target = run_dir
        target_is_directory = True
    elif entry == "scenario":
        link = run_dir / "scenario.json"
        target = tmp_path / "scenario-target.json"
        link.rename(target)
        target_is_directory = False
    elif entry == "checkpoints":
        link = run_dir / "checkpoints"
        target = tmp_path / "checkpoints-target"
        link.rename(target)
        target_is_directory = True
    else:
        link = run_dir / "checkpoints" / "best.pt"
        target = tmp_path / "checkpoint-target.pt"
        link.rename(target)
        target_is_directory = False
    try:
        os.symlink(target, link, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="plain|inventory|reparse|checkpoint"):
        validate_structured_run(link if entry == "root" else run_dir)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_validate_rejects_real_windows_junction(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    _, template = published_run_template
    target = clone_published_run(template, tmp_path / "junction-target")
    junction = tmp_path / "run-junction"
    create_windows_junction(junction, target)
    try:
        with pytest.raises(ValueError, match="plain|reparse"):
            validate_structured_run(junction)
    finally:
        os.rmdir(junction)


@pytest.mark.parametrize("component", (".", ".."))
def test_validate_rejects_dot_path_components_at_public_boundary(
    tmp_path: Path,
    component: str,
) -> None:
    run_path = f"{tmp_path}{os.sep}{component}{os.sep}run"

    with pytest.raises(ValueError, match="dot path component|traversal"):
        validate_structured_run(run_path)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_validate_rejects_missing_dotdot_windows_junction_bypass(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    _, template = published_run_template
    target = tmp_path / "junction-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    clone_published_run(template, ordinary_parent / "run")
    junction = tmp_path / "ancestor-junction"
    create_windows_junction(junction, target)
    candidate = (
        tmp_path / "missing" / ".." / junction.name / "ordinary-parent" / "run"
    )
    try:
        assert candidate.exists()
        with pytest.raises(ValueError, match="dot path component|traversal"):
            validate_structured_run(candidate)
    finally:
        os.rmdir(junction)


def test_validate_rejects_nested_symlink_ancestor(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    _, template = published_run_template
    target = tmp_path / "symlink-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    clone_published_run(template, ordinary_parent / "run")
    symlink = tmp_path / "ancestor-symlink"
    try:
        os.symlink(target, symlink, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    try:
        with pytest.raises(ValueError, match="ancestor|plain|reparse"):
            validate_structured_run(symlink / "ordinary-parent" / "run")
    finally:
        symlink.unlink()


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_validate_rejects_nested_windows_junction_ancestor(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    _, template = published_run_template
    target = tmp_path / "junction-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    clone_published_run(template, ordinary_parent / "run")
    junction = tmp_path / "ancestor-junction"
    create_windows_junction(junction, target)
    try:
        with pytest.raises(ValueError, match="ancestor|plain|reparse"):
            validate_structured_run(junction / "ordinary-parent" / "run")
    finally:
        os.rmdir(junction)


def test_validate_rejects_checkpoint_escape_and_preserves_run_bytes(
    tmp_path: Path,
    published_run_template: tuple[CheckpointCase, Path],
) -> None:
    _, template = published_run_template
    valid = clone_published_run(template, tmp_path / "valid-run")
    before = tree_snapshot(valid)

    validate_structured_run(valid)

    assert tree_snapshot(valid) == before
    escaped = clone_published_run(template, tmp_path / "escaped-run")
    rewrite_json(
        escaped / "run.json",
        lambda value: value.__setitem__("latest_checkpoint", "../outside.pt"),
    )
    with pytest.raises(ValueError, match="relative|latest_checkpoint"):
        validate_structured_run(escaped)


def test_publish_injected_failure_cleans_temporary_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_checkpoint as module

    case = make_case()
    run_dir = tmp_path / "failed-run"

    def fail() -> None:
        raise RuntimeError("injected after_checkpoint")

    monkeypatch.setattr(module, "_after_checkpoint_written", fail)
    with pytest.raises(RuntimeError, match="injected after_checkpoint"):
        publish_case_run(
            run_dir,
            case.result,
            case.corpus,
            case.scenario,
        )

    assert not run_dir.exists()
    assert not tuple(tmp_path.glob(".failed-run.tmp-*"))


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_publish_rejects_windows_reparse_parent(
    tmp_path: Path,
) -> None:
    case = make_case()
    target = tmp_path / "real-parent"
    target.mkdir()
    junction = tmp_path / "parent-junction"
    create_windows_junction(junction, target)
    try:
        with pytest.raises(ValueError, match="plain|reparse"):
            publish_case_run(
                junction / "run",
                case.result,
                case.corpus,
                case.scenario,
            )
    finally:
        os.rmdir(junction)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_publish_rejects_missing_dotdot_windows_junction_bypass_without_writing_target(
    tmp_path: Path,
) -> None:
    case = make_case()
    target = tmp_path / "junction-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    junction = tmp_path / "ancestor-junction"
    create_windows_junction(junction, target)
    candidate = (
        tmp_path / "missing" / ".." / junction.name / "ordinary-parent" / "run"
    )
    try:
        assert candidate.parent.exists()
        with pytest.raises(ValueError, match="dot path component|traversal"):
            publish_case_run(
                candidate,
                case.result,
                case.corpus,
                case.scenario,
            )
        assert not (ordinary_parent / "run").exists()
    finally:
        os.rmdir(junction)


def test_publish_rejects_nested_symlink_ancestor_without_writing_target(
    tmp_path: Path,
) -> None:
    case = make_case()
    target = tmp_path / "symlink-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    symlink = tmp_path / "ancestor-symlink"
    try:
        os.symlink(target, symlink, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    try:
        with pytest.raises(ValueError, match="ancestor|plain|reparse"):
            publish_case_run(
                symlink / "ordinary-parent" / "run",
                case.result,
                case.corpus,
                case.scenario,
            )
        assert not (ordinary_parent / "run").exists()
    finally:
        symlink.unlink()


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows reparse points")
def test_publish_rejects_nested_windows_junction_ancestor_without_writing_target(
    tmp_path: Path,
) -> None:
    case = make_case()
    target = tmp_path / "junction-target"
    ordinary_parent = target / "ordinary-parent"
    ordinary_parent.mkdir(parents=True)
    junction = tmp_path / "ancestor-junction"
    create_windows_junction(junction, target)
    try:
        with pytest.raises(ValueError, match="ancestor|plain|reparse"):
            publish_case_run(
                junction / "ordinary-parent" / "run",
                case.result,
                case.corpus,
                case.scenario,
            )
        assert not (ordinary_parent / "run").exists()
    finally:
        os.rmdir(junction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_training_publication_matches_cpu_with_declared_tolerance(
    tmp_path: Path,
) -> None:
    corpus = load_tiny_corpus_fixture()
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(4, 8, 16),
    )
    result = train_offline(
        corpus.train,
        corpus.validation,
        model_config,
        ObjectiveConfig(),
        TrainerConfig(
            seed=227,
            batch_size=4,
            learning_rate=3e-4,
            max_epochs=1,
            patience_epochs=1,
            gradient_clip_norm=1.0,
            device="cuda",
        ),
    )
    fixture_examples = corpus.validation[:2]
    expected_logits, expected_actions = fixture_logits_and_actions(
        result.model,
        fixture_examples,
    )

    run_dir = publish_case_run(
        tmp_path / "run",
        result,
        corpus,
        DUEL_IDENTITY_FIXTURE,
    )
    loaded = validate_structured_run(run_dir)
    actual_logits, actual_actions = fixture_logits_and_actions(
        loaded.model,
        loaded.fixture.examples,
    )

    assert next(result.model.parameters()).device.type == "cuda"
    assert next(loaded.model.parameters()).device.type == "cpu"
    assert loaded.metadata.trainer_config == result.trainer_config
    assert loaded.metadata.published_device == "cpu"
    for actual, expected in zip(actual_logits, expected_logits, strict=True):
        torch.testing.assert_close(
            actual,
            expected.cpu(),
            rtol=1e-5,
            atol=1e-6,
        )
    assert actual_actions == expected_actions


@pytest.mark.skipif(os.name != "nt", reason="Windows file leases enforce no-write/no-delete")
def test_publish_holds_corpus_file_leases_through_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_checkpoint as module

    case = copied_corpus_case(tmp_path)
    original = module._validate_partitions
    attacked = False

    def attack(train: tuple, validation: tuple) -> None:
        nonlocal attacked
        original(train, validation)
        attacked = True
        with pytest.raises(PermissionError):
            (case.corpus.root / "manifest.json").write_bytes(b"{}\n")

    monkeypatch.setattr(module, "_validate_partitions", attack)

    assert publish_case_run(
        tmp_path / "run",
        case.result,
        case.corpus,
        case.scenario,
    ) == tmp_path / "run"
    assert attacked


def test_publish_rejects_result_model_config_mismatch(tmp_path: Path) -> None:
    case = make_case()
    bad_result = replace(
        case.result,
        model_config=replace(case.result.model_config, horizon_turns=(3, 6)),
    )

    with pytest.raises(ValueError, match="model config"):
        publish_case_run(
            tmp_path / "run",
            bad_result,
            case.corpus,
            case.scenario,
        )


@pytest.mark.parametrize("mutation", ("model_config", "model_state_sha256"))
def test_save_rejects_inconsistent_supplied_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = make_case()
    if mutation == "model_config":
        metadata = replace(
            case.metadata,
            model_config=replace(case.metadata.model_config, horizon_turns=(3, 6)),
        )
        message = "model config"
    else:
        metadata = replace(case.metadata, model_state_sha256="f" * 64)
        message = "model state SHA-256"

    with pytest.raises(ValueError, match=message):
        save_structured_checkpoint(
            tmp_path / "model.pt",
            case.model,
            metadata,
            case.examples,
        )


def test_save_load_is_weights_only_cpu_and_replays_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = make_case()
    path = tmp_path / "model.pt"
    save_structured_checkpoint(path, case.model, case.metadata, case.examples)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"format_version", "metadata", "state_dict", "inference_fixture"}
    assert all(value.device.type == "cpu" and value.is_contiguous()
               for value in raw["state_dict"].values())
    original = torch.load
    calls: list[dict[str, object]] = []

    def spy(*args: object, **kwargs: object) -> object:
        calls.append({key: kwargs[key] for key in ("map_location", "weights_only")})
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    loaded = load_structured_checkpoint(path, case.metadata.identity.encoding_hash,
                                        case.metadata.identity.capacity_hash)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert loaded.metadata.model_state_sha256 != "0" * 64
    assert next(loaded.model.parameters()).device.type == "cpu"
    assert loaded.fixture.examples == case.examples


def test_checkpoint_contains_only_recursive_plain_values_and_state_tensors(
    tmp_path: Path,
) -> None:
    case = make_case()
    path = save_structured_checkpoint(
        tmp_path / "model.pt",
        case.model,
        case.metadata,
        case.examples,
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)

    def assert_plain(value: object) -> None:
        if value is None or type(value) in (str, int, float, bool):
            if type(value) is float:
                assert torch.isfinite(torch.tensor(value))
            return
        if type(value) is list:
            for item in value:
                assert_plain(item)
            return
        assert type(value) is dict
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_plain(item)

    assert set(raw) == {
        "format_version",
        "metadata",
        "state_dict",
        "inference_fixture",
    }
    assert_plain(raw["format_version"])
    assert_plain(raw["metadata"])
    assert_plain(raw["inference_fixture"])
    assert all(
        type(name) is str
        and isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.is_contiguous()
        for name, value in raw["state_dict"].items()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_top", "checkpoint fields"),
        ("missing_metadata", "metadata fields"),
        ("missing_trainer", "trainer_config"),
        ("missing_fixture", "selected_identities"),
        ("nested_tuple", "non-whitelisted"),
    ),
)
def test_load_rejects_missing_nested_keys_and_nonplain_container_types(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    case = make_case()
    source = save_structured_checkpoint(
        tmp_path / "source.pt",
        case.model,
        case.metadata,
        case.examples,
    )
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if mutation == "missing_top":
        del raw["state_dict"]
    elif mutation == "missing_metadata":
        del raw["metadata"]["objective_config"]
    elif mutation == "missing_trainer":
        del raw["metadata"]["trainer_config"]["seed"]
    elif mutation == "missing_fixture":
        del raw["inference_fixture"]["selected_identities"][0]["candidate_id"]
    elif mutation == "nested_tuple":
        raw["metadata"]["model_config"]["horizon_turns"] = (4, 8, 16)
    changed = tmp_path / f"{mutation}.pt"
    torch.save(raw, changed)

    with pytest.raises((TypeError, ValueError), match=message):
        load_structured_checkpoint(
            changed,
            case.metadata.identity.encoding_hash,
            case.metadata.identity.capacity_hash,
        )


def test_two_cpu_saves_are_semantically_identical_after_strict_load(
    tmp_path: Path,
    trained_cpu_case: CheckpointCase,
) -> None:
    loaded = []
    for name in ("a.pt", "b.pt"):
        path = save_structured_checkpoint(
            tmp_path / name,
            trained_cpu_case.model,
            trained_cpu_case.metadata,
            trained_cpu_case.examples,
        )
        loaded.append(load_structured_checkpoint(
            path,
            trained_cpu_case.metadata.identity.encoding_hash,
            trained_cpu_case.metadata.identity.capacity_hash,
        ))

    assert loaded[0].metadata == loaded[1].metadata
    assert loaded[0].fixture == loaded[1].fixture
    assert tuple(loaded[0].model.state_dict()) == tuple(loaded[1].model.state_dict())
    for name, value in loaded[0].model.state_dict().items():
        torch.testing.assert_close(
            value,
            loaded[1].model.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )


def test_cpu_save_load_preserves_logits_and_actions_exactly(
    tmp_path: Path,
    trained_cpu_case: CheckpointCase,
) -> None:
    expected_logits, expected_actions = fixture_logits_and_actions(
        trained_cpu_case.model,
        trained_cpu_case.examples,
    )
    path = save_structured_checkpoint(
        tmp_path / "model.pt",
        trained_cpu_case.model,
        trained_cpu_case.metadata,
        trained_cpu_case.examples,
    )
    loaded = load_structured_checkpoint(
        path,
        trained_cpu_case.metadata.identity.encoding_hash,
        trained_cpu_case.metadata.identity.capacity_hash,
    )
    actual_logits, actual_actions = fixture_logits_and_actions(
        loaded.model,
        loaded.fixture.examples,
    )

    for actual, expected in zip(actual_logits, expected_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual_actions == expected_actions


@pytest.mark.parametrize("field, value, message", [
    ("unknown", 1, "checkpoint fields"),
    ("best_epoch", True, "metadata.best_epoch"),
])
def test_load_rejects_tampered_values(tmp_path: Path, field: str, value: object, message: str) -> None:
    case = make_case()
    path = tmp_path / "model.pt"
    save_structured_checkpoint(path, case.model, case.metadata, case.examples)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if field == "unknown":
        raw[field] = value
    else:
        raw["metadata"][field] = value
    bad = tmp_path / "bad.pt"
    torch.save(raw, bad)
    with pytest.raises((TypeError, ValueError), match=message):
        load_structured_checkpoint(bad, case.metadata.identity.encoding_hash,
                                   case.metadata.identity.capacity_hash)


def test_load_rejects_hash_and_contract_mismatch(tmp_path: Path) -> None:
    case = make_case()
    path = tmp_path / "model.pt"
    save_structured_checkpoint(path, case.model, case.metadata, case.examples)
    with pytest.raises(ValueError, match="encoding hash"):
        load_structured_checkpoint(path, "0" * 64, case.metadata.identity.capacity_hash)
    with pytest.raises(ValueError, match="capacity hash"):
        load_structured_checkpoint(path, case.metadata.identity.encoding_hash, "0" * 64)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    first = next(iter(raw["state_dict"]))
    raw["state_dict"][first] = raw["state_dict"][first] + 1
    bad = tmp_path / "state-bad.pt"
    torch.save(raw, bad)
    with pytest.raises(ValueError, match="model state SHA-256"):
        load_structured_checkpoint(bad, case.metadata.identity.encoding_hash,
                                   case.metadata.identity.capacity_hash)


def test_publish_validates_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    case = make_case()
    run_dir = tmp_path / "run"
    assert publish_case_run(run_dir, case.result, case.corpus, case.scenario) == run_dir
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["evidence_status"] == "unsealed-experimental"
    assert manifest["config"]["algorithm"] == "structured_imitation"
    assert validate_structured_run(run_dir).fixture.examples == case.corpus.validation[:2]
    with pytest.raises(FileExistsError):
        publish_case_run(run_dir, case.result, case.corpus, case.scenario)
    manifest["dataset_manifest_sha256"] = "0" * 64
    (run_dir / "run.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="corpus SHA-256"):
        validate_structured_run(run_dir)

def test_train_cli_calls_real_sequence_then_only_publisher_owns_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import run_tactical_v3_imitation as cli

    case = make_case()
    calls: list[str] = []
    scenario_payload = json.loads(TRAINING_SCENARIO.read_text(encoding="utf-8"))
    policy_payload = json.loads(case.scenario.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    expected_model, expected_objective, expected_trainer = (
        cli._smoke_training_configs(0, "cpu")
    )

    def fake_parse(payload: object) -> object:
        assert payload == policy_payload
        calls.append("parse")
        return case.metadata.identity

    def fake_load(root: Path, expected: object) -> object:
        assert root == TINY_CORPUS_ROOT
        assert expected == case.metadata.identity
        calls.append("load")
        return case.corpus

    def fake_train(
        train_examples: tuple, validation_examples: tuple, model_config: TacticalV3ModelConfig,
        objective_config: ObjectiveConfig, trainer_config: TrainerConfig,
    ) -> TrainingResult:
        assert train_examples == case.corpus.train
        assert validation_examples == case.corpus.validation
        assert model_config == expected_model
        assert objective_config == expected_objective
        assert trainer_config == expected_trainer
        calls.append("train")
        return case.result

    def fake_publish(
        destination: Path,
        result: TrainingResult,
        corpus: object,
        *,
        training_scenario_path: Path,
        policy_identity: object,
    ) -> Path:
        assert destination == run_dir
        assert result is case.result
        assert corpus is case.corpus
        assert training_scenario_path == TRAINING_SCENARIO
        assert policy_identity == case.metadata.identity
        calls.append("publish")
        return destination

    monkeypatch.setattr(cli, "parse_spaces", fake_parse)
    monkeypatch.setattr(cli, "load_corpus", fake_load)
    monkeypatch.setattr(cli, "train_offline", fake_train)
    monkeypatch.setattr(cli, "publish_structured_run", fake_publish)
    assert cli.main([
        "train", "--corpus", str(TINY_CORPUS_ROOT),
        "--scenario", str(TRAINING_SCENARIO),
        "--policy-spaces", str(case.scenario),
        "--run-dir", str(run_dir), "--seed", "0", "--device", "cpu",
    ]) == 0
    assert calls == ["parse", "load", "train", "publish"]
    assert not run_dir.exists()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_save_preserves_training_model_device(tmp_path: Path) -> None:
    case = make_case()
    model = case.model.to(device="cuda")
    path = tmp_path / "model.pt"
    save_structured_checkpoint(path, model, case.metadata, case.examples)
    assert next(model.parameters()).device.type == "cuda"
    loaded = load_structured_checkpoint(
        path, case.metadata.identity.encoding_hash, case.metadata.identity.capacity_hash,
    )
