from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from ml_lab.tactical_v3_batching import collate_examples
from ml_lab.tactical_v3_checkpoint import (
    StructuredCheckpointMetadata,
    load_structured_checkpoint,
    publish_structured_run,
    save_structured_checkpoint,
    validate_structured_run,
)
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_training import (
    METRIC_KEYS,
    EpochMetrics,
    TrainerConfig,
    TrainingResult,
    _batch_to_device,
    train_offline,
)
from ml_lab.tactical_v3_corpus import load_corpus
from tests.tactical_v3_fixture_support import (
    DUEL_IDENTITY_FIXTURE,
    TINY_CORPUS_ROOT,
    load_duel_identity_fixture,
    load_tiny_corpus_fixture,
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


@pytest.fixture(scope="module")
def published_run_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CheckpointCase, Path]:
    case = make_case()
    root = tmp_path_factory.mktemp("published-run-template")
    return case, publish_structured_run(
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


def tree_snapshot(root: Path) -> tuple[tuple[str, int, int, str], ...]:
    rows: list[tuple[str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        rows.append((relative, info.st_mode, info.st_size, digest))
    return tuple(rows)


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
    run_dir = publish_structured_run(
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
        publish_structured_run(
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

    run_dir = publish_structured_run(
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
        publish_structured_run(
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
            publish_structured_run(
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
            publish_structured_run(
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
            publish_structured_run(
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
            publish_structured_run(
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

    run_dir = publish_structured_run(
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

    assert publish_structured_run(
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
        publish_structured_run(
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
    assert publish_structured_run(run_dir, case.result, case.corpus, case.scenario) == run_dir
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["evidence_status"] == "unsealed-experimental"
    assert manifest["config"]["algorithm"] == "structured_imitation"
    assert validate_structured_run(run_dir).fixture.examples == case.corpus.validation[:2]
    with pytest.raises(FileExistsError):
        publish_structured_run(run_dir, case.result, case.corpus, case.scenario)
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
    scenario_payload = json.loads(case.scenario.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"

    def fake_parse(payload: object) -> object:
        assert payload == scenario_payload
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
        assert model_config == TacticalV3ModelConfig()
        assert objective_config == ObjectiveConfig()
        assert trainer_config == TrainerConfig(seed=0, device="cpu")
        calls.append("train")
        return case.result

    def fake_publish(destination: Path, result: TrainingResult, corpus: object, scenario: Path) -> Path:
        assert destination == run_dir
        assert result is case.result
        assert result.model_config == TacticalV3ModelConfig()
        assert result.objective_config == ObjectiveConfig()
        assert result.trainer_config == TrainerConfig(seed=0, device="cpu")
        assert corpus is case.corpus
        assert scenario == case.scenario
        calls.append("publish")
        return destination

    monkeypatch.setattr(cli, "parse_spaces", fake_parse)
    monkeypatch.setattr(cli, "load_corpus", fake_load)
    monkeypatch.setattr(cli, "train_offline", fake_train)
    monkeypatch.setattr(cli, "publish_structured_run", fake_publish)
    assert cli.main([
        "train", "--corpus", str(TINY_CORPUS_ROOT), "--scenario", str(case.scenario),
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
