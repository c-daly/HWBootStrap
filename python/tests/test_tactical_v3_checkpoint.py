from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ml_lab.tactical_v3_checkpoint import (
    StructuredCheckpointMetadata,
    load_structured_checkpoint,
    publish_structured_run,
    save_structured_checkpoint,
    validate_structured_run,
)
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_training import TrainerConfig, TrainingResult
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


def make_case() -> CheckpointCase:
    identity = load_duel_identity_fixture()
    corpus = load_tiny_corpus_fixture()
    model = TacticalV3Policy(TacticalV3ModelConfig()).eval()
    metadata = StructuredCheckpointMetadata(
        1, "structured_imitation", identity, model.config, ObjectiveConfig(),
        TrainerConfig(), corpus.identity, "0" * 64, 0, 0.0, "cpu",
    )
    result = TrainingResult(model, 0, 0.0, False, ())
    return CheckpointCase(metadata, model, corpus.validation[:2], corpus,
                          DUEL_IDENTITY_FIXTURE, result)


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
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
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
