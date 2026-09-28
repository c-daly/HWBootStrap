from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ml_lab.controllers import ControllerResolver
from ml_lab.tactical_v3_batching import collate_decisions
from ml_lab.tactical_v3_controller import load_structured_controller, select_candidate
from ml_lab.tactical_v3_playable import export_playable_package, validate_playable_package
from ml_lab.tactical_v3_training import TrainerConfig
from tests.test_tactical_v3_controller import make_structured_run_case


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def package(tmp_path):
    case = make_structured_run_case(tmp_path)
    manifest = case.run_dir / "run.json"
    value = json.loads(manifest.read_text())
    value["state"] = "stopped"
    manifest.write_text(json.dumps(value))
    source = case.run_dir / "checkpoints/best.pt"
    evidence = tmp_path / "evidence.md"
    evidence.write_text("Finite development panel, not a promotion.\n")
    inputs = {p: p.read_bytes() for p in (source, manifest, evidence)}
    out = export_playable_package(source, tmp_path / "package", expected_checkpoint_sha256=sha(source),
        source_manifest=manifest, package_id="test-policy", evidence=(evidence,))
    assert all(p.read_bytes() == b for p, b in inputs.items())
    return case, out


def test_portable_export_keeps_stopped_provenance_and_exact_actions(package):
    case, out = package
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["completion_scope"] == "inference-package-export"
    assert manifest["source"]["state"] == "stopped"
    assert (out / "checkpoints/best.pt").read_bytes() == (case.run_dir / "checkpoints/best.pt").read_bytes()
    loaded = validate_playable_package(out, expected_package_sha256=manifest["package_sha256"])
    resolved = ControllerResolver(expected_structured_hashes=(case.identity.encoding_hash, case.identity.capacity_hash)).resolve(f"run:{out}")
    assert resolved.metadata()["checkpoint_sha256"] == sha(out / "checkpoints/best.pt")
    assert resolved.metadata()["package_id"] == "test-policy"
    assert resolved.promotable is False
    assert resolved.model.identity == loaded.metadata.identity
    with torch.inference_mode():
        expected, = loaded.model.select(collate_decisions(
            (case.view.decision,), loaded.model.config.horizon_turns, identity=case.identity))
    assert select_candidate(resolved.model, case.view) == expected


@pytest.mark.parametrize("name", ["checkpoints/best.pt", "policy-identity.json", "provenance/source-run.json", "evidence/00-evidence.md", "run.json"])
def test_tampered_export_rejected_before_tensor_load(package, monkeypatch, name):
    _, out = package
    (out / name).write_bytes((out / name).read_bytes() + b" ")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("tampered package reached tensor loader"))
    with pytest.raises(ValueError):
        validate_playable_package(out)


def test_resealed_identity_still_must_match_checkpoint(package):
    _, out = package
    identity = json.loads((out / "policy-identity.json").read_text())
    identity["scenario_id"] += "-tampered"
    (out / "policy-identity.json").write_text(json.dumps(identity))
    manifest = json.loads((out / "run.json").read_text())
    manifest["files"]["policy-identity.json"] = sha(out / "policy-identity.json")
    from ml_lab.tactical_v3_playable import _json_bytes
    manifest.pop("package_sha256")
    manifest["package_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    (out / "run.json").write_bytes(_json_bytes(manifest))
    with pytest.raises(ValueError):
        validate_playable_package(out)


def test_pinned_package_and_exact_inventory(package):
    _, out = package
    with pytest.raises(ValueError, match="pinned"):
        validate_playable_package(out, expected_package_sha256="0" * 64)
    (out / "unexpected.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        validate_playable_package(out)


def test_export_refuses_wrong_hash_and_overwrite(package, tmp_path):
    case, out = package
    source = case.run_dir / "checkpoints/best.pt"
    args = dict(source_manifest=case.run_dir / "run.json", package_id="test-policy")
    with pytest.raises(ValueError, match="expected hash"):
        export_playable_package(source, tmp_path / "wrong", expected_checkpoint_sha256="0" * 64, **args)
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(FileExistsError):
        export_playable_package(source, out, expected_checkpoint_sha256=sha(source), **args)


def test_cuda_training_metadata_does_not_require_gpu_for_cpu_inference(package, monkeypatch):
    case, out = package
    source = case.run_dir / "checkpoints/best.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["metadata"]["trainer_config"]["device"] = "cuda:7"
    torch.save(payload, source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        TrainerConfig(device="cuda:7")
    out2 = export_playable_package(source, out.parent / "cpu-export", expected_checkpoint_sha256=sha(source),
        source_manifest=case.run_dir / "run.json", package_id="cpu-export")
    loaded = validate_playable_package(out2)
    assert loaded.metadata.trainer_config.device == "cuda:7"
    assert next(loaded.model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps", "cuda:-1", "cuda:1.0", False])
def test_recorded_device_still_validated(device):
    with pytest.raises(ValueError):
        TrainerConfig.from_checkpoint_metadata(device=device)


def test_policy_server_cpu_jsonl_handshake_and_real_action(package):
    case, out = package
    fixture = Path(__file__).parent / "fixtures/tactical_v3/seed-41-decision.json"
    view = json.loads(fixture.read_text())
    command = [sys.executable, str(Path(__file__).parents[1] / "policy_server.py"),
        "--p0", f"run:{out}", "--p1", f"run:{out}",
        "--expected-environment", "tactical-v3", "--expected-contract-version", "tactical-v3",
        "--expected-encoding-hash", case.identity.encoding_hash, "--expected-capacity-hash", case.identity.capacity_hash]
    result = subprocess.run(command, input=json.dumps({"seat": view["seat"], "decision": view}) + '\n{"cmd":"close"}\n',
        capture_output=True, text=True, timeout=60, check=True)
    ready, action = map(json.loads, result.stdout.splitlines())
    assert ready["ready"] and ready["model_seats"] == [0, 1]
    assert all(x["checkpoint_sha256"] == sha(out / "checkpoints/best.pt") for x in ready["seat_models"])
    controller = load_structured_controller(out, case.identity.encoding_hash, case.identity.capacity_hash)
    assert action == asdict(select_candidate(controller, case.view))


def test_inference_import_does_not_require_training_environment_packages():
    code = '''
import importlib.abc, sys
class NoTrainingPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gymnasium','stable_baselines3','sb3_contrib'}:
            raise ImportError('training-only dependency: ' + fullname)
sys.meta_path.insert(0, NoTrainingPackages())
import policy_server
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=180)
