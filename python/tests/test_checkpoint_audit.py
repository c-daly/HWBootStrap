import hashlib
import json
from pathlib import Path

import pytest

from ml_lab.checkpoint_audit import (
    AuditDefinition,
    discover_audit_candidates,
    build_audit_definition,
)


ENCODING_HASH = "e" * 64


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _contract(*, contract_hash: str = "c" * 64, **changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "environment": "tactical-v2",
        "version": "tactical-v2",
        "contract_hash": contract_hash,
        "encoding_hash": ENCODING_HASH,
        "observation_size": 761,
        "action_size": 379,
        "board": {"width": 13, "height": 9},
    }
    result.update(changes)
    return result


def _write_run(
    root: Path,
    *,
    kind: str,
    checkpoints: tuple[int, ...],
    model_seed: int = 227,
    actor_init_source: str | None = None,
    scenario_bytes: bytes = b'{"environment":"tactical-v2","template_id":"standard-3v3"}\n',
    contract: dict[str, object] | None = None,
    latest_checkpoint: str | None = None,
    latest_checkpoint_step: int | None = None,
    state: str = "completed",
    checkpoint_history: list[int] | None = None,
) -> Path:
    root.mkdir()
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    for step in checkpoints:
        (checkpoint_dir / f"step_{step:09d}.zip").write_bytes(
            f"physical-checkpoint-{kind}-{step}".encode("ascii")
        )
    if latest_checkpoint is None and checkpoints:
        latest_checkpoint = f"checkpoints/step_{checkpoints[-1]:09d}.zip"
    if latest_checkpoint_step is None and checkpoints:
        latest_checkpoint_step = checkpoints[-1]
    config: dict[str, object] = {
        "algorithm": "maskable_ppo",
        "seed": model_seed,
        "model_seed": model_seed,
        "environment": "tactical-v2",
        "actor_init_source": actor_init_source,
    }
    if kind == "clone":
        config["behavioral_cloning"] = {"model_seed": model_seed}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "state": state,
        "timesteps": latest_checkpoint_step,
        "latest_checkpoint": latest_checkpoint,
        "latest_checkpoint_step": latest_checkpoint_step,
        "config": config,
        "contract": _contract() if contract is None else contract,
        "scenario": {"path": "scenario.json", "schema_version": 1},
        "model_seed": model_seed,
    }
    if checkpoint_history is not None:
        manifest["checkpoint_steps"] = checkpoint_history
    _write_json(root / "run.json", manifest)
    (root / "scenario.json").write_bytes(scenario_bytes)
    return root


@pytest.fixture
def source_runs(tmp_path: Path) -> tuple[Path, Path]:
    clone = _write_run(tmp_path / "clone", kind="clone", checkpoints=(0,))
    ppo = _write_run(
        tmp_path / "ppo",
        kind="ppo",
        checkpoints=(14_336, 26_624, 38_912),
        actor_init_source=str(clone),
        state="stopped",
        latest_checkpoint="checkpoints/step_000051200.zip",
        latest_checkpoint_step=51_036,
    )
    (ppo / "checkpoints" / "step_000051200.zip.partial").write_bytes(b"not-a-candidate")
    return clone, ppo


def test_candidate_discovery_uses_only_physical_checkpoint_trajectory(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs

    candidates = discover_audit_candidates(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
    )

    assert [(row.candidate_id, row.actual_step) for row in candidates] == [
        ("pure-bc-seed-227", 0),
        ("bc-ppo-seed-227-step-000014336", 14_336),
        ("bc-ppo-seed-227-step-000026624", 26_624),
        ("bc-ppo-seed-227-step-000038912", 38_912),
        ("random-anchor", None),
        ("bounded-search-anchor", None),
    ]
    assert all("51200" not in row.candidate_id for row in candidates)


def test_learned_candidates_capture_canonical_physical_provenance(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs

    candidates = discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)
    learned = candidates[:4]

    assert [row.source_run for row in learned] == [str(clone.resolve())] + [str(ppo.resolve())] * 3
    assert [row.source_scenario_sha256 for row in learned] == [
        hashlib.sha256((clone / "scenario.json").read_bytes()).hexdigest(),
        hashlib.sha256((ppo / "scenario.json").read_bytes()).hexdigest(),
        hashlib.sha256((ppo / "scenario.json").read_bytes()).hexdigest(),
        hashlib.sha256((ppo / "scenario.json").read_bytes()).hexdigest(),
    ]
    assert [row.checkpoint_sha256 for row in learned] == [
        hashlib.sha256(Path(row.checkpoint_path).read_bytes()).hexdigest() for row in learned
    ]
    assert all(Path(row.checkpoint_path).is_absolute() for row in learned)
    assert all(row.source_run_manifest_sha256 for row in learned)
    assert all(row.source_contract_hash for row in learned)
    assert all(row.source_encoding_hash == ENCODING_HASH for row in learned)
    assert all((row.observation_size, row.action_size) == (761, 379) for row in learned)
    assert json.loads(learned[0].controller) == {
        "kind": "run",
        "mode": "fixed",
        "path": str(clone.resolve()),
    }
    assert json.loads(learned[1].controller) == {
        "algorithm": "maskable_ppo",
        "kind": "snapshot",
        "path": str((ppo / "checkpoints" / "step_000014336.zip").resolve()),
        "source_run": str(ppo.resolve()),
        "step": 14_336,
    }


def test_controls_are_stable_and_do_not_claim_checkpoint_provenance(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs

    random_anchor, bounded_search = discover_audit_candidates(
        clone_run=clone, ppo_run=ppo, scratch_run=None
    )[-2:]

    assert (random_anchor.family, random_anchor.controller) == ("control", "random")
    assert (bounded_search.family, bounded_search.controller) == ("control", "bounded-search")
    assert all(
        getattr(candidate, field) is None
        for candidate in (random_anchor, bounded_search)
        for field in (
            "model_seed",
            "actual_step",
            "checkpoint_path",
            "checkpoint_sha256",
            "source_run",
            "source_run_manifest_sha256",
            "source_scenario_sha256",
            "source_contract_hash",
            "source_encoding_hash",
            "observation_size",
            "action_size",
        )
    )


def test_definition_explicitly_records_omitted_scratch_and_serializes_without_dataclass_magic(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs

    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)

    assert isinstance(definition, AuditDefinition)
    assert definition.omitted_optional_candidates == (
        {"family": "scratch_ppo", "reason": "no physical compatible run supplied"},
    )
    assert definition.to_dict() == {
        "schema_version": 1,
        "audit_id": "annihilation-checkpoint-audit-v1",
        "exploratory": True,
        "locked_panel_replacement": False,
        "schedule": {
            "seed_start": 16_000_000,
            "maps": 100,
            "both_seats": True,
            "profile": "standard-3v3",
            "opponent": "random",
        },
        "candidates": [candidate.to_dict() for candidate in definition.candidates],
        "omitted_optional_candidates": [
            {"family": "scratch_ppo", "reason": "no physical compatible run supplied"}
        ],
    }
    json.dumps(definition.to_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "checkpoint"),
        ("empty", "checkpoint"),
        ("clone_step_mismatch", "step"),
        ("clone_seed", "seed"),
        ("ppo_seed", "seed"),
        ("actor_source", "actor_init_source"),
        ("scenario", "scenario"),
        ("environment", "environment"),
        ("version", "version"),
        ("encoding", "encoding"),
        ("observation", "observation"),
        ("action", "action"),
    ],
)
def test_discovery_rejects_unphysical_or_incompatible_sources(
    source_runs: tuple[Path, Path], mutation: str, message: str
) -> None:
    clone, ppo = source_runs
    clone_manifest = json.loads((clone / "run.json").read_text(encoding="utf-8"))
    ppo_manifest = json.loads((ppo / "run.json").read_text(encoding="utf-8"))

    if mutation == "missing":
        (clone / "checkpoints" / "step_000000000.zip").unlink()
    elif mutation == "empty":
        (clone / "checkpoints" / "step_000000000.zip").write_bytes(b"")
    elif mutation == "clone_step_mismatch":
        clone_manifest["latest_checkpoint_step"] = 1
        _write_json(clone / "run.json", clone_manifest)
    elif mutation == "clone_seed":
        clone_manifest["model_seed"] = 211
        clone_manifest["config"]["model_seed"] = 211
        clone_manifest["config"]["seed"] = 211
        _write_json(clone / "run.json", clone_manifest)
    elif mutation == "ppo_seed":
        ppo_manifest["model_seed"] = 211
        ppo_manifest["config"]["model_seed"] = 211
        ppo_manifest["config"]["seed"] = 211
        _write_json(ppo / "run.json", ppo_manifest)
    elif mutation == "actor_source":
        ppo_manifest["config"]["actor_init_source"] = str(ppo)
        _write_json(ppo / "run.json", ppo_manifest)
    elif mutation == "scenario":
        (ppo / "scenario.json").write_bytes(b'{"environment":"tactical-v2","template_id":"other"}\n')
    else:
        key = {
            "environment": "environment",
            "version": "version",
            "encoding": "encoding_hash",
            "observation": "observation_size",
            "action": "action_size",
        }[mutation]
        ppo_manifest["contract"][key] = "different" if mutation in {"environment", "version", "encoding"} else 1
        _write_json(ppo / "run.json", ppo_manifest)

    with pytest.raises(ValueError, match=message):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)


def test_discovery_rejects_non_monotonic_or_duplicate_recorded_checkpoint_history(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    manifest = json.loads((ppo / "run.json").read_text(encoding="utf-8"))
    manifest["checkpoint_steps"] = [14_336, 26_624, 26_624]
    _write_json(ppo / "run.json", manifest)

    with pytest.raises(ValueError, match="checkpoint steps"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)

    manifest["checkpoint_steps"] = [26_624, 14_336, 38_912]
    _write_json(ppo / "run.json", manifest)
    with pytest.raises(ValueError, match="checkpoint steps"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)


@pytest.mark.parametrize("mutation", ["bc_initialized", "seed", "missing_checkpoint"])
def test_supplied_scratch_must_be_physical_uninitialized_and_seed_compatible(
    source_runs: tuple[Path, Path], tmp_path: Path, mutation: str
) -> None:
    clone, ppo = source_runs
    scratch = _write_run(
        tmp_path / "scratch",
        kind="scratch",
        checkpoints=(38_912,),
        actor_init_source=None,
    )
    if mutation == "bc_initialized":
        manifest = json.loads((scratch / "run.json").read_text(encoding="utf-8"))
        manifest["config"]["actor_init_source"] = str(clone)
        _write_json(scratch / "run.json", manifest)
    elif mutation == "seed":
        manifest = json.loads((scratch / "run.json").read_text(encoding="utf-8"))
        manifest["model_seed"] = 211
        manifest["config"]["model_seed"] = 211
        manifest["config"]["seed"] = 211
        _write_json(scratch / "run.json", manifest)
    else:
        (scratch / "checkpoints" / "step_000038912.zip").unlink()

    with pytest.raises(ValueError):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=scratch)


def test_supplied_scratch_is_added_only_from_a_physical_compatible_checkpoint(
    source_runs: tuple[Path, Path], tmp_path: Path
) -> None:
    clone, ppo = source_runs
    scratch = _write_run(tmp_path / "scratch", kind="scratch", checkpoints=(38_912,))

    candidates = discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=scratch)

    assert (candidates[-3].candidate_id, candidates[-3].family, candidates[-3].actual_step) == (
        "scratch-ppo-seed-227-step-000038912",
        "scratch_ppo",
        38_912,
    )


def test_discovery_never_names_manifest_only_51200_checkpoint(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs

    candidates = discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)

    assert all(candidate.actual_step != 51_200 for candidate in candidates)
    assert all("000051200" not in candidate.candidate_id for candidate in candidates)
