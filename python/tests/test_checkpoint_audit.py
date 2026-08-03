import base64
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import ml_lab.checkpoint_audit as audit_module
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
        "roster": ["command", "infantry", "armor"],
        "reward": {"terminal_win": 1.0, "terminal_loss": -1.0},
        "semantics": {"start_profiles": [{"id": "standard-3v3"}]},
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

def test_discovery_rejects_mutually_compatible_non_tactical_v2_sources(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    for source in (clone, ppo):
        manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
        manifest["config"]["environment"] = "tactical-v1"
        manifest["contract"]["environment"] = "tactical-v1"
        manifest["contract"]["version"] = "tactical-v1"
        _write_json(source / "run.json", manifest)

    with pytest.raises(ValueError, match="tactical-v2"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)
RUNTIME_CONTRACT = {
    **_contract(),
    "roster": ["command", "infantry", "armor"],
    "reward": {"terminal_win": 1.0, "terminal_loss": -1.0},
    "semantics": {"start_profiles": [{"id": "standard-3v3"}]},
}
REPOSITORY_IDENTITY = {"commit": "a" * 40, "dirty": False}
HALF_WILSON = {
    "low": 0.09453120573423074,
    "high": 0.9054687942657693,
    "confidence": 0.95,
}
ZERO_WILSON = {"low": 0.0, "high": 0.6576197724933468, "confidence": 0.95}


def _audit_definition(
    source_runs: tuple[Path, Path], *, maps: int = 1
) -> AuditDefinition:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    return replace(
        definition,
        schedule=replace(definition.schedule, maps=maps),
        candidates=(definition.candidates[0],),
    )


def _scripted_identity(name: str) -> dict[str, Any]:
    return {
        "kind": "scripted",
        "inference_mode": "deterministic",
        "path": None,
        "algorithm": None,
        "step": None,
        "contract_hash": None,
        "contract_version": None,
        "environment": None,
        "encoding_hash": None,
        "contract": None,
        "observation_size": None,
        "action_size": None,
        "legacy": False,
        "promotable": False,
        "name": name,
    }


def _candidate_identity(candidate: Any) -> dict[str, Any]:
    if candidate.family == "control":
        return _scripted_identity(candidate.controller)
    spec = json.loads(candidate.controller)
    identity = {
        "kind": spec["kind"],
        "inference_mode": "deterministic",
        "path": candidate.checkpoint_path,
        "algorithm": "maskable_ppo",
        "step": candidate.actual_step,
        "contract_hash": candidate.source_contract_hash,
        "contract_version": "tactical-v2",
        "environment": "tactical-v2",
        "encoding_hash": candidate.source_encoding_hash,
        "contract": _contract(),
        "observation_size": candidate.observation_size,
        "action_size": candidate.action_size,
        "legacy": False,
        "promotable": True,
    }
    if spec["kind"] == "run":
        identity["source_run"] = candidate.source_run
    return identity


class _FakeAuditEvaluator:
    def __init__(self, definition: AuditDefinition) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._candidates = {
            candidate.controller: candidate for candidate in definition.candidates
        }

    def __call__(self, p0: str, p1: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((p0, p1, dict(kwargs)))
        candidate = self._candidates[p0]
        seed = kwargs["seed_start"]
        evidence_dir = Path(kwargs["evidence_dir"])
        trace_dir = evidence_dir / "traces"
        replay_dir = evidence_dir / "replays"
        trace_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        matches: list[dict[str, Any]] = []
        for index, seat in enumerate((0, 1)):
            stem = f"match-{index:06d}-seed-{seed}-candidate-seat-{seat}"
            trace_path = trace_dir / f"{stem}.json"
            replay_path = replay_dir / f"{stem}.replay"
            trace_path.write_bytes(f"trace-{seed}-{seat}\n".encode("ascii"))
            replay_path.write_bytes(f"replay-{seed}-{seat}\n".encode("ascii"))
            is_draw = seat == 1
            matches.append(
                {
                    "seed": seed,
                    "candidate_seat": seat,
                    "winner": -1 if is_draw else 0,
                    "outcome": "draw" if is_draw else "win",
                    "start_profile": "standard-3v3",
                    "reference_seat": seat,
                    "terminated": not is_draw,
                    "truncated": is_draw,
                    "summary": {"turns": 8 + seat, "final_state_hash": f"state-{seat}"},
                    "classification": {"primary": "turn-limit"} if is_draw else None,
                    "trace_path": str(trace_path),
                    "replay_path": str(replay_path),
                }
            )
        result = {
            "schema_version": 1,
            "generated_at": "2026-08-03T12:00:00Z",
            "schedule": {
                "start_profile": "standard-3v3",
                "reference_seat_policy": "candidate-seat",
            },
            "candidate": _candidate_identity(candidate),
            "opponent": _scripted_identity("random"),
            "seed_start": seed,
            "seeds": [seed],
            "reciprocal": True,
            "games": 2,
            "wins": 1,
            "losses": 0,
            "draws": 1,
            "rates": {"win": 0.5, "loss": 0.0, "draw": 0.5},
            "confidence_intervals": {
                "win": HALF_WILSON,
                "loss": ZERO_WILSON,
                "draw": HALF_WILSON,
            },
            "seat_results": {
                "candidate_as_p0": {"wins": 1, "losses": 0, "draws": 0},
                "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 1},
            },
            "matches": matches,
            "evidence": {
                "retention": "all",
                "retained": 2,
                "draw_traces": 1,
                "control_traces": 1,
                "draw_categories": {"turn-limit": 1},
            },
        }
        _write_json(Path(kwargs["output_path"]), result)
        return result


def _stub_audit_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audit_module,
        "_runtime_contract",
        lambda _server_cmd: dict(RUNTIME_CONTRACT),
        raising=False,
    )
    monkeypatch.setattr(
        audit_module,
        "_repository_identity",
        lambda: dict(REPOSITORY_IDENTITY),
        raising=False,
    )


def _evaluate_fake_audit(
    definition: AuditDefinition,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeAuditEvaluator:
    _stub_audit_identity(monkeypatch)
    evaluator = _FakeAuditEvaluator(definition)
    audit_module.evaluate_audit(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=3,
        evaluator=evaluator,
        progress=lambda _message: None,
    )
    return evaluator


def test_schedule_keys_are_exactly_candidate_map_and_reciprocal_seat(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs, maps=100)
    clone, ppo = source_runs
    discovered = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    definition = replace(
        definition, candidates=(definition.candidates[0], discovered.candidates[-1])
    )
    output_root = tmp_path / "audit"

    evaluator = _evaluate_fake_audit(definition, output_root, monkeypatch)
    assert [(p0, p1) for p0, p1, _kwargs in evaluator.calls] == [
        (candidate.controller, "random")
        for candidate in definition.candidates
        for _map_seed in range(16_000_000, 16_000_100)
    ]

    assert [
        {
            "games": kwargs["games"],
            "seed_start": kwargs["seed_start"],
            "both_seats": kwargs["both_seats"],
            "environment": kwargs["environment"],
            "start_profile": kwargs["start_profile"],
            "capture_trace": kwargs["capture_trace"],
            "evidence_retention": kwargs["evidence_retention"],
        }
        for _p0, _p1, kwargs in evaluator.calls
    ] == [
        {
            "games": 1,
            "seed_start": map_seed,
            "both_seats": True,
            "environment": "tactical-v2",
            "start_profile": "standard-3v3",
            "capture_trace": True,
            "evidence_retention": "all",
        }
        for _candidate in definition.candidates
        for map_seed in range(16_000_000, 16_000_100)
    ]
    actual_schedule_keys = []
    for candidate in definition.candidates:
        for map_seed in range(16_000_000, 16_000_100):
            _evaluation, matches = audit_module.validate_physical_map(
                output_root, candidate, definition.schedule, map_seed
            )
            actual_schedule_keys.extend(
                (candidate.candidate_id, match["seed"], match["candidate_seat"])
                for match in matches
            )
    assert actual_schedule_keys == [
        (candidate.candidate_id, map_seed, seat)
        for candidate in definition.candidates
        for map_seed in range(16_000_000, 16_000_100)
        for seat in (0, 1)
    ]


def test_evaluation_reuse_reopens_and_validates_every_physical_artifact(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs, maps=2)
    output_root = tmp_path / "audit"
    first = _evaluate_fake_audit(definition, output_root, monkeypatch)
    second = _FakeAuditEvaluator(definition)

    audit_module.evaluate_audit(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=3,
        evaluator=second,
        progress=lambda _message: None,
    )

    assert len(first.calls) == 2
    assert second.calls == []
    for map_seed in range(16_000_000, 16_000_002):
        evaluation, matches = audit_module.validate_physical_map(
            output_root, definition.candidates[0], definition.schedule, map_seed
        )
        assert evaluation["audit_identity"]["checkpoint_sha256"] == (
            definition.candidates[0].checkpoint_sha256
        )
        assert all(not Path(match["trace_path"]).is_absolute() for match in matches)
        assert all(not Path(match["replay_path"]).is_absolute() for match in matches)


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_controller",
        "opponent",
        "seed",
        "seat",
        "profile",
        "runtime_contract",
        "outcome_totals",
        "summary",
        "trace_path",
        "replay_path",
        "trace_bytes",
        "replay_bytes",
        "checkpoint_bytes",
    ],
)
def test_evaluation_tamper_fails_closed_without_overwriting_existing_map(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    evaluation_path = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    trace_path = output_root / evaluation["matches"][0]["trace_path"]
    replay_path = output_root / evaluation["matches"][0]["replay_path"]

    if mutation == "candidate_controller":
        evaluation["candidate"]["path"] = str(tmp_path / "other.zip")
    elif mutation == "opponent":
        evaluation["opponent"]["name"] = "greedy"
    elif mutation == "seed":
        evaluation["seeds"] = [16_000_001]
    elif mutation == "seat":
        evaluation["matches"][1]["candidate_seat"] = 0
    elif mutation == "profile":
        evaluation["schedule"]["start_profile"] = "other-profile"
    elif mutation == "runtime_contract":
        evaluation["audit_identity"]["runtime_contract"]["encoding_hash"] = "f" * 64
    elif mutation == "outcome_totals":
        evaluation["wins"] = 2
    elif mutation == "summary":
        evaluation["matches"][0]["summary"] = None
    elif mutation == "trace_path":
        evaluation["matches"][0]["trace_path"] = "../escape.json"
    elif mutation == "replay_path":
        evaluation["matches"][0]["replay_path"] = "../escape.replay"
    elif mutation == "trace_bytes":
        trace_path.write_bytes(b"tampered trace\n")
    elif mutation == "replay_bytes":
        replay_path.write_bytes(b"tampered replay\n")
    else:
        Path(candidate.checkpoint_path).write_bytes(b"tampered checkpoint\n")

    if mutation not in {"trace_bytes", "replay_bytes", "checkpoint_bytes"}:
        _write_json(evaluation_path, evaluation)
    before_evaluation = evaluation_path.read_bytes()
    before_trace = trace_path.read_bytes()
    before_replay = replay_path.read_bytes()
    rejecting_evaluator = _FakeAuditEvaluator(definition)

    with pytest.raises(ValueError):
        audit_module.evaluate_audit(
            definition,
            output_root=output_root,
            server_cmd=["fake-gym-server"],
            workers=3,
            evaluator=rejecting_evaluator,
            progress=lambda _message: None,
        )

    assert rejecting_evaluator.calls == []
    assert evaluation_path.read_bytes() == before_evaluation
    assert trace_path.read_bytes() == before_trace
    assert replay_path.read_bytes() == before_replay


def test_interrupted_evaluation_resumes_only_missing_map_without_root_aggregate(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs, maps=100)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    missing_map_dir = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_099
    ).parent
    shutil.rmtree(missing_map_dir)
    resumed = _FakeAuditEvaluator(definition)

    audit_module.evaluate_audit(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=3,
        evaluator=resumed,
        progress=lambda _message: None,
    )

    assert len(resumed.calls) == 1
    assert resumed.calls[0][2]["seed_start"] == 16_000_099
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "in_progress"
    assert not (output_root / "audit.json").exists()
    assert not (output_root / "report.md").exists()

@pytest.mark.parametrize("mutation", ["schedule", "candidate", "scenario"])
def test_validator_rejects_self_consistent_root_definition_tamper(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation_path = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    if mutation == "schedule":
        manifest["definition"]["schedule"]["profile"] = "other-profile"
    elif mutation == "candidate":
        manifest["definition"]["candidates"][0]["controller"] = "random"
    else:
        changed_scenario = b'{"template_id":"other"}\n'
        changed_digest = hashlib.sha256(changed_scenario).hexdigest()
        manifest["scenario"]["bytes_base64"] = base64.b64encode(changed_scenario).decode("ascii")
        manifest["scenario"]["sha256"] = changed_digest
        manifest["source_contracts"][0]["scenario_sha256"] = changed_digest
        manifest["definition"]["candidates"][0]["source_scenario_sha256"] = changed_digest
        evaluation["audit_identity"]["scenario_sha256"] = changed_digest
    manifest["definition_sha256"] = hashlib.sha256(
        (json.dumps(manifest["definition"], sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    evaluation["audit_identity"]["definition_sha256"] = manifest["definition_sha256"]
    _write_json(manifest_path, manifest)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )
