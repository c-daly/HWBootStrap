import base64
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import ml_lab.checkpoint_audit as audit_module
from ml_lab.checkpoint_audit import (
    AuditDefinition,
    AuditSchedule,
    discover_audit_candidates,
    build_audit_definition,
)
from ml_lab.draw_classification import classify_draw, summarize_episode
from ml_lab.tactical_trace import CommandFrame, EpisodeTrace, SeatFrame, StateFrame, TransitionFrame


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


def test_clone_discovery_accepts_contract_identity_without_redundant_config_environment(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    clone_manifest = json.loads((clone / "run.json").read_text(encoding="utf-8"))
    clone_manifest["config"].pop("environment")
    _write_json(clone / "run.json", clone_manifest)

    candidates = discover_audit_candidates(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
    )

    assert candidates[0].candidate_id == "pure-bc-seed-227"
    clone_manifest["config"]["environment"] = "tactical-v1"
    _write_json(clone / "run.json", clone_manifest)
    with pytest.raises(ValueError, match="environment does not match its contract"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)


def test_ppo_discovery_accepts_config_seed_without_redundant_model_seed_fields(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    ppo_manifest = json.loads((ppo / "run.json").read_text(encoding="utf-8"))
    ppo_manifest.pop("model_seed")
    ppo_manifest["config"].pop("model_seed")
    _write_json(ppo / "run.json", ppo_manifest)

    candidates = discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)

    assert [row.model_seed for row in candidates if row.family == "bc_ppo"] == [227] * 3

    ppo_manifest["config"]["seed"] = 211
    _write_json(ppo / "run.json", ppo_manifest)
    with pytest.raises(ValueError, match="model seed must be 227"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)

    ppo_manifest["config"].pop("seed")
    _write_json(ppo / "run.json", ppo_manifest)
    with pytest.raises(ValueError, match="model seed must be 227"):
        discover_audit_candidates(clone_run=clone, ppo_run=ppo, scratch_run=None)


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
        "schema_version": 2,
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
        "source_roots": [
            {"role": "clone", "source_run": str(clone.resolve())},
            {"role": "ppo", "source_run": str(ppo.resolve())},
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
REPOSITORY_IDENTITY = {"repository": "C:/repo", "commit": "a" * 40, "dirty": False}
EVALUATION_SOURCE_IDENTITY = {
    "repository": "C:/repo",
    "commit": "a" * 40,
    "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
}


def _artifact_trace(winner: int) -> EpisodeTrace:
    alive = (1, 0) if winner == 0 else (0, 0)
    seats = tuple(
        SeatFrame(
            seat=seat,
            points=0,
            destroyed_value=0,
            alive_units=alive[seat],
            current_hit_points=10 * alive[seat],
            maximum_hit_points=10 * alive[seat],
            health_adjusted_material=float(10 * alive[seat]),
            can_damage_enemy=bool(alive[seat]),
            can_currently_attack_enemy=False,
            can_move=bool(alive[seat]),
            units=(),
        )
        for seat in (0, 1)
    )
    before = StateFrame(
        round=1,
        active_seat=1,
        is_game_over=False,
        winner=None,
        productive_legal_actions=0,
        seats=seats,
    )
    after = replace(
        before,
        round=2,
        active_seat=0,
        is_game_over=True,
        winner=None if winner == -1 else winner,
    )
    return EpisodeTrace(
        schema_version=1,
        transitions=(
            TransitionFrame(
                before=before,
                command=CommandFrame(
                    kind="end_turn",
                    issuer=1,
                    actor_id=None,
                    target_id=None,
                    q=None,
                    r=None,
                ),
                after=after,
            ),
        ),
    )


def _artifact_summary(trace: EpisodeTrace, candidate_seat: int) -> dict[str, Any]:
    summary = summarize_episode(trace, candidate_seat)
    return {
        "command_count": summary.command_count,
        "round_count": summary.round_count,
        "damage_by_seat": list(summary.damage_by_seat),
        "kills_by_seat": list(summary.kills_by_seat),
        "end_turns_by_seat": list(summary.end_turns_by_seat),
        "wasted_end_turns_by_seat": list(summary.wasted_end_turns_by_seat),
        "peak_normalized_advantage": summary.peak_normalized_advantage,
        "final_normalized_advantage": summary.final_normalized_advantage,
        "maximum_state_repetition": summary.maximum_state_repetition,
    }


def _artifact_classification(trace: EpisodeTrace, candidate_seat: int) -> dict[str, Any]:
    classification = classify_draw(
        trace,
        candidate_seat=candidate_seat,
        terminated=True,
        truncated=False,
        winner=None,
    )
    return {
        "primary": classification.primary.value,
        "flags": [flag.value for flag in classification.flags],
        "evidence": dict(classification.evidence),
    }


def _artifact_replay(winner: int) -> str:
    unit = "U 1 0 10 1 0 1 0 1 360 1 360 0 0 0\n" if winner == 0 else ""
    return (
        "HEXWARS-REPLAY 1\n"
        "META 2 1 1 0 0\n"
        "CONFIG win=1 fixedTemplates=0 templateSlots=0 generators=0\n"
        "TILES 1\n"
        "0 0 0 0\n"
        "ZONE0 1 0 0\n"
        "ZONE1 0\n"
        f"PLAYER 0 0 {1 if winner == 0 else 0} 0 0\n"
        f"{unit}"
        "PLAYER 1 0 0 0 0\n"
        "CMDS 1\n"
        "E 1\n"
    )


def _fake_replay_inspection(paths: Any) -> dict[Path, int]:
    return {
        Path(path).resolve(): 0 if "candidate-seat-0" in Path(path).name else -1
        for path in paths
    }


def test_runtime_contract_accepts_horizon_and_environment_kind_differences() -> None:
    source_contract = _contract(
        board={
            "width": 13,
            "height": 9,
            "max_steps": 808,
            "environment_kind": "tactical",
        }
    )
    runtime_contract = {
        **RUNTIME_CONTRACT,
        "board": {
            "width": 13,
            "height": 9,
            "max_steps": 1616,
            "environment_kind": "duel",
        },
    }

    audit_module._validate_runtime_contract(
        runtime_contract,
        [{"contract": source_contract}],
    )


def test_runtime_contract_accepts_multiple_sources_with_allowed_full_details() -> None:
    source_contracts = [
        {
            "contract": _contract(
                contract_hash="c" * 64,
                board={
                    "width": 13,
                    "height": 9,
                    "max_steps": 808,
                    "environment_kind": "tactical",
                },
            )
        },
        {
            "contract": _contract(
                contract_hash="d" * 64,
                board={
                    "width": 13,
                    "height": 9,
                    "max_steps": 4096,
                    "environment_kind": "duel",
                },
            )
        },
    ]

    audit_module._validate_runtime_contract(RUNTIME_CONTRACT, source_contracts)


def test_runtime_contract_rejects_geometry_mismatch_only_in_second_source() -> None:
    source_contracts = [
        {
            "contract": _contract(
                contract_hash="c" * 64,
                board={"width": 13, "height": 9},
            )
        },
        {
            "contract": _contract(
                contract_hash="d" * 64,
                board={
                    "width": 12,
                    "height": 9,
                    "max_steps": 808,
                    "environment_kind": "tactical",
                },
            )
        },
    ]

    with pytest.raises(ValueError, match="board geometry"):
        audit_module._validate_runtime_contract(RUNTIME_CONTRACT, source_contracts)


def test_runtime_contract_rejects_board_geometry_mismatch() -> None:
    runtime_contract = {
        **RUNTIME_CONTRACT,
        "board": {"width": 12, "height": 9},
    }

    with pytest.raises(ValueError, match="board geometry"):
        audit_module._validate_runtime_contract(
            runtime_contract,
            [{"contract": _contract()}],
        )


HALF_WILSON = {
    "low": 0.09453120573423074,
    "high": 0.9054687942657693,
    "confidence": 0.95,
}
ZERO_WILSON = {"low": 0.0, "high": 0.6576197724933468, "confidence": 0.95}


def _audit_definition(
    source_runs: tuple[Path, Path], *, maps: int = 100
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
            is_draw = seat == 1
            winner = -1 if is_draw else 0
            replay_path.write_text(_artifact_replay(winner), encoding="utf-8")
            trace = _artifact_trace(winner)
            _write_json(trace_path, trace.to_dict())
            summary = _artifact_summary(trace, seat)
            classification = _artifact_classification(trace, seat) if is_draw else None
            matches.append(
                {
                    "seed": seed,
                    "candidate_seat": seat,
                    "winner": -1 if is_draw else 0,
                    "outcome": "draw" if is_draw else "win",
                    "start_profile": "standard-3v3",
                    "reference_seat": seat,
                    "terminated": True,
                    "truncated": False,
                    "summary": summary,
                    "classification": classification,
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
                "draw_categories": {"invalid_scenario": 1},
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
    monkeypatch.setattr(
        audit_module,
        "_evaluation_source_identity",
        lambda: dict(EVALUATION_SOURCE_IDENTITY),
        raising=False,
    )
    monkeypatch.setattr(
        audit_module,
        "_inspect_replays",
        _fake_replay_inspection,
        raising=False,
    )
    # Physical-artifact tests intentionally use a reduced synthetic roster;
    # exact rediscovery is exercised independently against real source roots below.
    monkeypatch.setattr(
        audit_module, "_validate_exact_candidate_set", lambda _definition: None
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
    output_root = tmp_path / "audit"
    clone, ppo = source_runs
    discovered = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    definition = replace(
        definition, candidates=(definition.candidates[0], discovered.candidates[-1])
    )

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
    definition = _audit_definition(source_runs)
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

    assert len(first.calls) == 100
    assert second.calls == []
    for map_seed in range(16_000_000, 16_000_100):
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
        "evaluation_source_identity",
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
    elif mutation == "evaluation_source_identity":
        evaluation["audit_identity"]["evaluation_source_identity"][
            "tracked_diff_sha256"
        ] = "f" * 64
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

@pytest.mark.parametrize(
    "schedule",
    [
        AuditSchedule(seed_start=16_000_000, maps=99),
        AuditSchedule(seed_start=17_000_000, maps=100),
    ],
    ids=["short-development-panel", "reserved-final-namespace"],
)
def test_evaluate_audit_rejects_non_frozen_development_schedule(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: AuditSchedule,
) -> None:
    definition = replace(_audit_definition(source_runs), schedule=schedule)
    _stub_audit_identity(monkeypatch)

    def unexpected_evaluator(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("invalid schedule reached evaluator")

    with pytest.raises(ValueError, match="schedule"):
        audit_module.evaluate_audit(
            definition,
            output_root=tmp_path / "audit",
            server_cmd=["fake-gym-server"],
            workers=1,
            evaluator=unexpected_evaluator,
            progress=lambda _message: None,
        )


def test_programmatic_smoke_evaluates_aggregates_reopens_and_reuses(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    output_root = tmp_path / "smoke"
    _stub_audit_identity(monkeypatch)

    def smoke_summary(
        rows: list[dict[str, Any]], _expected_games: int = 200
    ) -> dict[str, Any]:
        assert _expected_games == 4
        assert len(rows) == 4
        return {
            "counts": {"games": 4, "wins": 2, "losses": 0, "draws": 2},
            "rates": {"win": 0.5, "loss": 0.0, "draw": 0.5},
            "draw_diagnostics": {
                "cycling": {"count": 0, "incidence": 0.0},
            },
        }

    monkeypatch.setattr(audit_module, "summarize_candidate", smoke_summary)
    first_evaluator = _FakeAuditEvaluator(definition)

    first = audit_module.run_programmatic_smoke(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=2,
        evaluator=first_evaluator,
        progress=lambda _message: None,
    )

    assert first["definition"]["schedule"] == {
        "seed_start": 16_000_000,
        "maps": 2,
        "both_seats": True,
        "profile": "standard-3v3",
        "opponent": "random",
    }
    assert first["evaluation"] == {
        "state": "in_progress",
        "manifest": str(output_root / "manifest.json"),
        "maps": 12,
        "games": 24,
        "reused": 0,
    }
    assert len(first_evaluator.calls) == 12
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["smoke"] is True
    assert manifest["exploratory"] is True
    assert manifest["locked_panel_replacement"] is False
    assert first["aggregate"]["smoke"] is True
    assert first["aggregate"]["decision"] == {
        "available": False,
        "reason": "two-map programmatic smoke is not a decision panel",
    }
    assert first["aggregate"]["physical_evidence"] == {
        "maps": 12,
        "games": 24,
        "traces": 24,
        "replays": 24,
    }

    smoke_schedule = AuditSchedule(seed_start=16_000_000, maps=2)
    for candidate in definition.candidates:
        for map_seed in (16_000_000, 16_000_001):
            _evaluation, matches = audit_module.validate_physical_map(
                output_root, candidate, smoke_schedule, map_seed
            )
            assert len(matches) == 2

    second_evaluator = _FakeAuditEvaluator(definition)
    second = audit_module.run_programmatic_smoke(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=2,
        evaluator=second_evaluator,
        progress=lambda _message: None,
    )

    assert second["evaluation"]["reused"] == 12
    assert second["aggregate"] == first["aggregate"]
    assert second_evaluator.calls == []


@pytest.mark.parametrize(
    "mutation",
    ["scenario", "source_contract", "runtime_geometry"],
)
def test_validator_rejects_root_identity_transplant_without_definition_mutation(
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

    if mutation == "scenario":
        changed_scenario = b'{"template_id":"transplanted"}\n'
        changed_digest = hashlib.sha256(changed_scenario).hexdigest()
        manifest["scenario"]["bytes_base64"] = base64.b64encode(changed_scenario).decode("ascii")
        manifest["scenario"]["sha256"] = changed_digest
        evaluation["audit_identity"]["scenario_sha256"] = changed_digest
    elif mutation == "source_contract":
        manifest["source_contracts"][0]["contract"]["contract_hash"] = "d" * 64
        evaluation["candidate"]["contract"]["contract_hash"] = "d" * 64
    else:
        manifest["source_contracts"][0]["contract"]["encoding_hash"] = "f" * 64
        manifest["runtime_contract"]["encoding_hash"] = "f" * 64
        evaluation["candidate"]["contract"]["encoding_hash"] = "f" * 64
        evaluation["audit_identity"]["runtime_contract"]["encoding_hash"] = "f" * 64
    _write_json(manifest_path, manifest)
    _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


def test_validator_rejects_winner_outside_engine_domain(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    evaluation_path = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["matches"][1]["winner"] = 2
    _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="winner"):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


def test_summary_reports_candidate_outcomes_seats_and_unavailable_policy_diagnostic() -> None:
    rows = [
        {
            "map_seed": 16_000_000 + index // 2,
            "candidate_seat": index % 2,
            "outcome": "win" if index < 50 else "loss" if index < 80 else "draw",
            "summary": {
                "rounds": 10,
                "decisions": 20,
                "end_turns_by_seat": [10, 10],
                "wasted_end_turns_by_seat": [2, 2],
                "peak_health_adjusted_advantage": 0.4,
                "final_health_adjusted_advantage": 0.2,
            },
            "classification": (
                {"primary": "cycling", "flags": ["cycling"], "evidence": {}}
                if index >= 80
                else None
            ),
        }
        for index in range(200)
    ]

    summary = audit_module.summarize_candidate(rows)

    assert summary["counts"] == {"games": 200, "wins": 50, "losses": 30, "draws": 120}
    assert summary["rates"] == {"win": 0.25, "loss": 0.15, "draw": 0.6}
    assert summary["seats"]["candidate_as_p0"]["games"] == 100
    assert summary["seats"]["candidate_as_p1"]["games"] == 100
    assert summary["end_turn_policy_diagnostics"] == {
        "available": False,
        "reason": "integer-action inference boundary does not expose action probabilities or ranks",
    }

def _full_summary_rows() -> list[dict[str, Any]]:
    by_seat = {
        0: ["win"] * 30 + ["loss"] * 10 + ["draw"] * 60,
        1: ["win"] * 20 + ["loss"] * 20 + ["draw"] * 60,
    }
    rows: list[dict[str, Any]] = []
    draw_index = {0: 0, 1: 0}
    for offset in range(100):
        for seat in (0, 1):
            outcome = by_seat[seat][offset]
            classification = None
            if outcome == "draw":
                position = draw_index[seat]
                draw_index[seat] += 1
                category = (
                    "cycling"
                    if position < 20
                    else "action_waste"
                    if position < 35
                    else "balanced_attrition"
                )
                classification = {"primary": category, "flags": [category], "evidence": {}}
            rows.append(
                {
                    "map_seed": 16_000_000 + offset,
                    "candidate_seat": seat,
                    "outcome": outcome,
                    "summary": {
                        "rounds": 10,
                        "decisions": 20,
                        "end_turns_by_seat": [10, 10],
                        "wasted_end_turns_by_seat": [2, 2],
                        "peak_health_adjusted_advantage": 0.4,
                        "final_health_adjusted_advantage": 0.2,
                    },
                    "classification": classification,
                }
            )
    return rows


def test_summary_accepts_private_smoke_game_count_without_changing_default() -> None:
    rows = _full_summary_rows()[:4]

    summary = audit_module.summarize_candidate(rows, _expected_games=4)

    assert summary["counts"] == {"games": 4, "wins": 4, "losses": 0, "draws": 0}
    assert summary["rates"] == {"win": 1.0, "loss": 0.0, "draw": 0.0}
    assert summary["normalized_advantage"]["all_games"] == {
        "final": 0.2,
        "peak": 0.4,
    }


def test_summary_maps_raw_trace_fields_to_complete_aggregate_metrics() -> None:
    summary = audit_module.summarize_candidate(_full_summary_rows())

    assert summary["confidence_intervals"] == {
        "win": {"low": 0.195081680068175, "high": 0.3143409831204583, "confidence": 0.95},
        "loss": {"low": 0.10713593562241996, "high": 0.20605579284166659, "confidence": 0.95},
        "draw": {"low": 0.53083672039262, "high": 0.6653942143319266, "confidence": 0.95},
    }
    assert summary["seats"] == {
        "candidate_as_p0": {
            "games": 100, "wins": 30, "losses": 10, "draws": 60,
            "rates": {"win": 0.3, "loss": 0.1, "draw": 0.6},
        },
        "candidate_as_p1": {
            "games": 100, "wins": 20, "losses": 20, "draws": 60,
            "rates": {"win": 0.2, "loss": 0.2, "draw": 0.6},
        },
    }
    assert summary["win_rate_p0_minus_p1"] == pytest.approx(0.1)
    assert summary["draw_diagnostics"] == {
        "cycling": {"count": 40, "incidence": 0.2},
        "action_waste": {"count": 30, "incidence": 0.15},
        "primary_categories": {
            "action_waste": 30, "balanced_attrition": 50, "cycling": 40
        },
    }
    assert summary["winning_games"] == {
        "round_count": {"mean": 10.0, "median": 10.0, "p90": 10.0},
        "command_count": {"mean": 20.0, "median": 20.0, "p90": 20.0},
    }
    assert summary["normalized_advantage"] == {
        "all_games": {"final": 0.2, "peak": 0.4},
        "draws": {"final": 0.2, "peak": 0.4},
    }
    assert summary["candidate_end_turns"] == {
        "total": 2000, "wasted": 400, "wasted_ratio": 0.2
    }

def _paired_rows(outcomes: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "map_seed": 16_000_000 + index // 2,
            "candidate_seat": index % 2,
            "outcome": outcome,
        }
        for index, outcome in enumerate(outcomes)
    ]


def test_paired_change_reports_full_transitions_and_exact_sign_test() -> None:
    transitions = (
        [("win", "win")] * 60
        + [("win", "draw")] * 15
        + [("win", "loss")] * 5
        + [("draw", "win")] * 25
        + [("draw", "draw")] * 25
        + [("draw", "loss")] * 10
        + [("loss", "win")] * 15
        + [("loss", "draw")] * 20
        + [("loss", "loss")] * 25
    )
    change = audit_module.paired_change(
        _paired_rows([left for left, _right in transitions]),
        _paired_rows([right for _left, right in transitions]),
    )

    assert change["transition_table"] == {
        "win": {"win": 60, "draw": 15, "loss": 5},
        "draw": {"win": 25, "draw": 25, "loss": 10},
        "loss": {"win": 15, "draw": 20, "loss": 25},
    }
    assert change["left_only_wins"] == 20
    assert change["right_only_wins"] == 40
    assert change["net_win_change"] == 20
    assert change["absolute_win_rate_change"] == pytest.approx(0.1)
    assert change["exact_sign_test_p_value"] == pytest.approx(0.01348929373119186)


def test_paired_change_rejects_duplicate_or_missing_schedule_keys() -> None:
    rows = _paired_rows(["draw"] * 200)
    duplicate = list(rows)
    duplicate[-1] = dict(duplicate[0])

    with pytest.raises(ValueError, match="duplicate"):
        audit_module.paired_change(rows, duplicate)
    with pytest.raises(ValueError, match="missing"):
        audit_module.paired_change(rows, rows[:-1])

def _decision_row(
    candidate_id: str,
    family: str,
    trajectory_order: int,
    *,
    wins: int,
    losses: int,
    draws: int,
    cycling: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "trajectory_order": trajectory_order,
        "summary": {
            "counts": {"games": 200, "wins": wins, "losses": losses, "draws": draws},
            "rates": {
                "win": wins / 200,
                "loss": losses / 200,
                "draw": draws / 200,
            },
            "draw_diagnostics": {
                "cycling": {"count": cycling, "incidence": cycling / 200}
            },
        },
    }


def test_decision_reports_independent_clauses_and_applies_precedence() -> None:
    clone = _decision_row(
        "clone", "pure_bc", 0, wins=120, losses=20, draws=60, cycling=10
    )
    qualifying = _decision_row(
        "ppo-1", "bc_ppo", 1, wins=140, losses=20, draws=40, cycling=10
    )
    regressed = _decision_row(
        "ppo-2", "bc_ppo", 2, wins=110, losses=30, draws=60, cycling=20
    )

    mixed = audit_module.choose_next_experiment([clone, qualifying, regressed])

    assert mixed == {
        "clauses": {
            "qualifying_ppo": ["ppo-1"],
            "consistent_improvement": True,
            "large_late_regression": True,
            "cycling_dominant": False,
            "all_ppo_below_half": False,
        },
        "recommended_next_step": "test_retained_imitation_constraint",
    }
    assert audit_module.choose_next_experiment([clone, qualifying])[
        "recommended_next_step"
    ] == "replicate_seeds_211_223"


def test_decision_distinguishes_dagger_and_inconclusive_trajectories() -> None:
    clone = _decision_row(
        "clone", "pure_bc", 0, wins=120, losses=20, draws=60, cycling=10
    )
    cycling = _decision_row(
        "ppo-low", "bc_ppo", 1, wins=60, losses=20, draws=120, cycling=80
    )
    middling = _decision_row(
        "ppo-mid", "bc_ppo", 1, wins=110, losses=20, draws=70, cycling=10
    )

    dagger = audit_module.choose_next_experiment([clone, cycling])
    assert dagger["clauses"]["all_ppo_below_half"] is True
    assert dagger["clauses"]["cycling_dominant"] is True
    assert dagger["recommended_next_step"] == "proceed_to_dagger"
    assert audit_module.choose_next_experiment([clone, middling])[
        "recommended_next_step"
    ] == "inconclusive_review_trajectory"

def test_aggregate_publication_marks_manifest_completed_and_revalidates_on_reopen(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone, ppo = source_runs
    discovered = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    definition = replace(
        discovered,
        candidates=(
            discovered.candidates[0],
            discovered.candidates[-2],
            discovered.candidates[-1],
        ),
    )
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    summarized_keys: list[list[tuple[int, int]]] = []

    def fake_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        summarized_keys.append(
            [(row["map_seed"], row["candidate_seat"]) for row in rows]
        )
        return _decision_row(
            "unused", "pure_bc", 0, wins=100, losses=0, draws=100, cycling=100
        )["summary"]

    monkeypatch.setattr(audit_module, "summarize_candidate", fake_summary)
    aggregate = audit_module.aggregate_audit(definition, output_root=output_root)

    assert aggregate["schema_version"] == 1
    assert aggregate["audit_id"] == "annihilation-checkpoint-audit-v1"
    assert aggregate["exploratory"] is True
    assert aggregate["locked_panel_replacement"] is False
    assert aggregate["anchors"] == ["random-anchor", "bounded-search-anchor"]
    assert [row["candidate_id"] for row in aggregate["trajectory"]] == [
        "pure-bc-seed-227"
    ]
    assert aggregate["paired_successive_changes"] == []
    assert aggregate["decision"]["clauses"]["large_late_regression"] is False
    assert aggregate["decision"]["recommended_next_step"] == (
        "inconclusive_review_trajectory"
    )
    assert aggregate["physical_evidence"] == {
        "maps": 300, "games": 600, "traces": 600, "replays": 600
    }
    assert summarized_keys[0][:4] == [
        (16_000_000, 0),
        (16_000_000, 1),
        (16_000_001, 0),
        (16_000_001, 1),
    ]
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_bytes = (output_root / "audit.json").read_bytes()
    assert manifest["state"] == "completed"
    assert manifest["aggregate_sha256"] == hashlib.sha256(audit_bytes).hexdigest()

    original_validate = audit_module.validate_physical_map
    reopened_maps: list[tuple[str, int]] = []

    def recording_validate(
        root: Path, candidate: Any, schedule: Any, seed: int, **kwargs: Any
    ) -> Any:
        reopened_maps.append((candidate.candidate_id, seed))
        return original_validate(root, candidate, schedule, seed, **kwargs)

    monkeypatch.setattr(audit_module, "validate_physical_map", recording_validate)
    assert audit_module.aggregate_audit(definition, output_root=output_root) == aggregate
    assert len(reopened_maps) == 300

def test_validator_rejects_reciprocal_matches_that_share_artifacts(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    evaluation_path = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    for field in ("trace_path", "replay_path", "trace_sha256", "replay_sha256"):
        evaluation["matches"][1][field] = evaluation["matches"][0][field]
    evaluation["audit_identity"]["artifacts"][1] = dict(
        evaluation["audit_identity"]["artifacts"][0]
    )
    _write_json(evaluation_path, evaluation)

    with pytest.raises(ValueError, match="distinct"):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


@pytest.mark.parametrize(
    "mutation", ["generated_at", "repository_identity", "state"]
)
def test_aggregate_rejects_malformed_root_manifest_shape(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    definition = _audit_definition(source_runs)
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "generated_at":
        manifest["generated_at"] = 7
    elif mutation == "repository_identity":
        manifest["repository_identity"] = {"commit": "bad", "dirty": "no"}
    else:
        manifest["state"] = "paused"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        audit_module,
        "summarize_candidate",
        lambda _rows: _decision_row(
            "unused", "pure_bc", 0, wins=100, losses=0, draws=100, cycling=100
        )["summary"],
    )

    with pytest.raises(ValueError, match="manifest"):
        audit_module.aggregate_audit(definition, output_root=output_root)

def test_aggregate_write_failure_leaves_root_manifest_in_progress(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    monkeypatch.setattr(
        audit_module,
        "summarize_candidate",
        lambda _rows: _decision_row(
            "unused", "pure_bc", 0, wins=100, losses=0, draws=100, cycling=100
        )["summary"],
    )
    real_atomic_write = audit_module.atomic_write_json

    def fail_aggregate_write(path: Path, value: object) -> None:
        if Path(path).name == "audit.json":
            raise OSError("simulated aggregate write failure")
        real_atomic_write(path, value)

    monkeypatch.setattr(audit_module, "atomic_write_json", fail_aggregate_write)
    with pytest.raises(OSError, match="simulated aggregate write failure"):
        audit_module.aggregate_audit(definition, output_root=output_root)

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "in_progress"
    assert "aggregate_sha256" not in manifest
    assert not (output_root / "audit.json").exists()

def test_summary_handles_candidates_with_no_wins_or_draws() -> None:
    rows = _full_summary_rows()
    for row in rows:
        row["outcome"] = "loss"
        row["classification"] = None

    summary = audit_module.summarize_candidate(rows)

    assert summary["winning_games"] == {
        "round_count": {"mean": None, "median": None, "p90": None},
        "command_count": {"mean": None, "median": None, "p90": None},
    }
    assert summary["normalized_advantage"]["draws"] == {
        "final": None, "peak": None
    }


def test_large_late_regression_uses_exact_ten_point_boundary() -> None:
    clone = _decision_row(
        "clone", "pure_bc", 0, wins=130, losses=20, draws=50, cycling=10
    )
    earlier = _decision_row(
        "ppo-140", "bc_ppo", 1, wins=140, losses=20, draws=40, cycling=10
    )
    later = _decision_row(
        "ppo-120", "bc_ppo", 2, wins=120, losses=20, draws=60, cycling=10
    )

    decision = audit_module.choose_next_experiment([clone, earlier, later])

    assert decision["clauses"]["large_late_regression"] is True
    assert decision["recommended_next_step"] == "test_retained_imitation_constraint"


def _write_self_consistent_audit_root(
    definition: AuditDefinition,
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root.mkdir()
    _stub_audit_identity(monkeypatch)
    scenario_bytes, scenario_sha256, source_contracts = audit_module._source_material(
        definition
    )
    manifest = audit_module._initial_manifest(
        definition,
        scenario_bytes=scenario_bytes,
        scenario_sha256=scenario_sha256,
        source_contracts=source_contracts,
        runtime_contract=RUNTIME_CONTRACT,
    )
    _write_json(output_root / "manifest.json", manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version",
        "audit_id",
        "exploratory",
        "locked_panel",
        "final_seed_namespace",
        "map_count",
        "nonreciprocal",
        "profile",
        "opponent",
    ],
)
def test_aggregate_independently_rejects_self_consistent_identity_or_isolation_mutation(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    definition = _audit_definition(source_runs)
    if mutation == "schema_version":
        definition = replace(definition, schema_version=3)
    elif mutation == "audit_id":
        definition = replace(definition, audit_id="other-audit")
    elif mutation == "exploratory":
        definition = replace(definition, exploratory=False)
    elif mutation == "locked_panel":
        definition = replace(definition, locked_panel_replacement=True)
    elif mutation == "final_seed_namespace":
        definition = replace(
            definition, schedule=replace(definition.schedule, seed_start=17_000_000)
        )
    elif mutation == "map_count":
        definition = replace(
            definition, schedule=replace(definition.schedule, maps=99)
        )
    elif mutation == "nonreciprocal":
        definition = replace(
            definition, schedule=replace(definition.schedule, both_seats=False)
        )
    elif mutation == "profile":
        definition = replace(
            definition, schedule=replace(definition.schedule, profile="other-profile")
        )
    else:
        definition = replace(
            definition, schedule=replace(definition.schedule, opponent="greedy")
        )
    output_root = tmp_path / "audit"
    _write_self_consistent_audit_root(definition, output_root, monkeypatch)

    def unexpected_validation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid global audit definition reached physical map validation")

    monkeypatch.setattr(audit_module, "validate_physical_map", unexpected_validation)
    with pytest.raises(ValueError, match="global identity/isolation contract"):
        audit_module.aggregate_audit(definition, output_root=output_root)


def test_validate_prepared_definition_rehashes_every_learned_checkpoint(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)

    prepared = audit_module.validate_prepared_definition(definition)

    assert prepared.scenario_sha256 == definition.candidates[0].source_scenario_sha256
    assert len(prepared.source_contracts) == 2
    changed = Path(definition.candidates[1].checkpoint_path)
    changed.write_bytes(changed.read_bytes() + b"-mutated-after-prepare")
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        audit_module.validate_prepared_definition(definition)


def test_validate_prepared_definition_rejects_dropped_final_physical_ppo_candidate(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    assert [(source.role, source.source_run) for source in definition.source_roots] == [
        ("clone", str(clone.resolve())),
        ("ppo", str(ppo.resolve())),
    ]
    final_ppo = max(
        (candidate for candidate in definition.candidates if candidate.family == "bc_ppo"),
        key=lambda candidate: candidate.actual_step,
    )
    tampered = replace(
        definition,
        candidates=tuple(
            candidate for candidate in definition.candidates if candidate != final_ppo
        ),
    )

    with pytest.raises(ValueError, match="candidate set"):
        audit_module.validate_prepared_definition(tampered)


@pytest.mark.parametrize("entrypoint", ["evaluate", "aggregate"])
def test_reopen_entrypoints_reject_dropped_physical_candidate_before_games(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    entrypoint: str,
) -> None:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    final_ppo = max(
        (candidate for candidate in definition.candidates if candidate.family == "bc_ppo"),
        key=lambda candidate: candidate.actual_step,
    )
    tampered = replace(
        definition,
        candidates=tuple(
            candidate
            for candidate in definition.candidates
            if candidate != final_ppo
        ),
    )
    output_root = tmp_path / entrypoint
    output_root.mkdir()

    with pytest.raises(ValueError, match="candidate set"):
        if entrypoint == "evaluate":
            audit_module.evaluate_audit(
                tampered,
                output_root=output_root,
                server_cmd=["must-not-start"],
                workers=1,
            )
        else:
            audit_module.aggregate_audit(tampered, output_root=output_root)


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_random_anchor",
        "drop_bounded_anchor",
        "duplicate_id",
        "unsafe_id",
        "reorder_trajectory",
        "noncontiguous_trajectory",
        "noncanonical_controller",
        "drop_ppo_family",
        "duplicate_checkpoint_membership",
        "missing_scratch_omission",
        "control_checkpoint_provenance",
    ],
)
def test_validate_prepared_definition_rejects_noncanonical_candidate_set_mutations(
    source_runs: tuple[Path, Path],
    mutation: str,
) -> None:
    clone, ppo = source_runs
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)
    candidates = list(definition.candidates)
    if mutation == "drop_random_anchor":
        candidates = [
            candidate for candidate in candidates if candidate.candidate_id != "random-anchor"
        ]
    elif mutation == "drop_bounded_anchor":
        candidates = [
            candidate
            for candidate in candidates
            if candidate.candidate_id != "bounded-search-anchor"
        ]
    elif mutation == "duplicate_id":
        candidates[1] = replace(candidates[1], candidate_id=candidates[0].candidate_id)
    elif mutation == "unsafe_id":
        candidates[1] = replace(candidates[1], candidate_id="../escaped-candidate")
    elif mutation == "reorder_trajectory":
        candidates[1], candidates[2] = candidates[2], candidates[1]
    elif mutation == "noncontiguous_trajectory":
        candidates[2] = replace(candidates[2], trajectory_order=99)
    elif mutation == "noncanonical_controller":
        candidates[1] = replace(
            candidates[1],
            controller=json.dumps(json.loads(candidates[1].controller), indent=2),
        )
    elif mutation == "drop_ppo_family":
        candidates = [candidate for candidate in candidates if candidate.family != "bc_ppo"]
    elif mutation == "duplicate_checkpoint_membership":
        candidates[2] = replace(
            candidates[2],
            checkpoint_path=candidates[1].checkpoint_path,
            checkpoint_sha256=candidates[1].checkpoint_sha256,
        )
    elif mutation == "control_checkpoint_provenance":
        control_index = next(
            index
            for index, candidate in enumerate(candidates)
            if candidate.family == "control"
        )
        candidates[control_index] = replace(
            candidates[control_index], checkpoint_sha256="a" * 64
        )
    tampered = replace(definition, candidates=tuple(candidates))
    if mutation == "missing_scratch_omission":
        tampered = replace(tampered, omitted_optional_candidates=())

    with pytest.raises(ValueError, match="candidate|source"):
        audit_module.validate_prepared_definition(tampered)


@pytest.mark.parametrize("mutation", ["drop_scratch_family", "claim_scratch_omitted"])
def test_validate_prepared_definition_rejects_inconsistent_supplied_scratch_set(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    clone, ppo = source_runs
    scratch = _write_run(
        tmp_path / "scratch-completeness",
        kind="scratch",
        checkpoints=(20_480, 40_960),
    )
    definition = build_audit_definition(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=scratch,
    )
    if mutation == "drop_scratch_family":
        tampered = replace(
            definition,
            candidates=tuple(
                candidate
                for candidate in definition.candidates
                if candidate.family != "scratch_ppo"
            ),
        )
    else:
        tampered = replace(
            definition,
            omitted_optional_candidates=(
                {
                    "family": "scratch_ppo",
                    "reason": "no physical compatible run supplied",
                },
            ),
        )

    with pytest.raises(ValueError, match="candidate"):
        audit_module.validate_prepared_definition(tampered)


def test_exact_candidate_validation_is_general_over_physical_ppo_steps(
    source_runs: tuple[Path, Path],
) -> None:
    clone, ppo = source_runs
    extra_step = 44_000
    (ppo / "checkpoints" / f"step_{extra_step:09d}.zip").write_bytes(b"extra-physical-ppo")
    definition = build_audit_definition(clone_run=clone, ppo_run=ppo, scratch_run=None)

    audit_module.validate_prepared_definition(definition)

    assert [
        candidate.actual_step
        for candidate in definition.candidates
        if candidate.family == "bc_ppo"
    ] == [14_336, 26_624, 38_912, extra_step]


def test_validator_rejects_coherent_metric_tamper_with_unchanged_artifacts(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    evaluation_path = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    draw_match = evaluation["matches"][1]
    evaluation["matches"][0].update(
        winner=-1,
        outcome="draw",
        terminated=True,
        truncated=False,
        summary=json.loads(json.dumps(draw_match["summary"])),
        classification=json.loads(json.dumps(draw_match["classification"])),
    )
    evaluation.update(
        wins=0,
        losses=0,
        draws=2,
        rates={"win": 0.0, "loss": 0.0, "draw": 1.0},
        confidence_intervals={
            "win": ZERO_WILSON,
            "loss": ZERO_WILSON,
            "draw": {"low": 0.3423802275066532, "high": 1.0, "confidence": 0.95},
        },
        seat_results={
            "candidate_as_p0": {"wins": 0, "losses": 0, "draws": 1},
            "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 1},
        },
        evidence={
            "retention": "all",
            "retained": 2,
            "draw_traces": 2,
            "control_traces": 0,
            "draw_categories": {"invalid_scenario": 2},
        },
    )
    _write_json(evaluation_path, evaluation)
    with pytest.raises(ValueError, match="trace|artifact|winner|summary|classification"):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


def test_validator_rejects_unreferenced_map_evidence_file(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    map_root = audit_module.audit_map_path(
        output_root, candidate.candidate_id, 16_000_000
    ).parent
    (map_root / "evidence" / "unexpected.bin").write_bytes(b"unreferenced\n")

    with pytest.raises(ValueError, match="unexpected|inventory|unreferenced"):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


def test_replay_inspector_uses_authoritative_engine_reconstruction(tmp_path: Path) -> None:
    p0 = tmp_path / "p0.replay"
    draw = tmp_path / "draw.replay"
    p0.write_text(_artifact_replay(0), encoding="utf-8")
    draw.write_text(_artifact_replay(-1), encoding="utf-8")

    assert audit_module._inspect_replays((p0, draw)) == {
        p0.resolve(): 0,
        draw.resolve(): -1,
    }


def test_validator_rejects_replay_terminal_winner_disagreement(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    candidate = definition.candidates[0]
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    monkeypatch.setattr(
        audit_module,
        "_inspect_replays",
        lambda paths: {Path(path).resolve(): -1 for path in paths},
        raising=False,
    )

    with pytest.raises(ValueError, match="replay.*winner|winner.*replay"):
        audit_module.validate_physical_map(
            output_root, candidate, definition.schedule, 16_000_000
        )


def test_aggregate_batches_authoritative_replay_inspection(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    calls: list[tuple[Path, ...]] = []

    def inspect(paths: Any) -> dict[Path, int]:
        batch = tuple(Path(path).resolve() for path in paths)
        calls.append(batch)
        return _fake_replay_inspection(batch)

    monkeypatch.setattr(audit_module, "_inspect_replays", inspect)
    audit_module.aggregate_audit(definition, output_root=output_root)

    assert len(calls) == 1
    assert len(calls[0]) == 200


@pytest.mark.parametrize("mutation", ["commit", "tracked_diff"])
def test_resume_rejects_changed_evaluation_source_before_runtime_or_evaluator(
    source_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    definition = _audit_definition(source_runs)
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    changed = dict(EVALUATION_SOURCE_IDENTITY)
    if mutation == "commit":
        changed["commit"] = "b" * 40
    else:
        changed["tracked_diff_sha256"] = "c" * 64
    monkeypatch.setattr(audit_module, "_evaluation_source_identity", lambda: changed)

    def unexpected_runtime(_server_cmd: Any) -> dict[str, Any]:
        raise AssertionError("changed source identity reached runtime server")

    monkeypatch.setattr(audit_module, "_runtime_contract", unexpected_runtime)
    evaluator = _FakeAuditEvaluator(definition)
    with pytest.raises(ValueError, match="evaluation source identity"):
        audit_module.evaluate_audit(
            definition,
            output_root=output_root,
            server_cmd=["fake-gym-server"],
            workers=1,
            evaluator=evaluator,
            progress=lambda _message: None,
        )
    assert evaluator.calls == []


def test_resume_reuses_maps_with_unchanged_evaluation_source_identity(
    source_runs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _audit_definition(source_runs)
    output_root = tmp_path / "audit"
    _evaluate_fake_audit(definition, output_root, monkeypatch)
    evaluator = _FakeAuditEvaluator(definition)
    replay_calls: list[tuple[Path, ...]] = []

    def inspect(paths: Any) -> dict[Path, int]:
        batch = tuple(Path(path).resolve() for path in paths)
        replay_calls.append(batch)
        return _fake_replay_inspection(batch)

    monkeypatch.setattr(audit_module, "_inspect_replays", inspect)
    audit_module.evaluate_audit(
        definition,
        output_root=output_root,
        server_cmd=["fake-gym-server"],
        workers=1,
        evaluator=evaluator,
        progress=lambda _message: None,
    )

    assert evaluator.calls == []
    assert len(replay_calls) == 1
    assert len(replay_calls[0]) == 200


def test_evaluation_source_identity_ignores_tracked_generated_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    source = repository / "python" / "ml_lab" / "checkpoint_audit.py"
    bytecode = repository / "python" / "hexwars_gym" / "__pycache__" / "env.pyc"
    source.parent.mkdir(parents=True)
    bytecode.parent.mkdir(parents=True)
    source.write_text("SOURCE = 1\n", encoding="utf-8")
    bytecode.write_bytes(b"generated-v1")
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Checkpoint Audit Test",
            "-c",
            "user.email=checkpoint-audit@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(audit_module, "__file__", str(source))
    clean = audit_module._evaluation_source_identity()

    bytecode.write_bytes(b"generated-v2")
    assert audit_module._evaluation_source_identity() == clean

    source.write_text("SOURCE = 2\n", encoding="utf-8")
    changed = audit_module._evaluation_source_identity()
    assert changed["commit"] == clean["commit"]
    assert changed["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
