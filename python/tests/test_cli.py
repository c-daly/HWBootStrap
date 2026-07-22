from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_lab.cli as cli_module
from ml_lab.contracts import EnvironmentContract, RunConfig, create_run
from ml_lab.io import atomic_write_json, read_json


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="c" * 64,
        observation_size=12,
        action_size=7,
        board={"width": 2, "height": 2},
        roster=["scout"],
        reward={"terminal_win": 1.0},
    )


def _config(run_name: str) -> RunConfig:
    return RunConfig(
        backend="sb3",
        algorithm="maskable_ppo",
        policy="HexCNN",
        run_name=run_name,
        seed=17,
        total_timesteps=64,
        checkpoint_interval=32,
        workers=1,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "greedy"},
        trackers=[{"kind": "local"}],
        resume_source=None,
    )


def _complete_fake_run(runs_root: Path, config: RunConfig) -> Path:
    run_dir = runs_root / config.run_name
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "state": "completed",
            "timesteps": config.total_timesteps,
            "config": config.to_dict(),
        },
    )
    return run_dir


def _invoke_json(argv: list[str], **kwargs) -> tuple[int, dict]:
    stdout = StringIO()
    exit_code = cli_module.main(argv, stdout=stdout, **kwargs)
    rendered = stdout.getvalue()
    assert rendered.endswith("\n")
    assert rendered.count("\n") == 1
    return exit_code, json.loads(rendered)


def _assert_envelope(payload: dict, command: str) -> dict:
    assert payload.keys() == {"schema_version", "command", "ok", "result"}
    assert payload["schema_version"] == 1
    assert payload["command"] == command
    assert payload["ok"] is True
    assert isinstance(payload["result"], dict)
    return payload["result"]


def test_doctor_checks_required_headless_dependencies_and_optional_capabilities(
    tmp_path: Path,
) -> None:
    from ml_lab.doctor import doctor_environment

    package_versions = {
        "gymnasium": "1.2.3",
        "stable_baselines3": "2.7.0",
        "sb3_contrib": "2.7.0",
        "numpy": "2.4.0",
    }
    handshakes: list[tuple[str, ...]] = []

    result = doctor_environment(
        server_cmd=["dotnet", "fake-server.dll"],
        runs_root=tmp_path,
        trackers=["wandb"],
        package_version=lambda name: package_versions[name],
        dotnet_version=lambda: "8.0.18",
        handshake=lambda command: handshakes.append(tuple(command))
        or {"contract_version": "tactical-v1", "contract_hash": "c" * 64},
        cuda_info=lambda: {"available": False, "detail": "CPU-only host"},
        tracker_available=lambda name: False,
        write_probe=lambda path: path == tmp_path,
    )

    assert result["ok"] is True
    assert handshakes == [("dotnet", "fake-server.dll")]
    checks = {check["name"]: check for check in result["checks"]}
    assert checks["python:gymnasium"]["detail"] == "1.2.3"
    assert checks["dotnet"]["required"] is True
    assert checks["gymserver_handshake"]["required"] is True
    assert checks["write_access"]["required"] is True
    assert checks["cuda"] == {
        "name": "cuda",
        "ok": False,
        "required": False,
        "detail": "CPU-only host",
    }
    assert checks["tracker:wandb"]["required"] is False


def test_doctor_json_is_one_stable_object_and_forwards_requested_trackers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[dict] = []

    def fake_doctor(**kwargs):
        received.append(kwargs)
        return {"ok": True, "checks": [{"name": "write_access", "ok": True}]}

    monkeypatch.setattr(cli_module, "doctor_environment", fake_doctor, raising=False)

    exit_code, payload = _invoke_json(
        [
            "doctor",
            "--server",
            "fake-server.dll",
            "--runs-root",
            str(tmp_path),
            "--tracker",
            "wandb",
            "--json",
        ]
    )

    assert exit_code == 0
    assert _assert_envelope(payload, "doctor") == {
        "ok": True,
        "checks": [{"name": "write_access", "ok": True}],
    }
    assert received == [
        {
            "server_cmd": ["dotnet", "fake-server.dll"],
            "runs_root": tmp_path,
            "trackers": ["wandb"],
        }
    ]


def test_train_json_reports_the_durable_completed_run(
    tmp_path: Path,
) -> None:
    received: list[RunConfig] = []

    def runner(config: RunConfig, *, runs_root: Path, server_cmd: list[str]) -> Path:
        received.append(config)
        assert server_cmd == ["dotnet", "fake-server.dll"]
        return _complete_fake_run(runs_root, config)

    exit_code, payload = _invoke_json(
        [
            "train",
            "--run",
            "json-train",
            "--timesteps",
            "64",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--json",
        ],
        runner=runner,
    )

    assert exit_code == 0
    result = _assert_envelope(payload, "train")
    assert result["run_dir"] == str((tmp_path / "json-train").resolve())
    assert result["run"]["state"] == "completed"
    assert result["run"]["timesteps"] == 64
    assert received[0].run_name == "json-train"


def test_train_serializes_wandb_and_custom_tracker_configuration_without_secrets(
    tmp_path: Path,
) -> None:
    received: list[RunConfig] = []

    def runner(config: RunConfig, *, runs_root: Path, server_cmd: list[str]) -> Path:
        received.append(config)
        return _complete_fake_run(runs_root, config)

    exit_code, payload = _invoke_json(
        [
            "train",
            "--run",
            "tracker-config",
            "--runs-root",
            str(tmp_path),
            "--tracker",
            "local",
            "--tracker",
            "tensorboard",
            "--tracker",
            "wandb",
            "--tracker",
            "custom=lab_hooks:mirror_metrics",
            "--wandb-project",
            "hexwars",
            "--wandb-entity",
            "research",
            "--wandb-mode",
            "offline",
            "--wandb-group",
            "seat-bias",
            "--wandb-tag",
            "held-out",
            "--wandb-tag",
            "reciprocal",
            "--wandb-upload-artifacts",
            "--json",
        ],
        runner=runner,
    )

    assert exit_code == 0
    _assert_envelope(payload, "train")
    assert received[0].trackers == [
        {"kind": "local"},
        {"kind": "tensorboard"},
        {
            "kind": "wandb",
            "project": "hexwars",
            "entity": "research",
            "mode": "offline",
            "group": "seat-bias",
            "tags": ["held-out", "reciprocal"],
            "upload_artifacts": True,
        },
        {"kind": "custom", "adapter": "lab_hooks:mirror_metrics"},
    ]
    assert "api_key" not in json.dumps(received[0].trackers).lower()


def test_resume_builds_a_new_run_from_authoritative_source_metadata(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source = create_run(tmp_path, _config("source-run"), contract)
    checkpoint = source / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"source-model")
    source_manifest = read_json(source / "run.json")
    source_manifest.update(
        {
            "state": "stopped",
            "timesteps": 64,
            "latest_checkpoint": "checkpoints/step_000000064.zip",
            "latest_checkpoint_step": 64,
        }
    )
    atomic_write_json(source / "run.json", source_manifest)
    received: list[RunConfig] = []

    def runner(config: RunConfig, *, runs_root: Path, server_cmd: list[str]) -> Path:
        received.append(config)
        assert server_cmd == ["dotnet", "fake-server.dll"]
        return _complete_fake_run(runs_root, config)

    exit_code, payload = _invoke_json(
        [
            "resume",
            str(source),
            "--run",
            "resumed-run",
            "--timesteps",
            "160",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--json",
        ],
        runner=runner,
    )

    assert exit_code == 0
    result = _assert_envelope(payload, "resume")
    assert result["run_dir"] == str((tmp_path / "resumed-run").resolve())
    resumed = received[0]
    assert resumed == replace(
        _config("source-run"),
        run_name="resumed-run",
        total_timesteps=160,
        resume_source=str(source.resolve()),
    )


def test_status_reads_local_truth_without_unity_or_remote_services(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("status-run"), contract)
    manifest = read_json(run_dir / "run.json")
    manifest.update({"state": "running", "pid": 1234, "timesteps": 48})
    atomic_write_json(run_dir / "run.json", manifest)

    exit_code, payload = _invoke_json(["status", str(run_dir), "--json"])

    assert exit_code == 0
    result = _assert_envelope(payload, "status")
    assert result["run_dir"] == str(run_dir.resolve())
    assert result["run"]["state"] == "running"
    assert result["run"]["pid"] == 1234
    assert result["run"]["timesteps"] == 48


def test_status_follow_waits_for_terminal_local_state_without_streaming_json(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("follow-run"), contract)
    manifest = read_json(run_dir / "run.json")
    manifest.update({"state": "running", "timesteps": 16})
    atomic_write_json(run_dir / "run.json", manifest)
    sleeps: list[float] = []

    def finish_during_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        updated = read_json(run_dir / "run.json")
        updated.update({"state": "completed", "pid": None, "timesteps": 64})
        atomic_write_json(run_dir / "run.json", updated)

    exit_code, payload = _invoke_json(
        [
            "status",
            str(run_dir),
            "--follow",
            "--interval",
            "0.01",
            "--json",
        ],
        sleeper=finish_during_sleep,
    )

    assert exit_code == 0
    assert sleeps == [0.01]
    result = _assert_envelope(payload, "status")
    assert result["run"]["state"] == "completed"
    assert result["run"]["timesteps"] == 64


def test_status_follow_human_output_emits_intermediate_local_updates(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("human-follow"), contract)
    manifest = read_json(run_dir / "run.json")
    manifest.update({"state": "running", "timesteps": 16})
    atomic_write_json(run_dir / "run.json", manifest)
    updates = iter([("running", 32), ("completed", 64)])

    def advance(_seconds: float) -> None:
        state, timesteps = next(updates)
        changed = read_json(run_dir / "run.json")
        changed.update({"state": state, "timesteps": timesteps})
        atomic_write_json(run_dir / "run.json", changed)

    stdout = StringIO()
    exit_code = cli_module.main(
        ["status", str(run_dir), "--follow", "--interval", "0.01"],
        sleeper=advance,
        stdout=stdout,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        f"{run_dir.resolve()}: running at 16 timesteps",
        f"{run_dir.resolve()}: running at 32 timesteps",
        f"{run_dir.resolve()}: completed at 64 timesteps",
    ]


def test_stop_after_checkpoint_atomically_updates_local_control(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("stop-run"), contract)

    exit_code, payload = _invoke_json(
        ["stop", str(run_dir), "--after-checkpoint", "--json"]
    )

    assert exit_code == 0
    control = read_json(run_dir / "control.json")
    assert control["request"] == "stop_after_checkpoint"
    result = _assert_envelope(payload, "stop")
    assert result["run_dir"] == str(run_dir.resolve())
    assert result["control"] == control


def test_inspect_model_json_exposes_checkpoint_algorithm_contract_and_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "model-run"
    metadata = {
        "kind": "run",
        "path": str(run_dir / "checkpoints" / "step_000000064.zip"),
        "algorithm": "maskable_ppo",
        "step": 64,
        "contract_hash": "c" * 64,
        "source_run": str(run_dir),
        "legacy": False,
        "promotable": True,
    }
    received: list[str] = []

    def fake_inspect(raw: str) -> dict:
        received.append(raw)
        return metadata

    monkeypatch.setattr(cli_module, "inspect_model", fake_inspect, raising=False)

    exit_code, payload = _invoke_json(
        ["inspect-model", str(run_dir), "--json"]
    )

    assert exit_code == 0
    assert received == [str(run_dir)]
    assert _assert_envelope(payload, "inspect-model") == metadata


def test_publish_checkpoint_creates_editor_lab_candidate_with_evaluation_evidence(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    from ml_lab.evaluation import publish_candidate

    run_dir = create_run(tmp_path, _config("candidate-source"), contract)
    checkpoint = run_dir / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"candidate-model")
    manifest = read_json(run_dir / "run.json")
    manifest.update(
        {
            "state": "completed",
            "timesteps": 64,
            "latest_checkpoint": "checkpoints/step_000000064.zip",
            "latest_checkpoint_step": 64,
        }
    )
    atomic_write_json(run_dir / "run.json", manifest)
    identity = {
        "kind": "run",
        "path": str(checkpoint.resolve()),
        "algorithm": "maskable_ppo",
        "step": 64,
        "contract_hash": contract.contract_hash,
        "legacy": False,
        "promotable": True,
    }
    evaluation = {
        "schema_version": 1,
        "candidate": identity,
        "opponent": {"kind": "scripted", "name": "greedy"},
        "games": 20,
        "wins": 12,
        "losses": 6,
        "draws": 2,
    }
    atomic_write_json(run_dir / "evaluation.json", evaluation)
    resolved = SimpleNamespace(path=checkpoint.resolve(), promotable=True, metadata=lambda: identity)
    resolver = SimpleNamespace(resolve=lambda raw: resolved)

    candidate_dir = publish_candidate(
        run_dir,
        "ppo-counter-candidate",
        resolver=resolver,
    )

    assert candidate_dir == run_dir / "candidates" / "ppo-counter-candidate"
    assert (candidate_dir / "model.zip").read_bytes() == b"candidate-model"
    candidate = read_json(candidate_dir / "candidate.json")
    assert candidate["publication_scope"] == "editor_lab_only"
    assert candidate["player_build_published"] is False
    assert candidate["source_run"] == str(run_dir.resolve())
    assert candidate["source_checkpoint"] == str(checkpoint.resolve())
    assert candidate["checkpoint_identity"] == identity
    assert candidate["evaluation"] == evaluation
    assert candidate_dir.resolve().is_relative_to(run_dir.resolve())


def test_publish_checkpoint_json_returns_candidate_manifest_without_player_build_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "source"
    candidate_dir = run_dir / "candidates" / "candidate-a"
    candidate = {
        "name": "candidate-a",
        "publication_scope": "editor_lab_only",
        "player_build_published": False,
        "candidate_dir": str(candidate_dir.resolve()),
    }
    received: list[tuple[Path, str]] = []

    def fake_publish(source: Path, name: str) -> Path:
        received.append((source, name))
        candidate_dir.mkdir(parents=True)
        atomic_write_json(candidate_dir / "candidate.json", candidate)
        return candidate_dir

    monkeypatch.setattr(cli_module, "publish_candidate", fake_publish, raising=False)

    exit_code, payload = _invoke_json(
        [
            "publish-checkpoint",
            str(run_dir),
            "--name",
            "candidate-a",
            "--json",
        ]
    )

    assert exit_code == 0
    assert received == [(run_dir, "candidate-a")]
    assert _assert_envelope(payload, "publish-checkpoint") == candidate


def test_evaluate_json_supports_arbitrary_models_reciprocal_seats_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evaluation.json"
    evaluation = {
        "schema_version": 1,
        "games": 4,
        "wins": 2,
        "losses": 1,
        "draws": 1,
    }
    received: list[dict] = []

    def fake_evaluate(p0: str, p1: str, **kwargs) -> dict:
        received.append({"p0": p0, "p1": p1, **kwargs})
        return evaluation

    monkeypatch.setattr(
        cli_module, "evaluate_controllers", fake_evaluate, raising=False
    )

    exit_code, payload = _invoke_json(
        [
            "evaluate",
            "--p0",
            "run:first-model",
            "--p1",
            '{"kind":"run","path":"second-model","mode":"fixed"}',
            "--games",
            "2",
            "--seed-start",
            "10000",
            "--both-seats",
            "--workers",
            "2",
            "--server",
            "fake-server.dll",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    assert _assert_envelope(payload, "evaluate") == evaluation
    assert received == [
        {
            "p0": "run:first-model",
            "p1": '{"kind":"run","path":"second-model","mode":"fixed"}',
            "games": 2,
            "seed_start": 10_000,
            "both_seats": True,
            "workers": 2,
            "server_cmd": ["dotnet", "fake-server.dll"],
            "output_path": output,
        }
    ]


def test_benchmark_json_reports_headless_protocol_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "schema_version": 1,
        "elapsed_seconds": 2.0,
        "reset_count": 4,
        "decision_count": 8,
        "resets_per_second": 2.0,
        "decisions_per_second": 4.0,
        "cpu_count": 12,
        "worker_count": 2,
        "protocol": {"total_bytes": 602},
    }
    received: list[dict] = []

    def fake_benchmark(**kwargs) -> dict:
        received.append(kwargs)
        return metrics

    monkeypatch.setattr(
        cli_module, "benchmark_gymserver", fake_benchmark, raising=False
    )

    exit_code, payload = _invoke_json(
        [
            "benchmark",
            "--games",
            "4",
            "--seed-start",
            "50000",
            "--workers",
            "2",
            "--server",
            "fake-server.dll",
            "--json",
        ]
    )

    assert exit_code == 0
    assert _assert_envelope(payload, "benchmark") == metrics
    assert received == [
        {
            "games": 4,
            "seed_start": 50_000,
            "workers": 2,
            "server_cmd": ["dotnet", "fake-server.dll"],
        }
    ]


def test_winrate_is_only_a_compatibility_wrapper_for_unified_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import winrate

    forwarded: list[list[str]] = []
    monkeypatch.setattr(winrate, "ml_main", lambda argv: forwarded.append(argv) or 0)

    exit_code = winrate.main(
        ["--p0", "run:first", "--p1", "greedy", "--games", "6", "--both-seats"]
    )

    assert exit_code == 0
    assert forwarded == [
        [
            "evaluate",
            "--p0",
            "run:first",
            "--p1",
            "greedy",
            "--games",
            "6",
            "--both-seats",
        ]
    ]
