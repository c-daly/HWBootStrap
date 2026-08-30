from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_lab.cli as cli_module
from ml_lab.contracts import EnvironmentContract, RunConfig, create_run as create_durable_run
from ml_lab.io import atomic_write_json, read_json
from ml_lab.scenarios import ResolvedScenario


ROOT = Path(__file__).resolve().parents[2]
TACTICAL_V3_SCENARIO = (
    ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
)


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="c" * 64,
        encoding_hash="d" * 64,
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
        environment="tactical-v1",
    )


def create_run(
    runs_root: Path,
    config: RunConfig,
    contract: EnvironmentContract,
) -> Path:
    template_id = (
        "tactical-standard"
        if config.environment == "tactical-v1"
        else "adaptive-standard"
    )
    scenario = cli_module.resolve_scenario(
        environment=config.environment,
        scenario_file=None,
        template_id=template_id,
    )
    return create_durable_run(
        runs_root,
        config,
        contract,
        scenario,
        opponent_snapshot=config.opponent,
    )


def parse_train(*options: str):
    return cli_module.build_parser().parse_args(["train", "--run", "locked-ppo", *options])


def test_cli_records_locked_ppo_options() -> None:
    args = parse_train(
        "--actor-init", "bc/run",
        "--learning-rate", "0.0003",
        "--ppo-epochs", "10",
        "--target-kl", "0.02",
        "--episode-seed-base", "13000000",
    )

    config = cli_module._training_config(args)

    assert config.algorithm_options == {
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "target_kl": 0.02,
    }
    assert config.actor_init_source == "bc/run"
    assert config.episode_seed_base == 13_000_000


def test_cli_records_explicit_adaptive_environment(tmp_path: Path) -> None:
    received: list[RunConfig] = []

    def runner(config: RunConfig, *, scenario: ResolvedScenario, **_kwargs) -> Path:
        received.append(config)
        assert scenario.template_id == "adaptive-standard"
        return _complete_fake_run(tmp_path, config)

    assert cli_module.main([
        "train", "--run", "adaptive-one", "--environment", "adaptive-v1",
        "--runs-root", str(tmp_path), "--json",
    ], runner=runner, stdout=StringIO()) == 0

    assert received[0].environment == "adaptive-v1"


def test_resume_manifest_without_explicit_environment_fails_closed(tmp_path: Path) -> None:
    source = create_run(tmp_path, _config("old-source"), EnvironmentContract(
        version="tactical-v1", contract_hash="c" * 64, encoding_hash="d" * 64, observation_size=12,
        action_size=7, board={"width": 2, "height": 2}, roster=["scout"],
        reward={"terminal_win": 1.0},
    ))
    manifest = read_json(source / "run.json")
    del manifest["config"]["environment"]
    atomic_write_json(source / "run.json", manifest)
    output = StringIO()
    assert cli_module.main([
        "resume", str(source), "--run", "old-resumed", "--timesteps", "128",
        "--runs-root", str(tmp_path), "--json",
    ], runner=lambda *_args, **_kwargs: source, stdout=output) == 1
    assert "environment" in output.getvalue()


def test_train_resume_inherits_adaptive_source_environment(tmp_path: Path) -> None:
    adaptive_contract = replace(
        EnvironmentContract(
            version="tactical-v1", contract_hash="c" * 64, encoding_hash="d" * 64, observation_size=12,
            action_size=7, board={"width": 2, "height": 2}, roster=["scout"],
            reward={"terminal_win": 1.0},
        ),
        version="adaptive-v1",
        semantics={"environment_kind": "adaptive_tactical"},
    )
    source = create_run(
        tmp_path,
        replace(_config("adaptive-source"), environment="adaptive-v1"),
        adaptive_contract,
    )
    (source / "scenario.json").unlink()
    received: list[RunConfig] = []

    def runner(config: RunConfig, *, scenario: ResolvedScenario, **_kwargs) -> Path:
        received.append(config)
        assert scenario.template_id == "legacy-default"
        return _complete_fake_run(tmp_path, config)

    assert cli_module.main([
        "train", "--run", "adaptive-resumed", "--resume", str(source),
        "--runs-root", str(tmp_path), "--json",
    ], runner=runner, stdout=StringIO()) == 0

    assert received[0].environment == "adaptive-v1"


def test_train_resume_inherits_locked_ppo_options(
    tmp_path: Path,
    contract: EnvironmentContract,
) -> None:
    source = create_run(
        tmp_path,
        replace(
            _config("locked-source"),
            algorithm_options={
                "learning_rate": 0.0007,
                "n_epochs": 4,
                "target_kl": 0.03,
            },
            episode_seed_base=13_000_000,
        ),
        contract,
    )
    args = cli_module.build_parser().parse_args(
        ["train", "--run", "locked-resume", "--resume", str(source)]
    )

    resumed = cli_module._training_config(args)

    assert resumed.algorithm_options == {
        "learning_rate": 0.0007,
        "n_epochs": 4,
        "target_kl": 0.03,
    }
    assert resumed.episode_seed_base == 13_000_000


def test_train_resume_rejects_ignored_ppo_option_override(
    tmp_path: Path,
    contract: EnvironmentContract,
) -> None:
    source = create_run(tmp_path, _config("override-source"), contract)
    args = cli_module.build_parser().parse_args(
        [
            "train",
            "--run",
            "override-resume",
            "--resume",
            str(source),
            "--learning-rate",
            "0.0007",
        ]
    )

    with pytest.raises(ValueError, match="cannot be overridden during resume"):
        cli_module._training_config(args)


def test_train_cli_rejects_template_and_file_together() -> None:
    with pytest.raises(SystemExit):
        cli_module.build_parser().parse_args([
            "train", "--run", "x", "--template", "tactical-standard",
            "--scenario-file", "custom.json",
        ])


def test_train_cli_passes_selected_template_to_runner(tmp_path: Path) -> None:
    received: list[ResolvedScenario] = []

    def runner(
        config: RunConfig, *, scenario: ResolvedScenario, **_kwargs
    ) -> Path:
        received.append(scenario)
        return _complete_fake_run(tmp_path, config)

    assert cli_module.main([
        "train", "--run", "large", "--environment", "tactical-v1",
        "--template", "tactical-large-battle", "--runs-root", str(tmp_path),
        "--json",
    ], runner=runner, stdout=StringIO()) == 0

    assert received[0].template_id == "tactical-large-battle"


def test_train_resume_uses_source_scenario_instead_of_new_selection(
    tmp_path: Path,
) -> None:
    source = create_run(
        tmp_path,
        _config("scenario-source"),
        EnvironmentContract(
            version="tactical-v1",
            contract_hash="c" * 64,
            encoding_hash="d" * 64,
            observation_size=12,
            action_size=7,
            board={"width": 2, "height": 2},
            roster=["scout"],
            reward={"terminal_win": 1.0},
        ),
    )
    source_scenario = cli_module.resolve_scenario(
        environment="tactical-v1",
        scenario_file=None,
        template_id="tactical-long-battle",
    )
    source_scenario.write(source / "scenario.json")
    received: list[ResolvedScenario] = []

    def runner(
        config: RunConfig, *, scenario: ResolvedScenario, **_kwargs
    ) -> Path:
        received.append(scenario)
        return _complete_fake_run(tmp_path, config)

    assert cli_module.main([
        "train", "--run", "scenario-resumed", "--resume", str(source),
        "--template", "tactical-large-battle", "--runs-root", str(tmp_path),
        "--json",
    ], runner=runner, stdout=StringIO()) == 0

    assert received[0].template_id == "tactical-long-battle"


def _complete_fake_run(runs_root: Path, config: RunConfig) -> Path:
    run_dir = runs_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
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

    from .test_gym_client import _valid_adaptive_spaces

    result = doctor_environment(
        server_cmd=["dotnet", "fake-server.dll"],
        environment="adaptive-v1",
        runs_root=tmp_path,
        trackers=["wandb"],
        package_version=lambda name: package_versions[name],
        dotnet_version=lambda: "8.0.18",
        handshake=lambda command: handshakes.append(tuple(command))
        or _valid_adaptive_spaces(),
        cuda_info=lambda: {"available": False, "detail": "CPU-only host"},
        tracker_available=lambda name: False,
        write_probe=lambda path: path == tmp_path,
    )

    assert result["ok"] is True
    assert handshakes == [("dotnet", "fake-server.dll", "--environment", "adaptive-v1")]
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


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(contract_version="tactical-v1"), "contract_version"),
        (
            lambda value: (
                value.update(environment_kind="adaptive_duel"),
                value["board"].update(environment_kind="adaptive_duel"),
                value["adaptive"].update(environment_kind="adaptive_duel"),
            ),
            "environment_kind",
        ),
        (lambda value: value["adaptive"].pop("phases"), "phases"),
    ],
)
def test_doctor_rejects_wrong_or_malformed_selected_contract(
    tmp_path: Path, mutation, expected: str
) -> None:
    from .test_gym_client import _valid_adaptive_spaces
    from ml_lab.doctor import doctor_environment

    response = copy.deepcopy(_valid_adaptive_spaces())
    mutation(response)
    result = doctor_environment(
        server_cmd=["dotnet", "fake-server.dll"],
        environment="adaptive-v1",
        runs_root=tmp_path,
        package_version=lambda _name: "1.0",
        dotnet_version=lambda: "8.0",
        handshake=lambda _command: response,
        cuda_info=lambda: {"available": False, "detail": "CPU"},
        tracker_available=lambda _name: True,
        write_probe=lambda _path: True,
    )

    check = next(item for item in result["checks"] if item["name"] == "gymserver_handshake")
    assert check["ok"] is False
    assert expected in check["detail"]
    assert result["ok"] is False


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
            "--environment",
            "adaptive-v1",
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
            "environment": "adaptive-v1",
            "runs_root": tmp_path,
            "trackers": ["wandb"],
        }
    ]


def test_train_json_reports_the_durable_completed_run(
    tmp_path: Path,
) -> None:
    received: list[RunConfig] = []

    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        scenario: ResolvedScenario,
    ) -> Path:
        received.append(config)
        assert server_cmd == ["dotnet", "fake-server.dll"]
        assert scenario.template_id == "tactical-v2-standard"
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


def test_train_no_console_output_suppresses_envelope_and_requests_file_only_logging(
    tmp_path: Path,
) -> None:
    received: list[bool] = []

    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        console_output: bool,
        scenario: ResolvedScenario,
    ) -> Path:
        del server_cmd, scenario
        received.append(console_output)
        return _complete_fake_run(runs_root, config)

    stdout = StringIO()
    exit_code = cli_module.main(
        [
            "train",
            "--run",
            "detached-train",
            "--timesteps",
            "64",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--no-console-output",
            "--json",
        ],
        stdout=stdout,
        runner=runner,
    )

    assert exit_code == 0
    assert stdout.getvalue() == ""
    assert received == [False]


def test_train_no_console_output_sinks_incidental_runner_stream_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        console_output: bool,
        scenario: ResolvedScenario,
    ) -> Path:
        del server_cmd, console_output, scenario
        print("tracker wrote to stdout")
        print("tracker wrote to stderr", file=sys.stderr)
        return _complete_fake_run(runs_root, config)

    exit_code = cli_module.main(
        [
            "train",
            "--run",
            "detached-streams",
            "--timesteps",
            "64",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--no-console-output",
            "--json",
        ],
        runner=runner,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_train_clean_run_creates_empty_stderr_log_file(tmp_path: Path) -> None:
    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        scenario: ResolvedScenario,
    ) -> Path:
        del server_cmd, scenario
        return _complete_fake_run(runs_root, config)

    exit_code = cli_module.main(
        [
            "train",
            "--run",
            "clean-stderr-log",
            "--timesteps",
            "64",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--json",
        ],
        runner=runner,
        stdout=StringIO(),
    )

    assert exit_code == 0
    log_path = tmp_path / "clean-stderr-log" / "train-err.log"
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8") == ""


def test_train_child_exception_traceback_lands_in_stderr_log(tmp_path: Path) -> None:
    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        scenario: ResolvedScenario,
    ) -> Path:
        del server_cmd, scenario
        # Simulate a trainer that gets partway through startup before crashing.
        run_dir = runs_root / config.run_name
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("simulated trainer crash after startup")

    output = StringIO()
    exit_code = cli_module.main(
        [
            "train",
            "--run",
            "crash-stderr-log",
            "--timesteps",
            "64",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--json",
        ],
        runner=runner,
        stdout=output,
    )

    assert exit_code == 1
    log_path = tmp_path / "crash-stderr-log" / "train-err.log"
    assert log_path.is_file()
    contents = log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in contents
    assert "RuntimeError" in contents
    assert "simulated trainer crash after startup" in contents

    # The --json stdout error protocol must stay untouched by the new stderr capture.
    payload = json.loads(output.getvalue())
    assert payload["ok"] is False
    assert payload["result"]["error"] == "RuntimeError"


def test_train_serializes_wandb_and_custom_tracker_configuration_without_secrets(
    tmp_path: Path,
) -> None:
    received: list[RunConfig] = []

    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        scenario: ResolvedScenario,
    ) -> Path:
        del scenario
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


def test_structured_train_forwards_independent_target_scenario_and_tensorboard(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-policy"
    source.mkdir()
    scenario = tmp_path / "target-scenario.json"
    scenario.write_text("{}\n", encoding="utf-8")
    received = []

    def structured_runner(config, *, runs_root: Path, server_cmd: list[str]) -> Path:
        received.append((config, server_cmd))
        run_dir = runs_root / config.run_name
        run_dir.mkdir(exist_ok=True)
        atomic_write_json(run_dir / "run.json", {
            "schema_version": 1,
            "state": "completed",
            "config": {
                "run_name": config.run_name,
                "total_timesteps": config.train_label_target,
                "learner_seat": config.learner_seat,
            },
        })
        return run_dir

    output = StringIO()
    exit_code = cli_module.main(
        [
            "train-structured",
            "--run", "latest-vs-greedy",
            "--source-run", str(source),
            "--scenario-file", str(scenario),
            "--opponent", "greedy",
            "--train-labels", "7500",
            "--validation-labels", "3000",
            "--seed", "227",
            "--device", "cuda:0",
            "--learner-seat", "alternating",
            "--tracker", "local",
            "--tracker", "tensorboard",
            "--runs-root", str(tmp_path / "runs"),
            "--server", "fake-server.dll",
            "--json",
        ],
        structured_runner=structured_runner,
        stdout=output,
    )

    assert exit_code == 0
    _assert_envelope(json.loads(output.getvalue()), "train-structured")
    config, command = received[0]
    assert config.source_run == source
    assert config.scenario_file == scenario
    assert config.opponent == "greedy"
    assert config.train_label_target == 7500
    assert config.validation_label_target == 3000
    assert config.device == "cuda:0"
    assert config.trackers == ({"kind": "local"}, {"kind": "tensorboard"})
    assert command == [
        "dotnet", "fake-server.dll", "--scenario-file", str(scenario),
    ]


def test_structured_preflight_parser_has_training_compatible_defaults(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    scenario = tmp_path / "scenario.json"

    args = cli_module.build_parser().parse_args([
        "preflight-structured",
        "--source-run", str(source),
        "--scenario-file", str(scenario),
        "--json",
    ])

    assert args.command == "preflight-structured"
    assert args.source_run == source
    assert args.scenario_file == scenario
    assert args.opponent == "greedy"
    assert args.seed == 227
    assert args.device == "auto"
    assert args.server == str(cli_module.DEFAULT_SERVER)
    assert args.json is True


def test_structured_preflight_authenticates_and_cross_checks_without_creating_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import ml_lab.tactical_v3_checkpoint as checkpoint_module
    import ml_lab.tactical_v3_client as client_module
    from ml_lab.tactical_v3_pilot import _pilot_configs
    from tests.tactical_v3_fixture_support import load_duel_identity_fixture

    identity = load_duel_identity_fixture()
    model_config, _, _ = _pilot_configs(227, "cpu")
    source = tmp_path / "source-policy"
    validated: list[Path] = []
    server_starts: list[tuple[list[str], str]] = []

    def validate(run_dir: Path):
        validated.append(run_dir)
        return SimpleNamespace(
            metadata=SimpleNamespace(identity=identity),
            model=SimpleNamespace(config=model_config),
        )

    class Client:
        def __init__(self, server_cmd, *, environment_kind: str):
            server_starts.append((list(server_cmd), environment_kind))
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(checkpoint_module, "validate_structured_run", validate)
    monkeypatch.setattr(client_module, "TacticalV3GymClient", Client)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    exit_code, payload = _invoke_json([
        "preflight-structured",
        "--source-run", str(source),
        "--scenario-file", str(TACTICAL_V3_SCENARIO),
        "--opponent", "random",
        "--seed", "227",
        "--device", "auto",
        "--server", "fake-server.dll",
        "--json",
    ])

    assert exit_code == 0
    result = _assert_envelope(payload, "preflight-structured")
    assert result.keys() == {
        "environment", "source", "target", "opponent", "device", "model_config",
    }
    assert result["environment"] == "tactical-v3"
    assert result["source"] == {
        "run_dir": str(source.resolve()),
        "checkpoint": str(source.resolve() / "checkpoints" / "best.pt"),
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }
    assert result["target"] == {
        "scenario_file": str(TACTICAL_V3_SCENARIO.resolve()),
        "scenario_id": identity.scenario_id,
        "scenario_schema_version": identity.scenario_schema_version,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }
    assert result["opponent"] == {"kind": "scripted", "name": "random"}
    assert result["device"] == {"requested": "auto", "effective": "cpu"}
    assert result["model_config"]["hidden_dim"] == 32
    assert validated == [source]
    assert server_starts == [([
        "dotnet", "fake-server.dll", "--scenario-file", str(TACTICAL_V3_SCENARIO),
    ], "duel")]
    assert not source.exists()


def test_structured_preflight_rejects_corrupt_source_before_starting_gymserver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_checkpoint as checkpoint_module
    import ml_lab.tactical_v3_client as client_module

    def reject(_run_dir: Path):
        raise ValueError("checkpoint state hash is inconsistent")

    class Client:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("GymServer must not start for a corrupt source")

    monkeypatch.setattr(checkpoint_module, "validate_structured_run", reject)
    monkeypatch.setattr(client_module, "TacticalV3GymClient", Client)

    exit_code, payload = _invoke_json([
        "preflight-structured",
        "--source-run", str(tmp_path / "corrupt-source"),
        "--scenario-file", str(TACTICAL_V3_SCENARIO),
        "--device", "cpu",
        "--json",
    ])

    assert exit_code == 1
    assert payload == {
        "schema_version": 1,
        "command": "preflight-structured",
        "ok": False,
        "result": {
            "error": "ValueError",
            "message": "checkpoint state hash is inconsistent",
        },
    }


def test_structured_preflight_rejects_wrong_source_architecture_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_checkpoint as checkpoint_module
    import ml_lab.tactical_v3_client as client_module
    from ml_lab.tactical_v3_pilot import _pilot_configs
    from tests.tactical_v3_fixture_support import load_duel_identity_fixture

    identity = load_duel_identity_fixture()
    expected, _, _ = _pilot_configs(227, "cpu")
    monkeypatch.setattr(
        checkpoint_module,
        "validate_structured_run",
        lambda _path: SimpleNamespace(
            metadata=SimpleNamespace(identity=identity),
            model=SimpleNamespace(config=replace(expected, hidden_dim=64)),
        ),
    )

    class Client:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("GymServer must not start for wrong architecture")

    monkeypatch.setattr(client_module, "TacticalV3GymClient", Client)

    exit_code, payload = _invoke_json([
        "preflight-structured",
        "--source-run", str(tmp_path / "wrong-architecture"),
        "--scenario-file", str(TACTICAL_V3_SCENARIO),
        "--device", "cpu",
        "--json",
    ])

    assert exit_code == 1
    assert payload["result"] == {
        "error": "ValueError",
        "message": "source policy model config does not match the continuation model",
    }


def test_structured_preflight_rejects_unavailable_or_out_of_range_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA but it is unavailable"):
        cli_module._structured_preflight_device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert cli_module._structured_preflight_device("cuda:0") == "cuda:0"
    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        cli_module._structured_preflight_device("cuda:1")


@pytest.mark.parametrize(
    ("mode", "expected_kind"),
    (("fixed", "fixed_run"), ("live", "live_run")),
)
def test_structured_preflight_authenticates_fixed_and_live_model_opponents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_kind: str,
) -> None:
    import ml_lab.tactical_v3_client as client_module
    from ml_lab.tactical_v3_checkpoint import publish_structured_run
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_pilot import _pilot_configs
    from ml_lab.tactical_v3_training import EpochMetrics, TrainingResult
    from tests.tactical_v3_fixture_support import (
        load_duel_identity_fixture,
        load_tiny_corpus_fixture,
    )

    identity = load_duel_identity_fixture()
    corpus = load_tiny_corpus_fixture()
    model_config, objective_config, trainer_config = _pilot_configs(227, "cpu")
    zero_metrics = {
        "total": 0.0,
        "policy": 0.0,
        "outcome": 0.0,
        "horizon": 0.0,
        "remaining_turns": 0.0,
    }
    result = TrainingResult(
        model=TacticalV3Policy(model_config).eval(),
        model_config=model_config,
        objective_config=objective_config,
        trainer_config=trainer_config,
        best_epoch=0,
        best_validation_policy_nll=0.0,
        stopped_early=False,
        history=(EpochMetrics(0, zero_metrics, zero_metrics, 0.0, True),),
    )
    policy_run = publish_structured_run(
        tmp_path / "policy",
        result,
        corpus,
        training_scenario_path=TACTICAL_V3_SCENARIO,
        policy_identity=identity,
    )

    class Client:
        def __init__(self, _server_cmd, *, environment_kind: str):
            assert environment_kind == "duel"
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(client_module, "TacticalV3GymClient", Client)
    opponent = (
        str(policy_run)
        if mode == "fixed"
        else json.dumps({"kind": "run", "path": str(policy_run), "mode": "live"})
    )

    exit_code, payload = _invoke_json([
        "preflight-structured",
        "--source-run", str(policy_run),
        "--scenario-file", str(TACTICAL_V3_SCENARIO),
        "--opponent", opponent,
        "--device", "cpu",
        "--json",
    ])

    assert exit_code == 0
    resolved = _assert_envelope(payload, "preflight-structured")["opponent"]
    assert resolved["kind"] == expected_kind
    assert resolved["mode"] == mode
    assert resolved["source_run"] == str(policy_run.resolve())
    assert resolved["checkpoint"] == str(policy_run / "checkpoints" / "best.pt")
    assert len(resolved["checkpoint_sha256"]) == 64
    assert resolved["algorithm"] == "structured_imitation"


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

    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        scenario: ResolvedScenario,
    ) -> Path:
        assert scenario.template_id == "tactical-standard"
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


def test_resume_does_not_reapply_actor_initialization(
    tmp_path: Path,
    contract: EnvironmentContract,
) -> None:
    source = create_run(
        tmp_path,
        replace(
            _config("actor-initialized-source"),
            actor_init_source="clone/source",
        ),
        contract,
    )
    args = cli_module.build_parser().parse_args(
        [
            "resume",
            str(source),
            "--run",
            "resumed-ppo",
            "--timesteps",
            "128",
        ]
    )

    resumed = cli_module._resume_config(args)

    assert resumed.resume_source == str(source.resolve())
    assert resumed.actor_init_source is None


def test_resume_no_console_output_suppresses_envelope_and_requests_file_only_logging(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source = create_run(tmp_path, _config("detached-source"), contract)
    received: list[bool] = []

    def runner(
        config: RunConfig,
        *,
        runs_root: Path,
        server_cmd: list[str],
        console_output: bool,
        scenario: ResolvedScenario,
    ) -> Path:
        del server_cmd
        assert scenario.template_id == "tactical-standard"
        received.append(console_output)
        return _complete_fake_run(runs_root, config)

    stdout = StringIO()
    exit_code = cli_module.main(
        [
            "resume",
            str(source),
            "--run",
            "detached-resume",
            "--timesteps",
            "160",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
            "--no-console-output",
            "--json",
        ],
        stdout=stdout,
        runner=runner,
    )

    assert exit_code == 0
    assert stdout.getvalue() == ""
    assert received == [False]


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


def test_run_result_aggregates_learner_seats_from_manifest_monitor_shards(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(
        tmp_path, replace(_config("seat-audit"), workers=2), contract
    )
    manifest = read_json(run_dir / "run.json")
    first, second = [run_dir / relative for relative in manifest["monitor_files"]]
    first.write_text(
        "worker_id,episode_index,episode_seed,learner_seat,"
        "episode_reward,episode_length,elapsed_seconds\n"
        "0,0,17,0,1.0,10,0.1\n"
        "0,1,19,1,-1.0,11,0.2\n"
        "0,2,21,0,1.0,12,0.3\n",
        encoding="utf-8",
    )
    second.write_text(
        "worker_id,episode_index,episode_seed,learner_seat,"
        "episode_reward,episode_length,elapsed_seconds\n"
        "1,0,18,1,1.0,9,0.1\n"
        "1,1,20,0,-1.0,8,0.2\n",
        encoding="utf-8",
    )

    audit = cli_module._run_result(run_dir)["seat_audit"]

    assert audit == {
        "seat_0_episodes": 3,
        "seat_1_episodes": 2,
        "readable": True,
        "balanced": True,
        "warning": "",
    }


def test_run_result_reports_path_for_malformed_monitor_seat(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("malformed-seat-audit"), contract)
    manifest = read_json(run_dir / "run.json")
    monitor_path = run_dir / manifest["monitor_files"][0]
    monitor_path.write_text(
        "worker_id,episode_index,episode_seed,learner_seat,"
        "episode_reward,episode_length,elapsed_seconds\n"
        "0,0,17,sideways,1.0,10,0.1\n",
        encoding="utf-8",
    )

    audit = cli_module._run_result(run_dir)["seat_audit"]

    assert audit["readable"] is False
    assert str(monitor_path) in audit["warning"]
    assert "learner_seat" in audit["warning"]


def test_run_result_reports_path_for_monitor_missing_learner_seat_header(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, _config("missing-header-audit"), contract)
    manifest = read_json(run_dir / "run.json")
    monitor_path = run_dir / manifest["monitor_files"][0]
    monitor_path.write_text(
        "worker_id,episode_reward\n0,1.0\n",
        encoding="utf-8",
    )

    audit = cli_module._run_result(run_dir)["seat_audit"]

    assert audit["readable"] is False
    assert str(monitor_path) in audit["warning"]
    assert "learner_seat" in audit["warning"]
    assert "header" in audit["warning"].lower()


@pytest.mark.parametrize(
    ("state", "learner_seat", "expect_warning"),
    [
        ("running", "alternating", False),
        ("completed", "alternating", True),
        ("completed", "0", False),
    ],
)
def test_run_result_only_warns_for_terminal_alternating_imbalance(
    tmp_path: Path,
    contract: EnvironmentContract,
    state: str,
    learner_seat: str,
    expect_warning: bool,
) -> None:
    run_dir = create_run(
        tmp_path,
        replace(_config("seat-warning"), learner_seat=learner_seat),
        contract,
    )
    manifest = read_json(run_dir / "run.json")
    manifest["state"] = state
    atomic_write_json(run_dir / "run.json", manifest)
    monitor_path = run_dir / manifest["monitor_files"][0]
    monitor_path.write_text(
        "worker_id,episode_index,episode_seed,learner_seat,"
        "episode_reward,episode_length,elapsed_seconds\n"
        "0,0,17,0,1.0,10,0.1\n"
        "0,1,18,0,1.0,10,0.2\n"
        "0,2,19,0,1.0,10,0.3\n",
        encoding="utf-8",
    )

    audit = cli_module._run_result(run_dir)["seat_audit"]

    assert audit["balanced"] is False
    assert bool(audit["warning"]) is expect_warning


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
            "environment": None,
            "start_profile": None,
            "capture_trace": False,
            "evidence_dir": None,
            "evidence_retention": "diagnostic",
        }
    ]


def test_evidence_directory_enables_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_controllers(*args, **kwargs):
        captured.update(kwargs)
        return {"wins": 0, "losses": 0, "draws": 1, "games": 1}

    monkeypatch.setattr(cli_module, "evaluate_controllers", fake_evaluate_controllers)
    assert cli_module.main(
        [
            "evaluate",
            "--p0",
            "greedy",
            "--p1",
            "random",
            "--games",
            "1",
            "--environment",
            "tactical-v2",
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ],
        stdout=StringIO(),
    ) == 0

    assert captured["capture_trace"] is True
    assert captured["evidence_dir"] == tmp_path / "evidence"
    assert captured["environment"] == "tactical-v2"


def test_evaluate_cli_propagates_profile_and_evidence_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_controllers(*args, **kwargs):
        captured.update(kwargs)
        return {"wins": 0, "losses": 0, "draws": 1, "games": 1}

    monkeypatch.setattr(cli_module, "evaluate_controllers", fake_evaluate_controllers)
    assert cli_module.main(
        [
            "evaluate",
            "--p0",
            "random",
            "--p1",
            "random",
            "--games",
            "1",
            "--both-seats",
            "--environment",
            "tactical-v2",
            "--start-profile",
            "standard-3v3",
            "--capture-trace",
            "--evidence-retention",
            "all",
            "--evidence-dir",
            r"C:\temp\audit-evidence",
        ],
        stdout=StringIO(),
    ) == 0

    assert captured["start_profile"] == "standard-3v3"
    assert captured["evidence_retention"] == "all"


def test_explicit_trace_capture_does_not_require_evidence_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_controllers(*args, **kwargs):
        captured.update(kwargs)
        return {"wins": 1, "losses": 0, "draws": 0, "games": 1}

    monkeypatch.setattr(cli_module, "evaluate_controllers", fake_evaluate_controllers)
    assert cli_module.main(
        [
            "evaluate",
            "--p0",
            "greedy",
            "--p1",
            "random",
            "--games",
            "1",
            "--capture-trace",
        ],
        stdout=StringIO(),
    ) == 0

    assert captured["capture_trace"] is True
    assert captured["evidence_dir"] is None
    assert captured["environment"] is None


def test_module_entrypoint_renders_evaluate_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_lab.cli",
            "evaluate",
            "--help",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--capture-trace" in completed.stdout
    assert "--evidence-dir" in completed.stdout
    assert "{tactical-v1,tactical-v2,adaptive-v1}" in completed.stdout


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
            "--environment",
            "adaptive-v1",
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
            "environment": "adaptive-v1",
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
