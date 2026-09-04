from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hexwars_ml
import ml_lab.cli as cli_module
import pytest


PYTHON_ROOT = Path(__file__).parents[1]
ENTRYPOINT = PYTHON_ROOT / "hexwars_ml.py"


def test_startup_error_path_is_limited_to_safe_training_run_names(
    tmp_path: Path,
) -> None:
    assert hexwars_ml._startup_error_path(
        [
            "train-structured",
            "--run",
            "continuation-1",
            "--runs-root",
            str(tmp_path),
        ]
    ) == tmp_path / "continuation-1" / "train-err.log"

    assert hexwars_ml._startup_error_path(
        [
            "retry-structured",
            "--collection-run",
            str(tmp_path / "old-collection"),
            "--run",
            "retry-1",
            "--runs-root",
            str(tmp_path),
        ]
    ) == tmp_path / "retry-1" / "train-err.log"

    assert hexwars_ml._startup_error_path(
        [
            "train-outcome",
            "--run", "close-candidate",
            "--runs-root", str(tmp_path),
        ]
    ) == tmp_path / "close-candidate" / "train-err.log"

    assert hexwars_ml._startup_error_path(
        ["evaluate", "--run", "evaluation", "--runs-root", str(tmp_path)]
    ) is None
    assert hexwars_ml._startup_error_path(
        ["train", "--run", "../outside", "--runs-root", str(tmp_path)]
    ) is None
    assert hexwars_ml._startup_error_path(
        [
            "train",
            "--run=earlier",
            "--run=equals-form",
            f"--runs-root={tmp_path}",
        ]
    ) == tmp_path / "equals-form" / "train-err.log"
    assert hexwars_ml._startup_error_path(
        ["train", "--", "--run=not-an-option", f"--runs-root={tmp_path}"]
    ) is None


@pytest.mark.parametrize("protected_kind", ("selected", "owner", "source"))
def test_retry_entrypoint_does_not_create_a_log_inside_a_protected_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    runs = tmp_path / "runs"
    selected = runs / "selected"
    selected.mkdir(parents=True)
    manifest: dict[str, object] = {}
    collection: dict[str, object] = {}

    if protected_kind == "selected":
        runs_root = selected
        run_name = "child"
        destination = selected / run_name
    elif protected_kind == "owner":
        destination = runs / "owner"
        destination.mkdir()
        manifest["collection_source_run"] = r"Z:\archived-runs\owner"
        runs_root = runs
        run_name = destination.name
    else:
        destination = runs / "source-policy"
        destination.mkdir()
        collection["source"] = {"run": r"Z:\archived-runs\source-policy"}
        runs_root = runs
        run_name = destination.name

    (selected / "run.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (selected / "collection.json").write_text(
        json.dumps(collection), encoding="utf-8",
    )
    argv = [
        "retry-structured",
        "--collection-run", str(selected),
        "--run", run_name,
        "--runs-root", str(runs_root),
        "--no-console-output",
        "--json",
    ]
    called = False

    def fake_main(_argv: list[str]) -> int:
        nonlocal called
        called = True
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml._startup_error_path(argv) is None
    assert hexwars_ml.run(argv) == 1
    assert called is True
    assert not (destination / "train-err.log").exists()
    if protected_kind == "selected":
        assert not destination.exists()


def test_retry_entrypoint_keeps_normal_sibling_startup_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    selected = runs / "selected"
    selected.mkdir(parents=True)
    (selected / "run.json").write_text("{}\n", encoding="utf-8")
    direct_source = tmp_path / "archived" / "safe-sibling"
    direct_source.mkdir(parents=True)
    (selected / "collection.json").write_text(
        json.dumps({"source": {"run": str(direct_source)}}),
        encoding="utf-8",
    )
    argv = [
        "retry-structured",
        "--collection-run", str(selected),
        "--run", "safe-sibling",
        "--runs-root", str(runs),
    ]
    monkeypatch.setattr(cli_module, "main", lambda _argv: 0)

    assert hexwars_ml.run(argv) == 0
    contents = (runs / "safe-sibling" / "train-err.log").read_text(
        encoding="utf-8",
    )
    assert "ML Lab startup began with pid " in contents
    assert "ML Lab startup exited with code 0" in contents


@pytest.mark.parametrize(
    "protected_kind",
    (
        "initialization",
        "fixed_opponent",
        "live_opponent",
        "direct_opponent",
        "file_opponent",
    ),
)
def test_outcome_entrypoint_does_not_open_a_log_inside_model_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    runs = tmp_path / "runs"
    source = runs / "protected"
    source.mkdir(parents=True)
    log = source / "train-err.log"
    log.write_text("source diagnostics\n", encoding="utf-8")
    (source / "run.json").write_text("{}\n", encoding="utf-8")
    spec_file = tmp_path / "opponent.json"
    spec_file.write_text(
        json.dumps({"kind": "run", "path": str(source), "mode": "fixed"}),
        encoding="utf-8",
    )
    opponent_values = {
        "fixed_opponent": f"run:{source}",
        "live_opponent": json.dumps({
            "kind": "run", "path": str(source), "mode": "live",
        }),
        "direct_opponent": str(source),
        "file_opponent": f"@{spec_file}",
    }
    source_arguments = ["--source-run", str(source)] if (
        protected_kind == "initialization"
    ) else ["--opponent", opponent_values[protected_kind]]
    argv = [
        "train-outcome",
        "--run", source.name,
        "--runs-root", str(runs),
        *source_arguments,
        "--json",
    ]
    received = []

    def fake_main(actual_argv: list[str]) -> int:
        received.append(actual_argv)
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml._startup_error_path(argv) is None
    assert hexwars_ml.run(argv) == 1
    assert received == [argv]
    assert log.read_text(encoding="utf-8") == "source diagnostics\n"


def test_outcome_entrypoint_keeps_normal_sibling_startup_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    source = runs / "source"
    source.mkdir(parents=True)
    source_log = source / "train-err.log"
    source_log.write_text("source diagnostics\n", encoding="utf-8")
    argv = [
        "train-outcome",
        "--run", "candidate",
        "--source-run", str(source),
        "--runs-root", str(runs),
    ]
    monkeypatch.setattr(cli_module, "main", lambda _argv: 0)

    destination_log = runs / "candidate" / "train-err.log"
    assert hexwars_ml._startup_error_path(argv) == destination_log
    assert hexwars_ml.run(argv) == 0
    assert source_log.read_text(encoding="utf-8") == "source diagnostics\n"
    contents = destination_log.read_text(encoding="utf-8")
    assert "ML Lab startup began with pid " in contents
    assert "ML Lab startup exited with code 0" in contents


def test_outcome_entrypoint_keeps_scripted_name_precedence_over_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_like_script = tmp_path / "random"
    named_like_script.mkdir()
    (named_like_script / "run.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runs = named_like_script / "runs"
    argv = [
        "train-outcome",
        "--run", "candidate",
        "--opponent", "random",
        "--runs-root", str(runs),
    ]
    monkeypatch.setattr(cli_module, "main", lambda _argv: 0)

    destination_log = runs / "candidate" / "train-err.log"
    assert hexwars_ml._startup_error_path(argv) == destination_log
    assert hexwars_ml.run(argv) == 0
    contents = destination_log.read_text(encoding="utf-8")
    assert "ML Lab startup began with pid " in contents
    assert "ML Lab startup exited with code 0" in contents


def test_outcome_entrypoint_does_not_log_inside_empty_source_run_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.chdir(source)
    argv = [
        "train-outcome",
        "--run", "child",
        "--source-run=",
        "--runs-root", ".",
    ]
    received = []

    def fake_main(actual_argv: list[str]) -> int:
        received.append(actual_argv)
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml._startup_error_path(argv) is None
    assert hexwars_ml.run(argv) == 1
    assert received == [argv]
    assert not (source / "child").exists()


@pytest.mark.parametrize("value", ("run:", "ppo:", "dqn:"))
def test_outcome_entrypoint_ignores_empty_controller_source_paths(value: str) -> None:
    assert hexwars_ml._controller_source_path(value) is None


def test_training_argparse_failure_is_retained_in_run_stderr_log(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRYPOINT),
            "train-structured",
            "--run",
            "entrypoint-failure",
            "--runs-root",
            str(tmp_path),
            "--source-run",
            str(tmp_path / "source"),
            "--scenario-file",
            str(tmp_path / "source" / "scenario.json"),
            "--train-labels",
            "2",
            "--validation-labels",
            "2",
            "--api-token",
            "do-not-persist-this-secret",
            "--not-a-real-option",
        ],
        cwd=PYTHON_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == ""

    log_path = tmp_path / "entrypoint-failure" / "train-err.log"
    contents = log_path.read_text(encoding="utf-8")
    assert "ML Lab startup began with pid " in contents
    assert "--not-a-real-option" in contents
    assert "error: unrecognized arguments:" in contents
    assert "do-not-persist-this-secret" not in contents
    assert "[REDACTED]" in contents
    assert "ML Lab startup exited with code 2" in contents


def test_valid_entrypoint_preserves_inner_native_stderr_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "native-stderr" / "train-err.log"

    def fake_main(_argv: list[str]) -> int:
        assert sys.stderr.fileno() == 2
        with cli_module._capture_stderr_to_file(log_path):
            os.write(2, b"native extension stderr reached the run log\n")
        return 0

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml.run(
        [
            "train-structured",
            "--run=native-stderr",
            f"--runs-root={tmp_path}",
        ]
    ) == 0
    contents = log_path.read_text(encoding="utf-8")
    assert "native extension stderr reached the run log" in contents
    assert "ML Lab startup exited with code 0" in contents


def test_startup_log_is_not_marked_exited_until_main_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "live-startup" / "train-err.log"

    def fake_main(_argv: list[str]) -> int:
        contents = log_path.read_text(encoding="utf-8")
        assert "ML Lab startup began with pid " in contents
        assert "ML Lab startup exited with code " not in contents
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml.run(
        ["train", "--run", "live-startup", "--runs-root", str(tmp_path)]
    ) == 1
    assert "ML Lab startup exited with code 1" in log_path.read_text(
        encoding="utf-8"
    )


def test_non_integer_system_exit_message_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_main(_argv: list[str]) -> int:
        raise SystemExit("import hook refused startup")

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml.run(
        ["train", "--run", "system-exit", "--runs-root", str(tmp_path)]
    ) == 1
    contents = (tmp_path / "system-exit" / "train-err.log").read_text(
        encoding="utf-8"
    )
    assert "import hook refused startup" in contents
    assert "ML Lab startup exited with code 1" in contents


def test_returned_failure_is_marked_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_main(_argv: list[str]) -> int:
        print("backend rejected secret-value", file=sys.stderr)
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml.run(
        [
            "train",
            "--run",
            "returned-failure",
            "--runs-root",
            str(tmp_path),
            "--api-token",
            "secret-value",
        ]
    ) == 1
    contents = (tmp_path / "returned-failure" / "train-err.log").read_text(
        encoding="utf-8"
    )
    assert "secret-value" not in contents
    assert "backend rejected [REDACTED]" in contents
    assert "ML Lab startup exited with code 1" in contents


def test_terminal_marker_is_appended_after_short_secret_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_main(_argv: list[str]) -> int:
        print("backend rejected 1", file=sys.stderr)
        return 1

    monkeypatch.setattr(cli_module, "main", fake_main)

    assert hexwars_ml.run(
        [
            "train",
            "--run",
            "short-secret",
            "--runs-root",
            str(tmp_path),
            "--api-token",
            "1",
        ]
    ) == 1
    contents = (tmp_path / "short-secret" / "train-err.log").read_text(
        encoding="utf-8"
    )
    assert "backend rejected [REDACTED]" in contents
    assert contents.rstrip().endswith("ML Lab startup exited with code 1")


def test_startup_log_refuses_linked_run_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_run = tmp_path / "linked-run"
    try:
        linked_run.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(OSError, match="linked ML Lab run directory"):
        hexwars_ml._open_startup_log(linked_run / "train-err.log")
    assert not (outside / "train-err.log").exists()
