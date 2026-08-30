from __future__ import annotations

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
    assert "native extension stderr reached the run log" in log_path.read_text(
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
