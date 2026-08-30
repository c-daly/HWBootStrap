"""Stable entry point launched by the Unity ML Lab and command-line users."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Iterator, Sequence, TextIO


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TRAINING_COMMANDS = frozenset({"train", "train-structured", "resume"})


def _option_value(argv: Sequence[str], option: str) -> str | None:
    value = None
    for index, argument in enumerate(argv):
        if argument == "--":
            break
        if argument == option:
            value = argv[index + 1] if index + 1 < len(argv) else None
        elif argument.startswith(option + "="):
            value = argument[len(option) + 1 :]
    return value


def _startup_error_path(argv: Sequence[str]) -> Path | None:
    if not argv or argv[0] not in _TRAINING_COMMANDS:
        return None
    run_name = _option_value(argv, "--run")
    if run_name is None or _SAFE_RUN_NAME.fullmatch(run_name) is None:
        return None
    runs_root = _option_value(argv, "--runs-root")
    root = Path(runs_root) if runs_root else Path(__file__).resolve().parent / "runs"
    return root / run_name / "train-err.log"


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _open_startup_log(path: Path) -> TextIO:
    """Open a run-local append log without following a run/log link."""
    requested_root = path.parent.parent
    requested_root.mkdir(parents=True, exist_ok=True)
    runs_root = requested_root.resolve(strict=True)
    run_dir = runs_root / path.parent.name
    if _is_link_like(run_dir):
        raise OSError(f"refusing linked ML Lab run directory: {run_dir}")
    run_dir.mkdir(exist_ok=True)
    if not run_dir.is_dir() or run_dir.resolve(strict=True).parent != runs_root:
        raise OSError(f"ML Lab run directory escaped its runs root: {run_dir}")

    log_path = run_dir / path.name
    if _is_link_like(log_path):
        raise OSError(f"refusing linked ML Lab startup log: {log_path}")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(log_path, flags | no_follow, 0o600)
    return open(
        fd,
        "a",
        buffering=1,
        encoding="utf-8",
        errors="replace",
        closefd=True,
    )


@contextmanager
def _capture_process_stderr(stream: TextIO) -> Iterator[None]:
    """Temporarily point Python and native process stderr at ``stream``."""
    original_stderr = sys.stderr
    saved_fd: int | None = None
    target_existed = True
    stderr_view: TextIO | None = None
    try:
        try:
            saved_fd = os.dup(2)
        except OSError:
            target_existed = False
        os.dup2(stream.fileno(), 2, inheritable=True)
        stderr_view = open(
            2,
            "w",
            buffering=1,
            encoding="utf-8",
            errors="replace",
            closefd=False,
        )
        sys.stderr = stderr_view
        yield
    finally:
        if stderr_view is not None:
            try:
                stderr_view.flush()
            except OSError:
                pass
        sys.stderr = original_stderr
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 2, inheritable=True)
            finally:
                os.close(saved_fd)
        elif not target_existed:
            try:
                os.close(2)
            except OSError:
                pass
        if stderr_view is not None:
            stderr_view.close()


def _sensitive_values(argv: Sequence[str]) -> set[str]:
    markers = ("api-key", "apikey", "password", "secret", "token")
    values: set[str] = set()
    for index, argument in enumerate(argv):
        if argument == "--":
            break
        if not argument.startswith("--"):
            continue
        option, separator, inline_value = argument.partition("=")
        if not any(marker in option.casefold() for marker in markers):
            continue
        if separator:
            if inline_value:
                values.add(inline_value)
        elif index + 1 < len(argv):
            values.add(argv[index + 1])
    return values


def _redact_startup_log(path: Path, argv: Sequence[str]) -> None:
    sensitive_values = sorted(_sensitive_values(argv), key=len, reverse=True)
    if not sensitive_values:
        return
    contents = path.read_text(encoding="utf-8")
    for value in sensitive_values:
        contents = contents.replace(value, "[REDACTED]")
    path.write_text(contents, encoding="utf-8")


def _record_log_setup_failure(path: Path, error: BaseException) -> None:
    """Best-effort diagnostic when even the requested run log cannot be opened."""
    fallback = Path(tempfile.gettempdir()) / "hexwars-ml-startup-errors.log"
    try:
        with fallback.open("a", encoding="utf-8", errors="replace") as stream:
            timestamp = datetime.now(timezone.utc).isoformat()
            stream.write(f"{timestamp} could not open {path}: {error}\n")
            traceback.print_exception(error, file=stream)
    except OSError:
        traceback.print_exception(error)


def run(argv: Sequence[str] | None = None) -> int:
    effective_argv = tuple(sys.argv[1:] if argv is None else argv)
    startup_log = _startup_error_path(effective_argv)
    if startup_log is None:
        from ml_lab.cli import main

        return main(list(effective_argv))

    try:
        stream = _open_startup_log(startup_log)
    except (OSError, ValueError) as error:
        _record_log_setup_failure(startup_log, error)
        return 1

    exit_code = 0
    failed = False
    with stream, _capture_process_stderr(stream):
        try:
            from ml_lab.cli import main

            exit_code = main(list(effective_argv))
            failed = exit_code != 0
            if failed:
                print(f"ML Lab startup exited with code {exit_code}", file=sys.stderr)
        except SystemExit as error:
            if error.code is None:
                exit_code = 0
            elif isinstance(error.code, int):
                exit_code = error.code
            else:
                print(error.code, file=sys.stderr)
                exit_code = 1
            failed = exit_code != 0
            if failed:
                print(f"ML Lab startup exited with code {exit_code}", file=sys.stderr)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
            failed = True

    if failed:
        _redact_startup_log(startup_log, effective_argv)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
