"""Stable entry point launched by the Unity ML Lab and command-line users."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Iterator, Sequence, TextIO


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TRAINING_COMMANDS = frozenset({
    "train", "train-structured", "train-outcome", "retry-structured", "resume",
})
_STARTUP_BEGAN_PREFIX = "ML Lab startup began with pid "
_STARTUP_EXITED_PREFIX = "ML Lab startup exited with code "


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


def _comparison_path(path: Path) -> Path:
    return Path(path).resolve(strict=False)


def _recorded_run_candidates(value: object, runs_root: Path) -> tuple[Path, ...]:
    if type(value) is not str or not value.strip():
        return ()
    direct = Path(value)
    basename = Path(value.replace("\\", "/")).name
    candidates = [direct]
    if basename:
        fallback = Path(runs_root) / basename
        if fallback != direct:
            candidates.append(fallback)
    return tuple(candidates)


def _recorded_run_protections(value: object, runs_root: Path) -> tuple[Path, ...]:
    """Protect candidates up to the one strict retry resolution would select."""

    protected: list[Path] = []
    for candidate in _recorded_run_candidates(value, runs_root):
        protected.append(candidate)
        is_junction = getattr(candidate, "is_junction", None)
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and not (is_junction is not None and is_junction())
        ):
            break
    return tuple(protected)


def _quiet_json_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return value if type(value) is dict else None


def _retry_protected_runs(collection_run: Path, runs_root: Path) -> set[Path]:
    """Discover retry sources without importing the ML stack or creating a log."""

    selected = _comparison_path(collection_run)
    protected = {selected}
    manifest = _quiet_json_object(selected / "run.json")
    owner_candidates: list[Path] = []
    if manifest is not None:
        for candidate in _recorded_run_protections(
            manifest.get("collection_source_run"), runs_root,
        ):
            resolved = _comparison_path(candidate)
            protected.add(resolved)
            owner_candidates.append(resolved)

        source_policy = manifest.get("source_policy")
        if type(source_policy) is dict:
            for candidate in _recorded_run_protections(
                source_policy.get("run"), runs_root,
            ):
                protected.add(_comparison_path(candidate))
        config = manifest.get("config")
        if type(config) is dict:
            for candidate in _recorded_run_protections(
                config.get("initialization_source"), runs_root,
            ):
                protected.add(_comparison_path(candidate))

    for run in (selected, *owner_candidates):
        collection = _quiet_json_object(run / "collection.json")
        if collection is None:
            continue
        source = collection.get("source")
        if type(source) is not dict:
            continue
        for candidate in _recorded_run_protections(source.get("run"), runs_root):
            protected.add(_comparison_path(candidate))
    return protected


def _retry_log_destination_conflicts(
    argv: Sequence[str], runs_root: Path, destination: Path,
) -> bool:
    if not argv or argv[0] != "retry-structured":
        return False
    collection_value = _option_value(argv, "--collection-run")
    if collection_value is None:
        return False
    resolved_destination = _comparison_path(destination)
    return any(
        resolved_destination == source or source in resolved_destination.parents
        for source in _retry_protected_runs(Path(collection_value), runs_root)
    )


def _startup_error_path(argv: Sequence[str]) -> Path | None:
    if not argv or argv[0] not in _TRAINING_COMMANDS:
        return None
    run_name = _option_value(argv, "--run")
    if run_name is None or _SAFE_RUN_NAME.fullmatch(run_name) is None:
        return None
    runs_root = _option_value(argv, "--runs-root")
    root = Path(runs_root) if runs_root else Path(__file__).resolve().parent / "runs"
    run_dir = root / run_name
    if _retry_log_destination_conflicts(argv, root, run_dir):
        return None
    return run_dir / "train-err.log"


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
        # Invalidate any prior terminal marker before entering main. The inner CLI
        # may subsequently truncate this line when it installs its native-stderr
        # capture; either a began marker or no terminal marker is fail-closed in
        # Unity until this process really exits.
        print(f"{_STARTUP_BEGAN_PREFIX}{os.getpid()}", file=sys.stderr)
        try:
            from ml_lab.cli import main

            exit_code = main(list(effective_argv))
            failed = exit_code != 0
        except SystemExit as error:
            if error.code is None:
                exit_code = 0
            elif isinstance(error.code, int):
                exit_code = error.code
            else:
                print(error.code, file=sys.stderr)
                exit_code = 1
            failed = exit_code != 0
        except BaseException:
            traceback.print_exc()
            exit_code = 1
            failed = True

    if failed:
        _redact_startup_log(startup_log, effective_argv)
    # Append after redaction so even a one-character secret matching the exit code
    # cannot corrupt the control marker. If this secure reopen fails, no marker is
    # published and ML Lab safely refuses to reuse the ambiguous startup shell.
    try:
        with _open_startup_log(startup_log) as marker_stream:
            marker_stream.write(f"{_STARTUP_EXITED_PREFIX}{exit_code}\n")
    except (OSError, ValueError) as error:
        _record_log_setup_failure(startup_log, error)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
