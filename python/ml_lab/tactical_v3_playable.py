"""Portable CPU inference packages, independent of training-run publication.

The checkpoint is copied byte for byte. Completion describes this export only;
source training state and supplied evidence are retained without promotion claims.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tempfile

import numpy as np
import torch

from .tactical_v3_checkpoint import (
    LoadedStructuredPolicy, _identity_manifest, _load_self_describing_checkpoint,
    load_structured_checkpoint, semantic_identity_wire,
)
from .tactical_v3_schema import parse_spaces


KIND = "tactical-v3-playable-export"
_FIELDS = {
    "schema_version", "kind", "package_id", "state", "completion_scope",
    "created_at", "config", "contract", "policy_identity", "latest_checkpoint",
    "latest_checkpoint_step", "checkpoint_sha256", "model_state_sha256",
    "evidence_status", "source", "runtime", "files", "package_sha256",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object) -> str:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _plain_file(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 1024:
        raise ValueError(f"package input must be a regular non-reparse file: {path}")
    return path.read_bytes()


def _relative(value: object) -> str:
    if type(value) is not str or "\\" in value:
        raise ValueError("package path must be relative POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or ":" in value or ".." in path.parts or str(path) != value or not path.parts:
        raise ValueError("package path must be canonical relative POSIX")
    return value


def _manifest(root: Path, expected_package_sha256: str | None = None) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("package root must be a plain directory")
    raw = _plain_file(root / "run.json")
    value = json.loads(raw)
    if type(value) is not dict or set(value) != _FIELDS or raw != _json_bytes(value):
        raise ValueError("invalid canonical playable package manifest")
    digest = _digest(_json_bytes({k: v for k, v in value.items() if k != "package_sha256"}))
    if _hash(value["package_sha256"]) != digest:
        raise ValueError("playable package manifest hash mismatch")
    if expected_package_sha256 is not None and digest != _hash(expected_package_sha256):
        raise ValueError("playable package does not match pinned hash")
    if (value["schema_version"] != 2 or value["kind"] != KIND
            or value["state"] != "completed"
            or value["completion_scope"] != "inference-package-export"
            or value["config"] != {"algorithm": "structured_imitation"}
            or value["evidence_status"] != "exported-unpromoted-checkpoint"
            or value["policy_identity"] != "policy-identity.json"
            or value["latest_checkpoint"] != "checkpoints/best.pt"):
        raise ValueError("unsupported playable package declaration")
    files = value["files"]
    if type(files) is not dict or not {"checkpoints/best.pt", "policy-identity.json", "provenance/source-run.json"} <= files.keys():
        raise ValueError("playable package inventory is incomplete")
    actual = set()
    for path in root.rglob("*"):
        info = path.lstat()
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 1024:
            raise ValueError("playable package contains a link or reparse point")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"run.json"}:
        raise ValueError("playable package inventory differs from manifest")
    for name, expected in files.items():
        if _digest(_plain_file(root / _relative(name))) != _hash(expected):
            raise ValueError(f"playable package file hash mismatch: {name}")
    if files["checkpoints/best.pt"] != _hash(value["checkpoint_sha256"]):
        raise ValueError("playable package checkpoint hash mismatch")
    return value


def validate_playable_package(
    root: Path, *, expected_package_sha256: str | None = None,
) -> LoadedStructuredPolicy:
    """Authenticate the portable inventory and replay the checkpoint's CPU fixture."""
    root = Path(root)
    before = _manifest(root, expected_package_sha256)
    identity = parse_spaces(json.loads(_plain_file(root / "policy-identity.json")))
    if before["contract"] != _identity_manifest(identity):
        raise ValueError("playable package identity does not match contract")
    source = before["source"]
    if type(source) is not dict or set(source) != {"run_name", "state", "manifest", "checkpoint", "checkpoint_sha256", "evidence"}:
        raise ValueError("playable package source lineage is invalid")
    original = json.loads(_plain_file(root / "provenance/source-run.json"))
    if (source["manifest"] != "provenance/source-run.json"
            or source["state"] != original.get("state")
            or source["checkpoint"] != original.get("latest_checkpoint")
            or source["checkpoint_sha256"] != before["checkpoint_sha256"]
            or not isinstance(source["state"], str) or not source["state"]):
        raise ValueError("playable package source snapshot does not match lineage")
    if type(source["evidence"]) is not list or any(
        name not in before["files"] or not name.startswith("evidence/")
        for name in source["evidence"]
    ):
        raise ValueError("playable package evidence inventory is invalid")
    if before["runtime"]["inference_device"] != "cpu":
        raise ValueError("playable package requires CPU inference")
    loaded = load_structured_checkpoint(root / "checkpoints/best.pt", identity.encoding_hash, identity.capacity_hash)
    if (loaded.metadata.identity != identity
            or loaded.metadata.model_state_sha256 != before["model_state_sha256"]
            or type(before["latest_checkpoint_step"]) is not int
            or loaded.metadata.best_epoch != before["latest_checkpoint_step"]):
        raise ValueError("playable package metadata does not match checkpoint")
    if _manifest(root, expected_package_sha256) != before:
        raise ValueError("playable package changed while loading")
    return loaded


def export_playable_package(
    checkpoint: Path, destination: Path, *, expected_checkpoint_sha256: str,
    source_manifest: Path, package_id: str, evidence: tuple[Path, ...] = (),
) -> Path:
    """Export one explicitly pinned checkpoint; never overwrite a package or source."""
    checkpoint, source_manifest, destination = map(Path, (checkpoint, source_manifest, destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite playable package {destination}")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", package_id) is None:
        raise ValueError("package_id must be a lowercase slug")
    checkpoint_bytes = _plain_file(checkpoint)
    if _digest(checkpoint_bytes) != _hash(expected_checkpoint_sha256):
        raise ValueError("source checkpoint does not match expected hash")
    source_bytes = _plain_file(source_manifest)
    source = json.loads(source_bytes)
    if type(source) is not dict or type(source.get("state")) is not str or not source["state"]:
        raise ValueError("source manifest must retain its recorded state")
    source_relative = _relative(source.get("latest_checkpoint"))
    if (source_manifest.parent / source_relative).resolve() != checkpoint.resolve():
        raise ValueError("source manifest does not identify the chosen checkpoint")
    snapshots = {checkpoint: checkpoint_bytes, source_manifest: source_bytes}
    for path in evidence:
        snapshots[Path(path)] = _plain_file(Path(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent))
    try:
        (temporary / "checkpoints").mkdir()
        (temporary / "provenance").mkdir()
        (temporary / "checkpoints/best.pt").write_bytes(checkpoint_bytes)
        (temporary / "provenance/source-run.json").write_bytes(source_bytes)
        loaded = _load_self_describing_checkpoint(temporary / "checkpoints/best.pt")
        (temporary / "policy-identity.json").write_bytes(_json_bytes(semantic_identity_wire(loaded.metadata.identity)))
        evidence_names = []
        for index, path in enumerate(evidence):
            name = f"evidence/{index:02d}-{Path(path).name}"
            (temporary / "evidence").mkdir(exist_ok=True)
            (temporary / name).write_bytes(snapshots[Path(path)])
            evidence_names.append(name)
        files = {path.relative_to(temporary).as_posix(): _digest(path.read_bytes())
                 for path in temporary.rglob("*") if path.is_file()}
        manifest = {
            "schema_version": 2, "kind": KIND, "package_id": package_id,
            "state": "completed", "completion_scope": "inference-package-export",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {"algorithm": "structured_imitation"},
            "contract": _identity_manifest(loaded.metadata.identity),
            "policy_identity": "policy-identity.json", "latest_checkpoint": "checkpoints/best.pt",
            "latest_checkpoint_step": loaded.metadata.best_epoch,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "model_state_sha256": loaded.metadata.model_state_sha256,
            "evidence_status": "exported-unpromoted-checkpoint",
            "source": {"run_name": source_manifest.parent.name, "state": source["state"],
                       "manifest": "provenance/source-run.json", "checkpoint": source_relative,
                       "checkpoint_sha256": expected_checkpoint_sha256, "evidence": evidence_names},
            "runtime": {"inference_device": "cpu", "torch": str(torch.__version__),
                        "numpy": np.__version__, "python": platform.python_version(),
                        "torch_threads": torch.get_num_threads(), "model_config": asdict(loaded.metadata.model_config),
                        "recorded_training_device": loaded.metadata.trainer_config.device,
                        "fixture": "embedded exact CPU logits and candidate identities"},
            "files": files,
        }
        manifest["package_sha256"] = _digest(_json_bytes(manifest))
        (temporary / "run.json").write_bytes(_json_bytes(manifest))
        validate_playable_package(temporary)
        if any(_plain_file(path) != data for path, data in snapshots.items()):
            raise ValueError("source checkpoint or evidence changed during export")
        # DrvFS does not support Linux renameat2(RENAME_NOREPLACE). Reserve the
        # output exclusively, then publish run.json last as the commit marker.
        # A racing export cannot overwrite this directory; readers fail closed
        # until the complete authenticated inventory and manifest are present.
        destination.mkdir()
        try:
            for child in temporary.iterdir():
                if child.name != "run.json":
                    shutil.move(str(child), str(destination / child.name))
            (temporary / "run.json").rename(destination / "run.json")
        except BaseException:
            shutil.rmtree(destination)
            raise
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = export_playable_package(args.checkpoint, args.output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        source_manifest=args.source_manifest, package_id=args.package_id,
        evidence=tuple(args.evidence))
    print(json.dumps({"status": "exported", "path": str(path),
                      "manifest_sha256": _digest((path / "run.json").read_bytes())}))


if __name__ == "__main__":
    main()
