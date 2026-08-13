"""Strict CPU inference adapter for tactical-v3 structured policy runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .tactical_v3_batching import collate_decisions
from .tactical_v3_checkpoint import validate_structured_run
from .tactical_v3_model import CandidateIdentity, TacticalV3Policy
from .tactical_v3_schema import TacticalV3SemanticIdentity, TacticalV3View, parse_spaces


@dataclass(frozen=True, slots=True)
class StructuredController:
    run_dir: Path
    checkpoint_path: Path
    policy: TacticalV3Policy
    identity: TacticalV3SemanticIdentity


def _run_identity_and_checkpoint(run_dir: Path) -> tuple[TacticalV3SemanticIdentity, Path]:
    root = Path(run_dir).resolve()
    try:
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        identity = parse_spaces(json.loads((root / "scenario.json").read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"structured run manifest is invalid: {root}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("structured run manifest must be an object")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or config.get("algorithm") != "structured_imitation":
        raise ValueError("structured run manifest must declare algorithm structured_imitation")
    checkpoint = manifest.get("latest_checkpoint")
    if not isinstance(checkpoint, str):
        raise ValueError("structured run manifest is missing latest_checkpoint")
    checkpoint_path = (root / checkpoint).resolve()
    checkpoints = (root / "checkpoints").resolve()
    if checkpoint_path.parent != checkpoints or checkpoint_path.suffix != ".pt":
        raise ValueError("structured run latest_checkpoint must name a checkpoints/*.pt file")
    contract = manifest.get("contract")
    expected_contract = {
        "environment": "tactical-v3",
        "version": "tactical-v3",
        "environment_kind": identity.environment_kind,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }
    if contract != expected_contract:
        raise ValueError("structured run manifest contract does not match scenario identity")
    return identity, checkpoint_path


def load_structured_controller(
    run_dir: Path,
    expected_encoding_hash: str,
    expected_capacity_hash: str,
) -> StructuredController:
    """Load only a sealed-by-validation structured run, never an isolated tensor file."""
    identity, checkpoint_path = _run_identity_and_checkpoint(run_dir)
    if identity.encoding_hash != expected_encoding_hash:
        raise ValueError("structured controller encoding hash does not match expected encoding hash")
    if identity.capacity_hash != expected_capacity_hash:
        raise ValueError("structured controller capacity hash does not match expected capacity hash")
    loaded = validate_structured_run(Path(run_dir))
    if loaded.metadata.identity != identity:
        raise ValueError("validated structured checkpoint identity does not match run scenario")
    policy = loaded.model.to(device="cpu")
    policy.eval()
    if next(policy.parameters()).device.type != "cpu" or policy.training:
        raise ValueError("structured controller must use CPU eval inference")
    return StructuredController(Path(run_dir).resolve(), checkpoint_path, policy, identity)


def select_candidate(
    controller: StructuredController,
    view: TacticalV3View,
) -> CandidateIdentity:
    """Return the exact decision/candidate identity selected from this view's legal rows."""
    if type(controller) is not StructuredController:
        raise ValueError("controller must be StructuredController")
    if type(view) is not TacticalV3View:
        raise ValueError("view must be TacticalV3View")
    batch = collate_decisions((view.decision,), controller.policy.config.horizon_turns)
    with torch.inference_mode():
        selected, = controller.policy.select(batch)
    return selected
