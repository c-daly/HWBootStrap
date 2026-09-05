"""Strict CPU inference adapter for tactical-v3 structured policy runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

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
    algorithm: Literal["structured_imitation", "structured_policy_gradient"] = (
        "structured_imitation"
    )
    checkpoint_step: int | None = None


def _run_identity_and_checkpoint(
    run_dir: Path,
) -> tuple[
    TacticalV3SemanticIdentity,
    Path,
    Literal["structured_imitation", "structured_policy_gradient"],
    int,
    str,
]:
    root = Path(run_dir).resolve()
    try:
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ValueError("structured run manifest must be an object")
        if manifest.get("policy_identity") != "policy-identity.json":
            raise ValueError("structured run must declare policy-identity.json")
        identity = parse_spaces(json.loads(
            (root / "policy-identity.json").read_text(encoding="utf-8")
        ))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"structured run manifest is invalid: {root}") from error
    config = manifest.get("config")
    algorithm = config.get("algorithm") if isinstance(config, Mapping) else None
    if algorithm not in {"structured_imitation", "structured_policy_gradient"}:
        raise ValueError(
            "structured run manifest must declare a supported structured algorithm"
        )
    expected_schema = 2 if algorithm == "structured_imitation" else 1
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(
            f"{algorithm} run schema version must be {expected_schema}"
        )
    checkpoint = manifest.get("latest_checkpoint")
    checkpoint_step = manifest.get("latest_checkpoint_step")
    state = manifest.get("state")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("structured run manifest is missing latest_checkpoint")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("structured run manifest is missing latest_checkpoint_step")
    if type(state) is not str or not state:
        raise ValueError("structured run manifest is missing state")
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
        raise ValueError(
            "structured run manifest contract does not match policy identity"
        )
    return identity, checkpoint_path, algorithm, checkpoint_step, state


def load_structured_controller(
    run_dir: Path,
    expected_encoding_hash: str,
    expected_capacity_hash: str,
) -> StructuredController:
    """Load only a sealed-by-validation structured run, never an isolated tensor file."""
    for _attempt in range(3):
        before = _run_identity_and_checkpoint(run_dir)
        identity, checkpoint_path, algorithm, checkpoint_step, _state = before
        if identity.encoding_hash != expected_encoding_hash:
            raise ValueError(
                "structured controller encoding hash does not match expected encoding hash"
            )
        if identity.capacity_hash != expected_capacity_hash:
            raise ValueError(
                "structured controller capacity hash does not match expected capacity hash"
            )
        if algorithm == "structured_imitation":
            loaded = validate_structured_run(Path(run_dir))
            validated_step = loaded.metadata.best_epoch
        else:
            from .tactical_v3_outcome_checkpoint import validate_outcome_run

            loaded = validate_outcome_run(Path(run_dir))
            validated_step = loaded.metadata.update
        after = _run_identity_and_checkpoint(run_dir)
        if before != after or checkpoint_step != validated_step:
            continue
        if loaded.metadata.identity != identity:
            raise ValueError(
                "validated structured checkpoint identity does not match policy identity"
            )
        policy = loaded.model.to(device="cpu")
        policy.eval()
        if next(policy.parameters()).device.type != "cpu" or policy.training:
            raise ValueError("structured controller must use CPU eval inference")
        return StructuredController(
            run_dir=Path(run_dir).resolve(),
            checkpoint_path=checkpoint_path,
            policy=policy,
            identity=identity,
            algorithm=algorithm,
            checkpoint_step=checkpoint_step,
        )
    raise ValueError("structured run changed repeatedly while loading its checkpoint")


def select_candidate(
    controller: StructuredController,
    view: TacticalV3View,
) -> CandidateIdentity:
    """Return the exact decision/candidate identity selected from this view's legal rows."""
    if type(controller) is not StructuredController:
        raise ValueError("controller must be StructuredController")
    if type(view) is not TacticalV3View:
        raise ValueError("view must be TacticalV3View")
    batch = collate_decisions(
        (view.decision,),
        controller.policy.config.horizon_turns,
        identity=controller.identity,
    )
    with torch.inference_mode():
        selected, = controller.policy.select(batch)
    return selected
