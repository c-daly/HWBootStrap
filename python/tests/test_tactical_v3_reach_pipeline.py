"""Reach objectives retain their semantics across existing model workflows."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml_lab.tactical_v3_controller import StructuredController, select_candidate
from ml_lab.tactical_v3_outcome import (
    _categorical_sample, _greedy_sample, optimize_outcome_rollout,
)
from ml_lab.tactical_v3_outcome_checkpoint import (
    load_outcome_checkpoint, save_outcome_checkpoint,
)
from ml_lab.tactical_v3_schema import TokenRef
from tests.test_tactical_v3_batching import _move_decision
from tests.test_tactical_v3_outcome import _game, _policy
from tests.test_tactical_v3_outcome_checkpoint import _case
from tests.test_tactical_v3_pilot import _identity, _reach_identity, _view


def _reach_decision():
    return _move_decision(target=TokenRef("cells", 1), terminal=True)


def test_cross_scenario_controller_uses_explicit_target_objective() -> None:
    model = _policy().eval()
    controller = StructuredController(Path("source"), Path("best.pt"), model, _identity())
    view = replace(_view(17), decision=_reach_decision())
    with pytest.raises(ValueError, match="annihilation move.target"):
        select_candidate(controller, view)
    chosen = select_candidate(controller, view, target_identity=_reach_identity())
    assert chosen.candidate_id == view.decision.candidates[0].candidate_id


@pytest.mark.parametrize("field", ["encoding_hash", "capacity_hash"])
def test_cross_scenario_controller_rejects_incompatible_target(field: str) -> None:
    controller = StructuredController(
        Path("source"), Path("best.pt"), _policy().eval(), _identity(),
    )
    target = replace(_reach_identity(), **{field: "0" * 64})
    with pytest.raises(ValueError, match=field):
        select_candidate(controller, _view(17), target_identity=target)


def test_reach_outcome_sampling_and_update_use_authenticated_objective() -> None:
    model = _policy().eval()
    identity, decision = _reach_identity(), _reach_decision()
    chosen, log_probability, entropy = _categorical_sample(
        model, decision, 234, identity=identity,
    )
    assert _greedy_sample(model, decision, identity=identity)[0] == chosen
    original = _game(model, 1.0, 0)
    record = replace(
        original.game.records[0], decision=decision,
        selected_candidate_id=chosen, log_probability=log_probability, entropy=entropy,
    )
    game = replace(original, game=replace(original.game, identity=identity, records=(record,)))
    optimize_outcome_rollout(
        model, torch.optim.Adam(model.parameters(), lr=3e-4),
        (game,), micro_batch_size=1,
    )


def test_reach_outcome_checkpoint_roundtrips_objective(tmp_path: Path) -> None:
    _, model, metadata, _, _ = _case(tmp_path)
    identity, decision = _reach_identity(), _reach_decision()
    metadata = replace(metadata, identity=identity)
    path = save_outcome_checkpoint(tmp_path / "beacon.pt", model, metadata, (decision,))
    loaded = load_outcome_checkpoint(path, identity.encoding_hash, identity.capacity_hash)
    assert loaded.metadata.identity == identity
    assert loaded.fixture.decisions == (decision,)
