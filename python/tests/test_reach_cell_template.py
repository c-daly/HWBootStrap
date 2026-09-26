from __future__ import annotations

from ml_lab.scenarios import resolve_scenario


def test_reach_cell_template_is_a_selectable_close_static_1v1_curriculum() -> None:
    reach = resolve_scenario(
        environment="tactical-v3",
        scenario_file=None,
        template_id="tactical-v3-reach-cell-v1",
    )
    close = resolve_scenario(
        environment="tactical-v3",
        scenario_file=None,
        template_id="tactical-v3-close-static-v1",
    )

    assert reach.template_id == "tactical-v3-reach-cell-v1"
    assert reach.name == "Reach Beacon 1v1"
    assert reach.document["board"] == close.document["board"]
    assert reach.document["rules"] == close.document["rules"]
    assert reach.document["episode"] == close.document["episode"]
    assert reach.document["reward"] == close.document["reward"]

    tactical_v3 = reach.document["tactical_v3"]
    legacy_tactical_v3 = close.document["tactical_v3"]
    assert tactical_v3["objective"] == {
        "kind": "reach_cell",
        "target_policy": "seeded_farthest_reachable_unoccupied_v1",
        "radius": 0,
    }
    for key in (
        "starting_unit_count",
        "max_controllable_units",
        "placement_policy",
        "capacity",
        "templates",
        "start_profiles",
        "start_distribution",
    ):
        assert tactical_v3[key] == legacy_tactical_v3[key]

    active_weights = [
        row for row in tactical_v3["start_distribution"] if row["basis_points"]
    ]
    assert len(active_weights) == 1
    assert active_weights[0] == {
        "profile_id": "conversion-1v1-near",
        "basis_points": 10_000,
    }
    profiles = {row["id"]: row for row in tactical_v3["start_profiles"]}
    active_profile = profiles[active_weights[0]["profile_id"]]
    assert active_profile["learner_units"] == 1
    assert active_profile["opponent_units"] == 1
