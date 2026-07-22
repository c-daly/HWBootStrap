from __future__ import annotations


def test_seat_models_is_structured_and_stably_ordered() -> None:
    from policy_server import seat_models

    class FakeSeat:
        def __init__(self, algorithm: str, step: int) -> None:
            self.algorithm = algorithm
            self.step = step

        def metadata(self) -> dict:
            return {
                "kind": "run",
                "path": f"{self.algorithm}.zip",
                "algorithm": self.algorithm,
                "step": self.step,
                "contract_hash": "c" * 64,
            }

    assert seat_models(
        {1: FakeSeat("masked_dqn", 96), 0: FakeSeat("maskable_ppo", 64)}
    ) == [
        {
            "seat": 0,
            "kind": "run",
            "path": "maskable_ppo.zip",
            "algorithm": "maskable_ppo",
            "step": 64,
            "contract_hash": "c" * 64,
        },
        {
            "seat": 1,
            "kind": "run",
            "path": "masked_dqn.zip",
            "algorithm": "masked_dqn",
            "step": 96,
            "contract_hash": "c" * 64,
        },
    ]
