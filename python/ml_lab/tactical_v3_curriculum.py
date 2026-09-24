"""Explicit, deterministic rehearsal across compatible tactical-v3 scenarios.

Each batch belongs to one authenticated scenario. Task weights control optimizer
updates, not the number of archived examples; small tasks cannot disappear behind
a larger replay collection. This module does not change model input geometry.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping

from .tactical_v3_corpus import StructuredExample
from .tactical_v3_schema import TacticalV3SemanticIdentity, canonical_sha256


@dataclass(frozen=True, slots=True)
class CurriculumTask:
    name: str
    identity: TacticalV3SemanticIdentity
    weight: int

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("curriculum task name must be alphanumeric with underscores")
        if type(self.weight) is not int or not 1 <= self.weight <= 10000:
            raise ValueError("curriculum task weight must be an integer from 1 to 10000")
        if type(self.identity) is not TacticalV3SemanticIdentity:
            raise TypeError("curriculum task requires a semantic identity")


@dataclass(frozen=True, slots=True)
class ScenarioMix:
    tasks: tuple[CurriculumTask, ...]

    def __post_init__(self) -> None:
        if type(self.tasks) is not tuple or not self.tasks:
            raise ValueError("curriculum tasks must be a nonempty immutable tuple")
        if any(type(task) is not CurriculumTask for task in self.tasks):
            raise TypeError("curriculum tasks must be CurriculumTask values")
        if len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("curriculum task names must be unique")
        if len({task.identity.contract_hash for task in self.tasks}) != len(self.tasks):
            raise ValueError("combine collections with the same contract into one task")
        primary = self.tasks[0].identity
        for task in self.tasks:
            identity = task.identity
            if identity.contract_version != "tactical-v3" or identity.environment_kind != "duel":
                raise ValueError("curriculum requires tactical-v3 duel identities")
            if (identity.encoding_hash, identity.capacity_hash) != (primary.encoding_hash, primary.capacity_hash):
                raise ValueError("curriculum model encoding and capacity must match exactly")

    @property
    def sha256(self) -> str:
        # Full identities, ordering, weights, and sampler semantics are resume state.
        from .tactical_v3_checkpoint import semantic_identity_wire

        return canonical_sha256({
            "sampler": "smooth-weighted-task-updates-shuffled-cycling-v1",
            "tasks": [{"name": task.name, "weight": task.weight,
                       "identity": semantic_identity_wire(task.identity)} for task in self.tasks],
        })

    def partitions(self, examples: tuple[StructuredExample, ...]) -> tuple[tuple[StructuredExample, ...], ...]:
        grouped: list[list[StructuredExample]] = [[] for _ in self.tasks]
        by_contract = {task.identity.contract_hash: index for index, task in enumerate(self.tasks)}
        for row in examples:
            if type(row) is not StructuredExample or row.contract_hash not in by_contract:
                raise ValueError("curriculum example has an undeclared contract")
            index = by_contract[row.contract_hash]
            identity = self.tasks[index].identity
            if (row.scenario_id, row.encoding_hash, row.capacity_hash) != (
                identity.scenario_id, identity.encoding_hash, identity.capacity_hash,
            ):
                raise ValueError("curriculum example provenance does not match its task")
            grouped[index].append(row)
        if any(not rows for rows in grouped):
            raise ValueError("every curriculum task needs both training and validation examples")
        return tuple(tuple(rows) for rows in grouped)

    def cycle(self) -> tuple[int, ...]:
        divisor = math.gcd(*(task.weight for task in self.tasks))
        weights = [task.weight // divisor for task in self.tasks]
        total = sum(weights)
        current = [0] * len(weights)
        cycle = []
        for _ in range(total):
            current = [score + weight for score, weight in zip(current, weights)]
            selected = max(range(len(weights)), key=lambda index: current[index])
            current[selected] -= total
            cycle.append(selected)
        return tuple(cycle)

    def batches(self, examples: tuple[StructuredExample, ...], batch_size: int, seed: int, epoch: int):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("curriculum batch size must be a positive integer")
        if type(seed) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError("curriculum seed and epoch must be integers with nonnegative epoch")
        partitions = self.partitions(examples)
        count = math.ceil(len(examples) / batch_size)
        cycle = self.cycle()
        rng = random.Random(f"curriculum-v1:{seed}:{epoch}")
        pools = [list(rows) for rows in partitions]
        for pool in pools:
            rng.shuffle(pool)
        positions = [0] * len(pools)
        for batch_index in range(count):
            task_index = cycle[(epoch * count + batch_index) % len(cycle)]
            pool = pools[task_index]
            rows = []
            for _ in range(batch_size):
                if positions[task_index] == len(pool):
                    rng.shuffle(pool)
                    positions[task_index] = 0
                rows.append(pool[positions[task_index]])
                positions[task_index] += 1
            yield self.tasks[task_index], tuple(rows)


@dataclass(frozen=True, slots=True)
class GameplayScore:
    """One complete reciprocal evaluation, never a partial/live win counter."""

    wins: tuple[int, int]
    draws: tuple[int, int]
    games: tuple[int, int]

    def __post_init__(self) -> None:
        if any(type(values) is not tuple or len(values) != 2 for values in (self.wins, self.draws, self.games)):
            raise ValueError("gameplay score must contain both seats")
        for seat in (0, 1):
            wins, draws, games = self.wins[seat], self.draws[seat], self.games[seat]
            if any(type(value) is not int for value in (wins, draws, games)) or games <= 0:
                raise ValueError("gameplay counts must be integers with positive games")
            if wins < 0 or draws < 0 or wins + draws > games:
                raise ValueError("gameplay wins/draws exceed completed games")
        if self.games[0] != self.games[1]:
            raise ValueError("gameplay evaluation must be reciprocal")


def passes_retention_gate(
    baseline: Mapping[str, GameplayScore], candidate: Mapping[str, GameplayScore],
    *, primary_task: str, min_new_task_wins: int = 1,
) -> bool:
    """Require no per-seat combat regression AND improvement on every new task.

    This is a conservative development-screen gate, not a statistical claim of
    equivalence. Confirmation uses a separate, previously untouched seed panel.
    """
    if set(baseline) != set(candidate) or primary_task not in baseline or len(baseline) < 2:
        raise ValueError("retention gate requires the same complete task panels")
    if type(min_new_task_wins) is not int or min_new_task_wins < 1:
        raise ValueError("minimum new-task improvement must be a positive win count")
    for name, original in baseline.items():
        actual = candidate[name]
        if type(original) is not GameplayScore or type(actual) is not GameplayScore:
            raise TypeError("retention gate requires validated gameplay scores")
        if original.games != actual.games:
            raise ValueError("retention gate panel sizes changed")
    for name, original in baseline.items():
        actual = candidate[name]
        for seat in (0, 1):
            if actual.wins[seat] < original.wins[seat]:
                return False
            if 2 * actual.wins[seat] + actual.draws[seat] < 2 * original.wins[seat] + original.draws[seat]:
                return False
        if name != primary_task and sum(actual.wins) < sum(original.wins) + min_new_task_wins:
            return False
    return True
