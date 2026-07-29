"""Deterministic reciprocal collection of validated scripted demonstrations."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_lab.cli import DEFAULT_SERVER
from ml_lab.contracts import EnvironmentContract
from ml_lab.evaluation import DuelClient
from ml_lab.imitation import CONVERSION_PROFILES, DemonstrationGame, DemonstrationWriter, sha256_file

from ml_lab.scenarios import resolve_scenario

@dataclass(frozen=True)
class CollectionSpec:
    dataset: Path
    partition: str
    scenario_hash: str
    contract: EnvironmentContract
    client_factory: Callable[[int], Any]
    workers: int = 1
    standard_pairs: int | None = None
    conversion_pairs: int | None = None
    standard_threshold: int = 100_000
    conversion_threshold: int = 50_000


def _pair_jobs(spec: CollectionSpec) -> list[tuple[str, str, int]]:
    if spec.partition not in {"train", "validation"} or spec.workers < 1:
        raise ValueError("collection partition or worker count is invalid")
    if spec.partition == "validation":
        standard, conversion, start = spec.standard_pairs if spec.standard_pairs is not None else 100, spec.conversion_pairs if spec.conversion_pairs is not None else 20, 12_000_000
    else:
        # Threshold mode deliberately collects complete pairs only; test/smoke callers can set exact pair counts.
        standard, conversion, start = spec.standard_pairs if spec.standard_pairs is not None else spec.standard_threshold, spec.conversion_pairs if spec.conversion_pairs is not None else spec.conversion_threshold, 11_000_000
    if standard < 0 or conversion < 0: raise ValueError("collection pair count is invalid")
    jobs = [("greedy", "standard-3v3", start + index) for index in range(standard)]
    conversion_start = 12_000_000 + standard if spec.partition == "validation" else 11_500_000
    profiles = ["conversion-3v1-near", "conversion-3v1-far", "conversion-2v1-near", "conversion-2v1-far", "conversion-1v1-near", "conversion-1v1-far"]
    if spec.partition == "validation" and spec.conversion_pairs is None:
        for profile_index, profile in enumerate(profiles):
            jobs.extend(("bounded-search", profile, conversion_start + profile_index * 20 + index) for index in range(20))
    else:
        for index in range(conversion):
            jobs.append(("bounded-search", profiles[index % len(profiles)], conversion_start + index))
    return jobs


def _controllers(teacher: str, seat: int) -> tuple[str, str]:
    return (teacher, "random") if seat == 0 else ("random", teacher)


def _outcome(winner: object, teacher_seat: int) -> str:
    return "win" if winner == teacher_seat else "loss" if winner in {0, 1} else "draw"


def _collect_one(client: Any, writer: DemonstrationWriter, spec: CollectionSpec, teacher: str, profile: str, seed: int, teacher_seat: int) -> None:
    key = (spec.partition, teacher, profile, seed, teacher_seat)
    if key in writer.completed_keys(): return
    p0, p1 = _controllers(teacher, teacher_seat)
    response = client.reset(seed=seed, p0=p0, p1=p1, start_profile=profile, reference_seat=teacher_seat)
    raw = client.drain_demonstrations()
    decisions = []
    for row in raw:
        if row.get("Seat", row.get("seat")) == teacher_seat:
            decisions.append({"observation": row.get("Observation", row.get("observation")), "legal_mask": row.get("LegalMask", row.get("legal_mask")), "action": row.get("Action", row.get("action")), "seat": teacher_seat, "command": row.get("Command", row.get("command")), "decision_index": len(decisions)})
    replay_path = Path("replays") / f"{spec.partition}-{teacher}-{profile}-seed-{seed}-seat-{teacher_seat}.replay"
    saved = client.save_replay(spec.dataset / replay_path)
    if Path(saved) != spec.dataset / replay_path: raise ValueError("duel client saved replay at an unexpected path")
    parameters: dict[str, Any] = {} if teacher == "greedy" else {"depth": 4, "expansion_budget": 512, "use_heuristic": True}
    writer.append_game(DemonstrationGame(spec.partition, teacher, parameters, "random", profile, seed, teacher_seat, replay_path.as_posix(), sha256_file(spec.dataset / replay_path), _outcome(response.get("winner"), teacher_seat), spec.scenario_hash, spec.contract.contract_hash, spec.contract.encoding_hash), decisions)


def collect_partition(spec: CollectionSpec) -> Path:
    """Collect whole reciprocal pairs in a deterministic worker-stride schedule."""
    writer = DemonstrationWriter.create(spec.dataset, contract=spec.contract)
    jobs = _pair_jobs(spec)
    clients: dict[int, Any] = {}
    try:
        for index, (teacher, profile, seed) in enumerate(jobs):
            pair_keys = {(spec.partition, teacher, profile, seed, 0), (spec.partition, teacher, profile, seed, 1)}
            partial_pair = bool(pair_keys & writer.completed_keys()) and not pair_keys <= writer.completed_keys()
            if spec.partition == "train" and teacher == "greedy" and spec.standard_pairs is None and writer.retained_decision_count("greedy") >= spec.standard_threshold and not partial_pair:
                continue
            if spec.partition == "train" and teacher == "bounded-search" and spec.conversion_pairs is None and writer.retained_decision_count("bounded-search") >= spec.conversion_threshold and not partial_pair:
                continue
            worker = index % spec.workers
            client = clients.get(worker)
            if client is None:
                client = spec.client_factory(worker)
                if getattr(client, "contract", spec.contract).contract_hash != spec.contract.contract_hash or getattr(client, "contract", spec.contract).encoding_hash != spec.contract.encoding_hash:
                    raise ValueError("collection client contract does not match specification")
                client.enable_demonstrations(True); clients[worker] = client
            # Seats are deliberately adjacent and committed in key order, so resume never duplicates either side.
            _collect_one(client, writer, spec, teacher, profile, seed, 0)
            _collect_one(client, writer, spec, teacher, profile, seed, 1)
    finally:
        for client in clients.values(): client.close()
        writer.close()
    return spec.dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation"), required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    args = parser.parse_args(argv)
    scenario = resolve_scenario(environment="tactical-v2", scenario_file=args.scenario, template_id=None, enforce_round_cap_minimum=True)
    scenario_hash = hashlib.sha256(scenario.canonical_json.encode("utf-8")).hexdigest()
    probe = DuelClient(["dotnet", str(args.server)], environment="tactical-v2")
    try:
        collect_partition(CollectionSpec(args.dataset, args.partition, scenario_hash, probe.contract, lambda worker: probe if worker == 0 else DuelClient(["dotnet", str(args.server)], environment="tactical-v2")))
    except BaseException:
        probe.close()
        raise


if __name__ == "__main__": main()
