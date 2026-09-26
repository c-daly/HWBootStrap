"""Headless mixed-scenario replay with explicit provenance and gameplay gates.

Run with ``python -m ml_lab.tactical_v3_curriculum_run --help``. This first
curriculum stage rehearses existing authenticated collections; it does not claim
to collect fresh on-policy examples. It never changes normal-game selections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .contracts import update_run_state, utc_now, validate_run_name
from .io import atomic_write_bytes, atomic_write_json, read_json
from .tactical_v3_checkpoint import (
    _canonical_json, _metrics_jsonl, adopt_structured_run, load_structured_checkpoint,
    load_training_resume_checkpoint, replace_structured_checkpoint,
    save_training_resume_checkpoint, semantic_identity_wire, validate_structured_run,
    structured_model_state_sha256,
)
from .tactical_v3_client import CandidateSelection, TacticalV3GymClient
from .tactical_v3_continuation import _inspect_reusable_collection, _load_reusable_partition
from .tactical_v3_controller import StructuredController, select_candidate
from .tactical_v3_curriculum import CurriculumTask, GameplayScore, ScenarioMix, passes_retention_gate
from .tactical_v3_training import TrainerConfig, train_offline
from .tactical_v3_schema import canonical_sha256
from .tactical_v3_model import TacticalV3Policy


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(value, fields, label):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError(f"{label} fields must be exactly {fields}")
    return value


def code_fingerprint():
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("tactical_v3_*.py"))}


def panel_schedule(panel):
    exact(panel, "seed_start pairs profiles", "evaluation panel")
    seed, pairs, profiles = panel["seed_start"], panel["pairs"], panel["profiles"]
    if type(seed) is not int or seed < 0 or type(pairs) is not int or pairs < 1:
        raise ValueError("evaluation seeds and pairs must be nonnegative/positive integers")
    if type(profiles) is not list or not profiles or any(type(p) is not str or not p for p in profiles):
        raise ValueError("evaluation needs named profiles")
    if len(set(profiles)) != len(profiles) or seed + pairs * len(profiles) > 2**31:
        raise ValueError("evaluation profiles must be unique and seeds fit int32")
    return tuple((profile, seed + index * pairs + pair, seat)
                 for index, profile in enumerate(profiles) for pair in range(pairs) for seat in (0, 1))


def load_recipe(path: Path):
    recipe = read_json(path)
    exact(recipe, "schema_version source_run primary_task trainer micro_batch_size tasks", "curriculum recipe")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ValueError("unsupported curriculum recipe")
    validate_run_name(recipe["source_run"])
    config = TrainerConfig(**recipe["trainer"])
    micro = recipe["micro_batch_size"]
    if type(micro) is not int or micro < 1 or micro > config.batch_size or config.batch_size % micro:
        raise ValueError("micro_batch_size must divide batch_size")
    tasks = recipe["tasks"]
    if type(tasks) is not list or len(tasks) < 2:
        raise ValueError("mixed curriculum requires at least two tasks")
    names, collections = set(), set()
    for task in tasks:
        exact(task, "name weight collections opponent screen confirmation", "curriculum task")
        name = task["name"]
        if type(name) is not str or not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("curriculum task names must be unique alphanumeric identifiers")
        names.add(name)
        if type(task["weight"]) is not int or not 1 <= task["weight"] <= 10000:
            raise ValueError("task weights must be positive bounded integers")
        if task["opponent"] not in ("random", "greedy", "passive"):
            raise ValueError("curriculum evaluation opponent must be a built-in controller")
        if type(task["collections"]) is not list or not task["collections"]:
            raise ValueError("each task requires saved collections")
        for collection in task["collections"]:
            validate_run_name(collection)
            if collection in collections:
                raise ValueError("a collection may be declared only once")
            collections.add(collection)
        screen, confirmation = panel_schedule(task["screen"]), panel_schedule(task["confirmation"])
        if {seed for _, seed, _ in screen} & {seed for _, seed, _ in confirmation}:
            raise ValueError("screen and confirmation seeds must be disjoint")
    if recipe["primary_task"] != tasks[0]["name"]:
        raise ValueError("the first task must be the primary deployment task")
    return recipe, config


def load_inputs(recipe, runs_root, run_name):
    """Reuse the existing strict continuation evidence reader, not raw JSON labels."""
    source = validate_structured_run(runs_root / recipe["source_run"])
    tasks, scenarios, train, validation, evidence = [], {}, [], [], []
    for task in recipe["tasks"]:
        identity = None
        for collection in task["collections"]:
            print(json.dumps({"phase": "authenticating", "collection": collection}), flush=True)
            header = _inspect_reusable_collection(runs_root / collection, run_name, runs_root, replay_only=True)
            if identity is not None and identity != header.identity:
                raise ValueError("collections within a task have different semantic identities")
            identity = header.identity
            scenarios[task["name"]] = json.loads(header.scenario.canonical_json)
            entry = {"run": collection, "task": task["name"],
                     "collection_sha256": hashlib.sha256(header.collection_bytes).hexdigest()}
            for partition, target in (("train", train), ("validation", validation)):
                episodes, digest = _load_reusable_partition(header, partition, replay_only=True)
                rows = tuple(record.example for episode in episodes for record in episode.records)
                target.extend(rows)
                entry[partition] = {"records_sha256": digest, "examples": len(rows)}
                print(json.dumps({"phase": "authenticated_partition", "collection": collection,
                                  "partition": partition, "examples": len(rows)}), flush=True)
                used_seeds = {row.episode_seed for row in rows}
                for panel in ("screen", "confirmation"):
                    if used_seeds & {seed for _, seed, _ in panel_schedule(task[panel])}:
                        raise ValueError("evaluation panel overlaps collection seeds")
                del episodes, rows
            evidence.append(entry)
        tasks.append(CurriculumTask(task["name"], identity, task["weight"]))
    mix = ScenarioMix(tuple(tasks))
    if source.metadata.identity != mix.tasks[0].identity:
        raise ValueError("source policy must match the primary deployment task exactly")
    return source, mix, scenarios, tuple(train), tuple(validation), evidence


def evaluate_panel(controller, task, identity, panel, server, scenario_file, progress):
    games = []
    schedule = panel_schedule(task[panel])
    with TacticalV3GymClient(["dotnet", str(server), "--scenario-file", str(scenario_file)], environment_kind="duel") as client:
        if client.identity != identity:
            raise ValueError("evaluation scenario identity changed")
        for profile, seed, seat in schedule:
            view = client.duel_reset(seed, "external" if seat == 0 else task["opponent"],
                task["opponent"] if seat == 0 else "external", seat, profile, seat)
            actions = {}
            while not view.terminated and not view.truncated:
                if view.seat != seat:
                    raise ValueError("evaluation learner seat changed")
                selected = select_candidate(controller, view, target_identity=identity)
                kind = next(c.kind for c in view.decision.candidates if c.candidate_id == selected.candidate_id)
                actions[kind] = actions.get(kind, 0) + 1
                view = client.duel_step(CandidateSelection(selected.decision_id, selected.candidate_id))
            if client.duel_status() != 0:
                raise ValueError("evaluation recorded an internal fallback")
            games.append({"profile": profile, "seed": seed, "seat": seat, "winner": view.winner,
                          "terminated": view.terminated, "truncated": view.truncated, "actions": actions})
            progress(games, len(schedule))
    score = GameplayScore(
        tuple(sum(g["winner"] == seat for g in games if g["seat"] == seat) for seat in (0, 1)),
        tuple(sum(g["winner"] == -1 for g in games if g["seat"] == seat) for seat in (0, 1)),
        tuple(sum(g["seat"] == seat for g in games) for seat in (0, 1)))
    return score, games


class StopAtCheckpoint(Exception):
    pass


def run(args):
    recipe, config = load_recipe(args.recipe)
    validate_run_name(args.run_name)
    runs_root, server = args.runs_root.resolve(), args.server.resolve()
    run_dir = runs_root / args.run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    if args.publish:
        validate_run_name(f"{args.run_name}-model")
        published_target = runs_root / f"{args.run_name}-model"
        if published_target.exists() or published_target.is_symlink():
            raise FileExistsError(f"refusing to overwrite {published_target}")
    server_sha = file_hash(server)
    code_hashes = code_fingerprint()
    source, mix, scenarios, train, validation, evidence = load_inputs(recipe, runs_root, args.run_name)
    mix.partitions(train)
    validation_parts = mix.partitions(validation)
    manifest = {"schema_version": 1, "kind": "tactical-v3-mixed-scenario-replay",
        "recipe": recipe, "curriculum_sha256": mix.sha256, "collections": evidence,
        "identities": {task.name: semantic_identity_wire(task.identity) for task in mix.tasks},
        "source_model_state_sha256": source.metadata.model_state_sha256, "server_sha256": server_sha,
        "code_sha256": code_hashes, "runtime": {"python": sys.version, "torch": str(torch.__version__),
                                               "cuda": torch.version.cuda}}
    corpus_sha = canonical_sha256(manifest)
    # Validate each scenario with the same real server that will evaluate it.
    # No source artifacts are modified: preflight can use the authenticated owner's file.
    for task, descriptor in zip(mix.tasks, recipe["tasks"]):
        header = _inspect_reusable_collection(runs_root / descriptor["collections"][0], args.run_name, runs_root, replay_only=True)
        with TacticalV3GymClient(["dotnet", str(server), "--scenario-file", str(header.owner_run / "scenario.json")], environment_kind="duel") as client:
            if client.identity != task.identity:
                raise ValueError("curriculum preflight scenario identity changed")
            for panel in ("screen", "confirmation"):
                available = {name for name, _ in header.start_distribution}
                if not set(descriptor[panel]["profiles"]) <= available:
                    raise ValueError("evaluation panel names an unavailable start profile")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA is not available in this process")
    if code_fingerprint() != code_hashes or file_hash(server) != server_sha:
        raise ValueError("curriculum code or server changed during preflight")
    print(json.dumps({"phase": "preflight_passed", "train": len(train), "validation": len(validation),
                      "task_weights": {task.name: task.weight for task in mix.tasks}}), flush=True)
    if args.preflight:
        return None
    resume = None
    if args.resume_from is not None:
        old = args.resume_from.resolve()
        if read_json(old / "run.json").get("state") not in {"stopped", "completed", "failed"}:
            raise ValueError("resume requires a terminal source run; active training is left alone")
        if read_json(old / "collection.json") != manifest:
            raise ValueError("resume recipe, server, or authenticated collections changed")
        resume = load_training_resume_checkpoint(old / "training" / "last.pt",
            expected_identity=source.metadata.identity, expected_corpus_sha256=corpus_sha,
            expected_source_model_state_sha256=source.metadata.model_state_sha256,
            expected_curriculum_sha256=mix.sha256)
    run_dir.mkdir()
    training_dir = run_dir / "training"
    training_dir.mkdir()
    (training_dir / "checkpoints").mkdir()
    best_path = training_dir / "checkpoints" / "best.pt"
    (run_dir / "scenarios").mkdir()
    for name, scenario in scenarios.items():
        atomic_write_json(run_dir / "scenarios" / f"{name}.json", scenario)
    atomic_write_bytes(run_dir / "collection.json", _canonical_json(manifest))
    atomic_write_json(run_dir / "control.json", {"request": None})
    atomic_write_json(run_dir / "run.json", {"schema_version": 1, "state": "created", "pid": os.getpid(),
        "created_at": utc_now(), "updated_at": utc_now(), "config": {"backend": "structured_curriculum",
        "algorithm": "structured_curriculum", "environment": "tactical-v3", "run_name": args.run_name,
        "device": config.device}, "curriculum_sha256": mix.sha256, "published_run": None})
    writer = SummaryWriter(str(run_dir / "tensorboard"), flush_secs=10)

    def status(phase, **fields):
        value = {"phase": phase, **fields}
        print(json.dumps(value), flush=True)
        with (run_dir / "train.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(value) + "\n")
        update_run_state(run_dir, "running", latest_message=phase, **fields)

    def step(name, metric):
        for key, value in metric.metrics.items():
            writer.add_scalar(f"tasks/{name}/steps/{metric.phase}/{key}", value, metric.global_step)

    def validation_epoch(epoch, name, metrics):
        for key, value in metrics.items():
            writer.add_scalar(f"tasks/{name}/epoch/validation/{key}", value, epoch + 1)
        with (training_dir / "task-metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch, "task": name, "validation": dict(metrics)}) + "\n")

    def checkpoint(state):
        save_training_resume_checkpoint(training_dir / "last.pt", state, identity=source.metadata.identity,
            corpus_sha256=corpus_sha, source_model_state_sha256=source.metadata.model_state_sha256)
        with torch.random.fork_rng(devices=[]):
            model = TacticalV3Policy(state.model_config).cpu()
        model.load_state_dict(dict(state.best_state), strict=True)
        metadata = replace(source.metadata, trainer_config=config, corpus_sha256=corpus_sha,
            best_epoch=state.best_epoch, best_validation_policy_nll=state.best_validation_policy_nll,
            model_state_sha256=structured_model_state_sha256(model))
        with torch.random.fork_rng(devices=[]):
            replace_structured_checkpoint(best_path, model, metadata, validation_parts[0][:2])
        atomic_write_bytes(training_dir / "metrics.jsonl", _metrics_jsonl(state.history))
        metric = state.history[-1]
        for phase in ("train", "validation"):
            for key, value in getattr(metric, phase).items():
                writer.add_scalar(f"epoch/{phase}/{key}", value, state.next_epoch)
        writer.flush()
        status("training", epoch=state.next_epoch, best_epoch=state.best_epoch + 1,
               latest_checkpoint="training/checkpoints/best.pt", latest_checkpoint_step=state.next_epoch,
               best_weighted_validation_nll=state.best_validation_policy_nll)
        if read_json(run_dir / "control.json").get("request") in {"stop_now", "stop_after_checkpoint"}:
            raise StopAtCheckpoint

    try:
        status("training", train_examples=len(train), validation_examples=len(validation))
        if resume is not None:
            checkpoint(resume)
        result = train_offline(train, validation, source.metadata.model_config, source.metadata.objective_config,
            config, scenario_mix=mix, micro_batch_size=recipe["micro_batch_size"],
            initial_state_dict=source.model.state_dict() if resume is None else None, resume_state=resume,
            task_step_callback=step, task_validation_callback=validation_epoch, checkpoint_callback=checkpoint)
        candidate = load_structured_checkpoint(best_path, source.metadata.identity.encoding_hash,
                                              source.metadata.identity.capacity_hash)
        controllers = {
            "baseline": StructuredController(runs_root / recipe["source_run"], Path("unused"), source.model.cpu().eval(), source.metadata.identity),
            "candidate": StructuredController(run_dir, best_path, candidate.model, candidate.metadata.identity)}
        gate_evidence = {"rule": "no-per-seat-regression-and-new-task-improvement-v1", "panels": {},
                         "candidate_sha256": file_hash(best_path),
                         "baseline_model_state_sha256": source.metadata.model_state_sha256, "server_sha256": server_sha}
        eligible = True
        for panel in ("screen", "confirmation"):
            if file_hash(server) != server_sha:
                raise ValueError("evaluation server changed during training")
            scores = {name: {} for name in controllers}
            panel_evidence = {}
            for task, descriptor in zip(mix.tasks, recipe["tasks"]):
                for label, controller in controllers.items():
                    def progress(games, total, task_name=task.name, label=label):
                        wins = sum(game["winner"] == game["seat"] for game in games)
                        writer.add_scalar(f"evaluation/{panel}/{task_name}/{label}/wins", wins, len(games))
                        writer.flush()
                        atomic_write_json(run_dir / "evaluation-progress.json", {
                            "panel": panel, "task": task_name, "controller": label, "games": games})
                        status("evaluating", panel=panel, task=task_name, controller=label,
                               games=len(games), total_games=total, wins=wins)
                        if read_json(run_dir / "control.json").get("request") in {"stop_now", "stop_after_checkpoint"}:
                            raise StopAtCheckpoint
                    score, games = evaluate_panel(controller, descriptor, task.identity, panel, server,
                        run_dir / "scenarios" / f"{task.name}.json", progress)
                    scores[label][task.name] = score
                    panel_evidence[f"{task.name}/{label}"] = {
                        "score": {key: list(value) for key, value in asdict(score).items()}, "games": games,
                    }
            eligible = passes_retention_gate(scores["baseline"], scores["candidate"], primary_task=recipe["primary_task"])
            gate_evidence["panels"][panel] = {"passed": eligible, "evaluations": panel_evidence}
            atomic_write_json(run_dir / "evaluation.json", gate_evidence)
            if not eligible:
                break  # Do not spend the reserved confirmation panel on a failed candidate.
        training_manifest = {"schema_version": 1, "kind": "mixed-scenario-replay-training",
            "corpus_sha256": corpus_sha, "curriculum_sha256": mix.sha256, "trainer": asdict(config),
            "best_epoch": result.best_epoch, "best_weighted_validation_nll": result.best_validation_policy_nll,
            "gameplay_gate_passed": eligible, "gate": gate_evidence}
        atomic_write_bytes(training_dir / "dagger-training.json", _canonical_json(training_manifest))
        published = None
        if eligible and args.publish:
            published = adopt_structured_run(runs_root / f"{args.run_name}-model",
                source_checkpoint_path=best_path, source_collection_path=run_dir / "collection.json",
                source_training_path=training_dir / "dagger-training.json", source_metrics_path=training_dir / "metrics.jsonl",
                training_scenario_path=run_dir / "scenarios" / f"{recipe['primary_task']}.json",
                expected_identity=source.metadata.identity, expected_checkpoint_sha256=file_hash(best_path),
                expected_collection_sha256=file_hash(run_dir / "collection.json"),
                expected_training_sha256=file_hash(training_dir / "dagger-training.json"),
                expected_metrics_sha256=file_hash(training_dir / "metrics.jsonl"),
                expected_scenario_sha256=file_hash(run_dir / "scenarios" / f"{recipe['primary_task']}.json"))
        update_run_state(run_dir, "completed", gameplay_gate_passed=eligible,
            latest_message="gameplay gates passed" if eligible else "candidate retained for analysis; gameplay gate failed",
            published_run=None if published is None else str(published))
    except StopAtCheckpoint:
        update_run_state(run_dir, "stopped", latest_message="stopped with exact-resume checkpoint preserved")
    except BaseException as error:
        update_run_state(run_dir, "failed", latest_message=str(error))
        raise
    finally:
        writer.close()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Resume into a new sibling run; never overwrite the stopped source")
    parser.add_argument("--publish", action="store_true", help="Publish only if both independent gameplay panels pass")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
