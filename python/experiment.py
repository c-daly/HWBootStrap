"""Experiment provenance: record each run's PARAMS (at train start) and RESULTS (after eval), so runs
are reproducible and comparable.

  - write_params(logdir, spaces_info, extra)  -> runs/<name>/params.json   (env config + training args)
  - record_result(name, params, results)      -> appends to experiments.jsonl + experiments.md

experiments.md is a human-readable table; experiments.jsonl is the machine-readable log of record_result.
"""
import json
import os

EXPERIMENTS_MD = "experiments.md"
EXPERIMENTS_JSONL = "experiments.jsonl"


def write_params(logdir, spaces_info, extra):
    """Dump the run's full parameters (server-reported env config + training args) to params.json."""
    os.makedirs(logdir, exist_ok=True)
    params = {"config": spaces_info, "train": extra}
    with open(os.path.join(logdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return params


def record_result(name, params, results, logdir=None):
    """Append one experiment's params+results to experiments.jsonl (+ a row in experiments.md), and write
    runs/<name>/results.json if logdir given. `results` is a dict of eval metrics."""
    entry = {"name": name, "params": params, "results": results}
    with open(EXPERIMENTS_JSONL, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    _append_md_row(name, params, results)
    return entry


def _append_md_row(name, params, results):
    cfg = (params or {}).get("config", {})
    header = ("| run | board | biomes | obs_len | n_actions | pts_w | draw_w | "
              "vs_base W/L/D | vs_greedy W/L/D | notes |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    if not os.path.exists(EXPERIMENTS_MD):
        with open(EXPERIMENTS_MD, "w") as f:
            f.write("# HexWars RL experiments\n\n" + header + "\n" + sep + "\n")
    board = f"{cfg.get('board_w','?')}x{cfg.get('board_h','?')}"
    row = (f"| {name} | {board} | {cfg.get('biomes','?')} | {cfg.get('obs_len','?')} | "
           f"{cfg.get('n_actions','?')} | {cfg.get('points_weight','?')} | {cfg.get('draw_credit_weight','?')} | "
           f"{results.get('vs_base','')} | {results.get('vs_greedy','')} | {results.get('notes','')} |")
    with open(EXPERIMENTS_MD, "a") as f:
        f.write(row + "\n")
