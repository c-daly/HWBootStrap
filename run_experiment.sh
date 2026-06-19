#!/usr/bin/env bash
# One-shot HexWars RL experiment: build the env-server, train two DIFFERENT MaskablePPO agents
# (logging to disk), then duel them into a replay. Run inside WSL2 from the repo root.
#
#   bash run_experiment.sh [timesteps_per_agent]      # default 200000
#
# To start it and walk away (detached, survives logout):
#   nohup bash run_experiment.sh 500000 > experiment.log 2>&1 &
#   tail -f experiment.log        # peek at progress any time
set -euo pipefail

TS="${1:-200000}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[1/4] building env-server (.NET)..."
dotnet build engine/HexWars.GymServer -c Release
export HEXWARS_SERVER="$ROOT/engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"

# use the venv if it exists (see python/README.md for one-time setup)
[ -f .venv/bin/activate ] && source .venv/bin/activate

cd python
echo "[2/4] training agent A (vs greedy baseline, $TS steps)..."
python train_maskable_ppo.py --opponent greedy --seed 1 --out ppo_a --timesteps "$TS" --logdir runs/ppo_a

echo "[3/4] training agent B (vs random baseline, $TS steps)..."
python train_maskable_ppo.py --opponent random --seed 2 --out ppo_b --timesteps "$TS" --logdir runs/ppo_b

echo "[4/4] dueling A vs B -> replay..."
python duel.py ppo_a.zip ppo_b.zip --out ../replays/ppo_a_vs_b.replay

echo "DONE."
echo "  models : python/ppo_a.zip, python/ppo_b.zip"
echo "  logs   : python/runs/ppo_a/progress.csv (+ monitor.csv), python/runs/ppo_b/..."
echo "  replay : replays/ppo_a_vs_b.replay   (Unity: HexWars -> Replay -> Open Replay File...)"
