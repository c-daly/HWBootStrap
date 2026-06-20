# HexWars RL (WSL2 / Linux)

Trains agents on the HexWars engine with Stable-Baselines3. **Everything runs inside WSL2** — the .NET
engine is cross-platform, so the env-server is a native *Linux* process, and Python drives it directly.

## No cross-OS networking

There is **no WSL↔Windows network communication** in this setup, by design:

- The env-server runs **inside WSL2** as a Linux .NET process — not a Windows process reached over a
  network. (The engine is pure netstandard2.1, no platform deps, so it builds/runs on Linux unchanged.)
- Python talks to it over **stdin/stdout pipes** (it launches the server as a subprocess) — **no TCP,
  no sockets, no ports**. Nothing for WSL/Windows networking to interfere with.
- The only Windows↔WSL touchpoint is the shared git repo (files). Clone into the WSL **native**
  filesystem (below) and even that goes away.

## One-time setup (in WSL2 / Ubuntu)

```bash
# 1. .NET 8 SDK
sudo apt-get update && sudo apt-get install -y dotnet-sdk-8.0   # or use Microsoft's install script

# 2. Get the repo — clone into the WSL native fs (fast builds; avoid /mnt/c which is slow under WSL)
git clone <repo-url> ~/HexWars && cd ~/HexWars

# 3. Build the env-server (Linux binary)
dotnet build engine/HexWars.GymServer -c Release

# 4. Python venv + deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r python/requirements.txt
```

## Train one agent

```bash
export HEXWARS_SERVER=$(pwd)/engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll
cd python && python train_maskable_ppo.py --opponent greedy --out ppo_a
```

Episode reward should climb as the policy learns to beat the scripted baseline (Greedy/Random are
*baselines only* — the learner is the real agent).

## Two learned agents playing each other

Train two **different** agents (here, against different opponents), then duel them — both seats are
driven by the policies, no scripted opponent:

```bash
python train_maskable_ppo.py --opponent greedy --seed 1 --out ppo_a --timesteps 200000
python train_maskable_ppo.py --opponent random --seed 2 --out ppo_b --timesteps 200000
python duel.py --p0 ppo:ppo_a.zip --p1 ppo:ppo_b.zip --out ../replays/ppo_a_vs_b.replay
```

Then watch it: Unity → **HexWars → Replay → Open Replay File…** → pick `replays/ppo_a_vs_b.replay`.

### Pick any controller per seat

`duel.py --p0 <spec> --p1 <spec>`, where each spec is `random | greedy | ppo:PATH | dqn:PATH`:

```bash
python duel.py --p0 ppo:ppo_a.zip --p1 greedy          # learned vs greedy baseline
python duel.py --p0 ppo:ppo_a.zip --p1 dqn:dqn_a.zip   # PPO vs DQN (different algorithms)
python duel.py --p0 greedy        --p1 random          # baselines only
```

### Different algorithms (DQN)

`train_dqn.py` is an **experimental** masked DQN (value-based) — SB3 has no native masking for DQN, so
it masks exploration/exploitation itself (Q-target left unmasked). MaskablePPO is the verified path; use
DQN when you want the two duelists to use *different* algorithms:

```bash
python train_dqn.py --opponent greedy --seed 3 --out dqn_a --timesteps 200000
python duel.py --p0 ppo:ppo_a.zip --p1 dqn:dqn_a.zip
```

### Self-play (train against a trained opponent)

Train a learner against a **frozen** policy; the opponent seat is played automatically by that model.
Optionally **iterate** — each round retrains against the previous round's winner (the bootstrapping
loop that pushes agents past the scripted baselines):

```bash
python selfplay.py --opponent ppo:ppo_a.zip --out sp --timesteps 200000             # one round vs frozen ppo_a
python selfplay.py --opponent ppo:ppo_a.zip --out sp --rounds 3 --timesteps 200000  # iterate: r0 vs ppo_a, r1 vs r0, r2 vs r1
```

The final policy is `sp_r<N-1>.zip`. Duel it against anything:

```bash
python duel.py --p0 ppo:sp_r2.zip --p1 ppo:ppo_a.zip --out ../replays/sp_vs_ppo_a.replay
```

## Quick sanity check (no Python deps)

The server speaks one JSON object per line — you can poke it by hand:

```bash
printf '{"cmd":"spaces"}\n{"cmd":"reset","seed":1}\n{"cmd":"step","action":0}\n{"cmd":"close"}\n' \
  | dotnet engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll
# -> {"obs_len":761,"n_actions":379}  then reset/step JSON
```

## Notes

- **Opponent** is selectable: `HexWarsEnv(server_cmd, opponent="greedy"|"random", seat=0|1)`. Self-play
  (a frozen policy as the opponent) is a later step.
- **Scope** is tactics-only (fixed rosters, no economy). Economy/design are future curriculum stages —
  transfer the trained policy and widen the action space.
- **Protocol** is JSON-over-stdio: simple and dependency-free. Fine for a first loop; if step throughput
  becomes the bottleneck, switch the payloads to binary/msgpack (same process model).
- **GPU**: install a CUDA build of torch in WSL2 for GPU training (WSL2 supports NVIDIA passthrough).
