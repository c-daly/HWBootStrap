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

## Train

```bash
export HEXWARS_SERVER=$(pwd)/engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll
cd python && python train_maskable_ppo.py
```

You should see the episode reward climb as the policy learns to beat the GreedyAgent.

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
