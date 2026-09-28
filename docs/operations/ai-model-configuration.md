# Configure the gameplay AI

Players choose a difficulty name. You control the available choices, their names,
and which model each uses in `Assets/StreamingAssets/ai-models.json`:

```json
"default_difficulty_id": "hard",
"difficulties": [
  { "id": "easy", "label": "Easy", "model_id": "random" },
  { "id": "normal", "label": "Normal", "model_id": "greedy" },
  { "id": "hard", "label": "Hard", "model_id": "focused-epoch2" }
]
```

- `label` is the name shown in the game; rename it freely.
- `model_id` links the difficulty to an entry in `models` in the same file.
- Add or remove a difficulty entry to change the choices offered to players.
- `default_difficulty_id` controls the initial choice and title demonstration.

IDs are stable configuration keys, separate from display names. Multiple
difficulties may use the same model. Model IDs, checkpoint paths, and training
metadata are not displayed as difficulty names. The shipped default is Hard,
which uses the focused trained model; the other entries retain the scripted
controllers as configurable alternatives.

Every trained model entry identifies an immutable package under `models/ai`, its
exact checkpoint SHA-256, and its observation/capacity contract. To add a model,
export a new package using [the package instructions](../../models/ai/README.md),
add its model entry, then map a difficulty to it. Keep older packages available
until the release has been validated. Never point gameplay at a mutable training
run or select a checkpoint by modification time.

## Local and hosted use

The Unity Editor and desktop player read the catalog from StreamingAssets.
`HEXWARS_AI_CONFIG_PATH` can select a deployment-specific catalog. Local trained
inference uses `python/policy_server.py` and the packaged checkpoint; set
`HEXWARS_AI_RUNTIME_ROOT` to the directory containing `python/` and `models/`, and
`HEXWARS_AI_PYTHON` to a Python executable with the inference dependencies. The
Windows project convention is `python/winenv/Scripts/python.exe`.

For a clean CPU environment:

```sh
python -m venv .ai-venv
.ai-venv/bin/python -m pip install -r python/requirements-inference.txt
export HEXWARS_AI_PYTHON="$PWD/.ai-venv/bin/python"
```

On Windows use `.ai-venv/Scripts/python.exe` for the second command and set
`$env:HEXWARS_AI_PYTHON = (Resolve-Path .ai-venv/Scripts/python.exe).Path` in
PowerShell. Launch the game or server from that configured environment. Training and
large evaluation batches use the separately configured CUDA environment. Gameplay
policy processes use CPU inference and retain the checkpoint's strict saved
numerical fixture.

Browsers load the current catalog from `/api/ai/models` on the game server and
send trained decisions to `/api/ai/decision`. The Docker image packages the same
catalog, checkpoint, and CPU Python runtime. Set `HEXWARS_AI_CONFIG_PATH` on the
server to override the packaged `/app/ai-models.json`; package paths remain
relative to `HEXWARS_AI_RUNTIME_ROOT`. A desktop client can use this same service
by setting `HEXWARS_AI_SERVER_URL` to its HTTP(S) base URL before launch.

New games, rematches, demonstrations, and unpinned developer duels refresh the
catalog. For a hosted release, ship the updated catalog and any new package.
The server rereads the catalog and discards workers from an older revision. A
match pins its difficulty, catalog revision, and checkpoint: changing a deployment during a
match stops stale inference rather than silently switching its opponent. A
temporary catalog failure can be retried by starting a new game.

## Supported gameplay and failures

The focused package supports Annihilation with fog disabled, within the
tactical-v3 limits of 512 board cells, 64 units, and 32 templates. The adapter
also validates the projected state and candidate-table capacity. Setup prevents
unsupported configurations; it does not imply that broader game modes were
trained or evaluated.

Missing packages, incompatible contracts, stale responses, timeouts, and invalid
actions produce a visible AI error. They never silently substitute Greedy.
To roll a difficulty back to Greedy, set its `model_id` to `greedy` and start a
new game (redeploy first for a hosted release). Rematches retain the selected
difficulty ID and refresh its configured mapping. Explicitly pinned ML Lab
experiments keep their selected controllers independently of gameplay settings.

The compiled WebGL client must be rebuilt and staged with
`engine/stage-webgl-deploy.ps1` for the first release of this integration; see
[the web deployment workflow](web-main-deployment.md). Later catalog changes do
not require recompiling the browser client if the inference contract is unchanged.
