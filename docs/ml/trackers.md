# Experiment tracking

HexWars treats local run files as the source of record. TensorBoard, Weights & Biases, and custom services are optional mirrors. If a mirror cannot initialize or raises while recording, its entry in `run.json.tracker_status` becomes `degraded`; local metrics and training continue.

## Local and TensorBoard

Local tracking is always safe to select and requires no service:

```powershell
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run local_example --tracker local
```

TensorBoard is an optional dependency and must be installed explicitly:

```powershell
.\python\winenv\Scripts\python.exe -m pip install tensorboard
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor --tracker tensorboard
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run tb_example --tracker local --tracker tensorboard
.\python\winenv\Scripts\python.exe -m tensorboard.main --logdir .\python\runs
```

Open the local URL TensorBoard prints. Numeric SB3 logger values are mirrored at their training step. `progress.csv`, monitor files, `run.json`, and `evaluation.json` remain the authoritative data.

## Weights & Biases

Install/login using W&B's normal tools or environment. Do not paste a token into Unity fields, a run name, CLI arguments, or a checked-in script. HexWars rejects tracker configuration keys containing token, secret, password, or API-key forms before they can enter a manifest.

```powershell
.\python\winenv\Scripts\python.exe -m pip install wandb
.\python\winenv\Scripts\python.exe -m wandb login
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor --tracker wandb

.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train `
  --run wandb_example `
  --tracker local `
  --tracker wandb `
  --wandb-project hexwars `
  --wandb-group counter-run1 `
  --wandb-tag ppo `
  --wandb-tag alternating-seats
```

Optional fields are `--wandb-entity`, `--wandb-mode online|offline|disabled`, group, and repeatable tags. Offline mode keeps W&B data under the local run directory for a later service-owned sync. The Unity Train tab exposes project, entity, mode, group, and artifact consent; use the CLI when repeatable tags are required.

Model/replay upload is off by default. Add `--wandb-upload-artifacts` only when the destination and data policy are understood. Local checkpoints are still written regardless.

## Custom tracker adapters

A custom adapter is one importable Python function with this signature:

```python
from typing import Any

def record(event: dict[str, Any]) -> None:
    event_type = event["type"]
    run_name = event["run_name"]
    timestamp = event["timestamp"]

    if event_type == "start":
        start_remote_run(run_name, timestamp)
    elif event_type == "metrics":
        send_metrics(run_name, event["step"], event["metrics"])
    elif event_type == "artifact":
        register_artifact(run_name, event["name"], event["path"])
    elif event_type == "finish":
        finish_remote_run(run_name, event["status"])
```

Put the module somewhere on the active Python import path, then use `module:function`:

```powershell
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor --tracker custom=my_team.hexwars_tracker:record
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run service_example --tracker local --tracker custom=my_team.hexwars_tracker:record
```

Events are normalized dictionaries:

| Type | Additional fields |
| --- | --- |
| `start` | none |
| `metrics` | integer `step`, dictionary `metrics` |
| `artifact` | local `path`, stable `name` |
| `finish` | terminal `status` |

The callback should return quickly; queue remote I/O inside the adapter if the service can block. Read credentials from the service SDK's environment/credential store. Never mutate the event or run files. Decide explicitly whether the adapter uploads artifacts—receiving an artifact event is not upload consent for a new integration.

Minimum adapter tests should prove import/configuration, every event shape, step preservation, credentials absent from manifests/logs, explicit artifact consent, and failure isolation. A test adapter should deliberately raise during a metric event and assert that local `progress.csv` still advances while `run.json.tracker_status` reports degradation.

For example, using the existing `run_dir` test fixture:

```python
def test_custom_tracker_receives_normalized_events(run_dir, monkeypatch):
    module = types.ModuleType("example_tracker")
    received = []
    module.record = received.append
    monkeypatch.setitem(sys.modules, module.__name__, module)

    hub = TrackerHub(run_dir, [{"kind": "custom", "adapter": "example_tracker:record"}])
    hub.start_run()
    hub.log_metrics({"mean_reward": 1.5}, step=64)
    hub.finish("completed")

    assert [event["type"] for event in received] == ["start", "metrics", "finish"]
    assert received[1]["step"] == 64

def test_custom_tracker_failure_does_not_lose_local_metrics(run_dir, monkeypatch):
    module = types.ModuleType("failing_tracker")
    module.record = lambda event: (_ for _ in ()).throw(RuntimeError("offline")) \
        if event["type"] == "metrics" else None
    monkeypatch.setitem(sys.modules, module.__name__, module)

    hub = TrackerHub(run_dir, [{"kind": "custom", "adapter": "failing_tracker:record"}])
    hub.start_run()
    hub.log_metrics({"mean_reward": 1.5}, step=64)

    assert read_json(run_dir / "run.json")["tracker_status"][0]["status"] == "degraded"
    assert "64" in (run_dir / "progress.csv").read_text(encoding="utf-8")
```

## Operational rules

- Run `doctor --tracker ...` before a long experiment. Tracker checks are optional health checks; required local/GymServer failures still block startup.
- Use unique run names across services so a dashboard cannot merge unrelated seeds.
- Treat service graphs as conveniences. Base promotion decisions on the preserved local evaluation and checkpoint identity.
- If a service fails, inspect the degraded message, repair credentials/network outside the trainer, and keep the local run. Do not restart a healthy learner just to make a dashboard prettier.
