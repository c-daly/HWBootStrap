# Live Training Stochastic Viewer Design

## Goal

Make **Start & Watch** show useful behavior from early Maskable PPO checkpoints without changing the reproducibility of deliberate Arena or evaluation duels. Early PPO checkpoints can choose the same legal deployment cancellation forever when inference always takes the highest-probability action. Training itself samples from the policy distribution and does not have that deterministic loop.

## Behavior

- **Start & Watch** launches its learner seat in a stochastic inference mode.
- Stochastic Maskable PPO inference samples from the masked policy distribution using SB3's native `deterministic: false` behavior. It never bypasses the legal-action mask.
- Manually configured Arena seats, fixed-run duels, live-run Arena duels, command-line evaluation, and official-AI inference remain deterministic by default.
- The opponent selected for Start & Watch retains its existing behavior. Scripted Greedy and Random opponents are unchanged.
- Reloading a newer checkpoint between games preserves the seat's chosen inference mode.
- The mode is developer-lab metadata only. It does not make unfinished models selectable by regular players.

## Data Flow

`MlLabWindow.StartTraining(watch: true)` continues to wait for the first published checkpoint. When it launches the live viewer, Unity builds a metadata-backed live-run controller specification that explicitly requests stochastic inference for the learner seat. `PolicyBridge` forwards that specification unchanged to `policy_server.py`. The Python controller binding validates the mode and supplies it to the algorithm-specific prediction function.

Specifications without an inference-mode field remain deterministic. This preserves compatibility with existing Arena configuration, evaluation commands, saved editor state, and official-model paths.

## Algorithm Semantics

- **Maskable PPO:** stochastic mode calls `model.predict(..., action_masks=mask, deterministic=False)`; deterministic mode continues to pass `True`.
- **Masked DQN:** this change does not invent a PPO-style sampler. Its existing deterministic masked argmax remains unchanged until DQN viewing behavior is designed separately.

## Failure Handling

- Unknown inference modes are rejected with a clear controller-resolution error.
- A malformed stochastic specification cannot silently fall back to deterministic behavior.
- Policy-server status metadata reports the resolved inference mode so Unity can show what was actually loaded.

## Testing

- Python controller tests prove the default remains deterministic.
- Python controller tests prove a stochastic PPO run passes `deterministic=False` while retaining the action mask.
- Controller-spec tests prove live reload retains the inference mode and rejects unknown values.
- Unity tests prove **Start & Watch** builds a stochastic live-run specification.
- Unity tests prove ordinary Arena `FixedRun` and `LiveRun` specifications remain deterministic/default.
- An end-to-end smoke test loads a metadata-backed adaptive checkpoint through the policy server and performs masked stochastic actions without changing evaluation defaults.

## Out of Scope

- Displaying arbitrary historical checkpoints in the Arena picker.
- Revealing either player's private deployment.
- Adding epsilon-greedy exploration to PPO.
- Changing training, evaluation, checkpoint publication, or official-AI promotion rules.
- Adding a deployment progress overlay or changing the Arena animation rate.
