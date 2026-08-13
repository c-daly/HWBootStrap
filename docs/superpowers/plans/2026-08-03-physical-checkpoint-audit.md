# Physical Checkpoint Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a restart-safe, physically auditable comparison of the seed-227 behavioral clone, every checkpoint that actually exists in its target-KL PPO continuation, Random, bounded search, and an optional explicitly supplied scratch PPO reference on the established 100-map reciprocal development schedule.

**Architecture:** Keep the exploratory audit outside the frozen 21-candidate imitation panel. Extend the public evaluation boundary with explicit start-profile and evidence-retention options, then add a focused `ml_lab.checkpoint_audit` domain module and a thin orchestration script. Every aggregate is rebuilt from reopened per-map manifests and retained trace/replay files; no aggregate row is accepted as evidence by itself.

**Tech Stack:** Python 3.11+, `uv`, pytest, NumPy, Stable-Baselines3, sb3-contrib, the existing headless .NET duel server, JSON research artifacts, Markdown reports.

## Global Constraints

- This plan implements only Experiment 1 from `docs/superpowers/specs/2026-08-03-training-improvement-experiments-design.md`. DAgger and tactical-v3 require separate gated plans.
- The audit is exploratory and must say so in every root manifest and report. It cannot populate, replace, or satisfy `python/panels/annihilation-imitation-v1/development`, its 21-candidate matrix, global-budget selection, or its final gate.
- Use exactly map seeds `16_000_000` through `16_000_099`, `standard-3v3`, both candidate seats, and Random as the opponent: 200 games per candidate.
- Do not read from or assign the `17_000_000` final namespace.
- Discover checkpoints from physical files and their source manifest. Never synthesize a 51,200-step checkpoint from the stopped in-memory state at 51,036 steps.
- The known PPO archive currently contains only `14_336`, `26_624`, and `38_912`. A later archive may add candidates only through a new explicit audit definition.
- The known clone and PPO archives share tactical-v2 encoding hash `2f334bc2163fd931d84c004e9dc8f44bae68934e46fbf2ec2c819fa3e297054a`, observation size 1,292, and action size 1,288, but their full source contract hashes differ. Preserve both source contracts and record the single evaluation-time duel contract. Do not edit archive manifests to make hashes agree.
- A scratch candidate is optional, never auto-selected by scanning arbitrary runs, and included only when the caller supplies a physical run that passes the same scenario/encoding/seed validation.
- Existing evaluator behavior remains the default: diagnostic trace retention and no forced start profile. New audit behavior must be opt-in.
- Policy EndTurn rank/probability is unavailable through the current integer-action inference boundary. Record this metric as unavailable with a reason; do not infer logits from selected actions or enlarge this audit into a policy API redesign.
- All JSON publication is atomic. Interrupted per-map work may be reused only after reopening and validating physical files.
- Use `uv` for all Python commands. From the repository root on Windows:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests -q
  ```

- Do not change C# or Unity assets in this plan. Unity compilation and PlayMode tests are therefore not required; the headless duel smoke is required.
- Do not stage or alter unrelated dirty worktree files. Commit only the files named by the current task.
- Never add attribution trailers to commits or PR bodies.

---

## Task 1: Make the public evaluator express the audit schedule and evidence policy

**Files:**

- Modify: `python/ml_lab/evaluation.py:612-860,929-989`
- Modify: `python/ml_lab/cli.py:150-190,600-623`
- Test: `python/tests/test_evaluation.py`
- Test: `python/tests/test_cli.py`

### Step 1: Write failing tests for full evidence retention

- [ ] Add a test beside the existing trace-capture tests that evaluates two reciprocal maps whose four games all have the same outcome. The existing diagnostic policy retains one non-draw control per seat/outcome stratum (two artifacts here); this test proves the audit policy retains all four games.

  ```python
  def test_evaluate_matchup_all_retention_publishes_every_trace_and_replay(
      tmp_path: Path,
  ) -> None:
      candidate, opponent, factory = _trace_evaluation_fixture(
          tmp_path, outcomes=("win", "win", "win", "win")
      )

      result = evaluate_matchup(
          candidate,
          opponent,
          games=2,
          both_seats=True,
          workers=1,
          client_factory=factory,
          capture_trace=True,
          evidence_dir=tmp_path / "evidence",
          evidence_retention="all",
      )

      assert result["evidence"] == {
          "retention": "all",
          "retained": 4,
          "draw_traces": 0,
          "control_traces": 4,
          "draw_categories": {},
      }
      assert all(Path(row["trace_path"]).is_file() for row in result["matches"])
      assert all(Path(row["replay_path"]).is_file() for row in result["matches"])
  ```

- [ ] Add validation tests that call the same fixture-backed `evaluate_matchup` invocation with `evidence_retention="unknown"`, then with `evidence_retention="all"` and `capture_trace=False`. Assert `ValueError` messages matching `evidence_retention` and `requires trace capture and an evidence directory`, respectively. Fail rather than silently degrading.

### Step 2: Run the focused tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_evaluation.py -k "retention" -q
  ```

  Expected: failures because `evaluate_matchup` does not accept `evidence_retention` and does not publish retention metadata.

### Step 3: Implement the opt-in retention policy

- [ ] Introduce a narrow type and extend `evaluate_matchup` without changing the default behavior:

  ```python
  from typing import Literal

  EvidenceRetention = Literal["diagnostic", "all"]

  def evaluate_matchup(
      candidate: ResolvedController,
      opponent: ResolvedController,
      *,
      games: int,
      seed_start: int = DEFAULT_HELD_OUT_SEED,
      both_seats: bool = True,
      workers: int = 1,
      client_factory: Callable[[int], Any],
      predict_action: Callable[[Any, str, np.ndarray, np.ndarray], int] = predict,
      output_path: Path | None = None,
      start_profile: str | None = None,
      confidence: float = 0.95,
      evidence_dir: Path | None = None,
      capture_trace: bool = False,
      evidence_retention: EvidenceRetention = "diagnostic",
  ) -> dict[str, Any]:
  ```

- [ ] Validate the policy before creating clients or directories. In the publication loop, make retention explicit:

  ```python
  if evidence_retention not in {"diagnostic", "all"}:
      raise ValueError("evidence_retention must be 'diagnostic' or 'all'")
  if evidence_retention == "all" and (not capture_trace or evidence_dir is None):
      raise ValueError(
          "evidence_retention='all' requires trace capture and an evidence directory"
      )

  retain = evidence_retention == "all" or match["outcome"] == "draw"
  ```

- [ ] Keep `draw_trace_count` semantically equal to retained draws. Under `all`, count every retained non-draw in `control_trace_count`, and publish:

  ```python
  evidence_summary = {
      "retention": evidence_retention,
      "retained": draw_trace_count + control_trace_count,
      "draw_traces": draw_trace_count,
      "control_traces": control_trace_count,
      "draw_categories": dict(sorted(draw_categories.items())),
  }
  ```

### Step 4: Write failing public-wrapper and CLI propagation tests

- [ ] Add `test_evaluate_controllers_propagates_profile_and_retention` in `test_evaluation.py`. Monkeypatch `evaluate_matchup`, call `evaluate_controllers(..., start_profile="standard-3v3", evidence_retention="all")`, and assert both values reach the lower boundary.
- [ ] Add a CLI dispatch test in `test_cli.py` invoking:

  ```text
  evaluate --p0 random --p1 random --games 1 --both-seats
           --environment tactical-v2 --start-profile standard-3v3
           --capture-trace --evidence-retention all
           --evidence-dir C:\temp\audit-evidence
  ```

  Assert that `evaluate_controllers` receives `start_profile="standard-3v3"` and `evidence_retention="all"`.

### Step 5: Run the propagation tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_evaluation.py python/tests/test_cli.py -k "profile_and_retention or evidence_retention" -q
  ```

  Expected: wrapper signature/CLI option failures. This also exposes the current bug where the CLI parses `--start-profile` but does not pass it to `evaluate_controllers`.

### Step 6: Extend the public wrapper and CLI

- [ ] Extend `evaluate_controllers` with:

  ```python
  start_profile: str | None = None,
  evidence_retention: EvidenceRetention = "diagnostic",
  ```

  and pass both arguments to `evaluate_matchup`.

- [ ] Add the CLI option:

  ```python
  evaluate.add_argument(
      "--evidence-retention",
      choices=("diagnostic", "all"),
      default="diagnostic",
      help="retain diagnostic traces or every trace/replay",
  )
  ```

- [ ] Pass both `args.start_profile` and `args.evidence_retention` through the evaluate command. Do not change other commands.

### Step 7: Run the complete evaluator and CLI test files

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_evaluation.py python/tests/test_cli.py -q
  ```

  Expected: PASS with the old diagnostic-retention tests unchanged.

### Step 8: Commit Task 1

- [ ] Stage only the four Task 1 files and commit:

  ```powershell
  git add python/ml_lab/evaluation.py python/ml_lab/cli.py python/tests/test_evaluation.py python/tests/test_cli.py
  git commit -m "feat: make evaluation evidence retention explicit"
  ```

---

## Task 2: Define and discover immutable physical audit candidates

**Files:**

- Create: `python/ml_lab/checkpoint_audit.py`
- Create: `python/tests/test_checkpoint_audit.py`

### Step 1: Write failing tests for the candidate manifest

- [ ] Build tiny clone/PPO run fixtures with real checkpoint bytes and version-1 `run.json`/`scenario.json` files. Test this exact trajectory:

  ```python
  candidates = discover_audit_candidates(
      clone_run=clone_run,
      ppo_run=ppo_run,
      scratch_run=None,
  )

  assert [(row.candidate_id, row.actual_step) for row in candidates] == [
      ("pure-bc-seed-227", 0),
      ("bc-ppo-seed-227-step-000014336", 14_336),
      ("bc-ppo-seed-227-step-000026624", 26_624),
      ("bc-ppo-seed-227-step-000038912", 38_912),
      ("random-anchor", None),
      ("bounded-search-anchor", None),
  ]
  ```

- [ ] Assert every learned candidate has a canonical absolute source path, SHA-256 of the physical checkpoint, complete source contract identity, scenario snapshot digest, observation/action sizes, and encoding hash. Assert controls have stable scripted controller specs and no fabricated checkpoint fields.
- [ ] Assert the PPO source manifest may be `state="stopped"`, but only files matching `checkpoints/step_<nine digits>.zip` enter the trajectory. The manifest target and stopped step do not create candidates.
- [ ] Add rejection tests for:

  - missing or malformed checkpoint bytes;
  - filename/recorded-step disagreement;
  - duplicate or decreasing checkpoint steps;
  - clone/PPO model seed other than 227;
  - PPO `actor_init_source` not resolving to the supplied clone;
  - scenario snapshot mismatch;
  - environment/version/encoding/observation/action incompatibility;
  - a supplied scratch run that is BC-initialized, seed-mismatched, or lacks a physical checkpoint;
  - any attempt to name a nonphysical 51,200 candidate.

### Step 2: Run the candidate tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -k "candidate or checkpoint or contract" -q
  ```

  Expected: import failure because `ml_lab.checkpoint_audit` does not exist.

### Step 3: Add the immutable domain types

- [ ] Implement these public records in `checkpoint_audit.py`:

  ```python
  @dataclass(frozen=True)
  class AuditSchedule:
      seed_start: int = 16_000_000
      maps: int = 100
      both_seats: bool = True
      profile: str = "standard-3v3"
      opponent: str = "random"

  @dataclass(frozen=True)
  class AuditCandidate:
      candidate_id: str
      family: Literal["pure_bc", "bc_ppo", "scratch_ppo", "control"]
      trajectory_order: int | None
      controller: str
      model_seed: int | None
      actual_step: int | None
      checkpoint_path: str | None
      checkpoint_sha256: str | None
      source_run: str | None
      source_run_manifest_sha256: str | None
      source_scenario_sha256: str | None
      source_contract_hash: str | None
      source_encoding_hash: str | None
      observation_size: int | None
      action_size: int | None

  @dataclass(frozen=True)
  class AuditDefinition:
      schema_version: int
      audit_id: str
      exploratory: bool
      locked_panel_replacement: bool
      schedule: AuditSchedule
      candidates: tuple[AuditCandidate, ...]
      omitted_optional_candidates: tuple[Mapping[str, str], ...]
  ```

- [ ] Use JSON controller specs for learned candidates. The clone uses `kind="run"`; physical intermediate checkpoints use `kind="snapshot"` with `path`, `source_run`, `algorithm`, and `step`. Controls use `random` and `bounded-search`.

### Step 4: Implement fail-closed discovery

- [ ] Implement:

  ```python
  def discover_audit_candidates(
      *,
      clone_run: Path,
      ppo_run: Path,
      scratch_run: Path | None,
  ) -> tuple[AuditCandidate, ...]:

  def build_audit_definition(
      *,
      clone_run: Path,
      ppo_run: Path,
      scratch_run: Path | None,
  ) -> AuditDefinition:
  ```

  Requirements:

  1. Resolve every input to a canonical absolute directory and reopen its `run.json` and `scenario.json`.
  2. Read the clone's `latest_checkpoint`; require actual step zero and hash the referenced file.
  3. Enumerate PPO `checkpoints/step_*.zip` from disk, parse steps from anchored filenames, sort numerically, and hash every file.
  4. Require PPO `actor_init_source` to resolve to the supplied clone and require `algorithm="maskable_ppo"`, seed/model seed 227, and the same resolved scenario bytes.
  5. Compare source environment, encoding version/hash, observation size, and action size. Record but do not require equality of full contract hash or board horizon.
  6. If `scratch_run` is omitted, add `{"family": "scratch_ppo", "reason": "no physical compatible run supplied"}` to `omitted_optional_candidates`. Do not scan `python/runs` and choose one heuristically.
  7. Append controls after learned candidates.

- [ ] Serialize through explicit `to_dict()` methods so tuples, `Path`, enums, and dataclasses never depend on incidental `json` behavior.

### Step 5: Run all domain tests

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -q
  ```

  Expected: PASS.

### Step 6: Commit Task 2

- [ ] Stage only the new module/test and commit:

  ```powershell
  git add python/ml_lab/checkpoint_audit.py python/tests/test_checkpoint_audit.py
  git commit -m "feat: define physical checkpoint audit candidates"
  ```

---

## Task 3: Evaluate each map restart-safely and validate physical evidence

**Files:**

- Modify: `python/ml_lab/checkpoint_audit.py`
- Modify: `python/tests/test_checkpoint_audit.py`

### Step 1: Write failing tests for the exact reciprocal schedule

- [ ] Inject a fake evaluator and assert `evaluate_audit` makes one call per candidate/map with:

  ```python
  {
      "games": 1,
      "seed_start": map_seed,
      "both_seats": True,
      "environment": "tactical-v2",
      "start_profile": "standard-3v3",
      "capture_trace": True,
      "evidence_retention": "all",
  }
  ```

- [ ] Assert the schedule keys are exactly:

  ```python
  [
      (candidate.candidate_id, map_seed, seat)
      for candidate in definition.candidates
      for map_seed in range(16_000_000, 16_000_100)
      for seat in (0, 1)
  ]
  ```

- [ ] Test idempotent reuse: on a second call, every `evaluation.json`, trace, replay, controller identity, schedule field, and checkpoint digest is reopened and validated; the evaluator receives zero calls.
- [ ] Parameterize tampering of the candidate controller, opponent, seed, seat, profile, runtime contract, outcome totals, summary, trace path, replay path, trace bytes, replay bytes, and checkpoint bytes. Every mutation must reject reuse.
- [ ] Test that an interrupted audit with 99 valid maps and no root completion manifest resumes only the missing map. A root aggregate must never be published from 199 games.

### Step 2: Run the evaluation tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -k "schedule or reuse or tamper or interrupted" -q
  ```

  Expected: failures because the evaluation functions do not exist.

### Step 3: Implement paths and per-map validation

- [ ] Use this stable layout beneath the caller-provided output root:

  ```text
  manifest.json
  candidates/
    <candidate-id>/
      map-16000000/
        evaluation.json
        evidence/traces/*.json
        evidence/replays/*.replay
      map-16000099/
        evaluation.json
        evidence/traces/*.json
        evidence/replays/*.replay
  audit.json
  report.md
  ```

- [ ] Implement:

  ```python
  def audit_map_path(root: Path, candidate_id: str, map_seed: int) -> Path:
      return root / "candidates" / candidate_id / f"map-{map_seed}" / "evaluation.json"

  def validate_physical_map(
      root: Path,
      candidate: AuditCandidate,
      schedule: AuditSchedule,
      map_seed: int,
  ) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], Mapping[str, Any]]]:

  def evaluate_audit(
      definition: AuditDefinition,
      *,
      output_root: Path,
      server_cmd: Sequence[str],
      workers: int,
      evaluator: Callable[..., Mapping[str, Any]] = evaluate_controllers,
      progress: Callable[[str], None] = print,
  ) -> Mapping[str, Any]:
  ```

- [ ] After a new evaluator result is published, reopen it, add an `audit_identity` object containing the frozen definition hash, scenario SHA-256, evaluation-time runtime contract, and SHA-256 for each match's trace and replay, then rewrite `evaluation.json` atomically. `validate_physical_map` must require:

  - schema version 1 and a timestamp;
  - the exact candidate controller identity and Random opponent;
  - one seed, reciprocal seats, exactly two games, and `standard-3v3` with candidate-seat reference;
  - an evaluation-time tactical-v2 contract with the expected encoding/geometry;
  - reconciled W/L/D totals, rates, seats, and Wilson fields;
  - trace summary and, for draws, draw classification;
  - non-null trace/replay paths for both matches, contained below the output root;
  - SHA-256 fields for each retained trace/replay, computed after publication;
  - current physical checkpoint bytes still matching the candidate manifest.

- [ ] Normalize artifact paths relative to the audit root before publication. Never accept `..`, an absolute artifact path, a symlink escape, or a missing file.

### Step 4: Implement evaluation and progress logging

- [ ] Before work, atomically write `manifest.json` containing the exact definition, definition SHA-256, repository identity, scenario bytes/hash, source contracts, and an explicit `state="in_progress"`. If it already exists, require byte-equivalent definition identity.
- [ ] For each candidate/map, first call `validate_physical_map`. Reuse only on success. A malformed existing map fails closed; do not overwrite evidence silently.
- [ ] If missing, invoke the evaluator and validate immediately. Log one concise line at candidate start, every ten completed maps, every reuse, and candidate completion. Include elapsed time and rolling ETA:

  ```text
  [2/6 bc-ppo-seed-227-step-000014336] maps 40/100, games 80/200, reused 12, elapsed 00:07:31, eta 00:11:16
  ```

- [ ] Do not set the root manifest to `completed` in this task. Completion belongs to physical aggregation in Task 4.

### Step 5: Run domain tests

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -q
  ```

  Expected: PASS.

### Step 6: Commit Task 3

- [ ] Stage only the Task 3 files and commit:

  ```powershell
  git add python/ml_lab/checkpoint_audit.py python/tests/test_checkpoint_audit.py
  git commit -m "feat: evaluate checkpoint audit with durable evidence"
  ```

---

## Task 4: Aggregate tactical metrics, paired changes, and deterministic decisions

**Files:**

- Modify: `python/ml_lab/checkpoint_audit.py`
- Modify: `python/tests/test_checkpoint_audit.py`

### Step 1: Write failing summary tests

- [ ] Create a 200-row synthetic candidate table with known outcomes, seats, summaries, and draw flags. Assert `summarize_candidate` returns:

  - W/L/D counts, rates, and 95% Wilson intervals;
  - per-seat counts/rates and `win_rate_p0_minus_p1`;
  - cycling/action-waste draw counts and incidence over all 200 games;
  - primary draw-category counts;
  - winning-game mean, median, and p90 for `round_count` and `command_count`;
  - all-game and draw-only mean final/peak normalized advantage;
  - candidate-seat wasted EndTurns and wasted-EndTurn ratio;
  - `end_turn_policy_diagnostics={"available": false, "reason": "integer-action inference boundary does not expose action probabilities or ranks"}`.

- [ ] Do not rename trace fields in raw evidence. Map them only in aggregate presentation:

  ```python
  TRACE_FIELDS = {
      "rounds": "round_count",
      "decisions": "command_count",
      "peak_health_adjusted_advantage": "peak_normalized_advantage",
      "final_health_adjusted_advantage": "final_normalized_advantage",
  }
  ```

### Step 2: Write failing paired-comparison tests

- [ ] Index every table by `(map_seed, candidate_seat)` and reject duplicates or missing keys. For each adjacent learned trajectory pair, assert a full 3-by-3 transition table (`win|draw|loss` to `win|draw|loss`), left/right-only wins, net win change, absolute win-rate change, and two-sided exact sign-test p-value.
- [ ] Assert controls are summarized as anchors but are not inserted into the PPO trajectory or used to trigger the retention rule.

### Step 3: Write failing decision tests

- [ ] Implement the approved rules with explicit operational definitions:

  ```python
  def choose_next_experiment(
      trajectory: Sequence[CandidateAggregate],
  ) -> Mapping[str, Any]:
      """Return independent clauses plus one recommended next step."""
  ```

  Definitions:

  - `qualifying_ppo`: a physical PPO checkpoint with win rate at least `0.65`.
  - `consistent_improvement`: from clone through the earliest qualifying PPO checkpoint, every adjacent absolute win-rate change is nonnegative and at least one is positive.
  - `large_late_regression`: among PPO checkpoints only, any earlier checkpoint exceeds any later checkpoint by at least `0.10` absolute win rate.
  - `cycling_dominant`: for the latest PPO checkpoint, cycling draws are a plurality over wins, losses, and all non-cycling draws combined.
  - `all_ppo_below_half`: every physical PPO checkpoint has win rate below `0.50`.

  Recommendation precedence:

  1. `test_retained_imitation_constraint` if `large_late_regression`;
  2. `replicate_seeds_211_223` if a qualifying checkpoint has consistent improvement;
  3. `proceed_to_dagger` if all PPO checkpoints are below half or cycling is dominant;
  4. `inconclusive_review_trajectory` otherwise.

- [ ] Report all independent clauses even when precedence chooses a different next step. This prevents a single label from hiding mixed evidence.

### Step 4: Run the aggregation tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -k "summary or paired or decision" -q
  ```

  Expected: failures because the aggregation functions do not exist.

### Step 5: Implement aggregation from reopened physical maps

- [ ] Implement:

  ```python
  def summarize_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:

  def paired_change(
      earlier: Sequence[Mapping[str, Any]],
      later: Sequence[Mapping[str, Any]],
  ) -> Mapping[str, Any]:

  def aggregate_audit(
      definition: AuditDefinition,
      *,
      output_root: Path,
  ) -> Mapping[str, Any]:
  ```

- [ ] `aggregate_audit` must reopen all 100 per-map manifests per candidate via `validate_physical_map`, require exactly 200 ordered match rows, then write `audit.json` atomically. Include:

  ```json
  {
    "schema_version": 1,
    "audit_id": "annihilation-checkpoint-audit-v1",
    "exploratory": true,
    "locked_panel_replacement": false,
    "schedule": {},
    "candidates": {},
    "trajectory": [],
    "paired_successive_changes": [],
    "anchors": ["random-anchor", "bounded-search-anchor"],
    "decision": {},
    "physical_evidence": {"maps": 0, "games": 0, "traces": 0, "replays": 0}
  }
  ```

- [ ] Only after successful aggregate validation, atomically update the root manifest to `state="completed"` with the SHA-256 of `audit.json`. Reopening a completed audit must revalidate every physical map before returning.

### Step 6: Run all audit-domain tests

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py -q
  ```

  Expected: PASS.

### Step 7: Commit Task 4

- [ ] Stage only the Task 4 files and commit:

  ```powershell
  git add python/ml_lab/checkpoint_audit.py python/tests/test_checkpoint_audit.py
  git commit -m "feat: aggregate checkpoint trajectory decisions"
  ```

---

## Task 5: Add the audit CLI, protocol, and human-readable report

**Files:**

- Create: `python/run_annihilation_checkpoint_audit.py`
- Create: `python/tests/test_annihilation_checkpoint_audit.py`
- Create: `python/panels/annihilation-checkpoint-audit-v1/PROTOCOL.md`

### Step 1: Write failing parser and dispatch tests

- [ ] Define these subcommands and test their exact required inputs:

  ```text
  prepare   --clone-run PATH --ppo-run PATH [--scratch-run PATH] --output-root PATH
  validate  --output-root PATH
  evaluate  --output-root PATH [--workers N]
  aggregate --output-root PATH
  report    --output-root PATH
  all       --clone-run PATH --ppo-run PATH [--scratch-run PATH]
            --output-root PATH [--workers N]
  ```

- [ ] Assert `all` executes `prepare -> validate -> evaluate -> aggregate -> report` and stops immediately on any failure. Assert `prepare` never launches a duel server and `report` never trusts an unvalidated `audit.json`.
- [ ] Assert `validate` prints the discovered physical candidate IDs/steps and explicitly prints that no 51,200 candidate exists when it is absent.

### Step 2: Run parser tests and observe failure

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_annihilation_checkpoint_audit.py -q
  ```

  Expected: import failure because the runner does not exist.

### Step 3: Implement the thin runner

- [ ] Keep orchestration in the script and all rules in `ml_lab.checkpoint_audit`. Use the established duel server command and scenario:

  ```python
  ROOT = Path(__file__).resolve().parents[1]
  SCENARIO = ROOT / "python" / "config" / "annihilation-imitation-v1.json"
  DEFAULT_OUTPUT = ROOT / "python" / "evidence" / "annihilation-checkpoint-audit-v1"

  def server_command() -> list[str]:
      return [
          "dotnet",
          str(ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"),
          "--scenario-file",
          str(SCENARIO),
      ]
  ```


- [ ] Use Python logging directed to both stdout and `<output-root>/audit.log`. Log timestamps, command, candidate, map progress, reuse, elapsed time, ETA, exceptions, and final artifact paths. Do not log full observations or model tensors.

- [ ] `prepare` writes the definition only when absent. If present, it rediscovers all physical inputs and requires the same serialized definition and hashes.
- [ ] `validate` opens one duel client before compute and verifies the runtime environment/version/encoding/observation/action identity against every learned candidate. It records the runtime contract separately; it does not demand equality with each full source contract hash.

### Step 4: Render a concise evidence-backed report

- [ ] Implement `render_report(aggregate)` with these sections:

  1. exploratory status and non-replacement warning;
  2. exact schedule and runtime/source identities;
  3. W/L/D, Wilson win interval, cycling/action-waste incidence, win rounds/decisions, advantage, and seat delta per candidate;
  4. successive paired transition table;
  5. Random and bounded-search anchors;
  6. deterministic decision clauses and recommended next experiment;
  7. unavailable EndTurn rank/probability explanation;
  8. omitted optional scratch reference, if any;
  9. physical artifact counts and hashes.

- [ ] The report must state `51,036 was a stopped in-memory training count, not an evaluated checkpoint` whenever the PPO manifest records that stop without a 51,200 file.

### Step 5: Write the protocol document

- [ ] In `PROTOCOL.md`, document the question, candidates, schedule, metric definitions, decision precedence, seed isolation, commands, output tree, recovery behavior, and why source full-contract hashes may differ while encoding compatibility remains required.
- [ ] Explicitly reserve DAgger's 18m/19m/20m banks without consuming them and leave 17m untouched.

### Step 6: Run runner and domain tests

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_checkpoint_audit.py python/tests/test_annihilation_checkpoint_audit.py -q
  ```

  Expected: PASS.

### Step 7: Commit Task 5

- [ ] Stage only the runner, tests, and protocol and commit:

  ```powershell
  git add python/run_annihilation_checkpoint_audit.py python/tests/test_annihilation_checkpoint_audit.py python/panels/annihilation-checkpoint-audit-v1/PROTOCOL.md
  git commit -m "feat: add physical checkpoint audit workflow"
  ```

---

## Task 6: Verify the implementation before spending the full evaluation budget

**Files:**

- Modify if tests expose defects: only files introduced or intentionally modified in Tasks 1-5
- Evidence only, not committed: a temporary audit directory under `$env:TEMP`

### Step 1: Run the focused suite from a fresh process

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests/test_evaluation.py python/tests/test_cli.py python/tests/test_checkpoint_audit.py python/tests/test_annihilation_checkpoint_audit.py -q
  ```

  Expected: PASS.

### Step 2: Run the entire Python suite

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python -m pytest python/tests -q
  ```

  Expected: PASS. Investigate any failure; do not dismiss it as unrelated without reproducing it against the pre-task revision.

### Step 3: Run the engine tests

- [ ] Run:

  ```powershell
  dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --no-restore
  ```

  Expected: PASS. Although this plan changes no C#, the live audit depends on the duel server and authoritative trace behavior.

### Step 4: Prepare and validate the real physical inputs without playing games

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  $auditRoot = Join-Path $env:TEMP 'hexwars-checkpoint-audit-preflight'
  uv run python python/run_annihilation_checkpoint_audit.py prepare `
    --clone-run 'C:\Users\cddal\HexWars\.worktrees\tactical-baseline-evidence\python\panels\annihilation-imitation-v1\bc-clones\seed-227' `
    --ppo-run 'C:\Users\cddal\HexWars\python\runs\bc227-ppo-random-s227-20260802-v2' `
    --output-root $auditRoot
  uv run python python/run_annihilation_checkpoint_audit.py validate --output-root $auditRoot
  ```

  Expected candidates: clone step 0; PPO steps 14,336, 26,624, and 38,912; Random; bounded search. Expected omission: scratch PPO. Expected absence: 51,200.

### Step 5: Run a bounded two-map smoke

- [ ] Add a test-only/programmatic smoke entry—not a publishable CLI schedule override—that constructs `AuditSchedule(seed_start=16_000_000, maps=2, both_seats=True, profile="standard-3v3", opponent="random")`, evaluates all discovered candidates, aggregates, and reopens all evidence. Mark its manifest `smoke=true`, `exploratory=true`, `locked_panel_replacement=false`.
- [ ] Run it through a short test script or pytest integration marker, then rerun it to prove complete physical reuse. Confirm the second run logs zero new games.
- [ ] Confirm every smoke match has a trace, replay, summary, and matching digest, including two same-outcome reciprocal games.

### Step 6: Review the diff and commit any verification-only fixes

- [ ] Run:

  ```powershell
  git diff --check
  git status --short
  git diff -- python/ml_lab/evaluation.py python/ml_lab/cli.py python/ml_lab/checkpoint_audit.py python/run_annihilation_checkpoint_audit.py python/tests/test_evaluation.py python/tests/test_cli.py python/tests/test_checkpoint_audit.py python/tests/test_annihilation_checkpoint_audit.py python/panels/annihilation-checkpoint-audit-v1/PROTOCOL.md
  ```

- [ ] If verification required fixes, commit only those named files:

  ```powershell
  git add python/ml_lab/evaluation.py python/ml_lab/cli.py python/ml_lab/checkpoint_audit.py python/run_annihilation_checkpoint_audit.py python/tests/test_evaluation.py python/tests/test_cli.py python/tests/test_checkpoint_audit.py python/tests/test_annihilation_checkpoint_audit.py python/panels/annihilation-checkpoint-audit-v1/PROTOCOL.md
  git commit -m "test: verify physical checkpoint audit"
  ```

---

## Task 7: Run the complete seed-227 physical audit

**Files:**

- Generated evidence, not source: `python/evidence/annihilation-checkpoint-audit-v1/`
- No source modifications expected

### Step 1: Freeze the physical definition

- [ ] From the clean reviewed implementation revision, run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python python/run_annihilation_checkpoint_audit.py prepare `
    --clone-run 'C:\Users\cddal\HexWars\.worktrees\tactical-baseline-evidence\python\panels\annihilation-imitation-v1\bc-clones\seed-227' `
    --ppo-run 'C:\Users\cddal\HexWars\python\runs\bc227-ppo-random-s227-20260802-v2' `
    --output-root 'C:\Users\cddal\HexWars\python\evidence\annihilation-checkpoint-audit-v1'
  uv run python python/run_annihilation_checkpoint_audit.py validate `
    --output-root 'C:\Users\cddal\HexWars\python\evidence\annihilation-checkpoint-audit-v1'
  ```

- [ ] Read `manifest.json` before compute. Confirm six candidates, the three physical PPO steps, no scratch candidate, no 51,200 candidate, the 16m schedule, and `locked_panel_replacement=false`.

### Step 2: Evaluate with visible logging

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python python/run_annihilation_checkpoint_audit.py evaluate `
    --output-root 'C:\Users\cddal\HexWars\python\evidence\annihilation-checkpoint-audit-v1' `
    --workers 4
  ```

  Expected workload: 1,200 games for six candidates. Keep training/Unity audio off; this is a headless duel-server evaluation.

- [ ] If interrupted, rerun the identical command. It must validate/reuse finished maps and log only missing work. Never delete or overwrite a malformed map; investigate the physical discrepancy.

### Step 3: Aggregate and report

- [ ] Run:

  ```powershell
  $env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path
  uv run python python/run_annihilation_checkpoint_audit.py aggregate `
    --output-root 'C:\Users\cddal\HexWars\python\evidence\annihilation-checkpoint-audit-v1'
  uv run python python/run_annihilation_checkpoint_audit.py report `
    --output-root 'C:\Users\cddal\HexWars\python\evidence\annihilation-checkpoint-audit-v1'
  ```

- [ ] Re-run `validate`, then independently verify counts:

  - six candidates;
  - 600 physical map manifests;
  - 1,200 ordered match rows;
  - 1,200 trace files and 1,200 replay files;
  - exactly 100 maps and both seats per candidate;
  - no seed outside 16,000,000-16,000,099;
  - completed manifest digest equals physical `audit.json` bytes.

### Step 4: Apply the decision without overclaiming

- [ ] Read `report.md`, `audit.json`, and at least one representative trace/replay from every draw category and every adjacent trajectory disagreement.
- [ ] Record the recommended next step exactly as computed:

  - reproduce seeds 211/223 only if the qualifying/consistent clause passes;
  - plan retained-imitation PPO if the ten-point late-regression clause passes;
  - proceed to the DAgger implementation plan if all PPO checkpoints remain below 50% or cycling is dominant;
  - otherwise document the trajectory as inconclusive before authorizing more compute.

- [ ] Do not merge these results into the locked imitation panel or consume final seeds. The output answers the checkpoint-trajectory question and selects the next experiment; it is not a production model promotion.

---

## Final Verification Checklist

- [ ] Every public evaluator test passes with diagnostic retention still the default.
- [ ] `--start-profile` reaches `evaluate_matchup`; it is no longer parsed and discarded.
- [ ] Full evidence retention publishes and hashes one trace/replay pair per game.
- [ ] Candidate discovery finds physical files only and cannot synthesize step 51,200.
- [ ] Source full-contract differences are preserved; encoding compatibility and one evaluation runtime are explicit.
- [ ] The optional scratch reference is either physically supplied and validated or explicitly omitted.
- [ ] Every aggregate can be regenerated by reopening 600 physical map manifests.
- [ ] Metrics include W/L/D, Wilson intervals, paired changes, draw pathologies, win speed, advantage, and seat asymmetry.
- [ ] EndTurn rank/probability is marked unavailable, not fabricated.
- [ ] Decision clauses are deterministic and the report remains explicitly exploratory.
- [ ] No final seeds, locked development artifacts, Unity assets, or unrelated dirty files were changed.
