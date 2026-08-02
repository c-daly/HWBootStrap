# Materialized Imitation Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-batch shard thrashing with exact-sequence vectorized sampling from validated materialized partitions and require at least 5x production input throughput before restarting behavioral cloning.

**Architecture:** Materialize each dataset partition once in canonical logical order and retain an immutable physical-reference-to-offset map. Keep the current seeded stratum cyclers and batch permutation unchanged, but translate their ordered references into vectorized NumPy gathers; add phase-timing evidence and a read-only 200-batch production benchmark command.

**Tech Stack:** Python 3.14, NumPy, PyTorch 2.12.1+cu130, Stable-Baselines3/MaskablePPO, pytest, PowerShell, .NET 8 GymServer.

## Global Constraints

- Work only in `C:\Users\cddal\HexWars\.worktrees\tactical-baseline-evidence`.
- Use only `C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe`; never use uv or WSL.
- Set `PYTHONPATH=python` and `PYTHONDONTWRITEBYTECODE=1` for every Python command.
- Preserve the exact physical reference sequence, batch contents, source ratio, seeds, batch size, model, optimizer, learning rate, clipping, patience, epoch limit, and optimizer-step order.
- Do not reorder examples for locality and do not preload the complete dataset onto CUDA.
- Preserve exact CPU fixture-logit reload equality and all device-provenance, identity, dataset-revision, and atomic-publication checks.
- The optimizer hot path must not hash, open, decode, or read physical shards after materialization.
- The production benchmark uses exactly 200 consecutive batches and must achieve at least 2,825 examples per second.
- Timing fields are exactly `sampling_seconds`, `transfer_forward_seconds`, `optimization_seconds`, `validation_seconds`, and `unclassified_seconds`.
- Use TDD for every behavior change. Do not commit datasets, models, staging, process logs, Python bytecode, or evidence archives.
- Never add attribution trailers to commits.
- Do not touch final evaluation seeds.

---

## File and Responsibility Map

- `python/ml_lab/imitation.py`: materialized-partition representation, grouped physical gathering, exact reference scheduling, vectorized batch gathering, phase timing, and sampler benchmark function.
- `python/tests/test_imitation.py`: exact-equivalence, no-hot-path-read, malformed-materialization, timing-schema, benchmark, and real-CUDA regression tests.
- `python/run_annihilation_imitation_panel.py`: retained-history timing validation and the identity-bound `benchmark-bc-input` command.
- `python/tests/test_annihilation_imitation_panel.py`: physical timing-history rejection, benchmark command, CLI dispatch, and locked-threshold tests.
- `python/panels/annihilation-imitation-v1/PROTOCOL.md`: materialized sampler, phase fields, benchmark threshold, and evidence-restart procedure.

### Task 1: Materialize Partitions and Preserve Exact Sampling

**Files:**
- Modify: `python/ml_lab/imitation.py:407-422, 633-685, 784-820, 1090-1146`
- Modify: `python/tests/test_imitation.py:24, 155-227, 633-655`

**Interfaces:**
- Produces: `MaterializedImitationPartition(partition: str, batch: ImitationBatch, offsets: Mapping[tuple[int, int], int])`.
- Produces: `materialize_imitation_partition(dataset: ImitationDataset, partition: str) -> MaterializedImitationPartition`.
- Produces: `StratifiedDecisionSampler._next_refs_and_sources() -> list[tuple[tuple[int, int], Source]]`.
- Changes: `StratifiedDecisionSampler(dataset, materialized, batch_size=1, standard_fraction=0.70, seed=0, partition="train")` requires the matching materialized view.
- Preserves: `next_batch() -> ImitationBatch` and the existing scheduler sequence.

- [ ] **Step 1: Add imports and an exact legacy-gather oracle in tests**

Add `from types import MappingProxyType` to the test module.

Add `MaterializedImitationPartition` and `materialize_imitation_partition` to the import from `ml_lab.imitation`.

Add this test-only oracle beside the sampler tests:

```python
def _physical_batch_for_scheduled_refs(
    dataset,
    refs_and_sources: list[tuple[tuple[int, int], Source]],
) -> ImitationBatch:
    refs = [ref for ref, _source in refs_and_sources]
    rows = dataset._row_data(refs)
    legal_masks = np.unpackbits(
        rows["packed_masks"], axis=1,
        count=dataset.contract.action_size, bitorder="little",
    ).astype(bool, copy=False)
    metadata = [dataset.games[int(game_id)] for game_id in rows["game_ids"]]
    return ImitationBatch(
        observations=rows["observations"].copy(),
        legal_masks=legal_masks.copy(),
        actions=rows["actions"].copy(),
        game_ids=rows["game_ids"].copy(),
        decision_indices=rows["decision_indices"].copy(),
        sources=np.asarray([source for _ref, source in refs_and_sources], dtype=object),
        profiles=np.asarray([game["profile"] for game in metadata], dtype=object),
        seats=rows["seats"].copy(),
        action_kinds=rows["action_kinds"].copy(),
        partitions=np.asarray([game["partition"] for game in metadata], dtype=object),
    )
```

- [ ] **Step 2: Write failing exact-equivalence and no-reread tests**

```python
def test_materialized_sampler_matches_physical_batches_and_never_rereads(
    sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    materialized = materialize_imitation_partition(dataset, "train")
    oracle_scheduler = StratifiedDecisionSampler(
        dataset, materialized, batch_size=100,
        standard_fraction=0.70, seed=211,
    )
    optimized = StratifiedDecisionSampler(
        dataset, materialized, batch_size=100,
        standard_fraction=0.70, seed=211,
    )
    expected_batches = [
        _physical_batch_for_scheduled_refs(
            dataset, oracle_scheduler._next_refs_and_sources()
        )
        for _ in range(4)
    ]
    monkeypatch.setattr(
        dataset._cache, "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("physical shard access entered sampler hot path")
        ),
    )
    actual_batches = [optimized.next_batch() for _ in range(4)]
    for expected, actual in zip(expected_batches, actual_batches, strict=True):
        for field in ImitationBatch.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(expected, field)
            )
```

```python
def test_materialized_partition_rejects_wrong_partition_and_missing_reference(
    sampled_dataset: Path,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    validation = materialize_imitation_partition(dataset, "validation")
    with pytest.raises(ValueError, match="partition"):
        StratifiedDecisionSampler(
            dataset, validation, batch_size=1, partition="train", seed=211,
        )
    train = materialize_imitation_partition(dataset, "train")
    with pytest.raises(ValueError, match="reference map"):
        MaterializedImitationPartition(
            partition=train.partition,
            batch=train.batch,
            offsets=MappingProxyType(dict(list(train.offsets.items())[1:])),
        )
```

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "materialized_sampler or materialized_partition"
```

Expected: collection fails because the materialized interfaces do not exist.

- [ ] **Step 4: Implement grouped physical gathering**

Replace the per-row body of `ImitationDataset._row_data` with shard-grouped, vectorized gathering while preserving the input reference order:

```python
def _row_data(self, refs: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
    if not refs:
        raise ValueError("row gathering requires at least one reference")
    grouped: dict[int, list[tuple[int, int]]] = {}
    for destination, (shard_index, local_row) in enumerate(refs):
        if shard_index not in range(len(self.shards)):
            raise ValueError("row reference shard is invalid")
        descriptor = self.shards[shard_index]
        if local_row not in range(descriptor.rows):
            raise ValueError("row reference offset is invalid")
        grouped.setdefault(shard_index, []).append((destination, local_row))

    selected: dict[str, np.ndarray] | None = None
    for shard_index, placements in grouped.items():
        descriptor = self.shards[shard_index]
        arrays = self._cache.get(
            self.root, descriptor, self.games[descriptor.game_id], self.contract,
        )
        destinations = np.asarray([item[0] for item in placements], dtype=np.int64)
        local_rows = np.asarray([item[1] for item in placements], dtype=np.int64)
        if selected is None:
            selected = {
                name: np.empty(
                    (len(refs), *values.shape[1:]), dtype=values.dtype,
                )
                for name, values in arrays.items()
            }
        for name in selected:
            selected[name][destinations] = arrays[name][local_rows]
    assert selected is not None
    return selected
```

Retain `_DecodedShardCache` and its two-entry policy for deferred physical validation outside the optimizer hot path.

- [ ] **Step 5: Implement the materialized partition**

Add immediately after `ImitationDataset`:

```python
@dataclass(frozen=True)
class MaterializedImitationPartition:
    partition: str
    batch: ImitationBatch
    offsets: Mapping[tuple[int, int], int]

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation"}:
            raise ValueError("materialized partition name is invalid")
        count = len(self.batch.actions)
        if count < 1 or len(self.offsets) != count:
            raise ValueError("materialized partition reference map is incomplete")
        if set(self.batch.partitions) != {self.partition}:
            raise ValueError("materialized partition metadata differs")
        if set(self.offsets.values()) != set(range(count)):
            raise ValueError("materialized partition offsets are invalid")
```

Refactor the canonical reference enumeration from `_partition_batch` into:

```python
def _partition_refs(dataset: ImitationDataset, partition: str) -> list[tuple[int, int]]:
    try:
        sources = dataset.index[partition]
    except KeyError as exc:
        raise ValueError(f"imitation dataset has no {partition} partition") from exc
    refs: list[tuple[int, int]] = []
    for source in sorted(sources, key=lambda value: value.value):
        for profile in sorted(sources[source]):
            for seat in sorted(sources[source][profile]):
                for kind in sorted(sources[source][profile][seat]):
                    refs.extend(sources[source][profile][seat][kind])
    if not refs or len(set(refs)) != len(refs):
        raise ValueError("partition references are empty or duplicated")
    return refs
```

Implement `materialize_imitation_partition` by calling `_partition_refs`, using the grouped `_row_data`, retaining the existing lexicographic `(game_id, decision_index)` order, and constructing the offset map after applying that same order. Do not create a second copy of the materialized arrays after the `ImitationBatch` is constructed.

- [ ] **Step 6: Implement exact reference scheduling and vectorized batch gathering**

Change the sampler constructor to require and validate the matching materialization. Extract the current lines that choose standard/conversion references and apply the seeded permutation into `_next_refs_and_sources` without changing their order or arithmetic.

Implement `next_batch` as:

```python
def next_batch(self) -> ImitationBatch:
    refs_and_sources = self._next_refs_and_sources()
    try:
        offsets = np.fromiter(
            (self.materialized.offsets[ref] for ref, _source in refs_and_sources),
            dtype=np.int64,
            count=len(refs_and_sources),
        )
    except KeyError as exc:
        raise RuntimeError("sampler selected a missing materialized reference") from exc
    batch = _take_batch(self.materialized.batch, offsets)
    scheduled_sources = np.asarray(
        [source for _ref, source in refs_and_sources], dtype=object,
    )
    if not np.array_equal(batch.sources, scheduled_sources):
        raise RuntimeError("materialized source metadata differs from scheduler")
    if set(batch.partitions) != {self.partition}:
        raise RuntimeError("materialized sampler crossed a partition")
    if not np.all(batch.legal_masks[np.arange(len(batch.actions)), batch.actions]):
        raise ValueError("selected teacher action is masked")
    return batch
```

- [ ] **Step 7: Wire the trainer and update existing sampler tests**

In `train_behavioral_clone`:

```python
training = materialize_imitation_partition(dataset, "train")
validation = materialize_imitation_partition(dataset, "validation")
fixtures = _fixture_batch(validation.batch)
sampler = StratifiedDecisionSampler(
    dataset,
    training,
    batch_size=config.batch_size,
    seed=config.model_seed,
    partition="train",
)
steps_per_epoch = max(
    1, int(np.ceil(len(training.batch.actions) / config.batch_size)),
)
```

Pass `validation.batch` to `_clone_metrics`. Update every test construction of `StratifiedDecisionSampler` to pass the matching materialized partition. Replace the obsolete test asserting that a first sampler batch decodes a physical shard with a test asserting that materialization performs the decode and subsequent batches do not.

- [ ] **Step 8: Run Task 1 gates**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "sampler or materialized or clone_rejects_any_validation"
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q
```

Expected: all selected tests and the complete imitation module pass with pristine output.

- [ ] **Step 9: Commit Task 1**

```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py
git diff --cached --check
git commit -m "Materialize imitation sampler batches"
```

### Task 2: Add Phase Timing and Physical History Validation

**Files:**
- Modify: `python/ml_lab/imitation.py:1034-1059, 1136-1192`
- Modify: `python/tests/test_imitation.py:693-757`
- Modify: `python/run_annihilation_imitation_panel.py:646-774`
- Modify: `python/tests/test_annihilation_imitation_panel.py:563-686`

**Interfaces:**
- Extends every `bc_epoch` event with the five exact phase fields from Global Constraints.
- Preserves `training-history.json` schema version 1 and adds the same fields to each epoch row.
- Extends `_validate_clone_run` to validate phase fields and their sum.

- [ ] **Step 1: Write failing trainer timing tests**

Extend `test_clone_trainer_emits_finite_epoch_and_completion_progress`:

```python
phase_fields = (
    "sampling_seconds",
    "transfer_forward_seconds",
    "optimization_seconds",
    "validation_seconds",
    "unclassified_seconds",
)
for key in phase_fields:
    assert math.isfinite(epoch[key])
    assert epoch[key] >= 0
assert math.isclose(
    sum(epoch[key] for key in phase_fields),
    epoch["epoch_seconds"],
    rel_tol=1e-9,
    abs_tol=1e-6,
)
```

Apply the same phase-field and sum assertions to
`test_behavioral_clone_real_cuda_training_publishes_cpu_artifact`. At the end of
that test, print one sorted JSON object containing `device`, `epoch_seconds`, and
the five phase fields. Normal pytest capture keeps the full suite quiet; the
real-hardware command in Task 3 uses `-s` to expose this evidence.

Add a unit test that passes an event with a missing phase, a negative phase, and an impossible sum to `_validate_behavioral_cloning_progress_event`, expecting `ValueError` for each mutation.

- [ ] **Step 2: Write failing physical-history tests**

Extend the valid clone fixture history with the five fields. Add:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("sampling_seconds"),
        lambda row: row.__setitem__("optimization_seconds", -1.0),
        lambda row: row.__setitem__("unclassified_seconds", 9.0),
    ],
)
def test_clone_validation_rejects_invalid_phase_timing(
    tmp_path: Path, mutation,
) -> None:
    run_dir, expected = _valid_clone_run(tmp_path)
    history_path = run_dir / "training-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    mutation(history["epochs"][0])
    history_path.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="timing"):
        _validate_fixture_run(run_dir, expected)
```

- [ ] **Step 3: Run timing tests and verify RED**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q -k "phase_timing or emits_finite_epoch"
```

Expected: failures for absent fields and absent physical validation.

- [ ] **Step 4: Add exact timing fields to the event validator**

Extend `_validate_behavioral_cloning_progress_event`'s exact required-field set. For `bc_epoch`, require each phase field to be a finite non-boolean number at least zero and require:

```python
if not math.isclose(
    sum(float(event[key]) for key in _BC_PHASE_FIELDS),
    float(event["epoch_seconds"]),
    rel_tol=1e-9,
    abs_tol=1e-6,
):
    raise ValueError("behavioral-cloning phase timing differs from epoch duration")
```

Define `_BC_PHASE_FIELDS` once near the event validator and reuse it in the trainer.

- [ ] **Step 5: Instrument the trainer without adding CUDA synchronization**

Initialize four phase accumulators at each epoch. Measure these exact boundaries:

```python
sampling_seconds = 0.0
transfer_forward_seconds = 0.0
optimization_seconds = 0.0
validation_seconds = 0.0
for _step in range(steps_per_epoch):
    phase_started = time.perf_counter()
    batch = sampler.next_batch()
    sampling_seconds += time.perf_counter() - phase_started
    if set(batch.partitions) != {"train"}:
        raise RuntimeError("validation rows entered behavioral-cloning optimization")

    phase_started = time.perf_counter()
    distribution, actions, _legal_masks = _distribution_tensors(model, batch)
    loss = -distribution.log_prob(actions).mean()
    losses.append(float(loss.detach().cpu()))
    transfer_forward_seconds += time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=1.0)
    optimizer.step()
    optimization_seconds += time.perf_counter() - phase_started

validation_started = time.perf_counter()
validation_metrics = _clone_metrics(model, validation.batch)
validation_seconds = time.perf_counter() - validation_started
```

After `epoch_elapsed`, compute:

```python
classified = (
    sampling_seconds + transfer_forward_seconds
    + optimization_seconds + validation_seconds
)
raw_unclassified = epoch_elapsed - classified
if raw_unclassified < -1e-6:
    raise RuntimeError("behavioral-cloning phase timing exceeds epoch duration")
unclassified_seconds = max(0.0, raw_unclassified)
```

Add all five values to the event. Do not call `torch.cuda.synchronize`.

- [ ] **Step 6: Extend physical retained-history validation**

In `_validate_clone_run`, require the five exact fields in every epoch row. Apply the same finite/nonnegative checks and `math.isclose` sum relationship used by the library validator. Do not change the existing NLL, examples/rate, patience, device, or publication checks.

- [ ] **Step 7: Run Task 2 gates**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q -k "phase_timing or training_history or emits_finite_epoch"
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q
```

Expected: all focused and combined tests pass.

- [ ] **Step 8: Commit Task 2**

```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py
git diff --cached --check
git commit -m "Report behavioral cloning phase timing"
```

### Task 3: Add the Production Benchmark and Restart Gate

**Files:**
- Modify: `python/ml_lab/imitation.py`
- Modify: `python/tests/test_imitation.py`
- Modify: `python/run_annihilation_imitation_panel.py:2572-2587, 2661-2687`
- Modify: `python/tests/test_annihilation_imitation_panel.py`
- Modify: `python/panels/annihilation-imitation-v1/PROTOCOL.md:43-85`

**Interfaces:**
- Produces: `benchmark_imitation_sampler(dataset, *, batch_size: int, seed: int, batches: int = 200) -> Mapping[str, Any]`.
- Produces: `_benchmark_bc_input_command(*, dataset_dir: Path = DATASET_PATH, execution_identity_path: Path = EXECUTION_IDENTITY_PATH, repository: Path = PROJECT_ROOT, repository_identity_provider: Callable[[Path], Mapping[str, Any]] | None = None) -> Mapping[str, Any]`.
- Adds CLI command: `benchmark-bc-input`.
- Locks: exactly 200 batches and minimum 2,825 examples per second.

- [ ] **Step 1: Write failing library benchmark tests**

```python
def test_sampler_benchmark_runs_exact_batches_and_reports_checksum(
    sampled_dataset: Path,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    result = benchmark_imitation_sampler(
        dataset, batch_size=100, seed=211, batches=4,
    )
    assert result["schema_version"] == 1
    assert result["batches"] == 4
    assert result["examples"] == 400
    assert result["examples_per_second"] > 0
    assert result["materialization_seconds"] >= 0
    assert result["sampling_seconds"] > 0
    assert len(result["sequence_sha256"]) == 64
```

The checksum must update from each batch's ordered `game_ids`, `decision_indices`, `actions`, and encoded `sources`, so it proves the benchmark consumes the deterministic sequence rather than optimizing the loop away.

- [ ] **Step 2: Write failing panel command and threshold tests**

Add a parser test requiring `build_parser().parse_args(["benchmark-bc-input"]).command == "benchmark-bc-input"`.

Test `_benchmark_bc_input_command` with injected benchmark and context providers:

```python
def test_benchmark_bc_input_locks_200_batches_and_threshold(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, "benchmark_imitation_sampler", lambda *args, **kwargs: calls.append(kwargs) or {
        "schema_version": 1,
        "batches": 200,
        "examples": 51200,
        "examples_per_second": 3000.0,
        "materialization_seconds": 1.0,
        "sampling_seconds": 17.0,
        "sequence_sha256": "a" * 64,
    })
    result = module._benchmark_bc_input_command(dataset_dir=tmp_path)
    assert calls == [{"batch_size": 256, "seed": 211, "batches": 200}]
    assert result["threshold_examples_per_second"] == 2825.0
    assert result["passed"] is True
```

Add a second test returning `2824.999` and require `ValueError("input benchmark throughput gate failed")`.

Give `_benchmark_bc_input_command` keyword-only `dataset_dir`,
`execution_identity_path`, `repository`, and `repository_identity_provider`
parameters with production defaults. Tests install call-recording fakes for
`_full_execution_context`, `_validate_dataset_execution_identity`,
`DuelClient`, `load_imitation_dataset`, and `benchmark_imitation_sampler`. They
must assert the identity validator received `dataset_dir` and the context's
identity before asserting the locked benchmark arguments.

- [ ] **Step 3: Run benchmark tests and verify RED**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q -k "sampler_benchmark or benchmark_bc_input"
```

Expected: failures for missing benchmark function and CLI command.

- [ ] **Step 4: Implement the read-only sampler benchmark**

Implement `benchmark_imitation_sampler` with strict integer validation for `batch_size`, `seed`, and `batches`. Materialize `train`, construct the exact sampler, start the sampling timer only after materialization, consume exactly `batches`, update SHA-256 with contiguous typed array bytes, and return an immutable mapping containing exactly:

```python
{
    "schema_version": 1,
    "batches": batches,
    "examples": batches * batch_size,
    "examples_per_second": examples / sampling_seconds,
    "materialization_seconds": materialization_seconds,
    "sampling_seconds": sampling_seconds,
    "sequence_sha256": digest.hexdigest(),
}
```

Reject a zero or non-finite elapsed duration instead of inventing throughput.

- [ ] **Step 5: Implement the identity-bound panel command**

Add constants:

```python
_BC_INPUT_BENCHMARK_BATCHES = 200
_BC_INPUT_MIN_EXAMPLES_PER_SECOND = 2825.0
```

`_benchmark_bc_input_command` must call `_full_execution_context` and then
`_validate_dataset_execution_identity(dataset_dir, identity)`. Create a
temporary `.benchmark-bc-runtime-` directory, materialize the immutable runtime
scenario there, start `DuelClient` from `_server_command`, capture its contract,
and close the client in `finally`. Remove only that verified temporary directory
in the outer `finally`, matching `_collect_command`. Then call
`load_imitation_dataset(dataset_dir, expected_contract=contract)`.

Invoke the benchmark as:

```python
result = benchmark_imitation_sampler(
    dataset,
    batch_size=panel["behavioral_cloning"]["batch_size"],
    seed=panel["model_seeds"][0],
    batches=_BC_INPUT_BENCHMARK_BATCHES,
)
```

Return the benchmark plus `threshold_examples_per_second` and `passed`. Raise `ValueError("input benchmark throughput gate failed")` before returning when throughput is below 2,825.

Add `commands.add_parser("benchmark-bc-input")` and dispatch it immediately before `train-bc` in `main`.

- [ ] **Step 6: Update protocol documentation**

Add the command to the documented sequence between `collect` and `train-bc`. Document exact-sequence materialization, five phase fields, the 200-batch/2,825 examples-per-second gate, diagnostic-only GPU utilization, the one-epoch CUDA proof, and mandatory recollection after the accepted code revision.

- [ ] **Step 7: Run all implementation gates**

Run independently:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests -q

dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo

dotnet build engine\HexWars.GymServer\HexWars.GymServer.csproj --nologo
```

Run `git diff --check`. Confirm the worktree contains only the five authorized tracked files and no generated artifact is staged.

- [ ] **Step 8: Commit Task 3**

```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py python/panels/annihilation-imitation-v1/PROTOCOL.md
git diff --cached --check
git commit -m "Gate behavioral cloning input throughput"
```

- [ ] **Step 9: Obtain per-task and final independent review**

Use the subagent-driven review loop after each task. After Task 3, generate one final review package from the pre-Task-1 base through `HEAD`. Require zero open Critical or Important findings before changing the current dataset or execution identity.

- [ ] **Step 10: Archive the superseded dataset only after review acceptance**

Verify the exact paths remain inside the isolated worktree. Move `python/datasets/annihilation-imitation-v1` into:

```text
python/panels/annihilation-imitation-v1/evidence/archive-task12-9e8bd19-low-gpu-interrupted/dataset
```

Do not delete or overwrite the already archived `bc-clones-staging`. Preserve the old process logs.

- [ ] **Step 11: Validate and recollect on the accepted clean commit**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' python/run_annihilation_imitation_panel.py validate
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' python/run_annihilation_imitation_panel.py collect
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' python/run_annihilation_imitation_panel.py collect
```

Require the first collection to publish a new clean identity-bound dataset and the second to return `reused: true` after physical validation.

- [ ] **Step 12: Run the real production performance gates**

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' python/run_annihilation_imitation_panel.py benchmark-bc-input
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -s -k "real_cuda_training"
```

Require benchmark `passed: true`, at least 2,825 examples per second, and a non-skipped real-CUDA test that publishes and validates a CPU controller.

- [ ] **Step 13: Restart full behavioral cloning only after every gate passes**

Launch `train-bc` with stdout/stderr redirected to ignored process logs. Verify the worker appears in the Windows per-process GPU counters. Monitor flushed `bc_epoch` events, not staging files. Continue to `evaluate-bc` only after all three clone runs publish and physically validate.
