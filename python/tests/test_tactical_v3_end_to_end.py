from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from typing import Literal, TextIO

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from hexwars_gym.env import no_window_creationflags
from ml_lab.tactical_v3_batching import collate_decisions, collate_examples
from ml_lab.tactical_v3_checkpoint import (
    publish_structured_run,
    validate_structured_run,
)
from ml_lab.tactical_v3_client import CandidateSelection, TacticalV3GymClient
from ml_lab.tactical_v3_controller import (
    StructuredController,
    load_structured_controller,
)
from ml_lab.tactical_v3_corpus import StructuredCorpus, StructuredExample
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_schema import (
    TacticalV3SemanticIdentity,
    TacticalV3View,
    parse_spaces,
)
from ml_lab.tactical_v3_training import TrainerConfig, TrainingResult, train_offline
from tests.tactical_v3_fixture_support import load_tiny_corpus_fixture
from run_tactical_v3_imitation import _smoke_training_configs


ROOT = Path(__file__).resolve().parents[2]
CHECKED_IN_SCENARIO = (
    ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
)
DUEL_IDENTITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "tactical_v3" / "seed-41-duel-spaces.json"
)
SCENARIO_24X16 = (
    Path(__file__).parent / "fixtures" / "tactical_v3" / "scenario-24x16.json"
)
SERVER_DLL = (
    ROOT
    / "engine"
    / "HexWars.GymServer"
    / "bin"
    / "Debug"
    / "net8.0"
    / "HexWars.GymServer.dll"
)
_READ_TIMEOUT_SECONDS = 30
_EXIT_TIMEOUT_SECONDS = 30


@pytest.fixture(scope="session")
def server_dll() -> Path:
    assert SERVER_DLL.is_file(), "build HexWars.GymServer before running Task 13"
    return SERVER_DLL


@dataclass(frozen=True, slots=True)
class EndToEndCase:
    corpus: StructuredCorpus
    identity: TacticalV3SemanticIdentity
    first_result: TrainingResult
    second_result: TrainingResult
    first_run_dir: Path
    second_run_dir: Path
    first_controller: StructuredController
    second_controller: StructuredController


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        cpu = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(cpu.dtype), "shape": list(cpu.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = cpu.view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_examples(
    examples: tuple[StructuredExample, ...],
) -> tuple[StructuredExample, ...]:
    return tuple(sorted(examples, key=lambda example: (
        example.scenario_id,
        example.episode_seed,
        example.learner_seat,
        example.profile_id,
        example.decision.decision_id,
    )))


def overfit_metrics(
    controller: StructuredController,
    examples: tuple[StructuredExample, ...],
) -> Mapping[str, float]:
    ordered = _canonical_examples(examples)
    batch = collate_examples(ordered, controller.policy.config.horizon_turns)
    controller.policy.eval()
    with torch.inference_mode():
        logits = controller.policy(batch).candidate_logits
    target_nll = F.cross_entropy(logits, batch.teacher_candidate_index)
    predictions = logits.argmax(dim=1)
    return {
        "policy_nll": float(target_nll.item()),
        "policy_accuracy": float(
            (predictions == batch.teacher_candidate_index).to(torch.float64).mean().item()
        ),
        "finite_logit_count": float(torch.isfinite(logits[batch.candidates.mask]).sum().item()),
    }


def total_valid_candidates(examples: tuple[StructuredExample, ...]) -> int:
    return sum(len(example.decision.candidates) for example in examples)


def gymserver_command(
    server_dll: Path,
    scenario: Path,
    role: Literal["tactical", "duel"],
) -> tuple[str, ...]:
    if role not in {"tactical", "duel"}:
        raise ValueError("role must be tactical or duel")
    # TacticalV3GymClient appends the environment. GymServer's current scenario
    # switch is --scenario-file; role selects the client's JSONL command family.
    return "dotnet", str(server_dll), "--scenario-file", str(scenario)


def _bounded_readline(process: subprocess.Popen[str], stream: TextIO) -> str:
    lines: Queue[str] = Queue(maxsize=1)
    reader = Thread(target=lambda: lines.put(stream.readline()), daemon=True)
    reader.start()
    try:
        return lines.get(timeout=_READ_TIMEOUT_SECONDS)
    except Empty as error:
        process.kill()
        process.wait(timeout=_EXIT_TIMEOUT_SECONDS)
        raise AssertionError("policy server JSONL read timed out") from error


class _PolicyServerProcess(AbstractContextManager["_PolicyServerProcess"]):
    def __init__(self, args: Sequence[str]) -> None:
        self.process = subprocess.Popen(
            tuple(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            creationflags=no_window_creationflags(),
        )
        assert self.process.stdout is not None
        ready_line = _bounded_readline(self.process, self.process.stdout)
        if not ready_line:
            _stdout, stderr = self.process.communicate(timeout=_EXIT_TIMEOUT_SECONDS)
            raise AssertionError(f"policy server exited before ready: {stderr[:4096]}")
        ready = json.loads(ready_line)
        assert type(ready) is dict and ready.get("ready") is True

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        self.process.stdin.flush()
        response = json.loads(_bounded_readline(self.process, self.process.stdout))
        assert type(response) is dict
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.write('{"cmd":"close"}\n')
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=_EXIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=_EXIT_TIMEOUT_SECONDS)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


def _view_wire(view: TacticalV3View) -> dict[str, object]:
    value = asdict(view)
    decision = value.pop("decision")
    assert type(decision) is dict
    return {**decision, **value}


def round_trip_via_policy_server(
    client: TacticalV3GymClient,
    run_dir: Path,
    identity: TacticalV3SemanticIdentity,
    seed: int,
) -> tuple[CandidateIdentity, TacticalV3View, TacticalV3View]:
    before = client.reset(seed)
    args = (
        sys.executable,
        str(ROOT / "python" / "policy_server.py"),
        "--p0",
        f"run:{run_dir}",
        "--expected-environment",
        "tactical-v3",
        "--expected-contract-version",
        "tactical-v3",
        "--expected-encoding-hash",
        identity.encoding_hash,
        "--expected-capacity-hash",
        identity.capacity_hash,
    )
    with _PolicyServerProcess(args) as server:
        response = server.request({"seat": before.seat, "decision": _view_wire(before)})
    assert set(response) == {"decision_id", "candidate_id"}
    assert type(response["decision_id"]) is int
    assert type(response["candidate_id"]) is int
    selected = CandidateIdentity(response["decision_id"], response["candidate_id"])
    after = client.step(CandidateSelection(selected.decision_id, selected.candidate_id))
    return selected, before, after


def controller_logits_and_selection(
    controller: StructuredController,
    view: TacticalV3View,
) -> tuple[Tensor, CandidateIdentity]:
    batch = collate_decisions((view.decision,), controller.policy.config.horizon_turns)
    with torch.inference_mode():
        output = controller.policy(batch)
        selection, = controller.policy.select(batch)
    return output.candidate_logits[0, batch.candidates.mask[0]].cpu(), selection


def controller_fixture_outputs(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
) -> tuple[tuple[Tensor, ...], tuple[CandidateIdentity, ...]]:
    ordered = _canonical_examples(examples)
    batch = collate_examples(ordered, model.config.horizon_turns)
    with torch.inference_mode():
        output = model(batch)
        actions = model.select(batch)
    logits = tuple(
        output.candidate_logits[index, batch.candidates.mask[index]].cpu()
        for index in range(len(ordered))
    )
    return logits, actions


def make_end_to_end_case(tmp_path: Path, server_dll: Path) -> EndToEndCase:
    assert server_dll.is_file()
    corpus = load_tiny_corpus_fixture()
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(4, 8, 16),
    )
    objective_config = ObjectiveConfig(
        policy_coefficient=1.0,
        outcome_coefficient=0.0,
        horizon_coefficient=0.0,
        remaining_turns_coefficient=0.0,
    )
    trainer_config = TrainerConfig(
        seed=227,
        batch_size=8,
        learning_rate=0.005,
        max_epochs=80,
        patience_epochs=80,
        gradient_clip_norm=1.0,
        device="cpu",
    )
    first_result = train_offline(
        corpus.train, corpus.validation, model_config, objective_config, trainer_config
    )
    second_result = train_offline(
        corpus.train, corpus.validation, model_config, objective_config, trainer_config
    )
    policy_identity = parse_spaces(json.loads(
        DUEL_IDENTITY_FIXTURE.read_text(encoding="utf-8")
    ))
    first_run_dir = publish_structured_run(
        tmp_path / "first-run",
        first_result,
        corpus,
        training_scenario_path=CHECKED_IN_SCENARIO,
        policy_identity=policy_identity,
    )
    second_run_dir = publish_structured_run(
        tmp_path / "second-run",
        second_result,
        corpus,
        training_scenario_path=CHECKED_IN_SCENARIO,
        policy_identity=policy_identity,
    )
    first_loaded = validate_structured_run(first_run_dir)
    second_loaded = validate_structured_run(second_run_dir)
    semantic_identity = first_loaded.metadata.identity
    assert second_loaded.metadata.identity == semantic_identity
    first_controller = load_structured_controller(
        first_run_dir, semantic_identity.encoding_hash, semantic_identity.capacity_hash
    )
    second_controller = load_structured_controller(
        second_run_dir, semantic_identity.encoding_hash, semantic_identity.capacity_hash
    )
    return EndToEndCase(
        corpus,
        semantic_identity,
        first_result,
        second_result,
        first_run_dir,
        second_run_dir,
        first_controller,
        second_controller,
    )


@pytest.fixture(scope='module')
def end_to_end_case(
    tmp_path_factory: pytest.TempPathFactory,
    server_dll: Path,
) -> EndToEndCase:
    return make_end_to_end_case(tmp_path_factory.mktemp('task-13'), server_dll)


def test_tiny_corpus_overfits_deterministically(
    end_to_end_case: EndToEndCase,
) -> None:
    case = end_to_end_case
    expected_configs = _smoke_training_configs(seed=227, device='cpu')
    assert expected_configs == (
        case.first_result.model_config,
        case.first_result.objective_config,
        case.first_result.trainer_config,
    )
    assert expected_configs == (
        case.second_result.model_config,
        case.second_result.objective_config,
        case.second_result.trainer_config,
    )
    assert case.first_result.history == case.second_result.history
    assert state_dict_sha256(case.first_result.model.state_dict()) == state_dict_sha256(
        case.second_result.model.state_dict()
    )
    for examples in (case.corpus.train, case.corpus.validation):
        first = overfit_metrics(case.first_controller, examples)
        second = overfit_metrics(case.second_controller, examples)
        assert first == second
        assert first["policy_accuracy"] == 1.0
        assert first["policy_nll"] < 0.02
        assert first["finite_logit_count"] == total_valid_candidates(examples)


def test_13x9_gymserver_policy_server_candidate_identity_round_trip(
    end_to_end_case: EndToEndCase, server_dll: Path,
) -> None:
    case = end_to_end_case
    command = gymserver_command(server_dll, CHECKED_IN_SCENARIO, role="tactical")
    with TacticalV3GymClient(command, environment_kind="tactical") as client:
        selection, before, after = round_trip_via_policy_server(
            client, case.first_run_dir, case.identity, seed=41
        )
    matches = [
        candidate
        for candidate in before.decision.candidates
        if candidate.candidate_id == selection.candidate_id
    ]
    assert len(matches) == 1
    assert selection.decision_id == before.decision.decision_id
    assert after.terminated or after.truncated or (
        after.decision.decision_id != before.decision.decision_id
    )


def test_same_checkpoint_infers_legally_on_24x16_without_rebuild(
    end_to_end_case: EndToEndCase, server_dll: Path,
) -> None:
    case = end_to_end_case
    before_parameter_count = sum(
        parameter.numel() for parameter in case.first_controller.policy.parameters()
    )
    command = gymserver_command(server_dll, SCENARIO_24X16, role="tactical")
    with TacticalV3GymClient(command, environment_kind="tactical") as client:
        view = client.reset(41)
        assert client.identity.encoding_hash == case.identity.encoding_hash
        assert client.identity.capacity_hash == case.identity.capacity_hash
        assert client.identity.contract_hash != case.identity.contract_hash
        logits, selection = controller_logits_and_selection(case.first_controller, view)
    assert torch.isfinite(logits).all()
    assert selection.candidate_id in {
        candidate.candidate_id for candidate in view.decision.candidates
    }
    assert sum(
        parameter.numel() for parameter in case.first_controller.policy.parameters()
    ) == before_parameter_count
    assert case.first_controller.policy.config == case.second_controller.policy.config


def test_two_publications_reload_with_identical_logits_and_actions(
    end_to_end_case: EndToEndCase,
) -> None:
    case = end_to_end_case
    first = validate_structured_run(case.first_run_dir)
    second = validate_structured_run(case.second_run_dir)
    assert first.metadata.model_state_sha256 == second.metadata.model_state_sha256
    for examples in (case.corpus.train, case.corpus.validation):
        first_logits, first_actions = controller_fixture_outputs(first.model, examples)
        second_logits, second_actions = controller_fixture_outputs(second.model, examples)
        for left, right in zip(first_logits, second_logits, strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        assert first_actions == second_actions
