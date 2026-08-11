from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from ml_lab.tactical_v3_client import TacticalV3GymClient
from ml_lab.tactical_v3_corpus import (
    TeacherEvidence,
    _select_tiny_candidate,
    create_tiny_corpus,
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
SERVER_DLL = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
FIXTURE = Path(__file__).parent / "fixtures" / "tactical_v3" / "tiny-corpus"


@pytest.fixture(scope="module")
def server_cmd() -> list[str]:
    assert SERVER_DLL.is_file(), "build HexWars.GymServer before running the corpus suite"
    return ["dotnet", str(SERVER_DLL), "--scenario-file", str(SCENARIO)]


@pytest.fixture(scope="module")
def expected_identity(server_cmd: list[str]):
    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        return client.identity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for row in rows))


def _reseal(root: Path) -> None:
    from ml_lab.tactical_v3_schema import canonical_sha256
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        data = path.read_bytes()
        item.update(byte_length=len(data), sha256=hashlib.sha256(data).hexdigest(), row_count=len(rows), seeds=sorted({row["episode_seed"] for row in rows}))
    unsigned = dict(manifest); unsigned.pop("identity", None)
    manifest["identity"] = canonical_sha256(unsigned)
    (root / "manifest.json").write_bytes((json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))


def _resealed_copy(tmp_path: Path) -> Path:
    copied = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE, copied)
    return copied


@pytest.mark.parametrize("mutation", ("profile", "seed", "seat", "three_rows", "candidate", "outcome"))
def test_loader_rejects_resealed_wrong_tiny_semantics(tmp_path: Path, expected_identity, mutation: str) -> None:
    copied = _resealed_copy(tmp_path)
    train_path = copied / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    if mutation == "profile": rows[0]["profile_id"] = "conversion-1v1-near"
    elif mutation == "seed":
        for row in rows[:4]: row["episode_seed"] = 4103
    elif mutation == "seat":
        for row in rows[:4]: row["learner_seat"] = row["decision"]["seat"] = 1
    elif mutation == "three_rows": rows.pop(3)
    elif mutation == "candidate": rows[0]["target"]["teacher_candidate_id"] = rows[0]["decision"]["candidates"][-1]["candidate_id"]
    elif mutation == "outcome":
        for row in rows[:4]: row["target"].update(terminal_outcome="loss", remaining_turns_to_victory=None)
    _write_canonical_jsonl(train_path, rows); _reseal(copied)
    with pytest.raises(ValueError, match="identity|seed|seat|profile|four|candidate"):
        load_corpus(copied, expected_identity)


def test_loader_rejects_resealed_duplicate_decision_across_partitions(tmp_path: Path, expected_identity, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml_lab.tactical_v3_corpus as module
    copied = _resealed_copy(tmp_path)
    train = [json.loads(line) for line in (copied / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    path = copied / "validation.jsonl"
    validation = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validation[0]["decision"] = train[0]["decision"]
    _write_canonical_jsonl(path, validation); _reseal(copied)
    manifest = json.loads((copied / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(module, "_EXPECTED_TINY_CORPUS_IDENTITY", manifest["identity"], raising=False)
    with pytest.raises(ValueError, match="duplicate decision"):
        load_corpus(copied, expected_identity)


def test_loader_rejects_duplicate_manifest_key_and_noncanonical_manifest(tmp_path: Path, expected_identity) -> None:
    copied = _resealed_copy(tmp_path)
    raw = (copied / "manifest.json").read_text(encoding="utf-8")
    (copied / "manifest.json").write_text(raw.replace('"label_source":', '"label_source":"wrong","label_source":', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(copied, expected_identity)
    copied = _resealed_copy(tmp_path / "second")
    path = copied / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical manifest"):
        load_corpus(copied, expected_identity)


@pytest.mark.parametrize("mutation", ("extra", "replace_partition"))
def test_loader_rejects_corpus_mutation_after_initial_snapshot(
    tmp_path: Path, expected_identity, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    import ml_lab.tactical_v3_corpus as module
    copied = _resealed_copy(tmp_path)
    original = module._validate_partitions
    def mutate(train, validation):
        original(train, validation)
        if mutation == "extra":
            (copied / "surprise.json").write_bytes(b"{}\n")
        else:
            (copied / "train.jsonl").write_bytes((copied / "validation.jsonl").read_bytes())
    monkeypatch.setattr(module, "_validate_partitions", mutate)
    with pytest.raises(ValueError, match="changed while"):
        load_corpus(copied, expected_identity)


def test_tiny_policy_selects_first_non_end_turn_or_the_only_end_turn() -> None:
    assert _select_tiny_candidate((
        SimpleNamespace(kind="end_turn", candidate_id=0),
        SimpleNamespace(kind="move", candidate_id=1),
        SimpleNamespace(kind="attack", candidate_id=2),
    )).candidate_id == 1
    assert _select_tiny_candidate((SimpleNamespace(kind="end_turn", candidate_id=7),)).candidate_id == 7
    with pytest.raises(ValueError, match="sole end_turn"):
        _select_tiny_candidate((
            SimpleNamespace(kind="end_turn", candidate_id=0),
            SimpleNamespace(kind="end_turn", candidate_id=1),
        ))


def test_tiny_corpus_is_exclusive_content_addressed_and_partitioned(
    tmp_path: Path, server_cmd: list[str], expected_identity,
) -> None:
    corpus = create_tiny_corpus(tmp_path / "corpus", server_cmd)

    assert 1 <= len(corpus.train) <= 8
    assert 1 <= len(corpus.validation) <= 4
    assert {row.episode_seed for row in corpus.train}.isdisjoint(
        row.episode_seed for row in corpus.validation
    )
    assert all(row.teacher == TeacherEvidence("tiny-fixture-policy-v1", 0, 0, 0, "none", None)
               for row in (*corpus.train, *corpus.validation))
    assert load_corpus(corpus.root, expected_identity) == corpus

    manifest = json.loads((corpus.root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "corpus_schema_version", "identity", "label_source", "scenario_id",
        "contract_hash", "encoding_hash", "capacity_hash", "environment_kind", "files",
    }
    assert len(manifest["identity"]) == 64
    with pytest.raises(FileExistsError):
        create_tiny_corpus(corpus.root, server_cmd)


def test_checked_in_tiny_corpus_is_immutable_canonical_and_reopens(
    expected_identity,
) -> None:
    before = {path.name: _sha256(path) for path in FIXTURE.iterdir() if path.is_file()}
    corpus = load_corpus(FIXTURE, expected_identity)
    after = {path.name: _sha256(path) for path in FIXTURE.iterdir() if path.is_file()}

    assert before == after
    assert corpus.root == FIXTURE
    assert corpus.train and corpus.validation
    for row in (*corpus.train, *corpus.validation):
        assert row.example_schema_version == 1
        assert row.target.teacher_candidate_id in {
            candidate.candidate_id for candidate in row.decision.candidates
        }
        assert row.target.remaining_turns_to_victory is None or row.target.terminal_outcome == "win"
        with pytest.raises(AttributeError):
            row.decision.candidates.append(row.decision.candidates[0])
    for path in (FIXTURE / "train.jsonl", FIXTURE / "validation.jsonl"):
        assert path.read_bytes().endswith(b"\n")
        assert all(
            json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"), allow_nan=False)
            == line
            for line in path.read_text(encoding="utf-8").splitlines()
        )


def test_loader_rejects_changed_row_bytes_even_when_json_is_valid(
    tmp_path: Path, expected_identity,
) -> None:
    copied = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE, copied)
    path = copied / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["target"]["teacher_candidate_id"] = (
        rows[0]["target"]["teacher_candidate_id"] + 1
    ) % len(rows[0]["decision"]["candidates"])
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_corpus(copied, expected_identity)


def test_loader_rejects_extra_inventory_and_cross_partition_seed(
    tmp_path: Path, expected_identity,
) -> None:
    copied = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE, copied)
    (copied / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        load_corpus(copied, expected_identity)

    (copied / "unexpected.json").unlink()
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = next(file for file in manifest["files"] if file["partition"] == "train")
    validation = next(file for file in manifest["files"] if file["partition"] == "validation")
    validation["seeds"] = [train["seeds"][0]]
    from ml_lab.tactical_v3_schema import canonical_sha256
    identity_input = dict(manifest)
    identity_input.pop("identity")
    manifest["identity"] = canonical_sha256(identity_input)
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="identity|seed"):
        load_corpus(copied, expected_identity)
