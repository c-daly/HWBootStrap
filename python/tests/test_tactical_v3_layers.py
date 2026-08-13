from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from ml_lab.tactical_v3_batching import (
    RELATION_KIND_COUNT,
    RELATION_KIND_IDS,
    TABLE_CATEGORICAL_CARDINALITIES,
    TABLE_ORDER,
    RaggedBatch,
    RelationNeighborhoodBatch,
    TokenTableBatch,
    collate_examples,
)
from ml_lab.tactical_v3_client import TacticalV3GymClient
from ml_lab.tactical_v3_corpus import StructuredExample, load_corpus
from ml_lab.tactical_v3_layers import (
    LocalHexMessagePassing,
    TacticalV3ModelConfig,
    TypedRelationalAttention,
    TypedTokenEncoders,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tactical_v3"
CORPUS_ROOT = FIXTURE_ROOT / "tiny-corpus"
CHECKED_IN_SCENARIO = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
SERVER_DLL = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
HORIZONS = (4, 8, 16)


@dataclass(frozen=True, slots=True)
class LayerTestCase:
    batch: RaggedBatch
    token_encoders: TypedTokenEncoders
    local_hex: LocalHexMessagePassing
    relational: TypedRelationalAttention


def canonical_example() -> StructuredExample:
    with TacticalV3GymClient(
        ["dotnet", str(SERVER_DLL), "--scenario-file", str(CHECKED_IN_SCENARIO)],
        environment_kind="duel",
    ) as client:
        identity = client.identity
    return load_corpus(CORPUS_ROOT, identity).train[0]


@pytest.fixture(scope="module")
def example_13x9() -> StructuredExample:
    return canonical_example()


def translate_cell_coordinates(
    example: StructuredExample, dq: int, dr: int
) -> StructuredExample:
    cells = tuple(
        replace(cell, q=cell.q + dq, r=cell.r + dr)
        for cell in example.decision.observation.cells
    )
    observation = replace(example.decision.observation, cells=cells)
    return replace(example, decision=replace(example.decision, observation=observation))


def make_layer_case(example: StructuredExample, seed: int) -> LayerTestCase:
    torch.manual_seed(seed)
    config = TacticalV3ModelConfig()
    return LayerTestCase(
        batch=collate_examples((example,), horizons=config.horizon_turns),
        token_encoders=TypedTokenEncoders(config).eval(),
        local_hex=LocalHexMessagePassing(config).eval(),
        relational=TypedRelationalAttention(config).eval(),
    )


def _remap_indices(indices: torch.Tensor, old_to_new: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp(min=0, max=old_to_new.numel() - 1)
    return old_to_new[safe]


def permute_table_and_remap(
    batch: RaggedBatch, table_name: str, seed: int
) -> tuple[RaggedBatch, torch.Tensor]:
    """Permute one table and every global reference to it in a single-sample batch."""
    assert batch.node_mask.shape[0] == 1
    table_slice = batch.table_slices[table_name]
    row_count = table_slice.stop - table_slice.start
    permutation = torch.randperm(
        row_count, generator=torch.Generator().manual_seed(seed)
    )
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(row_count)

    node_count = batch.node_mask.shape[1]
    old_to_new = torch.arange(node_count)
    old_to_new[table_slice] = table_slice.start + inverse
    new_order = torch.arange(node_count)
    new_order[table_slice] = table_slice.start + permutation

    tables = dict(batch.tables)
    table = tables[table_name]
    tables[table_name] = TokenTableBatch(
        numeric=table.numeric.index_select(1, permutation),
        categorical={
            name: value.index_select(1, permutation)
            for name, value in table.categorical.items()
        },
        boolean={
            name: value.index_select(1, permutation)
            for name, value in table.boolean.items()
        },
        mask=table.mask.index_select(1, permutation),
    )

    cell_index = _remap_indices(batch.cell_neighbor_index, old_to_new)
    cell_mask = batch.cell_neighbor_mask
    if table_name == "cells":
        cell_index = cell_index.index_select(1, permutation)
        cell_mask = cell_mask.index_select(1, permutation)

    neighborhoods = batch.neighborhoods
    remapped_neighborhoods = RelationNeighborhoodBatch(
        source_index=_remap_indices(neighborhoods.source_index, old_to_new).index_select(
            1, new_order
        ),
        kind=neighborhoods.kind.index_select(1, new_order),
        int_feature=neighborhoods.int_feature.index_select(1, new_order),
        float_feature=neighborhoods.float_feature.index_select(1, new_order),
        bool_feature=neighborhoods.bool_feature.index_select(1, new_order),
        mask=neighborhoods.mask.index_select(1, new_order),
    )
    candidates = replace(
        batch.candidates,
        reference_index=_remap_indices(batch.candidates.reference_index, old_to_new),
    )
    return (
        replace(
            batch,
            tables=tables,
            node_mask=batch.node_mask.index_select(1, new_order),
            cell_neighbor_index=cell_index,
            cell_neighbor_mask=cell_mask,
            neighborhoods=remapped_neighborhoods,
            candidates=candidates,
        ),
        inverse,
    )


def undo_table_rows(
    state: torch.Tensor, table_slice: slice, inverse: torch.Tensor
) -> torch.Tensor:
    restored = state[:, table_slice, :].index_select(1, inverse)
    return torch.cat(
        (state[:, : table_slice.start, :], restored, state[:, table_slice.stop :, :]),
        dim=1,
    )


def run_local_stack(case: LayerTestCase, batch: RaggedBatch) -> torch.Tensor:
    node_state = case.token_encoders(batch)
    return case.local_hex(
        node_state,
        batch.cell_neighbor_index,
        batch.cell_neighbor_mask,
        batch.node_mask,
        batch.table_slices["cells"],
    )


def run_encoder_stack(case: LayerTestCase, batch: RaggedBatch) -> torch.Tensor:
    local_state = run_local_stack(case, batch)
    edges = batch.neighborhoods
    return case.relational(
        local_state,
        edges.source_index,
        edges.kind,
        edges.int_feature,
        edges.float_feature,
        edges.bool_feature,
        edges.mask,
        batch.node_mask,
    )


def append_masked_padding(batch: RaggedBatch, fill: float) -> RaggedBatch:
    """Append one false-masked relation row, which is the final canonical table."""
    relation_table = batch.tables["relations"]
    batch_size = batch.node_mask.shape[0]
    tables = dict(batch.tables)
    tables["relations"] = TokenTableBatch(
        numeric=torch.cat(
            (
                relation_table.numeric,
                torch.full(
                    (batch_size, 1, relation_table.numeric.shape[2]),
                    fill,
                    dtype=torch.float32,
                ),
            ),
            dim=1,
        ),
        categorical={
            name: torch.cat((value, torch.zeros((batch_size, 1), dtype=torch.int64)), dim=1)
            for name, value in relation_table.categorical.items()
        },
        boolean={
            name: torch.cat((value, torch.ones((batch_size, 1), dtype=torch.bool)), dim=1)
            for name, value in relation_table.boolean.items()
        },
        mask=torch.cat(
            (relation_table.mask, torch.zeros((batch_size, 1), dtype=torch.bool)), dim=1
        ),
    )
    slices = dict(batch.table_slices)
    relation_slice = slices["relations"]
    slices["relations"] = slice(relation_slice.start, relation_slice.stop + 1)

    edges = batch.neighborhoods
    slots = edges.mask.shape[2]

    def append_int(value: torch.Tensor, payload: int) -> torch.Tensor:
        return torch.cat(
            (value, torch.full((batch_size, 1, slots), payload, dtype=value.dtype)), dim=1
        )

    padded_edges = RelationNeighborhoodBatch(
        source_index=append_int(edges.source_index, 1_000_000),
        kind=append_int(edges.kind, 1_000_000),
        int_feature=append_int(edges.int_feature, 1_000_000),
        float_feature=torch.cat(
            (edges.float_feature, torch.full((batch_size, 1, slots), fill)), dim=1
        ),
        bool_feature=torch.cat(
            (edges.bool_feature, torch.ones((batch_size, 1, slots), dtype=torch.bool)),
            dim=1,
        ),
        mask=torch.cat(
            (edges.mask, torch.zeros((batch_size, 1, slots), dtype=torch.bool)), dim=1
        ),
    )
    return replace(
        batch,
        tables=tables,
        table_slices=slices,
        node_mask=torch.cat(
            (batch.node_mask, torch.zeros((batch_size, 1), dtype=torch.bool)), dim=1
        ),
        neighborhoods=padded_edges,
    )


def insert_masked_cell_padding(batch: RaggedBatch, fill: float) -> RaggedBatch:
    """Insert a false cell row and remap every shifted global reference."""
    cells_slice = batch.table_slices["cells"]
    insert_at = cells_slice.stop
    batch_size, node_count = batch.node_mask.shape
    cells = batch.tables["cells"]
    tables = dict(batch.tables)
    tables["cells"] = TokenTableBatch(
        numeric=torch.cat(
            (
                cells.numeric,
                torch.full(
                    (batch_size, 1, cells.numeric.shape[2]),
                    fill,
                    dtype=cells.numeric.dtype,
                ),
            ),
            dim=1,
        ),
        categorical={
            name: torch.cat(
                (value, torch.zeros((batch_size, 1), dtype=value.dtype)), dim=1
            )
            for name, value in cells.categorical.items()
        },
        boolean={
            name: torch.cat(
                (value, torch.ones((batch_size, 1), dtype=value.dtype)), dim=1
            )
            for name, value in cells.boolean.items()
        },
        mask=torch.cat(
            (cells.mask, torch.zeros((batch_size, 1), dtype=torch.bool)), dim=1
        ),
    )

    old_index = torch.arange(node_count)
    old_to_new = old_index + (old_index >= insert_at).to(torch.int64)
    slices = {
        table_name: (
            slice(table_slice.start, table_slice.stop + 1)
            if table_name == "cells"
            else slice(table_slice.start + 1, table_slice.stop + 1)
        )
        for table_name, table_slice in batch.table_slices.items()
    }
    cell_index = _remap_indices(batch.cell_neighbor_index, old_to_new)
    cell_index = torch.cat(
        (
            cell_index,
            torch.full(
                (batch_size, 1, cell_index.shape[2]),
                1_000_000,
                dtype=torch.int64,
            ),
        ),
        dim=1,
    )
    cell_mask = torch.cat(
        (
            batch.cell_neighbor_mask,
            torch.zeros(
                (batch_size, 1, batch.cell_neighbor_mask.shape[2]),
                dtype=torch.bool,
            ),
        ),
        dim=1,
    )

    edges = batch.neighborhoods
    slots = edges.mask.shape[2]

    def insert_destination(value: torch.Tensor, payload: int | float | bool) -> torch.Tensor:
        inserted = torch.full(
            (batch_size, 1, slots), payload, dtype=value.dtype
        )
        return torch.cat(
            (value[:, :insert_at], inserted, value[:, insert_at:]), dim=1
        )

    neighborhoods = RelationNeighborhoodBatch(
        source_index=insert_destination(
            _remap_indices(edges.source_index, old_to_new), 1_000_000
        ),
        kind=insert_destination(edges.kind, 1_000_000),
        int_feature=insert_destination(edges.int_feature, 1_000_000),
        float_feature=insert_destination(edges.float_feature, fill),
        bool_feature=insert_destination(edges.bool_feature, True),
        mask=insert_destination(edges.mask, False),
    )
    candidates = replace(
        batch.candidates,
        reference_index=_remap_indices(batch.candidates.reference_index, old_to_new),
    )
    return replace(
        batch,
        tables=tables,
        table_slices=slices,
        node_mask=torch.cat(
            (
                batch.node_mask[:, :insert_at],
                torch.zeros((batch_size, 1), dtype=torch.bool),
                batch.node_mask[:, insert_at:],
            ),
            dim=1,
        ),
        cell_neighbor_index=cell_index,
        cell_neighbor_mask=cell_mask,
        neighborhoods=neighborhoods,
        candidates=candidates,
    )


def remove_inserted_node(state: torch.Tensor, index: int) -> torch.Tensor:
    return torch.cat((state[:, :index], state[:, index + 1 :]), dim=1)



def replace_masked_neighbor_payload(batch: RaggedBatch, fill: float) -> RaggedBatch:
    local_mask = batch.cell_neighbor_mask
    local_index = torch.where(
        local_mask,
        batch.cell_neighbor_index,
        torch.full_like(batch.cell_neighbor_index, 1_000_000),
    )
    edges = batch.neighborhoods
    edge_mask = edges.mask
    mutated_edges = RelationNeighborhoodBatch(
        source_index=torch.where(
            edge_mask, edges.source_index, torch.full_like(edges.source_index, 1_000_000)
        ),
        kind=torch.where(edge_mask, edges.kind, torch.full_like(edges.kind, 1_000_000)),
        int_feature=torch.where(
            edge_mask, edges.int_feature, torch.full_like(edges.int_feature, 1_000_000)
        ),
        float_feature=torch.where(
            edge_mask, edges.float_feature, torch.full_like(edges.float_feature, fill)
        ),
        bool_feature=torch.where(edge_mask, edges.bool_feature, torch.ones_like(edges.bool_feature)),
        mask=edge_mask,
    )
    return replace(batch, cell_neighbor_index=local_index, neighborhoods=mutated_edges)


def replace_first_valid_numeric(batch: RaggedBatch, value: float) -> RaggedBatch:
    cells = batch.tables["cells"]
    numeric = cells.numeric.clone()
    first = cells.mask.nonzero(as_tuple=False)[0]
    numeric[first[0], first[1], 0] = value
    tables = dict(batch.tables)
    tables["cells"] = replace(cells, numeric=numeric)
    return replace(batch, tables=tables)


def test_model_config_is_frozen_and_rejects_invalid_architectures() -> None:
    config = TacticalV3ModelConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.hidden_dim = 32  # type: ignore[misc]

    invalid = (
        {"hidden_dim": 0},
        {"categorical_dim": -1},
        {"cell_message_rounds": 0},
        {"relation_rounds": 0},
        {"attention_heads": 0},
        {"hidden_dim": 63, "attention_heads": 4},
        {"feed_forward_dim": 0},
        {"candidate_hidden_dim": 0},
        {"horizon_turns": ()},
        {"horizon_turns": (4, 4)},
        {"horizon_turns": (8, 4)},
        {"horizon_turns": (0, 4)},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            TacticalV3ModelConfig(**kwargs)


def test_typed_encoders_own_distinct_table_and_field_parameters(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=11)
    encoded = case.token_encoders(case.batch)
    assert encoded.shape == (*case.batch.node_mask.shape, 64)
    assert encoded.dtype == torch.float32
    assert {parameter.dtype for parameter in case.token_encoders.parameters()} == {
        torch.float32
    }
    assert tuple(case.token_encoders.numeric_projections) == tuple(
        table for table in TABLE_ORDER if case.batch.tables[table].numeric.shape[2]
    )
    assert len({id(layer) for layer in case.token_encoders.numeric_projections.values()}) == 7
    expected_categorical = sum(
        len(fields) for fields in TABLE_CATEGORICAL_CARDINALITIES.values()
    )
    assert len(case.token_encoders.categorical_embeddings) == expected_categorical
    assert len(case.token_encoders.boolean_embeddings) == 13
    assert all(
        embedding.num_embeddings == 2
        for embedding in case.token_encoders.boolean_embeddings.values()
    )


def test_centered_coordinates_make_encoder_translation_invariant(
    example_13x9: StructuredExample,
) -> None:
    shifted = translate_cell_coordinates(example_13x9, dq=17, dr=-9)
    left = collate_examples((example_13x9,), horizons=HORIZONS)
    right = collate_examples((shifted,), horizons=HORIZONS)
    torch.testing.assert_close(
        left.tables["cells"].numeric[..., :2],
        right.tables["cells"].numeric[..., :2],
        rtol=0.0,
        atol=0.0,
    )
    torch.manual_seed(13)
    encoder = TypedTokenEncoders(TacticalV3ModelConfig()).eval()
    torch.testing.assert_close(encoder(left), encoder(right), rtol=0.0, atol=0.0)


def test_local_hex_reads_batch_neighbors_and_writes_only_cells(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=17)
    encoded = case.token_encoders(case.batch)
    actual = run_local_stack(case, case.batch)
    non_cell_mask = case.batch.node_mask.clone()
    non_cell_mask[:, case.batch.table_slices["cells"]] = False
    torch.testing.assert_close(actual[non_cell_mask], encoded[non_cell_mask], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(actual[~case.batch.node_mask]) == 0


def test_local_hex_is_equivariant_to_cell_row_permutation(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=19)
    permuted, inverse = permute_table_and_remap(case.batch, "cells", seed=23)
    actual = undo_table_rows(
        run_local_stack(case, permuted), permuted.table_slices["cells"], inverse
    )
    torch.testing.assert_close(
        actual[case.batch.node_mask],
        run_local_stack(case, case.batch)[case.batch.node_mask],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "table_name",
    ("units", "templates", "capability_definitions", "capability_allocations"),
)
def test_relational_layer_is_equivariant_to_typed_table_row_permutations(
    example_13x9: StructuredExample, table_name: str
) -> None:
    case = make_layer_case(example_13x9, seed=29)
    permuted, inverse = permute_table_and_remap(case.batch, table_name, seed=31)
    actual = undo_table_rows(
        run_encoder_stack(case, permuted), permuted.table_slices[table_name], inverse
    )
    expected = run_encoder_stack(case, case.batch)
    torch.testing.assert_close(
        actual[case.batch.node_mask], expected[case.batch.node_mask], rtol=0.0, atol=1e-6
    )


def test_padding_and_masked_payloads_cannot_change_valid_embeddings(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=37)
    expected = run_encoder_stack(case, case.batch)

    padded = append_masked_padding(case.batch, fill=1_000_000.0)
    padded_actual = run_encoder_stack(case, padded)
    torch.testing.assert_close(
        padded_actual[:, : expected.shape[1]][case.batch.node_mask],
        expected[case.batch.node_mask],
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.count_nonzero(padded_actual[~padded.node_mask]) == 0

    cell_padded = insert_masked_cell_padding(case.batch, fill=1_000_000.0)
    cell_padding_index = case.batch.table_slices["cells"].stop
    cell_padded_actual = run_encoder_stack(case, cell_padded)
    restored = remove_inserted_node(cell_padded_actual, cell_padding_index)
    torch.testing.assert_close(
        restored[case.batch.node_mask],
        expected[case.batch.node_mask],
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.count_nonzero(
        cell_padded_actual[:, cell_padding_index, :]
    ) == 0

    mutated = replace_masked_neighbor_payload(case.batch, fill=1_000_000.0)
    mutated_actual = run_encoder_stack(case, mutated)
    torch.testing.assert_close(
        mutated_actual[case.batch.node_mask],
        expected[case.batch.node_mask],
        rtol=0.0,
        atol=1e-6,
    )




def test_masked_numeric_float32_max_is_zeroed_before_projection(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=59)
    padded = insert_masked_cell_padding(
        case.batch, fill=torch.finfo(torch.float32).max
    )
    actual = case.token_encoders(padded)
    padding_index = case.batch.table_slices["cells"].stop
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual[:, padding_index, :],
        torch.zeros_like(actual[:, padding_index, :]),
        rtol=0.0,
        atol=0.0,
    )


def test_local_hex_rejects_non_cell_sources_and_noncanonical_slice(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=61)
    encoded = case.token_encoders(case.batch)
    bad_source = case.batch.cell_neighbor_index.clone()
    first = case.batch.cell_neighbor_mask.nonzero(as_tuple=False)[0]
    bad_source[first[0], first[1], first[2]] = case.batch.table_slices["units"].start
    with pytest.raises(ValueError, match="cell_neighbor_index.*cells slice"):
        case.local_hex(
            encoded,
            bad_source,
            case.batch.cell_neighbor_mask,
            case.batch.node_mask,
            case.batch.table_slices["cells"],
        )

    cells = case.batch.table_slices["cells"]
    with pytest.raises(ValueError, match="cells_slice.*start.*zero|canonical"):
        case.local_hex(
            encoded,
            case.batch.cell_neighbor_index,
            case.batch.cell_neighbor_mask,
            case.batch.node_mask,
            slice(1, cells.stop + 1),
        )


def _tiny_config() -> TacticalV3ModelConfig:
    return TacticalV3ModelConfig(
        hidden_dim=4,
        categorical_dim=1,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=1,
        feed_forward_dim=4,
        candidate_hidden_dim=4,
        horizon_turns=(1,),
    )


def _zero_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_local_one_edge_source_selection_changes_only_its_destination() -> None:
    layer = LocalHexMessagePassing(_tiny_config()).eval()
    _zero_parameters(layer)
    with torch.no_grad():
        layer.message_mlp[0].weight[:, 4:8].copy_(torch.eye(4))
        layer.message_mlp[2].weight.copy_(torch.eye(4))

    state = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [3.0, 0.0, 0.0, 0.0],
          [7.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    node_mask = torch.ones((1, 4), dtype=torch.bool)
    edge_mask = torch.zeros((1, 3, 6), dtype=torch.bool)
    edge_mask[0, 1, 0] = True
    from_first = torch.zeros((1, 3, 6), dtype=torch.int64)
    from_third = from_first.clone()
    from_third[0, 1, 0] = 2

    first = layer(state, from_first, edge_mask, node_mask, slice(0, 3))
    third = layer(state, from_third, edge_mask, node_mask, slice(0, 3))
    torch.testing.assert_close(first[:, (0, 2, 3)], state[:, (0, 2, 3)])
    torch.testing.assert_close(third[:, (0, 2, 3)], state[:, (0, 2, 3)])
    torch.testing.assert_close(
        first[:, 1], torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        third[:, 1], torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    )


def _controlled_relational() -> TypedRelationalAttention:
    layer = TypedRelationalAttention(_tiny_config()).eval()
    _zero_parameters(layer)
    block = layer.blocks[0]
    with torch.no_grad():
        block.value.weight.copy_(torch.eye(4))
        block.output.weight.copy_(torch.eye(4))
        layer.relation_kind_embedding.weight[
            RELATION_KIND_IDS["occupies"], 0
        ] = 1.0
        layer.relation_kind_embedding.weight[
            RELATION_KIND_IDS["occupies_reverse"], 0
        ] = 2.0
        layer.integer_projection.weight[0, 0] = 1.0
        layer.float_projection.weight[0, 0] = 1.0
        layer.boolean_embedding.weight[1, 0] = 1.0
        layer.edge_projection.weight.copy_(torch.eye(4))
    return layer


def _run_controlled_relation(
    layer: TypedRelationalAttention,
    state: torch.Tensor,
    source_index: torch.Tensor,
    kind: torch.Tensor,
    integer: torch.Tensor,
    floating: torch.Tensor,
    boolean: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return layer(
        state,
        source_index,
        kind,
        integer,
        floating,
        boolean,
        mask,
        torch.ones(state.shape[:2], dtype=torch.bool),
    )


def test_relational_one_incoming_source_changes_only_its_destination() -> None:
    layer = _controlled_relational()
    state = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [3.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    shape = (1, 3, 1)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[0, 1, 0] = True
    source_first = torch.zeros(shape, dtype=torch.int64)
    source_third = source_first.clone()
    source_third[0, 1, 0] = 2
    kind = torch.zeros(shape, dtype=torch.int64)
    integer = torch.zeros(shape, dtype=torch.int64)
    floating = torch.zeros(shape, dtype=torch.float32)
    boolean = torch.zeros(shape, dtype=torch.bool)

    first = _run_controlled_relation(
        layer, state, source_first, kind, integer, floating, boolean, mask
    )
    third = _run_controlled_relation(
        layer, state, source_third, kind, integer, floating, boolean, mask
    )
    torch.testing.assert_close(first[:, (0, 2)], state[:, (0, 2)])
    torch.testing.assert_close(third[:, (0, 2)], state[:, (0, 2)])
    torch.testing.assert_close(
        first[:, 1], torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        third[:, 1], torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        ("kind", (1.0, 0.0, 0.0, 0.0)),
        ("integer", (0.0, 2.0, 0.0, 0.0)),
        ("floating", (0.0, 0.0, 3.0, 0.0)),
        ("boolean", (0.0, 0.0, 0.0, 1.0)),
    ),
)
def test_relational_consumes_each_typed_edge_feature(
    field: str, expected: tuple[float, float, float, float]
) -> None:
    layer = _controlled_relational()
    state = torch.zeros((1, 2, 4), dtype=torch.float32)
    shape = (1, 2, 1)
    source = torch.zeros(shape, dtype=torch.int64)
    kind = torch.zeros(shape, dtype=torch.int64)
    integer = torch.zeros(shape, dtype=torch.int64)
    floating = torch.zeros(shape, dtype=torch.float32)
    boolean = torch.zeros(shape, dtype=torch.bool)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[0, 1, 0] = True
    if field == "kind":
        kind[0, 1, 0] = RELATION_KIND_IDS["occupies"]
    elif field == "integer":
        integer[0, 1, 0] = 2
    elif field == "floating":
        floating[0, 1, 0] = 3.0
    else:
        boolean[0, 1, 0] = True

    actual = _run_controlled_relation(
        layer, state, source, kind, integer, floating, boolean, mask
    )
    torch.testing.assert_close(
        actual[:, 1], torch.tensor([expected], dtype=torch.float32)
    )
    assert torch.count_nonzero(actual[:, 0]) == 0


def test_task4_forward_reverse_kinds_keep_source_destination_orientation(
    example_13x9: StructuredExample,
) -> None:
    batch = collate_examples((example_13x9,), horizons=HORIZONS)
    relation = next(
        row
        for row in example_13x9.decision.observation.relations
        if row.kind == "occupies"
    )
    source = batch.table_slices[relation.source.table].start + relation.source.row
    destination = (
        batch.table_slices[relation.target.table].start + relation.target.row
    )
    edges = batch.neighborhoods
    forward_slots = edges.mask[0, destination]
    reverse_slots = edges.mask[0, source]
    assert any(
        int(edge_source) == source
        and int(edge_kind) == RELATION_KIND_IDS["occupies"]
        for edge_source, edge_kind in zip(
            edges.source_index[0, destination][forward_slots],
            edges.kind[0, destination][forward_slots],
            strict=True,
        )
    )
    assert any(
        int(edge_source) == destination
        and int(edge_kind) == RELATION_KIND_IDS["occupies_reverse"]
        for edge_source, edge_kind in zip(
            edges.source_index[0, source][reverse_slots],
            edges.kind[0, source][reverse_slots],
            strict=True,
        )
    )

    layer = _controlled_relational()
    state = torch.zeros((1, 2, 4), dtype=torch.float32)
    source_index = torch.tensor([[[1], [0]]], dtype=torch.int64)
    kind = torch.tensor(
        [[
            [RELATION_KIND_IDS["occupies_reverse"]],
            [RELATION_KIND_IDS["occupies"]],
        ]],
        dtype=torch.int64,
    )
    zeros_int = torch.zeros((1, 2, 1), dtype=torch.int64)
    zeros_float = torch.zeros((1, 2, 1), dtype=torch.float32)
    zeros_bool = torch.zeros((1, 2, 1), dtype=torch.bool)
    mask = torch.ones((1, 2, 1), dtype=torch.bool)
    actual = _run_controlled_relation(
        layer,
        state,
        source_index,
        kind,
        zeros_int,
        zeros_float,
        zeros_bool,
        mask,
    )
    torch.testing.assert_close(
        actual[0, :, 0], torch.tensor([2.0, 1.0], dtype=torch.float32)
    )

def test_all_masked_incoming_is_finite_zero_safe_and_repeatable(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=41)
    node_state = run_local_stack(case, case.batch)
    edges = case.batch.neighborhoods
    no_edges = RelationNeighborhoodBatch(
        source_index=torch.full_like(edges.source_index, 1_000_000),
        kind=torch.full_like(edges.kind, 1_000_000),
        int_feature=torch.full_like(edges.int_feature, 1_000_000),
        float_feature=torch.full_like(edges.float_feature, 1_000_000.0),
        bool_feature=torch.ones_like(edges.bool_feature),
        mask=torch.zeros_like(edges.mask),
    )

    def run() -> torch.Tensor:
        return case.relational(
            node_state,
            no_edges.source_index,
            no_edges.kind,
            no_edges.int_feature,
            no_edges.float_feature,
            no_edges.bool_feature,
            no_edges.mask,
            case.batch.node_mask,
        )

    with torch.no_grad():
        for block in case.relational.blocks:
            for parameter in block.feed_forward.parameters():
                parameter.zero_()
    first = run()
    second = run()
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first[~case.batch.node_mask]) == 0
    torch.testing.assert_close(
        first[case.batch.node_mask],
        node_state[case.batch.node_mask],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_full_encoder_stack_is_deterministic_on_cpu(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=43)
    first = run_encoder_stack(case, case.batch)
    second = run_encoder_stack(case, case.batch)
    assert first.dtype == torch.float32
    assert {
        parameter.dtype
        for module in (case.local_hex, case.relational)
        for parameter in module.parameters()
    } == {torch.float32}
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_nonfinite_table_input_fails_before_attention(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=47)
    bad = replace_first_valid_numeric(case.batch, value=float("nan"))
    with pytest.raises(FloatingPointError, match=r"nonfinite.*tables\.cells\.numeric"):
        run_encoder_stack(case, bad)

    encoded = case.token_encoders(case.batch)
    bad_state = encoded.clone()
    bad_state[case.batch.node_mask.nonzero(as_tuple=False)[0].unbind()] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite.*node_state"):
        case.local_hex(
            bad_state,
            case.batch.cell_neighbor_index,
            case.batch.cell_neighbor_mask,
            case.batch.node_mask,
            case.batch.table_slices["cells"],
        )

    edges = case.batch.neighborhoods
    bad_float = edges.float_feature.clone()
    bad_float[edges.mask.nonzero(as_tuple=False)[0].unbind()] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite.*incoming_float_feature"):
        case.relational(
            encoded,
            edges.source_index,
            edges.kind,
            edges.int_feature,
            bad_float,
            edges.bool_feature,
            edges.mask,
            case.batch.node_mask,
        )


def test_layers_fail_closed_on_tensor_contract_violations(
    example_13x9: StructuredExample,
) -> None:
    case = make_layer_case(example_13x9, seed=53)
    encoded = case.token_encoders(case.batch)
    cells = case.batch.table_slices["cells"]
    with pytest.raises(ValueError, match="cell_neighbor_mask.*dtype"):
        case.local_hex(
            encoded,
            case.batch.cell_neighbor_index,
            case.batch.cell_neighbor_mask.to(torch.int64),
            case.batch.node_mask,
            cells,
        )
    bad_index = case.batch.neighborhoods.source_index.clone()
    first = case.batch.neighborhoods.mask.nonzero(as_tuple=False)[0]
    bad_index[first[0], first[1], first[2]] = case.batch.node_mask.shape[1]
    edges = case.batch.neighborhoods
    with pytest.raises(ValueError, match="incoming_source_index.*range"):
        case.relational(
            encoded,
            bad_index,
            edges.kind,
            edges.int_feature,
            edges.float_feature,
            edges.bool_feature,
            edges.mask,
            case.batch.node_mask,
        )
    cells_table = case.batch.tables["cells"]
    terrain = cells_table.categorical["terrain"].clone()
    terrain[0, 0] = TABLE_CATEGORICAL_CARDINALITIES["cells"]["terrain"]
    tables = dict(case.batch.tables)
    tables["cells"] = replace(
        cells_table,
        categorical={**cells_table.categorical, "terrain": terrain},
    )
    with pytest.raises(ValueError, match=r"tables\.cells\.categorical\.terrain.*range"):
        case.token_encoders(replace(case.batch, tables=tables))
    assert RELATION_KIND_COUNT == 14
