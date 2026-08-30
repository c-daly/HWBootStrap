"""Typed neural layers for ragged tactical-v3 relational state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .tactical_v3_batching import (
    RELATION_KIND_COUNT,
    TABLE_BOOLEAN_FIELDS,
    TABLE_CATEGORICAL_CARDINALITIES,
    TABLE_CATEGORICAL_FIELDS,
    TABLE_NUMERIC_FIELDS,
    TABLE_ORDER,
    RaggedBatch,
)


@dataclass(frozen=True, slots=True)
class TacticalV3ModelConfig:
    hidden_dim: int = 64
    categorical_dim: int = 16
    cell_message_rounds: int = 2
    relation_rounds: int = 2
    attention_heads: int = 4
    feed_forward_dim: int = 128
    candidate_hidden_dim: int = 128
    horizon_turns: tuple[int, ...] = (4, 8, 16)

    def __post_init__(self) -> None:
        positive = (
            "hidden_dim",
            "categorical_dim",
            "cell_message_rounds",
            "relation_rounds",
            "attention_heads",
            "feed_forward_dim",
            "candidate_hidden_dim",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        horizons = self.horizon_turns
        if (
            type(horizons) is not tuple
            or not horizons
            or any(type(value) is not int or value <= 0 for value in horizons)
            or any(left >= right for left, right in zip(horizons, horizons[1:]))
        ):
            raise ValueError(
                "horizon_turns must be a non-empty strictly increasing tuple "
                "of positive integers"
            )


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value


def _require_dtype(value: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}")


def _require_device(value: torch.Tensor, device: torch.device, name: str) -> None:
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite values in {name}")


def _field_key(table_name: str, field_name: str) -> str:
    return f"{table_name}__{field_name}"


def _row_linear(layer: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    """Apply a linear map with a fixed feature-reduction order per row."""
    result = (value.unsqueeze(-2) * layer.weight).sum(dim=-1)
    if layer.bias is not None:
        result = result + layer.bias
    return result


class TypedTokenEncoders(nn.Module):
    """Encode each canonical token table without row or engine identities."""

    def __init__(self, config: TacticalV3ModelConfig) -> None:
        super().__init__()
        if type(config) is not TacticalV3ModelConfig:
            raise ValueError("config must be TacticalV3ModelConfig")
        self.config = config
        self.numeric_projections = nn.ModuleDict(
            {
                table_name: nn.Linear(
                    len(TABLE_NUMERIC_FIELDS[table_name]),
                    config.hidden_dim,
                )
                for table_name in TABLE_ORDER
                if TABLE_NUMERIC_FIELDS[table_name]
            }
        )
        self.categorical_embeddings = nn.ModuleDict(
            {
                _field_key(table_name, field_name): nn.Embedding(
                    TABLE_CATEGORICAL_CARDINALITIES[table_name][field_name],
                    config.categorical_dim,
                )
                for table_name in TABLE_ORDER
                for field_name in TABLE_CATEGORICAL_FIELDS[table_name]
            }
        )
        self.boolean_embeddings = nn.ModuleDict(
            {
                _field_key(table_name, field_name): nn.Embedding(
                    2, config.categorical_dim
                )
                for table_name in TABLE_ORDER
                for field_name in TABLE_BOOLEAN_FIELDS[table_name]
            }
        )
        self.table_projections = nn.ModuleDict()
        for table_name in TABLE_ORDER:
            input_dim = (
                config.hidden_dim if TABLE_NUMERIC_FIELDS[table_name] else 0
            ) + config.categorical_dim * (
                len(TABLE_CATEGORICAL_FIELDS[table_name])
                + len(TABLE_BOOLEAN_FIELDS[table_name])
            )
            self.table_projections[table_name] = nn.Linear(
                input_dim, config.hidden_dim
            )

    def _validate_batch(self, batch: RaggedBatch) -> tuple[int, int]:
        if type(batch) is not RaggedBatch:
            raise ValueError("batch must be RaggedBatch")
        if tuple(batch.tables) != TABLE_ORDER:
            raise ValueError("batch.tables must use canonical TABLE_ORDER")
        if tuple(batch.table_slices) != TABLE_ORDER:
            raise ValueError("batch.table_slices must use canonical TABLE_ORDER")
        node_mask = _require_tensor(batch.node_mask, "node_mask")
        _require_dtype(node_mask, torch.bool, "node_mask")
        if node_mask.ndim != 2 or node_mask.shape[0] <= 0:
            raise ValueError("node_mask shape must be [B, Nnode]")
        batch_size, node_count = node_mask.shape
        expected_start = 0
        table_masks: list[torch.Tensor] = []
        for table_name in TABLE_ORDER:
            table = batch.tables[table_name]
            numeric = _require_tensor(
                table.numeric, f"tables.{table_name}.numeric"
            )
            _require_dtype(
                numeric, torch.float32, f"tables.{table_name}.numeric"
            )
            expected_features = len(TABLE_NUMERIC_FIELDS[table_name])
            if (
                numeric.ndim != 3
                or numeric.shape[0] != batch_size
                or numeric.shape[2] != expected_features
            ):
                raise ValueError(
                    f"tables.{table_name}.numeric shape must be [B, N, "
                    f"{expected_features}]"
                )
            _require_device(
                numeric, node_mask.device, f"tables.{table_name}.numeric"
            )
            _require_finite(numeric, f"tables.{table_name}.numeric")
            row_count = numeric.shape[1]
            mask = _require_tensor(table.mask, f"tables.{table_name}.mask")
            _require_dtype(mask, torch.bool, f"tables.{table_name}.mask")
            if mask.shape != (batch_size, row_count):
                raise ValueError(
                    f"tables.{table_name}.mask shape must be [B, N]"
                )
            _require_device(mask, node_mask.device, f"tables.{table_name}.mask")
            if tuple(table.categorical) != TABLE_CATEGORICAL_FIELDS[table_name]:
                raise ValueError(
                    f"tables.{table_name}.categorical fields do not match schema"
                )
            for field_name in TABLE_CATEGORICAL_FIELDS[table_name]:
                name = f"tables.{table_name}.categorical.{field_name}"
                values = _require_tensor(table.categorical[field_name], name)
                _require_dtype(values, torch.int64, name)
                if values.shape != (batch_size, row_count):
                    raise ValueError(f"{name} shape must be [B, N]")
                _require_device(values, node_mask.device, name)
                cardinality = TABLE_CATEGORICAL_CARDINALITIES[table_name][
                    field_name
                ]
                active = values[mask]
                if active.numel() and not bool(
                    ((active >= 0) & (active < cardinality)).all()
                ):
                    raise ValueError(f"{name} active values are out of range")
            if tuple(table.boolean) != TABLE_BOOLEAN_FIELDS[table_name]:
                raise ValueError(
                    f"tables.{table_name}.boolean fields do not match schema"
                )
            for field_name in TABLE_BOOLEAN_FIELDS[table_name]:
                name = f"tables.{table_name}.boolean.{field_name}"
                values = _require_tensor(table.boolean[field_name], name)
                _require_dtype(values, torch.bool, name)
                if values.shape != (batch_size, row_count):
                    raise ValueError(f"{name} shape must be [B, N]")
                _require_device(values, node_mask.device, name)
            table_slice = batch.table_slices[table_name]
            if (
                type(table_slice) is not slice
                or table_slice.start != expected_start
                or table_slice.stop != expected_start + row_count
                or table_slice.step not in (None, 1)
            ):
                raise ValueError(
                    f"table_slices.{table_name} is not canonical"
                )
            expected_start = expected_start + row_count
            table_masks.append(mask)
        if expected_start != node_count:
            raise ValueError("table slices do not cover node_mask")
        expected_node_mask = torch.cat(table_masks, dim=1)
        if not torch.equal(node_mask, expected_node_mask):
            raise ValueError("node_mask must equal concatenated table masks")
        return batch_size, node_count

    def forward(self, batch: RaggedBatch) -> torch.Tensor:
        self._validate_batch(batch)
        encoded_tables: list[torch.Tensor] = []
        for table_name in TABLE_ORDER:
            table = batch.tables[table_name]
            mask = table.mask.unsqueeze(-1)
            fields: list[torch.Tensor] = []
            if TABLE_NUMERIC_FIELDS[table_name]:
                numeric = torch.where(
                    table.mask.unsqueeze(-1),
                    table.numeric,
                    torch.zeros_like(table.numeric),
                )
                fields.append(
                    _row_linear(
                        self.numeric_projections[table_name], numeric
                    )
                )
            for field_name in TABLE_CATEGORICAL_FIELDS[table_name]:
                values = torch.where(
                    table.mask,
                    table.categorical[field_name],
                    torch.zeros_like(table.categorical[field_name]),
                )
                fields.append(
                    self.categorical_embeddings[
                        _field_key(table_name, field_name)
                    ](values)
                )
            for field_name in TABLE_BOOLEAN_FIELDS[table_name]:
                values = torch.where(
                    table.mask,
                    table.boolean[field_name],
                    torch.zeros_like(table.boolean[field_name]),
                ).to(torch.int64)
                fields.append(
                    self.boolean_embeddings[
                        _field_key(table_name, field_name)
                    ](values)
                )
            combined = torch.cat(fields, dim=-1)
            encoded = _row_linear(
                self.table_projections[table_name], combined
            )
            encoded_tables.append(encoded * mask.to(encoded.dtype))
        result = torch.cat(encoded_tables, dim=1)
        return result * batch.node_mask.unsqueeze(-1).to(result.dtype)


def _validate_node_state(
    node_state: torch.Tensor,
    node_mask: torch.Tensor,
    hidden_dim: int,
) -> tuple[int, int]:
    node_state = _require_tensor(node_state, "node_state")
    node_mask = _require_tensor(node_mask, "node_mask")
    if node_state.dtype != torch.float32:
        raise ValueError("node_state dtype must be torch.float32")
    _require_dtype(node_mask, torch.bool, "node_mask")
    if node_state.ndim != 3 or node_state.shape[2] != hidden_dim:
        raise ValueError(f"node_state shape must be [B, Nnode, {hidden_dim}]")
    if node_mask.shape != node_state.shape[:2]:
        raise ValueError("node_mask shape must match node_state [B, Nnode]")
    _require_device(node_mask, node_state.device, "node_mask")
    _require_finite(node_state, "node_state")
    return node_state.shape[0], node_state.shape[1]


class LocalHexMessagePassing(nn.Module):
    """Update canonical cell rows from Task 4's authenticated hex neighbors."""

    def __init__(self, config: TacticalV3ModelConfig) -> None:
        super().__init__()
        if type(config) is not TacticalV3ModelConfig:
            raise ValueError("config must be TacticalV3ModelConfig")
        self.config = config
        self.message_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        node_state: torch.Tensor,
        cell_neighbor_index: torch.Tensor,
        cell_neighbor_mask: torch.Tensor,
        node_mask: torch.Tensor,
        cells_slice: slice,
    ) -> torch.Tensor:
        batch_size, node_count = _validate_node_state(
            node_state, node_mask, self.config.hidden_dim
        )
        index = _require_tensor(cell_neighbor_index, "cell_neighbor_index")
        mask = _require_tensor(cell_neighbor_mask, "cell_neighbor_mask")
        _require_dtype(index, torch.int64, "cell_neighbor_index")
        _require_dtype(mask, torch.bool, "cell_neighbor_mask")
        _require_device(index, node_state.device, "cell_neighbor_index")
        _require_device(mask, node_state.device, "cell_neighbor_mask")
        if (
            type(cells_slice) is not slice
            or type(cells_slice.start) is not int
            or type(cells_slice.stop) is not int
            or cells_slice.step not in (None, 1)
            or cells_slice.start != 0
            or cells_slice.stop < cells_slice.start
            or cells_slice.stop > node_count
        ):
            raise ValueError("cells_slice must be a bounded canonical slice")
        cell_count = cells_slice.stop - cells_slice.start
        if index.shape != (batch_size, cell_count, 6):
            raise ValueError("cell_neighbor_index shape must be [B, Ncell, 6]")
        if mask.shape != index.shape:
            raise ValueError(
                "cell_neighbor_mask shape must match cell_neighbor_index"
            )
        active = index[mask]
        if active.numel() and not bool(
            ((active >= cells_slice.start) & (active < cells_slice.stop)).all()
        ):
            raise ValueError(
                "cell_neighbor_index active values must lie inside cells slice"
            )
        if active.numel():
            sample = (
                torch.arange(batch_size, device=node_state.device)
                .view(batch_size, 1, 1)
                .expand_as(index)[mask]
            )
            if not bool(node_mask[sample, active].all()):
                raise ValueError(
                    "cell_neighbor_index active values must select valid nodes"
                )
        cell_node_mask = node_mask[:, cells_slice]
        if bool((mask & ~cell_node_mask.unsqueeze(-1)).any()):
            raise ValueError("padded cell destinations cannot have neighbors")

        state = node_state * node_mask.unsqueeze(-1).to(node_state.dtype)
        safe_index = torch.where(mask, index, torch.zeros_like(index))
        gather_index = safe_index.reshape(batch_size, -1).unsqueeze(-1).expand(
            -1, -1, self.config.hidden_dim
        )
        for _ in range(self.config.cell_message_rounds):
            neighbors = torch.gather(state, 1, gather_index).reshape(
                batch_size, cell_count, 6, self.config.hidden_dim
            )
            expanded_mask = mask.unsqueeze(-1)
            masked_neighbors = neighbors * expanded_mask.to(neighbors.dtype)
            count = expanded_mask.sum(dim=2).clamp(min=1).to(neighbors.dtype)
            mean = masked_neighbors.sum(dim=2) / count
            maximum_candidate = torch.where(
                expanded_mask,
                neighbors,
                torch.full_like(neighbors, torch.finfo(neighbors.dtype).min),
            ).max(dim=2).values
            maximum = torch.where(
                mask.any(dim=2, keepdim=True),
                maximum_candidate,
                torch.zeros_like(maximum_candidate),
            )
            cells = state[:, cells_slice, :]
            message_input = torch.cat((cells, mean, maximum), dim=-1)
            update = _row_linear(self.message_mlp[0], message_input)
            update = torch.relu(update)
            update = _row_linear(self.message_mlp[2], update)
            updated_cells = (cells + update) * cell_node_mask.unsqueeze(-1).to(
                cells.dtype
            )
            state = torch.cat(
                (
                    state[:, : cells_slice.start, :],
                    updated_cells,
                    state[:, cells_slice.stop :, :],
                ),
                dim=1,
            )
            state = state * node_mask.unsqueeze(-1).to(state.dtype)
        return state


class _RelationalAttentionBlock(nn.Module):
    def __init__(self, config: TacticalV3ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.heads = config.attention_heads
        self.head_dim = config.hidden_dim // config.attention_heads
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.feed_forward_dim),
            nn.ReLU(),
            nn.Linear(config.feed_forward_dim, config.hidden_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        source: torch.Tensor,
        edge_state: torch.Tensor,
        incoming_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, slot_count, _ = source.shape
        source_with_edge = source + edge_state
        query = _row_linear(self.query, state).reshape(
            batch_size, node_count, self.heads, self.head_dim
        )
        key = _row_linear(self.key, source_with_edge).reshape(
            batch_size, node_count, slot_count, self.heads, self.head_dim
        )
        value = _row_linear(self.value, source_with_edge).reshape(
            batch_size, node_count, slot_count, self.heads, self.head_dim
        )
        scores = (
            query.unsqueeze(2).to(torch.float64) * key.to(torch.float64)
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        active_scores = scores[incoming_mask.unsqueeze(-1).expand_as(scores)]
        if active_scores.numel() and not bool(torch.isfinite(active_scores).all()):
            raise FloatingPointError("nonfinite values before relational attention")
        head_mask = incoming_mask.unsqueeze(-1)
        masked_scores = torch.where(
            head_mask,
            scores,
            torch.full_like(scores, torch.finfo(scores.dtype).min),
        )
        weights = torch.softmax(masked_scores, dim=2)
        weights = weights * head_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1.0)
        message = (
            weights.unsqueeze(-1) * value.to(torch.float64)
        ).sum(dim=2).reshape(
            batch_size, node_count, self.hidden_dim
        ).to(state.dtype)
        message = _row_linear(self.output, message)
        destination_mask = node_mask.unsqueeze(-1).to(state.dtype)
        after_attention = (state + message) * destination_mask
        feed_forward = _row_linear(self.feed_forward[0], after_attention)
        feed_forward = torch.relu(feed_forward)
        feed_forward = _row_linear(self.feed_forward[2], feed_forward)
        return (after_attention + feed_forward) * destination_mask


class TypedRelationalAttention(nn.Module):
    """Apply typed, masked incoming attention without synthetic self-edges."""

    def __init__(self, config: TacticalV3ModelConfig) -> None:
        super().__init__()
        if type(config) is not TacticalV3ModelConfig:
            raise ValueError("config must be TacticalV3ModelConfig")
        self.config = config
        self.relation_kind_embedding = nn.Embedding(
            RELATION_KIND_COUNT, config.categorical_dim
        )
        self.integer_projection = nn.Linear(1, config.categorical_dim)
        self.float_projection = nn.Linear(1, config.categorical_dim)
        self.boolean_embedding = nn.Embedding(2, config.categorical_dim)
        self.edge_projection = nn.Linear(
            config.categorical_dim * 4, config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            _RelationalAttentionBlock(config)
            for _ in range(config.relation_rounds)
        )

    def forward(
        self,
        node_state: torch.Tensor,
        incoming_source_index: torch.Tensor,
        incoming_relation_kind: torch.Tensor,
        incoming_int_feature: torch.Tensor,
        incoming_float_feature: torch.Tensor,
        incoming_bool_feature: torch.Tensor,
        incoming_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count = _validate_node_state(
            node_state, node_mask, self.config.hidden_dim
        )
        tensors = (
            (incoming_source_index, torch.int64, "incoming_source_index"),
            (incoming_relation_kind, torch.int64, "incoming_relation_kind"),
            (incoming_int_feature, torch.int64, "incoming_int_feature"),
            (incoming_float_feature, torch.float32, "incoming_float_feature"),
            (incoming_bool_feature, torch.bool, "incoming_bool_feature"),
            (incoming_mask, torch.bool, "incoming_mask"),
        )
        shape: torch.Size | None = None
        for value, dtype, name in tensors:
            value = _require_tensor(value, name)
            _require_dtype(value, dtype, name)
            _require_device(value, node_state.device, name)
            if value.ndim != 3 or value.shape[:2] != (batch_size, node_count):
                raise ValueError(f"{name} shape must be [B, Nnode, Nincoming]")
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError("all incoming tensors must have identical shapes")
        assert shape is not None
        if shape[2] <= 0:
            raise ValueError("incoming tensors require at least one slot")
        _require_finite(incoming_float_feature, "incoming_float_feature")
        active_source = incoming_source_index[incoming_mask]
        if active_source.numel() and not bool(
            ((active_source >= 0) & (active_source < node_count)).all()
        ):
            raise ValueError("incoming_source_index active values are out of range")
        active_kind = incoming_relation_kind[incoming_mask]
        if active_kind.numel() and not bool(
            ((active_kind >= 0) & (active_kind < RELATION_KIND_COUNT)).all()
        ):
            raise ValueError("incoming_relation_kind active values are out of range")
        active_integer = incoming_int_feature[incoming_mask]
        if active_integer.numel() and not bool(
            ((active_integer >= -(2**31)) & (active_integer < 2**31)).all()
        ):
            raise ValueError("incoming_int_feature active values are out of int32 range")
        destination_has_edge = incoming_mask.any(dim=2)
        if bool((destination_has_edge & ~node_mask).any()):
            raise ValueError("incoming edges cannot target padded nodes")
        if active_source.numel():
            sample = (
                torch.arange(batch_size, device=node_state.device)
                .view(batch_size, 1, 1)
                .expand_as(incoming_source_index)[incoming_mask]
            )
            if not bool(node_mask[sample, active_source].all()):
                raise ValueError(
                    "incoming_source_index active values must select valid nodes"
                )

        safe_source = torch.where(
            incoming_mask,
            incoming_source_index,
            torch.zeros_like(incoming_source_index),
        )
        safe_kind = torch.where(
            incoming_mask,
            incoming_relation_kind,
            torch.zeros_like(incoming_relation_kind),
        )
        safe_integer = torch.where(
            incoming_mask,
            incoming_int_feature,
            torch.zeros_like(incoming_int_feature),
        )
        safe_float = torch.where(
            incoming_mask,
            incoming_float_feature,
            torch.zeros_like(incoming_float_feature),
        )
        safe_bool = torch.where(
            incoming_mask,
            incoming_bool_feature,
            torch.zeros_like(incoming_bool_feature),
        )
        gather_index = safe_source.reshape(batch_size, -1).unsqueeze(-1).expand(
            -1, -1, self.config.hidden_dim
        )
        state = node_state * node_mask.unsqueeze(-1).to(node_state.dtype)
        edge_fields = (
            self.relation_kind_embedding(safe_kind),
            _row_linear(
                self.integer_projection,
                safe_integer.to(torch.float32).unsqueeze(-1),
            ),
            _row_linear(self.float_projection, safe_float.unsqueeze(-1)),
            self.boolean_embedding(safe_bool.to(torch.int64)),
        )
        edge_state = _row_linear(
            self.edge_projection, torch.cat(edge_fields, dim=-1)
        )
        edge_state = edge_state * incoming_mask.unsqueeze(-1).to(edge_state.dtype)
        _require_finite(edge_state, "encoded incoming edge features")
        for block in self.blocks:
            source = torch.gather(state, 1, gather_index).reshape(
                batch_size,
                node_count,
                shape[2],
                self.config.hidden_dim,
            )
            source = source * incoming_mask.unsqueeze(-1).to(source.dtype)
            block_state = block(
                state, source, edge_state, incoming_mask, node_mask
            )
            _require_finite(block_state, "relational block output")
            state = block_state
        return state
