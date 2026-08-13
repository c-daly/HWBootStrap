"""Shared variable-candidate policy and auxiliary heads for tactical-v3."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .tactical_v3_batching import TABLE_ORDER, CandidateBatch, RaggedBatch
from .tactical_v3_layers import (
    LocalHexMessagePassing, TacticalV3ModelConfig,
    TypedRelationalAttention, TypedTokenEncoders,
)

_KIND_COUNT, _REFERENCE_COUNT, _INTEGER_COUNT, _BOOLEAN_COUNT = 4, 8, 7, 2

@dataclass(frozen=True, slots=True)
class PolicyOutput:
    candidate_logits: torch.Tensor
    outcome_logits: torch.Tensor
    horizon_logits: torch.Tensor
    remaining_turns: torch.Tensor

@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    decision_id: int
    candidate_id: int

def _row_linear(layer: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    result = (value.unsqueeze(-2) * layer.weight).sum(dim=-1)
    return result if layer.bias is None else result + layer.bias

class _RowLinear(nn.Linear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _row_linear(self, value)

def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value

def _contract(value: object, dtype: torch.dtype, shape: tuple[int, ...],
              device: torch.device, name: str) -> None:
    value = _tensor(value, name)
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}")

def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite values in {name}")

class TacticalV3Policy(nn.Module):
    """Encode ragged state once and score every legal candidate with shared weights."""
    state_table_names = TABLE_ORDER

    def __init__(self, config: TacticalV3ModelConfig) -> None:
        super().__init__()
        if type(config) is not TacticalV3ModelConfig:
            raise ValueError("config must be TacticalV3ModelConfig")
        self.config = config
        hidden, categorical = config.hidden_dim, config.categorical_dim
        self.token_encoders = TypedTokenEncoders(config)
        self.local_hex = LocalHexMessagePassing(config)
        self.relational = TypedRelationalAttention(config)
        self.state_projection = nn.Sequential(
            _RowLinear(len(TABLE_ORDER) * hidden * 2, hidden),
            nn.ReLU(), _RowLinear(hidden, hidden),
        )
        self.candidate_kind_embedding = nn.Embedding(_KIND_COUNT, categorical)
        self.projection_integer_projection = _RowLinear(_INTEGER_COUNT, hidden)
        self.projection_boolean_embeddings = nn.ModuleList(
            nn.Embedding(2, categorical) for _ in range(_BOOLEAN_COUNT)
        )
        candidate_width = (
            categorical + hidden + _BOOLEAN_COUNT * categorical
            + _REFERENCE_COUNT * hidden + _REFERENCE_COUNT
        )
        self.candidate_encoder = nn.Sequential(
            _RowLinear(candidate_width, hidden), nn.ReLU(), _RowLinear(hidden, hidden)
        )
        self.candidate_scorer = nn.Sequential(
            _RowLinear(hidden * 3, config.candidate_hidden_dim),
            nn.ReLU(), _RowLinear(config.candidate_hidden_dim, 1),
        )
        self.outcome_head = _RowLinear(hidden, 3)
        self.horizon_head = _RowLinear(hidden, len(config.horizon_turns))
        self.remaining_turns_head = _RowLinear(hidden, 1)

    def _validate_candidates(
        self, candidates: CandidateBatch, node_mask: torch.Tensor
    ) -> None:
        if type(candidates) is not CandidateBatch:
            raise ValueError("batch.candidates must be CandidateBatch")
        batch_size, node_count = node_mask.shape
        device = node_mask.device
        mask = _tensor(candidates.mask, "candidates.mask")
        if mask.dtype != torch.bool:
            raise ValueError("candidates.mask dtype must be torch.bool")
        if mask.device != device:
            raise ValueError(f"candidates.mask must be on device {device}")
        if mask.ndim != 2 or mask.shape[0] != batch_size or mask.shape[1] <= 0:
            raise ValueError("candidates.mask shape must be [B, C] with C > 0")
        count = mask.shape[1]
        contracts = (
            (candidates.candidate_id, torch.int64, (batch_size, count), "candidates.candidate_id"),
            (candidates.decision_id, torch.int64, (batch_size, count), "candidates.decision_id"),
            (candidates.kind, torch.int64, (batch_size, count), "candidates.kind"),
            (candidates.reference_index, torch.int64, (batch_size, count, 8), "candidates.reference_index"),
            (candidates.reference_mask, torch.bool, (batch_size, count, 8), "candidates.reference_mask"),
            (candidates.projection_integer, torch.int64, (batch_size, count, 7), "candidates.projection_integer"),
            (candidates.projection_boolean, torch.bool, (batch_size, count, 2), "candidates.projection_boolean"),
        )
        for value, dtype, shape, name in contracts:
            _contract(value, dtype, shape, device, name)
        for sample in range(batch_size):
            valid = mask[sample]
            if not bool(valid.any()):
                raise ValueError(f"sample {sample} has no valid candidates")
            ids = candidates.candidate_id[sample, valid]
            if not bool(((ids >= -(2**31)) & (ids < 2**31)).all()):
                raise ValueError("candidates.candidate_id active values are out of int32 range")
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError("candidate identity must be unique within a sample")
            decisions = candidates.decision_id[sample, valid]
            if not bool((decisions == decisions[0]).all()):
                raise ValueError("candidate decision_id values must agree within a sample")
        kinds = candidates.kind[mask]
        if not bool(((kinds >= 0) & (kinds < _KIND_COUNT)).all()):
            raise ValueError("candidates.kind active values are out of range")
        integers = candidates.projection_integer[
            mask.unsqueeze(-1).expand_as(candidates.projection_integer)
        ]
        if not bool(((integers >= -(2**31)) & (integers < 2**31)).all()):
            raise ValueError("candidates.projection_integer active values are out of int32 range")
        reference_mask = candidates.reference_mask & mask.unsqueeze(-1)
        references = candidates.reference_index[reference_mask]
        if references.numel() and not bool(
            ((references >= 0) & (references < node_count)).all()
        ):
            raise ValueError("candidates.reference_index active values are out of range")
        if references.numel():
            samples = (
                torch.arange(batch_size, device=device).view(batch_size, 1, 1)
                .expand_as(candidates.reference_index)[reference_mask]
            )
            if not bool(node_mask[samples, references].all()):
                raise ValueError(
                    "candidates.reference_index active values must select valid nodes"
                )

    def _encode_nodes(self, batch: RaggedBatch) -> torch.Tensor:
        state = self.token_encoders(batch)
        state = self.local_hex(
            state, batch.cell_neighbor_index, batch.cell_neighbor_mask,
            batch.node_mask, batch.table_slices["cells"],
        )
        edges = batch.neighborhoods
        return self.relational(
            state, edges.source_index, edges.kind, edges.int_feature,
            edges.float_feature, edges.bool_feature, edges.mask, batch.node_mask,
        )

    def _pool_state(self, nodes: torch.Tensor, batch: RaggedBatch) -> torch.Tensor:
        batch_size, hidden = nodes.shape[0], self.config.hidden_dim
        summaries: list[torch.Tensor] = []
        for table_name in TABLE_ORDER:
            rows = nodes[:, batch.table_slices[table_name], :]
            mask = batch.tables[table_name].mask
            if rows.shape[1] == 0:
                mean = torch.zeros(
                    (batch_size, hidden), dtype=nodes.dtype, device=nodes.device
                )
                maximum = torch.zeros_like(mean)
            else:
                expanded = mask.unsqueeze(-1)
                safe = torch.where(expanded, rows, torch.zeros_like(rows))
                count = expanded.sum(dim=1).clamp(min=1).to(torch.float64)
                mean = (safe.to(torch.float64).sum(dim=1) / count).to(nodes.dtype)
                candidate_max = torch.where(
                    expanded, rows,
                    torch.full_like(rows, torch.finfo(rows.dtype).min),
                ).max(dim=1).values
                maximum = torch.where(
                    mask.any(dim=1, keepdim=True), candidate_max,
                    torch.zeros_like(candidate_max),
                )
            summaries.extend((mean, maximum))
        state = self.state_projection(torch.cat(summaries, dim=-1))
        _finite(state, "shared state")
        return state

    def _encode_candidates(
        self, candidates: CandidateBatch, nodes: torch.Tensor
    ) -> torch.Tensor:
        mask = candidates.mask
        reference_mask = candidates.reference_mask & mask.unsqueeze(-1)
        kind = torch.where(mask, candidates.kind, torch.zeros_like(candidates.kind))
        integers = torch.where(
            mask.unsqueeze(-1), candidates.projection_integer,
            torch.zeros_like(candidates.projection_integer),
        )
        booleans = torch.where(
            mask.unsqueeze(-1), candidates.projection_boolean,
            torch.zeros_like(candidates.projection_boolean),
        )
        references = torch.where(
            reference_mask, candidates.reference_index,
            torch.zeros_like(candidates.reference_index),
        )
        batch_size, count, _ = references.shape
        gather = references.reshape(batch_size, -1).unsqueeze(-1).expand(
            -1, -1, self.config.hidden_dim
        )
        gathered = torch.gather(nodes, 1, gather).reshape(
            batch_size, count, _REFERENCE_COUNT, self.config.hidden_dim
        )
        gathered = gathered * reference_mask.unsqueeze(-1).to(gathered.dtype)
        boolean_fields = tuple(
            embedding(booleans[:, :, index].to(torch.int64))
            for index, embedding in enumerate(self.projection_boolean_embeddings)
        )
        fields = (
            self.candidate_kind_embedding(kind),
            self.projection_integer_projection(integers.to(torch.float32)),
            *boolean_fields,
            gathered.flatten(start_dim=2),
            reference_mask.to(nodes.dtype),
        )
        encoded = self.candidate_encoder(torch.cat(fields, dim=-1))
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        _finite(encoded[mask], "encoded candidates")
        return encoded

    def forward(self, batch: RaggedBatch) -> PolicyOutput:
        if type(batch) is not RaggedBatch:
            raise ValueError("batch must be RaggedBatch")
        node_mask = _tensor(batch.node_mask, "node_mask")
        if node_mask.ndim != 2 or node_mask.shape[0] <= 0:
            raise ValueError("node_mask shape must be [B, Nnode]")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("policy parameters must use torch.float32")
        if node_mask.device != parameter.device:
            raise ValueError(f"node_mask must be on device {parameter.device}")
        self._validate_candidates(batch.candidates, node_mask)
        nodes = self._encode_nodes(batch)
        state = self._pool_state(nodes, batch)
        candidates = self._encode_candidates(batch.candidates, nodes)
        expanded = state.unsqueeze(1).expand_as(candidates)
        score_input = torch.cat((expanded, candidates, expanded * candidates), dim=-1)
        raw_logits = self.candidate_scorer(score_input).squeeze(-1)
        _finite(raw_logits[batch.candidates.mask], "candidate_logits")
        outcome = self.outcome_head(state)
        horizon = self.horizon_head(state)
        remaining = self.remaining_turns_head(state).squeeze(-1)
        _finite(outcome, "outcome_logits")
        _finite(horizon, "horizon_logits")
        _finite(remaining, "remaining_turns")
        logits = torch.where(
            batch.candidates.mask, raw_logits,
            torch.full_like(raw_logits, -torch.inf),
        )
        return PolicyOutput(logits, outcome, horizon, remaining)

    @torch.inference_mode()
    def select(self, batch: RaggedBatch) -> tuple[CandidateIdentity, ...]:
        output = self(batch)
        selected: list[CandidateIdentity] = []
        for sample in range(output.candidate_logits.shape[0]):
            valid = batch.candidates.mask[sample]
            logits = output.candidate_logits[sample, valid]
            candidate_ids = batch.candidates.candidate_id[sample, valid]
            decision_ids = batch.candidates.decision_id[sample, valid]
            tied = logits == logits.max()
            winning_id = candidate_ids[tied].min()
            chosen = (candidate_ids == winning_id).nonzero(as_tuple=False)[0, 0]
            selected.append(CandidateIdentity(
                int(decision_ids[chosen].item()), int(candidate_ids[chosen].item())
            ))
        return tuple(selected)
