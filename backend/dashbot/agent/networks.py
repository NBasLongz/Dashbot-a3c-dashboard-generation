from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    feature_size: int
    hidden_size: int = 128
    max_columns: int = 10
    max_charts: int = 8
    aggregate_count: int = 6


class DashBotActorCritic(nn.Module):
    """Bi-LSTM actor-critic with sequential parameter heads from the paper."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size=config.feature_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        embedding_size = config.hidden_size * 2
        self.fuse = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(embedding_size, 1)

        self.action_block = SequentialHead(embedding_size, 4)
        self.key_column_block = SequentialHead(embedding_size, config.max_columns)
        self.mark_block = SequentialHead(embedding_size, 4)
        self.x_field_block = SequentialHead(embedding_size, config.max_columns)
        self.y_field_block = SequentialHead(embedding_size, config.max_columns)
        self.color_field_block = SequentialHead(embedding_size, config.max_columns + 1)
        self.x_agg_block = SequentialHead(embedding_size, config.aggregate_count)
        self.y_agg_block = SequentialHead(embedding_size, config.aggregate_count)
        self.remove_index_block = SequentialHead(embedding_size, config.max_charts)

    def forward(self, dashboard_features: torch.Tensor, masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(dashboard_features)
        pooled = encoded.mean(dim=1)
        shared = self.fuse(pooled)
        value = self.value_head(shared).squeeze(-1)

        outputs: dict[str, torch.Tensor] = {"value": value}
        context = shared
        for name, block in [
            ("action", self.action_block),
            ("key_column", self.key_column_block),
            ("mark", self.mark_block),
            ("x_field", self.x_field_block),
            ("y_field", self.y_field_block),
            ("color_field", self.color_field_block),
            ("x_agg", self.x_agg_block),
            ("y_agg", self.y_agg_block),
            ("remove_index", self.remove_index_block),
        ]:
            logits, context = block(shared, context)
            if masks and name in masks:
                logits = logits.masked_fill(~masks[name].bool(), -1e9)
            outputs[name] = logits
        return outputs


class DashBotIndependentActorCritic(nn.Module):
    """Ablation model for DashBot-ind.: shared Bi-LSTM with independent heads."""

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.LSTM(
            input_size=config.feature_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        embedding_size = config.hidden_size * 2
        self.fuse = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(embedding_size, 1)
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(embedding_size, 4),
                "key_column": nn.Linear(embedding_size, config.max_columns),
                "mark": nn.Linear(embedding_size, 4),
                "x_field": nn.Linear(embedding_size, config.max_columns),
                "y_field": nn.Linear(embedding_size, config.max_columns),
                "color_field": nn.Linear(embedding_size, config.max_columns + 1),
                "x_agg": nn.Linear(embedding_size, config.aggregate_count),
                "y_agg": nn.Linear(embedding_size, config.aggregate_count),
                "remove_index": nn.Linear(embedding_size, config.max_charts),
            }
        )

    def forward(self, dashboard_features: torch.Tensor, masks: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        encoded, _ = self.encoder(dashboard_features)
        pooled = encoded.mean(dim=1)
        shared = self.fuse(pooled)
        outputs: dict[str, torch.Tensor] = {"value": self.value_head(shared).squeeze(-1)}
        for name, head in self.heads.items():
            logits = head(shared)
            if masks and name in masks:
                logits = logits.masked_fill(~masks[name].bool(), -1e9)
            outputs[name] = logits
        return outputs


class SequentialHead(nn.Module):
    """One sequential classification block conditioned on previous context."""

    def __init__(self, embedding_size: int, output_size: int) -> None:
        super().__init__()
        self.intermediate = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU())
        self.context_fuse = nn.Sequential(nn.Linear(embedding_size * 2, embedding_size), nn.ReLU())
        self.output = nn.Linear(embedding_size, output_size)

    def forward(self, shared: torch.Tensor, previous_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = self.intermediate(previous_context)
        context = self.context_fuse(torch.cat([shared, intermediate], dim=-1))
        return self.output(context), context
