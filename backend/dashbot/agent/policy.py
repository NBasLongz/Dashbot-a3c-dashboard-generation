from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from dashbot.agent.feature_encoder import AGGREGATES, MARK_TYPES
from dashbot.core.models import ActionType, ChartSpec, DashboardState, DatasetProfile
from dashbot.rl_env.constraints import ConstraintSampler


ACTION_TYPES: tuple[ActionType, ...] = ("change", "add", "remove", "terminate")


@dataclass
class SampledDecision:
    action: ActionType
    params: dict[str, Any]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class PolicySampler:
    """Sample DashBot actions and parameters with constrained masks."""

    def __init__(self, constraints: ConstraintSampler | None = None) -> None:
        self.constraints = constraints or ConstraintSampler()

    def sample(
        self,
        outputs: dict[str, torch.Tensor],
        state: DashboardState,
        profile: DatasetProfile,
    ) -> SampledDecision:
        value = outputs["value"].squeeze(0)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        action = self._sample_name(
            outputs["action"].squeeze(0),
            ACTION_TYPES,
            self.constraints.action_mask(state.charts),
            log_probs,
            entropies,
        )

        params: dict[str, Any] = {}
        if action == "change":
            key_column = self._sample_column(
                outputs["key_column"].squeeze(0),
                profile,
                exclude=None,
                log_probs=log_probs,
                entropies=entropies,
            )
            params["key_column"] = key_column

        elif action == "add":
            chart = self._sample_chart(outputs, state, profile, log_probs, entropies)
            params["chart"] = chart

        elif action == "remove":
            valid_indices = {index: index < len(state.charts) for index in range(self.constraints.max_charts)}
            index = self._sample_index(
                outputs["remove_index"].squeeze(0),
                valid_indices,
                log_probs,
                entropies,
            )
            params["index"] = index

        return SampledDecision(
            action=action,
            params=params,
            log_prob=torch.stack(log_probs).sum(),
            entropy=torch.stack(entropies).sum(),
            value=value,
        )

    def _sample_chart(
        self,
        outputs: dict[str, torch.Tensor],
        state: DashboardState,
        profile: DatasetProfile,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> ChartSpec:
        mark_mask = self._available_mark_mask(profile)
        mark = self._sample_name(
            outputs["mark"].squeeze(0),
            MARK_TYPES,
            mark_mask,
            log_probs,
            entropies,
        )
        x_field, y_field = self._sample_fields_for_mark(outputs, mark, state, profile, log_probs, entropies)
        x_agg = self._sample_aggregate(outputs["x_agg"].squeeze(0), profile, x_field, log_probs, entropies)
        y_agg = self._sample_aggregate(outputs["y_agg"].squeeze(0), profile, y_field, log_probs, entropies)
        color_field = self._sample_optional_column(
            outputs["color_field"].squeeze(0),
            profile,
            exclude=state.key_column,
            log_probs=log_probs,
            entropies=entropies,
        )
        chart = ChartSpec(
            mark=mark,
            x=x_field,
            y=y_field,
            color=color_field,
            x_agg=x_agg,
            y_agg=y_agg,
            title=f"{y_field} by {x_field}",
        )
        return self._normalize_chart(chart, profile)

    def _sample_fields_for_mark(
        self,
        outputs: dict[str, torch.Tensor],
        mark: str,
        state: DashboardState,
        profile: DatasetProfile,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> tuple[str, str]:
        if mark == "line":
            x_types, y_types = {"T"}, {"Q"}
        elif mark == "point":
            x_types, y_types = {"Q"}, {"Q"}
        elif mark == "boxplot":
            x_types, y_types = {"N"}, {"Q"}
        else:
            x_types, y_types = {"N"}, {"Q"}

        x_field = self._sample_column(
            outputs["x_field"].squeeze(0),
            profile,
            exclude=None,
            log_probs=log_probs,
            entropies=entropies,
            allowed_types=x_types,
        )
        y_field = self._sample_column(
            outputs["y_field"].squeeze(0),
            profile,
            exclude=x_field,
            log_probs=log_probs,
            entropies=entropies,
            allowed_types=y_types,
        )
        return x_field, y_field

    def _sample_column(
        self,
        logits: torch.Tensor,
        profile: DatasetProfile,
        exclude: str | None,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
        allowed_types: set[str] | None = None,
    ) -> str:
        columns = profile.modeled_columns()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        names = []
        for index, column in enumerate(columns):
            names.append(column.name)
            allowed = allowed_types is None or column.type in allowed_types
            mask[index] = column.name != exclude and allowed
        if not mask.any() and allowed_types is not None:
            for index, column in enumerate(columns):
                mask[index] = column.name != exclude
        index = self._sample_tensor_index(logits, mask, log_probs, entropies)
        return names[index]

    def _sample_optional_column(
        self,
        logits: torch.Tensor,
        profile: DatasetProfile,
        exclude: str | None,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> str | None:
        columns = profile.modeled_columns()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        names: list[str | None] = []
        for index, column in enumerate(columns):
            names.append(column.name)
            mask[index] = column.name != exclude
        none_index = len(columns)
        if none_index < len(mask):
            names.append(None)
            mask[none_index] = True
        index = self._sample_tensor_index(logits, mask, log_probs, entropies)
        return names[index]

    def _sample_aggregate(
        self,
        logits: torch.Tensor,
        profile: DatasetProfile,
        field: str,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> str | None:
        aggregate_mask = self.constraints.aggregate_mask(profile, field)
        aggregate = self._sample_name(logits, AGGREGATES, aggregate_mask, log_probs, entropies)
        return None if aggregate == "none" else aggregate

    def _sample_name(
        self,
        logits: torch.Tensor,
        vocabulary: tuple[Any, ...],
        mask_dict: dict[Any, bool],
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> Any:
        mask = torch.tensor([mask_dict.get(item, False) for item in vocabulary], dtype=torch.bool, device=logits.device)
        index = self._sample_tensor_index(logits[: len(vocabulary)], mask, log_probs, entropies)
        return vocabulary[index]

    @staticmethod
    def _available_mark_mask(profile: DatasetProfile) -> dict[str, bool]:
        counts = {"Q": 0, "N": 0, "T": 0}
        for column in profile.modeled_columns():
            counts[column.type] += 1
        return {
            "bar": counts["Q"] > 0,
            "line": counts["T"] > 0 and counts["Q"] > 0,
            "point": counts["Q"] >= 2,
            "boxplot": counts["N"] > 0 and counts["Q"] > 0,
        }

    def _sample_index(
        self,
        logits: torch.Tensor,
        mask_dict: dict[int, bool],
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> int:
        mask = torch.tensor([mask_dict.get(index, False) for index in range(len(logits))], dtype=torch.bool, device=logits.device)
        return self._sample_tensor_index(logits, mask, log_probs, entropies)

    @staticmethod
    def _sample_tensor_index(
        logits: torch.Tensor,
        mask: torch.Tensor,
        log_probs: list[torch.Tensor],
        entropies: list[torch.Tensor],
    ) -> int:
        if not mask.any():
            mask = torch.ones_like(mask, dtype=torch.bool)
        masked_logits = logits.masked_fill(~mask, -1e9)
        distribution = Categorical(logits=masked_logits)
        index = distribution.sample()
        log_probs.append(distribution.log_prob(index))
        entropies.append(distribution.entropy())
        return int(index.item())

    @staticmethod
    def _normalize_chart(chart: ChartSpec, profile: DatasetProfile) -> ChartSpec:
        by_name = profile.by_name()
        x_type = by_name[chart.x].type if chart.x in by_name else "N"
        y_type = by_name[chart.y].type if chart.y and chart.y in by_name else None
        color_type = by_name[chart.color].type if chart.color and chart.color in by_name else None
        if chart.color in {chart.x, chart.y} or color_type != "N":
            chart.color = None
            chart.color_agg = None

        if chart.mark == "point":
            chart.x_agg = None
            chart.y_agg = None
        elif chart.mark == "line":
            chart.x_agg = None
            chart.y_agg = "mean" if y_type == "Q" else None
        elif chart.mark == "boxplot":
            chart.x_agg = None
            chart.y_agg = None
        elif chart.mark == "bar":
            if x_type == "Q" and not chart.y:
                chart.x_agg = "bin"
                chart.y_agg = None
            else:
                chart.x_agg = None
                chart.y_agg = "mean" if y_type == "Q" else "count"
        return chart
