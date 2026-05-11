from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

from dashbot.core.models import ActionType, ChartSpec, DatasetProfile, MarkType


VALID_AGGREGATES = {
    "Q": ["none", "mean", "max", "min", "count", "bin"],
    "N": ["none", "count"],
    "T": ["none", "count", "bin"],
}

T = TypeVar("T")


@dataclass(frozen=True)
class ConstraintMasks:
    actions: dict[ActionType, bool]
    marks: dict[MarkType, bool]
    columns: dict[str, bool]
    aggregates: dict[str, bool]


class ConstraintSampler:
    """Rule masks for constrained sampling before softmax."""

    action_space: tuple[ActionType, ...] = ("change", "add", "remove", "terminate")
    mark_space: tuple[MarkType, ...] = ("bar", "line", "point", "boxplot")

    def __init__(self, max_charts: int = 8) -> None:
        self.max_charts = max_charts

    def masks(
        self,
        profile: DatasetProfile,
        charts: list[ChartSpec],
        key_column: str | None,
        selected_field: str | None = None,
    ) -> ConstraintMasks:
        return ConstraintMasks(
            actions=self.action_mask(charts),
            marks={mark: True for mark in self.mark_space},
            columns=self.column_mask(profile, key_column),
            aggregates=self.aggregate_mask(profile, selected_field),
        )

    def action_mask(self, charts: list[ChartSpec]) -> dict[ActionType, bool]:
        return {
            "change": True,
            "add": len(charts) < self.max_charts,
            "remove": len(charts) > 0,
            "terminate": len(charts) > 0,
        }

    @staticmethod
    def column_mask(profile: DatasetProfile, key_column: str | None = None) -> dict[str, bool]:
        return {column.name: column.name != key_column for column in profile.modeled_columns()}

    @staticmethod
    def aggregate_mask(profile: DatasetProfile, selected_field: str | None) -> dict[str, bool]:
        by_name = profile.by_name()
        field_type = by_name[selected_field].type if selected_field in by_name else "Q"
        valid = set(VALID_AGGREGATES[field_type])
        return {aggregate: aggregate in valid for aggregate in ["none", "mean", "max", "min", "count", "bin"]}

    def valid_marks_for_fields(
        self,
        profile: DatasetProfile,
        x_field: str,
        y_field: str | None = None,
    ) -> dict[MarkType, bool]:
        by_name = profile.by_name()
        x_type = by_name[x_field].type if x_field in by_name else "N"
        y_type = by_name[y_field].type if y_field and y_field in by_name else None
        return {
            "bar": True,
            "line": x_type == "T" and y_type == "Q",
            "point": x_type == "Q" and y_type == "Q",
            "boxplot": y_type == "Q",
        }

    @staticmethod
    def filter_valid(items: Iterable[T], mask: dict[T, bool]) -> list[T]:
        return [item for item in items if mask.get(item, False)]
