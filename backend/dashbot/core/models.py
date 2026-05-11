from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ColumnType = Literal["Q", "N", "T"]
MarkType = Literal["bar", "line", "point", "boxplot"]
ActionType = Literal["change", "add", "remove", "terminate"]


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: ColumnType
    index: int
    missing_ratio: float
    cardinality: int
    unique_ratio: float
    entropy: float
    gini: float
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    skewness: float | None = None

    def feature_vector(self) -> list[float]:
        type_one_hot = {
            "Q": [1.0, 0.0, 0.0],
            "N": [0.0, 1.0, 0.0],
            "T": [0.0, 0.0, 1.0],
        }[self.type]
        numeric = [
            self.missing_ratio,
            float(self.cardinality),
            self.unique_ratio,
            self.entropy,
            self.gini,
            self.minimum or 0.0,
            self.maximum or 0.0,
            self.mean or 0.0,
            self.std or 0.0,
            self.skewness or 0.0,
        ]
        return type_one_hot + numeric

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    columns: list[ColumnProfile]
    max_modeled_columns: int = 10

    def by_name(self) -> dict[str, ColumnProfile]:
        return {column.name: column for column in self.columns}

    def modeled_columns(self) -> list[ColumnProfile]:
        return self.columns[: self.max_modeled_columns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "max_modeled_columns": self.max_modeled_columns,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass
class ChartSpec:
    mark: MarkType
    x: str
    y: str | None = None
    color: str | None = None
    x_agg: str | None = None
    y_agg: str | None = None
    color_agg: str | None = None
    insight_type: str | None = None
    title: str | None = None

    def fields(self) -> list[str]:
        return [field for field in [self.x, self.y, self.color] if field]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Insight:
    type: str
    columns: tuple[str, ...]
    score: float
    description: str
    arity: int

    @property
    def reward(self) -> int:
        return min(max(self.arity, 1), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "columns": list(self.columns),
            "score": self.score,
            "description": self.description,
            "arity": self.arity,
            "reward": self.reward,
        }


@dataclass
class DashboardState:
    key_column: str | None = None
    charts: list[ChartSpec] = field(default_factory=list)
    reward: float = 0.0

    def copy(self) -> "DashboardState":
        return DashboardState(
            key_column=self.key_column,
            charts=[ChartSpec(**chart.to_dict()) for chart in self.charts],
            reward=self.reward,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_column": self.key_column,
            "reward": self.reward,
            "charts": [chart.to_dict() for chart in self.charts],
        }
