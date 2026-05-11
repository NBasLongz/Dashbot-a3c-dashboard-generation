from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from dashbot.core.models import ChartSpec, ColumnProfile, DashboardState, DatasetProfile


MARK_TYPES = ("bar", "line", "point", "boxplot")
AGGREGATES = ("none", "mean", "max", "min", "count", "bin")


@dataclass(frozen=True)
class FeatureEncoderConfig:
    max_columns: int = 10
    max_charts: int = 8


class StateFeatureEncoder:
    """Encode DashboardState as the chart-sequence tensor described in Section 5.2."""

    def __init__(self, config: FeatureEncoderConfig | None = None) -> None:
        self.config = config or FeatureEncoderConfig()
        self.column_feature_size = len(self._zero_column())
        self.chart_feature_size = (
            len(MARK_TYPES)
            + 3
            + 3 * self.column_feature_size
            + 3 * len(AGGREGATES)
        )
        self.context_feature_size = (1 + self.config.max_columns) * self.column_feature_size
        self.feature_size = self.chart_feature_size + self.context_feature_size

    def encode(self, state: DashboardState, profile: DatasetProfile, shuffle_charts: bool = False) -> torch.Tensor:
        rows = []
        context = self._context_features(state, profile)
        charts = state.charts[: self.config.max_charts]
        if shuffle_charts:
            charts = list(charts)
            random.shuffle(charts)
        if not charts:
            rows.append(self._zero_chart() + context)
        else:
            rows.extend(self._chart_features(chart, profile) + context for chart in charts)

        while len(rows) < self.config.max_charts:
            rows.append(self._zero_chart() + context)

        return torch.tensor(rows[: self.config.max_charts], dtype=torch.float32)

    def encode_batch(self, states: list[DashboardState], profile: DatasetProfile) -> torch.Tensor:
        return torch.stack([self.encode(state, profile) for state in states], dim=0)

    def _context_features(self, state: DashboardState, profile: DatasetProfile) -> list[float]:
        by_name = profile.by_name()
        key_features = self._column_features(by_name.get(state.key_column or ""))
        column_features = []
        for column in profile.modeled_columns()[: self.config.max_columns]:
            column_features.extend(self._column_features(column))
        while len(column_features) < self.config.max_columns * self.column_feature_size:
            column_features.extend(self._zero_column())
        return key_features + column_features

    def _chart_features(self, chart: ChartSpec, profile: DatasetProfile) -> list[float]:
        by_name = profile.by_name()
        mark = self._one_hot(chart.mark, MARK_TYPES)
        channel_usage = [1.0, float(chart.y is not None), float(chart.color is not None)]
        fields = (
            self._column_features(by_name.get(chart.x))
            + self._column_features(by_name.get(chart.y or ""))
            + self._column_features(by_name.get(chart.color or ""))
        )
        aggregates = (
            self._one_hot(chart.x_agg or "none", AGGREGATES)
            + self._one_hot(chart.y_agg or "none", AGGREGATES)
            + self._one_hot(chart.color_agg or "none", AGGREGATES)
        )
        return mark + channel_usage + fields + aggregates

    def _zero_chart(self) -> list[float]:
        return [0.0] * self.chart_feature_size

    def _zero_column(self) -> list[float]:
        return [0.0] * 13

    def _column_features(self, column: ColumnProfile | None) -> list[float]:
        if column is None:
            return self._zero_column()
        return column.feature_vector()

    @staticmethod
    def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
        return [1.0 if value == item else 0.0 for item in vocabulary]
