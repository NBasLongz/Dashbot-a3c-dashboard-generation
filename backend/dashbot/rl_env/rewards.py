from __future__ import annotations

import math

import pandas as pd

from dashbot.core.insight_detector import InsightDetector
from dashbot.core.models import ChartSpec, DatasetProfile


class RewardEngine:
    """Presentation and insight rewards from the DashBot paper."""

    def __init__(
        self,
        insight_detector: InsightDetector | None = None,
        alpha: float = 3.0,
        n_best: int = 4,
        n_max: int = 8,
        w_diversity: float = 0.33,
        w_parsimony: float = 0.33,
        w_insight: float = 0.1,
        total_chart_types: int = 4,
    ) -> None:
        self.insight_detector = insight_detector or InsightDetector()
        self.alpha = alpha
        self.n_best = n_best
        self.n_max = n_max
        self.w_diversity = w_diversity
        self.w_parsimony = w_parsimony
        self.w_insight = w_insight
        self.total_chart_types = total_chart_types

    def dashboard_reward(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        charts: list[ChartSpec],
    ) -> float:
        diversity = self.chart_type_diversity(charts) + self.column_diversity(charts, profile)
        parsimony = self.parsimony(len(charts))
        insight_reward = self.insight_reward(frame, profile, charts)
        return (
            self.w_diversity * diversity
            + self.w_parsimony * parsimony_clamped(parsimony)
            + self.w_insight * insight_reward
        )

    def immediate_reward(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        previous_charts: list[ChartSpec],
        current_charts: list[ChartSpec],
    ) -> float:
        return self.dashboard_reward(frame, profile, current_charts) - self.dashboard_reward(
            frame, profile, previous_charts
        )

    def chart_type_diversity(self, charts: list[ChartSpec]) -> float:
        used = len({chart.mark for chart in charts})
        return self._diminishing_reward(used, self.total_chart_types)

    def column_diversity(self, charts: list[ChartSpec], profile: DatasetProfile) -> float:
        used = len({field for chart in charts for field in chart.fields()})
        return self._diminishing_reward(used, max(len(profile.columns), 1))

    def parsimony(self, chart_count: int) -> float:
        n = min(max(chart_count, 0), self.n_max)
        if n <= self.n_best:
            return math.sin((math.pi / 2.0) * n / self.n_best)
        return math.sin((math.pi / 2.0) * (1.0 + (n - self.n_best) / (self.n_max - self.n_best)))

    def insight_reward(self, frame: pd.DataFrame, profile: DatasetProfile, charts: list[ChartSpec]) -> float:
        insights = self.insight_detector.detect_dashboard(frame, profile, charts)
        return float(sum(insight.reward for insight in insights))

    def _diminishing_reward(self, used: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return float(1.0 - math.exp(-self.alpha * used / total))


def parsimony_clamped(value: float) -> float:
    return min(max(value, 0.0), 1.0)
