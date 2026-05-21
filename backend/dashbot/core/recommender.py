from __future__ import annotations

import itertools
from collections.abc import Iterable

import pandas as pd

from dashbot.core.chart_generator import ChartGenerator
from dashbot.core.data_profiler import DataProfiler
from dashbot.core.insight_detector import InsightDetector
from dashbot.core.models import ChartSpec, DatasetProfile
from dashbot.rl_env.rewards import RewardEngine


class GreedyDashboardRecommender:
    """A constrained non-RL baseline useful before A3C training finishes."""

    def __init__(
        self,
        profiler: DataProfiler | None = None,
        reward_engine: RewardEngine | None = None,
        chart_generator: ChartGenerator | None = None,
        max_charts: int = 5,
    ) -> None:
        self.profiler = profiler or DataProfiler()
        self.reward_engine = reward_engine or RewardEngine()
        self.chart_generator = chart_generator or ChartGenerator()
        self.max_charts = max_charts

    def recommend(self, frame: pd.DataFrame) -> dict:
        profile = self.profiler.profile(frame)
        candidates = self._candidate_charts(profile)
        selected: list[ChartSpec] = []

        while len(selected) < self.max_charts and candidates:
            best_chart = max(
                candidates,
                key=lambda chart: self.reward_engine.dashboard_reward(frame, profile, selected + [chart]),
            )
            best_reward = self.reward_engine.dashboard_reward(frame, profile, selected + [best_chart])
            current_reward = self.reward_engine.dashboard_reward(frame, profile, selected)
            if best_reward <= current_reward and selected:
                break
            selected.append(best_chart)
            candidates.remove(best_chart)

        insights = InsightDetector().detect_dashboard(frame, profile, selected)
        key_column = self._choose_key_column(profile, selected)
        recommendations = self._recommendation_charts(frame, profile, selected, candidates)
        return {
            "key_column": key_column,
            "reward": self.reward_engine.dashboard_reward(frame, profile, selected),
            "profile": profile.to_dict(),
            "charts": [self._chart_response(chart, profile) for chart in selected],
            "recommendations": [self._chart_response(chart, profile) for chart in recommendations],
            "insights": [insight.to_dict() for insight in insights],
        }

    @staticmethod
    def _choose_key_column(profile: DatasetProfile, charts: list[ChartSpec]) -> str | None:
        if not charts:
            return profile.columns[0].name if profile.columns else None
        field_counts: dict[str, int] = {}
        for chart in charts:
            for field in chart.fields():
                field_counts[field] = field_counts.get(field, 0) + 1
        return max(field_counts, key=field_counts.get)

    @staticmethod
    def _candidate_charts(profile: DatasetProfile) -> list[ChartSpec]:
        quantitative = [column.name for column in profile.modeled_columns() if column.type == "Q"]
        nominal = [
            column.name
            for column in profile.modeled_columns()
            if column.type == "N" and column.unique_ratio <= 0.6 and column.cardinality <= 20
        ]
        temporal = [column.name for column in profile.modeled_columns() if column.type == "T"]

        charts: list[ChartSpec] = []
        for q in quantitative:
            charts.append(ChartSpec("bar", x=q, x_agg="bin", title=f"Distribution of {q}"))

        for n, q in itertools.product(nominal, quantitative):
            charts.append(ChartSpec("bar", x=n, y=q, y_agg="mean", title=f"Average {q} by {n}"))

        for x, y in itertools.permutations(quantitative, 2):
            charts.append(ChartSpec("point", x=x, y=y, title=f"{y} vs {x}"))

        for t, q in itertools.product(temporal, quantitative):
            charts.append(ChartSpec("line", x=t, y=q, y_agg="mean", title=f"{q} over {t}"))

        for n, q in itertools.product(nominal, quantitative):
            charts.append(ChartSpec("boxplot", x=n, y=q, title=f"{q} distribution by {n}"))

        return charts

    def _recommendation_charts(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        selected: list[ChartSpec],
        candidates: list[ChartSpec],
        limit: int = 4,
    ) -> list[ChartSpec]:
        used = {self._chart_signature(chart) for chart in selected}
        selected_marks = {chart.mark for chart in selected}
        selected_fields = {field for chart in selected for field in chart.fields()}
        unique_candidates = self._dedupe_charts(chart for chart in candidates if self._chart_signature(chart) not in used)
        unique_candidates.sort(
            key=lambda chart: self._candidate_score(frame, profile, selected, chart, selected_marks, selected_fields),
            reverse=True,
        )
        return unique_candidates[:limit]

    def _candidate_score(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        selected: list[ChartSpec],
        chart: ChartSpec,
        selected_marks: set[str],
        selected_fields: set[str],
    ) -> float:
        reward = self.reward_engine.dashboard_reward(frame, profile, selected + [chart])
        type_bonus = 0.4 if chart.mark not in selected_marks else 0.0
        field_bonus = 0.05 * len(set(chart.fields()) - selected_fields)
        insight_bonus = 0.08 * len(InsightDetector().detect_for_chart(frame, profile, chart))
        return reward + type_bonus + field_bonus + insight_bonus

    def _chart_response(self, chart: ChartSpec, profile: DatasetProfile) -> dict:
        return {
            **chart.to_dict(),
            "vega_lite": self.chart_generator.to_vega_lite(chart, profile=profile),
        }

    @classmethod
    def _dedupe_charts(cls, charts: Iterable[ChartSpec]) -> list[ChartSpec]:
        selected: list[ChartSpec] = []
        used = set()
        for chart in charts:
            signature = cls._chart_signature(chart)
            if signature in used:
                continue
            selected.append(chart)
            used.add(signature)
        return selected

    @staticmethod
    def _chart_signature(chart: ChartSpec) -> tuple[str, str, str | None, str | None, str | None]:
        return (
            chart.mark,
            chart.x,
            chart.y,
            chart.x_agg,
            chart.y_agg,
        )
