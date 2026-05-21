from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from dashbot.core.models import ChartSpec, DatasetProfile, Insight


class InsightDetector:
    """Detect the insight categories defined in the DashBot paper."""

    def __init__(self, correlation_threshold: float = 0.5, top_k: int = 5) -> None:
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k

    def detect_for_chart(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        chart: ChartSpec,
    ) -> list[Insight]:
        column_types = profile.by_name()
        insights: list[Insight] = []

        x_profile = column_types.get(chart.x)
        y_profile = column_types.get(chart.y) if chart.y else None
        if x_profile is None:
            return insights

        if chart.mark == "bar" and x_profile.type == "Q" and chart.x_agg == "bin":
            insights.append(
                Insight("distribution", (chart.x,), 1.0, f"Distribution of {chart.x}", 1)
            )

        if chart.mark == "line" and y_profile and x_profile.type == "T" and y_profile.type == "Q":
            insights.append(Insight("trend", (chart.y or "", chart.x), 1.0, f"Trend of {chart.y} over {chart.x}", 2))

        if (
            chart.mark in {"line", "point"}
            and y_profile
            and x_profile.type == "Q"
            and y_profile.type == "Q"
        ):
            corr = self._pearson(frame, chart.x, chart.y or "")
            if abs(corr) >= self.correlation_threshold:
                insights.append(
                    Insight(
                        "correlation",
                        (chart.x, chart.y or ""),
                        abs(corr),
                        f"Correlation between {chart.x} and {chart.y}: {corr:.3f}",
                        2,
                    )
                )

        if chart.mark == "bar" and y_profile and x_profile.type == "N" and y_profile.type == "Q":
            grouped = self._group_numeric(frame, chart.x, chart.y or "")
            if len(grouped) >= min(self.top_k, 2):
                spread = self._normalized_spread(grouped.to_numpy())
                insights.append(
                    Insight(
                        "top/bottom k",
                        (chart.x, chart.y or ""),
                        spread,
                        f"Top/bottom {min(self.top_k, len(grouped))} {chart.x} by {chart.y}",
                        2,
                    )
                )

        return insights

    def detect_dashboard(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        charts: list[ChartSpec],
    ) -> list[Insight]:
        insights: list[Insight] = []
        for chart in charts:
            insights.extend(self.detect_for_chart(frame, profile, chart))

        insights.extend(self._co_correlations_from_chart_insights(insights))
        insights.extend(self._comparisons(insights))
        return insights

    def _co_correlations(self, frame: pd.DataFrame, profile: DatasetProfile) -> list[Insight]:
        """Dataset-level co-correlation helper kept for exploratory analysis."""
        quantitative = [column.name for column in profile.columns if column.type == "Q"]
        results: list[Insight] = []
        for a, b, c in itertools.combinations(quantitative, 3):
            corr_ab = abs(self._pearson(frame, a, b))
            corr_ac = abs(self._pearson(frame, a, c))
            if corr_ab >= self.correlation_threshold and corr_ac >= self.correlation_threshold:
                results.append(
                    Insight(
                        "co-correlation",
                        (a, b, c),
                        min(corr_ab, corr_ac),
                        f"{b} and {c} are both correlated with {a}",
                        3,
                    )
                )
        return results

    def _co_correlations_from_chart_insights(self, insights: list[Insight]) -> list[Insight]:
        correlations = [insight for insight in insights if insight.type == "correlation" and len(insight.columns) == 2]
        by_anchor: dict[str, list[Insight]] = {}
        for insight in correlations:
            left, right = insight.columns
            by_anchor.setdefault(left, []).append(insight)
            by_anchor.setdefault(right, []).append(insight)

        results: list[Insight] = []
        seen: set[tuple[str, str, str]] = set()
        for anchor, related in by_anchor.items():
            if len(related) < 2:
                continue
            for first, second in itertools.combinations(related, 2):
                other_first = first.columns[1] if first.columns[0] == anchor else first.columns[0]
                other_second = second.columns[1] if second.columns[0] == anchor else second.columns[0]
                if other_first == other_second:
                    continue
                columns = tuple(sorted((anchor, other_first, other_second)))
                if columns in seen:
                    continue
                seen.add(columns)
                results.append(
                    Insight(
                        "co-correlation",
                        columns,
                        min(first.score, second.score),
                        f"{other_first} and {other_second} are both correlated with {anchor}",
                        3,
                    )
                )
        return results

    @staticmethod
    def _comparisons(insights: list[Insight]) -> list[Insight]:
        results = []
        seen = set()
        for insight in insights:
            if insight.type != "top/bottom k" or insight.columns in seen:
                continue
            seen.add(insight.columns)
            results.append(
                Insight(
                    "comparison",
                    insight.columns,
                    insight.score,
                    f"Comparison insight for {insight.columns[0]} and {insight.columns[1]}",
                    3,
                )
            )
        return results

    @staticmethod
    def _pearson(frame: pd.DataFrame, left: str, right: str) -> float:
        if left == right or left not in frame.columns or right not in frame.columns:
            return 0.0
        left_values = InsightDetector._first_series(frame.loc[:, left])
        right_values = InsightDetector._first_series(frame.loc[:, right])
        pair = pd.concat([left_values, right_values], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            return 0.0
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if corr is None or np.isnan(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def _group_numeric(frame: pd.DataFrame, group_column: str, value_column: str) -> pd.Series:
        if group_column not in frame.columns or value_column not in frame.columns:
            return pd.Series(dtype=float)
        groups = InsightDetector._first_series(frame.loc[:, group_column])
        numeric = pd.to_numeric(InsightDetector._first_series(frame.loc[:, value_column]), errors="coerce")
        grouped = pd.DataFrame({"_dashbot_group": groups, "_dashbot_value": numeric}).dropna(subset=["_dashbot_value"])
        return grouped.groupby("_dashbot_group")["_dashbot_value"].mean().sort_values(ascending=False)

    @staticmethod
    def _first_series(value: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(value, pd.DataFrame):
            return value.iloc[:, 0]
        return value

    @staticmethod
    def _normalized_spread(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        denominator = abs(maximum) + abs(minimum) + 1e-9
        return float((maximum - minimum) / denominator)
