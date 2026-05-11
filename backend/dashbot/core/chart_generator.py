from __future__ import annotations

from typing import Any

from dashbot.core.models import ChartSpec, DatasetProfile


class ChartGenerator:
    """Convert internal chart objects into Vega-Lite-compatible specs."""

    def to_vega_lite(
        self,
        chart: ChartSpec,
        data_name: str = "table",
        profile: DatasetProfile | None = None,
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"name": data_name},
            "mark": self._mark(chart),
            "encoding": {},
        }

        spec["encoding"]["x"] = self._channel(chart.x, chart.x_agg, profile)
        if chart.y:
            spec["encoding"]["y"] = self._channel(chart.y, chart.y_agg, profile)
        if chart.color:
            spec["encoding"]["color"] = self._channel(chart.color, chart.color_agg, profile)
        if chart.title:
            spec["title"] = chart.title
        return spec

    @staticmethod
    def _mark(chart: ChartSpec) -> dict[str, Any] | str:
        if chart.mark == "point":
            return {"type": "point", "tooltip": True}
        if chart.mark == "bar":
            return {"type": "bar", "tooltip": True}
        if chart.mark == "line":
            return {"type": "line", "point": True, "tooltip": True}
        if chart.mark == "boxplot":
            return "boxplot"
        return chart.mark

    @staticmethod
    def _channel(
        field: str,
        aggregate: str | None = None,
        profile: DatasetProfile | None = None,
    ) -> dict[str, Any]:
        channel: dict[str, Any] = {"field": field}
        if profile:
            column = profile.by_name().get(field)
            if column:
                channel["type"] = {"Q": "quantitative", "N": "nominal", "T": "temporal"}[column.type]
        if aggregate and aggregate != "none":
            if aggregate == "bin":
                channel["bin"] = True
            else:
                channel["aggregate"] = aggregate
        return channel
