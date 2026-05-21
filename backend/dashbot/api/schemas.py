from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChartResponse(BaseModel):
    mark: str
    x: str
    y: str | None = None
    color: str | None = None
    x_agg: str | None = None
    y_agg: str | None = None
    color_agg: str | None = None
    insight_type: str | None = None
    title: str | None = None
    vega_lite: dict[str, Any]


class RecommendationResponse(BaseModel):
    method: str | None = None
    model_loaded: bool | None = None
    search_steps: int | None = None
    key_column: str | None
    reward: float
    profile: dict[str, Any]
    charts: list[ChartResponse]
    recommendations: list[ChartResponse] = Field(default_factory=list)
    insights: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class RecommendOptions(BaseModel):
    max_charts: int = Field(default=5, ge=1, le=8)
