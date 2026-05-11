from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from dashbot import __version__
from dashbot.api.schemas import HealthResponse, RecommendationResponse
from dashbot.core.a3c_recommender import A3CDashboardRecommender
from dashbot.core.data_profiler import DataProfiler
from dashbot.core.recommender import GreedyDashboardRecommender

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@router.post("/profile")
async def profile_csv(file: UploadFile = File(...)) -> dict:
    frame = await _read_csv(file)
    return DataProfiler().profile(frame).to_dict()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    file: UploadFile = File(...),
    max_charts: int = Query(default=5, ge=1, le=8),
    mode: str = Query(default="a3c", pattern="^(a3c|greedy)$"),
    search_steps: int = Query(default=1000, ge=1, le=5000),
) -> dict:
    frame = await _read_csv(file)
    if mode == "greedy":
        recommender = GreedyDashboardRecommender(max_charts=max_charts)
    else:
        recommender = A3CDashboardRecommender(max_charts=max_charts, search_steps=search_steps)
    return recommender.recommend(frame)


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc
