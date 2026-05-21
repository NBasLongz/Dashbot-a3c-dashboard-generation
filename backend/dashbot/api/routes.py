from __future__ import annotations

import io
from threading import Lock
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from dashbot import __version__
from dashbot.api.schemas import HealthResponse, RecommendationResponse
from dashbot.core.a3c_recommender import A3CDashboardRecommender
from dashbot.core.data_profiler import DataProfiler
from dashbot.core.recommender import GreedyDashboardRecommender

router = APIRouter(prefix="/api")
JOBS: dict[str, dict] = {}
JOB_LOCK = Lock()


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


@router.post("/recommend/jobs")
async def start_recommend_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_charts: int = Query(default=5, ge=1, le=8),
    mode: str = Query(default="a3c", pattern="^(a3c|greedy)$"),
    search_steps: int = Query(default=1000, ge=1, le=5000),
) -> dict:
    frame = await _read_csv(file)
    job_id = uuid4().hex
    _update_job(job_id, status="running", progress=5, message="Queued dashboard generation")
    background_tasks.add_task(_run_recommend_job, job_id, frame, max_charts, mode, search_steps)
    return _get_job(job_id, include_result=False)


@router.get("/recommend/jobs/{job_id}")
async def get_recommend_job(job_id: str) -> dict:
    return _get_job(job_id, include_result=True)


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}") from exc


def _run_recommend_job(job_id: str, frame: pd.DataFrame, max_charts: int, mode: str, search_steps: int) -> None:
    try:
        if mode == "greedy":
            _update_job(job_id, progress=40, message="Scoring candidate charts")
            result = GreedyDashboardRecommender(max_charts=max_charts).recommend(frame)
        else:
            def progress_callback(steps_used: int, total_steps: int, message: str) -> None:
                rollout_progress = 15 + int(80 * min(steps_used, total_steps) / max(total_steps, 1))
                _update_job(
                    job_id,
                    progress=min(95, rollout_progress),
                    message=f"A3C {steps_used}/{total_steps}",
                    steps_used=steps_used,
                    total_steps=total_steps,
                )

            recommender = A3CDashboardRecommender(
                max_charts=max_charts,
                search_steps=search_steps,
                progress_callback=progress_callback,
            )
            result = recommender.recommend(frame)
        _update_job(job_id, status="completed", progress=100, message="Dashboard ready", result=result)
    except Exception as exc:
        _update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))


def _update_job(job_id: str, **updates: object) -> None:
    with JOB_LOCK:
        current = JOBS.setdefault(job_id, {"job_id": job_id})
        current.update(updates)


def _get_job(job_id: str, include_result: bool) -> dict:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Recommendation job not found.")
        payload = dict(job)
    if not include_result:
        payload.pop("result", None)
    return payload
