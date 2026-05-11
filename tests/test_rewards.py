from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.core.data_profiler import DataProfiler
from dashbot.core.models import ChartSpec
from dashbot.rl_env.rewards import RewardEngine


def test_parsimony_peaks_at_n_best() -> None:
    engine = RewardEngine(n_best=4, n_max=8)
    assert engine.parsimony(4) > engine.parsimony(2)
    assert engine.parsimony(4) > engine.parsimony(8)


def test_correlation_chart_gets_insight_reward() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    profile = DataProfiler().profile(frame)
    chart = ChartSpec("point", x="x", y="y")
    reward = RewardEngine().insight_reward(frame, profile, [chart])
    assert reward >= 2


def test_column_diversity_is_positive() -> None:
    frame = pd.DataFrame({"category": ["a", "b", "c"], "value": [1, 2, 3]})
    profile = DataProfiler().profile(frame)
    chart = ChartSpec("bar", x="category", y="value", y_agg="mean")
    assert RewardEngine().column_diversity([chart], profile) > 0
