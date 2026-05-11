from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.rl_env.dashboard_env import DashboardEnv


def test_environment_add_chart_returns_reward() -> None:
    frame = pd.DataFrame(
        {
            "origin": ["USA", "USA", "Japan", "Japan"],
            "horsepower": [100, 120, 80, 90],
            "mpg": [20, 18, 32, 30],
        }
    )
    env = DashboardEnv(frame)
    env.reset("origin")
    chart = env.random_valid_chart()
    state, reward, done, info = env.step("add", {"chart": chart})
    assert not info["invalid"]
    assert len(state.charts) == 1
    assert isinstance(reward, float)
    assert done is False
