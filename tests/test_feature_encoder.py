from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

pytest.importorskip("torch")

from dashbot.agent.feature_encoder import StateFeatureEncoder
from dashbot.core.data_profiler import DataProfiler
from dashbot.core.models import ChartSpec, DashboardState


def test_state_feature_encoder_shape_is_stable() -> None:
    frame = pd.DataFrame(
        {
            "origin": ["USA", "Japan", "Europe"],
            "horsepower": [100, 80, 70],
            "mpg": [20, 32, 29],
        }
    )
    profile = DataProfiler().profile(frame)
    state = DashboardState(
        key_column="origin",
        charts=[ChartSpec("bar", x="origin", y="mpg", y_agg="mean")],
    )
    encoder = StateFeatureEncoder()
    encoded = encoder.encode(state, profile)
    assert encoded.shape == (encoder.config.max_charts, encoder.feature_size)
    assert encoded.sum().item() > 0
