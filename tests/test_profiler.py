from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.core.data_profiler import DataProfiler


def test_profiler_infers_basic_types() -> None:
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.5],
            "category": ["a", "b", "a"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    profile = DataProfiler().profile(frame)
    by_name = profile.by_name()
    assert by_name["value"].type == "Q"
    assert by_name["category"].type == "N"
    assert by_name["date"].type == "T"
