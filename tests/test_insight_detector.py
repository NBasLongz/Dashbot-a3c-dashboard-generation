import pandas as pd

from dashbot.core.data_profiler import DataProfiler
from dashbot.core.insight_detector import InsightDetector
from dashbot.core.models import ChartSpec


def test_co_correlation_requires_chart_level_correlations() -> None:
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [3, 6, 9, 12, 15],
        }
    )
    profile = DataProfiler().profile(frame)
    detector = InsightDetector(correlation_threshold=0.5)

    one_chart = detector.detect_dashboard(frame, profile, [ChartSpec("point", x="a", y="b")])
    two_charts = detector.detect_dashboard(
        frame,
        profile,
        [
            ChartSpec("point", x="a", y="b"),
            ChartSpec("point", x="a", y="c"),
        ],
    )

    assert not any(insight.type == "co-correlation" for insight in one_chart)
    assert any(insight.type == "co-correlation" for insight in two_charts)
