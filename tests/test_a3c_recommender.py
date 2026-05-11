import pandas as pd

from dashbot.core.a3c_recommender import A3CDashboardRecommender


def test_a3c_recommender_returns_dashboard_payload():
    frame = pd.DataFrame(
        {
            "origin": ["USA", "USA", "Japan", "Europe", "Japan", "Europe"],
            "year": [1975, 1976, 1977, 1978, 1979, 1980],
            "horsepower": [110, 95, 62, 80, 70, 88],
            "mpg": [18, 22, 34, 28, 33, 27],
        }
    )

    payload = A3CDashboardRecommender(max_charts=3, search_steps=5).recommend(frame)

    assert payload["method"] == "a3c"
    assert payload["search_steps"] == 5
    assert payload["charts"]
    assert payload["profile"]["row_count"] == len(frame)
    assert all("vega_lite" in chart for chart in payload["charts"])
