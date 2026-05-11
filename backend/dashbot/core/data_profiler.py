from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from dashbot.core.models import ColumnProfile, ColumnType, DatasetProfile


class DataProfiler:
    """Build VizML-like handcrafted features for tabular columns."""

    def __init__(self, max_modeled_columns: int = 10, temporal_parse_threshold: float = 0.8) -> None:
        self.max_modeled_columns = max_modeled_columns
        self.temporal_parse_threshold = temporal_parse_threshold

    def profile(self, frame: pd.DataFrame) -> DatasetProfile:
        columns = [
            self._profile_column(frame[column], index)
            for index, column in enumerate(frame.columns[: self.max_modeled_columns])
        ]
        return DatasetProfile(
            row_count=int(len(frame)),
            columns=columns,
            max_modeled_columns=self.max_modeled_columns,
        )

    def _profile_column(self, series: pd.Series, index: int) -> ColumnProfile:
        clean = series.dropna()
        column_type = self._infer_type(clean)
        missing_ratio = 0.0 if len(series) == 0 else float(series.isna().mean())
        cardinality = int(clean.nunique(dropna=True))
        unique_ratio = 0.0 if len(clean) == 0 else float(cardinality / len(clean))
        entropy = self._entropy(clean)
        gini = self._gini_impurity(clean)

        numeric_values = self._numeric_values(clean) if column_type == "Q" else pd.Series(dtype=float)
        stats = self._numeric_stats(numeric_values)
        return ColumnProfile(
            name=str(series.name),
            type=column_type,
            index=index,
            missing_ratio=missing_ratio,
            cardinality=cardinality,
            unique_ratio=unique_ratio,
            entropy=entropy,
            gini=gini,
            **stats,
        )

    def _infer_type(self, clean: pd.Series) -> ColumnType:
        if clean.empty:
            return "N"

        lowered_name = str(clean.name).lower()
        looks_temporal_name = any(token in lowered_name for token in ["date", "time", "year", "month", "day"])
        if looks_temporal_name:
            return "T"

        if pd.api.types.is_numeric_dtype(clean):
            return "Q"

        numeric = pd.to_numeric(clean, errors="coerce")
        if float(numeric.notna().mean()) >= 0.9:
            return "Q"

        if pd.api.types.is_datetime64_any_dtype(clean):
            return "T"

        sample = clean.astype(str).head(25)
        has_date_tokens = sample.str.contains(
            r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
            case=False,
            regex=True,
        ).any()
        if not has_date_tokens:
            return "N"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            parsed_dates = pd.to_datetime(clean, errors="coerce")
        if float(parsed_dates.notna().mean()) >= self.temporal_parse_threshold:
            return "T"

        return "N"

    @staticmethod
    def _numeric_values(clean: pd.Series) -> pd.Series:
        return pd.to_numeric(clean, errors="coerce").dropna().astype(float)

    @staticmethod
    def _numeric_stats(values: pd.Series) -> dict[str, float | None]:
        if values.empty:
            return {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "std": None,
                "skewness": None,
            }
        std = float(values.std(ddof=0))
        if len(values) < 3 or math.isclose(std, 0.0):
            skewness = 0.0
        else:
            centered = values - float(values.mean())
            skewness = float(np.mean((centered / std) ** 3))
        return {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "std": std,
            "skewness": skewness,
        }

    @staticmethod
    def _entropy(clean: pd.Series) -> float:
        if clean.empty:
            return 0.0
        probabilities = clean.astype(str).value_counts(normalize=True).to_numpy()
        return float(-(probabilities * np.log2(probabilities + 1e-12)).sum())

    @staticmethod
    def _gini_impurity(clean: pd.Series) -> float:
        if clean.empty:
            return 0.0
        probabilities = clean.astype(str).value_counts(normalize=True).to_numpy()
        return float(1.0 - np.square(probabilities).sum())
