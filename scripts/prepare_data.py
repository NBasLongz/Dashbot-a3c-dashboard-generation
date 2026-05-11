from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import urllib.request


VEGA_BASE_URL = "https://raw.githubusercontent.com/vega/vega-datasets/main/data"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Clean Vega datasets for DashBot training.")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "raw" / "vega_27_manifest.txt")
    parser.add_argument("--max-columns", type=int, default=10)
    parser.add_argument("--max-missing-ratio", type=float, default=0.45)
    parser.add_argument("--max-rows", type=int, default=10000)
    args = parser.parse_args()

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    names = read_manifest(args.manifest)
    processed_names: list[str] = []
    summary_rows: list[dict[str, object]] = []

    for name in names:
        raw_path = args.raw_dir / name
        if not raw_path.exists():
            try:
                download_raw_csv(name, raw_path)
            except Exception as exc:
                print(f"skip={name} reason=missing_and_download_failed:{exc}")
                continue
        try:
            raw = pd.read_csv(raw_path)
            cleaned = clean_frame(
                raw,
                max_columns=args.max_columns,
                max_missing_ratio=args.max_missing_ratio,
                max_rows=args.max_rows,
            )
        except Exception as exc:
            print(f"skip={name} reason={exc}")
            continue

        if cleaned.empty or cleaned.shape[1] < 2:
            print(f"skip={name} reason=not_enough_tabular_signal")
            continue

        out_name = name
        out_path = args.processed_dir / out_name
        cleaned.to_csv(out_path, index=False)
        processed_names.append(out_name)
        summary_rows.append(
            {
                "dataset": out_name,
                "raw_rows": raw.shape[0],
                "raw_cols": raw.shape[1],
                "processed_rows": cleaned.shape[0],
                "processed_cols": cleaned.shape[1],
            }
        )
        print(
            f"processed {name}: {raw.shape[0]}x{raw.shape[1]} -> "
            f"{cleaned.shape[0]}x{cleaned.shape[1]}"
        )

    (args.processed_dir / "vega_27_manifest.txt").write_text(
        "\n".join(processed_names) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(args.processed_dir / "summary.csv", index=False)
    print(f"manifest={args.processed_dir / 'vega_27_manifest.txt'} count={len(processed_names)}")
    print(f"summary={args.processed_dir / 'summary.csv'}")


def read_manifest(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def download_raw_csv(name: str, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    csv_url = f"{VEGA_BASE_URL}/{name}"
    try:
        urllib.request.urlretrieve(csv_url, raw_path)
        print(f"downloaded {name}")
        return
    except Exception:
        pass

    json_name = raw_path.with_suffix(".json").name
    json_url = f"{VEGA_BASE_URL}/{json_name}"
    with urllib.request.urlopen(json_url) as response:
        payload = json.load(response)
    frame = pd.DataFrame(payload)
    frame.to_csv(raw_path, index=False)
    print(f"downloaded {json_name} and converted to {name}")


def clean_frame(
    frame: pd.DataFrame,
    max_columns: int,
    max_missing_ratio: float,
    max_rows: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = dedupe_columns([normalize_column_name(column) for column in frame.columns])
    frame = frame.dropna(axis=1, how="all")

    missing_ratio = frame.isna().mean()
    frame = frame.loc[:, missing_ratio <= max_missing_ratio]

    for column in frame.columns:
        frame[column] = coerce_column(frame[column])

    frame = keep_informative_columns(frame)
    frame = choose_columns(frame, max_columns=max_columns)

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].fillna(frame[column].median())
        else:
            mode = frame[column].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "unknown"
            frame[column] = frame[column].fillna(fill).astype(str)

    frame = frame.drop_duplicates()
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=7).sort_index()
    return frame.reset_index(drop=True)


def normalize_column_name(column: object) -> str:
    name = str(column).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def dedupe_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for column in columns:
        count = seen.get(column, 0)
        seen[column] = count + 1
        result.append(column if count == 0 else f"{column}_{count + 1}")
    return result


def coerce_column(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.85:
            return numeric
    return series


def keep_informative_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    row_count = max(len(frame), 1)
    for column in frame.columns:
        unique = frame[column].nunique(dropna=True)
        if unique <= 1:
            continue
        if unique == row_count and not pd.api.types.is_numeric_dtype(frame[column]) and not looks_temporal(column):
            continue
        keep.append(column)
    return frame.loc[:, keep]


def choose_columns(frame: pd.DataFrame, max_columns: int) -> pd.DataFrame:
    scored = []
    for column in frame.columns:
        score = column_score(frame[column], column)
        scored.append((score, column))
    selected = [column for _, column in sorted(scored, reverse=True)[:max_columns]]
    selected_in_original_order = [column for column in frame.columns if column in selected]
    return frame.loc[:, selected_in_original_order]


def column_score(series: pd.Series, column: str) -> float:
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    missing_penalty = float(series.isna().mean())
    score = 1.0 - missing_penalty
    if pd.api.types.is_numeric_dtype(series):
        score += 2.0
    elif looks_temporal(column):
        score += 1.5
    elif unique_ratio <= 0.6:
        score += 1.0
    else:
        score -= 1.0
    if looks_temporal(column):
        score += 0.5
    return score


def looks_temporal(column: str) -> bool:
    return bool(re.search(r"(date|time|year|month|day)", column, re.IGNORECASE))


if __name__ == "__main__":
    main()
