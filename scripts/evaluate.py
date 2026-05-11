from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.core.a3c_recommender import A3CDashboardRecommender
from dashbot.core.recommender import GreedyDashboardRecommender
from scripts.train import default_data_dir, default_manifest, selected_csv_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DashBot recommendations on one CSV or the Vega manifest.")
    parser.add_argument("csv_path", nargs="?", type=Path, help="Optional single CSV path.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--manifest", type=Path, default=default_manifest())
    parser.add_argument("--mode", choices=["a3c", "greedy"], default="a3c")
    parser.add_argument("--max-charts", type=int, default=5)
    parser.add_argument("--search-steps", type=int, default=1000)
    parser.add_argument("--output-md", type=Path, default=ROOT / "reports" / "experiment_results.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "reports" / "experiment_results.csv")
    args = parser.parse_args()

    paths = [args.csv_path] if args.csv_path else selected_csv_paths(args.data_dir, args.manifest)
    paths = [path for path in paths if path and path.exists()]
    if not paths:
        raise SystemExit("No CSV files found for evaluation.")

    recommender = build_recommender(args.mode, args.max_charts, args.search_steps)
    rows = []
    for csv_path in paths:
        row = evaluate_one(csv_path, recommender, args.mode)
        rows.append(row)
        print(
            f"{row['dataset']}: reward={row['reward']:.4f} charts={row['charts']} "
            f"insights={row['insights']} runtime={row['runtime_sec']:.2f}s"
        )

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args)
    print(f"csv={args.output_csv}")
    print(f"markdown={args.output_md}")


def build_recommender(mode: str, max_charts: int, search_steps: int):
    if mode == "greedy":
        return GreedyDashboardRecommender(max_charts=max_charts)
    return A3CDashboardRecommender(max_charts=max_charts, search_steps=search_steps)


def evaluate_one(csv_path: Path, recommender: Any, mode: str) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    started = time.perf_counter()
    result = recommender.recommend(frame)
    runtime = time.perf_counter() - started
    chart_types = sorted({chart["mark"] for chart in result["charts"]})
    return {
        "dataset": csv_path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "method": result.get("method") or mode,
        "model_loaded": result.get("model_loaded"),
        "search_steps": result.get("search_steps"),
        "key_column": result.get("key_column"),
        "reward": float(result["reward"]),
        "charts": len(result["charts"]),
        "chart_types": len(chart_types),
        "chart_type_names": ", ".join(chart_types),
        "insights": len(result["insights"]),
        "runtime_sec": runtime,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    average_reward = sum(row["reward"] for row in rows) / len(rows)
    average_charts = sum(row["charts"] for row in rows) / len(rows)
    average_types = sum(row["chart_types"] for row in rows) / len(rows)
    average_runtime = sum(row["runtime_sec"] for row in rows) / len(rows)
    model_loaded = rows[0].get("model_loaded")

    lines = [
        "# DashBot Experimental Results",
        "",
        "## Setup",
        "",
        f"- Method: `{args.mode}`",
        f"- Datasets: `{len(rows)}` Vega CSV files",
        f"- Max charts: `{args.max_charts}`",
        f"- Search steps per dataset: `{args.search_steps if args.mode == 'a3c' else 'N/A'}`",
        f"- Model checkpoint loaded: `{model_loaded}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Average return | {average_reward:.4f} |",
        f"| Average charts/dashboard | {average_charts:.2f} |",
        f"| Average chart types/dashboard | {average_types:.2f} |",
        f"| Average runtime/dataset (sec) | {average_runtime:.2f} |",
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Rows | Cols | Key column | Return | Charts | Chart types | Insights | Runtime (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['rows']} | {row['columns']} | {row['key_column']} | "
            f"{row['reward']:.4f} | {row['charts']} | {row['chart_types']} | "
            f"{row['insights']} | {row['runtime_sec']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This table is generated by the local reimplementation, not copied from the paper.",
            "- For the closest paper reproduction, run `scripts/train_a3c.py --steps 500000` first, then rerun this evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
