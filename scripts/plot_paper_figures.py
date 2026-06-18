from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-style DashBot experiment figures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    curve = subparsers.add_parser("learning-curve", help="Plot Fig. 6-style training return curve from CSV logs.")
    curve.add_argument("--dashbot-log", type=Path, default=ROOT / "reports" / "training_curve_a3c.csv")
    curve.add_argument("--dashbot-ind-log", type=Path)
    curve.add_argument("--dashbot-pen-log", type=Path)
    curve.add_argument("--dqn-log", type=Path)
    curve.add_argument("--dashbot-logs", nargs="*", type=Path)
    curve.add_argument("--dashbot-ind-logs", nargs="*", type=Path)
    curve.add_argument("--dashbot-pen-logs", nargs="*", type=Path)
    curve.add_argument("--dqn-logs", nargs="*", type=Path)
    curve.add_argument("--output", type=Path, default=ROOT / "reports" / "fig6_learning_curve.png")

    args = parser.parse_args()
    if args.command == "learning-curve":
        plot_learning_curve(args)



def plot_learning_curve(args: argparse.Namespace) -> None:
    series = [
        ("DashBot", collect_paths(args.dashbot_log, args.dashbot_logs), "#1f77b4"),
        ("DashBot-ind.", collect_paths(args.dashbot_ind_log, args.dashbot_ind_logs), "#ff7f0e"),
        ("DashBot-pen.", collect_paths(args.dashbot_pen_log, args.dashbot_pen_logs), "#2ca02c"),
        ("DQN", collect_paths(args.dqn_log, args.dqn_logs), "#d62728"),
    ]
    plt.figure(figsize=(7.5, 4.2))
    plotted = False
    for label, paths, color in series:
        frames = read_training_logs(paths)
        if not frames:
            continue
        summary = summarize_training_logs(frames)
        plt.plot(summary["step"], summary["mean"], label=label, color=color, linewidth=1.5)
        plt.fill_between(
            summary["step"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=color,
            alpha=0.18,
        )
        plotted = True

    if not plotted:
        raise SystemExit("No valid training logs found. Train with --log-csv first.")

    plt.xlabel("training steps")
    plt.ylabel("mean return")
    plt.legend(frameon=False, loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    print(f"saved={args.output}")


def collect_paths(single_path: Path | None, extra_paths: list[Path] | None) -> list[Path]:
    paths: list[Path] = []
    if extra_paths:
        paths.extend(extra_paths)
    elif single_path:
        paths.append(single_path)
    return paths


def read_training_logs(paths: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path or not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "step" not in frame or "return" not in frame:
            continue
        frame = frame[["step", "return"]].dropna().sort_values("step")
        if frame.empty:
            continue
        frame["return"] = frame["return"].rolling(window=10, min_periods=1).mean()
        frames.append(frame)
    return frames


def summarize_training_logs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        frame = frames[0].copy()
        frame["mean"] = frame["return"]
        frame["std"] = frame["return"].rolling(window=10, min_periods=1).std().fillna(0.0)
        return frame[["step", "mean", "std"]]

    stacked = []
    for run_id, frame in enumerate(frames):
        copy = frame.copy()
        copy["run_id"] = run_id
        stacked.append(copy)
    combined = pd.concat(stacked, ignore_index=True)
    summary = combined.groupby("step")["return"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    return summary





if __name__ == "__main__":
    main()
