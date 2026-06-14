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

    study = subparsers.add_parser("user-study", help="Plot Fig. 8-style stacked user study votes.")
    study.add_argument("--input", type=Path, help="CSV with columns: metric,dashbot,neutral,baseline")
    study.add_argument("--output", type=Path, default=ROOT / "reports" / "fig8_user_study.png")
    study.add_argument("--baseline-name", default="MultiVision")

    args = parser.parse_args()
    if args.command == "learning-curve":
        plot_learning_curve(args)
    elif args.command == "user-study":
        plot_user_study(args)


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


def plot_user_study(args: argparse.Namespace) -> None:
    if args.input and args.input.exists():
        frame = pd.read_csv(args.input)
    else:
        frame = pd.DataFrame(
            [
                {"metric": "Overall Quality", "dashbot": 39, "neutral": 3, "baseline": 8},
                {"metric": "Understandability", "dashbot": 42, "neutral": 1, "baseline": 7},
                {"metric": "Aesthetic", "dashbot": 38, "neutral": 3, "baseline": 9},
                {"metric": "Insightfulness", "dashbot": 44, "neutral": 1, "baseline": 5},
            ]
        )

    metrics = frame["metric"].tolist()
    dashbot = frame["dashbot"].astype(float)
    neutral = frame["neutral"].astype(float)
    baseline = frame["baseline"].astype(float)
    y_pos = range(len(metrics))

    plt.figure(figsize=(7.5, 3.0))
    plt.barh(y_pos, dashbot, color="#4C93C3", label="DashBot is more preferable")
    plt.barh(y_pos, neutral, left=dashbot, color="#F28E2B", label="Neutral")
    plt.barh(y_pos, baseline, left=dashbot + neutral, color="#59A14F", label=f"{args.baseline_name} is more preferable")

    for index, (d_value, n_value, b_value) in enumerate(zip(dashbot, neutral, baseline)):
        if d_value:
            plt.text(d_value / 2, index, f"{int(d_value)}", va="center", ha="center", color="white", fontsize=8)
        if n_value:
            plt.text(d_value + n_value / 2, index, f"{int(n_value)}", va="center", ha="center", color="white", fontsize=8)
        if b_value:
            plt.text(d_value + n_value + b_value / 2, index, f"{int(b_value)}", va="center", ha="center", color="white", fontsize=8)

    plt.yticks(list(y_pos), metrics)
    plt.xlim(0, max((dashbot + neutral + baseline).max(), 1))
    plt.xlabel("number of ratings")
    plt.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
