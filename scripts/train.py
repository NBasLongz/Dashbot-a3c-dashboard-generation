from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.agent.a3c_worker import A3CConfig, A3CTrainer
from dashbot.agent.feature_encoder import FeatureEncoderConfig, StateFeatureEncoder
from dashbot.agent.memory import RolloutBuffer, Transition
from dashbot.agent.networks import DashBotActorCritic, NetworkConfig
from dashbot.agent.policy import PolicySampler
from dashbot.rl_env.dashboard_env import DashboardEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DashBot actor-critic agent.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--manifest", type=Path, default=default_manifest())
    parser.add_argument("--rollout-length", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-path", type=Path, default=ROOT / "backend" / "dashbot" / "weights" / "dashbot_actor_critic.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    frames = load_frames(args.data_dir, args.manifest)
    if not frames:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    feature_encoder = StateFeatureEncoder(FeatureEncoderConfig(max_columns=10, max_charts=8))
    model = DashBotActorCritic(
        NetworkConfig(
            feature_size=feature_encoder.feature_size,
            hidden_size=args.hidden_size,
            max_columns=10,
            max_charts=8,
        )
    )
    trainer = A3CTrainer(
        model,
        A3CConfig(
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            learning_rate=args.learning_rate,
        ),
    )
    policy = PolicySampler()
    buffer = RolloutBuffer()

    env = DashboardEnv(random.choice(frames))
    state = env.reset()
    episode_return = 0.0
    completed_episodes = 0

    for global_step in range(1, args.steps + 1):
        state_tensor = feature_encoder.encode(state, env.profile, shuffle_charts=True).unsqueeze(0)
        outputs = model(state_tensor)
        decision = policy.sample(outputs, state, env.profile)
        next_state, reward, done, info = env.step(decision.action, decision.params)

        buffer.append(
            Transition(
                state=state_tensor.detach(),
                action={"type": decision.action, **decision.params},
                reward=reward,
                done=done,
                log_prob=decision.log_prob,
                value=decision.value,
                entropy=decision.entropy,
            )
        )
        episode_return += reward

        if len(buffer.transitions) >= args.rollout_length or done:
            stats = trainer.update(buffer)
            buffer.clear()
        else:
            stats = None

        if done:
            completed_episodes += 1
            if completed_episodes % 10 == 0:
                loss_text = "" if stats is None else f" loss={stats['loss']:.4f}"
                print(
                    f"step={global_step} episodes={completed_episodes} "
                    f"return={episode_return:.4f} charts={len(next_state.charts)}{loss_text}"
                )
            env = DashboardEnv(random.choice(frames))
            state = env.reset()
            episode_return = 0.0
        else:
            state = next_state

    if buffer.transitions:
        trainer.update(buffer)
        buffer.clear()

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_size": feature_encoder.feature_size,
            "hidden_size": args.hidden_size,
            "max_columns": 10,
            "max_charts": 8,
        },
        args.save_path,
    )
    print(f"saved={args.save_path}")


def load_frames(data_dir: Path, manifest: Path | None = None) -> list[pd.DataFrame]:
    frames = []
    csv_paths = selected_csv_paths(data_dir, manifest)
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"skip={csv_path} reason={exc}")
            continue
        if not frame.empty and len(frame.columns) > 0:
            frames.append(frame)
    return frames


def selected_csv_paths(data_dir: Path, manifest: Path | None = None) -> list[Path]:
    if manifest and manifest.exists():
        names = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [data_dir / name for name in names]
    return sorted(data_dir.glob("*.csv"))


def default_data_dir() -> Path:
    processed = ROOT / "data" / "processed"
    if (processed / "vega_27_manifest.txt").exists():
        return processed
    return ROOT / "data" / "raw"


def default_manifest() -> Path:
    processed = ROOT / "data" / "processed" / "vega_27_manifest.txt"
    if processed.exists():
        return processed
    return ROOT / "data" / "raw" / "vega_27_manifest.txt"


if __name__ == "__main__":
    main()
