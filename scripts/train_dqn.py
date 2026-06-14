from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.agent.feature_encoder import AGGREGATES, MARK_TYPES, FeatureEncoderConfig, StateFeatureEncoder
from dashbot.agent.networks import NetworkConfig
from dashbot.agent.policy import PolicySampler
from dashbot.core.models import ChartSpec
from dashbot.rl_env.dashboard_env import DashboardEnv
from scripts.train import default_data_dir, default_manifest, selected_csv_paths
from scripts.train_a3c import is_feasible_chart


class DashBotQNetwork(nn.Module):
    """DQN baseline with the same Bi-LSTM dashboard encoder family."""

    def __init__(self, config: NetworkConfig, action_count: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=config.feature_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        embedding_size = config.hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, action_count),
        )

    def forward(self, dashboard_features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(dashboard_features)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


@dataclass(frozen=True)
class ReplayTransition:
    state: torch.Tensor
    action_index: int
    reward: float
    next_state: torch.Tensor
    next_mask: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.transitions: deque[ReplayTransition] = deque(maxlen=capacity)

    def append(self, transition: ReplayTransition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        return random.sample(self.transitions, batch_size)

    def __len__(self) -> int:
        return len(self.transitions)


class DQNActionSpace:
    def __init__(self, max_columns: int = 10, max_charts: int = 8) -> None:
        self.max_columns = max_columns
        self.max_charts = max_charts
        self.change_count = max_columns
        self.add_count = len(MARK_TYPES) * max_columns * max_columns * len(AGGREGATES)
        self.remove_count = max_charts
        self.terminate_count = 1
        self.action_count = self.change_count + self.add_count + self.remove_count + self.terminate_count
        self._add_mask_cache: dict[tuple[tuple[str, str], ...], torch.Tensor] = {}

    def decode(self, action_index: int, profile) -> tuple[str, dict]:
        columns = profile.modeled_columns()
        if action_index < self.change_count:
            if action_index >= len(columns):
                return "change", {"key_column": "__invalid_column"}
            return "change", {"key_column": columns[action_index].name}

        action_index -= self.change_count
        if action_index < self.add_count:
            aggregate_index = action_index % len(AGGREGATES)
            action_index //= len(AGGREGATES)
            y_index = action_index % self.max_columns
            action_index //= self.max_columns
            x_index = action_index % self.max_columns
            action_index //= self.max_columns
            mark = MARK_TYPES[action_index % len(MARK_TYPES)]
            x_field = columns[x_index].name if x_index < len(columns) else "__invalid_x"
            y_field = columns[y_index].name if y_index < len(columns) else "__invalid_y"
            aggregate = AGGREGATES[aggregate_index]
            chart = ChartSpec(
                mark=mark,
                x=x_field,
                y=y_field,
                y_agg=None if aggregate == "none" else aggregate,
                title=f"{y_field} by {x_field}",
            )
            if x_field in profile.by_name() and y_field in profile.by_name():
                chart = PolicySampler._normalize_chart(chart, profile)
            return "add", {"chart": chart}

        action_index -= self.add_count
        if action_index < self.remove_count:
            return "remove", {"index": action_index}
        return "terminate", {}

    def valid_mask(self, state, profile) -> torch.Tensor:
        mask = torch.zeros(self.action_count, dtype=torch.bool)
        columns = profile.modeled_columns()

        for index in range(min(len(columns), self.max_columns)):
            mask[index] = True

        offset = self.change_count
        add_mask = self._valid_add_mask(profile)
        mask[offset : offset + self.add_count] = add_mask if len(state.charts) < self.max_charts else False

        offset += self.add_count
        for index in range(self.max_charts):
            mask[offset + index] = index < len(state.charts)

        mask[-1] = len(state.charts) > 0
        return mask

    def _valid_add_mask(self, profile) -> torch.Tensor:
        key = tuple((column.name, column.type) for column in profile.modeled_columns())
        cached = self._add_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.zeros(self.add_count, dtype=torch.bool)
        offset = self.change_count
        for add_index in range(self.add_count):
            _action, params = self.decode(offset + add_index, profile)
            chart = params.get("chart")
            mask[add_index] = is_feasible_chart(chart, profile)
        self._add_mask_cache[key] = mask
        return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DashBot DQN ablation baseline.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--manifest", type=Path, default=default_manifest())
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--target-update-interval", type=int, default=5000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000)
    parser.add_argument("--invalid-penalty", type=float, default=-1.0)
    parser.add_argument("--use-transformed-features", action="store_true")
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--log-csv", type=Path, default=ROOT / "reports" / "ablation" / "training_curve_dqn.csv")
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "backend" / "dashbot" / "weights" / "ablation" / "checkpoints_dqn")
    parser.add_argument("--save-path", type=Path, default=ROOT / "backend" / "dashbot" / "weights" / "ablation" / "dashbot_dqn.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    csv_paths = [path for path in selected_csv_paths(args.data_dir, args.manifest) if path.exists()]
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    feature_encoder = StateFeatureEncoder(FeatureEncoderConfig(max_columns=10, max_charts=8))
    action_space = DQNActionSpace(max_columns=10, max_charts=8)
    config = NetworkConfig(feature_size=feature_encoder.feature_size, hidden_size=args.hidden_size, max_columns=10, max_charts=8)
    model = DashBotQNetwork(config, action_space.action_count)
    target_model = DashBotQNetwork(config, action_space.action_count)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay_size)
    init_log_csv(args.log_csv)

    env = DashboardEnv(load_random_frame(csv_paths))
    state = env.reset()
    episode_return = 0.0
    completed_episodes = 0
    latest_loss: float | None = None

    for step in range(1, args.steps + 1):
        feature_frame = env.frame if args.use_transformed_features else None
        state_tensor = feature_encoder.encode(state, env.profile, shuffle_charts=True, frame=feature_frame)
        mask = action_space.valid_mask(state, env.profile)
        action_index = choose_action(model, state_tensor, mask, epsilon_for_step(args, step))
        action, params = action_space.decode(action_index, env.profile)
        next_state, reward, done, info = env.step(action, params)
        if info.get("invalid"):
            reward += args.invalid_penalty

        next_tensor = feature_encoder.encode(next_state, env.profile, shuffle_charts=True, frame=feature_frame)
        next_mask = action_space.valid_mask(next_state, env.profile)
        replay.append(ReplayTransition(state_tensor.detach(), action_index, reward, next_tensor.detach(), next_mask, done))
        episode_return += reward

        if len(replay) >= args.batch_size:
            latest_loss = update_dqn(model, target_model, optimizer, replay, args.batch_size, args.gamma)

        if step % args.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0):
            print(
                f"worker=0 step={step} episodes={completed_episodes} return={episode_return:.4f} "
                f"charts={len(next_state.charts)} loss={latest_loss if latest_loss is not None else 0.0:.4f}",
                flush=True,
            )
            append_log_csv(args.log_csv, step, completed_episodes, episode_return, len(next_state.charts), latest_loss)
            maybe_save_checkpoint(model, args, step, feature_encoder.feature_size, action_space.action_count)

        if done:
            completed_episodes += 1
            env = DashboardEnv(load_random_frame(csv_paths))
            state = env.reset()
            episode_return = 0.0
        else:
            state = next_state

    save_checkpoint(args.save_path, model, args, args.steps, feature_encoder.feature_size, action_space.action_count)
    print(f"saved={args.save_path}")


def choose_action(model: DashBotQNetwork, state_tensor: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int:
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if len(valid_indices) == 0:
        return 0
    if random.random() < epsilon:
        return int(valid_indices[random.randrange(len(valid_indices))].item())
    with torch.no_grad():
        q_values = model(state_tensor.unsqueeze(0)).squeeze(0)
        q_values = q_values.masked_fill(~mask, -1e9)
        return int(torch.argmax(q_values).item())


def update_dqn(
    model: DashBotQNetwork,
    target_model: DashBotQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> float:
    batch = replay.sample(batch_size)
    states = torch.stack([transition.state for transition in batch])
    actions = torch.tensor([transition.action_index for transition in batch], dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32)
    next_states = torch.stack([transition.next_state for transition in batch])
    next_masks = torch.stack([transition.next_mask for transition in batch])
    dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32)

    q_values = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_model(next_states).masked_fill(~next_masks, -1e9)
        next_best = next_q.max(dim=1).values
        next_best = torch.where(torch.isfinite(next_best), next_best, torch.zeros_like(next_best))
        targets = rewards + gamma * next_best * (1.0 - dones)
    loss = F.smooth_l1_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


def epsilon_for_step(args: argparse.Namespace, step: int) -> float:
    fraction = min(max(step / max(args.epsilon_decay_steps, 1), 0.0), 1.0)
    return args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)


def load_random_frame(csv_paths: list[Path]) -> pd.DataFrame:
    for _ in range(10):
        csv_path = random.choice(csv_paths)
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if not frame.empty and len(frame.columns) > 0:
            return frame
    raise RuntimeError("Could not load any training CSV.")


def init_log_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["worker", "step", "episodes", "return", "charts", "loss", "policy_loss", "value_loss", "entropy"])


def append_log_csv(path: Path, step: int, episodes: int, episode_return: float, charts: int, loss: float | None) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([0, step, episodes, episode_return, charts, "" if loss is None else loss, "", "" if loss is None else loss, ""])


def maybe_save_checkpoint(
    model: DashBotQNetwork,
    args: argparse.Namespace,
    step: int,
    feature_size: int,
    action_count: int,
) -> None:
    if args.checkpoint_interval <= 0 or step % args.checkpoint_interval != 0:
        return
    save_checkpoint(args.checkpoint_dir / f"dashbot_dqn_step_{step}.pth", model, args, step, feature_size, action_count)


def save_checkpoint(
    path: Path,
    model: DashBotQNetwork,
    args: argparse.Namespace,
    step: int,
    feature_size: int,
    action_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_size": feature_size,
            "hidden_size": args.hidden_size,
            "max_columns": 10,
            "max_charts": 8,
            "action_count": action_count,
            "training": "dqn",
            "variant": "dqn",
            "steps": step,
        },
        path,
    )


if __name__ == "__main__":
    main()
