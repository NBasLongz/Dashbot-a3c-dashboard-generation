from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from dashbot.agent.a3c_worker import A3CConfig
from dashbot.agent.feature_encoder import FeatureEncoderConfig, StateFeatureEncoder
from dashbot.agent.memory import RolloutBuffer, Transition
from dashbot.agent.networks import DashBotActorCritic, NetworkConfig
from dashbot.agent.policy import PolicySampler
from dashbot.rl_env.dashboard_env import DashboardEnv
from scripts.train import default_data_dir, default_manifest, selected_csv_paths


class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared state for A3C worker processes."""

    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(parameter.data)
                state["exp_avg_sq"] = torch.zeros_like(parameter.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DashBot with asynchronous A3C workers.")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--manifest", type=Path, default=default_manifest())
    parser.add_argument("--rollout-length", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=5000)
    parser.add_argument("--log-csv", type=Path, default=ROOT / "reports" / "training_curve_a3c.csv")
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "backend" / "dashbot" / "weights" / "checkpoints")
    parser.add_argument("--save-path", type=Path, default=ROOT / "backend" / "dashbot" / "weights" / "dashbot_actor_critic.pth")
    args = parser.parse_args()

    csv_paths = [path for path in selected_csv_paths(args.data_dir, args.manifest) if path.exists()]
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    torch.manual_seed(args.seed)
    feature_encoder = StateFeatureEncoder(FeatureEncoderConfig(max_columns=10, max_charts=8))
    global_model = DashBotActorCritic(
        NetworkConfig(
            feature_size=feature_encoder.feature_size,
            hidden_size=args.hidden_size,
            max_columns=10,
            max_charts=8,
        )
    )
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=args.learning_rate)
    counter = mp.Value("i", 0)
    print_lock = mp.Lock()
    init_log_csv(args.log_csv)

    processes = []
    worker_count = max(1, args.workers)
    for worker_id in range(worker_count):
        process = mp.Process(
            target=worker_main,
            args=(worker_id, args, csv_paths, global_model, optimizer, counter, print_lock),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    save_checkpoint(args.save_path, global_model, args, worker_count, args.steps, feature_encoder.feature_size)
    print(f"saved={args.save_path}")


def worker_main(
    worker_id: int,
    args: argparse.Namespace,
    csv_paths: list[Path],
    global_model: DashBotActorCritic,
    optimizer: SharedAdam,
    counter,
    print_lock,
) -> None:
    random.seed(args.seed + worker_id)
    torch.manual_seed(args.seed + worker_id)
    feature_encoder = StateFeatureEncoder(FeatureEncoderConfig(max_columns=10, max_charts=8))
    local_model = DashBotActorCritic(
        NetworkConfig(
            feature_size=feature_encoder.feature_size,
            hidden_size=args.hidden_size,
            max_columns=10,
            max_charts=8,
        )
    )
    local_model.load_state_dict(global_model.state_dict())
    policy = PolicySampler()
    buffer = RolloutBuffer()
    config = A3CConfig(
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        learning_rate=args.learning_rate,
    )
    env = DashboardEnv(load_random_frame(csv_paths))
    state = env.reset()
    episode_return = 0.0
    completed_episodes = 0
    latest_stats: dict[str, float] | None = None

    while True:
        with counter.get_lock():
            if counter.value >= args.steps:
                break
            counter.value += 1
            global_step = counter.value

        state_tensor = feature_encoder.encode(state, env.profile, shuffle_charts=True).unsqueeze(0)
        outputs = local_model(state_tensor)
        decision = policy.sample(outputs, state, env.profile)
        next_state, reward, done, _ = env.step(decision.action, decision.params)
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
            stats = update_global(local_model, global_model, optimizer, buffer, config)
            latest_stats = stats
            buffer.clear()
            local_model.load_state_dict(global_model.state_dict())
        else:
            stats = None

        if should_log(global_step, args.log_interval, args.workers):
            with print_lock:
                loss_text = "" if latest_stats is None else f" loss={latest_stats['loss']:.4f}"
                print(
                    f"worker={worker_id} step={global_step} episodes={completed_episodes} "
                    f"return={episode_return:.4f} charts={len(next_state.charts)}{loss_text}",
                    flush=True,
                )
                append_log_csv(
                    args.log_csv,
                    worker_id,
                    global_step,
                    completed_episodes,
                    episode_return,
                    len(next_state.charts),
                    latest_stats,
                )
                maybe_save_periodic_checkpoint(global_model, args, global_step, feature_encoder.feature_size)

        if done:
            completed_episodes += 1
            env = DashboardEnv(load_random_frame(csv_paths))
            state = env.reset()
            episode_return = 0.0
        else:
            state = next_state


def update_global(
    local_model: DashBotActorCritic,
    global_model: DashBotActorCritic,
    optimizer: SharedAdam,
    rollout: RolloutBuffer,
    config: A3CConfig,
) -> dict[str, float]:
    if not rollout.transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    returns = torch.tensor(rollout.returns(config.gamma), dtype=torch.float32)
    values = torch.stack([transition.value for transition in rollout.transitions]).float()
    log_probs = torch.stack([transition.log_prob for transition in rollout.transitions]).float()
    entropies = torch.stack([transition.entropy for transition in rollout.transitions]).float()

    advantages = returns - values.detach()
    value_loss = F.mse_loss(values, returns)
    policy_loss = -(log_probs * advantages).mean()
    entropy = entropies.mean()
    loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
    ensure_shared_grads(local_model, global_model)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
    }


def ensure_shared_grads(local_model: DashBotActorCritic, global_model: DashBotActorCritic) -> None:
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            continue
        global_param._grad = local_param.grad


def should_log(step: int, log_interval: int, workers: int) -> bool:
    return log_interval > 0 and (step == 1 or step % log_interval == 0)


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


def append_log_csv(
    path: Path,
    worker_id: int,
    step: int,
    episodes: int,
    episode_return: float,
    charts: int,
    stats: dict[str, float] | None,
) -> None:
    stats = stats or {}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                worker_id,
                step,
                episodes,
                episode_return,
                charts,
                stats.get("loss", ""),
                stats.get("policy_loss", ""),
                stats.get("value_loss", ""),
                stats.get("entropy", ""),
            ]
        )


def maybe_save_periodic_checkpoint(
    global_model: DashBotActorCritic,
    args: argparse.Namespace,
    step: int,
    feature_size: int,
) -> None:
    if args.checkpoint_interval <= 0 or step % args.checkpoint_interval != 0:
        return
    checkpoint_path = args.checkpoint_dir / f"dashbot_actor_critic_step_{step}.pth"
    save_checkpoint(checkpoint_path, global_model, args, args.workers, step, feature_size)


def save_checkpoint(
    path: Path,
    model: DashBotActorCritic,
    args: argparse.Namespace,
    worker_count: int,
    step: int,
    feature_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_size": feature_size,
            "hidden_size": args.hidden_size,
            "max_columns": 10,
            "max_charts": 8,
            "training": "a3c",
            "workers": worker_count,
            "steps": step,
        },
        path,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
