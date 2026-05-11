from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dashbot.agent.memory import RolloutBuffer
from dashbot.agent.networks import DashBotActorCritic


@dataclass(frozen=True)
class A3CConfig:
    gamma: float = 1.0
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    learning_rate: float = 1e-4


class A3CTrainer:
    """Minimal synchronous trainer shell; can be expanded to true async workers."""

    def __init__(self, model: DashBotActorCritic, config: A3CConfig | None = None) -> None:
        self.model = model
        self.config = config or A3CConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        if not rollout.transitions:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        returns = torch.tensor(rollout.returns(self.config.gamma), dtype=torch.float32)
        values = torch.stack([transition.value for transition in rollout.transitions]).float()
        log_probs = torch.stack([transition.log_prob for transition in rollout.transitions]).float()
        entropies = torch.stack([transition.entropy for transition in rollout.transitions]).float()

        advantages = returns - values.detach()
        value_loss = F.mse_loss(values, returns)
        policy_loss = -(log_probs * advantages).mean()
        entropy = entropies.mean()
        loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
        }
