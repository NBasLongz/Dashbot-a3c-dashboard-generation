from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Transition:
    state: Any
    action: dict[str, Any]
    reward: float
    done: bool
    log_prob: Any | None = None
    value: Any | None = None
    entropy: Any | None = None


@dataclass
class RolloutBuffer:
    """Bộ nhớ lưu trữ quỹ đạo (trajectory) ngắn hạn phục vụ cho cập nhật actor-critic."""

    transitions: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    @property
    def states(self) -> list[Any]:
        return [transition.state for transition in self.transitions]

    @property
    def actions(self) -> list[dict[str, Any]]:
        return [transition.action for transition in self.transitions]

    @property
    def rewards(self) -> list[float]:
        return [transition.reward for transition in self.transitions]

    @property
    def dones(self) -> list[bool]:
        return [transition.done for transition in self.transitions]

    def returns(self, gamma: float = 1.0) -> list[float]:
        running = 0.0
        returns: list[float] = []
        for transition in reversed(self.transitions):
            running = transition.reward + gamma * running * (1.0 - float(transition.done))
            returns.append(running)
        return list(reversed(returns))

    def returns_tensor(self, gamma: float = 1.0, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(self.returns(gamma), dtype=torch.float32, device=device)
