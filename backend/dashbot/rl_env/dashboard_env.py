from __future__ import annotations

import random
from typing import Any

import pandas as pd

from dashbot.core.data_profiler import DataProfiler
from dashbot.core.models import ActionType, ChartSpec, DashboardState, DatasetProfile
from dashbot.rl_env.constraints import ConstraintSampler
from dashbot.rl_env.rewards import RewardEngine


class DashboardEnv:
    """DashBot MDP environment: state, action execution, reward delta."""

    def __init__(
        self,
        frame: pd.DataFrame,
        profiler: DataProfiler | None = None,
        reward_engine: RewardEngine | None = None,
        constraints: ConstraintSampler | None = None,
        max_steps: int = 50,
    ) -> None:
        self.frame = frame
        self.profiler = profiler or DataProfiler()
        self.profile: DatasetProfile = self.profiler.profile(frame)
        self.reward_engine = reward_engine or RewardEngine()
        self.constraints = constraints or ConstraintSampler()
        self.max_steps = max_steps
        self.state = DashboardState()
        self.steps = 0

    def reset(self, key_column: str | None = None) -> DashboardState:
        modeled = self.profile.modeled_columns()
        if not modeled:
            raise ValueError("DashboardEnv requires at least one modeled column.")
        self.steps = 0
        self.state = DashboardState(key_column=key_column or modeled[0].name, charts=[], reward=0.0)
        return self.state.copy()

    def step(self, action: ActionType, params: dict[str, Any] | None = None) -> tuple[DashboardState, float, bool, dict[str, Any]]:
        if self.state.key_column is None:
            self.reset()
        params = params or {}
        previous_charts = [ChartSpec(**chart.to_dict()) for chart in self.state.charts]

        valid_actions = self.constraints.action_mask(self.state.charts)
        if not valid_actions.get(action, False):
            return self.state.copy(), -1.0, False, {"invalid": True, "reason": f"Action {action} is masked"}

        if action == "change":
            self._change_key_column(params.get("key_column"))
        elif action == "add":
            chart = params.get("chart")
            self._add_chart(chart if isinstance(chart, ChartSpec) else ChartSpec(**chart))
        elif action == "remove":
            self._remove_chart(int(params.get("index", len(self.state.charts) - 1)))
        elif action == "terminate":
            reward = self.reward_engine.immediate_reward(self.frame, self.profile, previous_charts, self.state.charts)
            self.state.reward += reward
            return self.state.copy(), reward, True, {"terminated": True}

        reward = self.reward_engine.immediate_reward(self.frame, self.profile, previous_charts, self.state.charts)
        self.state.reward += reward
        self.steps += 1
        done = self.steps >= self.max_steps or len(self.state.charts) >= self.constraints.max_charts
        return self.state.copy(), reward, done, {"invalid": False}

    def random_valid_chart(self) -> ChartSpec:
        by_type = {column.type: column.name for column in self.profile.modeled_columns()}
        quantitative = [column.name for column in self.profile.modeled_columns() if column.type == "Q"]
        nominal = [column.name for column in self.profile.modeled_columns() if column.type == "N"]
        temporal = [column.name for column in self.profile.modeled_columns() if column.type == "T"]

        if quantitative and nominal:
            return ChartSpec("bar", x=random.choice(nominal), y=random.choice(quantitative), y_agg="mean")
        if len(quantitative) >= 2:
            x, y = random.sample(quantitative, 2)
            return ChartSpec("point", x=x, y=y)
        if temporal and quantitative:
            return ChartSpec("line", x=random.choice(temporal), y=random.choice(quantitative), y_agg="mean")
        first = self.profile.modeled_columns()[0].name
        return ChartSpec("bar", x=by_type.get("Q", first), x_agg="bin")

    def _change_key_column(self, key_column: str | None) -> None:
        valid_names = {column.name for column in self.profile.modeled_columns()}
        if key_column not in valid_names:
            key_column = next(iter(valid_names))
        self.state.key_column = key_column

    def _add_chart(self, chart: ChartSpec) -> None:
        self.state.charts.append(chart)

    def _remove_chart(self, index: int) -> None:
        if 0 <= index < len(self.state.charts):
            self.state.charts.pop(index)
