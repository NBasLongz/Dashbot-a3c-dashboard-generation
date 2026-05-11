from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from dashbot.agent.feature_encoder import FeatureEncoderConfig, StateFeatureEncoder
from dashbot.agent.networks import DashBotActorCritic, NetworkConfig
from dashbot.agent.policy import PolicySampler
from dashbot.core.chart_generator import ChartGenerator
from dashbot.core.data_profiler import DataProfiler
from dashbot.core.insight_detector import InsightDetector
from dashbot.core.models import ChartSpec, DashboardState
from dashbot.rl_env.dashboard_env import DashboardEnv
from dashbot.rl_env.rewards import RewardEngine


DEFAULT_WEIGHT_PATH = Path(__file__).resolve().parents[1] / "weights" / "dashbot_actor_critic.pth"


class A3CDashboardRecommender:
    """Realtime dashboard search driven by the DashBot actor-critic policy."""

    def __init__(
        self,
        max_charts: int = 5,
        search_steps: int = 1000,
        weight_path: Path | str | None = None,
        hidden_size: int = 128,
        device: str | torch.device = "cpu",
    ) -> None:
        self.max_charts = max(1, min(max_charts, 8))
        self.search_steps = max(1, search_steps)
        self.weight_path = Path(weight_path) if weight_path else DEFAULT_WEIGHT_PATH
        self.device = torch.device(device)
        self.profiler = DataProfiler()
        self.reward_engine = RewardEngine()
        self.chart_generator = ChartGenerator()
        self.insight_detector = InsightDetector()
        self.feature_encoder = StateFeatureEncoder(FeatureEncoderConfig(max_columns=10, max_charts=8))
        self.model = DashBotActorCritic(
            NetworkConfig(
                feature_size=self.feature_encoder.feature_size,
                hidden_size=hidden_size,
                max_columns=10,
                max_charts=8,
            )
        ).to(self.device)
        self.model_loaded = self._load_weights()
        self.model.eval()
        self.policy = PolicySampler()

    def recommend(self, frame: pd.DataFrame) -> dict:
        env = DashboardEnv(
            frame,
            profiler=self.profiler,
            reward_engine=self.reward_engine,
            max_steps=min(50, self.search_steps),
        )
        state = env.reset()
        best_state = state.copy()
        best_reward = self._dashboard_reward(frame, env, best_state.charts)
        seen_charts: dict[tuple[str, str, str | None, str | None, str | None], ChartSpec] = {}
        steps_used = 0

        with torch.no_grad():
            while steps_used < self.search_steps:
                state_tensor = self.feature_encoder.encode(state, env.profile).unsqueeze(0).to(self.device)
                outputs = self.model(state_tensor)
                decision = self.policy.sample(outputs, state, env.profile)
                next_state, _, done, info = env.step(decision.action, decision.params)
                steps_used += 1

                if not info.get("invalid"):
                    for chart in next_state.charts:
                        seen_charts[self._chart_signature(chart)] = chart
                    candidate_reward = self._dashboard_reward(frame, env, next_state.charts)
                    if next_state.charts and candidate_reward > best_reward:
                        best_state = next_state.copy()
                        best_reward = candidate_reward

                if done:
                    modeled = env.profile.modeled_columns()
                    key_column = modeled[steps_used % len(modeled)].name if modeled else None
                    state = env.reset(key_column=key_column)
                else:
                    state = next_state

        if not best_state.charts:
            fallback_chart = env.random_valid_chart()
            best_state = DashboardState(key_column=state.key_column, charts=[fallback_chart])
            best_reward = self._dashboard_reward(frame, env, best_state.charts)

        selected = self._dedupe_charts(best_state.charts)
        selected = self._fill_from_seen(frame, env, selected, seen_charts)
        selected = selected[: self.max_charts]
        final_reward = self._dashboard_reward(frame, env, selected)
        insights = self.insight_detector.detect_dashboard(frame, env.profile, selected)
        return {
            "method": "a3c",
            "model_loaded": self.model_loaded,
            "search_steps": steps_used,
            "key_column": best_state.key_column,
            "reward": final_reward,
            "profile": env.profile.to_dict(),
            "charts": [
                {
                    **chart.to_dict(),
                    "vega_lite": self.chart_generator.to_vega_lite(chart, profile=env.profile),
                }
                for chart in selected
            ],
            "insights": [insight.to_dict() for insight in insights],
        }

    def _load_weights(self) -> bool:
        if not self.weight_path.exists():
            return False
        checkpoint = torch.load(self.weight_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        return True

    def _dashboard_reward(self, frame: pd.DataFrame, env: DashboardEnv, charts: list[ChartSpec]) -> float:
        return self.reward_engine.dashboard_reward(frame, env.profile, charts)

    def _fill_from_seen(
        self,
        frame: pd.DataFrame,
        env: DashboardEnv,
        selected: list[ChartSpec],
        seen_charts: dict[tuple[str, str, str | None, str | None, str | None], ChartSpec],
    ) -> list[ChartSpec]:
        used = {self._chart_signature(chart) for chart in selected}
        candidates = [chart for signature, chart in seen_charts.items() if signature not in used]
        candidates.sort(
            key=lambda chart: self._dashboard_reward(frame, env, selected + [chart]),
            reverse=True,
        )
        for chart in candidates:
            signature = self._chart_signature(chart)
            if signature in used:
                continue
            selected.append(chart)
            used.add(signature)
            if len(selected) >= self.max_charts:
                break
        return selected

    def _dedupe_charts(self, charts: list[ChartSpec]) -> list[ChartSpec]:
        selected: list[ChartSpec] = []
        used = set()
        for chart in charts:
            signature = self._chart_signature(chart)
            if signature in used:
                continue
            selected.append(chart)
            used.add(signature)
        return selected

    @staticmethod
    def _chart_signature(chart: ChartSpec) -> tuple[str, str, str | None, str | None, str | None]:
        return (
            chart.mark,
            chart.x,
            chart.y,
            chart.x_agg,
            chart.y_agg,
        )
