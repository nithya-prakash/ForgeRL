"""Wrap a trained MaskablePPO model as a scheduling Policy (deterministic inference)."""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm

from factory_scheduler.environment import features as F
from factory_scheduler.simulation.engine import FactorySim
from factory_scheduler.simulation.model import Candidate


class PPOPolicy:
    def __init__(self, model: BaseAlgorithm | str | Path, name: str = "PPO"):
        self.name = name
        self.model = (
            MaskablePPO.load(str(model), device="cpu") if isinstance(model, str | Path) else model
        )

    def select(self, sim: FactorySim, candidates: list[Candidate]) -> int:
        ordered = F.ordered_candidates(sim)
        obs = F.build_obs(sim, ordered)
        action, _ = self.model.predict(  # type: ignore[call-arg]
            obs, action_masks=F.action_mask(ordered), deterministic=True
        )
        chosen = ordered[int(action)]
        return candidates.index(chosen)
