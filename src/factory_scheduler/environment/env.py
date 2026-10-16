"""Gymnasium environment: one step = one dispatch decision (choose a feasible slot).

Reward (see docs/rl_design.md): r_t = -(cost(t_now) - cost(t_prev)) / reward_scale with
  cost(t) = a * sum_j w_j * flow_j(t) + b * sum_j w_j * overdue_j(t)
where flow_j(t)/overdue_j(t) accrue from release/due-date until the job completes or the
episode ends. The undiscounted return is -(a*weighted flow + b*weighted tardiness)/scale,
exactly the evaluated quantities, so there is no proxy to hack: idling and ignoring a job
keep its cost accruing, and truncation at the horizon charges unfinished jobs up to T.
terminated = every job finished; truncated = horizon reached with unfinished work.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from factory_scheduler.config import ScenarioConfig
from factory_scheduler.environment import features as F
from factory_scheduler.simulation.engine import FactorySim, SimResult
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.model import Candidate, Instance

MAX_INVALID = 20


def accrued_cost(sim: FactorySim, t: int, a: float, b: float) -> float:
    total = 0.0
    for j in sim.inst.jobs:
        if t <= j.release:
            continue
        end = min(t, sim.completion.get(j.id, t))
        total += j.priority * (a * (end - j.release) + b * max(0, end - j.due))
    return total


class FactoryEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenarios: dict[str, ScenarioConfig],
        seeds: list[int],
        flow_weight: float = 1.0,
        tardiness_weight: float = 1.0,
        reward_scale: float = 500.0,
        sequential: bool = False,
    ):
        super().__init__()
        self.scenarios, self.seeds = scenarios, list(seeds)
        self.a, self.b, self.scale = flow_weight, tardiness_weight, reward_scale
        self.sequential = sequential
        self.observation_space = spaces.Box(-10.0, 10.0, (F.OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(F.N_SLOTS)
        self._cursor = 0
        self.sim: FactorySim | None = None
        self._cands: list[Candidate] = []
        self._cost = 0.0
        self._invalid = 0

    # ----------------------------------------------------------------- helpers
    def _pick_instance(self) -> Instance:
        names = sorted(self.scenarios)
        if self.sequential:
            k = self._cursor
            self._cursor += 1
            name, seed = names[k % len(names)], self.seeds[(k // len(names)) % len(self.seeds)]
        else:
            name = names[int(self.np_random.integers(len(names)))]
            seed = self.seeds[int(self.np_random.integers(len(self.seeds)))]
        return generate_instance(self.scenarios[name], seed)

    def _sync(self) -> bool:
        assert self.sim is not None
        more = self.sim.advance()
        self._cands = F.ordered_candidates(self.sim) if more else []
        return more

    def _obs(self) -> np.ndarray:
        assert self.sim is not None
        return F.build_obs(self.sim, self._cands)

    def action_masks(self) -> np.ndarray:
        return F.action_mask(self._cands)

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        inst = (options or {}).get("instance") or self._pick_instance()
        self.sim = FactorySim(inst)
        self._cost, self._invalid = 0.0, 0
        if not self._sync():  # zero jobs: nothing to decide
            pass
        return self._obs(), {"instance": inst.name, "instance_seed": inst.seed}

    def step(self, action: int):
        sim = self.sim
        assert sim is not None and not sim.done, "call reset() first"
        info: dict[str, Any] = {}
        a = int(action)
        valid = 0 <= a < min(len(self._cands), F.N_SLOTS)
        if not valid:  # infeasible choice: no state change, small penalty, bounded repeats
            self._invalid += 1
            info["invalid_action"] = True
            trunc = self._invalid >= MAX_INVALID
            if trunc:
                sim._finish(truncated=True)
            return self._obs(), -1.0 / self.scale * 10, False, trunc, info
        self._invalid = 0
        c = self._cands[a]
        sim.dispatch(c.job_id, c.machine_id)
        more = self._sync()
        cost = accrued_cost(sim, sim.now if more else sim.end_time, self.a, self.b)
        reward = -(cost - self._cost) / self.scale
        self._cost = cost
        terminated = (not more) and not sim.truncated
        truncated = (not more) and sim.truncated
        if not more:
            info["result"] = sim.result()
        return self._obs(), float(reward), terminated, truncated, info

    def result(self) -> SimResult:
        assert self.sim is not None
        return self.sim.result()
