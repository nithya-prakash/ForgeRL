"""Behaviour cloning of a dispatching rule into the PPO policy network (diagnostic tool).

Purpose (docs/rl_failure_analysis.md, E1): if the *same* observation and network can copy
ATC well, the observation is sufficient and PPO's gap is an optimisation/credit-assignment
problem; if cloning also fails, the representation is the bottleneck.
"""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from factory_scheduler.config import ScenarioConfig
from factory_scheduler.environment import features as F
from factory_scheduler.environment.env import FactoryEnv
from factory_scheduler.policies.base import Policy
from factory_scheduler.simulation.engine import FactorySim
from factory_scheduler.simulation.generator import generate_instance


def collect(
    teacher: Policy, scenarios: dict[str, ScenarioConfig], seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs, masks, acts = [], [], []
    for cfg in scenarios.values():
        for seed in seeds:
            sim = FactorySim(generate_instance(cfg, seed))
            while sim.advance():
                cands = sim.candidates()
                chosen = cands[teacher.select(sim, cands)]
                ordered = F.ordered_candidates(sim)
                slot = ordered.index(chosen)
                if slot < F.N_SLOTS:  # teacher choice invisible to the agent: skip sample
                    obs.append(F.build_obs(sim, ordered))
                    masks.append(F.action_mask(ordered))
                    acts.append(slot)
                sim.dispatch(chosen.job_id, chosen.machine_id)
    return np.stack(obs), np.stack(masks), np.array(acts)


def clone(
    model: MaskablePPO,
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 0,
) -> list[float]:
    """Cross-entropy on masked logits; returns per-epoch mean loss."""
    torch.manual_seed(seed)
    obs, masks, acts = (torch.as_tensor(x) for x in data)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n, losses = len(acts), []
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            _, logp, _ = model.policy.evaluate_actions(obs[idx], acts[idx], action_masks=masks[idx])
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        losses.append(tot / n)
    return losses


def new_model(scenarios, seeds, reward: dict, seed: int, **ppo_kw) -> MaskablePPO:
    env = FactoryEnv(scenarios, seeds, **reward)
    return MaskablePPO(
        "MlpPolicy", env, seed=seed, device="cpu", policy_kwargs={"net_arch": [128, 128]}, **ppo_kw
    )
