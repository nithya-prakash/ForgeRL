"""Static (offline) CP-SAT vs online heuristics on the CP-SAT-compatible problem.

Instances are `static_variant`s (no breakdowns, no setups). CP-SAT sees ALL jobs including
future releases, so it is an offline reference / lower bound, not an online competitor.
"""

from __future__ import annotations

import pandas as pd

from factory_scheduler.config import ScenarioConfig
from factory_scheduler.optimization.cpsat import solve_static, static_variant
from factory_scheduler.policies.base import run_policy
from factory_scheduler.policies.heuristics import HEURISTICS, get_policy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.metrics import compute_metrics
from factory_scheduler.simulation.validate import validate_schedule


def run_cpsat_bench(
    scenarios: dict[str, ScenarioConfig], seeds: list[int], objective: str, time_limit: float
) -> pd.DataFrame:
    rows = []
    for sname, cfg in scenarios.items():
        for seed in seeds:
            inst = static_variant(generate_instance(cfg, seed))
            sol, res = solve_static(inst, objective, time_limit, seed=0)
            base = {
                "scenario": sname,
                "seed": seed,
                "objective": objective,
                "n_jobs": len(inst.jobs),
                "n_ops": sum(len(j.ops) for j in inst.jobs),
            }
            if res is not None:
                assert validate_schedule(res) == []
                rows.append(
                    {
                        **base,
                        "policy": "CPSAT-static",
                        "solver_status": sol.status,
                        "solver_runtime_s": sol.runtime,
                        "time_limit_s": time_limit,
                        "gap": sol.gap,
                        "bound": sol.bound,
                        **compute_metrics(res),
                    }
                )
            else:
                rows.append(
                    {
                        **base,
                        "policy": "CPSAT-static",
                        "solver_status": sol.status,
                        "solver_runtime_s": sol.runtime,
                        "time_limit_s": time_limit,
                    }
                )
            for h in HEURISTICS:
                r = run_policy(inst, get_policy(h))
                assert validate_schedule(r) == []
                rows.append({**base, "policy": h, **compute_metrics(r)})
    return pd.DataFrame(rows)
