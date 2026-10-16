"""Run policies over (scenario, seed) grids on identical instances and save results."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from factory_scheduler.config import ScenarioConfig
from factory_scheduler.policies.base import Policy, run_policy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.metrics import compute_metrics, machine_stats
from factory_scheduler.simulation.validate import validate_schedule

# Seed ranges keep train / validation / test instances disjoint.
SEED_BASE = {"train": 0, "val": 100_000, "test": 200_000}


def split_seeds(split: str, n: int) -> list[int]:
    return [SEED_BASE[split] + i for i in range(n)]


def save_schedule(res, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inst = res.instance
    path.write_text(
        json.dumps(
            {
                "scenario": inst.name,
                "seed": inst.seed,
                "end_time": res.end_time,
                "truncated": res.truncated,
                "completion": res.completion,
                "unfinished": res.unfinished,
                "downtime": res.downtime,
                "segments": [asdict(s) for s in res.segments],
                "jobs": [
                    {
                        "id": j.id,
                        "release": j.release,
                        "due": j.due,
                        "priority": j.priority,
                        "n_ops": len(j.ops),
                    }
                    for j in inst.jobs
                ],
                "machines": [asdict(m) for m in inst.machines],
                "machine_stats": machine_stats(res),
            }
        )
    )


def run_grid(
    policies: Iterable[Policy],
    scenarios: dict[str, ScenarioConfig],
    seeds: list[int],
    split: str = "test",
    schedule_dir: Path | None = None,
    n_schedules: int = 1,
) -> pd.DataFrame:
    """One row per (policy, scenario, seed). Every schedule is independently validated."""
    rows = []
    policies = list(policies)
    for sname, cfg in scenarios.items():
        for si, seed in enumerate(seeds):
            inst = generate_instance(cfg, seed)  # same instance object for all policies
            for pol in policies:
                t0 = time.perf_counter()
                res = run_policy(inst, pol)
                wall = time.perf_counter() - t0
                errs = validate_schedule(res)
                if errs:
                    raise AssertionError(f"{pol.name}/{sname}/{seed} invalid: {errs[:3]}")
                row = {
                    "policy": pol.name,
                    "scenario": sname,
                    "seed": seed,
                    "split": split,
                    "wall_s": wall,
                    **compute_metrics(res),
                }
                rows.append(row)
                if schedule_dir and si < n_schedules:
                    save_schedule(res, schedule_dir / f"{pol.name}__{sname}__{seed}.json")
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, out_dir: Path, name: str, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir / f"{name}.csv"
