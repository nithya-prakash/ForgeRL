"""Command-line entry point: `factory-scheduler <command>`."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from factory_scheduler.config import RESULTS_DIR, ROOT, load_scenarios
from factory_scheduler.evaluation.benchmark import run_grid, save_results, split_seeds
from factory_scheduler.evaluation.stats import summarize
from factory_scheduler.policies.base import Policy
from factory_scheduler.policies.heuristics import HEURISTICS, get_policy

log = logging.getLogger("factory_scheduler")
RESULTS = RESULTS_DIR


def cmd_benchmark(a: argparse.Namespace) -> None:
    scen = load_scenarios()
    if a.scenarios:
        scen = {k: scen[k] for k in a.scenarios}
    seeds = split_seeds(a.split, a.n_seeds)
    df = run_grid(
        [get_policy(p) for p in (a.policies or HEURISTICS)],
        scen,
        seeds,
        a.split,
        schedule_dir=RESULTS / "schedules",
    )
    path = save_results(
        df,
        RESULTS,
        a.name,
        {
            "scenarios": {k: vars(v) for k, v in scen.items()},
            "seeds": seeds,
            "split": a.split,
            "policies": sorted(df.policy.unique()),
        },
    )
    cols = ["weighted_tardiness", "mean_flow_time", "makespan", "on_time_pct"]
    s = summarize(df, cols)
    print(s[["scenario", "policy", *cols]].round(1).to_string(index=False))
    print(f"\nsaved {path}")


def cmd_compare(a: argparse.Namespace) -> None:
    """All heuristics + PPO (+ rolling CP-SAT) on the held-out test split; paired stats."""
    import json

    import pandas as pd

    from factory_scheduler.evaluation.stats import paired_compare
    from factory_scheduler.rl.policy import PPOPolicy

    scen_all = load_scenarios()
    scen = {k: scen_all[k] for k in (a.scenarios or scen_all)}
    seeds = split_seeds("test", a.n_seeds)
    pols: list[Policy] = [get_policy(p) for p in HEURISTICS]
    pols.append(PPOPolicy(a.model, name="PPO"))
    if a.rolling:
        from factory_scheduler.optimization.cpsat import RollingCPSAT

        pols.append(RollingCPSAT(time_limit=a.rolling_time_limit))
    t0 = time.perf_counter()
    df = run_grid(pols, scen, seeds, "test", schedule_dir=RESULTS / "schedules")
    save_results(
        df,
        RESULTS,
        a.name,
        {
            "scenarios": {k: vars(v) for k, v in scen.items()},
            "seeds": seeds,
            "model": a.model,
            "policies": sorted(df.policy.unique()),
            "seconds": time.perf_counter() - t0,
        },
    )
    heur = [p for p in HEURISTICS]
    best = {
        sc: min(
            heur, key=lambda p: df[(df.scenario == sc) & (df.policy == p)].weighted_tardiness.mean()
        )
        for sc in scen
    }
    rows = []
    for sc in scen:
        r = paired_compare(df, "PPO", best[sc], "weighted_tardiness", sc)
        rows.append({"scenario": sc, "strongest_heuristic": best[sc], **r})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / f"{a.name}_ppo_vs_best.csv", index=False)
    print(
        tab[
            [
                "scenario",
                "strongest_heuristic",
                "n",
                "mean_diff",
                "ci95_low",
                "ci95_high",
                "wilcoxon_p",
                "a_better_frac",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )
    print("(mean_diff < 0 => PPO has lower weighted tardiness than the strongest heuristic)")
    print(json.dumps({"saved": str(RESULTS / f"{a.name}.csv")}))


def cmd_cpsat(a: argparse.Namespace) -> None:
    import pandas as pd

    from factory_scheduler.evaluation.cpsat_bench import run_cpsat_bench

    scen_all = load_scenarios()
    scen = {k: scen_all[k].replace(n_jobs=a.n_jobs) for k in (a.scenarios or scen_all)}
    seeds = split_seeds("test", a.n_seeds)
    frames = [
        run_cpsat_bench(scen, seeds, obj, a.time_limit)
        for obj in ("makespan", "weighted_tardiness")
    ]
    df = pd.concat(frames, ignore_index=True)
    save_results(
        df,
        RESULTS,
        a.name,
        {"n_jobs": a.n_jobs, "seeds": seeds, "time_limit": a.time_limit, "note": "offline CP-SAT"},
    )
    cp = df[df.policy == "CPSAT-static"]
    print(cp.groupby(["objective", "solver_status"]).size().to_string())
    for obj, col in (("makespan", "makespan"), ("weighted_tardiness", "weighted_tardiness")):
        d = df[df.objective == obj].groupby("policy")[col].mean().round(1).sort_values()
        print(f"\nmean {col} (static variant, offline CP-SAT vs online rules):\n{d.to_string()}")


def cmd_train(a: argparse.Namespace) -> None:
    from factory_scheduler.rl.train import smoke_overrides, train

    print(train(a.config, smoke_overrides() if a.smoke else None))


SWEEPS: dict[str, tuple[str, str, list[Any]]] = {  # name -> (base scenario, config field, levels)
    "load": ("hetero_full", "n_jobs", [15, 25, 35, 50]),
    "breakdown": ("hetero_full", "mtbf", [400.0, 200.0, 100.0, 50.0]),
    "arrival": ("dynamic_arrivals", "arrival_rate", [0.06, 0.1, 0.14, 0.2]),
}


def cmd_sweep(a: argparse.Namespace) -> None:
    """Sensitivity sweeps on held-out test seeds; one CSV per sweep (level = swept value)."""
    import pandas as pd

    from factory_scheduler.rl.policy import PPOPolicy

    base_name, field, levels = SWEEPS[a.sweep]
    base = load_scenarios()[base_name]
    pols: list[Policy] = [get_policy(p) for p in ("FIFO", "SPT", "EDD", "ATC")]
    if a.model:
        pols.append(PPOPolicy(a.model, name="PPO"))
    seeds = split_seeds("test", a.n_seeds)
    frames: list[pd.DataFrame] = []
    for lv in levels:
        df = run_grid(
            pols, {f"{base_name}@{field}={lv}": base.replace(**{field: lv})}, seeds, "test"
        )
        df["level"] = lv
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    save_results(
        out,
        RESULTS,
        f"sweep_{a.sweep}",
        {"base": base_name, "field": field, "levels": levels, "seeds": seeds},
    )
    print(out.groupby(["level", "policy"]).weighted_tardiness.mean().unstack().round(0))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="factory-scheduler")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="run heuristic policies on generated scenarios")
    b.add_argument("--name", default="heuristics")
    b.add_argument("--split", default="test", choices=["train", "val", "test"])
    b.add_argument("--n-seeds", type=int, default=20)
    b.add_argument("--scenarios", nargs="*")
    b.add_argument("--policies", nargs="*", choices=list(HEURISTICS))
    b.set_defaults(fn=cmd_benchmark)
    c = sub.add_parser("compare", help="PPO vs heuristics (+rolling CP-SAT) on test seeds")
    c.add_argument("--name", default="compare")
    c.add_argument("--model", default=str(ROOT / "artifacts/models/ppo_default/best.zip"))
    c.add_argument("--n-seeds", type=int, default=20)
    c.add_argument("--scenarios", nargs="*")
    c.add_argument("--rolling", action="store_true")
    c.add_argument("--rolling-time-limit", type=float, default=0.3)
    c.set_defaults(fn=cmd_compare)
    q = sub.add_parser("cpsat", help="offline CP-SAT vs heuristics on small static variants")
    q.add_argument("--name", default="cpsat")
    q.add_argument("--n-seeds", type=int, default=10)
    q.add_argument("--n-jobs", type=int, default=10)
    q.add_argument("--time-limit", type=float, default=20.0)
    q.add_argument("--scenarios", nargs="*")
    q.set_defaults(fn=cmd_cpsat)
    t = sub.add_parser("train", help="train PPO from configs/ppo.yaml")
    t.add_argument("--config", default=None)
    t.add_argument("--smoke", action="store_true")
    t.set_defaults(fn=cmd_train)
    w = sub.add_parser("sweep", help="sensitivity sweep (load / breakdown / arrival)")
    w.add_argument("sweep", choices=list(SWEEPS))
    w.add_argument("--model", default=None)
    w.add_argument("--n-seeds", type=int, default=15)
    w.set_defaults(fn=cmd_sweep)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
