"""Seeded synthetic scenario generator. Same (config, seed) -> identical Instance."""

from __future__ import annotations

import numpy as np

from factory_scheduler.config import ScenarioConfig
from factory_scheduler.simulation.model import (
    Breakdown,
    Instance,
    JobSpec,
    MachineSpec,
    OperationSpec,
)

FLEX_SLOWDOWN = 1.4


def generate_instance(cfg: ScenarioConfig, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    machines: list[MachineSpec] = []
    speed: dict[int, float] = {}
    for typ, count in cfg.machine_types.items():
        for _ in range(count):
            mid = len(machines)
            machines.append(MachineSpec(mid, typ))
            speed[mid] = 1.0 + rng.uniform(-cfg.speed_spread, cfg.speed_spread)
    types = list(cfg.machine_types)
    by_type = {t: [m.id for m in machines if m.type == t] for t in types}

    t = 0.0
    jobs = []
    for jid in range(cfg.n_jobs):
        if cfg.arrival == "poisson":
            t += rng.exponential(1.0 / cfg.arrival_rate)
        release = int(round(t)) if cfg.arrival == "poisson" else 0
        n_ops = int(rng.integers(cfg.ops_range[0], cfg.ops_range[1] + 1))
        ops, mean_work = [], 0.0
        for k in range(n_ops):
            base = int(rng.integers(cfg.proc_range[0], cfg.proc_range[1] + 1))
            typ = types[int(rng.integers(len(types)))]
            elig = {m: max(1, round(base / speed[m])) for m in by_type[typ]}
            if cfg.flex_prob > 0 and len(types) > 1 and rng.random() < cfg.flex_prob:
                other = types[int(rng.integers(len(types)))]
                for m in by_type[other]:
                    elig.setdefault(m, max(1, round(FLEX_SLOWDOWN * base / speed[m])))
            ops.append(OperationSpec(jid, k, int(rng.integers(cfg.n_families)), elig))
            mean_work += sum(elig.values()) / len(elig)
        due = release + int(round(cfg.due_tightness * mean_work))
        prio = int(rng.choice(cfg.priorities))
        jobs.append(JobSpec(jid, release, due, prio, tuple(ops)))

    bds: list[Breakdown] = []
    if cfg.mtbf > 0 and cfg.mttr > 0:
        for mach in machines:
            clock, clock_end = 0.0, -1
            while True:
                clock += rng.exponential(cfg.mtbf)
                start = max(int(round(clock)), int(clock_end) + 1)
                if start >= cfg.horizon:
                    break
                dur = max(1, int(round(rng.exponential(cfg.mttr))))
                bds.append(Breakdown(mach.id, start, dur))
                clock, clock_end = start + dur, start + dur
    inst = Instance(
        cfg.name,
        seed,
        tuple(machines),
        tuple(jobs),
        cfg.setup_time,
        tuple(bds),
        cfg.horizon,
        cfg.tardiness_cost,
        {"config_name": cfg.name},
    )
    inst.validate()
    return inst
