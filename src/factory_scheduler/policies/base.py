"""Policy interface + the loop that drives the simulator with any policy."""

from __future__ import annotations

import time
from typing import Protocol

from factory_scheduler.simulation.engine import FactorySim, SimResult
from factory_scheduler.simulation.model import Candidate, Instance


class Policy(Protocol):
    name: str

    def select(self, sim: FactorySim, candidates: list[Candidate]) -> int:
        """Return the index into `candidates` to dispatch. Must not mutate `sim`."""


def run_policy(instance: Instance, policy: Policy) -> SimResult:
    sim = FactorySim(instance)
    latencies: list[float] = []
    while sim.advance():
        cands = sim.candidates()
        t0 = time.perf_counter()
        i = policy.select(sim, cands)
        latencies.append(time.perf_counter() - t0)
        c = cands[i]
        sim.dispatch(c.job_id, c.machine_id)
    res = sim.result()
    res.decision_latencies = latencies
    return res
