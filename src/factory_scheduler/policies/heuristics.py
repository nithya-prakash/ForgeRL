"""Dispatching heuristics. Each ranks feasible (op, machine) pairs with a priority key.

All rules see exactly the same Candidate list, so they obey the same physical constraints.
Simplifications: rules are myopic, non-delay, and machine choice is folded into the key
(effective time = setup + processing, so slower/needing-setup machines rank worse).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from factory_scheduler.simulation.engine import FactorySim
from factory_scheduler.simulation.model import Candidate

KeyFn = Callable[[FactorySim, Candidate], tuple]


def _eff(c: Candidate) -> int:
    return c.setup + c.proc_time


class RuleBased:
    def __init__(self, name: str, key: KeyFn):
        self.name, self._key = name, key

    def select(self, sim: FactorySim, candidates: list[Candidate]) -> int:
        return min(range(len(candidates)), key=lambda i: self._key(sim, candidates[i]))


def fifo(sim: FactorySim, c: Candidate) -> tuple:
    return (c.ready_since, c.job_id, _eff(c), c.machine_id)


def spt(sim: FactorySim, c: Candidate) -> tuple:
    return (_eff(c), c.ready_since, c.job_id, c.machine_id)


def edd(sim: FactorySim, c: Candidate) -> tuple:
    return (c.due, c.job_id, _eff(c), c.machine_id)


def critical_ratio(sim: FactorySim, c: Candidate) -> tuple:
    """CR = (due - now) / remaining work; smallest first (negative = already late)."""
    cr = (c.due - sim.now) / max(1, c.work_left)
    return (cr, c.job_id, _eff(c), c.machine_id)


def least_slack(sim: FactorySim, c: Candidate) -> tuple:
    return (c.due - sim.now - c.work_left, c.job_id, _eff(c), c.machine_id)


def make_atc(k: float = 2.0):
    """Apparent Tardiness Cost, priority-weighted, workload- and machine-aware.

    index = w/p_eff * exp(-max(0, slack)/(k * pbar)) * (best_p / p)
    slack = due - now - remaining work after this op (remaining workload term);
    pbar = mean processing time of ready ops (current congestion); the last factor
    penalises assigning a flexible op to a machine slower than its best option.
    Higher index is better, so the key is negated.
    """

    def atc(sim: FactorySim, c: Candidate) -> tuple:
        ready = sim.ready_jobs()
        ops = [sim.current_op(j) for j in ready]
        pbar = sum(min(o.proc_times.values()) for o in ops if o) / max(1, len(ops))
        p = _eff(c)
        slack = c.due - sim.now - (c.work_left - min(sim.current_op(c.job_id).proc_times.values()))  # type: ignore[union-attr]
        best = min(sim.current_op(c.job_id).proc_times.values())  # type: ignore[union-attr]
        idx = (c.priority / p) * math.exp(-max(0.0, slack) / (k * pbar)) * (best / c.proc_time)
        return (-idx, c.job_id, c.machine_id)

    return atc


HEURISTICS: dict[str, KeyFn] = {
    "FIFO": fifo,
    "SPT": spt,
    "EDD": edd,
    "CR": critical_ratio,
    "LSL": least_slack,
    "ATC": make_atc(),
}


def get_policy(name: str) -> RuleBased:
    return RuleBased(name, HEURISTICS[name])
