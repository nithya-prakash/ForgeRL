# mypy: disable-error-code="attr-defined"
"""OR-Tools CP-SAT scheduling.

Three distinct things live here and must not be conflated (docs/optimization.md):
1. `solve_static`  - OFFLINE optimisation: all jobs (incl. future releases) known upfront,
   no breakdowns, no setups. A lower-bound/reference, NOT an online competitor.
2. `RollingCPSAT`  - ONLINE rolling-horizon policy: re-solves at every decision point using
   only released jobs; breakdowns/setups still occur in the simulator but are not modelled.
3. `solve_jobs`    - the shared model builder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from ortools.sat.python import cp_model

from factory_scheduler.simulation.engine import FactorySim, SimResult
from factory_scheduler.simulation.model import Candidate, Instance, JobSpec, Segment

STATUS = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


@dataclass
class Solution:
    status: str
    objective: float | None
    bound: float | None
    gap: float | None
    runtime: float
    time_limit: float
    assignment: dict[tuple[int, int], tuple[int, int]]  # (job, op) -> (machine, start)


def solve_jobs(
    jobs: list[JobSpec],
    first_op: dict[int, int],
    ready_at: dict[int, int],
    machine_free: dict[int, int],
    objective: str = "makespan",
    time_limit: float = 10.0,
    seed: int = 0,
    workers: int = 1,
) -> Solution:
    """Min makespan or min priority-weighted tardiness for the not-yet-started operations.

    first_op[j]: first unscheduled op index; ready_at[j]: earliest start of that op;
    machine_free[m]: earliest time machine m can start new work. workers=1 + fixed seed
    keeps the solve deterministic.
    """
    model = cp_model.CpModel()
    horizon = (
        max(list(ready_at.values()) + list(machine_free.values()) + [0])
        + sum(max(o.proc_times.values()) for j in jobs for o in j.ops[first_op[j.id] :])
        + 1
    )
    per_machine: dict[int, list] = {m: [] for m in machine_free}
    starts, ends, choice = {}, {}, {}
    job_end = {}
    for j in jobs:
        prev_end = None
        for op in j.ops[first_op[j.id] :]:
            s = model.NewIntVar(0, horizon, f"s{op.key}")
            e = model.NewIntVar(0, horizon, f"e{op.key}")
            starts[op.key], ends[op.key] = s, e
            if prev_end is None:
                model.Add(s >= ready_at[j.id])
            else:
                model.Add(s >= prev_end)
            lits = []
            for m, p in op.proc_times.items():
                lit = model.NewBoolVar(f"x{op.key}_{m}")
                itv = model.NewOptionalIntervalVar(s, p, e, lit, f"i{op.key}_{m}")
                model.Add(s >= machine_free[m]).OnlyEnforceIf(lit)
                per_machine[m].append(itv)
                choice[(op.key, m)] = lit
                lits.append(lit)
            model.AddExactlyOne(lits)
            prev_end = e
        if prev_end is not None:
            job_end[j.id] = prev_end
    for ivs in per_machine.values():
        model.AddNoOverlap(ivs)
    if objective == "makespan":
        obj = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(obj, list(job_end.values()) or [model.NewConstant(0)])
        model.Minimize(obj)
    elif objective == "weighted_tardiness":
        terms = []
        for j in jobs:
            if j.id in job_end:
                t = model.NewIntVar(0, horizon, f"t{j.id}")
                model.Add(t >= job_end[j.id] - j.due)
                terms.append(j.priority * t)
        model.Minimize(sum(terms))
    else:
        raise ValueError(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.perf_counter()
    st = solver.Solve(model)
    rt = time.perf_counter() - t0
    name = STATUS.get(st, "UNKNOWN")
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Solution(name, None, None, None, rt, time_limit, {})
    objv, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
    gap = abs(objv - bound) / max(1.0, abs(objv))
    assign = {}
    for (key, m), lit in choice.items():
        if solver.BooleanValue(lit):
            assign[key] = (m, solver.Value(starts[key]))
    return Solution(name, objv, bound, gap, rt, time_limit, assign)


def static_variant(inst: Instance) -> Instance:
    """The problem CP-SAT can model exactly: no breakdowns, no setups."""
    return replace(inst, breakdowns=(), setup_time=0, name=f"{inst.name}_static")


def solve_static(
    inst: Instance, objective: str = "makespan", time_limit: float = 10.0, seed: int = 0
) -> tuple[Solution, SimResult | None]:
    """OFFLINE solve. Requires an instance without breakdowns and setups."""
    if inst.breakdowns or inst.setup_time:
        raise ValueError(
            "static CP-SAT model supports neither breakdowns nor setups; "
            "use static_variant(instance)"
        )
    jobs = list(inst.jobs)
    sol = solve_jobs(
        jobs,
        {j.id: 0 for j in jobs},
        {j.id: j.release for j in jobs},
        {m.id: 0 for m in inst.machines},
        objective,
        time_limit,
        seed,
    )
    if not sol.assignment:
        return sol, None
    segs, comp = [], {}
    for j in jobs:
        for op in j.ops:
            m, s = sol.assignment[op.key]
            e = s + op.proc_times[m]
            segs.append(Segment(j.id, op.index, m, s, e, "process"))
            comp[j.id] = e
    segs.sort(key=lambda s: (s.start, s.machine_id))
    res = SimResult(
        inst, segs, comp, [], [], max(comp.values(), default=0), False, n_decisions=len(segs)
    )
    return sol, res


class RollingCPSAT:
    """Online rolling-horizon policy: re-plan at every decision point, execute the plan's
    immediate start. Uses only information visible to a real scheduler (released jobs,
    known processing times, current machine state, an assumed repair duration)."""

    def __init__(
        self,
        time_limit: float = 0.3,
        objective: str = "weighted_tardiness",
        repair_estimate: int = 10,
        seed: int = 0,
    ):
        self.name = "CPSAT-RH"
        self.time_limit, self.objective = time_limit, objective
        self.repair_estimate, self.seed = repair_estimate, seed
        self.statuses: list[str] = []

    def select(self, sim: FactorySim, candidates: list[Candidate]) -> int:
        jobs, first_op, ready_at = [], {}, {}
        for jid in sorted(sim.released):
            cur = sim.current_op(jid)
            if cur is None:
                continue
            job = sim.jobs[jid]
            if jid in sim.active_machine:  # op running: plan from the next one
                fin = sim.machine_free_at(sim.active_machine[jid], self.repair_estimate)
                nxt = cur.index + 1
                if nxt >= len(job.ops):
                    continue
                first_op[jid], ready_at[jid] = nxt, fin
            else:
                first_op[jid], ready_at[jid] = cur.index, sim.now
            jobs.append(job)
        free = {m: sim.machine_free_at(m, self.repair_estimate) for m in sim.machine_ids}
        sol = solve_jobs(jobs, first_op, ready_at, free, self.objective, self.time_limit, self.seed)
        self.statuses.append(sol.status)

        def rank(i: int) -> tuple:
            c = candidates[i]
            plan = sol.assignment.get(c.op_key)
            if plan is None:
                return (1, 0, c.due, c.job_id)
            m, start = plan
            return (0, start + (0 if m == c.machine_id else 10_000), c.due, c.job_id)

        return min(range(len(candidates)), key=rank)
