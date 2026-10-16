"""Observation construction shared by the Gym env and the trained-policy wrapper.

Observation = [global (5)] + [machines: MAX_MACHINES x 4] + [slots: N_SLOTS x SLOT_F]
Only information a real scheduler has at time `now` is used: released jobs, their known
routings/processing times, and machine status. Not used: unreleased jobs (not even their
count), future breakdowns, total job count.

global : now/horizon, #ready jobs/20, idle-machine frac, down-machine frac, #in-system/40
machine: one-hot(idle,busy,down), ready work eligible on it / 200   (0-padded)
slot   : one feasible (op, machine) candidate, sorted by (due, job, machine); 0 if empty
   proc/25, setup/10, slack=(due-now-work_left)/100 clipped, priority/3, wait/100,
   ops_left/6, work_left/100, proc/best_proc_for_op (>=1; speed penalty), 1 (valid flag)
"""

from __future__ import annotations

import numpy as np

from factory_scheduler.simulation.engine import BUSY, DOWN, IDLE, FactorySim
from factory_scheduler.simulation.model import Candidate

MAX_MACHINES = 8
N_SLOTS = 24
SLOT_F = 9
GLOBAL_F = 5
MACH_F = 4
OBS_DIM = GLOBAL_F + MAX_MACHINES * MACH_F + N_SLOTS * SLOT_F


def ordered_candidates(sim: FactorySim) -> list[Candidate]:
    """Deterministic candidate order; the first N_SLOTS are visible to the agent."""
    return sorted(sim.candidates(), key=lambda c: (c.due, c.job_id, c.machine_id))


def build_obs(sim: FactorySim, cands: list[Candidate]) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    m_ids = sim.machine_ids
    if len(m_ids) > MAX_MACHINES:
        raise ValueError(f"at most {MAX_MACHINES} machines supported")
    ready = sim.ready_jobs()
    n_sys = len(sim.released) - len(sim.completion)
    obs[0] = sim.now / sim.inst.horizon
    obs[1] = min(len(ready) / 20, 3)
    obs[2] = sum(sim.status[m] == IDLE for m in m_ids) / len(m_ids)
    obs[3] = sum(sim.status[m] == DOWN for m in m_ids) / len(m_ids)
    obs[4] = min(n_sys / 40, 3)
    work_on = {m: 0.0 for m in m_ids}
    for j in ready:
        op = sim.current_op(j)
        assert op is not None
        for m, p in op.proc_times.items():
            work_on[m] += p
    for i, m in enumerate(m_ids):
        base = GLOBAL_F + i * MACH_F
        st = sim.status[m]
        obs[base + (0 if st == IDLE else 1 if st == BUSY else 2)] = 1.0
        obs[base + 3] = min(work_on[m] / 200, 5)
    s0 = GLOBAL_F + MAX_MACHINES * MACH_F
    for i, c in enumerate(cands[:N_SLOTS]):
        op = sim.jobs[c.job_id].ops[c.op_index]
        best = min(op.proc_times.values())
        slack = (c.due - sim.now - c.work_left) / 100
        obs[s0 + i * SLOT_F : s0 + (i + 1) * SLOT_F] = [
            c.proc_time / 25,
            c.setup / 10,
            float(np.clip(slack, -5, 5)),
            c.priority / 3,
            (sim.now - c.ready_since) / 100,
            c.n_ops_left / 6,
            c.work_left / 100,
            c.proc_time / best,
            1.0,
        ]
    return obs


def action_mask(cands: list[Candidate]) -> np.ndarray:
    mask = np.zeros(N_SLOTS, dtype=bool)
    mask[: min(len(cands), N_SLOTS)] = True
    return mask
