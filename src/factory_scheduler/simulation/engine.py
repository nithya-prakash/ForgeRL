"""SimPy discrete-event job-shop simulator with explicit decision points.

The simulator never decides anything. It advances time until at least one idle, working
machine can start a ready operation (a *decision point*), exposes the feasible
`Candidate`s, and waits for `dispatch()`. Heuristics, CP-SAT replays and the RL
environment all drive this same engine.

Semantics (see docs/simulation_model.md):
* non-delay: when a candidate exists a dispatch is mandatory (no voluntary idling);
* breakdowns are preempt-resume: work already done is kept, the op resumes after repair;
* setup is paid when the machine's last completed family differs from the op family;
* only jobs already released are visible; breakdowns are invisible until they occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from factory_scheduler.simulation.model import (
    Candidate,
    Instance,
    JobSpec,
    OperationSpec,
    Segment,
)

IDLE, BUSY, DOWN = "idle", "busy", "down"


class DeadlockError(RuntimeError):
    """Raised if work remains but no future event could ever make progress."""


class InvalidDispatch(ValueError):
    """Raised when a dispatch is infeasible (wrong machine, precedence, busy...)."""


@dataclass
class SimResult:
    instance: Instance
    segments: list[Segment]
    completion: dict[int, int]  # job id -> completion time (completed jobs only)
    unfinished: list[int]
    downtime: list[tuple[int, int, int]]  # (machine, start, end) truncated at end_time
    end_time: int
    truncated: bool  # True if the horizon cut the run before all jobs finished
    n_decisions: int = 0
    decision_latencies: list[float] = field(default_factory=list)


class FactorySim:
    def __init__(self, instance: Instance):
        instance.validate()
        self.inst = instance
        self.env = simpy.Environment()
        self.jobs: dict[int, JobSpec] = {j.id: j for j in instance.jobs}
        self.machine_ids = [m.id for m in instance.machines]
        self.status: dict[int, str] = {m: IDLE for m in self.machine_ids}
        self.last_family: dict[int, int | None] = {m: None for m in self.machine_ids}
        self.released: set[int] = set()
        self.next_op: dict[int, int] = {j.id: 0 for j in instance.jobs}
        self.ready_since: dict[int, int] = {}
        self.op_active: set[int] = set()  # jobs whose current op is being processed
        self.completion: dict[int, int] = {}
        self.segments: list[Segment] = []
        self.downtime: list[tuple[int, int, int]] = []
        self._proc: dict[int, simpy.Process] = {}
        self._repaired: dict[int, simpy.Event] = {}
        self._piece: dict[int, tuple[int, int, int, int, int]] = {}
        self._down_since: dict[int, int] = {}
        self.active_machine: dict[int, int] = {}  # job id -> machine processing its op
        self._finish_at: dict[int, int] = {}  # machine -> projected completion if not interrupted
        self.n_decisions = 0
        self.end_time = 0
        self.truncated = False
        self._finished = False
        for j in instance.jobs:
            self.env.process(self._arrival(j))
        by_machine: dict[int, list] = {m: [] for m in self.machine_ids}
        for b in sorted(instance.breakdowns, key=lambda b: b.start):
            by_machine[b.machine_id].append(b)
        for m, bl in by_machine.items():
            self.env.process(self._breakdowns(m, bl))

    # ------------------------------------------------------------------ views
    @property
    def now(self) -> int:
        return int(self.env.now)

    @property
    def done(self) -> bool:
        return self._finished

    def n_unfinished(self) -> int:
        return len(self.jobs) - len(self.completion)

    def current_op(self, job_id: int) -> OperationSpec | None:
        k = self.next_op[job_id]
        ops = self.jobs[job_id].ops
        return ops[k] if k < len(ops) else None

    def ready_jobs(self) -> list[int]:
        """Released jobs whose next op is waiting (not in process), sorted by id."""
        return sorted(
            j for j in self.released if j not in self.op_active and self.current_op(j) is not None
        )

    def machine_free_at(self, m: int, repair_estimate: int = 0) -> int:
        """Time machine m is projected to be free. Repair end is unknown to a real
        scheduler, so a down machine is assumed free `repair_estimate` minutes from now."""
        if self.status[m] == IDLE:
            return self.now
        if self.status[m] == DOWN:
            return self.now + repair_estimate
        return self._finish_at[m]

    def setup_for(self, machine_id: int, family: int) -> int:
        last = self.last_family[machine_id]
        return self.inst.setup_time if last is not None and last != family else 0

    def candidates(self) -> list[Candidate]:
        out: list[Candidate] = []
        for j in self.ready_jobs():
            job, op = self.jobs[j], self.current_op(j)
            assert op is not None
            left = job.ops[op.index :]
            work_left = sum(min(o.proc_times.values()) for o in left)
            for m in sorted(op.proc_times):
                if self.status[m] == IDLE:
                    out.append(
                        Candidate(
                            j,
                            op.index,
                            m,
                            op.proc_times[m],
                            self.setup_for(m, op.family),
                            self.ready_since[j],
                            len(left),
                            work_left,
                            job.due,
                            job.priority,
                            job.release,
                        )
                    )
        return out

    # --------------------------------------------------------------- control
    def advance(self) -> bool:
        """Run until a decision is needed (True) or the episode is over (False)."""
        if self._finished:
            return False
        horizon = self.inst.horizon
        while True:
            while self.env.peek() == self.env.now:
                self.env.step()
            if len(self.completion) == len(self.jobs):
                self._finish(truncated=False)
                return False
            if self.candidates():
                return True
            nxt = self.env.peek()
            if nxt == float("inf"):
                raise DeadlockError(f"{self.n_unfinished()} jobs unfinished at t={self.now}")
            if nxt >= horizon:
                self._finish(truncated=True)
                return False
            self.env.step()

    def dispatch(self, job_id: int, machine_id: int) -> None:
        if self._finished:
            raise InvalidDispatch("episode is over")
        op = self.current_op(job_id) if job_id in self.released else None
        if op is None or job_id in self.op_active:
            raise InvalidDispatch(f"job {job_id} has no ready operation")
        if machine_id not in op.proc_times:
            raise InvalidDispatch(f"machine {machine_id} not eligible for op {op.key}")
        if self.status[machine_id] != IDLE:
            raise InvalidDispatch(f"machine {machine_id} is {self.status[machine_id]}")
        self.n_decisions += 1
        self.status[machine_id] = BUSY
        self.op_active.add(job_id)
        self.active_machine[job_id] = machine_id
        self._proc[machine_id] = self.env.process(self._run_op(op, machine_id))

    # ------------------------------------------------------------- processes
    def _arrival(self, job: JobSpec):
        if job.release > 0:
            yield self.env.timeout(job.release)
        self.released.add(job.id)
        self.ready_since[job.id] = self.now

    def _record(self, op: OperationSpec, m: int, a: int, b: int, off: int, setup: int):
        """Split the work interval [a,b) whose first minute is work offset `off`."""
        if b <= a:
            return
        s_end = max(0, min(setup - off, b - a))
        if s_end > 0:
            self.segments.append(Segment(op.job_id, op.index, m, a, a + s_end, "setup"))
        if a + s_end < b:
            self.segments.append(Segment(op.job_id, op.index, m, a + s_end, b, "process"))

    def _run_op(self, op: OperationSpec, m: int):
        setup = self.setup_for(m, op.family)
        total = setup + op.proc_times[m]
        done = 0
        while done < total:
            start = self.now
            self._piece[m] = (start, done, setup, op.job_id, op.index)
            self._finish_at[m] = start + total - done
            try:
                yield self.env.timeout(total - done)
            except simpy.Interrupt:
                del self._piece[m]
                self._record(op, m, start, self.now, done, setup)
                done += self.now - start
                while done < total:  # back-to-back failures may interrupt the wait itself
                    try:
                        yield self._repaired[m]
                        break
                    except simpy.Interrupt:
                        continue
                continue
            del self._piece[m]
            self._record(op, m, start, self.now, done, setup)
            done = total
        del self._proc[m]
        self.active_machine.pop(op.job_id, None)
        self._finish_at.pop(m, None)
        self.last_family[m] = op.family
        if self.status[m] != DOWN:  # op finished exactly as the machine failed
            self.status[m] = IDLE
        self.op_active.discard(op.job_id)
        self.next_op[op.job_id] += 1
        self.ready_since[op.job_id] = self.now
        if self.next_op[op.job_id] == len(self.jobs[op.job_id].ops):
            self.completion[op.job_id] = self.now

    def _breakdowns(self, m: int, bl):
        for b in bl:
            if b.start > self.now:
                yield self.env.timeout(b.start - self.now)
            self._repaired[m] = self.env.event()
            self._down_since[m] = self.now
            proc = self._proc.get(m)
            self.status[m] = DOWN
            if proc is not None:
                proc.interrupt()
            yield self.env.timeout(b.duration)
            self.downtime.append((m, self._down_since.pop(m), self.now))
            self.status[m] = BUSY if m in self._proc else IDLE
            self._repaired[m].succeed()

    # ---------------------------------------------------------------- finish
    def _finish(self, truncated: bool) -> None:
        self._finished = True
        self.truncated = truncated
        self.end_time = self.inst.horizon if truncated else max(self.completion.values(), default=0)
        if truncated:  # close in-flight pieces and open downtime at the horizon
            for m, (start, done, setup, j, k) in list(self._piece.items()):
                self._record(self.jobs[j].ops[k], m, start, self.end_time, done, setup)
            for m, since in self._down_since.items():
                self.downtime.append((m, since, self.end_time))

    def result(self) -> SimResult:
        if not self._finished:
            raise RuntimeError("episode not finished")
        dt = [(m, a, min(b, self.end_time)) for m, a, b in self.downtime if a < self.end_time]
        return SimResult(
            self.inst,
            sorted(self.segments, key=lambda s: (s.start, s.machine_id)),
            dict(self.completion),
            sorted(j for j in self.jobs if j not in self.completion),
            sorted(dt),
            self.end_time,
            self.truncated,
            self.n_decisions,
        )
