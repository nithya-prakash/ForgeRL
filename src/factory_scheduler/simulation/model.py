"""Plain data model shared by simulator, policies, optimizer and RL (no SimPy here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# All times are integer simulation minutes so that the simulator, the validator and
# CP-SAT agree exactly (no float tolerance issues).


@dataclass(frozen=True)
class MachineSpec:
    id: int
    type: str
    run_cost: float = 1.0  # cost per minute while setting up / processing
    idle_cost: float = 0.2  # cost per minute while up and idle
    down_cost: float = 0.0  # cost per minute while broken down (repair labour)


@dataclass(frozen=True)
class OperationSpec:
    job_id: int
    index: int  # position in the job's routing
    family: int  # product family -> sequence-dependent setups
    proc_times: dict[int, int]  # eligible machine id -> processing minutes

    @property
    def key(self) -> tuple[int, int]:
        return (self.job_id, self.index)


@dataclass(frozen=True)
class JobSpec:
    id: int
    release: int
    due: int
    priority: int  # weight in weighted tardiness (1 = normal, higher = more important)
    ops: tuple[OperationSpec, ...]


@dataclass(frozen=True)
class Breakdown:
    machine_id: int
    start: int
    duration: int

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class Instance:
    """One fully realised scenario: jobs, machines and the breakdown realisation.

    Breakdowns are pre-sampled in absolute time, so every policy faces identical
    failures (matched evaluation). Policies never see them before they occur.
    """

    name: str
    seed: int
    machines: tuple[MachineSpec, ...]
    jobs: tuple[JobSpec, ...]
    setup_time: int  # setup minutes when the next op's family differs from the last
    breakdowns: tuple[Breakdown, ...]
    horizon: int
    tardiness_cost: float = 1.0  # cost per priority-weighted minute of tardiness
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        mids = {m.id for m in self.machines}
        if len(mids) != len(self.machines):
            raise ValueError("duplicate machine ids")
        if len({j.id for j in self.jobs}) != len(self.jobs):
            raise ValueError("duplicate job ids")
        for j in self.jobs:
            if not j.ops:
                raise ValueError(f"job {j.id} has no operations")
            for k, op in enumerate(j.ops):
                if op.index != k or op.job_id != j.id:
                    raise ValueError(f"bad op indexing in job {j.id}")
                if not op.proc_times or not set(op.proc_times) <= mids:
                    raise ValueError(f"op {op.key} has no valid eligible machine")
                if min(op.proc_times.values()) <= 0:
                    raise ValueError(f"op {op.key} has non-positive processing time")


@dataclass(frozen=True)
class Segment:
    """A contiguous interval of work on a machine. kind is 'setup' or 'process'."""

    job_id: int
    op_index: int
    machine_id: int
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class Candidate:
    """A feasible (operation, machine) pairing offered at a decision point."""

    job_id: int
    op_index: int
    machine_id: int
    proc_time: int
    setup: int
    ready_since: int  # when the op became ready (previous op done / job released)
    n_ops_left: int  # including this op
    work_left: int  # sum of min proc times of this and later ops
    due: int
    priority: int
    release: int

    @property
    def op_key(self) -> tuple[int, int]:
        return (self.job_id, self.op_index)
