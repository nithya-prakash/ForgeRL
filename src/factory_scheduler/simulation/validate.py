"""Independent schedule validator: checks a SimResult's segments against the Instance
using only the instance data, never the code that produced the schedule."""

from __future__ import annotations

from collections import defaultdict

from factory_scheduler.simulation.engine import SimResult


def validate_schedule(res: SimResult, allow_setup_free: bool = False) -> list[str]:
    """Return a list of violations (empty = valid)."""
    inst, errs = res.instance, []
    jobs = {j.id: j for j in inst.jobs}
    per_machine = defaultdict(list)
    per_op = defaultdict(list)
    for s in res.segments:
        if s.end <= s.start:
            errs.append(f"empty/negative segment {s}")
        if s.machine_id not in {m.id for m in inst.machines}:
            errs.append(f"unknown machine in {s}")
        per_machine[s.machine_id].append(s)
        per_op[(s.job_id, s.op_index)].append(s)
    for m, segs in per_machine.items():
        segs.sort(key=lambda s: s.start)
        for a, b in zip(segs, segs[1:]):
            if b.start < a.end:
                errs.append(f"machine {m} overlap: {a} / {b}")
    for m, d0, d1 in res.downtime:
        for s in per_machine.get(m, []):
            if s.start < d1 and d0 < s.end:
                errs.append(f"machine {m} works during downtime [{d0},{d1}): {s}")
    for jid, job in jobs.items():
        prev_end = job.release
        for op in job.ops:
            segs = sorted(per_op.get(op.key, []), key=lambda s: s.start)
            completed = jid in res.completion
            if not segs:
                if completed:
                    errs.append(f"completed job {jid} has no segments for op {op.index}")
                continue
            ms = {s.machine_id for s in segs}
            if len(ms) != 1 or not ms <= set(op.proc_times):
                errs.append(f"op {op.key}: bad machine set {ms}")
                continue
            if segs[0].start < prev_end:
                errs.append(f"op {op.key} starts at {segs[0].start} before ready {prev_end}")
            proc = sum(s.end - s.start for s in segs if s.kind == "process")
            need = op.proc_times[next(iter(ms))]
            op_done = completed or any(j == jid and k > op.index for (j, k) in per_op)
            if proc > need or (op_done and proc != need):
                errs.append(f"op {op.key}: processed {proc} != required {need}")
            prev_end = segs[-1].end
        if jid in res.completion and res.completion[jid] != prev_end:
            errs.append(f"job {jid}: completion {res.completion[jid]} != last segment {prev_end}")
    return errs
