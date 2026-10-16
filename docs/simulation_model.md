# Simulation model

Simplified digital twin built on synthetic scenarios. It is **not validated against a real factory**.

* Time: integer minutes. Machines have a type, run/idle/down cost rates. Jobs have release, due date, priority weight and a routing of operations; each op has a product family and a per-machine processing time (eligible set). Heterogeneity = machine speed spread plus optional "flex" eligibility on another type at 1.4x time.
* Setup: `setup_time` minutes when the machine's last completed family differs from the op family (first op on a machine: none).
* Breakdowns: per machine, up-time ~ Exp(MTBF), repair ~ Exp(MTTR) (rounded, >=1), pre-sampled in absolute time per instance, so **all policies face identical failures**. Preempt-resume: finished work is kept and the op resumes on the same machine after repair. The scheduler learns of a breakdown only when it happens and never learns the repair end.
* Arrivals: static (all at t=0) or Poisson. A policy sees only released jobs. Future releases, their number, and future failures are never exposed.
* Decision points: when an idle, up machine and a ready op (predecessor finished, job released, machine eligible) coexist. Non-delay: a dispatch is mandatory (no voluntary idling). All simultaneous events are processed before a decision.
* Termination: all jobs done, or horizon. Unfinished jobs are reported, never counted as completed; their tardiness is charged to the horizon. A deadlock raises `DeadlockError`.
* Independent validation (`simulation/validate.py`): no machine overlap, no work during downtime, eligibility, one machine per op, precedence, release times, processing totals, completion times.

## Metrics (T = end time)
makespan = last completion (T if truncated); flow = C - release; tardiness = max(0, C - due), weighted by priority; on-time % counts unfinished as late; utilisation = (setup+process)/(T - repair time), i.e. **excludes downtime**; cost = run*busy + idle*idle + down*down + tardiness_cost*weighted tardiness. Decision latency = wall time of `select()`.
