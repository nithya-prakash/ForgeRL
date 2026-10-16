"""Operational metrics, computed only from a SimResult (never from policy internals).

Definitions (T = result.end_time; for a run that finishes, T = makespan):
  makespan      = max_j C_j over completed jobs (T when truncated at the horizon)
  throughput    = completed jobs / T                           [jobs per minute]
  flow time     = C_j - r_j                                    (completed jobs)
  tardiness     = max(0, C_j - d_j); weighted by job priority w_j
  on_time_pct   = 100 * #{C_j <= d_j} / #jobs                  (unfinished = late)
  utilization_m = busy_m / (T - down_m)  with busy = setup + processing minutes,
                  down_m = repair minutes inside [0, T]; i.e. excludes downtime.
  cost          = sum_m run_cost*busy + idle_cost*(T-busy-down) + down_cost*down
                  + tardiness_cost * sum_j w_j * tardiness_j
  Unfinished jobs are never counted as completed; they are reported separately and their
  tardiness is charged as max(0, T - d_j) (a lower bound) so truncation cannot look good.
"""

from __future__ import annotations

from factory_scheduler.simulation.engine import SimResult


def machine_stats(res: SimResult) -> dict[int, dict[str, float]]:
    T = res.end_time
    out = {m.id: {"setup": 0.0, "process": 0.0, "down": 0.0} for m in res.instance.machines}
    for s in res.segments:
        out[s.machine_id][s.kind] += s.end - s.start
    for m, a, b in res.downtime:
        out[m]["down"] += b - a
    for m, st in out.items():
        st["busy"] = st["setup"] + st["process"]
        up = T - st["down"]
        st["utilization"] = st["busy"] / up if up > 0 else 0.0
        st["idle"] = max(0.0, up - st["busy"])
    return out


def compute_metrics(res: SimResult) -> dict[str, float]:
    inst, T = res.instance, res.end_time
    jobs = {j.id: j for j in inst.jobs}
    n = len(jobs)
    flows, tard, wtard = [], [], []
    on_time = 0
    for jid, c in res.completion.items():
        j = jobs[jid]
        flows.append(c - j.release)
        t = max(0, c - j.due)
        tard.append(t)
        wtard.append(j.priority * t)
        on_time += c <= j.due
    unfinished_pen = sum(jobs[j].priority * max(0, T - jobs[j].due) for j in res.unfinished)
    ms = machine_stats(res)
    cost_by = {m.id: m for m in inst.machines}
    op_cost = sum(
        cost_by[m].run_cost * s["busy"]
        + cost_by[m].idle_cost * s["idle"]
        + cost_by[m].down_cost * s["down"]
        for m, s in ms.items()
    )
    total_w = sum(wtard) + unfinished_pen
    lat = res.decision_latencies
    return {
        "n_jobs": n,
        "n_completed": len(res.completion),
        "n_unfinished": len(res.unfinished),
        "makespan": float(
            T if res.truncated or not res.completion else max(res.completion.values())
        ),
        "throughput": len(res.completion) / T if T > 0 else 0.0,
        "mean_flow_time": sum(flows) / len(flows) if flows else 0.0,
        "mean_tardiness": sum(tard) / len(tard) if tard else 0.0,
        "max_tardiness": float(max(tard, default=0)),
        "weighted_tardiness": float(total_w),
        "on_time_pct": 100.0 * on_time / n if n else 100.0,
        "mean_utilization": sum(s["utilization"] for s in ms.values()) / len(ms),
        "total_downtime": sum(s["down"] for s in ms.values()),
        "total_setup_time": sum(s["setup"] for s in ms.values()),
        "total_cost": op_cost + inst.tardiness_cost * total_w,
        "n_decisions": res.n_decisions,
        "mean_decision_latency_ms": 1000 * sum(lat) / len(lat) if lat else 0.0,
    }
