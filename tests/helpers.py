from factory_scheduler.simulation.model import (
    Instance,
    JobSpec,
    MachineSpec,
    OperationSpec,
)


def make_instance(jobs, n_machines=2, breakdowns=(), setup=0, horizon=1000, name="toy"):
    """jobs: list of (release, due, prio, [(family, {machine: time}), ...])."""
    js = []
    for jid, (r, d, p, ops) in enumerate(jobs):
        js.append(
            JobSpec(
                jid,
                r,
                d,
                p,
                tuple(OperationSpec(jid, k, fam, dict(pt)) for k, (fam, pt) in enumerate(ops)),
            )
        )
    ms = tuple(MachineSpec(i, "m") for i in range(n_machines))
    return Instance(name, 0, ms, tuple(js), setup, tuple(breakdowns), horizon)
