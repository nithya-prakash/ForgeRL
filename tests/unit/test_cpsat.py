import pytest
from helpers import make_instance

from factory_scheduler.config import load_scenarios
from factory_scheduler.optimization.cpsat import RollingCPSAT, solve_static, static_variant
from factory_scheduler.policies.base import run_policy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.validate import validate_schedule


def test_known_optimum_two_machines():
    # 3 single-op jobs {4,4,4} on 2 machines: optimal makespan 8.
    inst = make_instance([(0, 99, 1, [(0, {0: 4, 1: 4})])] * 3)
    sol, res = solve_static(inst, "makespan", 5)
    assert sol.status == "OPTIMAL" and sol.objective == 8 and sol.gap == 0
    assert validate_schedule(res) == [] and res.end_time == 8


def test_precedence_and_release_respected():
    inst = make_instance([(5, 99, 1, [(0, {0: 3}), (0, {1: 4})])])
    sol, res = solve_static(inst, "makespan", 5)
    assert sol.objective == 12 and validate_schedule(res) == []  # 5 + 3 + 4


def test_rejects_unsupported_instance():
    from factory_scheduler.simulation.model import Breakdown

    inst = make_instance([(0, 9, 1, [(0, {0: 3})])], breakdowns=[Breakdown(0, 1, 1)])
    with pytest.raises(ValueError):
        solve_static(inst)
    assert solve_static(static_variant(inst))[0].status == "OPTIMAL"


def test_weighted_tardiness_objective():
    # one machine; J0 (w=1,p=5,due 5) J1 (w=3,p=5,due 5): best is J1 first -> 1*5 tardiness.
    inst = make_instance([(0, 5, 1, [(0, {0: 5})]), (0, 5, 3, [(0, {0: 5})])], n_machines=1)
    sol, res = solve_static(inst, "weighted_tardiness", 5)
    assert sol.objective == 5 and validate_schedule(res) == []


@pytest.mark.parametrize("seed", [1, 2])
def test_generated_instance_valid_and_rolling_horizon_feasible(seed):
    cfg = load_scenarios()["hetero_full"].replace(n_jobs=8)
    inst = generate_instance(cfg, seed)
    sol, res = solve_static(static_variant(inst), "makespan", 5)
    assert sol.status in ("OPTIMAL", "FEASIBLE") and validate_schedule(res) == []
    rh = run_policy(inst, RollingCPSAT(time_limit=0.2))  # full dynamics incl. breakdowns
    assert validate_schedule(rh) == [] and not rh.unfinished
