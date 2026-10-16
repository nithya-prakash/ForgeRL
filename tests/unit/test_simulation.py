import pytest
from helpers import make_instance

from factory_scheduler.config import load_scenarios
from factory_scheduler.policies.base import run_policy
from factory_scheduler.simulation.engine import FactorySim, InvalidDispatch
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.metrics import compute_metrics
from factory_scheduler.simulation.model import Breakdown
from factory_scheduler.simulation.validate import validate_schedule


class First:
    name = "first"

    def select(self, sim, c):
        return 0


def test_two_jobs_one_machine_manual():
    # J0: 5 min, J1: 3 min on the only machine, FIFO order J0 then J1.
    inst = make_instance([(0, 6, 1, [(0, {0: 5})]), (0, 9, 2, [(0, {0: 3})])], n_machines=1)
    res = run_policy(inst, First())
    assert res.completion == {0: 5, 1: 8}
    m = compute_metrics(res)
    assert m["makespan"] == 8 and m["mean_flow_time"] == 6.5
    assert m["mean_tardiness"] == 0 and m["on_time_pct"] == 100  # J0 C=5<=6, J1 C=8<=9
    assert m["mean_utilization"] == 1.0
    assert validate_schedule(res) == []


def test_tardiness_and_cost_manual():
    inst = make_instance([(0, 3, 2, [(0, {0: 5})])], n_machines=1)  # late by 2, weight 2
    m = compute_metrics(run_policy(inst, First()))
    assert m["max_tardiness"] == 2 and m["weighted_tardiness"] == 4 and m["on_time_pct"] == 0
    assert m["total_cost"] == 5 * 1.0 + 0 * 0.2 + 4  # run + idle + tardiness cost


def test_precedence_and_routing():
    inst = make_instance([(0, 99, 1, [(0, {0: 4}), (0, {1: 6})])])
    res = run_policy(inst, First())
    assert [(s.machine_id, s.start, s.end) for s in res.segments] == [(0, 0, 4), (1, 4, 10)]


def test_setup_applied_on_family_change_only():
    inst = make_instance(
        [(0, 99, 1, [(0, {0: 5})]), (0, 99, 1, [(1, {0: 5})]), (0, 99, 1, [(1, {0: 5})])],
        n_machines=1,
        setup=3,
    )
    res = run_policy(inst, First())
    assert res.completion == {0: 5, 1: 13, 2: 18}  # 5, +3+5, +0+5
    assert compute_metrics(res)["total_setup_time"] == 3
    assert validate_schedule(res) == []


def test_dynamic_release_respected():
    inst = make_instance([(10, 99, 1, [(0, {0: 4})])], n_machines=1)
    res = run_policy(inst, First())
    assert res.segments[0].start == 10 and res.completion[0] == 14


def test_breakdown_during_processing_preempt_resume():
    inst = make_instance(
        [(0, 99, 1, [(0, {0: 10})])], n_machines=1, breakdowns=[Breakdown(0, 4, 6)]
    )
    res = run_policy(inst, First())
    assert [(s.start, s.end) for s in res.segments] == [(0, 4), (10, 16)]
    assert res.completion[0] == 16 and res.downtime == [(0, 4, 10)]
    m = compute_metrics(res)
    assert m["total_downtime"] == 6 and m["mean_utilization"] == 1.0  # 10 busy / (16-6)
    assert validate_schedule(res) == []


def test_breakdown_on_idle_machine_blocks_dispatch():
    inst = make_instance(
        [(5, 99, 1, [(0, {0: 4})])], n_machines=1, breakdowns=[Breakdown(0, 3, 10)]
    )  # down [3,13)
    res = run_policy(inst, First())
    assert res.segments[0].start == 13


def test_breakdown_at_completion_instant_is_consistent():
    inst = make_instance(
        [(0, 99, 1, [(0, {0: 5}), (0, {0: 5})])], n_machines=1, breakdowns=[Breakdown(0, 5, 3)]
    )
    res = run_policy(inst, First())
    assert res.completion[0] == 13 and validate_schedule(res) == []


def test_invalid_dispatches_rejected():
    inst = make_instance([(0, 99, 1, [(0, {0: 4}), (0, {1: 4})])])
    sim = FactorySim(inst)
    assert sim.advance()
    with pytest.raises(InvalidDispatch):
        sim.dispatch(0, 1)  # not eligible for op 0
    sim.dispatch(0, 0)
    with pytest.raises(InvalidDispatch):
        sim.dispatch(0, 0)  # job already active


def test_zero_jobs():
    inst = make_instance([])
    res = run_policy(inst, First())
    m = compute_metrics(res)
    assert res.end_time == 0 and m["n_jobs"] == 0 and m["on_time_pct"] == 100


def test_horizon_truncation_never_counts_unfinished_as_done():
    inst = make_instance([(0, 5, 1, [(0, {0: 50})])], n_machines=1, horizon=20)
    res = run_policy(inst, First())
    assert res.truncated and res.unfinished == [0] and res.completion == {}
    m = compute_metrics(res)
    assert m["n_unfinished"] == 1 and m["n_completed"] == 0 and m["on_time_pct"] == 0
    assert m["weighted_tardiness"] == 15 and validate_schedule(res) == []


def test_unequal_op_counts_and_contention():
    inst = make_instance(
        [
            (0, 99, 1, [(0, {0: 3})]),
            (0, 99, 1, [(0, {0: 3}), (0, {0: 3}), (0, {1: 2})]),
        ]
    )
    res = run_policy(inst, First())
    assert validate_schedule(res) == [] and len(res.completion) == 2


def test_validator_catches_corruption():
    inst = make_instance([(0, 99, 1, [(0, {0: 4}), (0, {0: 4})])], n_machines=1)
    res = run_policy(inst, First())
    from dataclasses import replace

    bad = replace(res, segments=[replace(res.segments[1], start=2, end=6), res.segments[0]])
    assert validate_schedule(bad)


def test_deterministic_generation_and_simulation():
    cfg = load_scenarios()["hetero_full"]
    a, b = generate_instance(cfg, 7), generate_instance(cfg, 7)
    assert a == b and a != generate_instance(cfg, 8)
    r1, r2 = run_policy(a, First()), run_policy(b, First())
    assert r1.segments == r2.segments and r1.completion == r2.completion


@pytest.mark.parametrize(
    "name", ["stable", "breakdowns", "dynamic_arrivals", "hetero_full", "stress"]
)
def test_all_scenarios_valid(name):
    inst = generate_instance(load_scenarios()[name], 3)
    res = run_policy(inst, First())
    assert validate_schedule(res) == []
    assert len(res.completion) + len(res.unfinished) == len(inst.jobs)


def test_back_to_back_breakdowns_do_not_crash():
    inst = make_instance(
        [(0, 99, 1, [(0, {0: 10})])],
        n_machines=1,
        breakdowns=[Breakdown(0, 4, 3), Breakdown(0, 7, 3)],
    )
    res = run_policy(inst, First())
    assert res.completion[0] == 16 and validate_schedule(res) == []  # 4 done + 6 down + 6 left
