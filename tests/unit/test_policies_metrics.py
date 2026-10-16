import pandas as pd
from helpers import make_instance
from hypothesis import given, settings
from hypothesis import strategies as st

from factory_scheduler.config import load_scenarios
from factory_scheduler.evaluation.benchmark import run_grid, split_seeds
from factory_scheduler.evaluation.stats import paired_compare
from factory_scheduler.policies.base import run_policy
from factory_scheduler.policies.heuristics import HEURISTICS, get_policy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.validate import validate_schedule


def test_rules_choose_as_documented():
    # two ready jobs on one machine: SPT -> short first, EDD -> earliest due first
    inst = make_instance(
        [(0, 100, 1, [(0, {0: 9})]), (0, 20, 1, [(0, {0: 2})]), (0, 10, 1, [(0, {0: 5})])],
        n_machines=1,
    )
    first = {n: run_policy(inst, get_policy(n)).segments[0].job_id for n in HEURISTICS}
    assert first["SPT"] == 1 and first["EDD"] == 2 and first["FIFO"] == 0
    assert first["LSL"] == 2 and first["CR"] == 2


def test_machine_choice_prefers_faster_machine():
    inst = make_instance([(0, 100, 1, [(0, {0: 10, 1: 4})])])
    for n in HEURISTICS:
        assert run_policy(inst, get_policy(n)).segments[0].machine_id == 1, n


def test_benchmark_policies_see_identical_instances():
    cfg = {"s": load_scenarios()["breakdowns"].replace(n_jobs=10)}
    df = run_grid([get_policy("FIFO"), get_policy("EDD")], cfg, split_seeds("test", 3))
    assert df.groupby("seed").n_jobs.nunique().max() == 1
    assert set(df.policy) == {"FIFO", "EDD"} and (df.n_unfinished == 0).all()
    assert split_seeds("train", 5)[-1] < split_seeds("val", 1)[0] < split_seeds("test", 1)[0]


def test_paired_compare_manual():
    df = pd.DataFrame(
        {
            "policy": ["a", "b"] * 4,
            "scenario": ["s"] * 8,
            "seed": [0, 0, 1, 1, 2, 2, 3, 3],
            "m": [1.0, 2.0, 2.0, 4.0, 3.0, 3.0, 1.0, 5.0],
        }
    )
    r = paired_compare(df, "a", "b", "m")
    assert r["n"] == 4 and abs(r["mean_diff"] - (-1 - 2 + 0 - 4) / 4) < 1e-9


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    pol=st.sampled_from(sorted(HEURISTICS)),
    scen=st.sampled_from(["stable", "breakdowns", "hetero_full"]),
)
def test_property_every_policy_schedule_is_valid_and_complete(seed, pol, scen):
    inst = generate_instance(load_scenarios()[scen].replace(n_jobs=6), seed)
    res = run_policy(inst, get_policy(pol))
    assert validate_schedule(res) == []
    assert len(res.completion) + len(res.unfinished) == 6
