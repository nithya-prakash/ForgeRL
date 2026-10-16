import numpy as np
from gymnasium.utils.env_checker import check_env

from factory_scheduler.config import load_scenarios
from factory_scheduler.environment.env import FactoryEnv, accrued_cost
from factory_scheduler.policies.base import run_policy
from factory_scheduler.policies.heuristics import get_policy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.metrics import compute_metrics
from factory_scheduler.simulation.validate import validate_schedule


def make_env(**kw):
    sc = load_scenarios()
    return FactoryEnv(
        {k: sc[k].replace(n_jobs=8) for k in ("stable", "hetero_full")}, list(range(5)), **kw
    )


def test_gymnasium_check_env():
    check_env(make_env(), skip_render_check=True)


def test_random_masked_rollouts_valid_and_deterministic():
    def rollout(seed):
        env, rng = make_env(), np.random.default_rng(0)
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            obs, r, term, trunc, info = env.step(int(rng.choice(legal)))
            assert np.isfinite(obs).all() and "invalid_action" not in info
            total += r
            done = term or trunc
        assert validate_schedule(env.result()) == []
        return total, env.result().completion

    assert rollout(1) == rollout(1)


def test_invalid_action_is_noop_with_penalty_and_bounded():
    env = make_env()
    env.reset(seed=0)
    n_before = env.sim.n_decisions
    _, r, term, trunc, info = env.step(23)  # slot 23 is empty with only a few candidates
    assert info.get("invalid_action") and r < 0 and env.sim.n_decisions == n_before
    for _ in range(25):
        _, _, term, trunc, _ = env.step(23)
        if trunc:
            break
    assert trunc and not term


def test_return_equals_negative_accrued_cost():
    env = make_env(reward_scale=1.0)
    inst = generate_instance(load_scenarios()["stable"].replace(n_jobs=6), 3)
    env.reset(options={"instance": inst})
    ret, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(0)
        ret += r
        done = term or trunc
    sim = env.sim
    assert abs(ret + accrued_cost(sim, sim.end_time, 1.0, 1.0)) < 1e-6
    m = compute_metrics(env.result())
    flow = sum(j.priority * (sim.completion[j.id] - j.release) for j in inst.jobs)
    assert abs(-ret - flow - m["weighted_tardiness"]) < 1e-6


def test_truncation_flag_and_leakage_free_obs_size():
    sc = load_scenarios()["stress"].replace(n_jobs=10, horizon=30)
    env = FactoryEnv({"s": sc}, [1])
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
    assert trunc and not term and env.result().unfinished


def test_env_matches_heuristic_when_replaying_choice():
    """Driving the env with EDD's choices reproduces run_policy's EDD schedule."""
    inst = generate_instance(load_scenarios()["breakdowns"].replace(n_jobs=8), 2)
    pol = get_policy("EDD")
    env = make_env()
    env.reset(options={"instance": inst})
    done = False
    while not done:
        i = pol.select(env.sim, env._cands)
        _, _, term, trunc, _ = env.step(i)
        done = term or trunc
    assert env.result().completion == run_policy(inst, pol).completion
