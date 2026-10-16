from pathlib import Path

import factory_scheduler.tracking.mlflow_utils as mu
from factory_scheduler.config import load_scenarios
from factory_scheduler.policies.base import run_policy
from factory_scheduler.rl import train as T
from factory_scheduler.rl.policy import PPOPolicy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.validate import validate_schedule


def test_ppo_trains_saves_reloads_and_schedules_validly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(T, "MODELS", tmp_path / "models")
    monkeypatch.setattr(mu, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    ov = T.smoke_overrides()
    ov["train_scenarios"] = ["stable", "breakdowns"]
    out = T.train(overrides=ov)
    assert (out / "final.zip").exists() and (out / "meta.json").exists()
    pol = PPOPolicy(out / "final.zip")  # reload from disk
    inst = generate_instance(load_scenarios()["hetero_full"].replace(n_jobs=8), 300_000)
    res = run_policy(inst, pol)
    assert validate_schedule(res) == [] and not res.unfinished
