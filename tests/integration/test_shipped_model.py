import hashlib
from pathlib import Path

from factory_scheduler.config import ROOT, load_scenarios
from factory_scheduler.policies.base import run_policy
from factory_scheduler.rl.policy import PPOPolicy
from factory_scheduler.simulation.generator import generate_instance
from factory_scheduler.simulation.validate import validate_schedule

MODEL = ROOT / "models" / "ppo_1m_tardy_best.zip"


def test_shipped_model_matches_checksum_and_schedules_validly():
    expected = (ROOT / "models" / "SHA256SUMS").read_text().split()[0]
    assert hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() == expected
    inst = generate_instance(load_scenarios()["hetero_full"].replace(n_jobs=10), 200_000)
    res = run_policy(inst, PPOPolicy(MODEL))
    assert validate_schedule(res) == [] and not res.unfinished
