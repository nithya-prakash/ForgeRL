"""Experiment E1/E2 (docs/rl_failure_analysis.md). Separate from the main benchmark.

Usage: python scripts/rl_experiment_bc.py SEED [FINETUNE_STEPS]
E1: clone ATC into the PPO network (train seeds), evaluate on validation seeds.
E2: PPO fine-tune from the cloned policy (low LR, same tardiness-weighted reward).
Selection uses validation seeds only (10 per scenario). Test seeds are never touched here.
Writes data/results/experiments/bc_finetune/seed{SEED}.json.
"""

import json
import logging
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback

from factory_scheduler.config import RESULTS_DIR, load_scenarios
from factory_scheduler.evaluation.benchmark import run_grid, split_seeds
from factory_scheduler.policies.heuristics import get_policy
from factory_scheduler.rl.imitation import clone, collect, new_model
from factory_scheduler.rl.policy import PPOPolicy

seed = int(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
logging.basicConfig(level=logging.INFO)
scen = load_scenarios()
train_seeds, val_seeds = split_seeds("train", 200), split_seeds("val", 10)
reward = {"flow_weight": 0.2, "tardiness_weight": 2.0, "reward_scale": 500.0}
out = {"seed": seed, "finetune_steps": steps, "reward": reward}


def val(policy) -> dict:
    df = run_grid([policy], scen, val_seeds, "val")
    return {
        "weighted_tardiness": float(df.weighted_tardiness.mean()),
        "per_scenario": df.groupby("scenario").weighted_tardiness.mean().round(1).to_dict(),
        "per_instance": df.sort_values(["scenario", "seed"]).weighted_tardiness.tolist(),
    }


if seed == 0:  # reference values, computed once
    out["ATC_val"] = val(get_policy("ATC"))
    out["EDD_val"] = val(get_policy("EDD"))

data = collect(get_policy("ATC"), scen, train_seeds[:60])
out["bc_samples"] = len(data[2])
ppo_kw = {
    "learning_rate": 5e-5,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 5,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.0,
}
model = new_model(scen, train_seeds, reward, seed, **ppo_kw)
out["bc_losses"] = clone(model, data, seed=seed)
out["BC_val"] = val(PPOPolicy(model, "BC"))
logging.info("BC val %.0f", out["BC_val"]["weighted_tardiness"])


class Track(BaseCallback):
    def __init__(self):
        super().__init__()
        self.hist, self.best, self.best_path = [], float("inf"), None
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= 50_000:
            self._last = self.num_timesteps
            v = val(PPOPolicy(self.model, "FT"))
            self.hist.append({"timesteps": self.num_timesteps, **v})
            logging.info("finetune @%d val %.0f", self.num_timesteps, v["weighted_tardiness"])
        return True


tr = Track()
t0 = time.perf_counter()
model.learn(steps, callback=tr)
out["finetune_seconds"] = time.perf_counter() - t0
out["finetune_history"] = tr.hist
out["FT_final_val"] = val(PPOPolicy(model, "FT"))
d = RESULTS_DIR / "experiments" / "bc_finetune"
d.mkdir(parents=True, exist_ok=True)
model.save(str(d / f"seed{seed}_final"))
(d / f"seed{seed}.json").write_text(json.dumps(out, indent=1))
print(
    json.dumps({k: out[k] for k in ("seed", "bc_samples")}),
    "BC",
    out["BC_val"]["weighted_tardiness"],
    "FT",
    out["FT_final_val"]["weighted_tardiness"],
)
