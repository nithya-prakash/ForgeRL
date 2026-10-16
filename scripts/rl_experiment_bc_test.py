"""Test-split REPORTING of the E1/E2 models (no selection or tuning on these data).

Rebuilds the cloned-ATC model deterministically (same data, seed, epochs as
rl_experiment_bc.py), checks it reproduces the saved validation score, then evaluates the clone
and the saved fine-tuned model on the 100 test instances (5 scenarios x 20 seeds), paired vs ATC.
Writes data/results/experiments/bc_finetune/test_summary.json.
"""

import json

import pandas as pd

from factory_scheduler.config import RESULTS_DIR, load_scenarios
from factory_scheduler.evaluation.benchmark import run_grid, split_seeds
from factory_scheduler.evaluation.stats import paired_compare
from factory_scheduler.policies.heuristics import get_policy
from factory_scheduler.rl.imitation import clone, collect, new_model
from factory_scheduler.rl.policy import PPOPolicy

D = RESULTS_DIR / "experiments" / "bc_finetune"
scen = load_scenarios()
train_seeds, val_seeds, test_seeds = (
    split_seeds(s, n) for s, n in (("train", 200), ("val", 10), ("test", 20))
)
reward = {"flow_weight": 0.2, "tardiness_weight": 2.0, "reward_scale": 500.0}
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
data = collect(get_policy("ATC"), scen, train_seeds[:60])
atc = run_grid([get_policy("ATC")], scen, test_seeds, "test")
frames, out = [atc], {"atc_test_mean": float(atc.weighted_tardiness.mean()), "seeds": {}}
for seed in range(3):
    saved = json.loads((D / f"seed{seed}.json").read_text())
    model = new_model(scen, train_seeds, reward, seed, **ppo_kw)
    clone(model, data, seed=seed)
    v = run_grid([PPOPolicy(model, "BC")], scen, val_seeds, "val").weighted_tardiness.mean()
    out["seeds"][seed] = {
        "bc_val_rebuilt": float(v),
        "bc_val_saved": saved["BC_val"]["weighted_tardiness"],
    }
    frames.append(run_grid([PPOPolicy(model, f"BC_s{seed}")], scen, test_seeds, "test"))
    frames.append(
        run_grid([PPOPolicy(D / f"seed{seed}_final.zip", f"FT_s{seed}")], scen, test_seeds, "test")
    )
df = pd.concat(frames, ignore_index=True)
df.to_csv(D / "test_runs.csv", index=False)
for kind in ("BC", "FT"):
    pols = [f"{kind}_s{s}" for s in range(3)]
    sub = df[df.policy.isin(pols)]
    mean = float(sub.groupby("policy").weighted_tardiness.mean().mean())
    avg = (
        sub.groupby(["scenario", "seed"])
        .weighted_tardiness.mean()
        .reset_index()
        .assign(policy=kind)
    )
    r = paired_compare(pd.concat([avg, df[df.policy == "ATC"]]), kind, "ATC", "weighted_tardiness")
    out[kind] = {
        "per_seed_test_mean": {
            p: float(sub[sub.policy == p].weighted_tardiness.mean()) for p in pols
        },
        "mean_over_seeds": mean,
        "pct_vs_atc": 100 * (mean / out["atc_test_mean"] - 1),
        **r,
    }
(D / "test_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1))
print(
    {
        s: (round(x["bc_val_rebuilt"], 1), round(x["bc_val_saved"], 1))
        for s, x in out["seeds"].items()
    }
)
