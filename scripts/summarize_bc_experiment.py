"""Summarise data/results/experiments/bc_finetune/seed*.json (validation seeds only)."""

import json

import numpy as np
from scipy import stats

from factory_scheduler.config import RESULTS_DIR

d = [json.load(open(RESULTS_DIR / f"experiments/bc_finetune/seed{s}.json")) for s in range(3)]
atc = np.array(d[0]["ATC_val"]["per_instance"])
print("ATC val mean", round(atc.mean(), 1), "EDD", round(d[0]["EDD_val"]["weighted_tardiness"], 1))
for name, key in (("BC (clone of ATC)", "BC_val"), ("PPO fine-tuned from BC", "FT_final_val")):
    per_seed = [x[key]["weighted_tardiness"] for x in d]
    pooled = np.mean([np.array(x[key]["per_instance"]) for x in d], axis=0)
    diff = pooled - atc
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    h = stats.t.ppf(0.975, len(diff) - 1) * se
    print(
        f"{name}: seeds {[round(v) for v in per_seed]} mean {np.mean(per_seed):.0f} "
        f"({100 * (np.mean(per_seed) / atc.mean() - 1):+.1f}% vs ATC); paired diff "
        f"{diff.mean():+.0f} [{diff.mean() - h:+.0f}, {diff.mean() + h:+.0f}] over {len(diff)} "
        f"instances; Wilcoxon p={stats.wilcoxon(diff).pvalue:.2g}"
    )
