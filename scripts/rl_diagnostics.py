"""Cheap diagnostics for docs/rl_failure_analysis.md (validation seeds only).

Measures, per scenario, over decisions taken by ATC: number of feasible candidates, how often
more than N_SLOTS exist, and whether ATC's chosen pair is visible in the agent's slots.
"""

import json

import numpy as np

from factory_scheduler.config import RESULTS_DIR, load_scenarios
from factory_scheduler.environment import features as F
from factory_scheduler.evaluation.benchmark import split_seeds
from factory_scheduler.policies.heuristics import get_policy
from factory_scheduler.simulation.engine import FactorySim
from factory_scheduler.simulation.generator import generate_instance

atc = get_policy("ATC")
out = {}
for name, cfg in load_scenarios().items():
    n_c, vis, decisions, ep_len = [], [], 0, []
    for seed in split_seeds("val", 10):
        sim = FactorySim(generate_instance(cfg, seed))
        k = 0
        while sim.advance():
            cands = sim.candidates()
            i = atc.select(sim, cands)
            ordered = F.ordered_candidates(sim)
            n_c.append(len(cands))
            vis.append(ordered.index(cands[i]) < F.N_SLOTS)
            sim.dispatch(cands[i].job_id, cands[i].machine_id)
            k += 1
        ep_len.append(k)
    out[name] = {
        "mean_candidates": float(np.mean(n_c)),
        "max_candidates": int(np.max(n_c)),
        "frac_decisions_over_slots": float(np.mean(np.array(n_c) > F.N_SLOTS)),
        "frac_atc_choice_visible": float(np.mean(vis)),
        "mean_decisions_per_episode": float(np.mean(ep_len)),
        "frac_single_candidate": float(np.mean(np.array(n_c) == 1)),
    }
(RESULTS_DIR / "rl_diagnostics.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
