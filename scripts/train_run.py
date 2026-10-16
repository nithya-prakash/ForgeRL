"""Usage: python scripts/train_run.py RUN_NAME TIMESTEPS FLOW_W TARDY_W [SEED]"""

import logging
import sys

from factory_scheduler.rl.train import train

logging.basicConfig(level=logging.INFO)
name, steps, fw, tw = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
train(
    overrides={
        "run_name": name,
        "total_timesteps": steps,
        "eval_freq": 100_000,
        "seed": seed,
        "reward": {"flow_weight": fw, "tardiness_weight": tw, "reward_scale": 500.0},
    }
)
