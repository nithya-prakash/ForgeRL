"""PPO training (sb3-contrib MaskablePPO: invalid-action masking) with MLflow tracking.

Isolation: training samples instances from the *train* seed split; checkpoint selection
uses the *val* split; final reporting uses the *test* split elsewhere. Nothing here reads
test seeds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from factory_scheduler.config import CONFIG_DIR, ROOT, load_scenarios, load_yaml
from factory_scheduler.environment.env import FactoryEnv
from factory_scheduler.evaluation.benchmark import run_grid, split_seeds
from factory_scheduler.rl.policy import PPOPolicy
from factory_scheduler.tracking.mlflow_utils import log_file, tracked_run

log = logging.getLogger(__name__)
MODELS = ROOT / "artifacts" / "models"


class ValCallback(BaseCallback):
    """Every eval_freq steps: evaluate on validation instances, keep the best model."""

    def __init__(self, scenarios, val_seeds, out_dir: Path, eval_freq: int):
        super().__init__()
        self.scen, self.val_seeds, self.out, self.freq = scenarios, val_seeds, out_dir, eval_freq
        self.best = float("inf")
        self.history: list[dict] = []
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.freq:
            self._last = self.num_timesteps
            self.evaluate()
        return True

    def evaluate(self) -> None:
        df = run_grid([PPOPolicy(self.model)], self.scen, self.val_seeds, "val")
        wt = float(df.weighted_tardiness.mean())
        row = {
            "timesteps": self.num_timesteps,
            "val_weighted_tardiness": wt,
            "val_mean_flow_time": float(df.mean_flow_time.mean()),
            "val_on_time_pct": float(df.on_time_pct.mean()),
        }
        self.history.append(row)
        for k, v in row.items():
            if k != "timesteps":
                mlflow.log_metric(k, v, step=self.num_timesteps)
        if wt < self.best:
            self.best = wt
            self.model.save(str(self.out / "best"))
        log.info("val @%d: %s", self.num_timesteps, row)


class EpisodeLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep:
                self.rows.append(
                    {"timesteps": self.num_timesteps, "ep_return": ep["r"], "ep_len": ep["l"]}
                )
        return True


def train(config_path: str | Path | None = None, overrides: dict | None = None) -> Path:
    cfg = load_yaml(config_path or CONFIG_DIR / "ppo.yaml")
    cfg.update(overrides or {})
    scen_all = load_scenarios()
    scen = {k: scen_all[k] for k in cfg["train_scenarios"]}
    rw, hp = cfg["reward"], dict(cfg["ppo"])
    net_arch = hp.pop("net_arch")
    train_seeds = split_seeds("train", cfg["n_train_seeds"])
    val_seeds = split_seeds("val", cfg["n_val_seeds"])
    out = MODELS / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    def make(i: int):
        def _f():
            return Monitor(FactoryEnv(scen, train_seeds, **rw))

        return _f

    venv = DummyVecEnv([make(i) for i in range(cfg["n_envs"])])
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        seed=cfg["seed"],
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": net_arch},
        **hp,
    )
    val_cb = ValCallback(scen, val_seeds, out, cfg["eval_freq"])
    ep_cb = EpisodeLog()
    params = {
        **{f"ppo.{k}": v for k, v in hp.items()},
        "net_arch": net_arch,
        "total_timesteps": cfg["total_timesteps"],
        "seed": cfg["seed"],
        "n_envs": cfg["n_envs"],
        "train_scenarios": cfg["train_scenarios"],
        "train_seeds": f"{train_seeds[0]}..{train_seeds[-1]}",
        "val_seeds": f"{val_seeds[0]}..{val_seeds[-1]}",
        **{f"reward.{k}": v for k, v in rw.items()},
    }
    with tracked_run("ppo_training", cfg["run_name"], params, {"stage": "train"}) as run:
        t0 = time.perf_counter()
        model.learn(cfg["total_timesteps"], callback=[val_cb, ep_cb])
        train_s = time.perf_counter() - t0
        val_cb.evaluate()  # final point on the curve
        model.save(str(out / "final"))
        curves = pd.DataFrame(ep_cb.rows)
        curves.to_csv(out / "train_episodes.csv", index=False)
        pd.DataFrame(val_cb.history).to_csv(out / "val_history.csv", index=False)
        meta = {
            "run_name": cfg["run_name"],
            "mlflow_run_id": run.info.run_id,
            "train_seconds": train_s,
            "config": cfg,
            "best_val_weighted_tardiness": val_cb.best,
            "best_checkpoint": str(out / "best.zip"),
            "final_checkpoint": str(out / "final.zip"),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        mlflow.log_metric("train_seconds", train_s)
        mlflow.log_metric("best_val_weighted_tardiness", val_cb.best)
        for f in ("meta.json", "train_episodes.csv", "val_history.csv", "best.zip"):
            log_file(out / f, "model")
    log.info("trained %d steps in %.0fs -> %s", cfg["total_timesteps"], train_s, out)
    return out


def smoke_overrides() -> dict:
    return {
        "run_name": "ppo_smoke",
        "total_timesteps": 1024,
        "n_envs": 1,
        "eval_freq": 512,
        "n_train_seeds": 3,
        "n_val_seeds": 1,
        "ppo": {
            **load_yaml(CONFIG_DIR / "ppo.yaml")["ppo"],
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
        },
    }


_ = np  # numpy re-exported for type checkers
