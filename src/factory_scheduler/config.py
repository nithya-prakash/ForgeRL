"""Configuration loading (YAML) and scenario parameter dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
# Override with FACTORY_RESULTS_DIR (used by tests and to point the dashboard elsewhere).
RESULTS_DIR = Path(os.environ.get("FACTORY_RESULTS_DIR", ROOT / "data" / "results"))


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "custom"
    machine_types: dict[str, int] = field(default_factory=lambda: {"cut": 2, "mill": 2})
    speed_spread: float = 0.0
    flex_prob: float = 0.0
    n_jobs: int = 20
    ops_range: tuple[int, int] = (2, 4)
    proc_range: tuple[int, int] = (5, 25)
    n_families: int = 3
    setup_time: int = 4
    arrival: str = "static"
    arrival_rate: float = 0.0
    due_tightness: float = 3.0
    priorities: tuple[int, ...] = (1, 1, 2, 3)
    mtbf: float = 0.0
    mttr: float = 0.0
    horizon: int = 5000
    tardiness_cost: float = 1.0

    def replace(self, **kw: Any) -> ScenarioConfig:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(kw)
        return ScenarioConfig(**d)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_scenarios(path: str | Path | None = None) -> dict[str, ScenarioConfig]:
    raw = load_yaml(path or CONFIG_DIR / "scenarios.yaml")["scenarios"]
    out = {}
    for name, d in raw.items():
        d = dict(d)
        d["ops_range"], d["proc_range"] = tuple(d["ops_range"]), tuple(d["proc_range"])
        d["priorities"] = tuple(d["priorities"])
        out[name] = ScenarioConfig(name=name, **d)
    return out
