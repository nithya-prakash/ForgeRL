"""Summaries and matched-pair statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summarize(df: pd.DataFrame, metrics: list[str], by=("scenario", "policy")) -> pd.DataFrame:
    """Mean, std and 95% t-interval half-width across seeds."""
    g = df.groupby(list(by))[metrics]
    mean, std, n = g.mean(), g.std(ddof=1), g.count()
    half = std / np.sqrt(n) * stats.t.ppf(0.975, (n - 1).clip(lower=1))
    out = mean.join(std, rsuffix="_std").join(half, rsuffix="_ci95")
    return out.reset_index()


def paired_compare(df: pd.DataFrame, a: str, b: str, metric: str, scenario: str | None = None):
    """Matched-seed comparison of policy a vs b (same instance + breakdown realisation).

    Reports mean difference (a-b), 95% t-CI, paired Cohen's dz, Wilcoxon signed-rank p.
    Negative diff favours a for lower-is-better metrics.
    """
    d = df if scenario is None else df[df.scenario == scenario]
    piv = d.pivot_table(index=["scenario", "seed"], columns="policy", values=metric)
    diff = (piv[a] - piv[b]).dropna()
    n = len(diff)
    if n < 2:
        return {"n": n, "mean_diff": float(diff.mean()) if n else float("nan")}
    sd = float(diff.std(ddof=1))
    half = sd / np.sqrt(n) * stats.t.ppf(0.975, n - 1)
    try:
        p = float(stats.wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
    except ValueError:
        p = float("nan")
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "n": n,
        "mean_diff": float(diff.mean()),
        "ci95_low": float(diff.mean() - half),
        "ci95_high": float(diff.mean() + half),
        "cohen_dz": float(diff.mean() / sd) if sd > 0 else 0.0,
        "wilcoxon_p": p,
        "a_better_frac": float((diff < 0).mean()),
    }
