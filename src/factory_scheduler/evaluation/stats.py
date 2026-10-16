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


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across scenarios)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def reference_table(
    df: pd.DataFrame, policy: str, reference: dict[str, str], metric: str = "weighted_tardiness"
) -> pd.DataFrame:
    """`policy` vs a per-scenario reference policy fixed in advance (not chosen on these data).

    One row per scenario (paired over seeds, Holm-adjusted across scenarios) plus a pooled
    row over all (scenario, seed) pairs, which treats instances as independent pairs.
    """
    rows = []
    for sc, ref in reference.items():
        rows.append(
            {"scenario": sc, "reference": ref, **paired_compare(df, policy, ref, metric, sc)}
        )
    adj = holm_adjust([r["wilcoxon_p"] for r in rows])
    for r, p in zip(rows, adj, strict=True):
        r["wilcoxon_p_holm"] = p
        base = df[(df.scenario == r["scenario"]) & (df.policy == r["reference"])][metric].mean()
        r["pct_diff_vs_ref"] = 100 * r["mean_diff"] / base
    ref_rows = pd.concat(
        [
            df[(df.scenario == sc) & (df.policy == ref)].assign(policy="REF")
            for sc, ref in reference.items()
        ]
    )
    pooled = paired_compare(pd.concat([df[df.policy == policy], ref_rows]), policy, "REF", metric)
    pooled.update({"scenario": "ALL (pooled)", "reference": "per-scenario"})
    pooled["wilcoxon_p"] = float("nan")  # pairs are not independent across seeds/scenarios: CI only
    pooled["pct_diff_vs_ref"] = 100 * pooled["mean_diff"] / ref_rows[metric].mean()
    return pd.DataFrame(rows + [pooled])
