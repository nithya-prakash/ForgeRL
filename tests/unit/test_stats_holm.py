import pandas as pd

from factory_scheduler.evaluation.stats import holm_adjust, reference_table


def test_holm_matches_hand_calculation():
    # sorted p: .01 (x3), .02 (x2), .04 (x1) -> .03, .04, .04 (monotone max), original order kept
    assert holm_adjust([0.04, 0.01, 0.02]) == [0.04, 0.03, 0.04]
    assert holm_adjust([0.5, 0.9]) == [1.0, 1.0]  # capped at 1
    assert holm_adjust([]) == []


def test_reference_table_per_scenario_and_pooled():
    rows = []
    for sc, shift in (("a", 10.0), ("b", 30.0)):
        for seed in range(8):
            rows.append({"policy": "REFRULE", "scenario": sc, "seed": seed, "m": 100.0 + seed})
            rows.append({"policy": "X", "scenario": sc, "seed": seed, "m": 100.0 + seed + shift})
    t = reference_table(pd.DataFrame(rows), "X", {"a": "REFRULE", "b": "REFRULE"}, metric="m")
    assert list(t.scenario) == ["a", "b", "ALL (pooled)"]
    assert t.loc[0, "mean_diff"] == 10 and t.loc[1, "mean_diff"] == 30
    assert t.loc[2, "mean_diff"] == 20 and t.loc[2, "n"] == 16
    assert pd.isna(t.loc[2, "wilcoxon_p"])  # pooled row reports a CI only
    base_a = 100 + sum(range(8)) / 8
    assert abs(t.loc[0, "pct_diff_vs_ref"] - 100 * 10 / base_a) < 1e-9
    assert (t.loc[:1, "wilcoxon_p_holm"] >= t.loc[:1, "wilcoxon_p"]).all()
