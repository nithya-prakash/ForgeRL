import json

import pandas as pd
from streamlit.testing.v1 import AppTest

import factory_scheduler.dashboard.app as app_mod
from factory_scheduler import cli


def test_benchmark_cli_then_dashboard_shows_exactly_those_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(
        [
            "benchmark",
            "--n-seeds",
            "2",
            "--scenarios",
            "stable",
            "breakdowns",
            "--policies",
            "FIFO",
            "EDD",
            "--name",
            "compare",
        ]
    )
    df = pd.read_csv(tmp_path / "compare.csv")
    meta = json.loads((tmp_path / "compare.meta.json").read_text())
    assert len(df) == 2 * 2 * 2 and meta["seeds"] == [200000, 200001]
    assert list((tmp_path / "schedules").glob("*.json"))

    monkeypatch.setenv("FACTORY_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(app_mod.__file__, default_timeout=90)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Policy Comparison").run()
    assert not at.exception
    shown = {p for t in at.dataframe for p in t.value.get("policy", [])}
    assert shown == {"FIFO", "EDD"}  # only policies present in the saved CSV
    at.sidebar.radio[0].set_value("Schedule Visualization").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Machine Analytics").run()
    assert not at.exception


def test_dashboard_empty_state_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORY_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(app_mod.__file__, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Policy Comparison").run()
    assert not at.exception and any("No saved results" in w.value for w in at.warning)


def test_cpsat_cli_small(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    cli.main(
        ["cpsat", "--n-seeds", "1", "--n-jobs", "5", "--time-limit", "5", "--scenarios", "stable"]
    )
    df = pd.read_csv(tmp_path / "cpsat.csv")
    assert set(df[df.policy == "CPSAT-static"].solver_status) == {"OPTIMAL"}


def test_experiment_page_survives_empty_mlflow_db(tmp_path, monkeypatch):
    import factory_scheduler.dashboard.app as m

    monkeypatch.setattr("factory_scheduler.config.ROOT", tmp_path)
    (tmp_path / "mlflow.db").write_bytes(b"")  # what `touch mlflow.db` creates
    monkeypatch.setenv("FACTORY_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(m.__file__, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Experiment Tracking").run()
    assert not at.exception


def test_policy_page_shows_reference_table_from_saved_csv(tmp_path, monkeypatch):
    seeds = list(range(200000, 200006))
    rows = [
        {
            "policy": p,
            "scenario": sc,
            "seed": s,
            "weighted_tardiness": base + 10 * i + s % 7,
            "mean_flow_time": 1.0,
            "makespan": 1.0,
            "on_time_pct": 50.0,
            "n_unfinished": 0,
        }
        for sc in ("stable", "stress")
        for i, (p, base) in enumerate((("ATC", 100), ("PPO", 140)))
        for s in seeds
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "compare.csv", index=False)
    from factory_scheduler.evaluation.stats import reference_table

    reference_table(pd.DataFrame(rows), "PPO", {"stable": "ATC", "stress": "ATC"}).to_csv(
        tmp_path / "compare_ppo_vs_ref.csv", index=False
    )
    monkeypatch.setenv("FACTORY_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(app_mod.__file__, default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("Policy Comparison").run()
    assert not at.exception
    tables = [t.value for t in at.dataframe if "wilcoxon_p_holm" in t.value.columns]
    assert len(tables) == 1 and list(tables[0].scenario) == ["stable", "stress", "ALL (pooled)"]
