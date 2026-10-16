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
