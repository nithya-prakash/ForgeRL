"""Streamlit dashboard. Every number shown is loaded from saved result files, the MLflow
store, or computed by a clearly-labelled LIVE simulation. Nothing is hard-coded."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from factory_scheduler.config import RESULTS_DIR, ROOT, load_scenarios
from factory_scheduler.evaluation.stats import paired_compare, summarize
from factory_scheduler.visualization.plots import gantt, machine_table, tardiness_chart

RESULTS = Path(os.environ.get("FACTORY_RESULTS_DIR", RESULTS_DIR))  # re-read each run
MODELS = ROOT / "artifacts" / "models"
st.set_page_config(page_title="Smart Factory RL Scheduler", layout="wide")

LOWER_BETTER = {
    "weighted_tardiness": True,
    "mean_flow_time": True,
    "makespan": True,
    "mean_tardiness": True,
    "max_tardiness": True,
    "total_cost": True,
    "on_time_pct": False,
    "throughput": False,
    "mean_utilization": False,
}


@st.cache_data
def load_csv(name: str, root: str = "") -> pd.DataFrame | None:
    p = Path(root or RESULTS) / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def need(df, what: str):
    if df is None:
        st.warning(
            f"No saved results for **{what}**. Run the matching CLI command first "
            "(see README). Nothing is shown rather than made-up numbers."
        )
        st.stop()
    return df


PAGES = [
    "Factory Digital Twin",
    "Policy Comparison",
    "Machine Analytics",
    "Experiment Tracking",
    "Robustness",
    "Schedule Visualization",
]
_qp = st.query_params.get("page", PAGES[0])  # deep link, e.g. ?page=Machine%20Analytics
page = st.sidebar.radio("Page", PAGES, index=PAGES.index(_qp) if _qp in PAGES else 0)
st.sidebar.caption(
    "Times are minutes. Data source is labelled on every page: "
    "LIVE = simulated now; SAVED = loaded from data/results or MLflow."
)

schedules = (
    sorted((RESULTS / "schedules").glob("*.json")) if (RESULTS / "schedules").exists() else []
)


def pick_schedule(key: str) -> dict:
    if not schedules:
        st.warning("No saved schedules. Run `factory-scheduler benchmark` first.")
        st.stop()
    f = st.selectbox(
        "Saved schedule (policy__scenario__seed)", schedules, format_func=lambda p: p.stem, key=key
    )
    return json.loads(Path(f).read_text())


if page == "Factory Digital Twin":
    st.title("Factory Digital Twin")
    scen = load_scenarios()
    name = st.selectbox(
        "Scenario configuration (EXAMPLE CONFIG from configs/scenarios.yaml)", list(scen)
    )
    cfg = scen[name]
    c1, c2 = st.columns(2)
    c1.dataframe(
        pd.DataFrame({"value": {k: str(v) for k, v in vars(cfg).items() if k != "name"}}),
        height=330,
    )
    st.subheader("LIVE simulation")
    seed = c2.number_input("Seed", 0, 10**6, 0)
    from factory_scheduler.policies.base import run_policy
    from factory_scheduler.policies.heuristics import HEURISTICS, get_policy
    from factory_scheduler.simulation.engine import FactorySim
    from factory_scheduler.simulation.generator import generate_instance

    pol = c2.selectbox("Dispatching rule", list(HEURISTICS))
    inst = generate_instance(cfg, int(seed))
    t_view = c2.slider("Inspect time t", 0, int(cfg.horizon // 4), 60)
    c2.caption(
        f"{len(inst.jobs)} jobs, {len(inst.machines)} machines, "
        f"{len(inst.breakdowns)} sampled breakdowns"
    )
    sim = FactorySim(inst)
    p = get_policy(pol)
    while sim.advance() and sim.now <= t_view:
        cs = sim.candidates()
        c = cs[p.select(sim, cs)]
        sim.dispatch(c.job_id, c.machine_id)
    st.markdown(f"**State at t = {sim.now} min** ({pol}, seed {seed})")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "machine": f"M{m.id}",
                    "type": m.type,
                    "state": sim.status[m.id],
                    "last family": sim.last_family[m.id],
                }
                for m in inst.machines
            ]
        ),
        hide_index=True,
    )
    q = [
        {
            "job": j,
            "op": sim.current_op(j).index,  # type: ignore[union-attr]
            "ready since": sim.ready_since[j],
            "due": sim.jobs[j].due,
            "priority": sim.jobs[j].priority,
        }
        for j in sim.ready_jobs()
        if sim.current_op(j)
    ]
    st.markdown(
        f"Waiting queue ({len(q)} ready operations); "
        f"{len(inst.jobs) - len(sim.released)} jobs not yet released (unknown to policy)"
    )
    st.dataframe(pd.DataFrame(q), hide_index=True, height=260)
    res = run_policy(inst, get_policy(pol))
    done = len(res.completion)
    st.progress(done / max(1, len(inst.jobs)), f"full run: {done}/{len(inst.jobs)} jobs completed")

elif page == "Policy Comparison":
    st.title("Scheduling Policy Comparison (SAVED, held-out test seeds)")
    df = need(load_csv("compare", str(RESULTS)), "compare (run `factory-scheduler compare`)")
    metric = st.selectbox("Metric", [m for m in LOWER_BETTER if m in df], index=0)
    chosen = st.multiselect("Scenarios", sorted(df.scenario.unique()), sorted(df.scenario.unique()))
    d = df[df.scenario.isin(chosen)]
    s = summarize(d, [metric])
    st.caption(
        f"{'lower' if LOWER_BETTER[metric] else 'higher'} is better; mean ± 95% CI "
        f"over {d.seed.nunique()} matched seeds"
    )
    overall = s.groupby("policy")[metric].mean().sort_values(ascending=LOWER_BETTER[metric])
    st.markdown(
        "**Overall: unweighted mean over the selected scenarios.** `stress` dominates absolute "
        "tardiness, so also read the per-scenario chart."
    )
    st.dataframe(
        overall.rename(f"mean {metric}").round(2).to_frame().assign(rank=range(1, len(overall) + 1))
    )
    fig = px.bar(
        s, x="scenario", y=metric, color="policy", barmode="group", error_y=f"{metric}_ci95"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(s.round(2), hide_index=True)
    st.subheader("PPO vs. reference rule (paired)")
    pb = load_csv("compare_ppo_vs_ref", str(RESULTS))
    if pb is not None:
        st.caption(
            "Reference rule chosen on validation seeds. Per scenario: n = 20 matched test seeds, "
            "Holm-adjusted across scenarios. The pooled row treats all pairs as independent "
            "(CI only, no p-value). Positive diff = PPO worse (higher tardiness)."
        )
        cols = [
            "scenario",
            "reference",
            "n",
            "mean_diff",
            "pct_diff_vs_ref",
            "ci95_low",
            "ci95_high",
            "wilcoxon_p",
            "wilcoxon_p_holm",
        ]
        show = pb[cols].copy()
        for col in ("wilcoxon_p", "wilcoxon_p_holm"):
            show[col] = show[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1e}")
        st.dataframe(show.round(2), hide_index=True)
    else:
        legacy = load_csv("compare_ppo_vs_best", str(RESULTS))
        if legacy is not None:
            st.caption(
                "Legacy table: reference chosen on the test data (favours the rule), "
                "no multiple-comparison correction."
            )
            st.dataframe(legacy.round(3), hide_index=True)
    a, b = st.columns(2)
    pa = a.selectbox("Policy A", sorted(d.policy.unique()), index=0)
    pbn = b.selectbox("Policy B", sorted(d.policy.unique()), index=1)
    st.caption(
        "Pooled over all selected scenarios and seeds (pairs treated as independent; "
        "the p-value is uncorrected and descriptive only)."
    )
    if pa != pbn:
        st.json(paired_compare(d, pa, pbn, metric))
    cp = load_csv("cpsat", str(RESULTS))
    if cp is not None:
        st.subheader(
            "Offline CP-SAT vs online rules (small static instances, NOT a fair online comparison)"
        )
        mk = cp[cp.objective == "makespan"].groupby("policy").makespan.mean()
        wt = cp[cp.objective == "weighted_tardiness"].groupby("policy").weighted_tardiness.mean()
        st.caption(
            "makespan from the makespan-objective solve; weighted tardiness from the "
            "tardiness-objective solve (CP-SAT optimises one objective at a time)"
        )
        st.dataframe(pd.DataFrame({"mean makespan": mk, "mean weighted tardiness": wt}).round(1))
        st.dataframe(
            cp[cp.policy == "CPSAT-static"]
            .groupby(["objective", "solver_status"])
            .agg(n=("seed", "count"), runtime_s=("solver_runtime_s", "mean"), gap=("gap", "mean"))
            .round(3)
        )

elif page == "Machine Analytics":
    st.title("Machine Analytics (SAVED schedule)")
    sched = pick_schedule("ma")
    mt = machine_table(sched)
    fmt: dict = {c: "{:.0f}" for c in ["process", "setup", "down", "idle"]}
    fmt["utilization"] = "{:.0%}"
    st.dataframe(mt.style.format(fmt))
    st.plotly_chart(
        px.bar(
            mt.reset_index(),
            x="index",
            y=["process", "setup", "down", "idle"],
            labels={"index": "machine", "value": "minutes"},
            color_discrete_map={
                "process": "#2563eb",
                "setup": "#f59e0b",
                "down": "#dc2626",
                "idle": "#d1d5db",
            },
        ),
        use_container_width=True,
    )
    counts: dict[str, int] = {f"M{m['id']}": 0 for m in sched["machines"]}
    for m_id, _, _ in sched["downtime"]:
        counts[f"M{m_id}"] += 1
    st.write("Breakdowns in this run")
    st.dataframe(
        pd.DataFrame(
            {
                "count": counts,
                "mean repair (min)": {
                    k: round(float(mt["down"].astype(float)[k]) / v, 1) if v else 0.0
                    for k, v in counts.items()
                },
            }
        ).T
    )
    st.info(
        f"Bottleneck (highest utilisation, excluding downtime): "
        f"**{mt['utilization'].idxmax()}** ({mt['utilization'].max():.0%}). "
        "Utilisation = (setup+process)/(T − repair time)."
    )

elif page == "Experiment Tracking":
    st.title("Experiment Tracking (MLflow store + saved model metadata)")
    db = ROOT / "mlflow.db"
    has_runs = False
    if db.exists():
        con = sqlite3.connect(db)
        has_runs = bool(
            con.execute("select count(*) from sqlite_master where name='runs'").fetchone()[0]
        )
    if has_runs:
        runs = pd.read_sql(
            """select r.run_uuid, r.name, e.name as experiment, r.status,
              datetime(r.start_time/1000,'unixepoch') as started_utc from runs r
              join experiments e on e.experiment_id=r.experiment_id
              order by r.start_time desc""",
            con,
        )
        st.dataframe(runs, hide_index=True)
        if runs.empty:
            st.stop()
        rid: str = st.selectbox(
            "Run",
            runs.run_uuid,
            format_func=lambda r: runs.set_index("run_uuid").loc[r, "name"] + " · " + r[:8],
        )
        st.write("Parameters")
        st.dataframe(
            pd.read_sql("select key, value from params where run_uuid=?", con, params=(rid,)),
            hide_index=True,
        )
        met = pd.read_sql(
            "select key, value, step from metrics where run_uuid=?", con, params=(rid,)
        )
        if len(met):
            keys = sorted(met.key.unique())
            k = st.selectbox(
                "Metric (different units, so shown one at a time)",
                keys,
                index=keys.index("val_weighted_tardiness")
                if "val_weighted_tardiness" in keys
                else 0,
            )
            st.plotly_chart(
                px.line(
                    met[met.key == k],
                    x="step",
                    y="value",
                    markers=True,
                    labels={"step": "timesteps", "value": k},
                ),
                use_container_width=True,
            )
    else:
        st.warning("No MLflow runs found (mlflow.db missing or empty); train a model first.")
    for m in sorted(MODELS.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        with st.expander(f"model: {meta['run_name']}  (train {meta['train_seconds']:.0f}s)"):
            st.json(meta["config"])
            ep = m.parent / "train_episodes.csv"
            if ep.exists():
                e = pd.read_csv(ep)
                e["smoothed"] = e.ep_return.rolling(50, min_periods=1).mean()
                st.plotly_chart(
                    px.line(
                        e,
                        x="timesteps",
                        y="smoothed",
                        title="training episode return (rolling mean)",
                    )
                )
            vh = m.parent / "val_history.csv"
            if vh.exists():
                st.dataframe(pd.read_csv(vh).round(1), hide_index=True)

elif page == "Robustness":
    st.title("Robustness Analysis (SAVED sweep results)")
    sweeps = sorted(p.stem for p in RESULTS.glob("sweep_*.csv"))
    if not sweeps:
        st.warning("No sweeps saved. Run `factory-scheduler sweep`.")
        st.stop()
    name = st.selectbox("Sweep", sweeps)
    df = pd.read_csv(RESULTS / f"{name}.csv")
    metric = st.selectbox("Metric", [m for m in LOWER_BETTER if m in df])
    meta = json.loads((RESULTS / f"{name}.meta.json").read_text())
    st.caption(
        f"Base scenario `{meta['base']}`; x axis = `{meta['field']}` (levels {meta['levels']}); "
        f"{len(meta['seeds'])} held-out test seeds per level"
    )
    s = df.groupby(["level", "policy"])[metric].agg(["mean", "std"]).reset_index()
    st.plotly_chart(
        px.line(
            s,
            x="level",
            y="mean",
            color="policy",
            error_y="std",
            markers=True,
            labels={"level": meta["field"], "mean": f"mean {metric}"},
        ),
        use_container_width=True,
    )
    st.caption("Error bars: std across seeds.")
    base = df.groupby(["policy", "seed"])[metric].mean().reset_index()
    st.plotly_chart(px.box(base, x="policy", y=metric, points="all"), use_container_width=True)

else:
    st.title("Schedule Visualization (SAVED schedule)")
    sched = pick_schedule("sv")
    st.caption(
        f"scenario {sched['scenario']} seed {sched['seed']} · end t={sched['end_time']} · "
        f"unfinished jobs: {len(sched['unfinished'])} · red = breakdown/repair, grey = setup"
    )
    st.plotly_chart(gantt(sched), use_container_width=True)
    st.plotly_chart(tardiness_chart(sched), use_container_width=True)
