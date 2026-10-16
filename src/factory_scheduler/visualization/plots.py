"""Plotly figures built from saved result JSON/CSV (no demo data)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def gantt(sched: dict[str, Any]) -> go.Figure:
    """Machines on the y axis; setup (grey), processing (coloured by job), repairs (red)."""
    fig = go.Figure()
    for s in sched["segments"]:
        color = "#9aa0a6" if s["kind"] == "setup" else px.colors.qualitative.Set3[s["job_id"] % 12]
        fig.add_trace(
            go.Bar(
                y=[f"M{s['machine_id']}"],
                x=[s["end"] - s["start"]],
                base=[s["start"]],
                orientation="h",
                marker={"color": color, "line": {"width": 0.5, "color": "#333"}},
                hovertemplate=f"job {s['job_id']} op {s['op_index']} ({s['kind']})<br>"
                f"{s['start']}–{s['end']} min<extra></extra>",
                showlegend=False,
            )
        )
    for m, a, b in sched["downtime"]:
        fig.add_trace(
            go.Bar(
                y=[f"M{m}"],
                x=[b - a],
                base=[a],
                orientation="h",
                marker={"color": "rgba(220,38,38,0.65)"},
                showlegend=False,
                hovertemplate=f"breakdown/repair {a}–{b}<extra></extra>",
            )
        )
    for label, color in (("setup", "#9aa0a6"), ("breakdown/repair", "rgba(220,38,38,0.65)")):
        fig.add_trace(go.Bar(y=[None], x=[None], name=label, marker={"color": color}))
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "y": -0.25},
        barmode="overlay",
        xaxis_title="time (min)",
        yaxis_title="machine",
        height=120 + 55 * len(sched["machines"]),
        margin={"t": 20},
        yaxis={"categoryorder": "category descending"},
    )
    return fig


def tardiness_chart(sched: dict[str, Any]) -> go.Figure:
    comp = {int(k): v for k, v in sched["completion"].items()}
    rows = [
        {
            "job": j["id"],
            "completion": comp.get(j["id"]),
            "due": j["due"],
            "priority": j["priority"],
            "tardiness": max(0, comp[j["id"]] - j["due"]) if j["id"] in comp else None,
            "status": "completed" if j["id"] in comp else "UNFINISHED",
        }
        for j in sched["jobs"]
    ]
    df = pd.DataFrame(rows)
    return px.bar(
        df,
        x="job",
        y="tardiness",
        color="status",
        hover_data=["due", "completion"],
        labels={"tardiness": "tardiness (min)"},
    )


def machine_table(sched: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(sched["machine_stats"]).T
    df.index = [f"M{i}" for i in df.index]
    return df[["process", "setup", "down", "idle", "utilization"]].round(2)
