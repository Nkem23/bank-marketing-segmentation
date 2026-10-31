"""Monitoring tab for the dashboard. Reads reports/monitoring_summary.json (aggregates only)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_monitoring(root, blue, orange, grey, light):
    st.subheader("Monitoring baseline: drift and fairness checks")
    st.write("These checks run on the Module 4 hold-out data to set a baseline and to test "
             "the alert rules. No new labelled outcomes exist yet, so this is not live "
             "production monitoring.")
    path = root / "reports" / "monitoring_summary.json"
    if not path.exists():
        st.warning("Monitoring summary not found. Run: python -m src.monitoring")
        return
    mon = json.loads(path.read_text(encoding="utf-8"))
    drift, fair, rules = mon["drift"], mon["fairness"], mon["rules"]
    st.caption("Summary generated " + mon["generated_utc"] + " for " + mon["model"] + ".")

    st.markdown("**1. Data drift: did the subscription rate change?**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Earlier 80% of records", f"{drift['base_rate_earlier_pct']}%")
    c2.metric("Later 20% of records", f"{drift['base_rate_later_pct']}%")
    c3.metric("Shift", f"{drift['shift_pts']} points")
    if drift["alert"]:
        st.error(f"ALERT: the subscription rate moved {abs(drift['shift_pts'])} points, "
                 f"above the {rules['base_rate_shift_alert_pts']}-point rule. "
                 "Review is required before any use.")
    else:
        st.success("No drift alert.")
    fig_d = go.Figure(go.Bar(
        x=["Earlier 80%", "Later 20%"],
        y=[drift["base_rate_earlier_pct"], drift["base_rate_later_pct"]],
        marker_color=[blue, orange],
        text=[f"{drift['base_rate_earlier_pct']}%", f"{drift['base_rate_later_pct']}%"],
        textposition="outside"))
    fig_d.update_layout(height=320, yaxis_title="Subscription rate (%)",
                        showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig_d)
    mix = pd.DataFrame(drift["mix"])
    mix["abs_change"] = mix["change_pts"].abs()
    mix = mix.sort_values("abs_change", ascending=False).head(8).drop(columns="abs_change")
    mix.columns = ["Feature", "Category", "Earlier %", "Later %", "Change (points)"]
    st.caption("Largest shifts in the mix of contact channel, previous outcome and month:")
    st.dataframe(mix, hide_index=True)

    st.markdown("**2. Fairness: does each age band stay within 10 points of overall recall?**")
    bands = pd.DataFrame(fair["age_bands"])
    overall = fair["overall_recall_pct"]
    gap_rule = rules["recall_gap_alert_pts"]
    n_alert = int(bands["alert"].sum())
    if n_alert:
        st.error(f"ALERT: {n_alert} age band(s) are more than {gap_rule} points from "
                 f"overall recall ({overall}%). Escalate to the ethics committee.")
    else:
        st.success(f"No age band is outside the {gap_rule}-point rule "
                   f"(overall recall {overall}%).")
    fig_f = go.Figure()
    fig_f.add_hrect(y0=overall - gap_rule, y1=overall + gap_rule,
                    fillcolor=light, opacity=0.4, line_width=0)
    fig_f.add_hline(y=overall, line_dash="dash", line_color=grey)
    fig_f.add_trace(go.Bar(
        x=bands["age_band"], y=bands["recall_pct"], marker_color=blue,
        error_y=dict(type="data", symmetric=False,
                     array=(bands["ci_high_pct"] - bands["recall_pct"]).tolist(),
                     arrayminus=(bands["recall_pct"] - bands["ci_low_pct"]).tolist())))
    fig_f.update_layout(height=380, yaxis_title="Recall at the 30% cut (%)",
                        yaxis_range=[0, 100], showlegend=False, margin=dict(t=30))
    st.plotly_chart(fig_f)
    st.caption("Bars show recall with 95% intervals. The grey band is overall recall plus "
               "or minus 10 points. Wide intervals mean a band inside the rule is not "
               "proven fair, and selection rates still differ by age band.")
    table = bands[["age_band", "subscribers", "recall_pct", "ci_low_pct", "ci_high_pct",
                   "selection_rate_pct", "gap_pts"]].copy()
    table.columns = ["Age band", "Subscribers", "Recall %", "CI low %", "CI high %",
                     "Selection rate %", "Gap (points)"]
    st.dataframe(table, hide_index=True)

    st.markdown("**Alert rules**")
    st.write(f"- The subscription rate shifts by more than {rules['base_rate_shift_alert_pts']} points.\n"
             f"- Any age band with at least {rules['min_subscribers_per_band']} subscribers is "
             f"more than {rules['recall_gap_alert_pts']} points from overall recall.\n"
             "- A data-quality check fails (the Module 3 pipeline stops on failure).\n\n"
             "Any alert triggers investigation and review by the ethics committee.")
