"""Term-deposit outreach ranking: stakeholder dashboard (BAN6800 Module 5).

Academic prototype. Not deployed at a bank. Decision support only.
Serves the same configuration as app/main.py: term_deposit_ranker v1
(Gradient Boosting, no economic indicators) with age-band cutoffs.
All results shown come from the Module 4 hold-out evaluation (8,236 records).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monitoring_tab import render_monitoring

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "stratified_no_macro_gboost.joblib"
CUTOFF_PATH = ROOT / "models" / "age_band_cutoffs.json"
FIG_DIR = ROOT / "reports" / "figures"

# Same definitions as app/main.py
AGE_BINS = [17, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["17-24", "25-34", "35-44", "45-54", "55-64", "65+"]
FEATURES = ["job", "marital", "education", "default", "housing", "loan", "contact", "month",
            "day_of_week", "poutcome", "age", "previous", "previously_contacted",
            "days_since_previous_contact"]
JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown"]
MARITAL = ["divorced", "married", "single", "unknown"]
EDUCATION = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
             "professional.course", "university.degree", "unknown"]
YN = ["no", "yes", "unknown"]
MONTHS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = ["mon", "tue", "wed", "thu", "fri"]
CONTACT = ["cellular", "telephone"]
POUTCOME = ["nonexistent", "failure", "success"]

BLUE, ORANGE, GREY, LIGHT = "#4C78A8", "#F58518", "#9AA0A6", "#D3D3D3"

st.set_page_config(page_title="Term-deposit outreach ranking", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    cutoffs = json.loads(CUTOFF_PATH.read_text())["age_band_cutoffs"]
    return model, cutoffs


def age_band(age: int) -> str:
    return str(pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS, right=False)[0])


def make_frame(customers: list) -> pd.DataFrame:
    rows = []
    for c in customers:
        r = dict(c)
        prev = int(r["previous"])
        r["previously_contacted"] = int(prev > 0)
        r["days_since_previous_contact"] = float(r.pop("days_since")) if prev > 0 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)[FEATURES]
    df["days_since_previous_contact"] = df["days_since_previous_contact"].astype(float)
    return df


def score(model, customers: list) -> np.ndarray:
    return model.predict_proba(make_frame(customers))[:, 1]


PRESETS = {
    "Typical record (cellular, June, no earlier contact)": dict(
        age=38, job="admin.", marital="married", education="university.degree", default="no",
        housing="yes", loan="no", contact="cellular", month="jun", day_of_week="tue",
        poutcome="nonexistent", previous=0, days_since=0),
    "Earlier successful contact (cellular, September)": dict(
        age=52, job="retired", marital="married", education="high.school", default="no",
        housing="no", loan="no", contact="cellular", month="sep", day_of_week="thu",
        poutcome="success", previous=1, days_since=6),
    "First contact by landline (May)": dict(
        age=44, job="blue-collar", marital="married", education="basic.9y", default="unknown",
        housing="yes", loan="no", contact="telephone", month="may", day_of_week="mon",
        poutcome="nonexistent", previous=0, days_since=0),
}

# ---------------------------------------------------------------- header
st.title("Term-deposit outreach ranking: what the model found")
st.caption("BAN6800 Module 5 | Academic prototype built on Portuguese retail telemarketing data "
           "(2008-2010). Not deployed. Decision support only: a relationship manager makes every "
           "decision. Inputs on this page are illustrative; no real customer data is used.")

tab_overview, tab_score, tab_fair, tab_limits, tab_monitor = st.tabs(
    ["Results overview", "Score a customer and what-if", "Ethical compliance: fairness",
     "Limitations and transparency", "Monitoring"])

# ---------------------------------------------------------------- overview
with tab_overview:
    st.subheader("How many subscribers does the top 30% of the list reach?")
    st.write("Customers are ranked by the model's score and outreach goes to the top 30%. "
             "Random selection would reach about 30% of subscribers. The bars below show the "
             "share of all subscribers reached by each approach on the hold-out test set "
             "(8,236 records, 928 subscribers). Lines show 95% intervals where computed.")

    configs = [
        ("Random selection", 29.9, None, None, GREY),
        ("Previous-outcome rule", 46.1, None, None, GREY),
        ("Logistic Regression (no economic indicators)", 65.0, 62.4, 68.2, BLUE),
        ("Gradient Boosting (no economic indicators)", 67.2, 64.7, 70.0, BLUE),
        ("Gradient Boosting + age-band cutoffs (served here)", 62.4, 59.2, 65.7, ORANGE),
        ("Gradient Boosting without age, marital status, education", 66.7, None, None, BLUE),
        ("With economic indicators (upper bound only, not usable)", 73.5, 71.1, 76.4, LIGHT),
    ]
    names = [c[0] for c in configs][::-1]
    vals = [c[1] for c in configs][::-1]
    hi = [(c[3] - c[1]) if c[3] else 0 for c in configs][::-1]
    lo = [(c[1] - c[2]) if c[2] else 0 for c in configs][::-1]
    cols = [c[4] for c in configs][::-1]
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h", marker_color=cols,
        error_x=dict(type="data", array=hi, arrayminus=lo, color="#333"),
        text=[f"{v:.1f}%" for v in vals], textposition="inside", insidetextanchor="start"))
    fig.add_vline(x=70, line_dash="dash", line_color="#333",
                  annotation_text="Module 1 target: 70%", annotation_position="top")
    fig.update_layout(xaxis=dict(range=[0, 100], title="Share of subscribers reached (%)"),
                      height=430, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.caption("The 73.5% bar depends on economic indicators that mostly identify the campaign "
               "period, so it is shown only as an upper bound. The 66.7% result has no interval "
               "computed.")

    st.subheader("Progress against the three Module 1 goals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Subscribers reached in top 30%", "67.2%")
    c1.caption("Target 70%: not confidently met (CI 64.7-70.0)")
    c2.metric("Age-band recall gaps within 10 points", "0 of 6 bands beyond")
    c2.caption("Only with age-band cutoffs, which lower capture to 62.4%")
    c3.metric("Contacts per subscriber vs random", "Not established")
    c3.caption("7.83 vs 23.00 was for the full-feature model only")

    shap_img = FIG_DIR / "fig6_shap_summary_deployable.png"
    if shap_img.exists():
        st.subheader("What drives the scores")
        st.write("Main drivers: contact channel, whether an earlier contact is recorded, age, "
                 "and the months June and April. These show association, not cause, and may "
                 "reflect how the campaigns were run.")
        st.image(str(shap_img))
        st.caption("How to read this: each dot is one customer record. Dots to the right raise "
                   "the score and dots to the left lower it. Red means the feature is present "
                   "or high, blue means absent or low. 'missingindicator_days_since_previous_"
                   "contact' means no earlier contact is recorded: records that do have an "
                   "earlier contact (blue) are pushed up strongly. Month labels show the month "
                   "of contact.")

# ---------------------------------------------------------------- score + what-if
with tab_score:
    st.subheader("Score an illustrative customer")
    st.write("This uses the same scorer as the Module 4 API. The score is the model's estimate of "
             "the chance of subscribing. It is a ranking score, not a promise.")
    preset_name = st.selectbox("Start from an illustrative profile", list(PRESETS))
    p = PRESETS[preset_name]
    k = preset_name  # keys change with the preset so widgets reset to its values

    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", 17, 100, p["age"], key=f"age{k}")
        job = st.selectbox("Job", JOBS, JOBS.index(p["job"]), key=f"job{k}")
        marital = st.selectbox("Marital status", MARITAL, MARITAL.index(p["marital"]), key=f"mar{k}")
        education = st.selectbox("Education", EDUCATION, EDUCATION.index(p["education"]), key=f"edu{k}")
    with b:
        default = st.selectbox("Credit default", YN, YN.index(p["default"]), key=f"def{k}")
        housing = st.selectbox("Housing loan", YN, YN.index(p["housing"]), key=f"hou{k}")
        loan = st.selectbox("Personal loan", YN, YN.index(p["loan"]), key=f"loa{k}")
        poutcome = st.selectbox("Previous-campaign outcome", POUTCOME,
                                POUTCOME.index(p["poutcome"]), key=f"pout{k}")
    with c:
        contact = st.selectbox("Contact channel", CONTACT, CONTACT.index(p["contact"]), key=f"con{k}")
        month = st.selectbox("Month", MONTHS, MONTHS.index(p["month"]), key=f"mon{k}")
        day = st.selectbox("Weekday", DAYS, DAYS.index(p["day_of_week"]), key=f"day{k}")
        previous = st.number_input("Previous contacts", 0, 50, p["previous"], key=f"prev{k}")
        days_since = st.number_input("Days since previous contact", 0, 60, p["days_since"],
                                     key=f"ds{k}", disabled=(previous == 0))

    cust = dict(age=int(age), job=job, marital=marital, education=education, default=default,
                housing=housing, loan=loan, contact=contact, month=month, day_of_week=day,
                poutcome=poutcome, previous=int(previous), days_since=float(days_since))

    try:
        model, cutoffs = load_artifacts()
    except Exception as e:  # keep the rest of the dashboard usable
        st.error("The model file could not be loaded, so scoring is unavailable. "
                 f"Details: {type(e).__name__}: {e}")
        st.stop()

    prob = float(score(model, [cust])[0])
    band = age_band(cust["age"])
    cut = float(cutoffs[band])
    m1, m2, m3 = st.columns(3)
    m1.metric("Model score", f"{prob:.3f}")
    m2.metric(f"Age-band cutoff ({band})", f"{cut:.3f}")
    m3.metric("Against the cutoff", "At or above" if prob >= cut else "Below")
    st.info("Plain-language reading: the score estimates how likely this kind of customer record "
            "is to subscribe, based only on information known before the call. At or above the "
            "cutoff means the record would fall on the outreach list in the candidate "
            "configuration. The cutoff differs by age band (see the fairness tab). A relationship "
            "manager decides whether to act, and a low score does not mean a customer has no need.")

    st.subheader("What-if: how the score moves when only timing or channel changes")
    st.write("Everything else is held as set above. This shows how the score changes, not what "
             "would happen to a real customer. Month mostly reflects how the campaigns were run, "
             "so it is not something a relationship manager can act on.")
    rows, labels = [], []
    for ch in CONTACT:
        for m in MONTHS:
            rows.append({**cust, "contact": ch, "month": m})
            labels.append((ch, m))
    sc = score(model, rows)
    df_m = pd.DataFrame({"channel": [l[0] for l in labels], "month": [l[1] for l in labels],
                         "score": sc})
    fig_m = go.Figure()
    for ch, colr in zip(CONTACT, [BLUE, ORANGE]):
        d = df_m[df_m.channel == ch]
        fig_m.add_bar(x=d.month, y=d.score, name=ch, marker_color=colr)
    fig_m.add_hline(y=cut, line_dash="dash", line_color="#333",
                    annotation_text=f"Cutoff for age band {band}")
    fig_m.update_layout(barmode="group", height=360, yaxis_title="Model score",
                        xaxis_title="Month of contact", margin=dict(t=30, b=10))
    st.plotly_chart(fig_m)

    rows_d = [{**cust, "day_of_week": d_} for d_ in DAYS]
    sc_d = score(model, rows_d)
    fig_d = go.Figure(go.Bar(x=DAYS, y=sc_d, marker_color=BLUE))
    fig_d.add_hline(y=cut, line_dash="dash", line_color="#333")
    fig_d.update_layout(height=300, yaxis_title="Model score", xaxis_title="Weekday of contact",
                        margin=dict(t=20, b=10))
    st.plotly_chart(fig_d)
    st.caption("Module 4 counterfactual check: only 37.5% of customers not selected could reach "
               "the list within the same month, so the model leans heavily on calendar timing.")

# ---------------------------------------------------------------- fairness
with tab_fair:
    st.subheader("Ethical compliance dashboard: who gets reached?")
    st.write("Recall here means the share of each age group's subscribers who reach the outreach "
             "list. The Module 1 rule is that each group stays within 10 percentage points of "
             "overall recall (67.2% before mitigation, 62.4% after).")

    bands = AGE_LABELS
    n_sub = [50, 346, 252, 129, 89, 62]
    before = [96.0, 72.3, 49.2, 55.0, 77.5, 100.0]
    after = [56.0, 62.1, 61.9, 66.7, 59.6, 66.1]
    gap_b = [round(x - 67.2, 1) for x in before]
    gap_a = [round(x - 62.4, 1) for x in after]
    xlab = [f"{b_} (n={n})" for b_, n in zip(bands, n_sub)]

    fig_g = go.Figure()
    fig_g.add_bar(x=xlab, y=gap_b, name="Before mitigation", marker_color=BLUE,
                  text=[f"{g:+.1f}" for g in gap_b], textposition="outside")
    fig_g.add_bar(x=xlab, y=gap_a, name="After age-band cutoffs", marker_color=ORANGE,
                  text=[f"{g:+.1f}" for g in gap_a], textposition="outside")
    fig_g.add_hline(y=10, line_dash="dash", line_color="#333")
    fig_g.add_hline(y=-10, line_dash="dash", line_color="#333",
                    annotation_text="10-point rule", annotation_position="bottom right")
    fig_g.update_layout(barmode="group", height=430, yaxis_title="Recall gap vs overall (points)",
                        xaxis_title="Age band (subscribers in test set)",
                        margin=dict(t=30, b=10), yaxis=dict(range=[-25, 40]))
    st.plotly_chart(fig_g)
    st.caption("Before mitigation, 5 of 6 age bands were beyond the 10-point rule. After "
               "age-band cutoffs, none are, and the widest gap is 6.4 points. With only 50 to 346 "
               "subscribers per band, each estimate carries an interval of roughly 10 points, so "
               "no detectable gap does not prove equality.")

    st.dataframe(pd.DataFrame({"Age band": bands, "Subscribers (test)": n_sub,
                               "Recall before (%)": before, "Recall after (%)": after}),
                 hide_index=True)

    st.subheader("The three options and what each costs")
    st.dataframe(pd.DataFrame({
        "Option": ["No mitigation", "Refit without age, marital status, education",
                   "Age-band cutoffs (served in this dashboard)"],
        "Subscribers reached": ["67.2%", "66.7% (interval not computed)", "62.4% (CI 59.2-65.7)"],
        "Age bands beyond the rule": [5, 2, 0],
        "Age used at decision time": ["As a model input", "No", "Yes, to set cutoffs"],
    }), hide_index=True)
    st.write("Score cutoffs under the age-band option: a customer aged 17-24 needs a score of "
             "0.325 to be selected, and a customer aged 35-44 needs 0.077. At the same score, one "
             "could be selected and the other not.")

    st.subheader("Other fairness measures")
    f1, f2, f3 = st.columns(3)
    f1.metric("Disparate impact ratio (age), before", "0.161")
    f2.metric("Disparate impact ratio (age), after", "0.371")
    f2.caption("Screening convention: 0.80")
    f3.metric("Education groups beyond the rule", "3 before, 3 after")
    f3.caption("Education was not mitigated")
    st.write("The ratio stays below 0.80 because subscription rates differ widely by age (about "
             "7% to 52%) and the project targeted equal recall, not equal selection. The two "
             "measures cannot both be met with this approach. For education, the widest gap after "
             "mitigation is basic.9y at 31 points. Marital status had 2 groups beyond the rule "
             "before and 0 after. Gender cannot be assessed because the dataset has no gender "
             "column.")

# ---------------------------------------------------------------- limitations
with tab_limits:
    st.subheader("What this model can and cannot do")
    l, r = st.columns(2)
    with l:
        st.markdown("**It can**")
        st.markdown(
            "- Rank customer records using information known before outreach\n"
            "- Support relationship-manager planning when outreach slots are limited\n"
            "- Report performance and fairness by age, marital status and education")
    with r:
        st.markdown("**It cannot**")
        st.markdown(
            "- Decide who is contacted, or replace a relationship manager's judgement\n"
            "- Show that any channel or month causes subscriptions\n"
            "- Perform equally well for every group\n"
            "- Be assumed to work on Nigerian customers or on later periods\n"
            "- Be used for credit, pricing or eligibility decisions")

    st.subheader("Limitations")
    st.markdown(
        "- **Data:** one Portuguese bank, retail telemarketing, 2008-2010. Corporate relationship "
        "banking differs, and transfer to Nigeria is untested.\n"
        "- **Target not confidently met:** 67.2% against a 70% target (95% CI 64.7-70.0). The "
        "73.5% figure used economic indicators that mostly identify the campaign period.\n"
        "- **Drift:** subscription rates were 6.4% in the earlier records and 30.8% in the later "
        "ones. In the time-ordered stress test, capture fell to about 48-53%.\n"
        "- **Observational data:** drivers shown are associations, not causes.\n"
        "- **Efficiency claim:** 7.83 contacts per subscriber (vs 23.00) is descriptive and "
        "covers the full-feature model only.\n"
        "- **Identity:** the data has no customer identifier, so repeat customers cannot be ruled out.\n"
        "- **Gaps:** gender cannot be assessed, and the planned age-by-job analysis is incomplete.\n"
        "- **Prototype:** the API has no authentication, and the model file is a pickle that "
        "must only be loaded when produced by the bank itself.")

    st.subheader("Transparency statement and commitments")
    st.write("This is an academic prototype that supports outreach planning. Its commitments "
             "follow the project's Ethical AI Charter: fairness (group metrics reported "
             "openly), transparency (limitations and explanations disclosed), privacy (no "
             "direct identifiers; raw data not published), accountability (humans decide, and "
             "an ethics and governance committee can pause use) and responsible use (no credit "
             "decisions or coercive treatment).")
    st.caption("Sources: Module 4 Predictive Model Development and Validation Report and Model "
               "Card (25 October 2026); Moro et al. (2012) data set. Repository: "
               "github.com/Nkem23/bank-marketing-segmentation")

with tab_monitor:
    render_monitoring(ROOT, BLUE, ORANGE, GREY, LIGHT)
