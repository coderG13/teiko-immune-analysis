"""
dashboard.py - Interactive Streamlit dashboard for Teiko cell-count analysis.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Allow import from root when run as a module
sys.path.insert(0, os.path.dirname(__file__))
from analysis import (
    compute_frequency_table,
    run_statistical_analysis,
    get_melanoma_miraclib_pbmc,
    run_subset_analysis,
    POPULATIONS,
    DB_PATH,
)

st.set_page_config(
    page_title="Teiko Immune Cell Dashboard",
    page_icon="🧬",
    layout="wide",
)

# ── Shared state ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
    import load_data
    load_data.main()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_freq_df():
    return compute_frequency_table(get_conn())


@st.cache_data
def load_stat_df():
    return run_statistical_analysis(get_conn(), load_freq_df())


@st.cache_data
def load_subset():
    return run_subset_analysis(get_conn())


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://teiko.bio/favicon.ico",
    width=40,
)
st.sidebar.title("Teiko Teiknical")
st.sidebar.markdown("Immune Cell Population Analysis")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔬 Statistical Analysis", "🗂 Subset Analysis"],
)

# ── Page 1: Overview ──────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Part 2 — Cell Population Frequency Overview")
    st.markdown(
        "Relative frequency (%) of each immune cell population per sample. "
        "Each row = one population in one sample."
    )

    freq_df = load_freq_df()

    # Summary KPIs
    conn = get_conn()
    n_samples  = pd.read_sql("SELECT COUNT(*) AS n FROM samples", conn).iloc[0, 0]
    n_subjects = pd.read_sql("SELECT COUNT(*) AS n FROM subjects", conn).iloc[0, 0]
    n_projects = pd.read_sql("SELECT COUNT(*) AS n FROM projects", conn).iloc[0, 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{n_samples:,}")
    c2.metric("Total Subjects", f"{n_subjects:,}")
    c3.metric("Projects", n_projects)
    c4.metric("Cell Populations", len(POPULATIONS))

    st.divider()

    # Filters
    with st.expander("🔍 Filter samples", expanded=False):
        col1, col2, col3 = st.columns(3)
        pop_filter = col1.multiselect("Population", POPULATIONS, default=POPULATIONS)
        sample_search = col2.text_input("Sample ID contains", "")
        pct_range = col3.slider("Percentage range (%)", 0.0, 100.0, (0.0, 100.0))

    filtered = freq_df[
        freq_df["population"].isin(pop_filter) &
        freq_df["percentage"].between(*pct_range)
    ]
    if sample_search:
        filtered = filtered[filtered["sample"].str.contains(sample_search)]

    st.dataframe(filtered, use_container_width=True, height=400)
    st.caption(f"Showing {len(filtered):,} rows")

    # Bar: average % per population
    st.subheader("Average relative frequency by cell population")
    avg = freq_df.groupby("population")["percentage"].mean().reset_index()
    avg.columns = ["Population", "Avg %"]
    fig = px.bar(avg, x="Population", y="Avg %", color="Population",
                 text_auto=".2f", color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)


# ── Page 2: Statistical Analysis ──────────────────────────────────────────────

elif page == "🔬 Statistical Analysis":
    st.title("🔬 Part 3 — Statistical Analysis")
    st.markdown(
        "Comparing relative frequencies of **melanoma patients on miraclib** "
        "(PBMC samples only): **responders** vs **non-responders**."
    )

    freq_df = load_freq_df()
    stat_df = load_stat_df()
    conn    = get_conn()

    # Stats table
    st.subheader("Mann-Whitney U Test Results")
    display_stat = stat_df.copy()
    display_stat["significant"] = display_stat["significant"].map({True: "✅ Yes", False: "❌ No"})
    st.dataframe(display_stat, use_container_width=True)

    sig_pops = stat_df[stat_df["significant"] == True]["population"].tolist()
    if sig_pops:
        st.success(f"Significantly different populations (p < 0.05): **{', '.join(sig_pops)}**")
    else:
        st.info("No populations reached significance at p < 0.05.")

    st.divider()

    # Boxplots
    st.subheader("Boxplots: Responders vs Non-Responders")
    mel_df = get_melanoma_miraclib_pbmc(conn, freq_df)

    selected_pop = st.selectbox("Select population to inspect", POPULATIONS)

    pop_df = mel_df[mel_df["population"] == selected_pop]
    resp_vals    = pop_df[pop_df["response"] == "yes"]["percentage"]
    nonresp_vals = pop_df[pop_df["response"] == "no"]["percentage"]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=resp_vals, name="Responders",
        marker_color="#4C9BE8", boxmean=True,
    ))
    fig.add_trace(go.Box(
        y=nonresp_vals, name="Non-Responders",
        marker_color="#E8754C", boxmean=True,
    ))
    row = stat_df[stat_df["population"] == selected_pop].iloc[0]
    fig.update_layout(
        title=f"{selected_pop} — p={row['p_value']:.4f} {'(*)' if row['significant'] else ''}",
        yaxis_title="Relative Frequency (%)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # All populations side by side
    st.subheader("All Populations — Side-by-Side")
    fig2 = go.Figure()
    for response, color, label in [("yes", "#4C9BE8", "Responders"), ("no", "#E8754C", "Non-Responders")]:
        for i, pop in enumerate(POPULATIONS):
            sub = mel_df[(mel_df["population"] == pop) & (mel_df["response"] == response)]
            fig2.add_trace(go.Box(
                y=sub["percentage"],
                name=label,
                x=[pop] * len(sub),
                marker_color=color,
                legendgroup=label,
                showlegend=(i == 0),
                boxmean=True,
            ))
    fig2.update_layout(
        boxmode="group",
        yaxis_title="Relative Frequency (%)",
        height=500,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Page 3: Subset Analysis ───────────────────────────────────────────────────

elif page == "🗂 Subset Analysis":
    st.title("🗂 Part 4 — Subset Analysis")
    st.markdown(
        "Melanoma PBMC samples at **baseline (t=0)** from patients treated with **miraclib**."
    )

    subset = load_subset()
    df     = subset["dataframe"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(df))
    c2.metric("Responders", subset["subjects_by_response"].get("yes", 0))
    c3.metric("Non-Responders", subset["subjects_by_response"].get("no", 0))

    st.divider()

    col1, col2, col3 = st.columns(3)

    # Samples per project
    with col1:
        st.subheader("Samples per Project")
        proj_df = pd.DataFrame(
            subset["samples_per_project"].items(), columns=["Project", "Samples"]
        )
        fig = px.bar(proj_df, x="Project", y="Samples", color="Project",
                     text_auto=True, color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # By response
    with col2:
        st.subheader("Subjects by Response")
        resp_df = pd.DataFrame(
            subset["subjects_by_response"].items(), columns=["Response", "Count"]
        )
        resp_df["Response"] = resp_df["Response"].map({"yes": "Responders", "no": "Non-Responders"})
        fig2 = px.pie(resp_df, names="Response", values="Count",
                      color_discrete_sequence=["#4C9BE8", "#E8754C"])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # By sex
    with col3:
        st.subheader("Subjects by Sex")
        sex_df = pd.DataFrame(
            subset["subjects_by_sex"].items(), columns=["Sex", "Count"]
        )
        sex_df["Sex"] = sex_df["Sex"].map({"M": "Male", "F": "Female"})
        fig3 = px.pie(sex_df, names="Sex", values="Count",
                      color_discrete_sequence=["#6DBFB8", "#F4A261"])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    st.subheader("🎯 Key Result: Average B Cells (Melanoma Male Responders, t=0)")
    avg = subset["avg_bcell_male_resp"]
    n   = subset["n_male_resp_samples"]
    st.metric(
        label="Average B cell count",
        value=f"{avg:,.2f}",
        help=f"Based on {n} samples from melanoma male responders at baseline.",
    )

    st.subheader("Raw Subset Data")
    st.dataframe(df, use_container_width=True, height=350)
