
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime as dt

st.set_page_config(page_title="JetLearn Analytics", layout="wide")

# ----------------- Helpers -----------------

@st.cache_data(show_spinner=False)
def load_master(uploaded_file):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("Master_sheet-DB.csv")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(df, candidates):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        lc = c.lower()
        if lc in low:
            return low[lc]
    for c in cols:
        name = str(c).lower()
        for cand in candidates:
            if cand.lower() in name:
                return c
    return None

def _to_dt(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

# ----------------- Sidebar -----------------

with st.sidebar:
    st.title("JetLearn Analytics")
    uploaded = st.file_uploader("Upload Master_sheet-DB.csv", type=["csv"])
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)

df_raw = load_master(uploaded)
if df_raw is None or df_raw.empty:
    st.stop()

# (simple track filter placeholder - extend as per your pipeline logic)
pipe_col = _find_col(df_raw, ["Pipeline", "Program", "Course Type"])
if pipe_col and track != "Both":
    s = df_raw[pipe_col].astype(str).str.lower()
    if track == "AI Coding":
        mask = s.str.contains("ai") | s.str.contains("code") | s.str.contains("python") | s.str.contains("ml")
    else:
        mask = s.str.contains("math")
    df = df_raw[mask].copy()
else:
    df = df_raw.copy()

# ----------------- Navigation -----------------

MASTER_SECTIONS = {
    "Performance": ["Placeholder"],
    "Funnel & Movement": ["Placeholder"],
    "Insights & Forecast": ["Placeholder"],
    "Marketing": [
        "Plan vs Actual"
    ],
}

with st.sidebar:
    master = st.radio("Section", list(MASTER_SECTIONS.keys()), index=3)
    sub_views = MASTER_SECTIONS[master]
    nav_sub = st.radio("View", sub_views, index=0)

# ----------------- Placeholders for other pills -----------------

def _placeholder():
    st.info("This is a placeholder in this slim build. Your production file has full logic; merge this Plan vs Actual pill there.")

# ----------------- Marketing ▶ Plan vs Actual (FINAL) -----------------

def _render_marketing_plan_vs_actual(df):
    st.subheader("Plan vs Actual — JetLearn Deal Source × Country")

    # ---- Identify required columns ----
    src_col = _find_col(df, ["JetLearn Deal Source", "Jetlearn Deal Source", "Deal Source", "Lead Source", "Source"])
    country_col = _find_col(df, ["Country", "Country Name", "Geo", "Region"])
    create_col = _find_col(df, ["Create Date", "Created Date", "Deal create date", "Created On"])
    pay_col = _find_col(df, ["Payment Received Date", "Payment Received date", "Enrolment Date", "Deal Won Date"])

    if not src_col or not country_col or not create_col or not pay_col:
        st.error("Missing one of: JetLearn Deal Source, Country, Create Date, Payment Received Date.")
        return

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    src = df[src_col].astype(str)
    country = df[country_col].astype(str)
    create_dt = _to_dt(df[create_col])
    pay_dt = _to_dt(df[pay_col])

    all_sources = sorted(src.dropna().unique().tolist())
    all_countries = sorted(country.dropna().unique().tolist())

    if not all_sources or not all_countries:
        st.error("No sources or countries found.")
        return

    # ---- Controls: Lookback period & mode ----
    c1, c2, c3 = st.columns(3)
    with c1:
        lookback_months = st.selectbox(
            "Lookback period (for Plan conversion)",
            options=list(range(1, 25)),
            index=2,  # default 3 months
            format_func=lambda x: f"{x} month" if x == 1 else f"{x} months",
            key="pva_lb_months"
        )
    with c2:
        conv_mode = st.radio(
            "Conversion logic",
            ["MTD (deals & enrols in window)", "Cohort (deals in window, enrols anytime)"],
            index=0,
            key="pva_conv_mode"
        )
    with c3:
        # Actual window: free date range, default = current month
        today = dt.date.today()
        default_start = today.replace(day=1)
        actual_start = st.date_input("Actual period start", value=default_start, key="pva_actual_start")
        actual_end = st.date_input("Actual period end", value=today, key="pva_actual_end")
        if actual_start > actual_end:
            st.error("Actual start date cannot be after end date.")
            return

    # Compute lookback window
    today = dt.date.today()
    lb_end = pd.Timestamp(today)
    lb_start = lb_end - pd.DateOffset(months=lookback_months)
    lb_start = lb_start.normalize()
    lb_end = lb_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    ac_start_ts = pd.to_datetime(actual_start)
    ac_end_ts = pd.to_datetime(actual_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # ---- Helper: deals/enrols for source + country selection ----
    def _parse_countries(val):
        if not isinstance(val, str):
            return None
        txt = val.strip()
        if txt == "" or txt.lower() == "all":
            return None
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        return parts or None

    def _deals_enrol(src_val, countries_val, start_ts, end_ts, mode):
        m = (src == str(src_val))
        row_countries = _parse_countries(countries_val)
        if row_countries is None:
            m &= country.isin(all_countries)
        else:
            m &= country.isin(row_countries)

        if mode.startswith("MTD"):
            deals_mask = m & create_dt.between(start_ts, end_ts)
            enrl_mask = m & pay_dt.between(start_ts, end_ts)
        else:  # Cohort
            deals_mask = m & create_dt.between(start_ts, end_ts)
            enrl_mask = deals_mask & pay_dt.notna()

        deals = int(deals_mask.sum())
        enrol = int(enrl_mask.sum())
        return deals, enrol

    # ---- PLAN TABLE ----
    st.markdown("### 1️⃣ Plan Table (Editable)")

    # For each JetLearn Deal Source one row; user sets Plan Deals & Countries
    base_rows = []
    for s in all_sources:
        base_rows.append({
            "JetLearn Deal Source": s,
            "Countries (comma-separated or 'All')": "All",
            "Plan Deals": 0,
        })
    base_plan_df = pd.DataFrame(base_rows)

    plan_edit = st.data_editor(
        base_plan_df,
        key="pva_plan_editor",
        use_container_width=True,
        num_rows="dynamic"
    )

    plan_records = []
    for _, r in plan_edit.iterrows():
        s_val = r.get("JetLearn Deal Source")
        if pd.isna(s_val) or str(s_val).strip() == "":
            continue
        c_val = r.get("Countries (comma-separated or 'All')", "All")
        try:
            plan_deals = float(r.get("Plan Deals", 0) or 0)
        except Exception:
            plan_deals = 0.0

        lb_deals, lb_enrol = _deals_enrol(s_val, c_val, lb_start, lb_end, conv_mode)
        plan_conv = (lb_enrol / lb_deals) * 100.0 if lb_deals > 0 else 0.0
        plan_enrol = plan_deals * (plan_conv / 100.0)

        plan_records.append({
            "JetLearn Deal Source": s_val,
            "Plan Countries": c_val,
            "Lookback Deals": lb_deals,
            "Lookback Enrollments": lb_enrol,
            "Planned Deals": round(plan_deals, 2),
            "Planned Conversion % (from lookback)": round(plan_conv, 2),
            "Planned Enrollments": round(plan_enrol, 2),
        })

    if not plan_records:
        st.info("Add at least one valid Plan row.")
        return

    plan_df = pd.DataFrame(plan_records)

    st.dataframe(plan_df, use_container_width=True)

    # ---- ACTUAL TABLE ----
    st.markdown("### 2️⃣ Actual Table (Auto)")

    # Actual countries auto-populate per source within actual window
    actual_rows = []
    for s_val in plan_df["JetLearn Deal Source"].unique():
        # all countries for this source in actual scope
        mask_src = (src == str(s_val))
        mask_window = create_dt.between(ac_start_ts, ac_end_ts)
        # use all data for detecting countries; less strict
        src_countries = sorted(country[mask_src].dropna().unique().tolist())
        if not src_countries:
            src_countries = all_countries

        # Use "All" logically - i.e., same country set; details handled inside _deals_enrol
        c_val = "All"

        ac_deals, ac_enrol = _deals_enrol(s_val, c_val, ac_start_ts, ac_end_ts, conv_mode)
        ac_conv = (ac_enrol / ac_deals) * 100.0 if ac_deals > 0 else 0.0

        actual_rows.append({
            "JetLearn Deal Source": s_val,
            "Actual Countries (auto)": ", ".join(src_countries) if src_countries else "All",
            "Actual Deals": ac_deals,
            "Actual Enrollments": ac_enrol,
            "Actual Conversion %": round(ac_conv, 2),
        })

    actual_df = pd.DataFrame(actual_rows)
    st.dataframe(actual_df, use_container_width=True)

    # ---- COMPARISON TABLE ----
    st.markdown("### 3️⃣ Plan vs Actual Comparison")

    merged = pd.merge(
        plan_df,
        actual_df,
        on="JetLearn Deal Source",
        how="outer"
    ).fillna({
        "Planned Deals": 0,
        "Planned Enrollments": 0,
        "Planned Conversion % (from lookback)": 0,
        "Actual Deals": 0,
        "Actual Enrollments": 0,
        "Actual Conversion %": 0,
    })

    merged["Gap Deals (Actual - Plan)"] = merged["Actual Deals"] - merged["Planned Deals"]
    merged["Gap Enrollments (Actual - Plan)"] = merged["Actual Enrollments"] - merged["Planned Enrollments"]
    merged["Gap Conversion % (Actual - Plan)"] = merged["Actual Conversion %"] - merged["Planned Conversion % (from lookback)"]

    st.dataframe(merged, use_container_width=True)

    # ---- Chart Options ----
    st.markdown("### 4️⃣ Visual Comparison")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        metric_choice = st.selectbox(
            "Metric to compare",
            ["Conversion %", "Deals", "Enrollments"],
            index=0,
            key="pva_metric"
        )
    with col_m2:
        chart_type = st.selectbox(
            "Chart type",
            ["Grouped Bar", "Stacked Bar", "Line"],
            index=0,
            key="pva_chart_type"
        )

    chart_df = pd.DataFrame()
    if metric_choice == "Conversion %":
        chart_df = pd.DataFrame({
            "JetLearn Deal Source": merged["JetLearn Deal Source"],
            "Plan": merged["Planned Conversion % (from lookback)"],
            "Actual": merged["Actual Conversion %"],
        })
        y_title = "Conversion %"
    elif metric_choice == "Deals":
        chart_df = pd.DataFrame({
            "JetLearn Deal Source": merged["JetLearn Deal Source"],
            "Plan": merged["Planned Deals"],
            "Actual": merged["Actual Deals"],
        })
        y_title = "Deals"
    else:
        chart_df = pd.DataFrame({
            "JetLearn Deal Source": merged["JetLearn Deal Source"],
            "Plan": merged["Planned Enrollments"],
            "Actual": merged["Actual Enrollments"],
        })
        y_title = "Enrollments"

    chart_df = chart_df.melt(
        id_vars=["JetLearn Deal Source"],
        value_vars=["Plan", "Actual"],
        var_name="Type",
        value_name="Value"
    )

    if not chart_df.empty:
        if chart_type == "Grouped Bar":
            ch = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("JetLearn Deal Source:N", title="Deal Source"),
                y=alt.Y("Value:Q", title=y_title),
                color=alt.Color("Type:N", title=""),
                column=alt.Column("Type:N", title=""),
                tooltip=["JetLearn Deal Source", "Type", "Value"]
            )
        elif chart_type == "Stacked Bar":
            ch = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("JetLearn Deal Source:N", title="Deal Source"),
                y=alt.Y("Value:Q", title=y_title, stack="zero"),
                color=alt.Color("Type:N", title=""),
                tooltip=["JetLearn Deal Source", "Type", "Value"]
            )
        else:  # Line
            ch = alt.Chart(chart_df).mark_line(point=True).encode(
                x=alt.X("JetLearn Deal Source:N", title="Deal Source"),
                y=alt.Y("Value:Q", title=y_title),
                color=alt.Color("Type:N", title=""),
                tooltip=["JetLearn Deal Source", "Type", "Value"]
            )
        st.altair_chart(ch, use_container_width=True)

    # Download
    st.download_button(
        "Download CSV — Plan vs Actual Summary",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name="plan_vs_actual_summary.csv",
        mime="text/csv",
        key="pva_download"
    )

# ----------------- Router -----------------

if master == "Marketing" and nav_sub == "Plan vs Actual":
    _render_marketing_plan_vs_actual(df)
else:
    _placeholder()
