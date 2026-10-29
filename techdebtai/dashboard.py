"""Run: streamlit run dashboard.py"""
import os
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TechDebt-AI", layout="wide")
st.title("TechDebt-AI dashboard")

def load_csv(src):
    df = pd.read_csv(src)
    
    # 1. Standardize path column
    if "file name" in df.columns:
        df = df.rename(columns={"file name": "path"})
    if "path" in df.columns:
        df = df.set_index("path")
    else:
        df.index.name = "path"

    # 2. Map dataset.csv column names to dashboard expected names without creating duplicate column names
    col_map = {
        "cyclomatic_complexity": "avg_ccn",
        "max_complexity": "max_ccn",
        "loc": "nloc",
        "commit_count": "n_commits",
        "author_count": "n_authors",
        "bug_fix_commits": "bug_fixes",
        "churn_rate": "churn",
        "label": "risk"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Deduplicate columns if any duplicate headers exist
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Fill missing essential numeric columns
    for col in ["bug_fixes", "churn", "max_ccn", "nloc", "coverage"]:
        if col not in df.columns:
            df[col] = 0

    # 4. Auto-calculate debt_score if priority/risk/debt_score are missing
    if "priority" not in df and "risk" not in df and "debt_score" not in df:
        df["debt_score"] = (
            df["max_ccn"].rank(pct=True, fill_value=0) * 0.35 +
            df["churn"].rank(pct=True, fill_value=0) * 0.35 +
            df["bug_fixes"].rank(pct=True, fill_value=0) * 0.30
        ) * 100

    # 5. Add fallback suggestions
    if "suggestions" not in df.columns:
        df["suggestions"] = "Review complex logic | Increase test coverage"

    return df

# Initialize session state
if "df" not in st.session_state:
    if os.path.exists("report.csv"):
        st.session_state["df"] = load_csv("report.csv")
    elif os.path.exists("dataset.csv"):
        st.session_state["df"] = load_csv("dataset.csv")

# ---------------- sidebar: data source ----------------
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Open a report", "Analyze a repo"])
    if source == "Open a report":
        up = st.file_uploader("report.csv or dataset.csv", type="csv")
        if up is not None:
            st.session_state["df"] = load_csv(up)
    else:
        repo = st.text_input("Path to git repo")
        cmd = st.radio("Mode", ["predict", "scan"], horizontal=True)
        model_path = st.text_input("Model file", "model.joblib")
        gh = st.text_input("GitHub owner/repo (optional)")
        issues_file = st.text_input("Issues cache file (optional)", "issues.json")
        cov = st.text_input("Coverage file (optional)")
        token = st.text_input("GitHub token (optional)", type="password") or os.environ.get("GITHUB_TOKEN")
        if st.button("Run analysis", type="primary") and repo:
            from techdebt.issues import resolve_issues
            from techdebt.pipeline import run
            with st.spinner("Analyzing..."):
                issues = resolve_issues(gh or None, issues_file or None, token)
                df_new = run(repo, cmd, model_path, issues, cov or None, True)
                df_new.to_csv("report.csv")
                st.session_state["df"] = load_csv("report.csv")

if "df" not in st.session_state:
    st.info("Upload `dataset.csv` or `report.csv` in the sidebar to begin.")
    st.stop()

df = st.session_state["df"].copy()

# Automatically select available score metric
score = "priority" if "priority" in df else ("risk" if "risk" in df else "debt_score")
df["folder"] = [str(p).rsplit("/", 1)[0] if "/" in str(p) else "." for p in df.index]

# ---------------- filters ----------------
with st.sidebar:
    st.header("Filters")
    q = st.text_input("Path contains")
    max_score_val = float(df[score].max()) if not df[score].empty and pd.notna(df[score].max()) else 1.0
    min_score = st.slider(f"Min {score}", 0.0, max_score_val, 0.0)
    top_n = st.slider("Files in charts", 5, 50, 15)

f = df[df.index.astype(str).str.contains(q, case=False, regex=False) & (df[score] >= min_score)]

# ---------------- KPIs ----------------
k = st.columns(4)
k[0].metric("Files analysed", len(f))
k[1].metric("High priority/risk (>=0.5)" if score != "debt_score" else "High debt (>=75)",
            int((f[score] >= (0.5 if score != "debt_score" else 75)).sum()))
k[2].metric("Avg coverage", f"{f['coverage'].mean():.0f}%" if "coverage" in f and f["coverage"].sum() > 0 else "n/a")
k[3].metric("Total bug-fix commits", int(f["bug_fixes"].sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Hotspots", "Folders", "File detail", "Table"])
palette = alt.Scale(scheme="redyellowgreen", reverse=True)
tip = [c for c in ["path", "priority", "risk", "debt_score", "coverage", "max_ccn", "churn", "bug_fixes"] if c in f.reset_index().columns]

with tab1:
    c1, c2 = st.columns(2)
    top = f.nlargest(top_n, score).reset_index()
    c1.subheader(f"Top {top_n} by {score}")
    c1.altair_chart(alt.Chart(top).mark_bar().encode(
        x=alt.X(score), y=alt.Y("path:N", sort="-x", title=None),
        color=alt.Color(score, scale=palette, legend=None), tooltip=tip), use_container_width=True)
    c2.subheader("Complexity vs. churn (size = lines, colour = score)")
    c2.altair_chart(alt.Chart(f.reset_index()).mark_circle(opacity=0.7).encode(
        x=alt.X("churn", scale=alt.Scale(type="symlog")), y="max_ccn", size="nloc",
        color=alt.Color(score, scale=palette), tooltip=tip), use_container_width=True)

with tab2:
    g = f.groupby("folder").agg(files=(score, "size"), avg_score=(score, "mean"), max_score=(score, "max"),
                                bug_fixes=("bug_fixes", "sum"), churn=("churn", "sum")).round(3).reset_index()
    st.altair_chart(alt.Chart(g.nlargest(top_n, "avg_score")).mark_bar().encode(
        x="avg_score", y=alt.Y("folder:N", sort="-x", title=None),
        color=alt.Color("avg_score", scale=palette, legend=None),
        tooltip=["folder", "files", "avg_score", "max_score", "bug_fixes"]), use_container_width=True)
    st.dataframe(g.sort_values("avg_score", ascending=False), use_container_width=True)

with tab3:
    if f.empty:
        st.warning("No files match the filters.")
    else:
        path = st.selectbox("File", f.sort_values(score, ascending=False).index)
        r = f.loc[path]
        m = st.columns(5)
        for col, name in zip(m, ["priority", "risk", "debt_score", "coverage", "max_ccn"]):
            if name in r and pd.notna(r[name]):
                col.metric(name, f"{r[name]:.2f}" if name != "max_ccn" else int(r[name]))
        st.markdown("**Suggestions**")
        for s in str(r.get("suggestions", "No suggestions")).split(" | "):
            st.write("- " + s)

with tab4:
    show = [c for c in f.columns if c not in ["folder"]]
    st.dataframe(f[show].sort_values(score, ascending=False), use_container_width=True)
    st.download_button("Download CSV", f.to_csv().encode(), "techdebt_filtered.csv")