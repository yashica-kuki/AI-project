
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from techdebt.features import heuristic_debt_score
from techdebt.git_history import load_history
from techdebt.model import predict
from techdebt.suggest import add_suggestions

st.set_page_config(
    page_title="TechDebt AI",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 TechDebt AI")
st.subheader("AI-Powered Technical Debt Prediction & Codebase Health Analyzer")
st.write(
    "Analyze a public GitHub repository to discover technical debt, "
    "predict maintainability risks, and get actionable refactoring suggestions."
)
st.divider()

github_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repository",
)

top_n = st.slider("Files to display", min_value=5, max_value=50, value=20)


def validate_github_url(url):
    parsed = urlparse(url.strip())

    if parsed.scheme != "https" or parsed.hostname != "github.com":
        return None

    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        return None

    owner, repo = parts
    if not re.fullmatch(r"[A-Za-z0-9-]+", owner):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        return None

    return f"https://github.com/{owner}/{repo}.git"


if st.button("Analyze Repository", type="primary", use_container_width=True):
    clone_url = validate_github_url(github_url)

    if not clone_url:
        st.error("Enter a public repository URL like https://github.com/pallets/flask")
        st.stop()

    try:
        with tempfile.TemporaryDirectory(prefix="techdebt-") as temp_dir:
            repo_path = Path(temp_dir) / "repository"

            with st.spinner("Cloning repository history..."):
                result = subprocess.run(
                    [
                        "git", "clone", "--depth", "100",
                        clone_url, str(repo_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

            if result.returncode != 0:
                st.error("Could not clone this repository. Check that it is public and the URL is correct.")
                st.caption(result.stderr[-1500:])
                st.stop()

            with st.spinner("Analyzing files and predicting maintainability risk..."):
                history = load_history(str(repo_path))

                if history.empty:
                    st.error("This repository does not have enough Git history to analyze.")
                    st.stop()

                df = predict(str(repo_path), str(Path(__file__).parent / "model.joblib"))
                df["debt_score"] = heuristic_debt_score(df)
                df = add_suggestions(df)

                df = df.sort_values("risk", ascending=False)
                results = df.head(top_n).copy()

            st.success(f"Analysis complete — analyzed {len(df)} files.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Files analyzed", f"{len(df):,}")
            c2.metric("High-risk files", int((df["risk"] >= 0.7).sum()))
            c3.metric("Average debt score", f'{df["debt_score"].mean():.1f}/100')

            st.divider()
            st.subheader("Highest-risk files")

            display = results.reset_index()
            display = display.rename(columns={"path": "File"})
            display["risk"] = (display["risk"] * 100).round(1)
            display = display.rename(columns={
                "risk": "Predicted risk (%)",
                "debt_score": "Debt score",
                "max_ccn": "Max complexity",
                "nloc": "Lines of code",
                "fan_in": "Fan-in",
                "fan_out": "Fan-out",
                "churn": "Code churn",
                "bug_fixes": "Bug-fix commits",
                "n_authors": "Contributors",
                "suggestions": "Refactoring suggestions",
            })

            columns = [
                col for col in [
                    "File", "Predicted risk (%)", "Debt score",
                    "Max complexity", "Lines of code", "Fan-in",
                    "Fan-out", "Code churn", "Bug-fix commits",
                    "Contributors", "Refactoring suggestions",
                ] if col in display.columns
            ]

            st.dataframe(
                display[columns],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Explore individual files")
            selected_file = st.selectbox("Choose a file", df.index.astype(str))
            row = df.loc[selected_file]

            st.markdown(f"**File:** `{selected_file}`")
            a, b, c = st.columns(3)
            a.metric("Predicted risk", f'{row["risk"]:.1%}')
            b.metric("Technical debt score", f'{row["debt_score"]:.1f}/100')
            c.metric("Cyclomatic complexity", int(row["max_ccn"]))

            st.markdown("**Refactoring suggestions**")
            st.write(row["suggestions"])

            st.download_button(
                "Download analysis as CSV",
                data=df.to_csv().encode("utf-8"),
                file_name="techdebt_analysis.csv",
                mime="text/csv",
            )

    except subprocess.TimeoutExpired:
        st.error("Cloning took too long. Try a smaller repository.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.caption("Check the terminal for additional details.")
