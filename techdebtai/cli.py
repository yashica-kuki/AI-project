"""Usage:
  python cli.py train   REPO [--horizon 180] [--snapshots 4] [--github owner/repo] [--issues-file issues.json]
  python cli.py predict REPO [--top 25] [--coverage coverage.xml] [--github owner/repo] [--issues-file issues.json]
  python cli.py scan    REPO [--top 25] [--coverage coverage.xml]      # no ML, heuristic only
Then:  streamlit run dashboard.py
"""
import argparse
import os

from techdebt.issues import resolve_issues
from techdebt.model import train
from techdebt.pipeline import run
from techdebt.report import write_reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "scan"])
    ap.add_argument("repo")
    ap.add_argument("--horizon", type=int, default=180, help="prediction window in days")
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--github", help="owner/repo on GitHub to pull bug issues from (uses GITHUB_TOKEN if set)")
    ap.add_argument("--issues-file", help="JSON cache of issue numbers (read if exists, else written)")
    ap.add_argument("--coverage", help="coverage.xml (Cobertura) or lcov.info")
    ap.add_argument("--no-explain", action="store_true", help="skip SHAP")
    a = ap.parse_args()

    issues = resolve_issues(a.github, a.issues_file, os.environ.get("GITHUB_TOKEN"))
    if a.cmd == "train":
        train(a.repo, a.model, a.horizon, a.snapshots, issues)
        return

    df = run(a.repo, a.cmd, a.model, issues, a.coverage, not a.no_explain)
    cols = [c for c in ["priority", "risk", "debt_score", "coverage", "max_ccn", "churn", "bug_fixes", "why", "suggestions"] if c in df]
    print(df.head(a.top)[cols].to_string(max_colwidth=70))
    write_reports(df, a.top)


if __name__ == "__main__":
    main()
