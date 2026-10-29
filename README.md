# TechDebt-AI

Detects technical debt and predicts which files are likely to become hard to maintain.

## Setup
    pip install -r requirements.txt

## Commands
    python cli.py scan    REPO                        # heuristic debt score only (no training)
    python cli.py train   REPO --github owner/repo --issues-file issues.json
    python cli.py predict REPO --coverage coverage.xml --issues-file issues.json --top 25
    streamlit run dashboard.py                        # interactive dashboard (reads report.csv)

## Optional inputs
- Issue tracker: `--github owner/repo` (set env GITHUB_TOKEN to avoid rate limits) or
  `--issues-file issues.json` containing {"bug": [12, 40], "other": [5, 6]} (works for Jira/GitLab exports too)
- Coverage: `--coverage coverage.xml` (coverage.py: `coverage run -m pytest && coverage xml`; Jest: cobertura reporter)
  or an lcov.info file

## Modules
git_history.py  mine commits; label bug fixes (issue links + keywords)
issues.py       fetch/cache bug issues from GitHub
static_metrics.py  complexity, size, imports (lizard)
features.py     per-file features at a cutoff date (no leakage)
model.py        time-split Random Forest training
coverage_data.py  parse coverage.xml / lcov.info
explain.py      SHAP: why each file is risky
suggest.py      rule-based refactoring advice
pipeline.py     runs everything (used by CLI and dashboard)
report.py       report.csv + report.html
dashboard.py    Streamlit UI
