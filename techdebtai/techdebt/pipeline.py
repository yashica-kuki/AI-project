"""One function that runs everything; used by cli.py and dashboard.py."""
import joblib
import numpy as np

from .coverage_data import load_coverage
from .features import build_snapshot, heuristic_debt_score
from .git_history import load_history
from .suggest import add_suggestions


def run(repo, mode="predict", model_path="model.joblib", issues=None, coverage=None, explain=True):
    hist = load_history(repo, issues)
    df = build_snapshot(repo, hist, int(hist.ts.max()) + 1, rev="HEAD")
    df["debt_score"] = heuristic_debt_score(df)

    shap_df = None
    if mode == "predict":
        bundle = joblib.load(model_path)
        X = df[bundle["features"]]
        df["risk"] = bundle["model"].predict_proba(X)[:, 1].round(3)
        base = df["risk"]
        if explain:
            try:
                from .explain import explain as shap_explain, save_summary_plot
                df["why"], shap_df = shap_explain(bundle["model"], X)
                df = df.join(shap_df)
                save_summary_plot(shap_df, X)
            except ImportError:
                print("shap/matplotlib not installed -> skipping explanations (pip install shap matplotlib)")
    else:
        df["risk"] = np.nan
        base = df["debt_score"] / 100

    if coverage:
        # Files absent from the coverage report (tests, docs, unmeasured dirs) stay NaN = "unknown", not 0%
        df["coverage"] = load_coverage(coverage, df.index).reindex(df.index).round(1)
        df["priority"] = (base * (1 - df["coverage"].fillna(0) / 100)).round(3)   # risky AND untested first
    else:
        df["priority"] = base.round(3)

    df = add_suggestions(df)
    return df.sort_values("priority", ascending=False)
