"""Train the maintainability-risk model (time-based, leakage-safe)."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import DAY, FEATURES, build_snapshot, future_bug_label
from .git_history import load_history


def make_dataset(repo, horizon_days=180, n_snapshots=4, issues=None) -> pd.DataFrame:
    """Snapshot k looks back k*horizon from the last commit: features use the past, label uses the next window."""
    hist = load_history(repo, issues)
    end, start, H = hist.ts.max(), hist.ts.min(), horizon_days * DAY
    frames = []
    for k in range(1, n_snapshots + 1):
        cutoff = end - k * H
        if cutoff <= start:
            break
        df = build_snapshot(repo, hist, cutoff)
        if df.empty:
            continue
        df["label"] = future_bug_label(hist, cutoff, H).reindex(df.index).fillna(0).astype(int)
        df["snapshot"] = k
        frames.append(df)
        print(f"  snapshot {k}: {len(df)} files, {df.label.mean():.1%} positive")
    if not frames:
        raise SystemExit("Not enough history to build a training set.")
    return pd.concat(frames)


def train(repo, out="model.joblib", horizon_days=180, n_snapshots=4, issues=None):
    data = make_dataset(repo, horizon_days, n_snapshots, issues)
    if data.label.nunique() < 2:
        raise SystemExit("Only one class in labels - need more bug-fix history or a longer horizon.")

    test, train_ = data[data.snapshot == 1], data[data.snapshot > 1]   # newest snapshot = test set
    if train_.empty or train_.label.nunique() < 2 or test.label.nunique() < 2:
        train_, test = train_test_split(data, test_size=0.25, stratify=data.label, random_state=42)
        print("  (few snapshots -> using a random split instead of a time split)")

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ).fit(train_[FEATURES], train_.label)

    p = model.predict_proba(test[FEATURES])[:, 1]
    print(f"\nROC-AUC: {roc_auc_score(test.label, p):.3f} | "
          f"PR-AUC: {average_precision_score(test.label, p):.3f} | base rate: {test.label.mean():.1%}")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop drivers of risk:\n", imp.head(8).round(3).to_string())

    joblib.dump({"model": model, "features": FEATURES, "horizon_days": horizon_days}, out)
    print(f"\nSaved model -> {out}")
