"""Training-data store + model + scoring for the multi-repository platform.

  data/master_dataset.csv   every row ever used for training (original dataset + every repo you add)
  data/model.joblib         Random Forest trained on ALL rows of the master dataset
  data/scored.csv           every file scored by a model that never saw its own repository (leave-repos-out)
  data/meta.json            metrics, feature importances, per-repo AUC
"""
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

import extractor as E

DATA = Path(os.environ.get("TECHDEBT_DATA", Path(__file__).resolve().parent / "data"))
MASTER, MODEL, SCORED = DATA / "master_dataset.csv", DATA / "model.joblib", DATA / "scored.csv"
META, CFG, USERS = DATA / "meta.json", DATA / "config.json", DATA / "user_repos.json"

# history columns: usable, but partly tied to how the label is defined (label = bug-fix commits >= 2)
HISTORY = ["commit_count", "author_count", "days_since_change", "churn_rate", "refactor_commits"]
ALWAYS_DROP = ["bug_fix_commits"]            # the label is defined from it -> pure leakage
HIGH_RISK = 0.5


def _load(path, default):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def _save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ scoring helpers
def feature_list(df, mode):
    drop = {"label", "repo", "file name"} | set(ALWAYS_DROP)
    cols = [c for c in df.select_dtypes("number").columns if c not in drop]
    return [c for c in cols if c not in HISTORY] if mode == "static" else cols


def make_rf():
    # 150 shallower trees: same AUC as 250 deep ones, but ~3x faster SHAP (retraining happens on every added repo)
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_depth=16,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=42)


def heuristic_debt_score(df):
    w = {"max_complexity": .15, "cyclomatic_complexity": .10, "max_nesting_depth": .10, "sloc": .10,
         "dependency_count": .08, "halstead_effort": .07, "num_functions": .05, "avg_function_len": .05}
    s = sum(df[c].rank(pct=True) * wt for c, wt in w.items())
    s = s + (1 - df["maintainability_index"].rank(pct=True)) * .15 + (1 - df["is_tested"]) * .05
    return (100 * s / (sum(w.values()) + .20)).round(1)


def make_suggestions(df):
    q = lambda c, floor: max(df[c].quantile(.9), floor)
    thr = {"cc": q("max_complexity", 15), "nest": q("max_nesting_depth", 5), "sloc": q("sloc", 400),
           "alen": q("avg_function_len", 40), "dep": q("dependency_count", 12), "hal": q("halstead_effort", 20000)}
    mi_thr = min(df["maintainability_index"].quantile(.1), 40)

    def row(r):
        t = []
        if r.is_tested == 0 and (r.max_complexity >= 10 or r.sloc >= 200) and not r.name.split("/")[-1].startswith("test_"):
            t.append("No tests on non-trivial code: add tests BEFORE refactoring.")
        if r.max_complexity >= thr["cc"]:
            t.append(f"Very complex function (complexity {r.max_complexity:.0f}): split it, use guard clauses.")
        if r.max_nesting_depth >= thr["nest"]:
            t.append(f"Deep nesting ({r.max_nesting_depth:.0f} levels): flatten with early returns / helper functions.")
        if r.avg_function_len >= thr["alen"]:
            t.append(f"Long functions (avg {r.avg_function_len:.0f} lines): extract single-purpose functions.")
        if r.sloc >= thr["sloc"]:
            t.append(f"Large file ({r.sloc:.0f} source lines): split by responsibility.")
        if r.dependency_count >= thr["dep"]:
            t.append(f"Many imports ({r.dependency_count:.0f}): reduce coupling, invert dependencies.")
        if r.halstead_effort >= thr["hal"]:
            t.append("High Halstead effort: dense code - simplify expressions.")
        if r.maintainability_index <= mi_thr:
            t.append(f"Low maintainability index ({r.maintainability_index:.0f}): prioritise for refactoring.")
        if r.comment_ratio < 0.05 and r.sloc >= 200:
            t.append("Almost no comments/docstrings on a large file: document intent and public API.")
        return " | ".join(t) if t else "No major issues"
    return df.apply(row, axis=1)


def explain(model, X):
    try:
        import shap
    except ImportError:
        return None, None
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    shap_df = pd.DataFrame(sv, index=X.index, columns=["shap_" + c for c in X.columns])
    why = []
    for i in range(len(X)):
        top = np.argsort(-sv[i])[:3]
        why.append(", ".join(f"{X.columns[j]}={X.iloc[i, j]:.3g} ({sv[i, j]:+.2f})" for j in top if sv[i, j] > 0))
    return pd.Series(why, index=X.index, name="why"), shap_df


# ------------------------------------------------------------------ training
def init_base(base_csv):
    """First run: copy the original dataset into the master dataset and remember the outlier caps."""
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base_csv)
    missing = [c for c in E.COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{base_csv} is missing columns: {missing}")
    df[E.COLUMNS].to_csv(MASTER, index=False)
    caps = {}
    for c in ["commit_count", "churn_rate", "max_complexity"]:   # columns that were clipped at ~p99 in the dataset
        if (df[c] == df[c].max()).sum() >= 10:
            caps[c] = float(df[c].max())
    _save(CFG, {"mode": "static", "caps": caps})
    _save(USERS, {})


def train_all(mode=None, progress=print):
    t0 = time.time()
    cfg = _load(CFG, {"mode": "static", "caps": {}})
    mode = mode or cfg.get("mode", "static")
    cfg["mode"] = mode
    _save(CFG, cfg)

    df = pd.read_csv(MASTER)
    df.index = pd.Index(df["file name"].str.replace("\\", "/", regex=False), name="path")
    feats = feature_list(df, mode)
    X, y, g = df[feats], df["label"].astype(int), df["repo"]
    n_repos = g.nunique()

    progress(f"validating on {n_repos} repositories ({len(df)} files) - each repo is scored by a model that never saw it")
    if n_repos >= 3:
        cv = GroupKFold(n_splits=min(10, n_repos)); kw = {"groups": g}; how = f"leave-repos-out ({cv.n_splits}-fold)"
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=0); kw = {}; how = "random 5-fold (add more repos!)"
    oof = cross_val_predict(make_rf(), X, y, cv=cv, method="predict_proba", **kw)[:, 1]

    per_repo = {}
    for r, idx in df.groupby("repo").indices.items():
        yy = y.iloc[idx]
        if len(idx) >= 8 and yy.nunique() == 2:
            per_repo[r] = round(float(roc_auc_score(yy, oof[idx])), 3)

    progress("training the final model on all rows")
    model = make_rf().fit(X, y)
    joblib.dump({"model": model, "features": feats, "mode": mode}, MODEL)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

    progress("computing SHAP explanations and suggestions")
    scored = df.copy()
    scored["risk"] = np.round(oof, 4)
    scored["debt_score"] = heuristic_debt_score(df)
    scored["suggestions"] = make_suggestions(df)
    scored["rel"] = [p.split("/", 1)[1] if "/" in p else p for p in scored.index]
    why, shap_df = explain(model, X)
    if why is not None:
        scored["why"] = why
        scored = scored.join(shap_df)
    scored.to_csv(SCORED)

    meta = {"mode": mode, "features": feats, "n_rows": int(len(df)), "n_repos": int(n_repos), "validation": how,
            "roc_auc": round(float(roc_auc_score(y, oof)), 3), "pr_auc": round(float(average_precision_score(y, oof)), 3),
            "base_rate": round(float(y.mean()), 3),
            "importance": [{"feature": f, "value": round(float(v), 4)} for f, v in imp.items()],
            "per_repo_auc": per_repo, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    _save(META, meta)
    progress(f"done: leave-repos-out ROC-AUC {meta['roc_auc']} (PR-AUC {meta['pr_auc']}) in {meta['seconds']}s")
    return meta


# ------------------------------------------------------------------ adding a repository
def analyze_repo(source, name=None, progress=print):
    """Extract features for a repo (path or git URL), append to the master dataset, retrain, return its name."""
    cfg, users = _load(CFG, {"caps": {}}), _load(USERS, {})
    progress("resolving repository")
    path = E.clone_if_url(source, str(DATA / "clones"))
    name = (name or os.path.basename(path.rstrip("/\\"))).strip().replace("/", "-")
    master = pd.read_csv(MASTER)
    if name in set(master.repo) and name not in users:      # never overwrite a base-dataset repo
        name = name + "-new"
    rows = E.extract_repo(path, name, caps=cfg.get("caps"), progress=progress)
    master = pd.concat([master[master.repo != name], rows], ignore_index=True)
    master.to_csv(MASTER, index=False)
    users[name] = {"source": source, "added": time.strftime("%Y-%m-%d %H:%M"), "files": int(len(rows))}
    _save(USERS, users)
    progress(f"added {len(rows)} files of '{name}' to the training data (now {len(master)} rows) - retraining")
    train_all(progress=progress)
    return name


def delete_repo(name, progress=print):
    users = _load(USERS, {})
    if name not in users:
        raise ValueError("Only repositories you added can be deleted (base-dataset repos are protected).")
    master = pd.read_csv(MASTER)
    master[master.repo != name].to_csv(MASTER, index=False)
    users.pop(name)
    _save(USERS, users)
    train_all(progress=progress)
    return name


# ------------------------------------------------------------------ read API
def ready():
    return MODEL.exists() and SCORED.exists()


def repo_table():
    s = pd.read_csv(SCORED, index_col=0)
    meta, users = _load(META, {}), _load(USERS, {})
    g = s.groupby("repo")
    t = pd.DataFrame({"files": g.size(), "mean_risk": g.risk.mean().round(3),
                      "high_risk": g.risk.apply(lambda r: int((r >= HIGH_RISK).sum())),
                      "positive_rate": g.label.mean().round(3)})
    out = []
    for r, row in t.iterrows():
        d = {"repo": r, "files": int(row.files), "mean_risk": float(row.mean_risk), "high_risk": int(row.high_risk),
             "positive_rate": float(row.positive_rate), "auc": meta.get("per_repo_auc", {}).get(r), "source": "added" if r in users else "base"}
        if r in users:
            d["added"] = users[r]["added"]
        out.append(d)
    return out


def repo_report(name):
    s = pd.read_csv(SCORED, index_col=0)
    s = s[s.repo == name]
    if s.empty:
        raise KeyError(name)
    s = s.drop(columns=["file name", "repo"]).reset_index(drop=True)
    return json.loads(s.to_json(orient="records"))


def repo_csv(name):
    s = pd.read_csv(SCORED, index_col=0)
    s = s[s.repo == name]
    return s[[c for c in s.columns if not c.startswith("shap_")]].to_csv()


def ensure_ready(base=None, progress=print):
    if MASTER.exists() and ready():
        return
    if not MASTER.exists():
        cand = [base, DATA / "base_dataset.csv", Path("dataset.csv"), Path(__file__).parent / "dataset.csv"]
        src = next((Path(c) for c in cand if c and Path(c).exists()), None)
        if src is None:
            raise SystemExit("First run needs your training CSV:  python server.py --base path\\to\\dataset.csv")
        progress(f"first run: using {src} as the base training dataset")
        init_base(src)
    train_all(progress=progress)


if __name__ == "__main__":      # headless use:  python engine.py add <path-or-git-url> [name]
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "add":
        ensure_ready()
        print("added:", analyze_repo(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    else:
        print(__doc__)
