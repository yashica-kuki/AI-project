"""SHAP explanations: WHY the model thinks a file is risky."""
import numpy as np
import pandas as pd


def explain(model, X: pd.DataFrame, top_k: int = 3):
    """Returns (why: Series of text, shap_df: DataFrame of per-feature contributions to risk)."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # older shap: [class0, class1]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
        sv = sv[:, :, 1]
    shap_df = pd.DataFrame(sv, index=X.index, columns=["shap_" + c for c in X.columns])
    why = []
    for i in range(len(X)):
        top = np.argsort(-sv[i])[:top_k]
        why.append(", ".join(f"{X.columns[j]}={X.iloc[i, j]:.3g} ({sv[i, j]:+.2f})" for j in top if sv[i, j] > 0))
    return pd.Series(why, index=X.index, name="why"), shap_df


def save_summary_plot(shap_df: pd.DataFrame, X: pd.DataFrame, path: str = "shap_summary.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    shap.summary_plot(shap_df.values, X, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
