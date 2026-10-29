"""Turn metrics into concrete, explainable refactoring advice."""
import pandas as pd

# (feature, absolute minimum to ever flag, text). Also requires being in the repo's top decile.
RULES = [
    ("max_ccn", 10, "Very complex function (max cyclomatic complexity {v:.0f}): split it up, use guard clauses / strategy pattern."),
    ("max_func_nloc", 60, "Long function ({v:.0f} lines): extract smaller, single-purpose functions."),
    ("nloc", 500, "Large file ({v:.0f} lines): split by responsibility into modules."),
    ("fan_in", 5, "Many files depend on this ({v:.0f}): put a stable interface in front and add contract tests."),
    ("fan_out", 15, "Depends on many modules ({v:.0f}): reduce coupling, apply dependency inversion."),
    ("churn", 300, "Change hotspot ({v:.0f} lines churned): stabilise the design, add tests before more edits."),
    ("bug_fixes", 3, "Repeated bug fixes ({v:.0f}): add regression tests, do a root-cause review."),
    ("n_authors", 6, "Many authors ({v:.0f}) with diffuse ownership: assign an owner, document conventions."),
    ("todo_count", 3, "{v:.0f} TODO/FIXME markers: triage into tickets or resolve."),
]


def add_suggestions(df: pd.DataFrame, quantile: float = 0.9) -> pd.DataFrame:
    thr = {f: max(df[f].quantile(quantile), floor) for f, floor, _ in RULES}

    def per_row(row):
        tips = [txt.format(v=row[f]) for f, _, txt in RULES if row[f] >= thr[f]]
        cov = row.get("coverage")
        risky = row["max_ccn"] >= 10 or row["bug_fixes"] >= 2 or row["churn"] >= thr["churn"]
        if cov is not None and pd.notna(cov) and cov < 50 and risky:
            tips.insert(0, f"Only {cov:.0f}% test coverage on risky code: write tests BEFORE refactoring.")
        return " | ".join(tips) if tips else "No major issues"

    df = df.copy()
    df["suggestions"] = df.apply(per_row, axis=1)
    return df
