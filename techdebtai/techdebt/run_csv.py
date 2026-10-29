import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    # 1. Load dataset.csv
    print("Loading dataset.csv...")
    df = pd.read_csv("dataset.csv")

    # 2. Separate features (X) and target label (y)
    # Drop non-numeric / identifier columns
    ignore_cols = ["file name", "repo", "label"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    X = df[feature_cols].fillna(0)
    y = df["label"]

    print(f"Features: {feature_cols}")
    print(f"Dataset shape: {df.shape}")

    # 3. Train-Test Split & Random Forest Model Training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = clf.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, y_pred))

    # 5. Generate Top 20 Predictions / High Risk Files Report
    df["predicted_risk_prob"] = clf.predict_proba(X)[:, 1]
    top_20 = df.sort_values(by="predicted_risk_prob", ascending=False).head(20)

    output_cols = ["file name", "repo", "predicted_risk_prob"] + feature_cols[:5]
    top_20[output_cols].to_csv("report.csv", index=False)
    print("Saved top 20 risk predictions to report.csv")

    # 6. Output HTML Report
    top_20[output_cols].to_html("report.html", index=False)
    print("Saved HTML report to report.html")

if __name__ == "__main__":
    main()