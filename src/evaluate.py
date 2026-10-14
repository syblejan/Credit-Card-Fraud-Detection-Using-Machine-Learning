"""Leakage-aware credit-card fraud detection evaluation.

Requires the original creditcard.csv (Time,V1..V28,Amount,Class).
Do not use this educational experiment to approve/decline real transactions.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def read_data(path):
    df = pd.read_csv(path)
    required = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    if any(col not in df.columns for col in required):
        raise ValueError("Expected columns Time, V1..V28, Amount, Class")
    df = df[required].copy()
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite input data")
    if set(df["Class"].unique()) != {0, 1}:
        raise ValueError("Class must contain both 0 and 1")
    return df

def assess(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def run(data_path, output_dir):
    df = read_data(data_path)
    X, y = df.drop(columns="Class"), df["Class"].astype(int)
    # Split BEFORE any sampling, fitting, feature scaling or threshold tuning.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval,
        random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer(
            [("scale", StandardScaler(), ["Time", "Amount"])],
            remainder="passthrough")),
        ("model", LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=250, tol=1e-3,
            random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    # Threshold is chosen only from validation; final test is untouched.
    thresholds = np.arange(0.05, 0.951, 0.05)
    val_f1 = np.array([f1_score(y_val, val_probs >= t, zero_division=0)
                       for t in thresholds])
    chosen = float(thresholds[np.argmax(val_f1)])
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    report = {
        "dataset_rows": int(len(df)),
        "fraud_count": int(y.sum()),
        "fraud_prevalence": float(y.mean()),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "method": "Stratified 60/20/20 split before all fitting; balanced logistic regression; validation-only F1 threshold selection",
        "validation_selected_threshold": chosen,
        "test": assess(y_test, test_probs, chosen),
        "test_at_threshold_0_5": assess(y_test, test_probs, 0.5),
        "warning": "Educational dataset and results, not financial production validation."
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"actual": y_test.to_numpy(),
                  "fraud_probability": test_probs,
                  "predicted": (test_probs >= chosen).astype(int)}).to_csv(
        out / "holdout_predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", default=str(root / "data" / "creditcard.csv"))
    parser.add_argument("--out", default=str(root / "results"))
    args = parser.parse_args()
    run(args.data, args.out)
