# Credit Card Fraud Detection — Leakage-Aware Machine Learning

**Python · Pandas · Scikit-learn · Imbalanced Classification · Logistic Regression · PR-AUC**

An educational fraud detection experiment with a transparent, reproducible evaluation on the **original imbalanced transaction distribution**.

> **Not a banking or fraud-screening product.** No real transaction decisions should be based on this repository.

## Overview
The original notebook explores the well-known anonymized credit-card transactions dataset, examines imbalance, performs random undersampling, and trains Logistic Regression. This repository retains that notebook unchanged and adds a safer evaluation pipeline.

## Dataset
The original notebook output records **284,807 rows**, **492 fraudulent** and **284,315 legitimate** transactions, with 30 numeric features (`Time`, `V1`–`V28`, `Amount`) and binary target `Class` (0 = legitimate, 1 = fraud). Approximate fraud prevalence: **0.173%**.

The original CSV is **not included** in this repository. Obtain a legally usable copy of the public Credit Card Fraud Detection dataset (the ULB/Kaggle credit-card transactions dataset), verify its license/terms, and place it at `data/creditcard.csv`. The notebook originally loaded `/content/credit_data.csv`; rename or provide the matching path when running the new workflow. Do not commit financial or personal transaction records.

## Original Notebook — Historical Result
The unchanged `notebooks/original_credit_card_fraud_detection.ipynb` reported:
- **Train accuracy:** 94.155%
- **Test accuracy:** 93.909%
- Random sampling of 492 legitimate transactions followed by concatenation with all 492 fraud examples, *before splitting*.

**These accuracy figures came from an artificially balanced 984-row dataset** and are **not** evidence of performance on real-world, imbalanced transaction streams. The test set was only 197 observations. The original sampler also lacks a fixed random seed. Treat its results as an introductory baseline, not a production metric.

## What Was Improved
1. **Split first:** Stratified train / validation / test split at the original prevalence (60% / 20% / 20%) before any fitting, resampling, or threshold selection.
2. **Training-only preprocessing:** `StandardScaler` for `Time` and `Amount` fitted inside a pipeline on training data only. The anonymized `V` features pass through.
3. **Class imbalance:** `LogisticRegression(class_weight="balanced")` avoids discarding most legitimate transactions and gives rare fraud cases training weight.
4. **Validation-only threshold tuning:** Choose a decision threshold based on validation F1; report results on untouched holdout data.
5. **Appropriate fraud metrics:** Precision, recall, F1, **PR-AUC (average precision)**, ROC-AUC, and confusion matrix. Accuracy alone can be highly misleading for a ~0.17%-positive dataset.
6. **Reproducible experiments:** Fixed split random seed, explicit dependency list, machine-readable JSON and per-transaction holdout output.

Choosing F1 for threshold selection is an educational convention. A real fraud system would instead account for false-positive costs, missed fraud costs, calibration, temporal drift, privacy and regulatory responsibilities.

## Project Structure
```text
Credit-Card-Fraud-Detection-/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── evaluate.py
└── notebooks/
    ├── original_credit_card_fraud_detection.ipynb
    └── leakage_aware_fraud_detection.ipynb
```

Optional local `data/creditcard.csv` and generated `results/` are excluded from version control.

## Run Locally
```bash
git clone https://github.com/syblejan/Credit-Card-Fraud-Detection-.git
cd Credit-Card-Fraud-Detection-
python -m venv .venv
# Activate .venv for your platform
python -m pip install -r requirements.txt
# Place original dataset in data/creditcard.csv
python src/evaluate.py
```

Alternative source file:
```bash
python src/evaluate.py --data /path/to/creditcard.csv
```

Outputs (after a successful run): `results/evaluation.json`, `results/holdout_predictions.csv`.

## Verified Full-Dataset Execution

The corrected pipeline was **executed on the original uploaded 284,807-row CSV** using Python and scikit-learn, with the scalable `lbfgs` Logistic Regression solver (`max_iter=250`, `tol=1e-3`). Stratified train/validation/test split: **170,883 / 56,962 / 56,962**. The classification threshold **0.95** was selected using validation F1 only, without test-label selection.

| Test metric | Verified result |
|---|---:|
| Precision (fraud) | **29.79%** |
| Recall (fraud) | **88.78%** |
| F1 (fraud) | **44.62%** |
| PR-AUC / Average Precision | **0.7177** |
| ROC-AUC | **0.9728** |

### Confusion Matrix — Untouched Holdout

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual legitimate | 56,659 | 205 |
| Actual fraud | 11 | 87 |

Among the 98 actual fraudulent test cases, the model identified 87 and missed 11. However, it also flagged 205 legitimate transactions as fraud, so the **positive prediction precision is only 29.79%**. The ROC-AUC alone should not be interpreted as proof of a good deployed fraud system.

At the default threshold 0.50, fraud recall was higher (**91.84%**) but precision much lower (**6.06%**) with 1,395 false positives. This illustrates the precision–recall trade-off.

**Execution scope:** Successfully ran the corrected `src/evaluate.py` workflow with the original ZIP's `creditcard.csv`. It generated `results/evaluation.json` and `results/holdout_predictions.csv`. These output files are generated locally, not committed to the public GitHub repository.

## Limitations

- The original 93.91% accuracy was on a balanced subset and is not comparable to the new full-distribution scores.
- Stratified random holdout does not simulate live chronological fraud detection.
- No cost-sensitive business threshold, production monitoring, calibration review, model drift validation, or privacy/compliance assessment.
- The source CSV contains anonymized features; public provenance and usage terms should be independently checked.

**These results are educational and are not validated for real payment decisions.**

## Author
**Syble Jan** · BS Computer Science · AI/ML projects

[GitHub Profile](https://github.com/syblejan)
