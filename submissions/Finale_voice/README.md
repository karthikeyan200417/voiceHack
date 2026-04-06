# CareCaller Hackathon — Team Hakkuna Matata

A machine learning solution for predicting support ticket creation from healthcare call transcripts, built for the CareCaller Hackathon 2026.

## Problem Statement

Given call center data from a healthcare provider (CareCaller), predict whether a call will result in a support ticket being raised (`has_ticket`). The dataset is highly imbalanced (~67x more non-ticket calls than ticket calls), making this a challenging binary classification problem.

## Solution Overview

A **stacked ensemble** combining three base models with soft rule-based signals, optimized for F1 score on the validation set.

### Architecture

```
XGBoost ──┐
LightGBM ──┼──► Meta Learner (L2-regularized Logistic Regression) ──► Final Prediction
MLP ───────┤
Rule Score ┘
```

### Key Design Decisions

- **XGBoost & LightGBM** trained on original imbalanced data using `scale_pos_weight` (SMOTE hurt precision)
- **MLP** with heavy regularization (Dropout 0.5/0.4/0.3) and early stopping to prevent overfitting
- **Soft rule scores** (0–1 weighted signals) instead of hard overrides — rules had only ~0.03 precision as hard overrides
- **Meta-learner** with L2 penalty (C=0.1) and feature scaling to prevent MLP domination
- **Threshold optimized for F1** (not recall-biased)

## Features

The feature engineering pipeline extracts **128 features** across 10 groups:

| Group | Description |
|-------|-------------|
| Call Metadata | Duration, billing, attempt number, mismatch counts, form submission |
| Time Features | Hour/day cyclical encoding, weekend/after-hours flags |
| Pipeline Flags | Verification status, skip indicators |
| Outcome & Cycle | One-hot encoded outcomes, cycle status |
| Engineered Ratios | Completeness anomalies, duration per answer, word ratios |
| Q&A Analysis | Empty answers, suspicious weights, implausible goals |
| Transcript Keywords | Opt-out, wrong number, medical advice, escalation, frustration patterns |
| Validation Notes | Mismatch, fabrication, skipping, error signals |
| Cross-feature Anomalies | Misclassified outcomes, missed escalations |
| Anomaly Score | Isolation Forest on key numeric features |

Additionally, **40 TF-IDF SVD components** are extracted from combined transcript text.

## Results

| Metric | Value |
|--------|-------|
| Best Threshold | 0.93 |
| F1 Score | 0.429 |
| Recall | 0.522 |
| Precision | 0.364 |

### Component Comparison (at threshold 0.40)

| Model | F1 |
|-------|----|
| XGBoost | 0.089 |
| LightGBM | 0.000 |
| MLP | ~0.453 (best val) |
| Rules | — |
| **Stacked Ensemble** | **0.429** |

## Requirements

```
xgboost
lightgbm
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
shap
torch
```

Install all dependencies:

```bash
pip install xgboost lightgbm scikit-learn imbalanced-learn pandas numpy matplotlib seaborn shap torch
```

## Usage

The solution is provided as a Jupyter notebook. Two versions are available:

- `carecaller_problem1_Team_Hakkuna Matata_combined.ipynb` — Full combined pipeline (recommended)
- `carecaller_problem1_Team_Hakkuna Matata.ipynb` — Original version

### Data

Place the following CSV files in the working directory:

- `hackathon_train.csv` — Training data (8,029 rows, 55 columns)
- `hackathon_val.csv` — Validation data (1,715 rows, 55 columns)
- `Total.csv` — Test data (11,480 rows, labels hidden)

### Running the Notebook

Open and run all cells in order:

1. **Cell 1** — Install dependencies
2. **Cell 2** — Imports
3. **Cell 3** — Load data
4. **Cell 4** — Pattern definitions & helper functions
5. **Cell 5** — Feature engineering function
6. **Cell 6** — Build features + TF-IDF
7. **Cell 7** — Train XGBoost
8. **Cell 8** — Train LightGBM
9. **Cell 9** — Train MLP (regularized)
10. **Cell 10** — Compute soft rule scores
11. **Cell 11** — OOF stacking & meta-learner
12. **Cell 12** — Threshold sweep (F1-optimized)
13. **Cell 13** — Validation report + per-outcome analysis
14. **Cell 14** — SHAP feature importance

## Team

**Team Hakkuna Matata**

## Version History

- **v3 (current)** — Precision-Recall balanced
  - Rules converted to soft votes (not hard overrides)
  - MLP uses heavy regularization + shorter training
  - Meta-learner gets L2 penalty + feature scaling
  - Threshold optimized for F1
  - XGB/LGB trained on original data (not SMOTE)
