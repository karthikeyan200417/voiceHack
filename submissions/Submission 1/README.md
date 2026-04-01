# Submission 1 — Call Quality Auto-Flagger
### CareCaller Hackathon 2026 | Team: Hakkuna Matata

A classical ML pipeline that compares three models to automatically detect healthcare AI voice agent calls requiring a human review ticket.

---

## 📋 Overview

| Property | Detail |
|----------|--------|
| Environment | Local Python script |
| Language | Python 3 |
| NLP Approach | TF-IDF + keyword extraction |
| Class Imbalance | SMOTE |
| Models | Random Forest, Gradient Boosting, XGBoost |
| Feature Count | 11 |
| Bonus Category | ❌ |

---

## 🗂️ Files

```
Submission 1/
├── ticket_classification_model.py     # Main pipeline script
├── test_predictions.csv               # Output: call_id + predictions
├── test_predictions_distribution.png  # Output: probability histogram & pie chart
└── README.md
```

> The following are generated on run and saved to the working directory:
> `feature_importance.png`, `roc_curves.png`, `confusion_matrices.png`

---

## ⚙️ Setup & Run

**Install dependencies**
```bash
pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn tqdm
```

**Run**
```bash
python "Submission 1/ticket_classification_model.py"
```

> Expects the following CSVs relative to the workspace root:
> ```
> Datasets/csv/hackathon_train.csv
> Datasets/csv/hackathon_val.csv
> Datasets/csv/hackathon_test.csv
> ```

---

## 🔧 Pipeline

```
Raw CSVs
   │
   ▼
Preprocessing & Feature Engineering (11 features)
   │
   ▼
SMOTE (class balancing)
   │
   ▼
Train 3 Models in parallel
   │
   ▼
Evaluate on Validation Set → Select Best by F1
   │
   ▼
Threshold Analysis (0.3 → 0.7)
   │
   ▼
Predict on Test Set → Save submission
```

---

## 🧠 Feature Engineering

| Feature | Description |
|---------|-------------|
| `call_duration` | Raw call length in seconds |
| `whisper_mismatch_count` | STT disagreement count |
| `response_completeness` | Fraction of questions answered |
| `outcome_encoded` | Label-encoded call outcome |
| `is_high_risk_outcome` | 1 if outcome is wrong_number / escalated / incomplete / voicemail |
| `response_count` | Number of Q&A pairs parsed from `responses_json` |
| `critical_value_mentions` | Count of health keywords (weight, BP, medication, etc.) in answers |
| `validation_flag_count` | Count of error / warning / mismatch in validation notes |
| `transcript_length` | Character count of transcript |
| `word_count` | Word count of transcript |
| `duration_to_response_ratio` | Call duration ÷ response count (pace anomaly) |

---

## 🤖 Models

| Model | Key Hyperparameters |
|-------|---------------------|
| Random Forest | `n_estimators=100`, `max_depth=15`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5` |
| XGBoost | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6` |

All models are trained on SMOTE-balanced data. The best model is selected by validation F1 score.

---

## ⚖️ Class Imbalance — SMOTE

The dataset is imbalanced (more no-ticket calls than ticket calls). SMOTE generates synthetic minority class samples before training to give each model a balanced view of both classes.

```
Before SMOTE:  Class 0 >> Class 1
After SMOTE:   Class 0 ≈ Class 1
```

---

## 📊 Evaluation Metrics

Each model is evaluated on the validation set with:

- F1 Score *(primary)*
- Recall / Precision
- Accuracy
- AUC-ROC
- Cohen's Kappa
- Matthews Correlation Coefficient
- Overfitting gap (Train F1 − Val F1)

Threshold sweep from 0.3 to 0.7 is performed on the best model to find the optimal operating point.

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `test_predictions.csv` | `call_id`, `predicted_has_ticket` for all 159 test calls |
| `feature_importance.png` | Top 10 feature importances (best model) |
| `roc_curves.png` | ROC curves for all 3 models |
| `confusion_matrices.png` | Confusion matrices for all 3 models |
| `test_predictions_distribution.png` | Probability histogram + prediction pie chart |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
tqdm
```
