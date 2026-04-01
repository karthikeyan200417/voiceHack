# CareCaller Hackathon 2026 — Problem 1: Call Quality Auto-Flagger
### Team: Hakkuna Matata

Automatically detect healthcare AI voice agent calls that require a human review ticket, using structured metadata, transcript NLP, and machine learning.

---

## Problem Statement

Given call transcripts and metadata from a healthcare AI voice agent (TrimRX medication refill check-ins), predict whether a call will generate a support ticket (`has_ticket`). The system must catch bad calls (high recall) while minimising false alarms (reasonable precision).

---

## Dataset

| Split | Rows | Tickets |
|-------|------|---------|
| Train | 689  | labeled |
| Val   | 144  | labeled |
| Test  | 159  | hidden  |

Key fields: `call_duration`, `outcome`, `whisper_mismatch_count`, `response_completeness`, `transcript_text`, `responses_json`, `validation_notes`, `has_ticket`.

See [`../Datasets/DATA_DICTIONARY.md`](../Datasets/DATA_DICTIONARY.md) for the full field reference.

---

## Submissions Overview

| | Submission 1 | Submission 2 |
|---|---|---|
| File | `ticket_classification_model.py` | `carecaller_problem1_Team_Hakkuna Matata_File2.ipynb` |
| Environment | Local Python script | Google Colab notebook |
| NLP approach | TF-IDF + keyword extraction | Regex pattern matching |
| Models | Random Forest, Gradient Boosting, XGBoost ensemble | XGBoost + deterministic rule engine |
| Class imbalance | SMOTE | `scale_pos_weight` |
| Feature count | 11 | 50+ |
| Bonus (category) | No | Yes |

---

## Submission 1 — `ticket_classification_model.py`

### Approach

A classical ML pipeline with three models compared head-to-head. Focuses on simplicity and interpretability with a small, clean feature set.

### Pipeline

```
Raw CSVs → Preprocessing → SMOTE → Model Training → Threshold Analysis → Test Predictions
```

**Feature Engineering (11 features)**
- `call_duration`, `whisper_mismatch_count`, `response_completeness` — raw metadata
- `outcome_encoded` — label-encoded call outcome
- `is_high_risk_outcome` — binary flag for wrong_number / escalated / incomplete / voicemail
- `response_count`, `critical_value_mentions` — parsed from `responses_json`
- `validation_flag_count` — error/warning/mismatch count in validation notes
- `transcript_length`, `word_count` — text length signals
- `duration_to_response_ratio` — pace anomaly detector

**Models Trained**
- Random Forest (`n_estimators=100, max_depth=15`)
- Gradient Boosting (`n_estimators=100, learning_rate=0.1`)
- XGBoost (`n_estimators=100, learning_rate=0.1, max_depth=6`)

All trained on SMOTE-balanced data. Best model selected by validation F1.

**Threshold Analysis**
Tests thresholds 0.3 → 0.7 to find the optimal precision/recall trade-off.

### Output Files

| File | Description |
|------|-------------|
| `test_predictions.csv` | `predicted_has_ticket`, `ticket_probability` for all 159 test calls |
| `feature_importance.png` | Top 10 feature importances for best model |
| `roc_curves.png` | ROC curves for all 3 models |
| `confusion_matrices.png` | Confusion matrices for all 3 models |
| `test_predictions_distribution.png` | Probability histogram + prediction pie chart |

### How to Run

```bash
pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn tqdm
python "Submission 1/ticket_classification_model.py"
```

> Expects `Datasets/csv/hackathon_train.csv`, `hackathon_val.csv`, `hackathon_test.csv` relative to the workspace root.

---

## Submission 2 — `carecaller_problem1_Team_Hakkuna Matata_File2.ipynb`

### Approach

A hybrid system combining a deterministic rule engine with a tuned XGBoost classifier in a weighted soft-vote ensemble. Uses deep feature engineering across 5 groups and includes the bonus ticket category prediction.

### Pipeline

```
Raw CSVs → Feature Engineering (5 groups) → Rule Engine + XGBoost → Ensemble → Submission
                                                                              ↓
                                                                    Bonus: Category Model
```

**Feature Engineering (50+ features across 5 groups)**

| Group | Features |
|-------|----------|
| 1 — Raw metadata | call duration, attempt number, mismatch count, completeness, turn/word counts, cyclical hour/day encoding, outcome one-hot |
| 2 — Structural anomalies | completed + low completeness, completed + short call, completed + no form, high mismatch, duration-per-answer, agent word ratio, questions gap |
| 3 — Q&A integrity | empty answer count, answered count, suspicious weight value, answered count discrepancy, possible fabrication flag |
| 4 — Transcript NLP | opt-out signal, wrong number signal, medical advice signal, escalation signal, agent question count, question-answer gap, word/char count, validation notes flags |
| 5 — Cross-field sanity | opted-out but classified completed, wrong number misclassified, weight answer not found in whisper transcript, composite anomaly score |

**Rule Engine (7 deterministic rules)**

| Rule | Condition |
|------|-----------|
| R1 | `outcome == completed` AND `response_completeness < 0.7` |
| R2 | `whisper_mismatch_count >= 3` |
| R3 | Opted-out signal but classified as `wrong_number` |
| R4 | Opted-out signal but classified as `completed` |
| R5 | Medical advice pattern detected in transcript |
| R6 | Suspicious weight value AND `outcome == completed` |
| R7 | Validation notes flag fabrication or skipping |

**XGBoost Classifier**
- `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`
- `scale_pos_weight` set to negative/positive class ratio
- Early stopping on validation AUCPR (patience=30)
- Threshold tuned on validation set (sweep 0.05 → 0.70)

**Weighted Ensemble**
```
combined_score = 0.35 × rule_flag + 0.65 × xgb_probability
predicted = combined_score >= best_threshold
```

**5-Fold Cross-Validation** on combined train+val data before final model retraining.

**Bonus — Ticket Category Prediction**
For flagged calls, a second XGBoost multiclass model predicts the ticket category:
`audio_issue | elevenlabs | openai | supabase | scheduler_aws | other`

### Output Files

| File | Description |
|------|-------------|
| `submission_File2.csv` | `call_id`, `predicted_ticket` for all 159 test calls |
| `submission_with_categories.csv` | Same + `ticket_category` for flagged calls |

### How to Run

**Google Colab (recommended)**
1. Open `carecaller_problem1_Team_Hakkuna Matata_File2.ipynb` in Colab
2. Upload `hackathon_train.csv`, `hackathon_val.csv`, `hackathon_test.csv`
3. Run all cells in order (Cell 1 installs dependencies)
4. Cell 16 auto-downloads `submission.csv`

**Local**
```bash
pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn shap
jupyter notebook "Submission 2/carecaller_problem1_Team_Hakkuna Matata_File2.ipynb"
```

---

## Key Design Decisions

**Why SMOTE in Submission 1 vs `scale_pos_weight` in Submission 2?**
SMOTE generates synthetic minority samples before training, which works well with smaller feature sets. `scale_pos_weight` is more efficient with large feature sets and tree-based models as it adjusts the loss function directly without inflating the dataset.

**Why a rule engine in Submission 2?**
Certain failure modes (e.g. completed call with <70% completeness, 3+ STT mismatches) are near-deterministic signals. Hard-coding these as rules gives the model a reliable prior and improves recall on edge cases that XGBoost might miss at low training frequencies.

**Why an ensemble?**
Rules have high precision on the patterns they cover but miss novel failure modes. XGBoost generalises across all features but can be overconfident. The 35/65 weighted combination balances both.

---

## Dependencies

```
pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn tqdm
```

Optional for CUDA acceleration (Submission 2 / `check.py`):
```
cupy-cuda12x cudf-cu12
```
