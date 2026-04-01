# Submission 2 — Call Quality Auto-Flagger
### CareCaller Hackathon 2026 | Team: Hakkuna Matata

A hybrid system combining a deterministic rule engine with a tuned XGBoost classifier in a weighted soft-vote ensemble. Includes deep feature engineering across 5 groups and a bonus ticket category predictor.

---

## 📋 Overview

| Property | Detail |
|----------|--------|
| Environment | Google Colab notebook |
| Language | Python 3 |
| NLP Approach | Regex pattern matching |
| Class Imbalance | `scale_pos_weight` |
| Models | Rule Engine + XGBoost ensemble |
| Feature Count | 50+ |
| Bonus Category | ✅ |

---

## 🗂️ Files

```
Submission 2/
├── carecaller_problem1_Team_Hakkuna Matata_File2.ipynb   # Main notebook
├── submission_File2.csv                                   # Output: binary predictions
└── README.md
```

> `submission_with_categories.csv` is generated on run (bonus task output).

---

## ⚙️ Setup & Run

### Google Colab *(recommended)*

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload `hackathon_train.csv`, `hackathon_val.csv`, `hackathon_test.csv`
3. Run all cells top to bottom — Cell 1 installs all dependencies automatically
4. Cell 16 auto-downloads `submission.csv` to your machine

### Local

```bash
pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib seaborn shap
jupyter notebook "Submission 2/carecaller_problem1_Team_Hakkuna Matata_File2.ipynb"
```

> When running locally, place the three CSV files in the same directory as the notebook.

---

## 🔧 Pipeline

```
Raw CSVs
   │
   ▼
Feature Engineering (5 groups, 50+ features)
   │
   ├──────────────────────┐
   ▼                      ▼
Rule Engine           XGBoost Classifier
(7 hard rules)        (scale_pos_weight + early stopping)
   │                      │
   └──────────┬───────────┘
              ▼
     Weighted Soft-Vote Ensemble
     (0.35 × rules + 0.65 × xgb_proba)
              │
              ▼
     Threshold Tuning on Val Set
              │
              ▼
     5-Fold Cross-Validation
              │
              ▼
     Final Model (retrained on Train + Val)
              │
         ┌────┴────┐
         ▼         ▼
   submission   Bonus: Category
     _File2.csv   Prediction
```

---

## 🧠 Feature Engineering

### Group 1 — Raw Metadata
Call duration, attempt number, whisper mismatch count, response completeness, question/answer counts, turn counts, word counts, form submitted, hour of day.

Cyclical encoding applied to hour and day of week:
```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Outcome one-hot encoded across 7 classes: `completed`, `incomplete`, `opted_out`, `scheduled`, `escalated`, `wrong_number`, `voicemail`.

### Group 2 — Structural Anomalies

| Feature | Condition |
|---------|-----------|
| `completed_low_completeness` | outcome=completed AND completeness < 0.8 |
| `completed_short_call` | outcome=completed AND duration < 120s |
| `completed_no_form` | outcome=completed AND form not submitted |
| `high_mismatch_completed` | mismatch ≥ 2 AND outcome=completed |
| `any_high_mismatch` | mismatch ≥ 3 |
| `duration_per_answer` | call duration ÷ answered count (pace check) |
| `agent_word_ratio` | agent words ÷ total words (agent-dominated = possible skipping) |
| `questions_gap` | question_count − answered_count |

### Group 3 — Q&A Integrity
Parsed directly from `responses_json`:

| Feature | Description |
|---------|-------------|
| `qa_empty_answer_count` | Questions with blank answers |
| `qa_answered_count` | Non-empty answers |
| `qa_suspicious_weight` | Weight value outside plausible range (< 80 or > 550 lbs) |
| `answered_count_discrepancy` | Metadata answered_count vs actual non-empty responses |
| `possible_fabrication` | High answer count on a very short call (< 100s) |

### Group 4 — Transcript NLP

Regex pattern matching across four signal types:

| Feature | Patterns Detected |
|---------|-------------------|
| `transcript_opt_out_signal` | "not interested", "don't want", "take me off", "no thanks" |
| `transcript_wrong_number_signal` | "wrong number", "no one here", "wrong person" |
| `transcript_medical_advice` | "you should take", "the dosage is", "try reducing" |
| `transcript_escalation` | "chest pain", "difficult to breathe", "call 911", "severe" |
| `agent_question_count` | Agent turns containing `?` |
| `question_answer_gap` | Agent questions asked − answers recorded |

Validation notes signals: `val_notes_has_mismatch`, `val_notes_has_fabricated`, `val_notes_has_skipped`.

### Group 5 — Cross-Field Sanity Checks

| Feature | Description |
|---------|-------------|
| `opted_out_classified_completed` | Opt-out signal in transcript but outcome=completed |
| `wrong_number_misclassified` | Opt-out signal but outcome=wrong_number |
| `weight_answer_not_in_whisper` | Recorded weight answer not found in raw STT transcript |
| `anomaly_score` | Sum of all binary flags above |

---

## 📏 Rule Engine

7 deterministic rules fire before the ML model. Any rule match sets the call as flagged.

| Rule | Condition | Label |
|------|-----------|-------|
| R1 | `outcome=completed` AND `response_completeness < 0.7` | `skipped_questions` |
| R2 | `whisper_mismatch_count >= 3` | `stt_mismatch` |
| R3 | Opt-out signal AND `outcome=wrong_number` | `wrong_number_misclassify` |
| R4 | Opt-out signal AND `outcome=completed` | `outcome_mismatch` |
| R5 | Medical advice pattern in transcript | `medical_advice` |
| R6 | Suspicious weight AND `outcome=completed` | `suspicious_weight_stt` |
| R7 | Validation notes flag fabrication or skipping | `validation_flag` |

---

## 🤖 XGBoost Classifier

```python
XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=neg_count / pos_count,
    eval_metric='aucpr',
    early_stopping_rounds=30
)
```

`scale_pos_weight` is computed per split as `negative_count / positive_count` to handle class imbalance without inflating the dataset.

---

## 🔀 Weighted Ensemble

```
combined_score = 0.35 × rule_flag + 0.65 × xgb_probability
predicted_ticket = combined_score >= best_threshold
```

`best_threshold` is found by sweeping 0.05 → 0.70 on the validation set and selecting the value that maximises F1.

**Why an ensemble?**
Rules have high precision on known failure patterns but miss novel ones. XGBoost generalises across all features but can be overconfident on rare edge cases. The weighted combination balances both.

---

## ✅ Cross-Validation

5-fold stratified cross-validation is run on the combined train+val set before the final model is retrained on all labeled data.

```
Mean F1     reported across 5 folds
Mean Recall reported across 5 folds
```

---

## 🎁 Bonus — Ticket Category Prediction

For calls predicted as flagged, a second XGBoost multiclass model predicts the ticket category:

| Category | Description |
|----------|-------------|
| `audio_issue` | Audio / STT quality problem |
| `elevenlabs` | Voice agent issue |
| `openai` | AI processing issue |
| `supabase` | Database issue |
| `scheduler_aws` | Scheduling issue |
| `other` | Uncategorised ticket |

Trained only on labeled ticket calls from train+val. Output is appended to `submission_with_categories.csv`.

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `submission_File2.csv` | `call_id`, `predicted_ticket` for all 159 test calls |
| `submission_with_categories.csv` | Same + `ticket_category` for flagged calls |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
```
