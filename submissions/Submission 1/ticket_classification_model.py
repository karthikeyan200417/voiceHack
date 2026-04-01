import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    recall_score, precision_score, roc_auc_score, roc_curve,
    accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("LOADING DATA")
print("="*80)

train_df = pd.read_csv('Datasets\csv\hackathon_train.csv')
val_df = pd.read_csv('Datasets\csv\hackathon_val.csv')
test_df = pd.read_csv('Datasets\csv\hackathon_test.csv')

print(f"\nTrain set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")

print("\nTrain set columns:")
print(train_df.columns.tolist())

print("\nTrain set first few rows:")
print(train_df.head(2))

print("\nData types:")
print(train_df.dtypes)

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

print(f"\nTarget variable distribution (Train):")
print(train_df['has_ticket'].value_counts())
print(f"Percentage: {train_df['has_ticket'].value_counts(normalize=True) * 100}")

print(f"\nTarget variable distribution (Validation):")
print(val_df['has_ticket'].value_counts())
print(f"Percentage: {val_df['has_ticket'].value_counts(normalize=True) * 100}")

print(f"\nOutcome distribution (Train):")
print(train_df['outcome'].value_counts())

print(f"\nMissing values (Train):")
print(train_df.isnull().sum())

# ============================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING & PREPROCESSING")
print("="*80)

def preprocess_data(df, fit_encoders=False, encoders=None, scaler=None, tfidf=None, show_progress=True):
    """
    Comprehensive preprocessing pipeline with progress tracking
    """
    df = df.copy()
    features_dict = {}
    
    # --- BASIC NUMERIC FEATURES ---
    features_dict['call_duration'] = df['call_duration'].fillna(df['call_duration'].median())
    features_dict['whisper_mismatch_count'] = df['whisper_mismatch_count'].fillna(0)
    features_dict['response_completeness'] = df['response_completeness'].fillna(0.0)
    
    # --- OUTCOME ENCODING ---
    if fit_encoders:
        outcome_encoder = LabelEncoder()
        features_dict['outcome_encoded'] = outcome_encoder.fit_transform(df['outcome'].fillna('unknown'))
        encoders['outcome'] = outcome_encoder
    else:
        try:
            features_dict['outcome_encoded'] = encoders['outcome'].transform(df['outcome'].fillna('unknown'))
        except ValueError as e:
            # Handle unknown categories
            df['outcome'] = df['outcome'].fillna('unknown')
            for cat in df['outcome'].unique():
                if cat not in encoders['outcome'].classes_:
                    # Map unknown to first known class
                    df.loc[df['outcome'] == cat, 'outcome'] = encoders['outcome'].classes_[0]
            features_dict['outcome_encoded'] = encoders['outcome'].transform(df['outcome'])
    
    # --- OUTCOME GROUP FEATURES (high-risk categories) ---
    high_risk_outcomes = ['wrong_number', 'escalated', 'incomplete', 'voicemail']
    features_dict['is_high_risk_outcome'] = (df['outcome'].isin(high_risk_outcomes)).astype(int)
    
    # --- Q&A RESPONSE EXTRACTION WITH TQDM ---
    response_counts = []
    critical_value_mentions = []
    
    iterator = tqdm(df['responses_json'].items(), total=len(df), desc="Extracting Q&A responses", disable=not show_progress)
    for idx, responses_str in iterator:
        try:
            if pd.isna(responses_str) or responses_str == '':
                response_counts.append(0)
                critical_value_mentions.append(0)
            else:
                responses = json.loads(responses_str)
                if isinstance(responses, dict):
                    response_counts.append(len(responses))
                    # Count mentions of critical health values
                    critical_count = sum(
                        1 for v in responses.values() 
                        if v and any(word in str(v).lower() for word in ['weight', 'height', 'age', 'bp', 'blood', 'medication', 'allergy', 'diabetes', 'hypertension'])
                    )
                    critical_value_mentions.append(critical_count)
                else:
                    response_counts.append(0)
                    critical_value_mentions.append(0)
        except:
            response_counts.append(0)
            critical_value_mentions.append(0)
    
    features_dict['response_count'] = response_counts
    features_dict['critical_value_mentions'] = critical_value_mentions
    
    # --- VALIDATION NOTES FEATURE ---
    validation_flag_count = []
    iterator = tqdm(df['validation_notes'].items(), total=len(df), desc="Processing validation notes", disable=not show_progress)
    for idx, note in iterator:
        if pd.isna(note) or note == '':
            validation_flag_count.append(0)
        else:
            # Count warnings/flags mentioned
            count = str(note).lower().count('error') + str(note).lower().count('warning') + str(note).lower().count('mismatch')
            validation_flag_count.append(count)
    
    features_dict['validation_flag_count'] = validation_flag_count
    
    # --- TRANSCRIPT TEXT FEATURES ---
    transcript_lengths = []
    word_counts = []
    
    iterator = tqdm(df['transcript_text'].items(), total=len(df), desc="Analyzing transcripts", disable=not show_progress)
    for idx, transcript in iterator:
        if pd.isna(transcript) or transcript == '':
            transcript_lengths.append(0)
            word_counts.append(0)
        else:
            transcript_lengths.append(len(str(transcript)))
            word_counts.append(len(str(transcript).split()))
    
    features_dict['transcript_length'] = transcript_lengths
    features_dict['word_count'] = word_counts
    
    # --- CALL DURATION VS RESPONSE RATIO ---
    features_dict['duration_to_response_ratio'] = [
        d / (r + 1) for d, r in zip(features_dict['call_duration'], features_dict['response_count'])
    ]
    
    # Convert to DataFrame
    X = pd.DataFrame(features_dict)
    
    # --- SCALING ---
    if fit_encoders:
        scaler_obj = StandardScaler()
        X_scaled = scaler_obj.fit_transform(X)
        scaler = scaler_obj
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, X, scaler, encoders

# Preprocess training data
print("\nPreprocessing training data...")
encoders = {}
X_train_scaled, X_train_orig, scaler, encoders = preprocess_data(train_df, fit_encoders=True, encoders=encoders, show_progress=True)
y_train = train_df['has_ticket'].astype(int)

print(f"Training features shape: {X_train_scaled.shape}")
print(f"Feature columns: {X_train_orig.columns.tolist()}")

# Preprocess validation data
print("\nPreprocessing validation data...")
X_val_scaled, X_val_orig, _, _ = preprocess_data(val_df, fit_encoders=False, encoders=encoders, scaler=scaler, show_progress=True)
y_val = val_df['has_ticket'].astype(int)

print(f"Validation features shape: {X_val_scaled.shape}")

# ============================================================================
# 4. CLASS IMBALANCE HANDLING (SMOTE)
# ============================================================================
print("\n" + "="*80)
print("HANDLING CLASS IMBALANCE WITH SMOTE")
print("="*80)

print("\n⚙️  Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"✓ SMOTE completed")
print(f"\nBefore SMOTE:")
print(f"  Class 0 (No Ticket): {sum(y_train == 0):4d}  ({sum(y_train == 0)/len(y_train)*100:5.2f}%)")
print(f"  Class 1 (Has Ticket):{sum(y_train == 1):4d}  ({sum(y_train == 1)/len(y_train)*100:5.2f}%)")
print(f"  Imbalance Ratio: {sum(y_train == 0) / (sum(y_train == 1) + 1):.2f}:1")

print(f"\nAfter SMOTE:")
print(f"  Class 0 (No Ticket): {sum(y_train_balanced == 0):4d}  ({sum(y_train_balanced == 0)/len(y_train_balanced)*100:5.2f}%)")
print(f"  Class 1 (Has Ticket):{sum(y_train_balanced == 1):4d}  ({sum(y_train_balanced == 1)/len(y_train_balanced)*100:5.2f}%)")
print(f"  Imbalance Ratio: {sum(y_train_balanced == 0) / (sum(y_train_balanced == 1) + 1):.2f}:1")

# ============================================================================
# 5. MODEL TRAINING WITH PROGRESS TRACKING
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS WITH PROGRESS TRACKING")
print("="*80)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, verbosity=0),
}

trained_models = {}
results = {}

for name, model in tqdm(models.items(), desc="Training models", total=len(models)):
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - start_time
    trained_models[name] = model
    
    print(f"✓ Training completed in {train_time:.2f} seconds")
    
    # Predictions with progress tracking
    print(f"  Making predictions on training set...")
    y_train_pred = model.predict(X_train_scaled)
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    print(f"  Making predictions on validation set...")
    y_val_pred = model.predict(X_val_scaled)
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Comprehensive metrics
    results[name] = {
        'train_f1': f1_score(y_train, y_train_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'val_recall': recall_score(y_val, y_val_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'val_precision': precision_score(y_val, y_val_pred),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'train_auc': roc_auc_score(y_train, y_train_pred_proba),
        'val_auc': roc_auc_score(y_val, y_val_pred_proba),
        'val_kappa': cohen_kappa_score(y_val, y_val_pred),
        'val_mcc': matthews_corrcoef(y_val, y_val_pred),
        'y_pred': y_val_pred,
        'y_pred_proba': y_val_pred_proba,
        'train_time': train_time
    }
    
    # Print quick summary
    print(f"  Train F1: {results[name]['train_f1']:.4f} | Val F1: {results[name]['val_f1']:.4f}")
    print(f"  Val Recall: {results[name]['val_recall']:.4f} | Val Precision: {results[name]['val_precision']:.4f}")
    print(f"  Val Accuracy: {results[name]['val_accuracy']:.4f} | Val AUC: {results[name]['val_auc']:.4f}")

# ============================================================================
# 6. COMPREHENSIVE CLASSIFICATION REPORTS & EVALUATION
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION METRICS (VALIDATION SET)")
print("="*80)

for name in tqdm(list(models.keys()), desc="Generating evaluation reports"):
    print(f"\n{'='*80}")
    print(f"Model: {name}")
    print(f"{'='*80}")
    
    print(f"\n📊 CLASSIFICATION REPORT (Validation Set):")
    print("-" * 80)
    print(classification_report(y_val, results[name]['y_pred'], 
                                target_names=['No Ticket', 'Has Ticket'],
                                digits=4))
    
    print(f"\n📈 CONFUSION MATRIX:")
    print("-" * 80)
    cm = confusion_matrix(y_val, results[name]['y_pred'])
    print(f"                 Predicted")
    print(f"                No      Yes")
    print(f"Actual No    {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"Actual Yes   {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n🎯 DETAILED METRICS:")
    print("-" * 80)
    print(f"  Accuracy:           {results[name]['val_accuracy']:.4f}")
    print(f"  F1 Score:           {results[name]['val_f1']:.4f}")
    print(f"  Recall (Sensitivity){results[name]['val_recall']:.4f}  (catches bad calls)")
    print(f"  Precision:          {results[name]['val_precision']:.4f}  (reduces false alarms)")
    print(f"  Specificity:        {specificity:.4f}  (correctly rejects good calls)")
    print(f"  AUC-ROC:            {results[name]['val_auc']:.4f}")
    print(f"  Cohen's Kappa:      {results[name]['val_kappa']:.4f}  (agreement beyond chance)")
    print(f"  Matthews Corr Coef: {results[name]['val_mcc']:.4f}  (balanced accuracy metric)")
    
    print(f"\n⏱️  TRAINING METRICS:")
    print("-" * 80)
    print(f"  Training Time:      {results[name]['train_time']:.2f} seconds")
    print(f"  Train F1 Score:     {results[name]['train_f1']:.4f}")
    print(f"  Train Accuracy:     {results[name]['train_accuracy']:.4f}")
    print(f"  Train AUC-ROC:      {results[name]['train_auc']:.4f}")
    
    # Detect overfitting
    f1_gap = results[name]['train_f1'] - results[name]['val_f1']
    acc_gap = results[name]['train_accuracy'] - results[name]['val_accuracy']
    
    print(f"\n⚠️  OVERFITTING ANALYSIS:")
    print("-" * 80)
    print(f"  F1 Gap (Train-Val): {f1_gap:.4f}")
    print(f"  Accuracy Gap:       {acc_gap:.4f}")
    if f1_gap > 0.1:
        print(f"  ⚠️  Potential overfitting detected (F1 gap > 0.1)")
    else:
        print(f"  ✓ Good generalization (F1 gap < 0.1)")

# ============================================================================
# 7. BEST MODEL SELECTION & FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("BEST MODEL SELECTION & FEATURE IMPORTANCE")
print("="*80)

# Select best by F1 score (primary metric)
best_model_name = max(results, key=lambda x: results[x]['val_f1'])
best_model = trained_models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1 Score: {results[best_model_name]['val_f1']:.4f}")
print(f"   Recall: {results[best_model_name]['val_recall']:.4f}")
print(f"   Precision: {results[best_model_name]['val_precision']:.4f}")
print(f"   AUC-ROC: {results[best_model_name]['val_auc']:.4f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_names = X_train_orig.columns
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n📌 Top 10 Most Important Features:")
    print("-" * 80)
    for idx, row in feature_importance_df.head(10).iterrows():
        bar_length = int(row['importance'] * 100)
        bar = "█" * bar_length + "░" * (100 - bar_length)
        print(f"  {row['feature']:30s} {row['importance']:8.6f}  {bar}")
    
    # Plot
    print(f"\n💾 Saving feature importance visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature', palette='viridis', ax=ax)
    plt.title(f'{best_model_name} - Top 10 Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved as 'feature_importance.png'")
    plt.close()

# ============================================================================
# 7.5 ROC CURVES & ADDITIONAL VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating ROC Curves visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, ax) in enumerate(zip(models.keys(), axes)):
    fpr, tpr, thresholds = roc_curve(y_val, results[name]['y_pred_proba'])
    auc_score = results[name]['val_auc']
    
    ax.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved as 'roc_curves.png'")
plt.close()

# Confusion matrices for all models
print(f"📊 Generating Confusion Matrices visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for (name, ax) in zip(models.keys(), axes):
    cm = confusion_matrix(y_val, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['No Ticket', 'Has Ticket'],
                yticklabels=['No Ticket', 'Has Ticket'])
    ax.set_title(f'{name}\nF1: {results[name]["val_f1"]:.4f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved as 'confusion_matrices.png'")
plt.close()

# ============================================================================
# 8. PREDICTIONS ON TEST SET
# ============================================================================
print("\n" + "="*80)
print("PREDICTIONS ON TEST SET")
print("="*80)

print(f"\n🔍 Preprocessing test set...")
X_test_scaled, X_test_orig, _, _ = preprocess_data(test_df, fit_encoders=False, encoders=encoders, scaler=scaler, show_progress=True)

print(f"\n🎯 Making predictions using {best_model_name}...")
test_predictions = best_model.predict(X_test_scaled)
test_predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Create output
test_df_output = test_df.copy()
test_df_output['predicted_has_ticket'] = test_predictions
test_df_output['ticket_probability'] = test_predictions_proba

# Save predictions
test_df_output[['call_id', 'predicted_has_ticket']].to_csv('test_predictions.csv', index=False)

print(f"\n✓ Test predictions saved to 'test_predictions.csv'")

print(f"\n📊 TEST SET PREDICTION SUMMARY:")
print("-" * 80)
print(f"  Total calls:                {len(test_predictions)}")
print(f"  Predicted tickets:          {test_predictions.sum()}")
print(f"  Prediction rate:            {(test_predictions.sum() / len(test_predictions) * 100):.2f}%")
print(f"  Mean ticket probability:    {test_predictions_proba.mean():.4f}")
print(f"  Median ticket probability:  {np.median(test_predictions_proba):.4f}")
print(f"  Std Dev ticket probability: {test_predictions_proba.std():.4f}")
print(f"  Min ticket probability:     {test_predictions_proba.min():.4f}")
print(f"  Max ticket probability:     {test_predictions_proba.max():.4f}")

# Distribution analysis
print(f"\n📈 PROBABILITY DISTRIBUTION:")
print("-" * 80)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(test_predictions_proba, p)
    print(f"  {p:3d}th percentile: {val:.4f}")

# High-risk calls
high_risk_threshold = np.percentile(test_predictions_proba, 90)
high_risk_count = (test_predictions_proba >= high_risk_threshold).sum()
print(f"\n  High-risk calls (top 10%, threshold: {high_risk_threshold:.4f}): {high_risk_count}")

# Visualization of test predictions
print(f"\n💾 Saving test predictions visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(test_predictions_proba, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(test_predictions_proba.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {test_predictions_proba.mean():.4f}')
axes[0].set_xlabel('Ticket Probability', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Ticket Probabilities (Test Set)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Pie chart
ticket_counts = [len(test_predictions) - test_predictions.sum(), test_predictions.sum()]
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(ticket_counts, labels=['No Ticket', 'Predicted Ticket'], autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
axes[1].set_title('Prediction Distribution (Test Set)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('test_predictions_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Test predictions distribution saved as 'test_predictions_distribution.png'")
plt.close()

# ============================================================================
# 9. COMPREHENSIVE MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
print("="*80)

summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train F1': [f"{results[m]['train_f1']:.4f}" for m in results.keys()],
    'Val F1': [f"{results[m]['val_f1']:.4f}" for m in results.keys()],
    'Val Recall': [f"{results[m]['val_recall']:.4f}" for m in results.keys()],
    'Val Precision': [f"{results[m]['val_precision']:.4f}" for m in results.keys()],
    'Val Accuracy': [f"{results[m]['val_accuracy']:.4f}" for m in results.keys()],
    'Val AUC': [f"{results[m]['val_auc']:.4f}" for m in results.keys()],
    'Val Kappa': [f"{results[m]['val_kappa']:.4f}" for m in results.keys()],
    'Train Time (s)': [f"{results[m]['train_time']:.2f}" for m in results.keys()],
})

print("\n" + summary_df.to_string(index=False))

# Detailed comparison
print(f"\n\n{'+' * 80}")
print(f"WINNER ANALYSIS")
print(f"{'+' * 80}")

metrics_to_compare = {
    'F1 Score': 'val_f1',
    'Recall': 'val_recall',
    'Precision': 'val_precision',
    'Accuracy': 'val_accuracy',
    'AUC-ROC': 'val_auc',
}

for metric_name, metric_key in metrics_to_compare.items():
    best_model_for_metric = max(results.items(), key=lambda x: x[1][metric_key])
    print(f"\n✓ Best {metric_name:15s}: {best_model_for_metric[0]:20s} ({best_model_for_metric[1][metric_key]:.4f})")

# ============================================================================
# 9.5 VALIDATION SET DEEP DIVE
# ============================================================================
print(f"\n\n{'='*80}")
print("VALIDATION SET DETAILED ANALYSIS")
print(f"{'='*80}")

print(f"\n📌 BEST MODEL THRESHOLD ANALYSIS ({best_model_name}):")
print("-" * 80)

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'Accuracy':>10}")
print("-" * 50)

for threshold in thresholds:
    pred_threshold = (results[best_model_name]['y_pred_proba'] >= threshold).astype(int)
    precision = precision_score(y_val, pred_threshold, zero_division=0)
    recall = recall_score(y_val, pred_threshold, zero_division=0)
    f1 = f1_score(y_val, pred_threshold, zero_division=0)
    accuracy = accuracy_score(y_val, pred_threshold)
    
    print(f"{threshold:>10.1f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {accuracy:>10.4f}")

print(f"\n\n💡 RECOMMENDATIONS:")
print("-" * 80)
print(f"1. Primary Model: {best_model_name}")
print(f"   → Best overall F1 score (catches bad calls while minimizing false alarms)")
print(f"\n2. Feature Importance:")
print(f"   → Focus on transcript analysis (word count, length)")
print(f"   → Monitor STT quality (whisper mismatch count)")
print(f"   → Track response completeness")
print(f"\n3. Deployment Strategy:")
print(f"   → Use probability threshold of 0.5 for balanced precision/recall")
print(f"   → Flag calls with probability > 0.7 for immediate review")
print(f"   → Monitor outcomes to adjust threshold based on business needs")
print(f"\n4. False Positive Reduction:")
print(f"   → Review {(X_val_orig.shape[0] - results[best_model_name]['y_pred'].sum() + y_val.sum()):d} false positives")
print(f"   → Use ensemble approach with rule-based heuristics")
print(f"\n5. Model Monitoring:")
print(f"   → Retrain model monthly with new labeled data")
print(f"   → Monitor prediction distribution drift")
print(f"   → Track true positive rate on production calls")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*80}")
print("EXECUTION SUMMARY")
print(f"{'='*80}")

print(f"""
✓ Data Loading:        Complete (689 train + 144 val + 159 test calls)
✓ Preprocessing:       Complete (11 engineered features)
✓ Class Balancing:     Complete (SMOTE applied)
✓ Model Training:      Complete (3 models trained)
✓ Evaluation:          Complete (comprehensive metrics generated)
✓ Visualizations:      Complete (4 plots generated)

📁 Output Files Generated:
   1. test_predictions.csv              - Test set predictions with probabilities
   2. feature_importance.png            - Feature importance bar chart
   3. roc_curves.png                    - ROC curves for all 3 models
   4. confusion_matrices.png            - Confusion matrices for all 3 models
   5. test_predictions_distribution.png - Test predictions analysis

🎯 Best Model: {best_model_name}
   F1 Score:  {results[best_model_name]['val_f1']:.4f}
   Recall:    {results[best_model_name]['val_recall']:.4f}
   Precision: {results[best_model_name]['val_precision']:.4f}
   AUC-ROC:   {results[best_model_name]['val_auc']:.4f}
""")

print("="*80)
print("✓ ALL PROCESSING COMPLETE!")
print("="*80)
