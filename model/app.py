"""
🚀 HAZARDOUS ASTEROID CLASSIFICATION - COMPLETE SOLUTION
Dataset: dataset.csv (24,000+ asteroids)
Ready for Hackathon Presentation!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("🚀 HAZARDOUS ASTEROID CLASSIFICATION SYSTEM")
print("="*80)
print("Dataset: dataset.csv")
print("Objective: Classify asteroids as Hazardous or Non-Hazardous")
print("="*80)

# ==================== STEP 1: LOAD DATA ====================
print("\n" + "="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

try:
    df = pd.read_csv('/Users/shubhamjha/Desktop/coding/ML/pythonforML/hackathon/dataset.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total Records: {len(df):,}")
    print(f"📋 Total Columns: {len(df.columns)}")
    print(f"\n🔍 First few rows:")
    print(df.head(3))
except FileNotFoundError:
    print("❌ Error: 'dataset.csv' not found!")
    print("📁 Please ensure 'dataset.csv' is in the same directory as this script.")
    exit()

# ==================== STEP 2: DATA EXPLORATION ====================
print("\n" + "="*80)
print("STEP 2: DATA EXPLORATION")
print("="*80)

print(f"\n📈 Dataset Shape: {df.shape}")
print(f"\n📋 Column Names ({len(df.columns)} columns):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n🔍 Data Types:")
print(df.dtypes.value_counts())

print(f"\n❓ Missing Values Summary:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("✅ No missing values detected!")

print(f"\n🎯 Target Variable Distribution:")
if 'Hazardous' in df.columns:
    hazard_counts = df['Hazardous'].value_counts()
    print(hazard_counts)
    print(f"\nPercentage Distribution:")
    for val, count in hazard_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {val}: {count:,} ({pct:.2f}%)")

# ==================== STEP 3: DATA PREPROCESSING ====================
print("\n" + "="*80)
print("STEP 3: DATA PREPROCESSING")
print("="*80)

# Create a clean copy
df_clean = df.copy()

# Convert Hazardous to binary (handle different formats)
if df_clean['Hazardous'].dtype == 'object' or df_clean['Hazardous'].dtype == 'bool':
    df_clean['Hazardous'] = df_clean['Hazardous'].map({
        'TRUE': 1, 'FALSE': 0, 
        True: 1, False: 0,
        'True': 1, 'False': 0,
        1: 1, 0: 0
    })
    print("✅ Target variable converted to binary (0/1)")

# Identify columns to exclude from features
exclude_cols = [
    'Name', 'Hazardous',
    'Epoch Date Close Approach', 
    'approach_year', 'approach_month', 'approach_day',
    'Epoch Osculation', 'Perihelion Time'
]

# Select numerical features
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numerical_cols if col not in exclude_cols]

print(f"\n✅ Selected {len(feature_cols)} numerical features for modeling")
print(f"\n📊 Feature Categories:")
print(f"  • Velocity features: {len([c for c in feature_cols if 'Velocity' in c or 'velocity' in c])}")
print(f"  • Distance features: {len([c for c in feature_cols if 'Dist' in c or 'distance' in c])}")
print(f"  • Orbital features: {len([c for c in feature_cols if any(x in c for x in ['Orbital', 'Semi', 'Asc', 'Perihelion', 'Aphelion'])])}")
print(f"  • Other features: {len(feature_cols) - len([c for c in feature_cols if any(x in c for x in ['Velocity', 'Dist', 'Orbital', 'Semi', 'Asc', 'Perihelion', 'Aphelion'])])}")

# Handle missing values
print(f"\n🔧 Handling missing values...")
X = df_clean[feature_cols].copy()

# Fill missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        
print(f"✅ All missing values handled using median imputation")

# Create target variable
y = df_clean['Hazardous']

print(f"\n✅ Final dataset prepared:")
print(f"  • Features (X): {X.shape}")
print(f"  • Target (y): {y.shape}")
print(f"  • Hazardous asteroids: {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"  • Non-hazardous asteroids: {(y==0).sum():,} ({(1-y.mean())*100:.2f}%)")

# ==================== STEP 4: FEATURE ENGINEERING ====================
print("\n" + "="*80)
print("STEP 4: FEATURE ENGINEERING")
print("="*80)

X_eng = X.copy()

# Create composite features
created_features = []

# 1. Energy proxy (velocity-based risk)
if 'Relative Velocity km per sec' in X_eng.columns:
    X_eng['Velocity_Risk_Score'] = X_eng['Relative Velocity km per sec'] / 1000
    created_features.append('Velocity_Risk_Score')

# 2. Proximity risk (distance-based)
if 'Miss Dist.(Astronomical)' in X_eng.columns:
    X_eng['Proximity_Risk'] = 1 / (X_eng['Miss Dist.(Astronomical)'] + 0.001)
    created_features.append('Proximity_Risk')

# 3. Combined risk score
if 'Relative Velocity km per sec' in X_eng.columns and 'Miss Dist.(Astronomical)' in X_eng.columns:
    X_eng['Combined_Risk'] = (X_eng['Relative Velocity km per sec'] / 10000) * X_eng['Proximity_Risk']
    created_features.append('Combined_Risk')

# 4. Orbital eccentricity risk
if 'Eccentricity' in X_eng.columns:
    X_eng['Eccentric_Risk'] = X_eng['Eccentricity'] ** 2
    created_features.append('Eccentric_Risk')

print(f"✅ Created {len(created_features)} engineered features:")
for feat in created_features:
    print(f"  • {feat}")

print(f"\n✅ Total features for modeling: {X_eng.shape[1]}")

# ==================== STEP 5: TRAIN-TEST SPLIT ====================
print("\n" + "="*80)
print("STEP 5: SPLITTING DATA & FEATURE SCALING")
print("="*80)

# Split the data (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✅ Data split completed:")
print(f"  • Training set: {X_train.shape[0]:,} samples ({(X_train.shape[0]/len(X_eng))*100:.1f}%)")
print(f"  • Test set: {X_test.shape[0]:,} samples ({(X_test.shape[0]/len(X_eng))*100:.1f}%)")

print(f"\n📊 Class distribution in training set:")
print(f"  • Non-Hazardous: {(y_train==0).sum():,}")
print(f"  • Hazardous: {(y_train==1).sum():,}")

print(f"\n📊 Class distribution in test set:")
print(f"  • Non-Hazardous: {(y_test==0).sum():,}")
print(f"  • Hazardous: {(y_test==1).sum():,}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled using StandardScaler")
print(f"  • Mean ≈ 0, Standard Deviation ≈ 1")

# ==================== STEP 6: MODEL TRAINING ====================
print("\n" + "="*80)
print("STEP 6: TRAINING MULTIPLE MODELS")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

trained_models = {}

print("\n🔄 Training models (this may take a few minutes)...\n")

for name, model in models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print("✅ Done!")

print(f"\n✅ All {len(models)} models trained successfully!")

# ==================== STEP 7: MODEL EVALUATION ====================
print("\n" + "="*80)
print("STEP 7: EVALUATING MODELS")
print("="*80)

results = []

for name, model in trained_models.items():
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print('='*70)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📈 Performance Metrics:")
    print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  • F1-Score:  {f1:.4f}")
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  • ROC-AUC:   {auc:.4f}")
    else:
        auc = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"  • True Negatives:  {cm[0,0]:,}")
    print(f"  • False Positives: {cm[0,1]:,}")
    print(f"  • False Negatives: {cm[1,0]:,} ⚠️ (Missed Threats)")
    print(f"  • True Positives:  {cm[1,1]:,}")
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': auc if auc else 0,
        'False_Negatives': cm[1,0]
    })

# Results DataFrame
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n" + "="*80)
print("📊 MODEL COMPARISON SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# Identify best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"  • Accuracy: {results_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"  • F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
print(f"  • False Negatives: {int(results_df.iloc[0]['False_Negatives'])}")

# ==================== STEP 8: FEATURE IMPORTANCE ====================
print("\n" + "="*80)
print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_eng.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🎯 Top 20 Most Important Features:")
    print("-" * 70)
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row.name+1:2d}. {row['Feature']:<40} {row['Importance']:.6f}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_results.csv', index=False)
    print(f"\n✅ Full feature importance saved to 'feature_importance_results.csv'")
else:
    print(f"\n⚠️ {best_model_name} does not support feature importance")

# ==================== STEP 9: VISUALIZATIONS ====================
print("\n" + "="*80)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*80)

# 1. Confusion Matrix
y_pred_best = best_model.predict(X_test_scaled)
cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_best, annot=True, fmt=',d', cmap='Blues', cbar=True,
            xticklabels=['Non-Hazardous', 'Hazardous'],
            yticklabels=['Non-Hazardous', 'Hazardous'],
            annot_kws={'size': 16})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved: 'confusion_matrix.png'")
plt.close()

# 2. ROC Curve
if hasattr(best_model, 'predict_proba'):
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
    auc_best = roc_auc_score(y_test, y_pred_proba_best)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='purple', lw=3, label=f'ROC Curve (AUC = {auc_best:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - Asteroid Classification', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    print("✅ ROC curve saved: 'roc_curve.png'")
    plt.close()

# 3. Model Comparison
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(results_df))
width = 0.2

bars1 = ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', color='#8b5cf6')
bars2 = ax.bar(x - 0.5*width, results_df['Precision'], width, label='Precision', color='#3b82f6')
bars3 = ax.bar(x + 0.5*width, results_df['Recall'], width, label='Recall', color='#10b981')
bars4 = ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score', color='#f59e0b')

ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Model comparison saved: 'model_comparison.png'")
plt.close()

# 4. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'], color='purple', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontsize=14, fontweight='bold')
    plt.ylabel('Features', fontsize=14, fontweight='bold')
    plt.title('Top 20 Feature Importances', fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance plot saved: 'feature_importance.png'")
    plt.close()

print("\n✅ All visualizations created successfully!")

# ==================== STEP 10: SAVE MODEL ====================
print("\n" + "="*80)
print("STEP 10: SAVING MODEL & ARTIFACTS")
print("="*80)

# Save the best model and scaler
joblib.dump(best_model, 'best_asteroid_classifier.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(X_eng.columns.tolist(), 'feature_names.pkl')

print(f"✅ Best model saved: 'best_asteroid_classifier.pkl'")
print(f"✅ Scaler saved: 'feature_scaler.pkl'")
print(f"✅ Feature names saved: 'feature_names.pkl'")

# Save results summary
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"✅ Results saved: 'model_comparison_results.csv'")

# ==================== STEP 11: FINAL SUMMARY ====================
print("\n" + "="*80)
print("🎉 ANALYSIS COMPLETE - HACKATHON READY!")
print("="*80)

print(f"\n📊 FINAL SUMMARY:")
print(f"  • Dataset Size: {len(df):,} asteroids")
print(f"  • Features Used: {X_eng.shape[1]}")
print(f"  • Best Model: {best_model_name}")
print(f"  • Test Accuracy: {results_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"  • ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
print(f"  • False Negatives: {int(results_df.iloc[0]['False_Negatives'])} (Missed Threats)")

print(f"\n📁 Generated Files:")
print(f"  1. confusion_matrix.png")
print(f"  2. roc_curve.png")
print(f"  3. model_comparison.png")
print(f"  4. feature_importance.png")
print(f"  5. best_asteroid_classifier.pkl")
print(f"  6. feature_scaler.pkl")
print(f"  7. feature_names.pkl")
print(f"  8. model_comparison_results.csv")
print(f"  9. feature_importance_results.csv")

print(f"\n🚀 READY FOR PRESENTATION!")
print("="*80)

# ==================== PREDICTION FUNCTION ====================
def predict_new_asteroid(features_dict):
    """
    Predict if a new asteroid is hazardous
    
    Parameters:
    features_dict: Dictionary with feature values
    
    Returns:
    prediction, probability
    """
    # Load models
    model = joblib.load('best_asteroid_classifier.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    # Create feature array
    features = np.array([[features_dict.get(name, 0) for name in feature_names]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    result = "🔴 HAZARDOUS" if prediction == 1 else "🟢 NON-HAZARDOUS"
    confidence = probability[prediction] * 100
    
    print(f"\n{'='*60}")
    print(f"🔮 PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"Classification: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nProbability Breakdown:")
    print(f"  Non-Hazardous: {probability[0]*100:.2f}%")
    print(f"  Hazardous:     {probability[1]*100:.2f}%")
    print(f"{'='*60}")
    
    return prediction, probability

print("\n💡 To make predictions on new asteroids, use:")
print("   predict_new_asteroid({'Relative Velocity km per sec': 30000, ...})")
print("\n✨ Good luck with your hackathon!")
