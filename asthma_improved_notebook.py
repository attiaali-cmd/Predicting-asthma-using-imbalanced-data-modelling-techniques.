"""
Integration script for improved asthma model resampling
This complements your existing asthma_brffs_2029.ipynb notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. ENHANCED RESAMPLING METHODS COMPARISON
# ============================================================================

def advanced_resampling_evaluation(df, target='Current_Asthma', scale_features=True):
    """
    Enhanced version of your evaluate_models_resampling function
    Includes scaling, advanced samplers, and more metrics
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Advanced resampling strategies
    samplers = {
        "No Resampling": None,
        "Random Over-Sampling": RandomOverSampler(random_state=42),
        "Random Under-Sampling": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(k_neighbors=5, random_state=42),
        "Borderline SMOTE": BorderlineSMOTE(k_neighbors=5, random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTE + Tomek": SMOTETomek(random_state=42),
        "SMOTE + ENN": SMOTEENN(random_state=42),
        "Tomek Links": TomekLinks(),
        "NearMiss": NearMiss(version=2, n_neighbors=3)
    }
    
    # Enhanced models with better hyperparameters
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=10, 
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel='rbf', C=1.0, probability=True, 
            class_weight='balanced', random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights='distance')
    }
    
    results = []
    
    for sampler_name, sampler in samplers.items():
        # Apply resampling
        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        
        for model_name, model in models.items():
            try:
                model.fit(X_res, y_res)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                metrics = {
                    "Resampling": sampler_name,
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
                }
                results.append(metrics)
            except Exception as e:
                print(f"Error with {model_name} + {sampler_name}: {str(e)}")
    
    return pd.DataFrame(results).round(4)


# ============================================================================
# 2. CLASS WEIGHT BALANCING (Alternative to resampling)
# ============================================================================

def evaluate_with_class_weights(df, target='Current_Asthma', scale_features=True):
    """
    Evaluate models using class_weight='balanced' parameter
    This is often as effective as resampling and computationally faster
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Calculate class weights manually for reference
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    print(f"\nClass weights: {dict(zip(classes, weights))}")
    
    models_balanced = {
        "Logistic Regression (Balanced)": LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=42
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "SVM (Balanced)": SVC(
            kernel='rbf', class_weight='balanced', probability=True, random_state=42
        )
    }
    
    results = []
    for model_name, model in models_balanced.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        }
        results.append(metrics)
    
    return pd.DataFrame(results).round(4)


# ============================================================================
# 3. CROSS-VALIDATION WITH RESAMPLING
# ============================================================================

def cross_validate_with_resampling(df, target='Current_Asthma', cv=5):
    """
    Perform stratified cross-validation with best resampling method
    More robust than single train/test split
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = []
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    }
    
    for model_name, model in models.items():
        # Use SMOTE resampling in cross-validation
        smote = SMOTE(random_state=42)
        
        scores_f1 = []
        scores_roc = []
        scores_recall = []
        
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train_cv, X_test_cv = X_scaled[train_idx], X_scaled[test_idx]
            y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
            
            # Resample training data
            X_train_res, y_train_res = smote.fit_resample(X_train_cv, y_train_cv)
            
            # Train and evaluate
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train_res, y_train_res)
            
            y_pred = model_clone.predict(X_test_cv)
            y_prob = model_clone.predict_proba(X_test_cv)[:, 1] if hasattr(model_clone, "predict_proba") else None
            
            scores_f1.append(f1_score(y_test_cv, y_pred, zero_division=0))
            scores_recall.append(recall_score(y_test_cv, y_pred, zero_division=0))
            if y_prob is not None:
                scores_roc.append(roc_auc_score(y_test_cv, y_prob))
        
        results.append({
            "Model": model_name,
            "F1_Mean": np.mean(scores_f1),
            "F1_Std": np.std(scores_f1),
            "Recall_Mean": np.mean(scores_recall),
            "Recall_Std": np.std(scores_recall),
            "ROC_AUC_Mean": np.mean(scores_roc),
            "ROC_AUC_Std": np.std(scores_roc)
        })
    
    return pd.DataFrame(results).round(4)


# ============================================================================
# 4. THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_decision_threshold(df, target='Current_Asthma', model_type='random_forest'):
    """
    Optimize decision threshold to maximize F1 or Recall
    Default threshold is 0.5, but may not be optimal for imbalanced data
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=2000, class_weight='balanced')
    
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\nThreshold Optimization Results ({model_type}):")
    print(f"  Default threshold (0.5) F1: {f1_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0):.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Optimal F1: {f1_scores[optimal_idx]:.4f}")
    print(f"  Recall at optimal: {recall[optimal_idx]:.4f}")
    print(f"  Precision at optimal: {precision[optimal_idx]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1])
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    plt.axvline(0.5, color='gray', linestyle=':', label='Default: 0.5')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold


# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(df, target='Current_Asthma', top_n=15):
    """
    Extract and visualize feature importance from Random Forest
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    feature_names = X.columns
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pd.DataFrame({'Feature': top_features, 'Importance': top_importances})


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

def run_improved_analysis(brfss_2019_michigan):
    """
    Main function to run all improved analyses
    
    Usage:
        from asthma_improved_notebook import run_improved_analysis
        run_improved_analysis(brfss_2019_michigan)
    """
    
    print("="*80)
    print("IMPROVED ASTHMA MODEL ANALYSIS")
    print("="*80)
    
    # 1. Advanced resampling comparison
    print("\n1. ADVANCED RESAMPLING COMPARISON (with scaling and advanced methods)")
    print("-" * 80)
    results_advanced = advanced_resampling_evaluation(brfss_2019_michigan, scale_features=True)
    print(results_advanced)
    print("\nTop 10 configurations by F1:")
    print(results_advanced.nlargest(10, 'F1'))
    
    # 2. Class weight balancing
    print("\n2. CLASS WEIGHT BALANCING (Faster alternative)")
    print("-" * 80)
    results_weights = evaluate_with_class_weights(brfss_2019_michigan)
    print(results_weights)
    
    # 3. Cross-validation
    print("\n3. CROSS-VALIDATION WITH RESAMPLING (5-Fold)")
    print("-" * 80)
    results_cv = cross_validate_with_resampling(brfss_2019_michigan)
    print(results_cv)
    
    # 4. Threshold optimization
    print("\n4. DECISION THRESHOLD OPTIMIZATION")
    print("-" * 80)
    optimal_threshold = optimize_decision_threshold(brfss_2019_michigan)
    
    # 5. Feature importance
    print("\n5. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)
    feature_importance_df = analyze_feature_importance(brfss_2019_michigan, top_n=15)
    print(feature_importance_df)
    
    # Save all results
    results_advanced.to_csv('asthma_advanced_resampling.csv', index=False)
    results_weights.to_csv('asthma_class_weights.csv', index=False)
    results_cv.to_csv('asthma_cross_validation.csv', index=False)
    feature_importance_df.to_csv('asthma_feature_importance.csv', index=False)
    
    print("\n" + "="*80)
    print("Analysis complete! Results saved to CSV files.")
    print("="*80)
    
    return {
        'advanced_resampling': results_advanced,
        'class_weights': results_weights,
        'cross_validation': results_cv,
        'feature_importance': feature_importance_df
    }


if __name__ == "__main__":
    print("Import this module and use: run_improved_analysis(your_dataframe)")
