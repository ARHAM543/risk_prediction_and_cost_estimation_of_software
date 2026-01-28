"""
PRMS - Model Validation Script
Evaluates trained models on test data with detailed metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    r2_score, 
    mean_absolute_error,
    mean_squared_error
)
import joblib
import os


def load_models(model_dir: str = "models"):
    """Load all trained models and artifacts."""
    clf = joblib.load(os.path.join(model_dir, "risk_classifier.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    reg = joblib.load(os.path.join(model_dir, "schedule_regressor.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
    
    return clf, label_encoder, reg, scaler, feature_cols


def validate_classifier(clf, label_encoder, X_test, y_test):
    """Validate the risk classifier with detailed metrics."""
    print("\n" + "="*60)
    print("RISK CLASSIFIER VALIDATION")
    print("="*60)
    
    # Encode labels
    y_encoded = label_encoder.transform(y_test)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))
    
    # F1 Scores
    f1_weighted = f1_score(y_encoded, y_pred, average='weighted')
    f1_macro = f1_score(y_encoded, y_pred, average='macro')
    
    print(f"\nF1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro):    {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_encoded, y_pred)
    print("\nConfusion Matrix:")
    print(f"Classes: {label_encoder.classes_}")
    print(cm)
    
    # Per-class accuracy
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(label_encoder.classes_):
        cls_total = (y_encoded == i).sum()
        cls_correct = cm[i, i]
        cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
        print(f"   {cls}: {cls_accuracy:.2%} ({cls_correct}/{cls_total})")
    
    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }


def validate_regressor(reg, X_test, y_test):
    """Validate the schedule regressor with detailed metrics."""
    print("\n" + "="*60)
    print("SCHEDULE REGRESSOR VALIDATION")
    print("="*60)
    
    # Predictions
    y_pred = reg.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nR2 Score:              {r2:.4f}")
    print(f"Mean Absolute Error:   {mae:.4f} days")
    print(f"Root Mean Sq Error:    {rmse:.4f} days")
    
    # Prediction distribution
    print("\nPrediction Statistics:")
    print(f"   Actual Mean:     {y_test.mean():.2f} days")
    print(f"   Predicted Mean:  {y_pred.mean():.2f} days")
    print(f"   Actual Std:      {y_test.std():.2f} days")
    print(f"   Predicted Std:   {y_pred.std():.2f} days")
    
    # Residual analysis
    residuals = y_test - y_pred
    print("\nResidual Analysis:")
    print(f"   Mean Residual:   {residuals.mean():.4f}")
    print(f"   Std Residual:    {residuals.std():.4f}")
    print(f"   Max Overest:     {residuals.min():.2f} days")
    print(f"   Max Underest:    {residuals.max():.2f} days")
    
    return {
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse
    }


def run_cross_validation(clf, reg, X, y_risk, y_schedule, label_encoder, cv=5):
    """Run cross-validation for both models."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    # Encode labels
    y_risk_encoded = label_encoder.fit_transform(y_risk)
    
    # Classifier CV
    clf_scores = cross_val_score(clf, X, y_risk_encoded, cv=cv, scoring='f1_weighted')
    print(f"\nClassifier F1 (weighted) - {cv}-Fold CV:")
    print(f"   Mean: {clf_scores.mean():.4f}")
    print(f"   Std:  {clf_scores.std():.4f}")
    print(f"   Scores: {[f'{s:.3f}' for s in clf_scores]}")
    
    # Regressor CV
    reg_scores = cross_val_score(reg, X, y_schedule, cv=cv, scoring='r2')
    print(f"\nRegressor R2 - {cv}-Fold CV:")
    print(f"   Mean: {reg_scores.mean():.4f}")
    print(f"   Std:  {reg_scores.std():.4f}")
    print(f"   Scores: {[f'{s:.3f}' for s in reg_scores]}")
    
    return {
        'clf_cv_mean': clf_scores.mean(),
        'clf_cv_std': clf_scores.std(),
        'reg_cv_mean': reg_scores.mean(),
        'reg_cv_std': reg_scores.std()
    }


def main():
    """Main validation pipeline."""
    print("PRMS Model Validation Pipeline")
    print("="*60)
    
    # Load data
    data_path = "software_project_risk_data.csv"
    print(f"\nLoading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # Load models
    print("Loading trained models...")
    clf, label_encoder, reg, scaler, feature_cols = load_models()
    
    # Prepare data
    X = df[feature_cols].copy()
    y_risk = df['Risk_Level'].copy()
    y_schedule = df['Schedule_Deviation_Days'].copy()
    
    # Split data (same seed as training for consistent test set)
    X_train, X_test, y_risk_train, y_risk_test, y_schedule_train, y_schedule_test = train_test_split(
        X, y_risk, y_schedule,
        test_size=0.2,
        random_state=42,
        stratify=y_risk
    )
    
    print(f"   Test set size: {len(X_test)} samples")
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)
    
    # Validate models
    clf_metrics = validate_classifier(clf, label_encoder, X_test_scaled, y_risk_test)
    reg_metrics = validate_regressor(reg, X_test_scaled, y_schedule_test)
    
    # Cross-validation
    cv_metrics = run_cross_validation(clf, reg, X_scaled, y_risk, y_schedule, label_encoder)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"\nRisk Classifier:")
    print(f"   F1 Score (Weighted): {clf_metrics['f1_weighted']:.4f}")
    print(f"   CV Mean F1:          {cv_metrics['clf_cv_mean']:.4f} +/- {cv_metrics['clf_cv_std']:.4f}")
    
    print(f"\nSchedule Regressor:")
    print(f"   R2 Score:            {reg_metrics['r2_score']:.4f}")
    print(f"   CV Mean R2:          {cv_metrics['reg_cv_mean']:.4f} +/- {cv_metrics['reg_cv_std']:.4f}")
    print(f"   MAE:                 {reg_metrics['mae']:.2f} days")
    
    # Quality thresholds
    print("\n" + "="*60)
    print("QUALITY THRESHOLDS CHECK")
    print("="*60)
    
    f1_pass = clf_metrics['f1_weighted'] >= 0.75
    r2_pass = reg_metrics['r2_score'] >= 0.60
    
    print(f"   Classifier F1 >= 0.75: {'PASS' if f1_pass else 'FAIL'}")
    print(f"   Regressor R2 >= 0.60:  {'PASS' if r2_pass else 'FAIL'}")
    
    if f1_pass and r2_pass:
        print("\n[OK] All quality thresholds met!")
    else:
        print("\n[WARNING] Some thresholds not met. Consider retraining with tuned hyperparameters.")
    
    return {
        'classifier': clf_metrics,
        'regressor': reg_metrics,
        'cross_validation': cv_metrics
    }


if __name__ == "__main__":
    main()
