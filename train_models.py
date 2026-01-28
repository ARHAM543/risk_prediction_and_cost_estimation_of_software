"""
PRMS - Model Training Pipeline
Trains Random Forest models for risk classification and schedule deviation prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, r2_score, mean_absolute_error
import joblib
import os


def load_and_preprocess_data(filepath: str) -> tuple:
    """Load and preprocess the project risk dataset."""
    df = pd.read_csv(filepath)
    
    # Feature columns for prediction
    feature_cols = [
        'Team_Size', 
        'Budget_Allocated', 
        'Requirement_Changes', 
        'Code_Churn', 
        'Team_Experience_Score'
    ]
    
    X = df[feature_cols].copy()
    
    # Target variables
    y_risk = df['Risk_Level'].copy()  # Classification target
    y_schedule = df['Schedule_Deviation_Days'].copy()  # Regression target
    
    return X, y_risk, y_schedule, feature_cols


def train_risk_classifier(X_train, y_train) -> tuple:
    """Train Random Forest Classifier for Risk Level prediction."""
    # Encode risk labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_encoded)
    
    return clf, label_encoder


def train_schedule_regressor(X_train, y_train) -> RandomForestRegressor:
    """Train Random Forest Regressor for Schedule Deviation prediction."""
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    reg.fit(X_train, y_train)
    
    return reg


def save_models(clf, label_encoder, reg, scaler, feature_cols, model_dir: str = "models"):
    """Save trained models and artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(clf, os.path.join(model_dir, "risk_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    joblib.dump(reg, os.path.join(model_dir, "schedule_regressor.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.joblib"))
    
    print(f"[OK] Models saved to '{model_dir}/' directory")


def evaluate_models(clf, reg, label_encoder, scaler, X_test, y_risk_test, y_schedule_test):
    """Evaluate model performance on test data."""
    X_test_scaled = scaler.transform(X_test)
    
    # Classification metrics
    y_risk_encoded = label_encoder.transform(y_risk_test)
    y_risk_pred = clf.predict(X_test_scaled)
    
    print("\n" + "="*50)
    print("RISK CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(
        y_risk_encoded, 
        y_risk_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Regression metrics
    y_schedule_pred = reg.predict(X_test_scaled)
    r2 = r2_score(y_schedule_test, y_schedule_pred)
    mae = mean_absolute_error(y_schedule_test, y_schedule_pred)
    
    print("\n" + "="*50)
    print("SCHEDULE DEVIATION REGRESSION REPORT")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f} days")
    
    return {
        'classification_report': classification_report(y_risk_encoded, y_risk_pred, output_dict=True),
        'r2_score': r2,
        'mae': mae
    }


def main():
    """Main training pipeline."""
    print("PRMS Model Training Pipeline")
    print("="*50)
    
    # Load data
    data_path = "software_project_risk_data.csv"
    print(f"Loading data from '{data_path}'...")
    X, y_risk, y_schedule, feature_cols = load_and_preprocess_data(data_path)
    print(f"   Loaded {len(X)} samples with {len(feature_cols)} features")
    
    # Split data
    X_train, X_test, y_risk_train, y_risk_test, y_schedule_train, y_schedule_test = train_test_split(
        X, y_risk, y_schedule, 
        test_size=0.2, 
        random_state=42,
        stratify=y_risk
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models
    print("\nTraining Risk Classifier...")
    clf, label_encoder = train_risk_classifier(X_train_scaled, y_risk_train)
    
    print("Training Schedule Regressor...")
    reg = train_schedule_regressor(X_train_scaled, y_schedule_train)
    
    # Evaluate models
    metrics = evaluate_models(clf, reg, label_encoder, scaler, X_test, y_risk_test, y_schedule_test)
    
    # Save models
    save_models(clf, label_encoder, reg, scaler, feature_cols)
    
    print("\n[OK] Training complete!")
    return metrics


if __name__ == "__main__":
    main()
