"""
PRMS - Model Training Tests
Unit tests for model training pipeline and predictions.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_models import (
    load_and_preprocess_data,
    train_risk_classifier,
    train_schedule_regressor,
)


class TestDataLoading:
    """Tests for data loading and preprocessing."""
    
    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a temporary CSV file with sample data."""
        data = {
            'Project_ID': ['PROJ-001', 'PROJ-002', 'PROJ-003', 'PROJ-004', 'PROJ-005'],
            'Team_Size': [5, 10, 7, 3, 12],
            'Budget_Allocated': [50000, 100000, 75000, 30000, 120000],
            'Requirement_Changes': [2, 5, 3, 1, 6],
            'Code_Churn': [0.1, 0.3, 0.15, 0.05, 0.4],
            'Team_Experience_Score': [1.2, 0.8, 1.0, 1.4, 0.7],
            'Schedule_Deviation_Days': [0.5, 3.5, 1.5, 0.0, 5.0],
            'Risk_Level': ['Low', 'High', 'Medium', 'Low', 'High']
        }
        df = pd.DataFrame(data)
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        return str(filepath)
    
    def test_load_returns_correct_shapes(self, sample_data_file):
        """Test that loaded data has correct structure."""
        X, y_risk, y_schedule, feature_cols = load_and_preprocess_data(sample_data_file)
        
        assert len(X) == 5
        assert len(y_risk) == 5
        assert len(y_schedule) == 5
        assert len(feature_cols) == 5
    
    def test_feature_columns_correct(self, sample_data_file):
        """Test that correct feature columns are extracted."""
        X, _, _, feature_cols = load_and_preprocess_data(sample_data_file)
        
        expected_cols = [
            'Team_Size', 'Budget_Allocated', 'Requirement_Changes',
            'Code_Churn', 'Team_Experience_Score'
        ]
        assert feature_cols == expected_cols
    
    def test_risk_levels_extracted(self, sample_data_file):
        """Test that risk levels are correctly extracted."""
        _, y_risk, _, _ = load_and_preprocess_data(sample_data_file)
        
        unique_risks = set(y_risk.unique())
        assert unique_risks == {'Low', 'Medium', 'High'}


class TestRiskClassifier:
    """Tests for risk classification model training."""
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = pd.Series(np.random.choice(['Low', 'Medium', 'High'], 100))
        return X_train, y_train
    
    def test_classifier_trains_successfully(self, training_data):
        """Test that classifier trains without errors."""
        X_train, y_train = training_data
        clf, label_encoder = train_risk_classifier(X_train, y_train)
        
        assert clf is not None
        assert label_encoder is not None
    
    def test_classifier_has_correct_classes(self, training_data):
        """Test that classifier learns all risk classes."""
        X_train, y_train = training_data
        clf, label_encoder = train_risk_classifier(X_train, y_train)
        
        classes = set(label_encoder.classes_)
        assert classes == {'Low', 'Medium', 'High'}
    
    def test_classifier_can_predict(self, training_data):
        """Test that classifier can make predictions."""
        X_train, y_train = training_data
        clf, label_encoder = train_risk_classifier(X_train, y_train)
        
        # Predict on training data
        predictions = clf.predict(X_train)
        
        assert len(predictions) == len(X_train)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_classifier_predict_proba(self, training_data):
        """Test that classifier outputs probabilities."""
        X_train, y_train = training_data
        clf, label_encoder = train_risk_classifier(X_train, y_train)
        
        probas = clf.predict_proba(X_train)
        
        assert probas.shape == (100, 3)  # 100 samples, 3 classes
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestScheduleRegressor:
    """Tests for schedule deviation regression model."""
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.uniform(0, 5, 100)  # Days deviation
        return X_train, y_train
    
    def test_regressor_trains_successfully(self, regression_data):
        """Test that regressor trains without errors."""
        X_train, y_train = regression_data
        reg = train_schedule_regressor(X_train, y_train)
        
        assert reg is not None
    
    def test_regressor_can_predict(self, regression_data):
        """Test that regressor can make predictions."""
        X_train, y_train = regression_data
        reg = train_schedule_regressor(X_train, y_train)
        
        predictions = reg.predict(X_train)
        
        assert len(predictions) == len(X_train)
    
    def test_regressor_predictions_reasonable(self, regression_data):
        """Test that regressor predictions are in reasonable range."""
        X_train, y_train = regression_data
        reg = train_schedule_regressor(X_train, y_train)
        
        predictions = reg.predict(X_train)
        
        # Predictions should be in a reasonable range
        assert all(p >= -10 for p in predictions)
        assert all(p <= 15 for p in predictions)


class TestModelIntegration:
    """Integration tests for the full training pipeline."""
    
    @pytest.fixture
    def full_training_data(self):
        """Create comprehensive training dataset."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'Team_Size': np.random.randint(3, 15, n_samples),
            'Budget_Allocated': np.random.uniform(30000, 150000, n_samples),
            'Requirement_Changes': np.random.randint(0, 8, n_samples),
            'Code_Churn': np.random.uniform(0, 0.5, n_samples),
            'Team_Experience_Score': np.random.uniform(0.5, 1.5, n_samples)
        })
        
        y_risk = pd.Series(np.random.choice(['Low', 'Medium', 'High'], n_samples))
        y_schedule = pd.Series(np.random.uniform(0, 6, n_samples))
        
        return X, y_risk, y_schedule
    
    def test_full_pipeline_execution(self, full_training_data):
        """Test complete pipeline from data to predictions."""
        X, y_risk, y_schedule = full_training_data
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        clf, label_encoder = train_risk_classifier(X_scaled, y_risk)
        reg = train_schedule_regressor(X_scaled, y_schedule)
        
        # Make predictions
        risk_preds = clf.predict(X_scaled)
        schedule_preds = reg.predict(X_scaled)
        
        assert len(risk_preds) == len(X)
        assert len(schedule_preds) == len(X)
    
    def test_feature_importance_available(self, full_training_data):
        """Test that feature importances are extractable."""
        X, y_risk, _ = full_training_data
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf, _ = train_risk_classifier(X_scaled, y_risk)
        
        importances = clf.feature_importances_
        
        assert len(importances) == 5
        assert abs(sum(importances) - 1.0) < 0.01  # Should sum to ~1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
