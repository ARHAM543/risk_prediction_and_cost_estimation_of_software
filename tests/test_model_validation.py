"""
PRMS - Model Validation Tests
Unit tests for model validation utilities.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tempfile
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_validation import (
    load_models,
    validate_classifier,
    validate_regressor,
    run_cross_validation
)


class TestModelLoading:
    """Tests for model loading functionality."""
    
    @pytest.fixture
    def mock_models_dir(self, tmp_path):
        """Create temporary directory with mock models."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Create mock classifier
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        clf.fit(X, y)
        
        # Create mock regressor
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        y_reg = np.random.uniform(0, 5, 50)
        reg.fit(X, y_reg)
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['High', 'Low', 'Medium'])
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Feature columns
        feature_cols = ['Team_Size', 'Budget_Allocated', 'Requirement_Changes',
                       'Code_Churn', 'Team_Experience_Score']
        
        # Save all
        joblib.dump(clf, model_dir / "risk_classifier.joblib")
        joblib.dump(reg, model_dir / "schedule_regressor.joblib")
        joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
        joblib.dump(scaler, model_dir / "scaler.joblib")
        joblib.dump(feature_cols, model_dir / "feature_cols.joblib")
        
        return str(model_dir)
    
    def test_load_models_returns_all_components(self, mock_models_dir):
        """Test that all model components are loaded."""
        clf, label_encoder, reg, scaler, feature_cols = load_models(mock_models_dir)
        
        assert clf is not None
        assert label_encoder is not None
        assert reg is not None
        assert scaler is not None
        assert feature_cols is not None
    
    def test_load_classifier_is_fitted(self, mock_models_dir):
        """Test that loaded classifier is fitted."""
        clf, _, _, _, _ = load_models(mock_models_dir)
        
        # Should be able to predict
        X_test = np.random.randn(5, 5)
        predictions = clf.predict(X_test)
        
        assert len(predictions) == 5
    
    def test_load_regressor_is_fitted(self, mock_models_dir):
        """Test that loaded regressor is fitted."""
        _, _, reg, _, _ = load_models(mock_models_dir)
        
        X_test = np.random.randn(5, 5)
        predictions = reg.predict(X_test)
        
        assert len(predictions) == 5
    
    def test_feature_columns_correct(self, mock_models_dir):
        """Test that feature columns are correctly loaded."""
        _, _, _, _, feature_cols = load_models(mock_models_dir)
        
        expected = ['Team_Size', 'Budget_Allocated', 'Requirement_Changes',
                   'Code_Churn', 'Team_Experience_Score']
        assert feature_cols == expected


class TestClassifierValidation:
    """Tests for classifier validation metrics."""
    
    @pytest.fixture
    def classifier_setup(self):
        """Create classifier and test data."""
        np.random.seed(42)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        X_train = np.random.randn(200, 5)
        y_train = np.random.randint(0, 3, 200)
        clf.fit(X_train, y_train)
        
        # Create test data
        X_test = np.random.randn(50, 5)
        y_test = pd.Series(['Low', 'Medium', 'High'] * 16 + ['Low', 'Medium'])
        
        # Create label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(['High', 'Low', 'Medium'])
        
        return clf, label_encoder, X_test, y_test
    
    def test_validate_classifier_returns_metrics(self, classifier_setup):
        """Test that validation returns expected metrics."""
        clf, label_encoder, X_test, y_test = classifier_setup
        
        metrics = validate_classifier(clf, label_encoder, X_test, y_test)
        
        assert 'f1_weighted' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_f1_scores_in_valid_range(self, classifier_setup):
        """Test that F1 scores are between 0 and 1."""
        clf, label_encoder, X_test, y_test = classifier_setup
        
        metrics = validate_classifier(clf, label_encoder, X_test, y_test)
        
        assert 0 <= metrics['f1_weighted'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_confusion_matrix_shape(self, classifier_setup):
        """Test confusion matrix has correct shape."""
        clf, label_encoder, X_test, y_test = classifier_setup
        
        metrics = validate_classifier(clf, label_encoder, X_test, y_test)
        
        cm = metrics['confusion_matrix']
        assert cm.shape == (3, 3)  # 3 classes


class TestRegressorValidation:
    """Tests for regressor validation metrics."""
    
    @pytest.fixture
    def regressor_setup(self):
        """Create regressor and test data."""
        np.random.seed(42)
        
        # Train regressor
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        X_train = np.random.randn(200, 5)
        y_train = np.random.uniform(0, 5, 200)
        reg.fit(X_train, y_train)
        
        # Test data
        X_test = np.random.randn(50, 5)
        y_test = pd.Series(np.random.uniform(0, 5, 50))
        
        return reg, X_test, y_test
    
    def test_validate_regressor_returns_metrics(self, regressor_setup):
        """Test that validation returns expected metrics."""
        reg, X_test, y_test = regressor_setup
        
        metrics = validate_regressor(reg, X_test, y_test)
        
        assert 'r2_score' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
    
    def test_r2_score_reasonable(self, regressor_setup):
        """Test R2 score is in reasonable range."""
        reg, X_test, y_test = regressor_setup
        
        metrics = validate_regressor(reg, X_test, y_test)
        
        # R2 can be negative, but shouldn't be extremely so
        assert metrics['r2_score'] > -5
        assert metrics['r2_score'] <= 1
    
    def test_mae_non_negative(self, regressor_setup):
        """Test that MAE is non-negative."""
        reg, X_test, y_test = regressor_setup
        
        metrics = validate_regressor(reg, X_test, y_test)
        
        assert metrics['mae'] >= 0
    
    def test_rmse_non_negative(self, regressor_setup):
        """Test that RMSE is non-negative."""
        reg, X_test, y_test = regressor_setup
        
        metrics = validate_regressor(reg, X_test, y_test)
        
        assert metrics['rmse'] >= 0


class TestCrossValidation:
    """Tests for cross-validation functionality."""
    
    @pytest.fixture
    def cv_setup(self):
        """Create models and data for cross-validation."""
        np.random.seed(42)
        
        # Create models
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        reg = RandomForestRegressor(n_estimators=20, random_state=42)
        
        # Create data
        X = np.random.randn(100, 5)
        y_risk = pd.Series(['Low', 'Medium', 'High'] * 33 + ['Low'])
        y_schedule = pd.Series(np.random.uniform(0, 5, 100))
        
        # Label encoder
        label_encoder = LabelEncoder()
        
        return clf, reg, X, y_risk, y_schedule, label_encoder
    
    def test_cv_returns_metrics(self, cv_setup):
        """Test that cross-validation returns expected metrics."""
        clf, reg, X, y_risk, y_schedule, label_encoder = cv_setup
        
        metrics = run_cross_validation(clf, reg, X, y_risk, y_schedule, label_encoder, cv=3)
        
        assert 'clf_cv_mean' in metrics
        assert 'clf_cv_std' in metrics
        assert 'reg_cv_mean' in metrics
        assert 'reg_cv_std' in metrics
    
    def test_cv_mean_in_valid_range(self, cv_setup):
        """Test that CV means are in valid range."""
        clf, reg, X, y_risk, y_schedule, label_encoder = cv_setup
        
        metrics = run_cross_validation(clf, reg, X, y_risk, y_schedule, label_encoder, cv=3)
        
        # F1 should be between 0 and 1
        assert 0 <= metrics['clf_cv_mean'] <= 1
        # R2 could be negative for bad models
        assert metrics['reg_cv_mean'] <= 1
    
    def test_cv_std_non_negative(self, cv_setup):
        """Test that CV standard deviations are non-negative."""
        clf, reg, X, y_risk, y_schedule, label_encoder = cv_setup
        
        metrics = run_cross_validation(clf, reg, X, y_risk, y_schedule, label_encoder, cv=3)
        
        assert metrics['clf_cv_std'] >= 0
        assert metrics['reg_cv_std'] >= 0


class TestValidationEdgeCases:
    """Tests for edge cases in validation."""
    
    def test_perfect_classifier_f1_is_one(self):
        """Test that perfect predictions give F1 = 1."""
        np.random.seed(42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        X = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]] * 50)
        y = [0, 1, 2] * 50
        clf.fit(X, y)
        
        label_encoder = LabelEncoder()
        label_encoder.fit(['High', 'Low', 'Medium'])
        
        y_test = pd.Series(['High', 'Low', 'Medium'] * 10)
        X_test = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]] * 10)
        
        metrics = validate_classifier(clf, label_encoder, X_test, y_test)
        
        # F1 should be high (close to 1) for perfect predictions
        assert metrics['f1_weighted'] > 0.8
    
    def test_perfect_regressor_r2_is_one(self):
        """Test that perfect predictions give R2 close to 1."""
        np.random.seed(42)
        
        # Create perfectly predictable data
        X = np.random.randn(100, 5)
        y_true = X[:, 0] + X[:, 1]  # Simple linear relationship
        
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X, y_true)
        
        y_test = pd.Series(y_true)
        
        metrics = validate_regressor(reg, X, y_test)
        
        # R2 should be high for predictable data
        assert metrics['r2_score'] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
