"""
PRMS - Data Generation Tests
Unit tests for synthetic data generation.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import generate_synthetic_data


class TestDataGenerationBasics:
    """Tests for basic data generation functionality."""
    
    def test_generates_correct_number_of_samples(self):
        """Test that correct number of samples are generated."""
        for n in [10, 100, 500]:
            df = generate_synthetic_data(n)
            assert len(df) == n
    
    def test_generates_all_required_columns(self):
        """Test that all required columns are present."""
        df = generate_synthetic_data(100)
        
        required_columns = [
            "Project_ID", "Team_Size", "Budget_Allocated",
            "Requirement_Changes", "Code_Churn", "Team_Experience_Score",
            "Actual_Hours", "Defect_Density",
            "Actual_Cost", "Cost_Variance", "CPI",
            "Schedule_Deviation_Days", "Risk_Level"
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_no_null_values(self):
        """Test that generated data has no null values."""
        df = generate_synthetic_data(100)
        assert df.isnull().sum().sum() == 0


class TestProjectIDGeneration:
    """Tests for Project ID generation."""
    
    def test_project_id_format(self):
        """Test that project IDs follow the PROJ-XXX format."""
        df = generate_synthetic_data(50)
        
        for pid in df['Project_ID']:
            assert pid.startswith('PROJ-')
            assert len(pid) == 8  # PROJ-XXX
    
    def test_project_ids_are_strings(self):
        """Test that project IDs are string type."""
        df = generate_synthetic_data(10)
        assert df['Project_ID'].dtype == object


class TestTeamMetrics:
    """Tests for team-related metrics."""
    
    def test_team_size_range(self):
        """Test that team sizes are within expected range (3-15)."""
        df = generate_synthetic_data(200)
        
        assert df['Team_Size'].min() >= 3
        assert df['Team_Size'].max() <= 15
    
    def test_team_experience_range(self):
        """Test that team experience is within expected range (0.5-1.5)."""
        df = generate_synthetic_data(200)
        
        assert df['Team_Experience_Score'].min() >= 0.5
        assert df['Team_Experience_Score'].max() <= 1.5


class TestBudgetMetrics:
    """Tests for budget-related metrics."""
    
    def test_budget_is_positive(self):
        """Test that all budgets are positive."""
        df = generate_synthetic_data(100)
        assert (df['Budget_Allocated'] > 0).all()
    
    def test_actual_cost_is_positive(self):
        """Test that actual costs are positive."""
        df = generate_synthetic_data(100)
        assert (df['Actual_Cost'] > 0).all()
    
    def test_cpi_is_positive(self):
        """Test that CPI values are positive."""
        df = generate_synthetic_data(100)
        assert (df['CPI'] > 0).all()
    
    def test_cost_variance_calculation(self):
        """Test that cost variance = budget - actual."""
        df = generate_synthetic_data(100)
        
        calculated_cv = df['Budget_Allocated'] - df['Actual_Cost']
        difference = abs(df['Cost_Variance'] - calculated_cv)
        
        assert (difference < 0.01).all()


class TestCodeQualityMetrics:
    """Tests for code quality metrics."""
    
    def test_code_churn_range(self):
        """Test that code churn is between 0 and 1."""
        df = generate_synthetic_data(200)
        
        assert (df['Code_Churn'] >= 0).all()
        assert (df['Code_Churn'] <= 1).all()
    
    def test_defect_density_non_negative(self):
        """Test that defect density is non-negative."""
        df = generate_synthetic_data(200)
        assert (df['Defect_Density'] >= 0).all()
    
    def test_requirement_changes_non_negative(self):
        """Test that requirement changes are non-negative integers."""
        df = generate_synthetic_data(200)
        assert (df['Requirement_Changes'] >= 0).all()


class TestScheduleMetrics:
    """Tests for schedule-related metrics."""
    
    def test_schedule_deviation_non_negative(self):
        """Test that schedule deviation is non-negative."""
        df = generate_synthetic_data(200)
        assert (df['Schedule_Deviation_Days'] >= 0).all()
    
    def test_actual_hours_positive(self):
        """Test that actual hours are positive."""
        df = generate_synthetic_data(200)
        assert (df['Actual_Hours'] > 0).all()


class TestRiskLevelAssignment:
    """Tests for risk level assignment logic."""
    
    def test_risk_levels_valid(self):
        """Test that only valid risk levels are assigned."""
        df = generate_synthetic_data(200)
        
        valid_levels = {'Low', 'Medium', 'High'}
        actual_levels = set(df['Risk_Level'].unique())
        
        assert actual_levels.issubset(valid_levels)
    
    def test_high_risk_conditions(self):
        """Test that High risk is assigned for severe conditions."""
        df = generate_synthetic_data(500)
        
        high_risk = df[df['Risk_Level'] == 'High']
        
        # All high risk should have schedule > 3 OR cpi < 0.85
        for _, row in high_risk.iterrows():
            assert row['Schedule_Deviation_Days'] > 3 or row['CPI'] < 0.85
    
    def test_low_risk_conditions(self):
        """Test that Low risk has good metrics."""
        df = generate_synthetic_data(500)
        
        low_risk = df[df['Risk_Level'] == 'Low']
        
        # Low risk should have schedule <= 1 AND cpi >= 0.95
        for _, row in low_risk.iterrows():
            assert row['Schedule_Deviation_Days'] <= 1 and row['CPI'] >= 0.95
    
    def test_risk_distribution_reasonable(self):
        """Test that risk distribution is reasonable."""
        df = generate_synthetic_data(1000)
        
        risk_counts = df['Risk_Level'].value_counts()
        
        # Each risk level should represent a meaningful portion
        for level in ['Low', 'Medium', 'High']:
            assert risk_counts.get(level, 0) > 50  # At least 5%


class TestDataReproducibility:
    """Tests for data generation reproducibility with seeds."""
    
    def test_same_seed_same_data(self):
        """Test that same seed produces same data structure."""
        np.random.seed(42)
        df1 = generate_synthetic_data(100)
        
        np.random.seed(42)
        df2 = generate_synthetic_data(100)
        
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)


class TestStatisticalProperties:
    """Tests for statistical properties of generated data."""
    
    def test_team_size_variability(self):
        """Test that team sizes have reasonable variability."""
        df = generate_synthetic_data(500)
        
        std = df['Team_Size'].std()
        assert std > 1  # Should have some variability
    
    def test_budget_variability(self):
        """Test that budgets have reasonable variability."""
        df = generate_synthetic_data(500)
        
        std = df['Budget_Allocated'].std()
        assert std > 10000  # Should have meaningful spread
    
    def test_cpi_distribution(self):
        """Test that CPI has expected distribution characteristics."""
        df = generate_synthetic_data(500)
        
        mean_cpi = df['CPI'].mean()
        # CPI should generally be around 1.0 (some over, some under budget)
        assert 0.8 < mean_cpi < 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
