"""
PRMS - Analytics Tests
Unit tests for EVM calculations and technical debt utilities.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import (
    EVMMetrics,
    calculate_evm,
    calculate_simple_cpi,
    calculate_cost_variance,
    calculate_technical_debt_impact,
    get_risk_color,
    get_cpi_status,
    get_spi_status
)


class TestEVMMetrics:
    """Tests for EVMMetrics dataclass calculations."""
    
    def test_cost_variance_under_budget(self):
        """Test CV calculation when under budget."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=40000
        )
        assert evm.cost_variance == 10000  # EV - AC = 50000 - 40000
    
    def test_cost_variance_over_budget(self):
        """Test CV calculation when over budget."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=60000
        )
        assert evm.cost_variance == -10000  # Negative = over budget
    
    def test_cpi_healthy(self):
        """Test CPI > 1.0 (under budget)."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=40000
        )
        assert evm.cost_performance_index == 1.25  # 50000 / 40000
    
    def test_cpi_over_budget(self):
        """Test CPI < 1.0 (over budget)."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=60000
        )
        assert pytest.approx(evm.cost_performance_index, 0.01) == 0.833
    
    def test_cpi_zero_actual_cost(self):
        """Test CPI when actual cost is zero."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=0
        )
        assert evm.cost_performance_index == 0.0
    
    def test_spi_on_schedule(self):
        """Test SPI = 1.0 (on schedule)."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=50000
        )
        assert evm.schedule_performance_index == 1.0
    
    def test_spi_behind_schedule(self):
        """Test SPI < 1.0 (behind schedule)."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=40000,
            actual_cost=50000
        )
        assert evm.schedule_performance_index == 0.8
    
    def test_schedule_variance(self):
        """Test SV calculation."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=40000,
            actual_cost=50000
        )
        assert evm.schedule_variance == -10000  # Behind schedule
    
    def test_estimate_at_completion(self):
        """Test EAC calculation."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=60000
        )
        # EAC = BAC / CPI = 100000 / (50000/60000) = 120000
        assert pytest.approx(evm.estimate_at_completion, 1) == 120000
    
    def test_variance_at_completion(self):
        """Test VAC calculation."""
        evm = EVMMetrics(
            budget_at_completion=100000,
            planned_value=50000,
            earned_value=50000,
            actual_cost=60000
        )
        # VAC = BAC - EAC = 100000 - 120000 = -20000
        assert pytest.approx(evm.variance_at_completion, 1) == -20000


class TestSimpleCPI:
    """Tests for simple CPI calculation."""
    
    def test_cpi_under_budget(self):
        """Test CPI when under budget."""
        cpi = calculate_simple_cpi(100000, 80000)
        assert cpi == 1.25
    
    def test_cpi_over_budget(self):
        """Test CPI when over budget."""
        cpi = calculate_simple_cpi(100000, 120000)
        assert pytest.approx(cpi, 0.01) == 0.833
    
    def test_cpi_zero_cost(self):
        """Test CPI when actual cost is zero."""
        cpi = calculate_simple_cpi(100000, 0)
        assert cpi == 0.0


class TestCostVariance:
    """Tests for cost variance calculation."""
    
    def test_positive_variance(self):
        """Test positive variance (under budget)."""
        cv = calculate_cost_variance(100000, 80000)
        assert cv == 20000
    
    def test_negative_variance(self):
        """Test negative variance (over budget)."""
        cv = calculate_cost_variance(100000, 120000)
        assert cv == -20000
    
    def test_zero_variance(self):
        """Test zero variance (on budget)."""
        cv = calculate_cost_variance(100000, 100000)
        assert cv == 0


class TestTechnicalDebtImpact:
    """Tests for technical debt impact calculations."""
    
    def test_low_debt_impact(self):
        """Test low technical debt scenario."""
        impact = calculate_technical_debt_impact(
            defect_density=2.0,
            code_churn=0.05,
            team_size=5,
            hourly_rate=50
        )
        assert impact.quality_risk_level == "Low"
        assert impact.estimated_rework_hours > 0
        assert impact.estimated_rework_cost > 0
    
    def test_high_debt_impact(self):
        """Test high technical debt scenario."""
        impact = calculate_technical_debt_impact(
            defect_density=15.0,
            code_churn=0.40,
            team_size=10,
            hourly_rate=60
        )
        assert impact.quality_risk_level == "High"
    
    def test_medium_debt_impact(self):
        """Test medium technical debt scenario."""
        # risk_score = (defect_density * 5) + (code_churn * 100 * 2)
        # For Medium: 40 <= score < 70
        # With defect_density=6.0 and code_churn=0.10: score = 30 + 20 = 50
        impact = calculate_technical_debt_impact(
            defect_density=6.0,
            code_churn=0.10,
            team_size=7,
            hourly_rate=55
        )
        assert impact.quality_risk_level == "Medium"


class TestColorUtilities:
    """Tests for color/status utility functions."""
    
    def test_risk_colors(self):
        """Test risk level color mapping."""
        assert get_risk_color("High") == "#e74c3c"
        assert get_risk_color("Medium") == "#f39c12"
        assert get_risk_color("Low") == "#27ae60"
        assert get_risk_color("Unknown") == "#95a5a6"
    
    def test_cpi_status_healthy(self):
        """Test CPI status for healthy project."""
        status, color = get_cpi_status(1.1)
        assert status == "On/Under Budget"
        assert color == "#27ae60"
    
    def test_cpi_status_warning(self):
        """Test CPI status for warning level."""
        status, color = get_cpi_status(0.92)
        assert status == "Slightly Over Budget"
        assert color == "#f39c12"
    
    def test_cpi_status_critical(self):
        """Test CPI status for critical level."""
        status, color = get_cpi_status(0.8)
        assert status == "Significantly Over Budget"
        assert color == "#e74c3c"
    
    def test_spi_status_healthy(self):
        """Test SPI status for on-time project."""
        status, color = get_spi_status(1.0)
        assert status == "On/Ahead of Schedule"
        assert color == "#27ae60"
    
    def test_spi_status_warning(self):
        """Test SPI status for slight delay."""
        status, color = get_spi_status(0.95)
        assert status == "Slightly Behind Schedule"
        assert color == "#f39c12"
    
    def test_spi_status_critical(self):
        """Test SPI status for major delay."""
        status, color = get_spi_status(0.7)
        assert status == "Significantly Behind Schedule"
        assert color == "#e74c3c"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
