"""
PRMS - Analytics Module
Earned Value Management (EVM) calculations and Technical Debt utilities.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EVMMetrics:
    """Container for Earned Value Management metrics."""
    budget_at_completion: float  # BAC
    planned_value: float         # PV
    earned_value: float          # EV
    actual_cost: float           # AC
    
    @property
    def cost_variance(self) -> float:
        """CV = EV - AC (Negative = Over Budget)"""
        return self.earned_value - self.actual_cost
    
    @property
    def schedule_variance(self) -> float:
        """SV = EV - PV (Negative = Behind Schedule)"""
        return self.earned_value - self.planned_value
    
    @property
    def cost_performance_index(self) -> float:
        """CPI = EV / AC (< 1.0 = Over Budget)"""
        if self.actual_cost == 0:
            return 0.0
        return self.earned_value / self.actual_cost
    
    @property
    def schedule_performance_index(self) -> float:
        """SPI = EV / PV (< 1.0 = Behind Schedule)"""
        if self.planned_value == 0:
            return 0.0
        return self.earned_value / self.planned_value
    
    @property
    def estimate_at_completion(self) -> float:
        """EAC = BAC / CPI"""
        cpi = self.cost_performance_index
        if cpi == 0:
            return float('inf')
        return self.budget_at_completion / cpi
    
    @property
    def variance_at_completion(self) -> float:
        """VAC = BAC - EAC"""
        return self.budget_at_completion - self.estimate_at_completion
    
    @property
    def to_complete_performance_index(self) -> float:
        """TCPI = (BAC - EV) / (BAC - AC)"""
        denominator = self.budget_at_completion - self.actual_cost
        if denominator == 0:
            return float('inf')
        return (self.budget_at_completion - self.earned_value) / denominator


def calculate_evm(
    budget_allocated: float,
    planned_hours: float,
    actual_hours: float,
    percent_complete: float,
    hourly_rate: float
) -> EVMMetrics:
    """
    Calculate EVM metrics from project parameters.
    
    Args:
        budget_allocated: Total project budget (BAC)
        planned_hours: Hours planned to date
        actual_hours: Hours actually worked
        percent_complete: Work completion percentage (0-100)
        hourly_rate: Cost per hour
    
    Returns:
        EVMMetrics dataclass with calculated values
    """
    bac = budget_allocated
    pv = (planned_hours / (planned_hours if planned_hours > 0 else 1)) * bac
    ev = (percent_complete / 100) * bac
    ac = actual_hours * hourly_rate
    
    return EVMMetrics(
        budget_at_completion=bac,
        planned_value=pv,
        earned_value=ev,
        actual_cost=ac
    )


def calculate_simple_cpi(budget_allocated: float, actual_cost: float) -> float:
    """
    Simple CPI calculation: Budget / Actual Cost.
    CPI < 1.0 means over budget.
    """
    if actual_cost == 0:
        return 0.0
    return budget_allocated / actual_cost


def calculate_cost_variance(budget_allocated: float, actual_cost: float) -> float:
    """
    Cost Variance = Budget - Actual.
    Negative means over budget.
    """
    return budget_allocated - actual_cost


# =============================================================================
# Technical Debt Utilities
# =============================================================================

@dataclass
class TechnicalDebtImpact:
    """Container for technical debt analysis results."""
    defect_density: float
    code_churn: float
    estimated_rework_hours: float
    estimated_rework_cost: float
    quality_risk_score: float  # 0-100
    
    @property
    def quality_risk_level(self) -> str:
        """Categorize quality risk based on score."""
        if self.quality_risk_score >= 70:
            return "High"
        elif self.quality_risk_score >= 40:
            return "Medium"
        return "Low"


def calculate_technical_debt_impact(
    defect_density: float,
    code_churn: float,
    team_size: int,
    hourly_rate: float,
    hours_per_defect: float = 4.0
) -> TechnicalDebtImpact:
    """
    Calculate the financial impact of technical debt.
    
    Args:
        defect_density: Defects per 1K lines of code
        code_churn: Percentage of code rewritten (0-1)
        team_size: Number of team members
        hourly_rate: Cost per hour
        hours_per_defect: Average hours to fix a defect
    
    Returns:
        TechnicalDebtImpact with estimated costs
    """
    # Estimate total defects based on assumed codebase size
    estimated_kloc = team_size * 2  # ~2K LOC per developer
    total_defects = defect_density * estimated_kloc
    
    # Rework from defects
    defect_rework_hours = total_defects * hours_per_defect
    
    # Rework from churn (high churn = more review/testing)
    churn_rework_hours = code_churn * team_size * 20  # ~20 hours impact per team member
    
    total_rework_hours = defect_rework_hours + churn_rework_hours
    total_rework_cost = total_rework_hours * hourly_rate
    
    # Quality risk score (0-100)
    # Based on weighted combination of defect density and churn
    risk_score = min(100, (defect_density * 5) + (code_churn * 100 * 2))
    
    return TechnicalDebtImpact(
        defect_density=defect_density,
        code_churn=code_churn,
        estimated_rework_hours=round(total_rework_hours, 1),
        estimated_rework_cost=round(total_rework_cost, 2),
        quality_risk_score=round(risk_score, 1)
    )


def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level visualization."""
    colors = {
        "High": "#e74c3c",    # Red
        "Medium": "#f39c12",  # Orange
        "Low": "#27ae60"      # Green
    }
    return colors.get(risk_level, "#95a5a6")


def get_cpi_status(cpi: float) -> Tuple[str, str]:
    """Get status label and color based on CPI value."""
    if cpi >= 1.0:
        return "On/Under Budget", "#27ae60"
    elif cpi >= 0.9:
        return "Slightly Over Budget", "#f39c12"
    else:
        return "Significantly Over Budget", "#e74c3c"


def get_spi_status(spi: float) -> Tuple[str, str]:
    """Get status label and color based on SPI value."""
    if spi >= 1.0:
        return "On/Ahead of Schedule", "#27ae60"
    elif spi >= 0.9:
        return "Slightly Behind Schedule", "#f39c12"
    else:
        return "Significantly Behind Schedule", "#e74c3c"


def explain_risk(inputs: dict, risk_level: str) -> list:
    """
    Generate explainable AI reasons for the predicted risk level.
    
    Args:
        inputs: Dictionary of input parameters
        risk_level: Predicted risk level (High, Medium, Low)
        
    Returns:
        List of string explanations
    """
    reasons = []
    
    # Thresholds (logic derived from typical software risk factors)
    if risk_level == "High":
        if inputs['code_churn'] > 0.20:
            reasons.append(f"High Code Churn ({inputs['code_churn']*100:.1f}%) indicates instability and potential rework.")
        if inputs['team_experience'] < 0.8:
            reasons.append(f"Low Team Experience Score ({inputs['team_experience']}) increases likelihood of defects/delays.")
        if inputs['req_changes'] > 4:
            reasons.append(f"Frequent Requirement Changes ({inputs['req_changes']}) destabilize the development plan.")
        if inputs['budget_allocated'] < 20000: # Threshold in USD
            reasons.append("Tight Budget allocation might lead to resource constraints.")
    
    elif risk_level == "Medium":
        if 0.10 < inputs['code_churn'] <= 0.20:
            reasons.append(f"Moderate Code Churn ({inputs['code_churn']*100:.1f}%) suggests evolving stability.")
        if 0.8 <= inputs['team_experience'] < 1.2:
            reasons.append("Team Experience is adequate but may face challenges with complex tasks.")
            
    if not reasons and risk_level != "Low":
        reasons.append("Combination of factors (Churn vs Experience ratios) resulted in elevated risk.")
        
    if risk_level == "Low":
        reasons.append("All project parameters are within healthy ranges.")
        reasons.append(f"Team Experience ({inputs['team_experience']}) effectively offsets Code Churn.")

    return reasons


def get_mitigation_plan(inputs: dict) -> list:
    """
    Generate actionable suggestions to reduce risk.
    """
    suggestions = []
    
    # Analyze Code Churn
    if inputs['code_churn'] > 0.15:
        suggestions.append("📉 **Reduce Code Churn**: Freeze core modules and enforce stricter code reviews to lower churn below 15%.")
    
    # Analyze Requirements
    if inputs['req_changes'] > 2:
        suggestions.append("📝 **Lock Requirements**: Implement a 'Requirement Freeze' period in the next sprint to stabilize scope.")
        
    # Analyze Experience
    if inputs['team_experience'] < 1.0:
        suggestions.append("👨‍🏫 **Senior Mentorship**: pair junior devs with seniors or conduct daily stand-ups to mitigate low experience impact.")
    
    # Budget check (Example logic)
    # Convert input budget (assumed USD for logic here)
    if inputs['budget_allocated'] < 30000 and inputs['team_size'] > 5:
        suggestions.append("💰 **Budget Review**: Budget seems low for the team size. Consider requesting a 15-20% contingency fund.")
        
    if not suggestions:
        suggestions.append("✅ Project is on solid track. Maintain current practices and monitor daily stand-ups.")
        
    return suggestions


def explain_schedule_deviation(inputs: dict, deviation_days: float) -> dict:
    """
    Generate a detailed breakdown of schedule deviation.
    The breakdown is proportionally scaled to match the actual model prediction.
    
    Returns:
        Dictionary with base_schedule, deviation_breakdown, total, and explanations
    """
    team_size = inputs['team_size']
    code_churn = inputs['code_churn']
    req_changes = inputs['req_changes']
    team_experience = inputs['team_experience']
    
    # Base schedule calculation (heuristic: 6 weeks * 5 days)
    base_weeks = 6
    base_days = base_weeks * 5  # 30 working days baseline
    
    # Calculate relative weights for each factor (these are proportional, not absolute)
    churn_weight = code_churn * 3  # Higher churn = more weight
    req_weight = req_changes * 0.5  # Each req change contributes
    experience_weight = max(0, (1.0 - team_experience) * 2)  # Low experience adds weight
    team_weight = max(0, (10 - team_size) * 0.2) if team_size < 10 else 0  # Small teams
    
    # Total weight
    total_weight = churn_weight + req_weight + experience_weight + team_weight
    
    # Proportionally distribute the ACTUAL deviation across factors
    if total_weight > 0 and deviation_days > 0:
        churn_impact = (churn_weight / total_weight) * deviation_days
        req_impact = (req_weight / total_weight) * deviation_days
        experience_impact = (experience_weight / total_weight) * deviation_days
        team_impact = (team_weight / total_weight) * deviation_days
    else:
        churn_impact = req_impact = experience_impact = team_impact = 0
    
    breakdown = {
        'base_schedule_days': base_days,
        'churn_impact_days': round(churn_impact, 1),
        'req_change_impact_days': round(req_impact, 1),
        'experience_impact_days': round(experience_impact, 1),
        'team_size_impact_days': round(team_impact, 1),
        'total_deviation_days': round(deviation_days, 1),
        'final_schedule_days': round(base_days + deviation_days, 1)
    }
    
    # Explanations for each deviation factor
    explanations = []
    
    if churn_impact > 0.1:
        explanations.append(f"🔄 **Code Churn ({code_churn*100:.0f}%)** contributes ~{churn_impact:.1f} days due to rework and testing overhead.")
    
    if req_impact > 0.1:
        explanations.append(f"📋 **{req_changes} Requirement Changes** contribute ~{req_impact:.1f} days for analysis and re-testing.")
    
    if experience_impact > 0.1:
        explanations.append(f"👤 **Team Experience ({team_experience})** contributes ~{experience_impact:.1f} days due to learning curve.")
    
    if team_impact > 0.1:
        explanations.append(f"👥 **Team Size ({team_size})** contributes ~{team_impact:.1f} days as fewer resources available.")
    
    if not explanations:
        explanations.append("✅ Project parameters are optimal - minimal schedule deviation expected.")
    
    breakdown['explanations'] = explanations
    
    return breakdown


