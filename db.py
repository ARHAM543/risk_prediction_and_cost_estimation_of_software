import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_samples=1000):
    data = []

    for _ in range(num_samples):
        # --- 1. PROJECT PLANNING METRICS (The Baseline) ---
        project_id = f"PROJ-{random.randint(100, 999)}"
        team_size = random.randint(3, 15)
        planned_sprint_duration = 14  # standard 2 weeks in days
        
        # Budget per developer per hour (approx $40-$80)
        hourly_rate = np.round(np.random.uniform(40, 80), 2)
        
        # Total budget for the sprint (Team * Hours * Rate)
        # Assuming 6 effective hours per day * 10 working days
        planned_hours = team_size * 6 * 10
        budget_allocated = planned_hours * hourly_rate

        # --- 2. EXECUTION METRICS (The "Chaos" Factors) ---
        # Requirement Changes (Scope Creep): 0 to 8 changes per sprint
        # Higher complexity projects get more changes
        req_changes = np.random.poisson(2) 
        
        # Code Churn: Percentage of code rewritten (0% to 50%)
        # High churn often leads to bugs
        code_churn = np.round(np.random.beta(2, 10), 2) 

        # Team Experience Factor (0.5 = Junior, 1.5 = Senior)
        team_experience = np.random.uniform(0.5, 1.5)

        # --- 3. CALCULATING OUTCOMES (The Effects) ---
        
        # Effect 1: Actual Hours Worked
        # Hours increase if there are changes or high churn. 
        # Senior teams (high experience) work faster.
        inefficiency_factor = (1 + (req_changes * 0.05) + (code_churn * 0.2)) / team_experience
        actual_hours = planned_hours * inefficiency_factor
        
        # Add some random noise (unexpected issues)
        actual_hours += np.random.normal(0, 10) 
        actual_hours = max(actual_hours, planned_hours * 0.8) # Can't be impossibly low

        # Effect 2: Defect Density (Bugs per 1K lines)
        # Correlated with Churn and Low Experience
        base_defects = np.random.normal(5, 2)
        defect_density = base_defects + (code_churn * 20) - (team_experience * 2)
        defect_density = max(0, defect_density) # No negative bugs

        # --- 4. FINANCIAL METRICS (EVM) ---
        actual_cost = actual_hours * hourly_rate
        
        # Cost Variance (Negative means Over Budget)
        cost_variance = budget_allocated - actual_cost
        
        # Cost Performance Index (CPI)
        # CPI < 1.0 means you are over budget
        cpi = budget_allocated / actual_cost if actual_cost > 0 else 0

        # --- 5. SCHEDULE METRICS ---
        # Calculate delay based on extra hours needed vs team capacity
        # If actual_hours > planned_hours, we have a delay
        overtime_hours = max(0, actual_hours - planned_hours)
        
        # Assume team can absorb 10% overtime, rest becomes delay
        unabsorbed_hours = max(0, overtime_hours - (planned_hours * 0.10))
        daily_capacity = team_size * 6
        schedule_deviation_days = np.round(unabsorbed_hours / daily_capacity, 1)

        # --- 6. RISK LABELS (Target Variable) ---
        # Logic to define "Risk" based on Cost and Time
        if schedule_deviation_days > 3 or cpi < 0.85:
            risk_level = "High"
        elif schedule_deviation_days > 1 or cpi < 0.95:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Append row
        data.append([
            project_id, team_size, budget_allocated, 
            req_changes, code_churn, team_experience, 
            actual_hours, defect_density, 
            actual_cost, cost_variance, cpi, 
            schedule_deviation_days, risk_level
        ])

    # Convert to DataFrame
    columns = [
        "Project_ID", "Team_Size", "Budget_Allocated", 
        "Requirement_Changes", "Code_Churn", "Team_Experience_Score", 
        "Actual_Hours", "Defect_Density", 
        "Actual_Cost", "Cost_Variance", "CPI", 
        "Schedule_Deviation_Days", "Risk_Level"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate and view data
df_projects = generate_synthetic_data(1000)

# Display first 5 rows
print(df_projects.head())

# Optional: Save to CSV for your project
df_projects.to_csv("software_project_risk_data.csv", index=False)