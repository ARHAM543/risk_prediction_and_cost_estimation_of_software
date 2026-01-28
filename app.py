"""
PRMS - Streamlit Dashboard
Predictive Risk Monitoring System with ML-powered forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

from analytics import (
    calculate_simple_cpi,
    calculate_cost_variance,
    calculate_technical_debt_impact,
    get_risk_color,
    get_cpi_status,
    explain_risk,
    get_mitigation_plan,
    explain_schedule_deviation,
    TechnicalDebtImpact
)

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="PRMS - Risk Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #e74c3c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #f39c12;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #27ae60;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_historical_data():
    """Load historical project data."""
    return pd.read_csv("software_project_risk_data.csv")


@st.cache_resource
def load_models():
    """Load trained ML models."""
    model_dir = "models"
    try:
        clf = joblib.load(os.path.join(model_dir, "risk_classifier.joblib"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        reg = joblib.load(os.path.join(model_dir, "schedule_regressor.joblib"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))
        return clf, label_encoder, reg, scaler, feature_cols
    except FileNotFoundError:
        return None, None, None, None, None


# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.markdown("## 🎯 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "📈 Analytics", "🔄 What-If Comparison"],
    label_visibility="collapsed"
)

# =============================================================================
# Dashboard Page
# =============================================================================
def render_dashboard():
    st.markdown('<h1 class="main-header">📊 Project Risk Dashboard & Simulator</h1>', unsafe_allow_html=True)
    
    models = load_models()
    clf, label_encoder, reg, scaler, feature_cols = models
    
    if clf is None:
        st.error("⚠️ Models not found! Please run `python train_models.py` first to train the models.")
        return

    # Container for inputs
    with st.container():
        st.subheader("🛠️ Project Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team_size = st.slider("Team Size", min_value=3, max_value=20, value=8, help="Total number of members in the project team")
            # Budget in INR. Default 50k USD * 83 ~= 4,150,000 INR
            budget_inr = st.number_input(
                "Budget Allocated (₹)", 
                min_value=500000, 
                max_value=50000000, 
                value=4000000,
                step=100000,
                format="%d",
                help="Total financial budget allocated for the project in INR"
            )
            
        with col2:
            req_changes = st.slider("Expected Requirement Changes", min_value=0, max_value=20, value=3, help="Number of significant changes to requirements expected during the project")
            # Convert INR input to USD for model (Approx 83 INR = 1 USD)
            budget_usd = budget_inr / 83.0
            
        with col3:
            code_churn = st.slider("Code Churn (%)", min_value=0, max_value=100, value=15, help="Percentage of code expected to be rewritten or deleted") / 100
            team_experience = st.slider(
                "Team Experience Score", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0,
                step=0.1,
                help="Average experience level of the team (0.5=Junior, 1.0=Mid, 2.0=Senior)"
            )

    if st.button("🚀 Predict & Analyze Risk", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[team_size, budget_usd, req_changes, code_churn, team_experience]])
        input_scaled = scaler.transform(input_data)
        
        # Predictions
        risk_pred_encoded = clf.predict(input_scaled)[0]
        risk_pred = label_encoder.inverse_transform([risk_pred_encoded])[0]
        risk_proba = clf.predict_proba(input_scaled)[0]
        schedule_deviation = reg.predict(input_scaled)[0]
        
        # --- Calculations ---
        hourly_rate_usd = 60
        hourly_rate_inr = hourly_rate_usd * 83
        
        planned_hours = team_size * 6 * 10 # heuristic
        estimated_actual_hours = planned_hours * (1 + (req_changes * 0.05) + (code_churn * 0.2)) / team_experience
        estimated_cost_inr = estimated_actual_hours * hourly_rate_inr
        
        cpi = budget_inr / estimated_cost_inr if estimated_cost_inr > 0 else 0
        
        # Explanations
        input_dict = {
            'team_size': team_size, 
            'budget_allocated': budget_usd, 
            'req_changes': req_changes, 
            'code_churn': code_churn, 
            'team_experience': team_experience
        }
        explanations = explain_risk(input_dict, risk_pred)
        suggestions = get_mitigation_plan(input_dict)
        
        # --- UI Results ---
        st.divider()
        
        # 1. Main Risk Card
        alert_class = f"alert-{risk_pred.lower()}"
        alert_icon = "🔴" if risk_pred == "High" else "🟡" if risk_pred == "Medium" else "🟢"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <h2 style='margin:0'>{alert_icon} Risk Level: {risk_pred}</h2>
            <p style='margin-top:5px; font-size:1.1rem'>Predicted Schedule Deviation: <strong>{max(0, schedule_deviation):.1f} days</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1.5, 1])
        
        with col_res1:
            # 2. Explainable AI Section
            st.subheader("💡 Why is this the risk level?")
            for reason in explanations:
                st.markdown(f"- {reason}")
                
            st.subheader("🛡️ Recommended Mitigation Steps")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
                
        with col_res2:
            # 3. Cost Estimation Section
            st.subheader("💰 Cost Estimation (INR)")
            
            delta_cost = budget_inr - estimated_cost_inr
            delta_color = "normal" if delta_cost >= 0 else "inverse"
            
            st.metric(
                "Estimated Final Cost", 
                f"₹{estimated_cost_inr:,.0f}", 
                delta=f"₹{delta_cost:,.0f} (Variance)",
                delta_color=delta_color
            )
            st.metric("Allocated Budget", f"₹{budget_inr:,.0f}")
            st.metric("Predicted CPI", f"{cpi:.2f}", delta="Needs > 1.0" if cpi < 1 else "Good")
        
        # --- Schedule Breakdown Section ---
        st.divider()
        st.subheader("📅 Schedule Breakdown & Deviation Analysis")
        
        schedule_data = explain_schedule_deviation(input_dict, max(0, schedule_deviation))
        
        # Schedule Timeline Metrics
        col_sch1, col_sch2, col_sch3 = st.columns(3)
        
        with col_sch1:
            st.metric("Base Schedule", f"{schedule_data['base_schedule_days']} days")
        with col_sch2:
            st.metric("Predicted Deviation", f"+{schedule_data['total_deviation_days']} days", delta_color="inverse")
        with col_sch3:
            st.metric("Final Estimated Schedule", f"{schedule_data['final_schedule_days']} days")
        
        # Deviation Breakdown Bar Chart
        breakdown_df = pd.DataFrame({
            'Factor': ['Code Churn', 'Requirement Changes', 'Team Experience', 'Team Size'],
            'Days Added': [
                schedule_data['churn_impact_days'],
                schedule_data['req_change_impact_days'],
                schedule_data['experience_impact_days'],
                schedule_data['team_size_impact_days']
            ]
        })
        
        fig_breakdown = px.bar(
            breakdown_df,
            x='Days Added',
            y='Factor',
            orientation='h',
            color='Days Added',
            color_continuous_scale='Reds',
            title='Deviation Breakdown by Factor'
        )
        fig_breakdown.update_layout(
            showlegend=False,
            height=250,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Deviation Explanations
        st.markdown("#### 🔍 Why the extra days?")
        for explanation in schedule_data['explanations']:
            st.markdown(f"- {explanation}")



# =============================================================================
# Risk Prediction Page
# =============================================================================
def render_prediction():
    st.markdown('<h1 class="main-header">🔮 Predict Project Risk</h1>', unsafe_allow_html=True)
    
    models = load_models()
    clf, label_encoder, reg, scaler, feature_cols = models
    
    if clf is None:
        st.error("⚠️ Models not found! Please run `python train_models.py` first to train the models.")
        st.code("python train_models.py", language="bash")
        return
    
    st.info("Enter your project parameters below to predict risk level and schedule deviation.")
    
    # Input Form
    col1, col2 = st.columns(2)
    
    with col1:
        team_size = st.slider("Team Size", min_value=3, max_value=15, value=8)
        budget_allocated = st.number_input(
            "Budget Allocated ($)", 
            min_value=10000, 
            max_value=500000, 
            value=50000,
            step=5000
        )
        req_changes = st.slider("Expected Requirement Changes", min_value=0, max_value=10, value=2)
    
    with col2:
        code_churn = st.slider("Code Churn (%)", min_value=0, max_value=50, value=15) / 100
        team_experience = st.slider(
            "Team Experience Score", 
            min_value=0.5, 
            max_value=1.5, 
            value=1.0,
            step=0.1,
            help="0.5 = Junior, 1.0 = Mid, 1.5 = Senior"
        )
    
    if st.button("🎯 Predict Risk", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[team_size, budget_allocated, req_changes, code_churn, team_experience]])
        input_scaled = scaler.transform(input_data)
        
        # Predictions
        risk_pred_encoded = clf.predict(input_scaled)[0]
        risk_pred = label_encoder.inverse_transform([risk_pred_encoded])[0]
        risk_proba = clf.predict_proba(input_scaled)[0]
        schedule_deviation = reg.predict(input_scaled)[0]
        
        # Calculate additional metrics
        hourly_rate = 60  # Assume average rate
        planned_hours = team_size * 6 * 10
        estimated_actual_hours = planned_hours * (1 + (req_changes * 0.05) + (code_churn * 0.2)) / team_experience
        estimated_cost = estimated_actual_hours * hourly_rate
        cpi = budget_allocated / estimated_cost if estimated_cost > 0 else 0
        
        # Technical debt impact
        defect_density = 5 + (code_churn * 20) - (team_experience * 2)
        debt_impact = calculate_technical_debt_impact(defect_density, code_churn, team_size, hourly_rate)
        
        st.divider()
        
        # Results
        st.subheader("📋 Prediction Results")
        
        # Alert based on risk level
        alert_class = f"alert-{risk_pred.lower()}"
        alert_icon = "🔴" if risk_pred == "High" else "🟡" if risk_pred == "Medium" else "🟢"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <h3>{alert_icon} Risk Level: {risk_pred}</h3>
            <p>Predicted Schedule Deviation: <strong>{max(0, schedule_deviation):.1f} days</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk Level", risk_pred)
        with col2:
            st.metric("Schedule Delay", f"{max(0, schedule_deviation):.1f} days")
        with col3:
            st.metric("Est. CPI", f"{cpi:.2f}")
        with col4:
            st.metric("Quality Risk", debt_impact.quality_risk_level)
        
        # Probability Chart
        st.subheader("Risk Probability Distribution")
        prob_df = pd.DataFrame({
            'Risk Level': label_encoder.classes_,
            'Probability': risk_proba
        })
        fig = px.bar(
            prob_df,
            x='Risk Level',
            y='Probability',
            color='Risk Level',
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
        )
        fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Debt Impact
        st.subheader("💸 Technical Debt Impact")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Rework Hours", f"{debt_impact.estimated_rework_hours:.0f} hrs")
        with col2:
            st.metric("Estimated Rework Cost", f"${debt_impact.estimated_rework_cost:,.0f}")
        with col3:
            st.metric("Quality Risk Score", f"{debt_impact.quality_risk_score:.0f}/100")


# =============================================================================
# Analytics Page
# =============================================================================
def render_analytics():
    st.markdown('<h1 class="main-header">📈 Historical Analytics</h1>', unsafe_allow_html=True)
    
    df = load_historical_data()
    
    # --- moved from Dashboard ---
    st.subheader("Project Portfolio Overview")
    
    # Summary Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Projects", len(df))
    
    with col2:
        high_risk = len(df[df['Risk_Level'] == 'High'])
        st.metric("High Risk", high_risk, delta=f"{high_risk/len(df)*100:.1f}%")
    
    with col3:
        avg_cpi = df['CPI'].mean()
        st.metric("Avg CPI", f"{avg_cpi:.2f}", delta="On Budget" if avg_cpi >= 1 else "Over Budget")
    
    with col4:
        avg_deviation = df['Schedule_Deviation_Days'].mean()
        st.metric("Avg Delay", f"{avg_deviation:.1f} days")
    
    with col5:
        # Show Budget in INR (approx conversion for display)
        total_budget_usd = df['Budget_Allocated'].sum()
        total_budget_inr = total_budget_usd * 83 / 1e7 # In Crores
        st.metric("Total Budget", f"₹{total_budget_inr:.1f} Cr")
    
    st.divider()
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = df['Risk_Level'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'},
            hole=0.4
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Schedule Deviation Distribution")
        fig = px.histogram(
            df,
            x='Schedule_Deviation_Days',
            nbins=30,
            color='Risk_Level',
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'},
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)

    # High Risk Projects Table
    st.markdown("### ⚠️ High Risk Projects History")
    high_risk_df = df[df['Risk_Level'] == 'High'].sort_values('Schedule_Deviation_Days', ascending=False).head(10)
    st.dataframe(
        high_risk_df[['Project_ID', 'Team_Size', 'Budget_Allocated', 'CPI', 'Schedule_Deviation_Days', 'Defect_Density']],
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Feature Correlations
    st.subheader("📊 Feature Correlations")
    
    numeric_cols = ['Team_Size', 'Budget_Allocated', 'Requirement_Changes', 'Code_Churn', 
                   'Team_Experience_Score', 'Defect_Density', 'CPI', 'Schedule_Deviation_Days']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# What-If Comparison Page (Interactive Feature)
# =============================================================================
def render_comparison():
    st.markdown('<h1 class="main-header">🔄 What-If Scenario Comparison</h1>', unsafe_allow_html=True)
    
    models = load_models()
    clf, label_encoder, reg, scaler, feature_cols = models
    
    if clf is None:
        st.error("⚠️ Models not found! Please run `python train_models.py` first.")
        return
    
    st.info("Compare two project scenarios side-by-side to see the impact of changes.")
    
    col_a, col_b = st.columns(2)
    
    # Scenario A
    with col_a:
        st.markdown("### 🅰️ Scenario A (Baseline)")
        team_a = st.slider("Team Size (A)", 3, 20, 8, key="team_a", help="Total number of members in the team")
        budget_a = st.number_input("Budget (₹)", 500000, 50000000, 4000000, step=100000, key="budget_a", format="%d", help="Total allocated budget in INR")
        req_a = st.slider("Req. Changes (A)", 0, 20, 3, key="req_a", help="Expected requirement changes")
        churn_a = st.slider("Code Churn % (A)", 0, 100, 15, key="churn_a", help="Percentage of code rewrite expected") / 100
        exp_a = st.slider("Experience (A)", 0.5, 2.0, 1.0, step=0.1, key="exp_a", help="Team experience score (0.5-2.0)")
        
    # Scenario B
    with col_b:
        st.markdown("### 🅱️ Scenario B (Modified)")
        team_b = st.slider("Team Size (B)", 3, 20, 10, key="team_b", help="Total number of members in the team")
        budget_b = st.number_input("Budget (₹)", 500000, 50000000, 5000000, step=100000, key="budget_b", format="%d", help="Total allocated budget in INR")
        req_b = st.slider("Req. Changes (B)", 0, 20, 2, key="req_b", help="Expected requirement changes")
        churn_b = st.slider("Code Churn % (B)", 0, 100, 10, key="churn_b", help="Percentage of code rewrite expected") / 100
        exp_b = st.slider("Experience (B)", 0.5, 2.0, 1.2, step=0.1, key="exp_b", help="Team experience score (0.5-2.0)")
    
    if st.button("⚖️ Compare Scenarios", type="primary", use_container_width=True):
        # Predictions for A
        input_a = np.array([[team_a, budget_a / 83.0, req_a, churn_a, exp_a]])
        input_a_scaled = scaler.transform(input_a)
        risk_a = label_encoder.inverse_transform([clf.predict(input_a_scaled)[0]])[0]
        proba_a = clf.predict_proba(input_a_scaled)[0]
        delay_a = max(0, reg.predict(input_a_scaled)[0])
        
        # Predictions for B
        input_b = np.array([[team_b, budget_b / 83.0, req_b, churn_b, exp_b]])
        input_b_scaled = scaler.transform(input_b)
        risk_b = label_encoder.inverse_transform([clf.predict(input_b_scaled)[0]])[0]
        proba_b = clf.predict_proba(input_b_scaled)[0]
        delay_b = max(0, reg.predict(input_b_scaled)[0])
        
        st.divider()
        st.subheader("📊 Comparison Results")
        
        res_a, res_b = st.columns(2)
        
        with res_a:
            icon_a = "🔴" if risk_a == "High" else "🟡" if risk_a == "Medium" else "🟢"
            st.markdown(f"#### {icon_a} Scenario A: **{risk_a}** Risk")
            st.metric("Schedule Delay", f"{delay_a:.1f} days")
            
            # Gauge Chart for A
            fig_a = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(proba_a) * 100,
                title={'text': f"Confidence ({risk_a})"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': get_risk_color(risk_a)}}
            ))
            fig_a.update_layout(height=250, margin=dict(t=50, b=10))
            st.plotly_chart(fig_a, use_container_width=True)
            
        with res_b:
            icon_b = "🔴" if risk_b == "High" else "🟡" if risk_b == "Medium" else "🟢"
            st.markdown(f"#### {icon_b} Scenario B: **{risk_b}** Risk")
            st.metric("Schedule Delay", f"{delay_b:.1f} days")
            
            # Gauge Chart for B
            fig_b = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(proba_b) * 100,
                title={'text': f"Confidence ({risk_b})"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': get_risk_color(risk_b)}}
            ))
            fig_b.update_layout(height=250, margin=dict(t=50, b=10))
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Improvement summary
        st.divider()
        delay_diff = delay_a - delay_b
        if delay_diff > 0:
            st.success(f"✅ Scenario B saves approximately **{delay_diff:.1f} days** compared to Scenario A.")
        elif delay_diff < 0:
            st.warning(f"⚠️ Scenario B adds approximately **{abs(delay_diff):.1f} days** compared to Scenario A.")
        else:
            st.info("Both scenarios have similar schedule predictions.")


# =============================================================================
# Main Routing
# =============================================================================
if page == "📊 Dashboard":
    render_dashboard()
elif page == "📈 Analytics":
    render_analytics()
elif page == "🔄 What-If Comparison":
    render_comparison()

# Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("**PRMS v2.0**")
st.sidebar.markdown("Predictive Risk Monitoring System")
st.sidebar.markdown("_Now with Explainable AI_")
