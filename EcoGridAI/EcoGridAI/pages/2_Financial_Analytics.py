import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from utils.financial_calculations import FinancialCalculator

st.set_page_config(page_title="Financial Analytics", page_icon="💰", layout="wide")

st.title("💰 Financial Analytics Dashboard")
st.markdown("---")

# Initialize financial calculator
calc = FinancialCalculator()

# Generate financial data
def generate_financial_data():
    """Generate comprehensive financial data for analysis"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    financial_data = []
    
    for i, date in enumerate(dates):
        # Energy generation and consumption
        solar_kwh = 25 + 15 * np.sin(i * 2 * np.pi / 30) + random.uniform(-5, 5)
        wind_kwh = 20 + 10 * np.sin((i + 7) * 2 * np.pi / 30) + random.uniform(-8, 8)
        consumption_kwh = 35 + 10 * np.sin((i + 3) * 2 * np.pi / 30) + random.uniform(-5, 5)
        
        # Pricing (varies with demand and time)
        base_rate = 0.12
        peak_multiplier = 1.5 if date.weekday() < 5 and 16 <= date.hour <= 20 else 1.0
        rate_per_kwh = base_rate * peak_multiplier + random.uniform(-0.02, 0.03)
        
        # Financial calculations
        generation_kwh = solar_kwh + wind_kwh
        net_kwh = generation_kwh - consumption_kwh
        
        # Costs and revenues
        if net_kwh > 0:
            # Surplus - selling to grid
            energy_cost = 0
            energy_revenue = net_kwh * rate_per_kwh * 0.8  # Feed-in tariff is typically lower
        else:
            # Deficit - buying from grid
            energy_cost = abs(net_kwh) * rate_per_kwh
            energy_revenue = 0
        
        # Additional costs (maintenance, etc.)
        maintenance_cost = random.uniform(0.5, 2.0)
        
        financial_data.append({
            'date': date,
            'solar_kwh': max(0, solar_kwh),
            'wind_kwh': max(0, wind_kwh),
            'consumption_kwh': consumption_kwh,
            'generation_kwh': generation_kwh,
            'net_kwh': net_kwh,
            'rate_per_kwh': rate_per_kwh,
            'energy_cost': energy_cost,
            'energy_revenue': energy_revenue,
            'maintenance_cost': maintenance_cost,
            'net_savings': energy_revenue - energy_cost - maintenance_cost
        })
    
    return pd.DataFrame(financial_data)

# Generate data
df_financial = generate_financial_data()

# Calculate key financial metrics
total_savings = df_financial['net_savings'].sum()
total_revenue = df_financial['energy_revenue'].sum()
total_costs = df_financial['energy_cost'].sum() + df_financial['maintenance_cost'].sum()
avg_daily_savings = df_financial['net_savings'].mean()
total_generation = df_financial['generation_kwh'].sum()
total_consumption = df_financial['consumption_kwh'].sum()

# ROI Calculations
initial_investment = 25000  # Assumed initial system cost
annual_savings = total_savings * 12  # Extrapolate monthly to annual
roi_years = initial_investment / annual_savings if annual_savings > 0 else float('inf')
roi_percentage = (annual_savings / initial_investment) * 100

# Key Financial Metrics
st.subheader("📊 Key Financial Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💵 Total Savings (30 days)", f"${total_savings:.2f}", f"${avg_daily_savings:.2f}/day")

with col2:
    st.metric("💰 Total Revenue", f"${total_revenue:.2f}", f"${(total_revenue/30):.2f}/day")

with col3:
    st.metric("💸 Total Costs", f"${total_costs:.2f}", f"${(total_costs/30):.2f}/day")

with col4:
    roi_color = "normal" if roi_years <= 10 else "inverse"
    st.metric("📈 ROI Period", f"{roi_years:.1f} years" if roi_years != float('inf') else "∞", 
             f"{roi_percentage:.1f}% annual", delta_color=roi_color)

# Financial Overview Charts
st.subheader("📈 Financial Performance Analysis")

# Create comprehensive financial dashboard
fig_financial = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Daily Savings/Costs', 'Cumulative Savings', 
                   'Energy Trading Revenue', 'Cost Breakdown'),
    specs=[[{"secondary_y": True}, {"secondary_y": False}],
           [{"secondary_y": False}, {"type": "pie"}]]
)

# Daily savings/costs
fig_financial.add_trace(
    go.Bar(x=df_financial['date'], y=df_financial['energy_revenue'], 
           name='Revenue', marker_color='#4CAF50'),
    row=1, col=1
)
fig_financial.add_trace(
    go.Bar(x=df_financial['date'], y=-df_financial['energy_cost'], 
           name='Costs', marker_color='#FF5722'),
    row=1, col=1
)
fig_financial.add_trace(
    go.Scatter(x=df_financial['date'], y=df_financial['net_savings'], 
              name='Net Savings', line=dict(color='#2E7D32', width=3)),
    row=1, col=1, secondary_y=True
)

# Cumulative savings
cumulative_savings = df_financial['net_savings'].cumsum()
fig_financial.add_trace(
    go.Scatter(x=df_financial['date'], y=cumulative_savings, 
              name='Cumulative Savings', line=dict(color='#1565C0', width=3),
              fill='tozeroy'),
    row=1, col=2
)

# Energy trading revenue breakdown
solar_revenue = df_financial['solar_kwh'] * df_financial['rate_per_kwh'] * 0.8
wind_revenue = df_financial['wind_kwh'] * df_financial['rate_per_kwh'] * 0.8

fig_financial.add_trace(
    go.Scatter(x=df_financial['date'], y=solar_revenue, 
              name='Solar Revenue', line=dict(color='#FF8F00', width=2),
              stackgroup='one'),
    row=2, col=1
)
fig_financial.add_trace(
    go.Scatter(x=df_financial['date'], y=wind_revenue, 
              name='Wind Revenue', line=dict(color='#1565C0', width=2),
              stackgroup='one'),
    row=2, col=1
)

# Cost breakdown pie chart
cost_breakdown = {
    'Grid Purchases': total_costs - df_financial['maintenance_cost'].sum(),
    'Maintenance': df_financial['maintenance_cost'].sum(),
    'System Fees': total_costs * 0.1
}

fig_financial.add_trace(
    go.Pie(labels=list(cost_breakdown.keys()), values=list(cost_breakdown.values()),
           marker_colors=['#FF5722', '#FF8F00', '#757575']),
    row=2, col=2
)

fig_financial.update_layout(height=800, showlegend=True, 
                           title_text="Comprehensive Financial Analysis")
st.plotly_chart(fig_financial, use_container_width=True)

# Advanced Financial Analytics
st.subheader("🔍 Advanced Financial Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 Energy Cost Analysis")
    
    # Cost per kWh analysis
    df_financial['cost_per_kwh_consumed'] = np.where(
        df_financial['consumption_kwh'] > 0,
        df_financial['energy_cost'] / df_financial['consumption_kwh'],
        0
    )
    
    fig_cost_kwh = go.Figure()
    fig_cost_kwh.add_trace(go.Scatter(
        x=df_financial['date'],
        y=df_financial['cost_per_kwh_consumed'],
        mode='lines+markers',
        name='Actual Cost per kWh',
        line=dict(color='#FF5722', width=2)
    ))
    fig_cost_kwh.add_trace(go.Scatter(
        x=df_financial['date'],
        y=df_financial['rate_per_kwh'],
        mode='lines',
        name='Grid Rate per kWh',
        line=dict(color='#757575', width=2, dash='dash')
    ))
    
    fig_cost_kwh.update_layout(
        title="Cost per kWh Analysis",
        xaxis_title="Date",
        yaxis_title="Cost ($/kWh)",
        height=400
    )
    st.plotly_chart(fig_cost_kwh, use_container_width=True)

with col2:
    st.subheader("📊 Budget Projections")
    
    # Monthly budget projection
    monthly_projection = pd.DataFrame({
        'Month': [f"Month {i+1}" for i in range(12)],
        'Projected_Savings': [avg_daily_savings * 30] * 12,
        'Cumulative_Savings': [avg_daily_savings * 30 * (i+1) for i in range(12)]
    })
    
    fig_projection = go.Figure()
    fig_projection.add_trace(go.Bar(
        x=monthly_projection['Month'],
        y=monthly_projection['Projected_Savings'],
        name='Monthly Savings',
        marker_color='#4CAF50'
    ))
    fig_projection.add_trace(go.Scatter(
        x=monthly_projection['Month'],
        y=monthly_projection['Cumulative_Savings'],
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='#1565C0', width=3),
        yaxis='y2'
    ))
    
    fig_projection.update_layout(
        title="12-Month Savings Projection",
        xaxis_title="Month",
        yaxis=dict(title="Monthly Savings ($)", side="left"),
        yaxis2=dict(title="Cumulative Savings ($)", side="right", overlaying="y"),
        height=400
    )
    st.plotly_chart(fig_projection, use_container_width=True)

# Investment Analysis
st.subheader("💼 Investment Analysis & ROI")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🎯 ROI Analysis")
    
    # ROI gauge chart
    fig_roi = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=roi_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Annual ROI %"},
        delta={'reference': 10, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 25]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 5], 'color': "#FF5722"},
                {'range': [5, 10], 'color': "#FF8F00"},
                {'range': [10, 25], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 15
            }
        }
    ))
    fig_roi.update_layout(height=300)
    st.plotly_chart(fig_roi, use_container_width=True)

with col2:
    st.markdown("#### 💰 Payback Analysis")
    
    # Payback period calculation
    payback_data = []
    cumulative_investment = initial_investment
    
    for year in range(1, 16):
        annual_return = annual_savings
        cumulative_investment -= annual_return
        payback_data.append({
            'Year': year,
            'Remaining_Investment': max(0, cumulative_investment),
            'Cumulative_Returns': annual_return * year
        })
    
    df_payback = pd.DataFrame(payback_data)
    
    fig_payback = go.Figure()
    fig_payback.add_trace(go.Scatter(
        x=df_payback['Year'],
        y=df_payback['Remaining_Investment'],
        name='Remaining Investment',
        line=dict(color='#FF5722', width=3)
    ))
    fig_payback.add_trace(go.Scatter(
        x=df_payback['Year'],
        y=df_payback['Cumulative_Returns'],
        name='Cumulative Returns',
        line=dict(color='#4CAF50', width=3)
    ))
    
    fig_payback.update_layout(
        title="Investment Payback Timeline",
        xaxis_title="Years",
        yaxis_title="Amount ($)",
        height=300
    )
    st.plotly_chart(fig_payback, use_container_width=True)

with col3:
    st.markdown("#### 📈 Performance Metrics")
    
    # Key performance indicators
    energy_independence = (total_generation / total_consumption) * 100
    cost_savings_rate = ((total_costs - total_savings) / total_costs) * 100 if total_costs > 0 else 0
    
    st.metric("🔋 Energy Independence", f"{energy_independence:.1f}%")
    st.metric("💲 Cost Savings Rate", f"{cost_savings_rate:.1f}%")
    st.metric("⚡ Avg Generation", f"{df_financial['generation_kwh'].mean():.1f} kWh/day")
    st.metric("🏠 Avg Consumption", f"{df_financial['consumption_kwh'].mean():.1f} kWh/day")

# Financial Recommendations
st.subheader("🎯 Smart Financial Recommendations")

recommendations = []

if roi_percentage < 8:
    recommendations.append(("📈 Consider system optimization", "warning", 
                          "ROI is below market average. Consider adding more capacity or optimizing usage patterns."))

if avg_daily_savings < 5:
    recommendations.append(("💰 Low daily savings", "info",
                          "Daily savings are low. Review energy consumption patterns during peak pricing."))

if energy_independence < 80:
    recommendations.append(("🔋 Increase energy independence", "info",
                          "Consider expanding renewable capacity to reduce grid dependency."))

if total_revenue > total_costs * 2:
    recommendations.append(("🌟 Excellent performance!", "success",
                          "Your system is performing exceptionally well with strong returns."))

if not recommendations:
    st.success("✅ Financial performance is optimal!")
else:
    for title, rec_type, description in recommendations:
        if rec_type == "warning":
            st.warning(f"**{title}**: {description}")
        elif rec_type == "info":
            st.info(f"**{title}**: {description}")
        else:
            st.success(f"**{title}**: {description}")

# Payment history simulation
st.subheader("📋 Recent Payment History")

payment_history = []
for i in range(10):
    date = datetime.now() - timedelta(days=i*3)
    amount = random.uniform(-50, 100)  # Negative = payment to you, positive = you pay
    payment_type = "Grid Sale" if amount < 0 else "Grid Purchase"
    
    payment_history.append({
        'Date': date.strftime('%Y-%m-%d'),
        'Type': payment_type,
        'Amount': f"${abs(amount):.2f}",
        'Status': 'Completed',
        'Balance_Impact': 'Credit' if amount < 0 else 'Debit'
    })

df_payments = pd.DataFrame(payment_history)
st.dataframe(df_payments, use_container_width=True)

# Export options
if st.button("📊 Export Financial Report"):
    st.success("Financial report exported! (Feature would generate PDF/Excel in production)")

if st.button("🔄 Refresh Financial Data"):
    st.rerun()
