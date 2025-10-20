import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Carbon Tracker", page_icon="🌱", layout="wide")

st.title("🌱 Carbon Footprint Tracker")
st.markdown("---")

def generate_carbon_data():
    """Generate comprehensive carbon tracking data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    
    carbon_data = []
    
    for i, date in enumerate(dates):
        # Energy consumption and generation
        solar_kwh = max(0, 25 + 15 * np.sin(i * 2 * np.pi / 30) + random.uniform(-5, 5))
        wind_kwh = max(0, 20 + 10 * np.sin((i + 7) * 2 * np.pi / 30) + random.uniform(-8, 8))
        grid_kwh = max(0, 15 + 8 * np.sin((i + 3) * 2 * np.pi / 30) + random.uniform(-5, 5))
        consumption_kwh = 40 + 10 * np.sin((i + 1) * 2 * np.pi / 30) + random.uniform(-5, 5)
        
        # Carbon intensity factors (kg CO2/kWh)
        grid_carbon_intensity = 0.45 + random.uniform(-0.05, 0.05)  # Grid electricity
        solar_carbon_intensity = 0.05  # Solar lifecycle emissions
        wind_carbon_intensity = 0.02   # Wind lifecycle emissions
        
        # Carbon calculations
        carbon_from_grid = grid_kwh * grid_carbon_intensity
        carbon_from_solar = solar_kwh * solar_carbon_intensity
        carbon_from_wind = wind_kwh * wind_carbon_intensity
        
        total_carbon = carbon_from_grid + carbon_from_solar + carbon_from_wind
        
        # Carbon savings from renewables (vs. all grid electricity)
        baseline_carbon = consumption_kwh * grid_carbon_intensity
        carbon_saved = baseline_carbon - total_carbon
        
        carbon_data.append({
            'date': date,
            'solar_kwh': solar_kwh,
            'wind_kwh': wind_kwh,
            'grid_kwh': grid_kwh,
            'consumption_kwh': consumption_kwh,
            'carbon_from_grid': carbon_from_grid,
            'carbon_from_solar': carbon_from_solar,
            'carbon_from_wind': carbon_from_wind,
            'total_carbon': total_carbon,
            'baseline_carbon': baseline_carbon,
            'carbon_saved': max(0, carbon_saved),
            'carbon_intensity': grid_carbon_intensity,
            'renewable_percentage': ((solar_kwh + wind_kwh) / consumption_kwh) * 100 if consumption_kwh > 0 else 0
        })
    
    return pd.DataFrame(carbon_data)

# Generate carbon data
df_carbon = generate_carbon_data()

# Key Carbon Metrics
st.subheader("🌍 Carbon Impact Overview")

col1, col2, col3, col4 = st.columns(4)

total_carbon_saved = df_carbon['carbon_saved'].sum()
avg_daily_savings = df_carbon['carbon_saved'].mean()
total_renewable_kwh = df_carbon['solar_kwh'].sum() + df_carbon['wind_kwh'].sum()
avg_carbon_intensity = df_carbon['total_carbon'].sum() / df_carbon['consumption_kwh'].sum()

with col1:
    st.metric("🌱 Total Carbon Saved", f"{total_carbon_saved:.1f} kg CO₂", f"{avg_daily_savings:.1f} kg/day")

with col2:
    st.metric("♻️ Renewable Energy", f"{total_renewable_kwh:.0f} kWh", f"{df_carbon['renewable_percentage'].mean():.1f}%")

with col3:
    st.metric("📉 Carbon Intensity", f"{avg_carbon_intensity:.3f} kg CO₂/kWh", f"-{random.uniform(5, 15):.1f}%")

with col4:
    # Carbon offset equivalent (trees)
    trees_equivalent = total_carbon_saved / 22  # Average tree absorbs ~22 kg CO2/year
    st.metric("🌳 Trees Equivalent", f"{trees_equivalent:.1f} trees", f"+{trees_equivalent/30:.1f}/month")

# Carbon Visualization Dashboard
st.subheader("📊 Carbon Footprint Analysis")

# Main carbon tracking charts
fig_carbon = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Daily Carbon Emissions by Source', 'Carbon Savings Over Time',
                   'Energy Source Carbon Impact', 'Carbon Intensity Trends'),
    specs=[[{"secondary_y": True}, {"secondary_y": False}],
           [{"type": "bar"}, {"secondary_y": True}]]
)

# Daily carbon emissions by source
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['carbon_from_grid'], 
              name='Grid Carbon', line=dict(color='#FF5722', width=2),
              stackgroup='one'),
    row=1, col=1
)
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['carbon_from_solar'], 
              name='Solar Carbon', line=dict(color='#FF8F00', width=2),
              stackgroup='one'),
    row=1, col=1
)
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['carbon_from_wind'], 
              name='Wind Carbon', line=dict(color='#1565C0', width=2),
              stackgroup='one'),
    row=1, col=1
)
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['baseline_carbon'], 
              name='Baseline (No Renewables)', line=dict(color='#757575', width=2, dash='dash')),
    row=1, col=1, secondary_y=True
)

# Carbon savings over time
cumulative_savings = df_carbon['carbon_saved'].cumsum()
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=cumulative_savings, 
              name='Cumulative Carbon Savings', line=dict(color='#4CAF50', width=3),
              fill='tozeroy'),
    row=1, col=2
)

# Energy source carbon impact
carbon_by_source = {
    'Solar': df_carbon['carbon_from_solar'].sum(),
    'Wind': df_carbon['carbon_from_wind'].sum(),  
    'Grid': df_carbon['carbon_from_grid'].sum()
}

fig_carbon.add_trace(
    go.Bar(x=list(carbon_by_source.keys()), y=list(carbon_by_source.values()),
           marker_color=['#FF8F00', '#1565C0', '#FF5722']),
    row=2, col=1
)

# Carbon intensity trends
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['carbon_intensity'], 
              name='Grid Carbon Intensity', line=dict(color='#FF5722', width=2)),
    row=2, col=2
)
fig_carbon.add_trace(
    go.Scatter(x=df_carbon['date'], y=df_carbon['total_carbon']/df_carbon['consumption_kwh'], 
              name='Your Carbon Intensity', line=dict(color='#4CAF50', width=3)),
    row=2, col=2, secondary_y=True
)

fig_carbon.update_layout(height=800, showlegend=True, 
                        title_text="Comprehensive Carbon Footprint Analysis")
st.plotly_chart(fig_carbon, use_container_width=True)

# Carbon Goals and Targets
st.subheader("🎯 Carbon Reduction Goals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Progress Towards Goals")
    
    # Set carbon reduction goals
    annual_target = st.number_input("Annual Carbon Reduction Target (kg CO₂)", 
                                   min_value=100, max_value=5000, value=1500)
    
    # Calculate progress
    projected_annual_savings = total_carbon_saved * 12  # Extrapolate monthly to annual
    progress_percentage = min(100, (projected_annual_savings / annual_target) * 100)
    
    # Progress visualization
    fig_progress = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=progress_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Goal Progress (%)"},
        delta={'reference': 100, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 150]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 50], 'color': "#FF5722"},
                {'range': [50, 80], 'color': "#FF8F00"},
                {'range': [80, 150], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    fig_progress.update_layout(height=300)
    st.plotly_chart(fig_progress, use_container_width=True)
    
    if progress_percentage >= 100:
        st.success(f"🎉 Congratulations! You're exceeding your carbon reduction goal by {progress_percentage-100:.1f}%")
    elif progress_percentage >= 80:
        st.info(f"🌟 Great progress! You're at {progress_percentage:.1f}% of your goal")
    else:
        st.warning(f"📊 {progress_percentage:.1f}% of goal achieved. Consider additional renewable sources")

with col2:
    st.markdown("#### 🌍 Environmental Impact")
    
    # Environmental impact metrics
    st.metric("🌳 Forest Area Preserved", f"{total_carbon_saved * 0.001:.2f} hectares")
    st.metric("🚗 Car Miles Offset", f"{total_carbon_saved * 2.3:.0f} miles")
    st.metric("⛽ Gas Gallons Saved", f"{total_carbon_saved / 8.9:.1f} gallons")
    st.metric("🏠 Home Days Powered", f"{total_renewable_kwh / 30:.0f} days")
    
    # Carbon trend analysis
    recent_trend = df_carbon['carbon_saved'].tail(7).mean() - df_carbon['carbon_saved'].head(7).mean()
    trend_direction = "📈 Improving" if recent_trend > 0 else "📉 Declining" if recent_trend < -0.5 else "➡️ Stable"
    
    st.markdown("#### 📊 Recent Trends")
    st.write(f"**Carbon Savings Trend**: {trend_direction}")
    st.write(f"**Weekly Change**: {recent_trend:+.2f} kg CO₂/day")

# Carbon Reduction Recommendations
st.subheader("💡 Carbon Reduction Recommendations")

recommendations = []

# Generate recommendations based on data analysis
avg_grid_percentage = df_carbon['grid_kwh'].sum() / df_carbon['consumption_kwh'].sum() * 100
avg_renewable_percentage = df_carbon['renewable_percentage'].mean()

if avg_grid_percentage > 30:
    recommendations.append({
        "category": "🔋 Energy Storage",
        "priority": "High",
        "description": "Add battery storage to reduce grid dependency during peak carbon intensity hours",
        "potential_savings": f"{avg_grid_percentage * 0.4:.0f} kg CO₂/month"
    })

if avg_renewable_percentage < 70:
    recommendations.append({
        "category": "☀️ Solar Expansion", 
        "priority": "Medium",
        "description": "Consider additional solar panels to increase renewable energy generation",
        "potential_savings": f"{(70 - avg_renewable_percentage) * 2:.0f} kg CO₂/month"
    })

if df_carbon['carbon_intensity'].mean() > 0.4:
    recommendations.append({
        "category": "⚡ Smart Usage",
        "priority": "Medium", 
        "description": "Shift energy usage to times with lower grid carbon intensity",
        "potential_savings": f"{df_carbon['carbon_intensity'].mean() * 100:.0f} kg CO₂/month"
    })

# Always include energy efficiency
recommendations.append({
    "category": "💡 Energy Efficiency",
    "priority": "Low",
    "description": "Implement energy-efficient appliances and LED lighting",
    "potential_savings": f"{df_carbon['consumption_kwh'].mean() * 0.15 * 0.45:.0f} kg CO₂/month"
})

# Display recommendations
for i, rec in enumerate(recommendations):
    priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[rec["priority"]]
    
    with st.expander(f"{priority_color} {rec['category']} - {rec['priority']} Priority"):
        st.write(f"**Recommendation**: {rec['description']}")
        st.write(f"**Potential Carbon Savings**: {rec['potential_savings']}")
        
        if st.button(f"Learn More", key=f"learn_more_{i}"):
            st.info("Detailed implementation guide would be provided here")

# Carbon Offset Options
st.subheader("🌍 Carbon Offset Opportunities")

col1, col2, col3 = st.columns(3)

remaining_carbon = max(0, df_carbon['total_carbon'].sum() - total_carbon_saved)

with col1:
    st.markdown("#### 🌳 Forest Projects")
    forest_cost = remaining_carbon * 0.15  # $0.15 per kg CO₂
    st.write(f"Offset remaining {remaining_carbon:.1f} kg CO₂")
    st.write(f"**Cost**: ${forest_cost:.2f}")
    st.write("**Impact**: Reforestation in developing countries")
    if st.button("🌳 Purchase Forest Offsets"):
        st.success("Forest offset purchase initiated!")

with col2:
    st.markdown("#### ⚡ Renewable Energy")
    renewable_cost = remaining_carbon * 0.12  # $0.12 per kg CO₂
    st.write(f"Offset remaining {remaining_carbon:.1f} kg CO₂")
    st.write(f"**Cost**: ${renewable_cost:.2f}")
    st.write("**Impact**: Support wind/solar projects")
    if st.button("⚡ Purchase Renewable Offsets"):
        st.success("Renewable offset purchase initiated!")

with col3:
    st.markdown("#### 🏭 Carbon Capture")
    capture_cost = remaining_carbon * 0.25  # $0.25 per kg CO₂
    st.write(f"Offset remaining {remaining_carbon:.1f} kg CO₂")
    st.write(f"**Cost**: ${capture_cost:.2f}")
    st.write("**Impact**: Direct air capture technology")
    if st.button("🏭 Purchase Capture Offsets"):
        st.success("Carbon capture offset purchase initiated!")

# Certification and Reporting
st.subheader("📋 Carbon Certification & Reporting")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 Carbon Achievements")
    
    achievements = []
    
    if total_carbon_saved > 100:
        achievements.append("🥉 Carbon Saver - 100+ kg CO₂ reduced")
    if total_carbon_saved > 500:
        achievements.append("🥈 Climate Champion - 500+ kg CO₂ reduced") 
    if total_carbon_saved > 1000:
        achievements.append("🥇 Environmental Hero - 1000+ kg CO₂ reduced")
    if avg_renewable_percentage > 80:
        achievements.append("⚡ Renewable Leader - 80%+ renewable energy")
    
    if achievements:
        for achievement in achievements:
            st.success(achievement)
    else:
        st.info("Complete carbon reduction goals to earn achievements!")

with col2:
    st.markdown("#### 📊 Reporting & Verification")
    
    if st.button("📄 Generate Carbon Report"):
        st.success("Carbon footprint report generated!")
        
        # Sample report summary
        report_data = {
            'Metric': ['Total Carbon Footprint', 'Carbon Saved', 'Renewable Energy %', 
                      'Carbon Intensity', 'Trees Equivalent'],
            'Value': [f"{df_carbon['total_carbon'].sum():.1f} kg CO₂",
                     f"{total_carbon_saved:.1f} kg CO₂", 
                     f"{avg_renewable_percentage:.1f}%",
                     f"{avg_carbon_intensity:.3f} kg CO₂/kWh",
                     f"{trees_equivalent:.1f} trees"]
        }
        
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    if st.button("🔍 Third-Party Verification"):
        st.info("Third-party verification process initiated. Results available in 5-7 business days.")
    
    if st.button("📤 Share Results"):
        st.success("Carbon reduction results shared to social media!")

# Real-time Carbon Monitoring
st.subheader("⏱️ Real-Time Carbon Monitoring")

# Current carbon metrics
current_hour = datetime.now().hour
current_carbon_intensity = 0.45 + 0.1 * np.sin(current_hour * np.pi / 12)
current_renewable_mix = random.uniform(60, 90)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌐 Current Grid Carbon Intensity", 
             f"{current_carbon_intensity:.3f} kg CO₂/kWh",
             f"{random.uniform(-10, 10):+.1f}%")

with col2:
    status = "🟢 Low" if current_carbon_intensity < 0.4 else "🟡 Medium" if current_carbon_intensity < 0.5 else "🔴 High"
    st.metric("📊 Carbon Status", status)

with col3:
    st.metric("♻️ Current Renewable Mix", 
             f"{current_renewable_mix:.1f}%",
             f"{random.uniform(-2, 5):+.1f}%")

# Live carbon intensity chart
st.markdown("#### 📈 24-Hour Carbon Intensity Forecast")

hours = list(range(24))
carbon_forecast = [0.45 + 0.1 * np.sin((h + random.uniform(-1, 1)) * np.pi / 12) + 
                  random.uniform(-0.05, 0.05) for h in hours]

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=hours, y=carbon_forecast,
    mode='lines+markers',
    name='Carbon Intensity',
    line=dict(color='#FF5722', width=3),
    fill='tozeroy'
))

# Add current hour marker
fig_forecast.add_vline(x=current_hour, line_dash="dash", line_color="green",
                      annotation_text="Now")

fig_forecast.update_layout(
    title="Grid Carbon Intensity Forecast",
    xaxis_title="Hour of Day", 
    yaxis_title="kg CO₂/kWh",
    height=300
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Carbon alerts
if current_carbon_intensity > 0.5:
    st.error("🚨 **High Carbon Alert**: Grid carbon intensity is high. Consider reducing usage or switching to battery power.")
elif current_carbon_intensity < 0.35:
    st.success("✅ **Low Carbon Period**: Great time for energy-intensive activities!")
else:
    st.info("ℹ️ **Moderate Carbon Period**: Normal grid carbon intensity.")

if st.button("🔄 Refresh Carbon Data"):
    st.rerun()
