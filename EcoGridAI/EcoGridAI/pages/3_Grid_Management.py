import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time

st.set_page_config(page_title="Grid Management", page_icon="🔌", layout="wide")

st.title("🔌 Smart Grid Management Dashboard")
st.markdown("---")

# Initialize session state for grid data
if 'grid_state' not in st.session_state:
    st.session_state.grid_state = {
        'load_balancing': 'auto',
        'demand_response': True,
        'trading_mode': 'optimized',
        'alerts': []
    }

def generate_grid_data():
    """Generate real-time grid management data"""
    now = datetime.now()
    
    # Grid zones simulation
    zones = ['North', 'South', 'East', 'West', 'Central']
    grid_data = []
    
    for zone in zones:
        # Base load with realistic variations
        base_load = random.uniform(80, 120)
        peak_factor = 1.3 if 17 <= now.hour <= 20 else 1.0  # Evening peak
        current_load = base_load * peak_factor * random.uniform(0.9, 1.1)
        
        # Generation capacity
        solar_capacity = random.uniform(40, 80) if 6 <= now.hour <= 18 else 0
        wind_capacity = random.uniform(30, 70)
        conventional_capacity = random.uniform(50, 100)
        
        total_generation = solar_capacity + wind_capacity + conventional_capacity
        
        # Grid stability metrics
        frequency = 50.0 + random.uniform(-0.2, 0.2)  # Hz
        voltage = 230 + random.uniform(-5, 5)  # V
        power_factor = random.uniform(0.85, 0.95)
        
        grid_data.append({
            'zone': zone,
            'current_load': current_load,
            'solar_generation': solar_capacity,
            'wind_generation': wind_capacity,
            'conventional_generation': conventional_capacity,
            'total_generation': total_generation,
            'net_balance': total_generation - current_load,
            'frequency': frequency,
            'voltage': voltage,
            'power_factor': power_factor,
            'stability_score': min(100, max(0, 
                100 - abs(frequency - 50) * 100 - abs(voltage - 230) * 2))
        })
    
    return pd.DataFrame(grid_data)

# Generate grid data
df_grid = generate_grid_data()

# Control Panel
st.subheader("⚙️ Grid Control Panel")

col1, col2, col3, col4 = st.columns(4)

with col1:
    load_balancing = st.selectbox(
        "🔄 Load Balancing",
        ["Auto", "Manual", "Scheduled"],
        index=0
    )
    st.session_state.grid_state['load_balancing'] = load_balancing.lower()

with col2:
    demand_response = st.toggle("📊 Demand Response", value=True)
    st.session_state.grid_state['demand_response'] = demand_response

with col3:
    trading_mode = st.selectbox(
        "💹 Trading Mode",
        ["Optimized", "Conservative", "Aggressive"],
        index=0
    )
    st.session_state.grid_state['trading_mode'] = trading_mode.lower()

with col4:
    emergency_mode = st.button("🚨 Emergency Override", type="secondary")
    if emergency_mode:
        st.warning("Emergency mode activated!")

# Real-time Grid Status
st.subheader("📊 Real-Time Grid Status")

# Key grid metrics
col1, col2, col3, col4 = st.columns(4)

total_load = df_grid['current_load'].sum()
total_generation = df_grid['total_generation'].sum()
avg_frequency = df_grid['frequency'].mean()
avg_stability = df_grid['stability_score'].mean()

with col1:
    st.metric("⚡ Total Grid Load", f"{total_load:.1f} MW", f"{random.uniform(-2, 2):.1f}")

with col2:
    st.metric("🔋 Total Generation", f"{total_generation:.1f} MW", f"{random.uniform(-3, 3):.1f}")

with col3:
    frequency_status = "🟢" if 49.8 <= avg_frequency <= 50.2 else "🟡"
    st.metric(f"{frequency_status} Grid Frequency", f"{avg_frequency:.2f} Hz", f"{random.uniform(-0.05, 0.05):.3f}")

with col4:
    stability_status = "🟢" if avg_stability > 95 else "🟡" if avg_stability > 85 else "🔴"
    st.metric(f"{stability_status} Grid Stability", f"{avg_stability:.1f}%", f"{random.uniform(-2, 2):.1f}")

# Grid Visualization
st.subheader("🗺️ Interactive Grid Network")

# Create grid network visualization
fig_network = go.Figure()

# Add grid zones as nodes
for i, zone in enumerate(df_grid['zone']):
    x = np.cos(2 * np.pi * i / len(df_grid))
    y = np.sin(2 * np.pi * i / len(df_grid))
    
    # Color based on load balance
    balance = df_grid.loc[df_grid['zone'] == zone, 'net_balance'].iloc[0]
    color = '#4CAF50' if balance > 0 else '#FF5722' if balance < -10 else '#FF8F00'
    size = 30 + abs(balance) * 0.5
    
    fig_network.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=size, color=color, opacity=0.8),
        text=f"{zone}<br>{balance:.1f}MW",
        textposition="middle center",
        name=zone,
        hovertemplate=f"<b>{zone} Zone</b><br>" +
                     f"Load: {df_grid.loc[df_grid['zone'] == zone, 'current_load'].iloc[0]:.1f} MW<br>" +
                     f"Generation: {df_grid.loc[df_grid['zone'] == zone, 'total_generation'].iloc[0]:.1f} MW<br>" +
                     f"Balance: {balance:.1f} MW<extra></extra>"
    ))

# Add connections between zones (simplified)
for i in range(len(df_grid)):
    for j in range(i+1, len(df_grid)):
        x1, y1 = np.cos(2 * np.pi * i / len(df_grid)), np.sin(2 * np.pi * i / len(df_grid))
        x2, y2 = np.cos(2 * np.pi * j / len(df_grid)), np.sin(2 * np.pi * j / len(df_grid))
        
        # Line thickness based on power flow
        flow = random.uniform(5, 20)
        fig_network.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(width=flow/4, color='rgba(100,100,100,0.5)'),
            showlegend=False,
            hoverinfo='skip'
        ))

fig_network.update_layout(
    title="Real-Time Grid Network Status",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=500,
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    annotations=[
        dict(
            text="🟢 Surplus Generation<br>🟡 Balanced<br>🔴 Deficit",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bordercolor="#c7c7c7", borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    ]
)

st.plotly_chart(fig_network, use_container_width=True)

# Detailed Grid Analytics
st.subheader("📈 Grid Performance Analytics")

# Create comprehensive grid dashboard
fig_grid = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Load vs Generation by Zone', 'Grid Stability Metrics', 
                   'Power Flow Distribution', 'Renewable Energy Mix'),
    specs=[[{"secondary_y": True}, {"secondary_y": True}],
           [{"type": "bar"}, {"type": "pie"}]]
)

# Load vs Generation by zone
fig_grid.add_trace(
    go.Bar(x=df_grid['zone'], y=df_grid['current_load'], 
           name='Current Load', marker_color='#FF5722'),
    row=1, col=1
)
fig_grid.add_trace(
    go.Bar(x=df_grid['zone'], y=df_grid['total_generation'], 
           name='Total Generation', marker_color='#4CAF50'),
    row=1, col=1
)

# Grid stability metrics
fig_grid.add_trace(
    go.Scatter(x=df_grid['zone'], y=df_grid['frequency'], 
              name='Frequency (Hz)', line=dict(color='#1565C0', width=3)),
    row=1, col=2
)
fig_grid.add_trace(
    go.Scatter(x=df_grid['zone'], y=df_grid['voltage'], 
              name='Voltage (V)', line=dict(color='#FF8F00', width=3)),
    row=1, col=2, secondary_y=True
)

# Power flow distribution
power_flows = ['Import', 'Export', 'Local']
flow_values = [
    df_grid[df_grid['net_balance'] < 0]['current_load'].sum(),
    df_grid[df_grid['net_balance'] > 0]['total_generation'].sum(),
    abs(df_grid['net_balance']).sum()
]

fig_grid.add_trace(
    go.Bar(x=power_flows, y=flow_values, 
           marker_color=['#FF5722', '#4CAF50', '#FF8F00']),
    row=2, col=1
)

# Renewable energy mix
total_renewable = df_grid['solar_generation'].sum() + df_grid['wind_generation'].sum()
total_conventional = df_grid['conventional_generation'].sum()

fig_grid.add_trace(
    go.Pie(labels=['Solar', 'Wind', 'Conventional'], 
           values=[df_grid['solar_generation'].sum(), 
                  df_grid['wind_generation'].sum(), 
                  total_conventional],
           marker_colors=['#FF8F00', '#1565C0', '#757575']),
    row=2, col=2
)

fig_grid.update_layout(height=800, showlegend=True, 
                      title_text="Comprehensive Grid Management Dashboard")
st.plotly_chart(fig_grid, use_container_width=True)

# AI-Powered Load Balancing
st.subheader("🤖 AI-Powered Load Balancing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎯 Current Optimization Status")
    
    # Simulated AI recommendations
    ai_recommendations = [
        "Redistribute 15 MW from North to South zone",
        "Increase wind generation by 8% in East zone", 
        "Schedule demand response in Central zone",
        "Optimize battery storage discharge timing"
    ]
    
    optimization_score = random.uniform(85, 98)
    
    st.metric("🎯 Optimization Score", f"{optimization_score:.1f}%")
    
    for i, rec in enumerate(ai_recommendations):
        status = "✅" if random.choice([True, False]) else "⏳"
        st.write(f"{status} {rec}")

with col2:
    st.markdown("#### 📊 Demand Response Program")
    
    # Demand response participation
    dr_participants = random.randint(150, 300)
    dr_capacity = random.uniform(20, 45)
    dr_savings = random.uniform(5, 15)
    
    st.metric("👥 Active Participants", f"{dr_participants}")
    st.metric("⚡ Available Capacity", f"{dr_capacity:.1f} MW")
    st.metric("💰 Cost Savings", f"${dr_savings:.1f}k/hour")
    
    # DR activation chart
    hours = list(range(24))
    dr_activation = [random.uniform(0, 100) if 16 <= h <= 20 else random.uniform(0, 30) for h in hours]
    
    fig_dr = go.Figure(data=go.Scatter(
        x=hours, y=dr_activation,
        mode='lines+markers',
        line=dict(color='#2E7D32', width=3),
        fill='tozeroy'
    ))
    fig_dr.update_layout(
        title="Demand Response Activation (24h)",
        xaxis_title="Hour",
        yaxis_title="Activation %",
        height=250
    )
    st.plotly_chart(fig_dr, use_container_width=True)

# Energy Trading Interface
st.subheader("💹 Smart Energy Trading")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 💱 Current Market Prices")
    
    # Market prices simulation
    market_data = {
        'Time Slot': ['Now', '+1h', '+2h', '+3h'],
        'Buy Price ($/MWh)': [45.5, 48.2, 52.1, 49.8],
        'Sell Price ($/MWh)': [42.8, 45.1, 48.9, 46.5],
        'Demand': ['Medium', 'High', 'Peak', 'Medium']
    }
    
    df_market = pd.DataFrame(market_data)
    st.dataframe(df_market, use_container_width=True)

with col2:
    st.markdown("#### 📈 Trading Performance")
    
    daily_trades = random.randint(15, 35)
    trading_profit = random.uniform(200, 800)
    success_rate = random.uniform(85, 95)
    
    st.metric("📊 Daily Trades", f"{daily_trades}")
    st.metric("💰 Trading Profit", f"${trading_profit:.2f}")
    st.metric("🎯 Success Rate", f"{success_rate:.1f}%")

with col3:
    st.markdown("#### 🔮 AI Predictions")
    
    # AI trading predictions
    predictions = [
        ("Price spike expected at 6 PM", "🔴"),
        ("Surplus generation in 2 hours", "🟢"),
        ("High demand in South zone", "🟡"),
        ("Optimal selling window: 3-4 PM", "🟢")
    ]
    
    for prediction, status in predictions:
        st.write(f"{status} {prediction}")

# Grid Alerts and Warnings
st.subheader("🚨 Grid Alerts & Monitoring")

# Simulate grid alerts
alerts = []

if avg_frequency < 49.9 or avg_frequency > 50.1:
    alerts.append(("⚠️ Frequency deviation detected", "warning", "Grid frequency outside normal range"))

if any(df_grid['stability_score'] < 90):
    alerts.append(("🔴 Stability issue in grid zones", "error", "Some zones showing instability"))

if total_load > total_generation * 0.95:
    alerts.append(("⚡ High load conditions", "warning", "Grid approaching capacity limits"))

if any(df_grid['net_balance'] < -20):
    alerts.append(("🔋 High import requirement", "info", "Some zones require significant imports"))

# Display alerts
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🚨 Active Alerts")
    if not alerts:
        st.success("✅ All systems operating normally")
    else:
        for alert_text, alert_type, description in alerts:
            if alert_type == "error":
                st.error(f"{alert_text}: {description}")
            elif alert_type == "warning":
                st.warning(f"{alert_text}: {description}")
            else:
                st.info(f"{alert_text}: {description}")

with col2:
    st.markdown("#### 📊 System Health")
    
    # System health metrics
    health_metrics = {
        'Grid Frequency': '🟢 Normal' if 49.9 <= avg_frequency <= 50.1 else '🟡 Caution',
        'Load Balance': '🟢 Optimal' if abs(total_generation - total_load) < 10 else '🟡 Managing',
        'Stability': '🟢 Stable' if avg_stability > 95 else '🟡 Monitoring',
        'Trading': '🟢 Active' if len(alerts) < 2 else '🟡 Limited'
    }
    
    for metric, status in health_metrics.items():
        st.write(f"**{metric}**: {status}")

# Control Actions
st.subheader("🎮 Grid Control Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Rebalance Loads"):
        st.success("Load rebalancing initiated")

with col2:
    if st.button("⚡ Activate DR"):
        st.success("Demand response activated")

with col3:
    if st.button("🔋 Deploy Storage"):
        st.success("Battery storage deployed")

with col4:
    if st.button("📊 Export Report"):
        st.success("Grid report generated")

# Auto-refresh
if st.sidebar.button("🔄 Refresh Grid Data"):
    st.rerun()

# Real-time updates
st.sidebar.markdown("**Real-time monitoring active**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
