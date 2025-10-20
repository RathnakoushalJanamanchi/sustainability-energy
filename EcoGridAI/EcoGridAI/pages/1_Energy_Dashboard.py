import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Energy Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ Energy Dashboard")
st.markdown("---")

# Initialize session state for data persistence
if 'energy_data' not in st.session_state:
    st.session_state.energy_data = []

def generate_energy_data():
    """Generate realistic energy data for visualization"""
    timestamps = []
    solar_data = []
    wind_data = []
    consumption_data = []
    battery_data = []
    
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(144):  # 24 hours * 6 (10-minute intervals)
        timestamp = base_time + timedelta(minutes=i*10)
        hour = timestamp.hour
        
        # Solar generation (peaks at noon)
        solar = max(0, 100 * np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
        solar += random.uniform(-10, 10)
        
        # Wind generation (more consistent but variable)
        wind = 45 + 25 * np.sin(hour * np.pi / 12) + random.uniform(-15, 15)
        
        # Consumption (peaks in evening)
        base_consumption = 40 + 30 * np.sin((hour - 8) * np.pi / 16)
        consumption = max(20, base_consumption + random.uniform(-10, 10))
        
        # Battery level simulation
        generation = solar + wind
        net_flow = generation - consumption
        battery_change = net_flow * 0.1
        if i == 0:
            battery = 60
        else:
            battery = max(0, min(100, battery_data[-1] + battery_change))
        
        timestamps.append(timestamp)
        solar_data.append(max(0, solar))
        wind_data.append(max(0, wind))
        consumption_data.append(consumption)
        battery_data.append(battery)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'solar': solar_data,
        'wind': wind_data,
        'consumption': consumption_data,
        'battery': battery_data,
        'total_generation': [s + w for s, w in zip(solar_data, wind_data)]
    })

# Generate or update data
df = generate_energy_data()

# Key metrics
col1, col2, col3, col4 = st.columns(4)

current_solar = df['solar'].iloc[-1]
current_wind = df['wind'].iloc[-1]
current_consumption = df['consumption'].iloc[-1]
current_battery = df['battery'].iloc[-1]

with col1:
    st.metric("🌞 Solar Generation", f"{current_solar:.1f} kW", f"{df['solar'].iloc[-1] - df['solar'].iloc[-2]:.1f}")

with col2:
    st.metric("💨 Wind Generation", f"{current_wind:.1f} kW", f"{df['wind'].iloc[-1] - df['wind'].iloc[-2]:.1f}")

with col3:
    st.metric("⚡ Consumption", f"{current_consumption:.1f} kW", f"{df['consumption'].iloc[-1] - df['consumption'].iloc[-2]:.1f}")

with col4:
    st.metric("🔋 Battery", f"{current_battery:.1f}%", f"{df['battery'].iloc[-1] - df['battery'].iloc[-2]:.1f}")

# Main energy flow visualization
st.subheader("🔄 24-Hour Energy Flow Analysis")

fig_main = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Generation vs Consumption', 'Battery Level Over Time', 
                   'Energy Sources Breakdown', 'Net Energy Balance'),
    specs=[[{"secondary_y": True}, {"secondary_y": False}],
           [{"type": "pie"}, {"secondary_y": True}]]
)

# Generation vs Consumption
fig_main.add_trace(
    go.Scatter(x=df['timestamp'], y=df['solar'], name='Solar', 
              line=dict(color='#FF8F00', width=2), fill='tonexty'),
    row=1, col=1
)
fig_main.add_trace(
    go.Scatter(x=df['timestamp'], y=df['wind'], name='Wind',
              line=dict(color='#1565C0', width=2), fill='tonexty'),
    row=1, col=1
)
fig_main.add_trace(
    go.Scatter(x=df['timestamp'], y=df['consumption'], name='Consumption',
              line=dict(color='#2E7D32', width=3, dash='dash')),
    row=1, col=1
)

# Battery level
fig_main.add_trace(
    go.Scatter(x=df['timestamp'], y=df['battery'], name='Battery Level',
              line=dict(color='#4CAF50', width=3), fill='tozeroy'),
    row=1, col=2
)

# Energy sources pie chart
total_solar = df['solar'].sum()
total_wind = df['wind'].sum()
fig_main.add_trace(
    go.Pie(labels=['Solar', 'Wind'], values=[total_solar, total_wind],
           marker_colors=['#FF8F00', '#1565C0']),
    row=2, col=1
)

# Net energy balance
net_balance = df['total_generation'] - df['consumption']
fig_main.add_trace(
    go.Scatter(x=df['timestamp'], y=net_balance, name='Net Balance',
              line=dict(color='#FF5722', width=2), fill='tozeroy'),
    row=2, col=2
)

fig_main.update_layout(height=800, showlegend=True, title_text="Comprehensive Energy Analysis Dashboard")
st.plotly_chart(fig_main, use_container_width=True)

# Real-time animated Sankey diagram
st.subheader("🌊 Real-Time Energy Flow (Sankey Diagram)")

# Calculate current flows
solar_to_consumption = min(current_solar, current_consumption * 0.4)
wind_to_consumption = min(current_wind, current_consumption * 0.4)
grid_to_consumption = max(0, current_consumption - solar_to_consumption - wind_to_consumption)
solar_to_battery = max(0, current_solar - solar_to_consumption) * 0.6
wind_to_battery = max(0, current_wind - wind_to_consumption) * 0.6
solar_to_grid = max(0, current_solar - solar_to_consumption - solar_to_battery)
wind_to_grid = max(0, current_wind - wind_to_consumption - wind_to_battery)

sankey_fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["Solar Panels", "Wind Turbines", "Grid Import", "Battery Charge",
               "Home Consumption", "Grid Export", "Battery Storage"],
        color=["#FF8F00", "#1565C0", "#757575", "#4CAF50", 
               "#2E7D32", "#FF5722", "#4CAF50"],
        x=[0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        y=[0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6]
    ),
    link=dict(
        source=[0, 1, 2, 3, 0, 1, 0, 1],
        target=[4, 4, 4, 4, 6, 6, 5, 5],
        value=[solar_to_consumption, wind_to_consumption, grid_to_consumption, 
               current_battery * 0.1, solar_to_battery, wind_to_battery,
               solar_to_grid, wind_to_grid],
        color=["rgba(255,143,0,0.4)", "rgba(21,101,192,0.4)", "rgba(117,117,117,0.4)",
               "rgba(76,175,80,0.4)", "rgba(255,143,0,0.2)", "rgba(21,101,192,0.2)",
               "rgba(255,87,34,0.4)", "rgba(255,87,34,0.4)"]
    )
)])

sankey_fig.update_layout(
    title_text="Current Energy Flow Distribution",
    font_size=12,
    height=500,
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(sankey_fig, use_container_width=True)

# Energy efficiency metrics
st.subheader("📊 Energy Efficiency Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    efficiency = (df['total_generation'].sum() / df['consumption'].sum()) * 100
    st.metric("⚡ Energy Self-Sufficiency", f"{efficiency:.1f}%")
    
    # Gauge chart for efficiency
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=efficiency,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Self-Sufficiency %"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 150]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 50], 'color': "#FF5722"},
                {'range': [50, 80], 'color': "#FF8F00"},
                {'range': [80, 150], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    peak_demand = df['consumption'].max()
    avg_demand = df['consumption'].mean()
    load_factor = (avg_demand / peak_demand) * 100
    st.metric("📈 Load Factor", f"{load_factor:.1f}%")
    
    # Bar chart for demand analysis
    hourly_avg = df.groupby(df['timestamp'].dt.hour).agg({
        'consumption': 'mean',
        'total_generation': 'mean'
    }).reset_index()
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_avg['timestamp'],
        y=hourly_avg['consumption'],
        name='Avg Consumption',
        marker_color='#2E7D32'
    ))
    fig_hourly.add_trace(go.Bar(
        x=hourly_avg['timestamp'],
        y=hourly_avg['total_generation'],
        name='Avg Generation',
        marker_color='#FF8F00'
    ))
    fig_hourly.update_layout(title="Hourly Energy Profile", height=300)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col3:
    renewable_percentage = (df['total_generation'].sum() / 
                          (df['total_generation'].sum() + df['consumption'].sum() - df['total_generation'].sum())) * 100
    st.metric("🌱 Renewable Energy %", f"{renewable_percentage:.1f}%")
    
    # Donut chart for energy mix
    grid_import = df['consumption'].sum() - df['total_generation'].sum()
    if grid_import < 0:
        grid_import = 0
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Solar', 'Wind', 'Grid'],
        values=[total_solar, total_wind, grid_import],
        hole=0.3,
        marker_colors=['#FF8F00', '#1565C0', '#757575']
    )])
    fig_donut.update_layout(title="Energy Mix", height=300)
    st.plotly_chart(fig_donut, use_container_width=True)

# Performance alerts and recommendations
st.subheader("🚨 Smart Alerts & Recommendations")

alerts = []

if efficiency < 70:
    alerts.append(("⚠️ Low self-sufficiency", "warning", "Consider adding more renewable capacity"))

if current_battery < 30:
    alerts.append(("🔋 Low battery level", "error", "Battery level is critically low"))

if current_consumption > current_solar + current_wind:
    alerts.append(("⚡ High grid dependency", "info", "Currently importing from grid"))

if load_factor < 60:
    alerts.append(("📊 Poor load factor", "warning", "Consider load balancing strategies"))

if not alerts:
    st.success("✅ All systems operating optimally!")
else:
    for alert_text, alert_type, recommendation in alerts:
        if alert_type == "error":
            st.error(f"{alert_text}: {recommendation}")
        elif alert_type == "warning":
            st.warning(f"{alert_text}: {recommendation}")
        else:
            st.info(f"{alert_text}: {recommendation}")

# Auto-refresh
if st.button("🔄 Refresh Dashboard"):
    st.rerun()
