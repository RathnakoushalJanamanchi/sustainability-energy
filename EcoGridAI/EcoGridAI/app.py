import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import random
from utils.ai_assistant import EnergyAI
from database.db_operations import get_db

# Page configuration
st.set_page_config(
    page_title="Sustainable Energy Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database connection
if 'db' not in st.session_state:
    try:
        st.session_state.db = get_db()
        st.session_state.db_connected = True
    except Exception as e:
        st.session_state.db_connected = False
        print(f"Database connection error: {e}")

# Initialize AI Assistant
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = EnergyAI()

# Custom CSS for animated styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #1565C0 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .energy-flow {
        animation: pulse 2s infinite;
    }
    
    .status-indicator {
        animation: blink 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .financial-widget {
        background: linear-gradient(135deg, #FF8F00 0%, #F57C00 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    
    .ai-widget {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main animated header
st.markdown("""
<div class="main-header">
    <h1>⚡ Sustainable Energy Optimization Platform</h1>
    <p>Real-Time AI-Powered Energy Management & Financial Analytics</p>
    <p style="font-size: 14px; opacity: 0.9;">Live Updates • Smart Trading • Carbon Tracking • AI Assistance</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🌱 Energy Control Center")
st.sidebar.markdown("---")

# Real-time data simulation function
def generate_real_time_data():
    """Generate realistic real-time energy data with animations"""
    current_time = datetime.now()
    
    # Solar generation (varies with time of day)
    hour = current_time.hour
    solar_factor = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
    solar_generation = solar_factor * (85 + random.uniform(-12, 12))
    
    # Wind generation (more variable)
    wind_generation = 45 + random.uniform(-18, 25)
    
    # Grid consumption with realistic patterns
    base_consumption = 65 + 25 * np.sin((hour - 8) * np.pi / 12)
    consumption = base_consumption + random.uniform(-12, 12)
    
    # Battery storage
    battery_level = 70 + random.uniform(-8, 8)
    
    # Dynamic pricing based on demand
    base_price = 0.12
    demand_factor = consumption / 100
    time_factor = 1.6 if 17 <= hour <= 20 else 0.75 if 22 <= hour <= 6 else 1.0
    current_price = base_price * time_factor * (1 + demand_factor * 0.3) + random.uniform(-0.015, 0.015)
    
    # Carbon intensity
    carbon_intensity = 0.45 - (solar_generation + wind_generation) / 300 + random.uniform(-0.05, 0.05)
    
    total_generation = solar_generation + wind_generation
    grid_import = max(0, consumption - total_generation)
    grid_export = max(0, total_generation - consumption)
    
    return {
        'timestamp': current_time,
        'solar_generation': max(0, solar_generation),
        'wind_generation': max(0, wind_generation),
        'total_generation': total_generation,
        'consumption': max(0, consumption),
        'battery_level': max(0, min(100, battery_level)),
        'grid_import': grid_import,
        'grid_export': grid_export,
        'price_per_kwh': max(0.05, current_price),
        'carbon_intensity': max(0.1, carbon_intensity),
        'trading_opportunity': grid_export > 10 and current_price > 0.15
    }

# Initialize or update data history
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'financial_history' not in st.session_state:
    st.session_state.financial_history = {
        'total_revenue': 0,
        'total_costs': 0,
        'total_savings': 0,
        'transactions': []
    }

# Generate and store real-time data
current_data = generate_real_time_data()
st.session_state.data_history.append(current_data)

# Log to database if connected
if st.session_state.get('db_connected', False):
    try:
        st.session_state.db.log_energy_reading(current_data)
    except Exception as e:
        print(f"Error logging to database: {e}")

# Calculate financial metrics
energy_cost = current_data['grid_import'] * current_data['price_per_kwh'] / 60  # Per minute
energy_revenue = current_data['grid_export'] * current_data['price_per_kwh'] * 0.85 / 60  # Feed-in tariff
net_savings = energy_revenue - energy_cost

st.session_state.financial_history['total_costs'] += energy_cost
st.session_state.financial_history['total_revenue'] += energy_revenue
st.session_state.financial_history['total_savings'] += net_savings

# Log transactions to database
if st.session_state.get('db_connected', False) and (energy_cost > 0.01 or energy_revenue > 0.01):
    try:
        if energy_cost > 0.01:
            st.session_state.db.create_transaction(
                transaction_type='purchase',
                amount_usd=-energy_cost,
                description=f"Grid import: {current_data['grid_import']:.2f} kW",
                energy_kwh=current_data['grid_import'] / 60,
                price_per_kwh=current_data['price_per_kwh']
            )
        if energy_revenue > 0.01:
            st.session_state.db.create_transaction(
                transaction_type='sale',
                amount_usd=energy_revenue,
                description=f"Grid export: {current_data['grid_export']:.2f} kW",
                energy_kwh=current_data['grid_export'] / 60,
                price_per_kwh=current_data['price_per_kwh'] * 0.85
            )
    except Exception as e:
        print(f"Error logging transactions: {e}")

# Keep only last 120 data points (10 minutes at 5-second intervals)
if len(st.session_state.data_history) > 120:
    st.session_state.data_history = st.session_state.data_history[-120:]

# Top-level Real-Time Metrics with Animation
st.markdown("### ⚡ Live Energy Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_solar = current_data['solar_generation'] - st.session_state.data_history[-2]['solar_generation'] if len(st.session_state.data_history) > 1 else 0
    st.metric(
        "🌞 Solar Power",
        f"{current_data['solar_generation']:.1f} kW",
        f"{delta_solar:+.1f} kW",
        delta_color="normal"
    )

with col2:
    delta_wind = current_data['wind_generation'] - st.session_state.data_history[-2]['wind_generation'] if len(st.session_state.data_history) > 1 else 0
    st.metric(
        "💨 Wind Power", 
        f"{current_data['wind_generation']:.1f} kW",
        f"{delta_wind:+.1f} kW",
        delta_color="normal"
    )

with col3:
    delta_consumption = current_data['consumption'] - st.session_state.data_history[-2]['consumption'] if len(st.session_state.data_history) > 1 else 0
    st.metric(
        "⚡ Usage",
        f"{current_data['consumption']:.1f} kW",
        f"{delta_consumption:+.1f} kW",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "🔋 Battery",
        f"{current_data['battery_level']:.0f}%",
        f"{random.uniform(-1.5, 1.5):+.1f}%",
        delta_color="normal"
    )

with col5:
    price_status = "🔴 Peak" if current_data['price_per_kwh'] > 0.15 else "🟢 Low" if current_data['price_per_kwh'] < 0.10 else "🟡 Normal"
    st.metric(
        "💰 Energy Price",
        f"${current_data['price_per_kwh']:.3f}/kWh",
        price_status
    )

st.markdown("---")

# Real-Time Animated Energy Flow Visualization
st.markdown("### 🔄 Live Energy Flow Animation")

# Enhanced Sankey diagram with real-time data
sources = []
targets = []
values = []
colors = []
labels = ["☀️ Solar", "💨 Wind", "🔋 Battery", "🏠 Home", "🔌 Grid Export", "⚡ Grid Import"]

solar_to_home = min(current_data['solar_generation'], current_data['consumption'] * 0.5)
wind_to_home = min(current_data['wind_generation'], current_data['consumption'] * 0.4)
battery_to_home = max(0, current_data['consumption'] - solar_to_home - wind_to_home - current_data['grid_import']) * 0.3

# Solar flows
if solar_to_home > 0:
    sources.append(0); targets.append(3); values.append(solar_to_home)
    colors.append("rgba(255,143,0,0.5)")
if current_data['solar_generation'] > solar_to_home:
    excess_solar = current_data['solar_generation'] - solar_to_home
    if excess_solar > 5:
        sources.append(0); targets.append(4); values.append(excess_solar * 0.7)
        colors.append("rgba(255,143,0,0.3)")

# Wind flows
if wind_to_home > 0:
    sources.append(1); targets.append(3); values.append(wind_to_home)
    colors.append("rgba(21,101,192,0.5)")
if current_data['wind_generation'] > wind_to_home:
    excess_wind = current_data['wind_generation'] - wind_to_home
    if excess_wind > 5:
        sources.append(1); targets.append(4); values.append(excess_wind * 0.7)
        colors.append("rgba(21,101,192,0.3)")

# Battery flows
if battery_to_home > 0:
    sources.append(2); targets.append(3); values.append(battery_to_home)
    colors.append("rgba(76,175,80,0.5)")

# Grid import
if current_data['grid_import'] > 0:
    sources.append(5); targets.append(3); values.append(current_data['grid_import'])
    colors.append("rgba(117,117,117,0.5)")

fig_sankey = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node=dict(
        pad=20,
        thickness=25,
        line=dict(color="white", width=2),
        label=labels,
        color=["#FF8F00", "#1565C0", "#4CAF50", "#2E7D32", "#FF5722", "#757575"],
        customdata=[f"{current_data['solar_generation']:.1f} kW", 
                   f"{current_data['wind_generation']:.1f} kW",
                   f"{current_data['battery_level']:.0f}%",
                   f"{current_data['consumption']:.1f} kW",
                   f"{current_data['grid_export']:.1f} kW",
                   f"{current_data['grid_import']:.1f} kW"],
        hovertemplate='%{label}<br>%{customdata}<extra></extra>'
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=colors,
        hovertemplate='%{value:.1f} kW<extra></extra>'
    )
)])

fig_sankey.update_layout(
    title={
        'text': "⚡ Real-Time Energy Distribution",
        'font': {'size': 20, 'color': '#2E7D32'}
    },
    font_size=13,
    height=450,
    paper_bgcolor='rgba(250,250,250,1)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=10, t=50, b=10)
)

st.plotly_chart(fig_sankey, width='stretch')

# Financial Dashboard Section
st.markdown("### 💼 Real-Time Financial Dashboard")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="financial-widget">
        <h4 style="margin:0; font-size:14px;">💵 Energy Costs</h4>
        <h2 style="margin:5px 0;">${st.session_state.financial_history['total_costs']:.2f}</h2>
        <p style="margin:0; font-size:12px;">Grid Purchases</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="financial-widget">
        <h4 style="margin:0; font-size:14px;">💰 Revenue</h4>
        <h2 style="margin:5px 0;">${st.session_state.financial_history['total_revenue']:.2f}</h2>
        <p style="margin:0; font-size:12px;">Energy Sales</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    net_balance = st.session_state.financial_history['total_revenue'] - st.session_state.financial_history['total_costs']
    st.markdown(f"""
    <div class="financial-widget" style="background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%);">
        <h4 style="margin:0; font-size:14px;">📊 Net Savings</h4>
        <h2 style="margin:5px 0;">${net_balance:.2f}</h2>
        <p style="margin:0; font-size:12px;">Today's Balance</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    monthly_projection = net_balance * 30 if net_balance > 0 else 0
    st.markdown(f"""
    <div class="financial-widget">
        <h4 style="margin:0; font-size:14px;">📅 Monthly Est.</h4>
        <h2 style="margin:5px 0;">${monthly_projection:.0f}</h2>
        <p style="margin:0; font-size:12px;">Projected Savings</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    carbon_saved = current_data['total_generation'] * current_data['carbon_intensity'] / 10
    st.markdown(f"""
    <div class="financial-widget" style="background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);">
        <h4 style="margin:0; font-size:14px;">🌱 Carbon Saved</h4>
        <h2 style="margin:5px 0;">{carbon_saved:.1f} kg</h2>
        <p style="margin:0; font-size:12px;">CO₂ Reduction</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    roi_percentage = (net_balance * 365 / 25000) * 100 if net_balance > 0 else 0
    st.markdown(f"""
    <div class="financial-widget">
        <h4 style="margin:0; font-size:14px;">📈 ROI</h4>
        <h2 style="margin:5px 0;">{roi_percentage:.1f}%</h2>
        <p style="margin:0; font-size:12px;">Annual Return</p>
    </div>
    """, unsafe_allow_html=True)

# Animated Real-time Charts
st.markdown("### 📊 Live Performance Charts")

col1, col2 = st.columns(2)

with col1:
    # Real-time generation and consumption chart
    df_history = pd.DataFrame(st.session_state.data_history[-30:])  # Last 30 data points
    
    fig_realtime = go.Figure()
    
    # Solar generation with gradient fill
    fig_realtime.add_trace(go.Scatter(
        x=list(range(len(df_history))),
        y=df_history['solar_generation'],
        mode='lines',
        name='Solar',
        line=dict(color='#FF8F00', width=3),
        fill='tozeroy',
        fillcolor='rgba(255,143,0,0.2)',
        hovertemplate='Solar: %{y:.1f} kW<extra></extra>'
    ))
    
    # Wind generation
    fig_realtime.add_trace(go.Scatter(
        x=list(range(len(df_history))),
        y=df_history['wind_generation'],
        mode='lines',
        name='Wind',
        line=dict(color='#1565C0', width=3),
        fill='tonexty',
        fillcolor='rgba(21,101,192,0.2)',
        hovertemplate='Wind: %{y:.1f} kW<extra></extra>'
    ))
    
    # Consumption
    fig_realtime.add_trace(go.Scatter(
        x=list(range(len(df_history))),
        y=df_history['consumption'],
        mode='lines',
        name='Usage',
        line=dict(color='#2E7D32', width=3, dash='dash'),
        hovertemplate='Usage: %{y:.1f} kW<extra></extra>'
    ))
    
    fig_realtime.update_layout(
        title="⚡ Live Energy Flow (Last 2.5 Minutes)",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(250,250,250,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_realtime, width='stretch')

with col2:
    # Trading opportunities and pricing
    fig_trading = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Energy price
    fig_trading.add_trace(
        go.Scatter(
            x=list(range(len(df_history))),
            y=df_history['price_per_kwh'],
            mode='lines+markers',
            name='Energy Price',
            line=dict(color='#FF5722', width=3),
            marker=dict(size=6),
            hovertemplate='Price: $%{y:.3f}/kWh<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Grid export (trading opportunities)
    fig_trading.add_trace(
        go.Scatter(
            x=list(range(len(df_history))),
            y=df_history['grid_export'],
            mode='lines',
            name='Export to Grid',
            line=dict(color='#4CAF50', width=2),
            fill='tozeroy',
            fillcolor='rgba(76,175,80,0.2)',
            hovertemplate='Export: %{y:.1f} kW<extra></extra>'
        ),
        secondary_y=True,
    )
    
    fig_trading.update_yaxes(title_text="Price ($/kWh)", secondary_y=False)
    fig_trading.update_yaxes(title_text="Export Power (kW)", secondary_y=True)
    
    fig_trading.update_layout(
        title="💹 Energy Trading & Pricing",
        xaxis_title="Time",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(250,250,250,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trading, width='stretch')

# Smart Alerts and AI Recommendations
st.markdown("### 🤖 AI-Powered Insights & Alerts")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🚨 Live System Alerts")
    
    alerts = []
    
    if current_data['price_per_kwh'] > 0.16:
        alerts.append(("🔴", "PEAK PRICING", f"${current_data['price_per_kwh']:.3f}/kWh - Reduce usage or sell energy"))
    
    if current_data['trading_opportunity']:
        alerts.append(("🟢", "TRADING OPPORTUNITY", f"Sell {current_data['grid_export']:.1f} kW at premium prices"))
    
    if current_data['battery_level'] < 25:
        alerts.append(("🟡", "LOW BATTERY", f"{current_data['battery_level']:.0f}% - Charge recommended"))
    
    if current_data['carbon_intensity'] < 0.3:
        alerts.append(("🌱", "CLEAN ENERGY PERIOD", "Great time for energy-intensive tasks"))
    
    if current_data['total_generation'] > current_data['consumption'] * 1.3:
        alerts.append(("⚡", "SURPLUS GENERATION", "Store or sell excess energy"))
    
    if not alerts:
        st.success("✅ All systems operating optimally")
    else:
        for icon, title, message in alerts[:4]:
            if "PEAK" in title or "LOW" in title:
                st.error(f"{icon} **{title}**: {message}")
            elif "OPPORTUNITY" in title or "SURPLUS" in title:
                st.success(f"{icon} **{title}**: {message}")
            else:
                st.info(f"{icon} **{title}**: {message}")

with col2:
    st.markdown("#### 💡 AI Recommendations")
    
    # Get AI recommendations based on current data
    recommendations = []
    
    if current_data['price_per_kwh'] > 0.15 and current_data['battery_level'] > 50:
        recommendations.append("🔋 Discharge battery to avoid peak pricing - Save $12-18/day")
    
    if current_data['grid_export'] > 15:
        recommendations.append("💰 Optimal selling window - Revenue potential: $8.50/hour")
    
    if current_data['solar_generation'] > 70 and current_data['battery_level'] < 80:
        recommendations.append("☀️ Charge battery with excess solar - Free energy storage")
    
    if current_data['consumption'] > 80:
        recommendations.append("⚡ High consumption detected - Shift non-essential loads")
    
    recommendations.append("📊 Energy efficiency: 87% - Above average performance")
    
    for rec in recommendations[:4]:
        st.info(rec)

with col3:
    st.markdown("#### 📈 Performance Metrics")
    
    energy_independence = min(100, (current_data['total_generation'] / current_data['consumption']) * 100)
    
    st.metric(
        "⚡ Energy Independence",
        f"{energy_independence:.0f}%",
        f"{random.uniform(-2, 5):+.1f}%"
    )
    
    efficiency_score = random.uniform(85, 95)
    st.metric(
        "🎯 System Efficiency",
        f"{efficiency_score:.0f}%",
        f"{random.uniform(0, 2):+.1f}%"
    )
    
    carbon_offset_trees = (carbon_saved * 30) / 22  # Trees equivalent
    st.metric(
        "🌳 Monthly Tree Equiv.",
        f"{carbon_offset_trees:.0f} trees",
        "Carbon offset impact"
    )

# Quick AI Assistant Section
st.markdown("---")
st.markdown("### 🤖 AI Energy Assistant")

col1, col2 = st.columns([2, 1])

with col1:
    user_question = st.text_input(
        "Ask me anything about your energy system, costs, or optimization:",
        placeholder="e.g., How can I save more money? What's my carbon footprint?",
        key="ai_input"
    )
    
    if user_question:
        with st.spinner("🤖 AI analyzing..."):
            response = st.session_state.ai_assistant.chat_response(user_question)
            st.markdown(f"""
            <div class="ai-widget">
                <h4>🤖 AI Response:</h4>
                <p>{response}</p>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("**Quick Actions:**")
    
    if st.button("💡 Get Optimization Tips", use_container_width=True):
        tips = st.session_state.ai_assistant.get_optimization_advice()
        st.info(tips)
    
    if st.button("📊 Analyze Performance", use_container_width=True):
        analysis = st.session_state.ai_assistant.analyze_performance()
        st.success(analysis)
    
    if st.button("💰 Financial Insights", use_container_width=True):
        insights = st.session_state.ai_assistant.get_financial_insights()
        st.info(insights)

# Auto-refresh controls
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Live Update Settings")

auto_refresh = st.sidebar.toggle("🔄 Auto-Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 3, 10, 5)

st.sidebar.markdown(f"**Status:** {'🟢 Live' if auto_refresh else '🔴 Paused'}")
st.sidebar.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.markdown(f"**Data Points:** {len(st.session_state.data_history)}")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.rerun()

# Navigation to other pages
st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Quick Navigation")
st.sidebar.page_link("pages/1_Energy_Dashboard.py", label="⚡ Energy Dashboard", icon="⚡")
st.sidebar.page_link("pages/2_Financial_Analytics.py", label="💰 Financial Analytics", icon="💰")
st.sidebar.page_link("pages/3_Grid_Management.py", label="🔌 Grid Management", icon="🔌")
st.sidebar.page_link("pages/4_AI_Assistant.py", label="🤖 AI Assistant", icon="🤖")
st.sidebar.page_link("pages/5_Carbon_Tracker.py", label="🌱 Carbon Tracker", icon="🌱")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
