import streamlit as st
import json
import os
from utils.ai_assistant import EnergyAI
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Energy Assistant")
st.markdown("---")

# Initialize AI assistant
ai_assistant = EnergyAI()

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI Energy Assistant. I can help you with energy optimization, financial analysis, carbon footprint questions, and renewable energy advice. How can I assist you today?",
            "timestamp": datetime.now()
        }
    ]

# Sidebar with AI capabilities
st.sidebar.title("🎯 AI Capabilities")
st.sidebar.markdown("""
**I can help you with:**
- 🔋 Energy optimization strategies
- 💰 Financial analysis & ROI calculations  
- 🌱 Carbon footprint reduction tips
- ⚡ Grid management advice
- 📊 Performance analytics insights
- 💹 Energy trading recommendations
- 🔮 Predictive maintenance alerts
- 📈 Usage pattern analysis
""")

# Quick action buttons
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")

if st.sidebar.button("💡 Optimization Tips"):
    optimization_query = "Give me 3 specific energy optimization tips based on current data"
    response = ai_assistant.get_optimization_advice(optimization_query)
    st.session_state.chat_history.append({
        "role": "user",
        "content": optimization_query,
        "timestamp": datetime.now()
    })
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response,
        "timestamp": datetime.now()
    })

if st.sidebar.button("📊 Analyze Performance"):
    performance_query = "Analyze my current energy performance and suggest improvements"
    response = ai_assistant.analyze_performance()
    st.session_state.chat_history.append({
        "role": "user",
        "content": performance_query,
        "timestamp": datetime.now()
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now()
    })

if st.sidebar.button("💰 Financial Insights"):
    financial_query = "Provide financial insights and cost-saving opportunities"
    response = ai_assistant.get_financial_insights()
    st.session_state.chat_history.append({
        "role": "user",
        "content": financial_query,
        "timestamp": datetime.now()
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now()
    })

# Main chat interface
st.subheader("💬 Chat with AI Assistant")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
                st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                st.caption(f"🤖 {message['timestamp'].strftime('%H:%M:%S')}")

# Chat input
user_input = st.chat_input("Ask me anything about energy management...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Get AI response
    with st.spinner("🤖 Thinking..."):
        response = ai_assistant.chat_response(user_input)
    
    # Add AI response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now()
    })
    
    # Rerun to update chat display
    st.rerun()

# AI Insights Dashboard
st.subheader("🎯 AI-Generated Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔮 Predictive Analytics")
    
    # Generate predictive insights
    insights = ai_assistant.get_predictive_insights()
    
    # Create prediction visualization
    dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
    predicted_generation = [random.uniform(80, 120) + 20 * np.sin(i * np.pi / 3) for i in range(7)]
    predicted_consumption = [random.uniform(60, 100) + 15 * np.sin((i + 1) * np.pi / 4) for i in range(7)]
    
    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Scatter(
        x=dates, y=predicted_generation,
        mode='lines+markers',
        name='Predicted Generation',
        line=dict(color='#4CAF50', width=3)
    ))
    fig_prediction.add_trace(go.Scatter(
        x=dates, y=predicted_consumption,
        mode='lines+markers',
        name='Predicted Consumption', 
        line=dict(color='#FF5722', width=3)
    ))
    
    fig_prediction.update_layout(
        title="7-Day Energy Forecast",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        height=300
    )
    st.plotly_chart(fig_prediction, use_container_width=True)
    
    # Display insights
    for insight in insights:
        st.info(f"🔮 {insight}")

with col2:
    st.markdown("#### 🎯 Smart Recommendations")
    
    # Get personalized recommendations
    recommendations = ai_assistant.get_smart_recommendations()
    
    # Recommendation impact visualization
    categories = ['Energy Savings', 'Cost Reduction', 'Carbon Impact', 'Efficiency Gain']
    impact_scores = [random.uniform(70, 95) for _ in categories]
    
    fig_impact = go.Figure(data=[
        go.Bar(x=categories, y=impact_scores,
               marker_color=['#4CAF50', '#FF8F00', '#1565C0', '#2E7D32'])
    ])
    fig_impact.update_layout(
        title="Recommendation Impact Scores",
        yaxis_title="Impact Score (%)",
        height=300
    )
    st.plotly_chart(fig_impact, use_container_width=True)
    
    # Display recommendations
    for rec in recommendations:
        st.success(f"✨ {rec}")

# AI Learning Dashboard
st.subheader("🧠 AI Learning & Adaptation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📚 Learning Progress")
    
    learning_metrics = {
        'Pattern Recognition': 94.2,
        'Forecast Accuracy': 89.7,
        'Optimization Success': 91.5,
        'User Satisfaction': 96.1
    }
    
    for metric, score in learning_metrics.items():
        st.metric(metric, f"{score:.1f}%", f"+{random.uniform(0.1, 1.5):.1f}%")

with col2:
    st.markdown("#### 🎯 Model Performance")
    
    # Performance gauge
    overall_performance = np.mean(list(learning_metrics.values()))
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_performance,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI Performance Score"},
        delta={'reference': 90},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E7D32"},
            'steps': [
                {'range': [0, 70], 'color': "#FF5722"},
                {'range': [70, 85], 'color': "#FF8F00"},
                {'range': [85, 100], 'color': "#4CAF50"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col3:
    st.markdown("#### 🔄 Continuous Learning")
    
    st.write("**Recent Learning Events:**")
    learning_events = [
        "🔄 Updated pricing prediction model",
        "📊 Improved consumption forecasting",
        "🌱 Enhanced carbon calculation accuracy",
        "⚡ Optimized load balancing algorithm"
    ]
    
    for event in learning_events:
        st.write(f"• {event}")
    
    st.metric("Data Points Processed", "2.4M+", "+127K today")

# Specialized AI Tools
st.subheader("🛠️ Specialized AI Tools")

tool_tabs = st.tabs(["📊 Data Analysis", "🔮 Forecasting", "💡 Optimization", "🌱 Sustainability"])

with tool_tabs[0]:
    st.markdown("#### 📊 AI Data Analysis Tool")
    
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Energy Pattern Analysis", "Cost Breakdown Analysis", "Efficiency Trends", "Carbon Footprint Analysis"]
    )
    
    if st.button("🔍 Run Analysis"):
        with st.spinner("🤖 Analyzing data..."):
            # Simulate AI analysis
            analysis_result = ai_assistant.analyze_data(analysis_type)
            st.success("Analysis Complete!")
            st.write(analysis_result)
            
            # Generate sample visualization based on analysis type
            if "Pattern" in analysis_type:
                x = pd.date_range(start=datetime.now()-timedelta(days=30), periods=30, freq='D')
                y = [50 + 20 * np.sin(i * np.pi / 15) + random.uniform(-5, 5) for i in range(30)]
                
                fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
                fig.update_layout(title="Energy Usage Patterns", height=300)
                st.plotly_chart(fig, use_container_width=True)

with tool_tabs[1]:
    st.markdown("#### 🔮 AI Forecasting Tool")
    
    forecast_period = st.slider("Forecast Period (days)", 1, 30, 7)
    forecast_type = st.radio("Forecast Type:", ["Generation", "Consumption", "Pricing", "Carbon Intensity"])
    
    if st.button("🔮 Generate Forecast"):
        with st.spinner("🤖 Generating forecast..."):
            forecast_data = ai_assistant.generate_forecast(forecast_type, forecast_period)
            st.success(f"Forecast generated for next {forecast_period} days")
            
            # Display forecast
            dates = pd.date_range(start=datetime.now(), periods=forecast_period, freq='D')
            values = [random.uniform(50, 150) for _ in range(forecast_period)]
            
            fig = go.Figure(data=go.Scatter(x=dates, y=values, mode='lines+markers',
                                          line=dict(color='#1565C0', width=3)))
            fig.update_layout(title=f"{forecast_type} Forecast", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(forecast_data)

with tool_tabs[2]:
    st.markdown("#### 💡 AI Optimization Tool")
    
    optimization_goal = st.selectbox(
        "Optimization Goal:",
        ["Minimize Costs", "Maximize Generation", "Reduce Carbon Footprint", "Improve Efficiency"]
    )
    
    constraints = st.multiselect(
        "Constraints:",
        ["Budget Limit", "Equipment Capacity", "Grid Stability", "Weather Dependency"]
    )
    
    if st.button("⚡ Optimize System"):
        with st.spinner("🤖 Optimizing..."):
            optimization_result = ai_assistant.optimize_system(optimization_goal, constraints)
            st.success("Optimization Complete!")
            
            # Show optimization results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Potential Savings", f"${random.uniform(100, 500):.2f}/month")
                st.metric("Efficiency Improvement", f"+{random.uniform(5, 15):.1f}%")
                
            with col2:
                st.metric("Payback Period", f"{random.uniform(2, 8):.1f} months")
                st.metric("Carbon Reduction", f"{random.uniform(10, 30):.1f} kg CO₂/month")
            
            st.write("**Recommended Actions:**")
            st.write(optimization_result)

with tool_tabs[3]:
    st.markdown("#### 🌱 AI Sustainability Advisor")
    
    sustainability_focus = st.selectbox(
        "Sustainability Focus:",
        ["Carbon Neutrality", "Renewable Integration", "Waste Reduction", "Sustainable Practices"]
    )
    
    if st.button("🌱 Get Sustainability Plan"):
        with st.spinner("🤖 Creating sustainability plan..."):
            sustainability_plan = ai_assistant.create_sustainability_plan(sustainability_focus)
            st.success("Sustainability plan created!")
            
            # Sustainability metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Carbon Savings", f"{random.uniform(100, 500):.0f} kg CO₂/year")
                
            with col2:
                st.metric("Renewable %", f"{random.uniform(80, 95):.1f}%")
                
            with col3:
                st.metric("Sustainability Score", f"{random.uniform(85, 98):.1f}/100")
            
            st.write("**Sustainability Action Plan:**")
            st.write(sustainability_plan)

# Chat management
st.subheader("💬 Chat Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Chat history cleared. How can I help you today?",
                "timestamp": datetime.now()
            }
        ]
        st.success("Chat history cleared!")

with col2:
    if st.button("📥 Export Chat"):
        # In a real implementation, this would create a downloadable file
        st.success("Chat exported! (Feature would generate file in production)")

with col3:
    if st.button("🔄 Refresh AI"):
        st.success("AI assistant refreshed!")

# Usage statistics
st.sidebar.markdown("---")
st.sidebar.markdown("**Usage Statistics:**")
st.sidebar.metric("Chat Messages", len(st.session_state.chat_history))
st.sidebar.metric("AI Responses", len([m for m in st.session_state.chat_history if m["role"] == "assistant"]))
st.sidebar.metric("Session Duration", f"{random.randint(5, 45)} min")
