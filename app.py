import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Robot Predictive Maintenance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/robot_daily_summary.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    health = pd.read_csv("data/robot_health_summary.csv")
    joint = pd.read_csv("data/robot_joint_analysis.csv")
    financial = pd.read_csv("data/financial_summary.csv")
    return df, health, joint, financial

df, health, joint, financial = load_data()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🤖 Dashboard Controls")
st.sidebar.markdown("---")

selected_robots = st.sidebar.multiselect(
    "Select Robots",
    options=df['robot_id'].unique(),
    default=df['robot_id'].unique()
)

selected_location = st.sidebar.multiselect(
    "Select Location",
    options=df['location'].unique(),
    default=df['location'].unique()
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['timestamp'].min(), df['timestamp'].max())
)

# Filter data
filtered_df = df[
    (df['robot_id'].isin(selected_robots)) &
    (df['location'].isin(selected_location))
]

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("🏭 AI-Powered Predictive Maintenance")
st.subheader("Robotic Assembly Line Health Command Center")
st.markdown("---")

# ===== KPI ROW =====
col1, col2, col3, col4, col5 = st.columns(5)

total_robots = len(selected_robots)
healthy = len(filtered_df[filtered_df['avg_anomaly_score'] < 45]['robot_id'].unique())
warning = len(filtered_df[(filtered_df['avg_anomaly_score'] >= 45) & 
                          (filtered_df['avg_anomaly_score'] < 75)]['robot_id'].unique())
critical = len(filtered_df[filtered_df['avg_anomaly_score'] >= 75]['robot_id'].unique())
avg_oee = filtered_df['avg_oee'].mean()

col1.metric("🤖 Total Robots", total_robots)
col2.metric("🟢 Healthy", healthy)
col3.metric("🟡 Warning", warning)
col4.metric("🔴 Critical", critical)
col5.metric("📈 OEE Score", f"{avg_oee:.1%}")

st.markdown("---")

# ===== CHARTS ROW 1 =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Anomaly Score Trend")
    fig_trend = px.line(
        filtered_df, x='timestamp', y='avg_anomaly_score',
        color='robot_id',
        title="Anomaly Score Over Time",
        labels={'avg_anomaly_score': 'Anomaly Score', 'timestamp': 'Date'}
    )
    fig_trend.add_hline(y=75, line_dash="dash", line_color="red",
                        annotation_text="Critical Threshold")
    fig_trend.add_hline(y=45, line_dash="dash", line_color="orange",
                        annotation_text="Warning Threshold")
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("🔵 Torque vs Vibration")
    fig_scatter = px.scatter(
        filtered_df, x='avg_torque', y='avg_vibration',
        color='avg_anomaly_score',
        size='avg_anomaly_score',
        hover_data=['robot_id', 'timestamp'],
        title="Torque vs Vibration (Color = Anomaly Score)",
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ===== CHARTS ROW 2 =====
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ OEE Trend")
    fig_oee = px.line(
        filtered_df, x='timestamp', y='avg_oee',
        color='robot_id',
        title="OEE Score Over Time"
    )
    st.plotly_chart(fig_oee, use_container_width=True)

with col2:
    st.subheader("💰 Financial Impact")
    fig_fin = px.bar(
        financial[financial['robot_id'].isin(selected_robots)],
        x='robot_id', y=['total_failure_cost', 'total_savings'],
        barmode='group',
        title="Failure Cost vs Savings by Robot"
    )
    st.plotly_chart(fig_fin, use_container_width=True)

# ===== JOINT ANALYSIS =====
st.markdown("---")
st.subheader("🔧 Joint-Level Analysis")

fig_joint = px.density_heatmap(
    joint[joint['robot_id'].isin(selected_robots)],
    x='robot_id', y='joint_name', z='avg_anomaly_score',
    histfunc='avg',
    title="Joint Anomaly Heatmap",
    color_continuous_scale='RdYlGn_r'
)
st.plotly_chart(fig_joint, use_container_width=True)

# ===== SUSTAINABILITY =====
st.markdown("---")
st.subheader("🌱 Sustainability Impact")

col1, col2, col3 = st.columns(3)

total_carbon = financial['total_carbon_saving'].sum()
total_energy_waste = financial['total_energy_waste'].sum()
total_savings_val = financial['total_savings'].sum()

col1.metric("🌿 Carbon Saved (kg CO₂)", f"{total_carbon:,.0f}")
col2.metric("⚡ Energy Waste (kWh)", f"{total_energy_waste:,.0f}")
col3.metric("💰 Total Savings ($)", f"${total_savings_val:,.0f}")

st.markdown("---")
st.caption("Built with ❤️ using AI + IoT + Power BI | Predictive Maintenance System")
