import pandas as pd
import numpy as np
import json

print("[LOAD] Reading CSV files...")

daily = pd.read_csv("data/robot_daily_summary.csv")
health = pd.read_csv("data/robot_health_summary.csv")
joint = pd.read_csv("data/robot_joint_analysis.csv")
financial = pd.read_csv("data/financial_summary.csv")

daily['timestamp'] = pd.to_datetime(daily['timestamp'])

# ---- KPI Data ----
robots = daily['robot_id'].unique().tolist()
locations = daily['location'].unique().tolist()

total_robots = len(robots)
avg_oee = round(daily['avg_oee'].mean() * 100, 1)

robot_avg_scores = daily.groupby('robot_id')['avg_anomaly_score'].mean()
healthy_count = int((robot_avg_scores < 45).sum())
warning_count = int(((robot_avg_scores >= 45) & (robot_avg_scores < 75)).sum())
critical_count = int((robot_avg_scores >= 75).sum())

total_failure_cost = round(financial['total_failure_cost'].sum(), 0)
total_savings = round(financial['total_savings'].sum(), 0)
savings_pct = round(total_savings / (total_failure_cost + total_savings) * 100, 1) if (total_failure_cost + total_savings) > 0 else 0
total_downtime = round(financial['total_downtime'].sum(), 1)
avg_defect = round(financial['avg_defect_rate'].mean(), 1)
total_energy_waste = round(financial['total_energy_waste'].sum(), 1)
total_carbon = round(financial['total_carbon_saving'].sum(), 1)

# ---- Robot Health Table ----
health_table = []
for _, row in health.iterrows():
    score = round(100 - row['avg_anomaly_score'], 1)
    status = "Critical" if row['avg_anomaly_score'] >= 75 else ("Warning" if row['avg_anomaly_score'] >= 45 else "Normal")
    health_table.append({
        "robot_id": row['robot_id'],
        "health_score": score,
        "status": status,
        "avg_oee": round(row['avg_oee'] * 100, 1),
        "avg_vibration": round(row['avg_vibration'], 2),
        "avg_torque": round(row['avg_torque'], 1)
    })

# ---- Status Distribution ----
status_counts = daily.copy()
status_counts['status'] = status_counts['avg_anomaly_score'].apply(
    lambda x: 'Critical' if x >= 75 else ('Warning' if x >= 45 else 'Normal')
)
status_dist = status_counts['status'].value_counts().to_dict()

# ---- OEE Trend (daily avg per robot) ----
oee_trend = {}
for robot in robots:
    rdata = daily[daily['robot_id'] == robot].sort_values('timestamp')
    oee_trend[robot] = {
        "dates": rdata['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        "values": rdata['avg_oee'].round(3).tolist()
    }

# ---- Anomaly Score Trend ----
anomaly_trend = {}
for robot in robots:
    rdata = daily[daily['robot_id'] == robot].sort_values('timestamp')
    anomaly_trend[robot] = {
        "dates": rdata['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        "values": rdata['avg_anomaly_score'].round(2).tolist()
    }

# ---- Scatter: Torque vs Vibration ----
scatter_data = []
for _, row in daily.iterrows():
    scatter_data.append({
        "x": round(row['avg_torque'], 2),
        "y": round(row['avg_vibration'], 2),
        "score": round(row['avg_anomaly_score'], 1),
        "robot": row['robot_id']
    })

# Sample scatter data (too many points otherwise)
np.random.seed(42)
if len(scatter_data) > 500:
    indices = np.random.choice(len(scatter_data), 500, replace=False)
    scatter_data = [scatter_data[i] for i in indices]

# ---- Joint Analysis ----
joint_data = []
for _, row in joint.iterrows():
    joint_data.append({
        "robot_id": row['robot_id'],
        "joint_name": row['joint_name'],
        "avg_anomaly_score": round(row['avg_anomaly_score'], 1),
        "count_anomalies": int(row['count_anomalies']),
        "avg_vibration": round(row['avg_vibration'], 2)
    })

# ---- Financial by Robot ----
fin_data = []
for _, row in financial.iterrows():
    fin_data.append({
        "robot_id": row['robot_id'],
        "failure_cost": round(row['total_failure_cost'], 0),
        "savings": round(row['total_savings'], 0),
        "downtime": round(row['total_downtime'], 1),
        "defect_rate": round(row['avg_defect_rate'], 1),
        "energy_waste": round(row['total_energy_waste'], 1),
        "carbon_saving": round(row['total_carbon_saving'], 1)
    })

# ---- Cost Trend Over Time ----
cost_trend = daily.groupby(daily['timestamp'].dt.strftime('%Y-%m-%d')).agg(
    cost=('total_cost_failure', 'sum'),
    savings=('total_savings', 'sum'),
    energy=('total_energy_waste', 'sum'),
    carbon=('total_carbon_saving', 'sum')
).round(0).reset_index()

cost_trend_data = {
    "dates": cost_trend['timestamp'].tolist(),
    "cost": cost_trend['cost'].tolist(),
    "savings": cost_trend['savings'].tolist(),
    "energy": cost_trend['energy'].tolist(),
    "carbon": cost_trend['carbon'].tolist()
}

# ---- Location Health ----
loc_health = daily.groupby('location')['avg_anomaly_score'].mean().round(1).to_dict()

# ---- Sensor Trends (daily avg) ----
sensor_trend = daily.groupby(daily['timestamp'].dt.strftime('%Y-%m-%d')).agg(
    torque=('avg_torque', 'mean'),
    vibration=('avg_vibration', 'mean'),
    temperature=('avg_temperature', 'mean'),
    power=('avg_power', 'mean')
).round(2).reset_index()

sensor_data = {
    "dates": sensor_trend['timestamp'].tolist(),
    "torque": sensor_trend['torque'].tolist(),
    "vibration": sensor_trend['vibration'].tolist(),
    "temperature": sensor_trend['temperature'].tolist(),
    "power": sensor_trend['power'].tolist()
}

# ---- Build final JSON ----
dashboard_data = {
    "kpi": {
        "total_robots": total_robots,
        "healthy": healthy_count,
        "warning": warning_count,
        "critical": critical_count,
        "avg_oee": avg_oee,
        "total_failure_cost": total_failure_cost,
        "total_savings": total_savings,
        "savings_pct": savings_pct,
        "total_downtime": total_downtime,
        "avg_defect": avg_defect,
        "total_energy_waste": total_energy_waste,
        "total_carbon": total_carbon
    },
    "robots": robots,
    "locations": locations,
    "health_table": health_table,
    "status_dist": status_dist,
    "oee_trend": oee_trend,
    "anomaly_trend": anomaly_trend,
    "scatter": scatter_data,
    "joint": joint_data,
    "financial": fin_data,
    "cost_trend": cost_trend_data,
    "loc_health": loc_health,
    "sensor": sensor_data
}

# Save JSON
with open("dashboard/data.js", "w") as f:
    f.write("const DASHBOARD_DATA = ")
    json.dump(dashboard_data, f)
    f.write(";")

print(f"[OK] Dashboard data exported!")
print(f"[INFO] Robots: {total_robots}, Records: {len(daily)}")
print(f"[INFO] Open dashboard/index.html in your browser")
