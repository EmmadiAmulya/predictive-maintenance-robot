import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Load Data
# ============================================================

print("[LOAD] Loading data...")
df = pd.read_csv("data/robot_sensor_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"[OK] Loaded {len(df)} records")

# ============================================================
# STEP 2: Feature Engineering
# ============================================================

print("\n[FEAT] Engineering features...")

# Select sensor features for anomaly detection
sensor_features = ['torque_nm', 'vibration_mm_s', 'power_consumption_w',
                   'temperature_c', 'joint_movement_deg']

# Create additional features
df['torque_vibration_ratio'] = df['torque_nm'] / (df['vibration_mm_s'] + 0.001)
df['power_per_torque'] = df['power_consumption_w'] / (df['torque_nm'] + 0.001)
df['temp_vibration_index'] = df['temperature_c'] * df['vibration_mm_s']

# Rolling averages (per robot)
df = df.sort_values(['robot_id', 'timestamp'])
for feat in sensor_features:
    df[f'{feat}_rolling_mean_7'] = df.groupby('robot_id')[feat].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df[f'{feat}_rolling_std_7'] = df.groupby('robot_id')[feat].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
    )

# Rate of change
for feat in sensor_features:
    df[f'{feat}_diff'] = df.groupby('robot_id')[feat].transform(
        lambda x: x.diff().fillna(0)
    )

print(f"[OK] Total features: {len(df.columns)}")

# ============================================================
# STEP 3: Prepare Data for Model
# ============================================================

print("\n[DATA] Preparing model data...")

# All features for model
model_features = sensor_features + [
    'torque_vibration_ratio', 'power_per_torque', 'temp_vibration_index'
]

# Add rolling features
for feat in sensor_features:
    model_features.extend([f'{feat}_rolling_mean_7', f'{feat}_rolling_std_7', f'{feat}_diff'])

# Remove any NaN/inf
df[model_features] = df[model_features].replace([np.inf, -np.inf], np.nan)
df[model_features] = df[model_features].fillna(0)

X = df[model_features].copy()

# Create binary label: 1 = anomaly (Warning/Critical), 0 = normal
df['true_anomaly'] = df['actual_status'].apply(
    lambda x: 1 if x in ['Warning', 'Critical'] else 0
)
y = df['true_anomaly']

print(f"[OK] Anomaly rate: {y.mean()*100:.1f}%")
print(f"[OK] Features for model: {len(model_features)}")

# ============================================================
# STEP 4: Scale the Data
# ============================================================

print("\n[SCALE] Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("[OK] Scaler saved")

# ============================================================
# STEP 5: Train Isolation Forest
# ============================================================

print("\n[TRAIN] Training Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.35,  # Approximate anomaly ratio
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
    bootstrap=False
)

iso_forest.fit(X_scaled)

# Save model
joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')
print("[OK] Model saved")

# ============================================================
# STEP 6: Predict Anomalies
# ============================================================

print("\n[PREDICT] Predicting anomalies...")

# Isolation Forest: -1 = anomaly, 1 = normal
raw_predictions = iso_forest.predict(X_scaled)

# Convert: -1 -> 1 (anomaly), 1 -> 0 (normal)
df['predicted_anomaly'] = (raw_predictions == -1).astype(int)

# Get anomaly scores (lower = more anomalous)
anomaly_scores = iso_forest.decision_function(X_scaled)

# Normalize scores to 0-100 (higher = more anomalous)
df['anomaly_score'] = (
    (anomaly_scores.max() - anomaly_scores) / 
    (anomaly_scores.max() - anomaly_scores.min()) * 100
).round(2)

# ============================================================
# STEP 7: Classify into Normal/Warning/Critical
# ============================================================

def classify_health(score):
    if score >= 75:
        return "Critical"
    elif score >= 45:
        return "Warning"
    else:
        return "Normal"

df['predicted_status'] = df['anomaly_score'].apply(classify_health)

# ============================================================
# STEP 8: Calculate Health Score per Robot
# ============================================================

health_scores = df.groupby('robot_id').agg(
    avg_anomaly_score=('anomaly_score', 'mean'),
    max_anomaly_score=('anomaly_score', 'max'),
    pct_critical=('predicted_status', lambda x: (x == 'Critical').mean() * 100),
    pct_warning=('predicted_status', lambda x: (x == 'Warning').mean() * 100),
    avg_torque=('torque_nm', 'mean'),
    avg_vibration=('vibration_mm_s', 'mean'),
    avg_power=('power_consumption_w', 'mean'),
    avg_temperature=('temperature_c', 'mean'),
    avg_oee=('oee_score', 'mean'),
    total_savings=('savings_if_predictive_usd', 'sum'),
    total_carbon_saving=('carbon_saving_kg_co2', 'sum'),
    latest_status=('predicted_status', 'last')
).round(2)

health_scores['health_score'] = (100 - health_scores['avg_anomaly_score']).round(1)

print("\n[SUMMARY] Robot Health Summary:")
print(health_scores[['health_score', 'latest_status', 'avg_oee']])

# ============================================================
# STEP 9: Evaluate Model
# ============================================================

print("\n[EVAL] Model Evaluation:")
print("=" * 50)

print("\nConfusion Matrix:")
cm = confusion_matrix(y, df['predicted_anomaly'])
print(cm)

print("\nClassification Report:")
print(classification_report(y, df['predicted_anomaly'],
                           target_names=['Normal', 'Anomaly']))

# Accuracy
accuracy = (y == df['predicted_anomaly']).mean()
print(f"\n[OK] Overall Accuracy: {accuracy*100:.1f}%")

# Anomaly detection rate (recall for anomalies)
anomaly_recall = cm[1][1] / (cm[1][0] + cm[1][1])
print(f"[OK] Anomaly Detection Rate: {anomaly_recall*100:.1f}%")

# ============================================================
# STEP 10: Save Results for Power BI
# ============================================================

print("\n[SAVE] Saving results for Power BI...")

# Save main dataset with predictions
df.to_csv("data/robot_predictions_output.csv", index=False)

# Save health summary
health_scores.to_csv("data/robot_health_summary.csv")

# Save daily aggregation for Power BI
daily_agg = df.groupby([
    pd.Grouper(key='timestamp', freq='D'),
    'robot_id',
    'location'
]).agg(
    avg_torque=('torque_nm', 'mean'),
    avg_vibration=('vibration_mm_s', 'mean'),
    avg_power=('power_consumption_w', 'mean'),
    avg_temperature=('temperature_c', 'mean'),
    avg_joint_movement=('joint_movement_deg', 'mean'),
    avg_anomaly_score=('anomaly_score', 'mean'),
    avg_oee=('oee_score', 'mean'),
    count_critical=('predicted_status', lambda x: (x == 'Critical').sum()),
    count_warning=('predicted_status', lambda x: (x == 'Warning').sum()),
    count_normal=('predicted_status', lambda x: (x == 'Normal').sum()),
    total_downtime_hrs=('estimated_downtime_hrs', 'sum'),
    total_cost_failure=('cost_of_failure_usd', 'sum'),
    total_savings=('savings_if_predictive_usd', 'sum'),
    total_carbon_saving=('carbon_saving_kg_co2', 'sum'),
    avg_defect_rate=('defect_rate_pct', 'mean'),
    total_energy_waste=('energy_waste_kwh', 'sum')
).round(2).reset_index()

daily_agg.to_csv("data/robot_daily_summary.csv", index=False)

# Save joint-level analysis
joint_analysis = df.groupby(['robot_id', 'joint_name']).agg(
    avg_torque=('torque_nm', 'mean'),
    avg_vibration=('vibration_mm_s', 'mean'),
    avg_temperature=('temperature_c', 'mean'),
    avg_anomaly_score=('anomaly_score', 'mean'),
    count_anomalies=('predicted_anomaly', 'sum')
).round(2).reset_index()

joint_analysis.to_csv("data/robot_joint_analysis.csv", index=False)

# Save financial summary
financial_summary = df.groupby('robot_id').agg(
    total_failure_cost=('cost_of_failure_usd', 'sum'),
    total_savings=('savings_if_predictive_usd', 'sum'),
    total_downtime=('estimated_downtime_hrs', 'sum'),
    avg_defect_rate=('defect_rate_pct', 'mean'),
    total_energy_waste=('energy_waste_kwh', 'sum'),
    total_carbon_saving=('carbon_saving_kg_co2', 'sum')
).round(2).reset_index()

financial_summary.to_csv("data/financial_summary.csv", index=False)

print("\n[OK] All files saved successfully!")
print("""
Output Files:
1. robot_predictions_output.csv   - Full data with predictions
2. robot_health_summary.csv      - Robot health scores
3. robot_daily_summary.csv       - Daily aggregated data
4. robot_joint_analysis.csv      - Joint-level analysis
5. financial_summary.csv         - Financial impact data
6. isolation_forest_model.pkl    - Trained model
7. scaler.pkl                    - Feature scaler
""")
