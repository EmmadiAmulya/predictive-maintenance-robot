import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ============================================================
# STEP 1: Define Robot Configuration
# ============================================================

robot_ids = ["IRB_1200_01", "IRB_1200_02", "IRB_1200_03",
             "IRB_1200_04", "IRB_1200_05", "IRB_6700_01",
             "IRB_6700_02", "IRB_6700_03"]

locations = ["Assembly_Line_A", "Assembly_Line_B",
             "Welding_Station", "Painting_Station",
             "Packaging_Line", "Quality_Check"]

joint_names = ["Joint_1_Base", "Joint_2_Shoulder",
               "Joint_3_Elbow", "Joint_4_Wrist1",
               "Joint_5_Wrist2", "Joint_6_Flange"]

# ============================================================
# STEP 2: Generate Time-Series Data
# ============================================================

np.random.seed(42)

start_date = datetime(2024, 1, 1)
num_days = 90  # 3 months of data
records_per_day = 96  # Every 15 minutes

total_records = num_days * records_per_day * len(robot_ids)

data = []

for robot in robot_ids:
    # Each robot has a different "health profile"
    base_torque = np.random.uniform(40, 80)
    base_vibration = np.random.uniform(0.5, 2.0)
    base_power = np.random.uniform(200, 500)
    base_temp = np.random.uniform(35, 55)
    base_joint_movement = np.random.uniform(80, 120)

    # Assign location
    location = random.choice(locations)

    # Random degradation start point
    degradation_start = random.randint(30, 60)  # Day when problems begin

    for day in range(num_days):
        for record in range(records_per_day):
            timestamp = start_date + timedelta(
                days=day,
                minutes=record * 15
            )

            # ---- Normal behavior with small noise ----
            torque = base_torque + np.random.normal(0, 2)
            vibration = base_vibration + np.random.normal(0, 0.1)
            power = base_power + np.random.normal(0, 10)
            temperature = base_temp + np.random.normal(0, 1)
            joint_movement = base_joint_movement + np.random.normal(0, 3)

            # ---- Degradation kicks in after degradation_start ----
            if day >= degradation_start:
                degradation_factor = (day - degradation_start) / (num_days - degradation_start)

                # Gradual increase in problems
                torque += degradation_factor * np.random.uniform(5, 20)
                vibration += degradation_factor * np.random.uniform(1.0, 4.0)
                power += degradation_factor * np.random.uniform(50, 150)
                temperature += degradation_factor * np.random.uniform(5, 20)

                # Movement becomes erratic
                joint_movement -= degradation_factor * np.random.uniform(5, 20)

                # Occasional spikes (anomalies)
                if random.random() < 0.1 * degradation_factor:
                    torque += np.random.uniform(20, 50)
                    vibration += np.random.uniform(3, 8)
                    power += np.random.uniform(100, 300)
                    temperature += np.random.uniform(10, 25)

            # Ensure non-negative values
            torque = max(0, torque)
            vibration = max(0, vibration)
            power = max(0, power)
            temperature = max(0, temperature)
            joint_movement = max(0, joint_movement)

            # Pick a random joint for this reading
            joint = random.choice(joint_names)

            # Calculate OEE components
            availability = max(0.5, min(1.0, 1.0 - (vibration / 20)))
            performance = max(0.5, min(1.0, joint_movement / 120))
            quality = max(0.5, min(1.0, 1.0 - (torque - base_torque) / 100))
            oee = round(availability * performance * quality, 3)

            # Determine actual status
            if day >= degradation_start:
                degradation_factor = (day - degradation_start) / (num_days - degradation_start)
                if degradation_factor > 0.7:
                    status = "Critical"
                elif degradation_factor > 0.3:
                    status = "Warning"
                else:
                    status = "Normal"
            else:
                status = "Normal"

            # Calculate financial metrics
            if status == "Critical":
                estimated_downtime_hrs = round(np.random.uniform(4, 12), 1)
                cost_of_failure = round(np.random.uniform(5000, 20000), 0)
                defect_rate = round(np.random.uniform(5, 15), 1)
                energy_waste_kwh = round(np.random.uniform(10, 50), 1)
            elif status == "Warning":
                estimated_downtime_hrs = round(np.random.uniform(1, 4), 1)
                cost_of_failure = round(np.random.uniform(1000, 5000), 0)
                defect_rate = round(np.random.uniform(2, 5), 1)
                energy_waste_kwh = round(np.random.uniform(5, 15), 1)
            else:
                estimated_downtime_hrs = 0
                cost_of_failure = 0
                defect_rate = round(np.random.uniform(0.1, 1.0), 1)
                energy_waste_kwh = round(np.random.uniform(0, 3), 1)

            # Savings if predictive maintenance is applied
            savings = round(cost_of_failure * 0.7, 0) if status != "Normal" else 0

            # Carbon footprint reduction (kg CO2)
            carbon_saving = round(energy_waste_kwh * 0.5 * 0.6, 2) if status != "Normal" else 0

            data.append({
                "timestamp": timestamp,
                "robot_id": robot,
                "location": location,
                "joint_name": joint,
                "torque_nm": round(torque, 2),
                "vibration_mm_s": round(vibration, 2),
                "power_consumption_w": round(power, 2),
                "temperature_c": round(temperature, 2),
                "joint_movement_deg": round(joint_movement, 2),
                "oee_score": oee,
                "availability": round(availability, 3),
                "performance": round(performance, 3),
                "quality_score": round(quality, 3),
                "actual_status": status,
                "estimated_downtime_hrs": estimated_downtime_hrs,
                "cost_of_failure_usd": cost_of_failure,
                "defect_rate_pct": defect_rate,
                "energy_waste_kwh": energy_waste_kwh,
                "savings_if_predictive_usd": savings,
                "carbon_saving_kg_co2": carbon_saving
            })

# ============================================================
# STEP 3: Create DataFrame and Save
# ============================================================

df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# Save to CSV
df.to_csv("data/robot_sensor_data.csv", index=False)

print(f"[OK] Dataset generated successfully!")
print(f"[DATA] Total Records: {len(df)}")
print(f"[ROBOTS] Robots: {df['robot_id'].nunique()}")
print(f"[DATE] Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"[STATS] Status Distribution:\n{df['actual_status'].value_counts()}")
print(f"\n[COLS] Column Names:\n{list(df.columns)}")
print(f"\n[SAMPLE] Sample Data:\n{df.head()}")
