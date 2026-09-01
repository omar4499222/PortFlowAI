import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 5000

df = pd.DataFrame({
    "port_capacity_teu": np.random.randint(8000, 30000, N),
    "containers_in_yard": np.random.randint(3000, 29000, N),
    "vessel_arrivals_24h": np.random.randint(2, 25, N),
    "avg_container_volume_per_vessel": np.random.randint(300, 4500, N),
    "berth_utilization": np.random.uniform(.35, .99, N),
    "truck_arrivals_24h": np.random.randint(500, 7000, N),
    "truck_queue_minutes": np.random.randint(5, 360, N),
    "avg_handling_time_hours": np.random.uniform(1.5, 18, N),
    "customs_clearance_hours": np.random.uniform(1, 48, N),
    "weather_severity": np.random.uniform(0, 1, N),
    "crane_utilization": np.random.uniform(.35, .99, N),
    "historical_delay_rate": np.random.uniform(0, .8, N),
    "rail_utilization": np.random.uniform(.1, .95, N),
})

capacity_pressure = df.containers_in_yard / df.port_capacity_teu
score = (
    2.5*capacity_pressure
    + 1.8*df.berth_utilization
    + 0.9*df.truck_queue_minutes/360
    + 0.8*df.crane_utilization
    + 0.7*df.vessel_arrivals_24h/25
    + 0.6*df.weather_severity
    + 0.8*df.historical_delay_rate
    + np.random.normal(0, .25, N)
)
df["congestion_risk"] = (score > 3.25).astype(int)

delay_score = (
    1.7*df.congestion_risk
    + .8*df.customs_clearance_hours/48
    + .8*df.truck_queue_minutes/360
    + .6*df.weather_severity
    + .9*df.historical_delay_rate
    + .4*df.avg_handling_time_hours/18
    + np.random.normal(0, .2, N)
)
df["shipment_delayed"] = (delay_score > 1.65).astype(int)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/port_operations.csv", index=False)
print("Generated", len(df), "synthetic records.")
