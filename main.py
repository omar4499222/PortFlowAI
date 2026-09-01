from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="PortFlow AI API", version="0.1.0")
bundle = joblib.load("models/congestion_model.joblib")
model, FEATURES = bundle["model"], bundle["features"]

class PortSnapshot(BaseModel):
    port_capacity_teu: int
    containers_in_yard: int
    vessel_arrivals_24h: int
    avg_container_volume_per_vessel: int
    berth_utilization: float
    truck_arrivals_24h: int
    truck_queue_minutes: int
    avg_handling_time_hours: float
    customs_clearance_hours: float
    weather_severity: float
    crane_utilization: float
    historical_delay_rate: float
    rail_utilization: float

def recommendation(x):
    actions=[]
    if x["berth_utilization"] >= .85: actions.append("Redistribute vessel/berth schedule.")
    if x["truck_queue_minutes"] >= 180: actions.append("Redirect or stagger truck arrivals.")
    if x["containers_in_yard"]/x["port_capacity_teu"] >= .8: actions.append("Prioritize yard evacuation and container handling.")
    if x["weather_severity"] >= .7: actions.append("Activate weather contingency plan.")
    if x["customs_clearance_hours"] >= 24: actions.append("Prioritize customs-clearance exceptions.")
    return actions or ["Continue monitoring; no immediate intervention required."]

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
def predict(snapshot: PortSnapshot):
    row = pd.DataFrame([snapshot.model_dump()])[FEATURES]
    risk = float(model.predict_proba(row)[0,1])
    risk_level = "HIGH" if risk >= .70 else "MEDIUM" if risk >= .40 else "LOW"
    return {
        "congestion_probability": round(risk,4),
        "risk_level": risk_level,
        "recommendations": recommendation(snapshot.model_dump())
    }
