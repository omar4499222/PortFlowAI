import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA = "data/port_operations.csv"
FEATURES = [
    "port_capacity_teu","containers_in_yard","vessel_arrivals_24h",
    "avg_container_volume_per_vessel","berth_utilization","truck_arrivals_24h",
    "truck_queue_minutes","avg_handling_time_hours","customs_clearance_hours",
    "weather_severity","crane_utilization","historical_delay_rate","rail_utilization"
]

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df["congestion_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)
model = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)
p = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

print("Accuracy:", round(accuracy_score(y_test,p),3))
print("Precision:", round(precision_score(y_test,p),3))
print("Recall:", round(recall_score(y_test,p),3))
print("F1:", round(f1_score(y_test,p),3))
print("ROC-AUC:", round(roc_auc_score(y_test,proba),3))

Path("models").mkdir(exist_ok=True)
joblib.dump({"model":model, "features":FEATURES}, "models/congestion_model.joblib")
print("Saved models/congestion_model.joblib")
