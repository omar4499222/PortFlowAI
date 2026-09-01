# PortFlow AI
AI-powered early warning and decision-support prototype for port congestion and shipment delays.

## MVP
- Synthetic port operations dataset generator
- XGBoost-compatible ML pipeline (RandomForest fallback if xgboost is unavailable)
- Congestion-risk classification
- Shipment-delay probability
- Explainable feature importance
- FastAPI backend
- Streamlit dashboard
- Proactive recommendations
- Docker configuration

## Run
```bash
pip install -r requirements.txt
python data/generate_data.py
python ml/train.py
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

API: http://localhost:8000/docs
Dashboard: http://localhost:8501

## Important
The included data is synthetic/demo data. Replace it with authorized real port data for production use.
