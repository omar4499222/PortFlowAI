import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="PortFlow AI", layout="wide")
st.title("🚢 PortFlow AI")
st.caption("Predictive port congestion & shipment-delay decision support")

st.sidebar.header("Port Snapshot")
capacity = st.sidebar.number_input("Port capacity (TEU)", 8000, 30000, 20000)
yard = st.sidebar.number_input("Containers in yard", 1000, 30000, 17000)
vessels = st.sidebar.number_input("Vessel arrivals / 24h", 1, 30, 12)
volume = st.sidebar.number_input("Avg container volume / vessel", 100, 5000, 1800)
berth = st.sidebar.slider("Berth utilization", 0.0, 1.0, .82)
trucks = st.sidebar.number_input("Truck arrivals / 24h", 100, 8000, 3000)
queue = st.sidebar.number_input("Truck queue (minutes)", 0, 600, 160)
handling = st.sidebar.number_input("Avg handling time (hours)", 1.0, 24.0, 8.0)
customs = st.sidebar.number_input("Customs clearance (hours)", 1.0, 72.0, 16.0)
weather = st.sidebar.slider("Weather severity", 0.0, 1.0, .25)
crane = st.sidebar.slider("Crane utilization", 0.0, 1.0, .75)
hist = st.sidebar.slider("Historical delay rate", 0.0, 1.0, .25)
rail = st.sidebar.slider("Rail utilization", 0.0, 1.0, .55)

payload = {
"port_capacity_teu":capacity,"containers_in_yard":yard,"vessel_arrivals_24h":vessels,
"avg_container_volume_per_vessel":volume,"berth_utilization":berth,
"truck_arrivals_24h":trucks,"truck_queue_minutes":queue,
"avg_handling_time_hours":handling,"customs_clearance_hours":customs,
"weather_severity":weather,"crane_utilization":crane,
"historical_delay_rate":hist,"rail_utilization":rail}

if st.button("Run PortFlow AI"):
    try:
        r=requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        out=r.json()
        a,b,c=st.columns(3)
        a.metric("Congestion Probability", f"{out['congestion_probability']*100:.1f}%")
        b.metric("Risk Level", out["risk_level"])
        c.metric("Yard Utilization", f"{yard/capacity*100:.1f}%")
        st.subheader("Why is the risk elevated?")
        factors = pd.DataFrame({
            "Factor":["Yard pressure","Berth utilization","Truck queue","Crane utilization","Weather","Historical delay"],
            "Value":[yard/capacity,berth,queue/360,crane,weather,hist]
        })
        st.bar_chart(factors.set_index("Factor"))
        st.subheader("AI Proactive Recommendations")
        for item in out["recommendations"]: st.success(item)
    except Exception as e:
        st.error("Start the API first: uvicorn api.main:app --reload")
