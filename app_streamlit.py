# app_streamlit.py
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Bengaluru_House_Data.csv"

st.set_page_config(page_title="Bengaluru House Price Prediction", page_icon="🏠")
st.title("🏠 Bengaluru House Price Prediction")
st.caption("Enter property details to estimate price (Lakhs).")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run training first: `python train.py`.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_locations():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if "location" in df.columns:
            locs = sorted(set(df["location"].dropna().astype(str).str.strip()))
            return [l for l in locs if l]
    return ["Whitefield", "Koramangala", "HSR Layout", "Indiranagar", "Marathahalli", "Electronic City"]

model = load_model()
locations = load_locations()

col1, col2 = st.columns(2)
with col1:
    total_sqft = st.number_input("Total Sqft", min_value=200, max_value=20000, value=1200, step=50)
    bhk = st.number_input("BHK", min_value=1, max_value=12, value=2, step=1)
with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=9, value=2, step=1)
    location = st.selectbox("Location", options=locations, index=(locations.index("Whitefield") if "Whitefield" in locations else 0))

if st.button("Predict Price"):
    X = pd.DataFrame([{
        "total_sqft": float(total_sqft),
        "bhk": float(bhk),
        "bath": float(bath),
        "location": str(location)
    }])
    price = float(model.predict(X)[0])
    st.success(f"Estimated Price: ₹ {price:.2f} Lakhs")
