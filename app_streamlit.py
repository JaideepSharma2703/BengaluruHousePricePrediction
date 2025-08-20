# app_streamlit.py
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import os
import requests

from sklearn.externals.array_api_compat.torch import result_type

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model_compressed.joblib"
DATA_PATH = BASE_DIR / "data" / "Bengaluru_House_Data.csv"

MODEL_URL = "https://drive.google.com/uc?export=download&id=14tcnjOR_y5zM97-w5ziynETI-IdxEYcH"

st.set_page_config(page_title="Bengaluru House Price Prediction", page_icon="🏠")
st.title("🏠 Bengaluru House Price Prediction")
st.caption("Enter property details to estimate price (Lakhs).")



@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.info("Downloading model...")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
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
bhk_options = list(range(1,13))
bath_options = list(range(1,9))

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Prediction" , "Health Check" , "Model Info"])

#Page Routing
if page == "Prediction":
    st.title("🏠 Bengaluru House Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        total_sqft = st.number_input("Total Sqft", min_value=200, max_value=20000, value=1200, step=50)
        bhk = st.selectbox("BHK" , options=bhk_options , index = 0)
    with col2:
        bath = st.selectbox("Bathrooms", options=bath_options , index = 0)
        location = st.selectbox("Location", options=locations, index=(locations.index("Whitefield") if "Whitefield" in locations else 0))

    if st.button("Predict Price"):
        try:
            """X = pd.DataFrame([{
                "total_sqft": float(total_sqft),
                "bhk": float(bhk),
                "bath": float(bath),
                "location": str(location)
            }])"""
            data  = {
                "total_sqft": total_sqft,
                "bhk": bhk,
                "bath": bath,
                "location": location
            }
            response = requests.post("http://127.0.0.1:5000/predict", json=data)

            if response.status_code == 200:
                result =  response.json()
                price = result.get("predicted_price")
                "price = float(model.predict(X)[0])"
                st.success(f"Estimated Price: ₹ {price:.2f} Lakhs")
            else:
                st.error("Prediction failed. Please try again.")

        except Exception as e:
            st.error(f"Error: {e}")

    elif page == "Health Check":
        st.title("🩺 API Health Status")
        try:
            res = requests.get("http://127.0.0.1:5000/health")
            if res.status_code == 200:
                st.success(f"{res.json()['status']}")
            else:
                st.error("API is not healthy.")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

    elif page == "Model Info":
        st.title("ℹ️ Model Information.")
        try:
            res = requests.get("http://127.0.0.1:5000/model_info")
            if res.status_code == 200:
                info= res.json()
                st.json(info)
            else:
                st.error("Failed to fetch model info.")
        except Exception as e:
            st.error(f"Error : {e}")


