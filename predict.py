# predict.py
from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model.joblib"

def predict_price(total_sqft: float, bhk: float, bath: float, location: str) -> float:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([{
        "total_sqft": float(total_sqft),
        "bhk": float(bhk),
        "bath": float(bath),
        "location": str(location)
    }])
    return float(model.predict(X)[0])

def main():
    parser = argparse.ArgumentParser(description="Predict Bengaluru House Price (Lakhs).")
    parser.add_argument("--total_sqft", type=float, required=True)
    parser.add_argument("--bhk", type=float, required=True)
    parser.add_argument("--bath", type=float, required=True)
    parser.add_argument("--location", type=str, required=True)
    args = parser.parse_args()

    price = predict_price(args.total_sqft, args.bhk, args.bath, args.location)
    print(f"Estimated Price: ₹ {price:.2f} Lakhs")

if __name__ == "__main__":
    main()
