# train.py
from __future__ import annotations
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from preprocessing import clean_dataframe


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Bengaluru_House_Data.csv"

MODEL_PATH = BASE_DIR / "models" / "bhp_model.joblib"

def main():
    print("📥 Loading data from:", DATA_PATH)
    df_raw = pd.read_csv(DATA_PATH)
    print("Rows (raw):", len(df_raw))

    print("🧹 Cleaning & feature engineering...")
    df = clean_dataframe(df_raw)
    print("Rows (clean):", len(df))

    if "price" not in df.columns:
        raise KeyError("Column 'price' not found after cleaning. Check your CSV column names.")

    # Choose features that are usually present in Kaggle dataset
    candidate_features = ["total_sqft", "bhk", "bath", "location"]
    features = [c for c in candidate_features if c in df.columns]
    print("Using features:", features)

    X = df[features].copy()
    y = df["price"].astype(float).copy()  # Kaggle price in Lakhs

    # Identify numeric vs categorical
    num_features = [c for c in features if c != "location"]
    cat_features = [c for c in features if c == "location"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])

    print("🔀 Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚂 Training...")
    pipe.fit(X_train, y_train)

    print("📊 Evaluating...")
    preds = pipe.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2:   {r2:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"💾 Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
