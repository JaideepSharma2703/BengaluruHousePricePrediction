
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model_compressed.joblib"

# Load model
pipe = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected keys: total_sqft, bhk, bath, location
        required_fields = ["total_sqft", "bhk", "bath", "location"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert into DataFrame (pipeline expects DataFrame, not dict)
        df = pd.DataFrame([data])

        # Prediction
        prediction = pipe.predict(df)[0]

        return jsonify({
            "input": data,
            "predicted_price_lakhs": f"{round(float(prediction), 2)} lakhs"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/locations", methods=["GET"])
def get_locations():
    try:
        # Get the ColumnTransformer (fitted one from pipeline)
        preprocessor = pipe.named_steps["pre"]

        # Loop through to find categorical transformer
        cat_transformer = None
        for name, transformer, columns in preprocessor.transformers_:
            if name == "cat":
                cat_transformer = transformer
                break

        if cat_transformer is None:
            return jsonify({"error": "Categorical transformer not found"}), 500

        # Ensure we are accessing the OneHotEncoder correctly
        if hasattr(cat_transformer, "named_steps"):
            ohe = cat_transformer.named_steps["ohe"]   # Pipeline inside
        else:
            return jsonify({"error": "Unexpected transformer structure"}), 500

        # Now fetch the fitted categories
        if hasattr(ohe, "categories_"):
            locations = list(ohe.categories_[0])
            return jsonify({"locations": locations})
        else:
            return jsonify({"error": "OneHotEncoder is not fitted"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK" ,
        "message": "API is up and running successfully!!"

    }) , 200

@app.route("/model_info", methods=["GET"])
def get_model_info():
    """Return metadata about the trained model."""
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error" : "Model file not found"}), 404

    try:
        model = joblib.load(MODEL_PATH)

        info = {
            "model_type" : type(model).__name__,
            "is_fitted" : hasattr(model, "predict"),
            "model_parameters" : str(model.get_params() if hasattr(model, "get_params") else "N/A"),
            "file_path" : str(MODEL_PATH),
            "status" : "Model Load Successfully !!",
        }
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
