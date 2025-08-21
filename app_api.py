from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model_compressed.joblib"

app = Flask(__name__)

# Lazy-loaded model
pipe = None

def get_model():
    """Load the model only once when needed."""
    global pipe
    if pipe is None:
        pipe = joblib.load(MODEL_PATH)
    return pipe

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = ["total_sqft", "bhk", "bath", "location"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        df = pd.DataFrame([data])
        model = get_model()
        prediction = model.predict(df)[0]

        return jsonify({
            "input": data,
            "predicted_price_lakhs": f"{round(float(prediction), 2)} lakhs"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/locations", methods=["GET"])
def get_locations():
    try:
        model = get_model()
        preprocessor = model.named_steps["pre"]

        cat_transformer = None
        for name, transformer, columns in preprocessor.transformers_:
            if name == "cat":
                cat_transformer = transformer
                break

        if cat_transformer is None or not hasattr(cat_transformer, "named_steps"):
            return jsonify({"error": "Categorical transformer not found"}), 500

        ohe = cat_transformer.named_steps["ohe"]

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
        "status": "OK",
        "message": "API is up and running successfully!!"
    }), 200

@app.route("/model_info", methods=["GET"])
def get_model_info():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model file not found"}), 404

    try:
        model = get_model()
        info = {
            "model_type": type(model).__name__,
            "is_fitted": hasattr(model, "predict"),
            "model_parameters": str(model.get_params() if hasattr(model, "get_params") else "N/A"),
            "file_path": str(MODEL_PATH),
            "status": "Model Loaded Successfully!!",
        }
        return jsonify(info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)