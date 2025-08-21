from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import os
import traceback


# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bhp_model_compressed.joblib"

print("MODEL_PATH -> " , MODEL_PATH)
print("MODEL_EXISTS? ->" , MODEL_PATH.exists())

app = Flask(__name__)


@app.route("/debug-files", methods=["GET"])
def debug_files():
    try:
        base_dir = Path(__file__).resolve().parent
        models_dir = base_dir / "models"

        # List all files inside /models
        if models_dir.exists():
            files = os.listdir(models_dir)
        else:
            files = ["❌ models directory not found"]

        return jsonify({
            "base_dir": str(base_dir),
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "models_dir_files": files
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/test-model", methods=["GET"])
def test_model():
    try:
        import os
        from pathlib import Path
        import joblib

        path = Path(__file__).resolve().parent / "models" / "bhp_model_compressed.joblib"
        exists = path.exists()

        if not exists:
            return jsonify({"status": "not found", "path": str(path)})

        # Try loading
        model = joblib.load(path)
        return jsonify({
            "status": "loaded",
            "model_type": str(type(model)),
            "steps_attr": hasattr(model, "named_steps")
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route("/debug-model", methods=["GET"])
def debug_model():
    try:
        model = get_model()
        steps = list(model.named_steps.keys()) if hasattr(model, "named_steps") else "No named_steps"
        return jsonify({
            "model_type": str(type(model)),
            "steps": str(steps)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

#cache variable
_cached_model = None

def get_model():
    """Load the model only once when needed."""
    global _cached_model

    if _cached_model is None:
        print("Loading model from file...")  # Runs only first time
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _cached_model = joblib.load(MODEL_PATH)
    else:
        print("Using cached model...")       # Runs every other time

    return _cached_model


@app.route("/")
def home():
    return jsonify({
        "Status": "OK",
        "message": "Welcome to Bengaluru House Price Prediction API 🚀",
        "routes" : {
            "/health" : "Check health status",
            "/predict" : "Send JSON with features to get predictions",
        },
        "endpoints": ["/health", "/predict", "/locations", "/model_info"]
    })


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
        print("🔍 /locations endpoint hit")

        # Load model
        model = get_model()
        print("✅ Model Loaded:", model)

        # Show pipeline steps
        if not hasattr(model, "named_steps"):
            return jsonify({"error": "Model has no named_steps (not a Pipeline?)"}), 500
        print("📌 Model steps:", list(model.named_steps.keys()))

        # Get preprocessor
        if "pre" not in model.named_steps:
            return jsonify({"error": "No 'pre' step found in model pipeline"}), 500
        preprocessor = model.named_steps["pre"]
        print("✅ Found preprocessor step")

        # Inspect transformers inside preprocessor
        if not hasattr(preprocessor, "transformers_"):
            return jsonify({"error": "Preprocessor has no transformers_"}), 500
        print("📌 Preprocessor transformers:", preprocessor.transformers_)

        # Find categorical transformer
        cat_transformer = None
        for name, transformer, columns in preprocessor.transformers_:
            print(f"🔎 Checking transformer: {name}, columns={columns}")
            if name == "cat":
                cat_transformer = transformer
                print("✅ Found categorical transformer:", cat_transformer)
                break

        if cat_transformer is None:
            return jsonify({"error": "Categorical transformer not found (check its name)"}), 500

        # Inspect categorical pipeline steps
        if not hasattr(cat_transformer, "named_steps"):
            return jsonify({"error": "Categorical transformer has no named_steps"}), 500
        print("📌 Cat transformer steps:", list(cat_transformer.named_steps.keys()))

        # Get OneHotEncoder
        if "ohe" not in cat_transformer.named_steps:
            return jsonify({"error": "No 'ohe' step found in categorical transformer"}), 500
        ohe = cat_transformer.named_steps["ohe"]
        print("✅ Found OneHotEncoder:", ohe)

        # Extract locations
        if hasattr(ohe, "categories_"):
            locations = list(ohe.categories_[0])
            print("✅ Extracted locations:", locations)
            return jsonify({"locations": locations})
        else:
            return jsonify({"error": "OneHotEncoder is not fitted"}), 500

    except Exception as e:
        print("❌ Exception in /locations:", str(e))
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
        print("error in /locations" ,traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
