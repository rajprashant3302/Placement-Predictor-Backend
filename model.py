import sys
import json
import pandas as pd
import joblib
import os

try:
    # --- Read JSON input from command-line argument ---
    if len(sys.argv) < 2:
        raise ValueError("No input received from Node.js")
    
    raw_input = sys.argv[1]
    features = json.loads(raw_input)

    # --- Load model ---
    model_path = os.path.join(os.path.dirname(__file__), "placement_package_predictor.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    pipeline = joblib.load(model_path)

    # --- Convert features to DataFrame ---
    df_in = pd.DataFrame([features])

    # --- Predict ---
    predicted_package = pipeline.predict(df_in)
    predicted_value = max(0, float(predicted_package[0]))  # Ensure non-negative

    # --- Output ---
    print(predicted_value, flush=True)

except Exception as e:
    # Send error to stderr
    sys.stderr.write(str(e) + "\n")
    print("0", flush=True)  # Return 0 if error occurs
    sys.exit(1)
