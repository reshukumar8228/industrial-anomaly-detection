import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Paths
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/autoencoder.keras"   # ✅ updated format


class AnomalyDetector:
    def __init__(self, threshold):
        # Load scaler
        self.scaler = joblib.load(SCALER_PATH)

        # Load model (no compile needed for inference)
        self.model = load_model(MODEL_PATH, compile=False)

        self.threshold = threshold

    def predict(self, sample):
        try:
            # Convert to numpy array
            sample = np.array(sample).reshape(1, -1)

            # Scale
            sample_scaled = self.scaler.transform(sample)

            # Reconstruction
            recon = self.model.predict(sample_scaled, verbose=0)

            # Error
            error = np.mean((sample_scaled - recon) ** 2)

            # Decision
            is_anomaly = error > self.threshold

            return {
                "error": float(error),
                "anomaly": bool(is_anomaly)
            }

        except Exception as e:
            return {"error": str(e), "anomaly": None}


# ✅ Run test when file is executed
if __name__ == "__main__":
    # Use your evaluated threshold
    threshold = 0.08   # adjust based on your evaluation output

    detector = AnomalyDetector(threshold=threshold)

    print("\n🔍 Testing Inference...\n")

    # Option 1: Manual sample
    sample = [20.5, 35.0, 100.0, 70.0, 0.02]

    result = detector.predict(sample)

    print("Input Sample:", sample)
    print("Prediction:", result)

    # Option 2: Test with real dataset row
    try:
        df = pd.read_excel("data/industrial_sensor_dataset2.xlsx")
        real_sample = df.iloc[0].drop("label").values.tolist()

        result_real = detector.predict(real_sample)

        print("\nReal Data Sample:", real_sample)
        print("Prediction:", result_real)

    except Exception as e:
        print("\nCould not load dataset sample:", e)