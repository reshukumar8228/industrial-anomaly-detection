import numpy as np
import joblib
from tensorflow.keras.models import load_model

SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"


class LSTMAnomalyDetector:
    def __init__(self, threshold, window_size=10):
        self.scaler = joblib.load(SCALER_PATH)
        self.model = load_model(MODEL_PATH, compile=False)
        self.threshold = threshold
        self.window_size = window_size
        self.buffer = []

    def predict(self, sample):
        self.buffer.append(sample)

        if len(self.buffer) < self.window_size:
            return {"status": "waiting_for_sequence"}

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        sequence = np.array(self.buffer)
        sequence_scaled = self.scaler.transform(sequence)
        sequence_scaled = sequence_scaled.reshape(1, self.window_size, -1)

        recon = self.model.predict(sequence_scaled, verbose=0)
        error = np.mean((sequence_scaled - recon) ** 2)

        return {
            "error": float(error),
            "anomaly": error > self.threshold
        }


if __name__ == "__main__":
    detector = LSTMAnomalyDetector(threshold=0.1)

    # Simulate streaming
    samples = [
        [20.5, 35.0, 100.0, 70.0, 0.02],
        [20.6, 35.1, 100.5, 70.2, 0.02],
        [20.7, 35.2, 101.0, 70.1, 0.02],
        [20.8, 35.3, 101.2, 70.3, 0.02],
        [20.9, 35.4, 101.5, 70.4, 0.02],
        [21.0, 35.5, 101.8, 70.5, 0.02],
        [21.2, 35.6, 102.0, 70.6, 0.02],
        [21.3, 35.7, 102.2, 70.7, 0.02],
        [21.5, 35.8, 102.5, 70.8, 0.02],
        [21.6, 35.9, 102.7, 70.9, 0.02],
    ]

    for s in samples:
        result = detector.predict(s)
        print(result)