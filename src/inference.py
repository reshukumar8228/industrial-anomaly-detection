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
    # Dynamically generates a random array shape matching your model's expected inputs
    n_features = detector.scaler.n_features_in_
    samples = np.random.rand(15, n_features).tolist()

    for s in samples:
        result = detector.predict(s)
        print(result)