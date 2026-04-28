import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from preprocessing import load_data, preprocess_data, scale_data, create_sequences
from tensorflow.keras.models import load_model

DATA_PATH = "data/boiler_dataset.csv"
MODEL_PATH = "models/lstm_autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"

WINDOW_SIZE = 10


def visualize():
    # Load data
    df = load_data(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Preprocess
    X = preprocess_data(df)
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    # Create sequences
    X_seq = create_sequences(X_scaled, WINDOW_SIZE)

    # Load model
    model = load_model(MODEL_PATH, compile=False)

    # Predict
    recon = model.predict(X_seq)
    mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

    threshold = np.percentile(mse, 97)
    y_pred = (mse > threshold).astype(int)

    time = X.index[WINDOW_SIZE:]

    # 🔵 GRAPH 1: Sensor with anomalies
    plt.figure(figsize=(12, 4))

    sensor = X['pressure'].iloc[WINDOW_SIZE:]

    plt.plot(time, sensor, label='Normal')

    plt.scatter(time[y_pred == 1], sensor[y_pred == 1],
                color='red', label='Anomaly', s=10)

    plt.title("Sensor Signal with Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("reports/sensor_plot.png")
    plt.show()

    # 🔴 GRAPH 2: Reconstruction Error
    plt.figure(figsize=(12, 4))

    plt.plot(time, mse, label='Error')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')

    plt.title("Reconstruction Error")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("reports/error_plot.png")
    plt.show()


if __name__ == "__main__":
    visualize()