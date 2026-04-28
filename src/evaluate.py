import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

from tensorflow.keras.models import load_model

from preprocessing import load_data, preprocess_data, scale_data, create_sequences

DATA_PATH = "data/boiler_dataset.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

WINDOW_SIZE = 10


def evaluate():

    df = load_data(DATA_PATH)
    X = preprocess_data(df)

    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    X_seq = create_sequences(X_scaled, WINDOW_SIZE)

    model = load_model(MODEL_PATH, compile=False)

    recon = model.predict(X_seq, verbose=0)
    mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

    # ✅ FIXED threshold
    threshold = np.percentile(mse, 95)

    anomalies = (mse > threshold).astype(int)

    print("\nThreshold:", threshold)

    os.makedirs("reports", exist_ok=True)

    # ✅ SAVE EVERYTHING
    metrics = {
        "threshold": float(threshold),
        "total_points": int(len(mse)),
        "anomalies_detected": int(np.sum(anomalies)),
        "anomaly_rate": float(np.mean(anomalies))
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metrics saved correctly!")

    # Save results
    df_result = X.iloc[WINDOW_SIZE:].copy()
    df_result["error"] = mse
    df_result["anomaly"] = anomalies
    df_result.to_csv("reports/evaluation_results.csv")
    print("✅ Results saved to reports/evaluation_results.csv")

    # Plot
    plt.figure()
    plt.hist(mse[anomalies == 0], bins=50, alpha=0.6, label='Normal')
    plt.hist(mse[anomalies == 1], bins=50, alpha=0.6, label='Anomaly')

    plt.axvline(threshold, linestyle='--', color='red')
    plt.title("Reconstruction Error")
    plt.legend()

    plt.savefig("reports/lstm_error_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate()