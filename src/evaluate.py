import numpy as np
import matplotlib.pyplot as plt
import json
import os

from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model

from src.preprocessing import load_data, preprocess_data, scale_data, create_sequences

# Paths
DATA_PATH = "data/high_anomaly_dataset.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

WINDOW_SIZE = 10


def evaluate():
    # ================= LOAD =================
    df = load_data(DATA_PATH)

    # ================= PREPROCESS =================
    X, y = preprocess_data(df)

    # ================= SCALE =================
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    # ================= CREATE SEQUENCES =================
    X_seq = create_sequences(X_scaled, WINDOW_SIZE)
    y_seq = y[WINDOW_SIZE:].reset_index(drop=True)

    # ================= LOAD MODEL =================
    model = load_model(MODEL_PATH, compile=False)

    # ================= RECONSTRUCTION =================
    recon = model.predict(X_seq, verbose=0)
    mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

    # ================= THRESHOLD =================
    threshold = np.percentile(mse, 85)

    # ================= PREDICTIONS =================
    y_pred = (mse > threshold).astype(int)

    # ================= METRICS =================
    print("\nThreshold:", threshold)
    print("\nClassification Report:\n")
    print(classification_report(y_seq, y_pred))
    print("ROC-AUC:", roc_auc_score(y_seq, mse))

    # ================= SAVE RESULTS =================
    os.makedirs("reports", exist_ok=True)

    df_result = df.iloc[WINDOW_SIZE:].copy()
    df_result["error"] = mse
    df_result["prediction"] = y_pred

    df_result.to_csv("reports/lstm_results.csv", index=False)

    # ================= SAVE METRICS (FOR DASHBOARD) =================
    metrics = {
        "accuracy": float(accuracy_score(y_seq, y_pred)),
        "recall": float(recall_score(y_seq, y_pred)),
        "f1_score": float(f1_score(y_seq, y_pred)),
        "roc_auc": float(roc_auc_score(y_seq, mse)),
        "total_points": int(len(y_seq)),
        "anomalies_detected": int(np.sum(y_pred))
    }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metrics saved to reports/metrics.json")

    # ================= PLOT =================
    plt.figure()

    plt.hist(mse[y_seq == 0], bins=50, alpha=0.6, label='Normal')
    plt.hist(mse[y_seq == 1], bins=50, alpha=0.6, label='Anomaly')

    plt.axvline(threshold, linestyle='--', color='red')
    plt.xlim(0, np.percentile(mse, 99))

    plt.legend()
    plt.title("LSTM Reconstruction Error")

    plt.savefig("reports/lstm_error_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate()