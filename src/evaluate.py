import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import load_model

from src.preprocessing import load_data, preprocess_data, scale_data, create_sequences

# Paths
DATA_PATH = "data/industrial_sensor_dataset2.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

WINDOW_SIZE = 10


def evaluate():
    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    X, y = preprocess_data(df)

    # Scale
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    # Create sequences
    X_seq = create_sequences(X_scaled, WINDOW_SIZE)
    y_seq = y[WINDOW_SIZE:].reset_index(drop=True)

    # Load model
    model = load_model(MODEL_PATH, compile=False)

    # Reconstruction
    recon = model.predict(X_seq)
    mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

    # Threshold
    threshold = np.percentile(mse, 97)

    # Predictions
    y_pred = (mse > threshold).astype(int)

    # Metrics
    print("\nThreshold:", threshold)
    print("\nClassification Report:\n")
    print(classification_report(y_seq, y_pred))
    print("ROC-AUC:", roc_auc_score(y_seq, mse))

    # Save results
    os.makedirs("reports", exist_ok=True)

    df_result = df.iloc[WINDOW_SIZE:].copy()
    df_result["error"] = mse
    df_result["prediction"] = y_pred

    df_result.to_csv("reports/lstm_results.csv", index=False)

    # Plot
    plt.figure()

    plt.hist(mse[y_seq == 0], bins=50, alpha=0.6, label='Normal')
    plt.hist(mse[y_seq == 1], bins=50, alpha=0.6, label='Anomaly')

    plt.axvline(threshold, linestyle='--')
    plt.xlim(0, np.percentile(mse, 99))

    plt.legend()
    plt.title("LSTM Reconstruction Error")

    plt.savefig("reports/lstm_error_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate()