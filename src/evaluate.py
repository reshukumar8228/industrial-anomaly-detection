import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import load_model

from src.preprocessing import load_data, preprocess_data, scale_data


# Paths
DATA_PATH = "data/industrial_sensor_dataset2.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/autoencoder.keras"   # or .h5 if you didn’t change


def evaluate():
    # 🔹 Load data
    df = load_data(DATA_PATH)

    # 🔹 Preprocess
    X, y = preprocess_data(df)

    # 🔹 Scale using saved scaler
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    # 🔹 Load trained model
    model = load_model(MODEL_PATH, compile=False)

    # 🔹 Reconstruction
    recon = model.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - recon), axis=1)

    # 🔹 Threshold (tuned for better precision)
    threshold = np.percentile(mse, 95)

    # 🔹 Predictions
    y_pred = (mse > threshold).astype(int)

    # 🔹 Metrics
    print("\nThreshold:", threshold)
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, mse))

    # 🔹 Visualization (FIXED SCALE)
    plt.figure(figsize=(8,5))

    plt.hist(mse[y == 0], bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(mse[y == 1], bins=50, alpha=0.6, label='Anomaly', density=True)

    plt.axvline(threshold, linestyle='--')

    plt.xlim(0, np.percentile(mse, 99))
    plt.yscale('log')  # ✅ makes anomalies visible

    plt.legend()
    plt.title("Reconstruction Error Distribution")

    plt.savefig("reports/error_distribution.png")
    plt.show()


if __name__ == "__main__":
    evaluate()