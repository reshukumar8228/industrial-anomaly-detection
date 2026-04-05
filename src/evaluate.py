import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocessing import load_data, preprocess_data, scale_data
from tensorflow.keras.models import load_model

DATA_PATH = "data/industrial_sensor_dataset2.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/autoencoder.h5"


def evaluate():
    # Load data
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)

    # Scale using saved scaler
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

    # Load model
    model = load_model(MODEL_PATH)

    # Reconstruction
    recon = model.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - recon), axis=1)

    # Threshold (95th percentile)
    threshold = np.percentile(mse, 95)

    # Predictions
    y_pred = (mse > threshold).astype(int)

    # Metrics
    print("Threshold:", threshold)
    print(classification_report(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, mse))


if __name__ == "__main__":
    evaluate()