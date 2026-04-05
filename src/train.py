import os
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, preprocess_data, scale_data, create_sequences
from src.model import build_lstm_autoencoder

# Paths
DATA_PATH = "data/industrial_sensor_dataset2.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

WINDOW_SIZE = 10


def train():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    X, y = preprocess_data(df)

    # Scale
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=True)

    # Create sequences
    X_seq = create_sequences(X_scaled, WINDOW_SIZE)
    y_seq = y[WINDOW_SIZE:].reset_index(drop=True)

    # Train only on normal data
    X_train = X_seq[y_seq == 0]

    # Split
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Build model
    model = build_lstm_autoencoder(
        timesteps=WINDOW_SIZE,
        n_features=X_train.shape[2]
    )

    # Train
    model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val)
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print("✅ LSTM Training complete. Model saved.")


if __name__ == "__main__":
    train()