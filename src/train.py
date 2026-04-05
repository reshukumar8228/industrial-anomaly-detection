import os
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, preprocess_data, scale_data, get_normal_data
from src.model import build_autoencoder


DATA_PATH = "data/industrial_sensor_dataset2.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/autoencoder.keras"


def train():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    X, y = preprocess_data(df)

    # Scale
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=True)

    # Train only on normal data
    X_train = get_normal_data(X_scaled, y)

    # Split for validation
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # Build model
    model = build_autoencoder(input_dim=X_train.shape[1])

    # Train
    history = model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        shuffle=True
    )

    # Save model
    model.save(MODEL_PATH)

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()