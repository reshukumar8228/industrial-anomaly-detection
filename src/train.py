import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, preprocess_data, scale_data, create_sequences
from src.model import build_lstm_autoencoder

# Paths
DATA_PATH = "data/high_anomaly_dataset.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

WINDOW_SIZE = 10


def train():
    # ================= LOAD =================
    df = load_data(DATA_PATH)

    # ================= PREPROCESS =================
    X, y = preprocess_data(df)

    # ================= SCALE =================
    X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=True)

    # ================= CREATE SEQUENCES =================
    X_seq = create_sequences(X_scaled, WINDOW_SIZE)
    y_seq = y[WINDOW_SIZE:].reset_index(drop=True)

    # ================= TRAIN DATA (ONLY NORMAL) =================
    X_train = X_seq[y_seq == 0]

    # ================= SPLIT =================
    X_train, X_val = train_test_split(
        X_train, test_size=0.2, random_state=42
    )

    # ================= BUILD MODEL =================
    model = build_lstm_autoencoder(
        timesteps=WINDOW_SIZE,
        n_features=X_train.shape[2]
    )

    # ================= TRAIN =================
    print("🚀 Training started...\n")

    history = model.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, X_val),
        verbose=1  # ✅ shows epoch logs in terminal
    )

    # ================= SAVE MODEL =================
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print("\n✅ LSTM Training complete. Model saved.")

    # ================= SAVE LOSS GRAPH =================
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title("MSE Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save graph
    plt.savefig("reports/training_loss.png")

    print("📊 Training loss graph saved at: reports/training_loss.png")

    # Optional: show graph
    plt.show()


if __name__ == "__main__":
    train()