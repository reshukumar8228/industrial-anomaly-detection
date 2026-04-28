import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def preprocess_data(df):
    # Convert and sort timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)

    # Keep only numeric columns
    df = df.select_dtypes(include=['float64', 'int64'])

    # Ensure unsupervised learning (drop label if exists)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # Handle missing values
    df = df.ffill().bfill()

    return df


def scale_data(X, scaler_path=None, fit=True):
    scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)

        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return X_scaled


def create_sequences(X, window_size=10):
    sequences = []

    for i in range(len(X) - window_size):
        sequences.append(X[i:i + window_size])

    return np.array(sequences)