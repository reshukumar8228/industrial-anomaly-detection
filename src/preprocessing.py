import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df


def preprocess_data(df):
    # Drop timestamp
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    X = df.drop(columns=['label'])
    y = df['label']

    return X, y


def scale_data(X, scaler_path=None, fit=True):
    scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return X_scaled


def get_normal_data(X_scaled, y):
    return X_scaled[y == 0]