import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model

from src.preprocessing import load_data, preprocess_data, scale_data, create_sequences

# Config
st.set_page_config(layout="wide")

DATA_PATH = "data/high_anomaly_dataset.csv"
MODEL_PATH = "models/lstm_autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"
WINDOW_SIZE = 10

# Title
st.title("🚀 Industrial Anomaly Detection Dashboard")

# Sidebar
st.sidebar.header("Controls")

sensor_choice = st.sidebar.selectbox(
    "Select Sensor",
    ["pressure", "temperature", "flow", "level", "vibration"]
)

threshold_percentile = st.sidebar.slider(
    "Threshold Percentile", 90, 99, 97
)

run = st.sidebar.button("Run Analysis")

# Load data
df = load_data(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

X, y = preprocess_data(df)
X_scaled = scale_data(X, scaler_path=SCALER_PATH, fit=False)

X_seq = create_sequences(X_scaled, WINDOW_SIZE)
y_seq = y[WINDOW_SIZE:].reset_index(drop=True)

model = load_model(MODEL_PATH, compile=False)

recon = model.predict(X_seq)
mse = np.mean((X_seq - recon) ** 2, axis=(1, 2))

threshold = np.percentile(mse, threshold_percentile)
y_pred = (mse > threshold).astype(int)

time = df['timestamp'][WINDOW_SIZE:]
sensor_data = df[sensor_choice][WINDOW_SIZE:]

# Layout
col1, col2 = st.columns([3, 1])

# 📊 MAIN GRAPH
with col1:
    st.subheader("📊 Sensor Signal with Anomalies")

    fig = px.line(x=time, y=sensor_data)

    fig.add_scatter(
        x=time[y_pred == 1],
        y=sensor_data[y_pred == 1],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Error graph
    st.subheader("⚠ Reconstruction Error")

    fig2 = px.line(x=time, y=mse)

    fig2.add_hline(y=threshold, line_dash="dash", line_color="red")

    st.plotly_chart(fig2, use_container_width=True)

# 📊 SIDE PANEL
with col2:
    st.subheader("📈 Summary")

    st.metric("Total Windows", len(mse))
    st.metric("Detected Anomalies", int(np.sum(y_pred)))
    st.metric("Anomaly Rate", f"{np.mean(y_pred)*100:.2f}%")

    st.subheader("🔍 Sensor Contribution")

    idx = st.slider("Select anomaly index", 0, len(X_seq)-1, 0)

    error = np.abs(X_seq[idx] - recon[idx])
    contribution = np.mean(error, axis=0)

    features = ["pressure", "temperature", "flow", "level", "vibration"]

    contrib_df = pd.DataFrame({
        "feature": features,
        "contribution": contribution
    })

    st.bar_chart(contrib_df.set_index("feature"))