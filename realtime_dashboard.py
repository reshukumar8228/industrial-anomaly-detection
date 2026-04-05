import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# ================= CONFIG =================
st.set_page_config(layout="wide")

DATA_PATH = "data/high_anomaly_dataset.csv"
MODEL_PATH = "models/lstm_autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"

WINDOW_SIZE = 10
MAX_POINTS = 150
DISPLAY_WINDOW = 100

features = ["pressure", "temperature", "flow", "level", "vibration"]

# ================= LOAD =================
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH, compile=False)

# ================= UI =================
st.title("🚀 Industrial Real-Time Anomaly Detection Dashboard")
st.caption("Real-time LSTM-based anomaly detection using reconstruction loss")

with st.sidebar:
    threshold_percentile = st.slider("Threshold Percentile", 90, 99, 97)
    run = st.button("Run Analysis")

# ================= SOUND FIX =================
def play_sound():
    st.markdown("""
        <audio autoplay>
        <source src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)

# ================= MAIN =================
if run:

    buffer = []
    errors = []
    anomaly_idx = []

    sensor_data = {f: [] for f in features}
    timestamps = []

    threshold = 0

    placeholder = st.empty()
    alert_placeholder = st.empty()  # ✅ FIX notification spam

    for i in range(min(len(df), MAX_POINTS)):

        row = df.iloc[i]
        t = row["timestamp"]

        timestamps.append(t)

        for f in features:
            sensor_data[f].append(row[f])

        buffer.append(row[features])

        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

        # ================= MODEL =================
        if len(buffer) == WINDOW_SIZE:

            seq_df = pd.DataFrame(buffer, columns=features)
            seq_scaled = scaler.transform(seq_df)
            seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, -1)

            recon = model.predict(seq_scaled, verbose=0)
            error = np.mean((seq_scaled - recon) ** 2)

            errors.append(error)

            if len(errors) > 10:
                threshold = np.percentile(errors, threshold_percentile)

            if threshold > 0 and error > threshold:
                anomaly_idx.append(i)

                # 🔊 SOUND
                play_sound()

                # 🚨 SINGLE ALERT
                alert_placeholder.error(f"⚠️ Anomaly detected at {t}")

        else:
            errors.append(0)

        # ================= LIVE DISPLAY =================
        with placeholder.container():

            st.subheader("📡 Sensor Signals (Live)")

            cols = st.columns(2)

            for idx, f in enumerate(features):

                fig = go.Figure()

                # ✅ MOVING WINDOW
                x_window = timestamps[-DISPLAY_WINDOW:]
                y_window = sensor_data[f][-DISPLAY_WINDOW:]

                fig.add_trace(go.Scatter(
                    x=x_window,
                    y=y_window,
                    mode='lines',
                    name='Normal'
                ))

                # anomalies in window
                anomaly_x = [timestamps[j] for j in anomaly_idx if j >= len(timestamps)-DISPLAY_WINDOW]
                anomaly_y = [sensor_data[f][j] for j in anomaly_idx if j >= len(timestamps)-DISPLAY_WINDOW]

                fig.add_trace(go.Scatter(
                    x=anomaly_x,
                    y=anomaly_y,
                    mode='markers',
                    marker=dict(color='red', size=6),
                    name='Anomaly'
                ))

                fig.update_layout(title=f"{f.capitalize()} Sensor")

                cols[idx % 2].plotly_chart(fig, width="stretch")

            # ================= ERROR GRAPH =================
            st.subheader("⚠️ Reconstruction Error")

            fig_err = go.Figure()

            fig_err.add_trace(go.Scatter(
                x=timestamps[-DISPLAY_WINDOW:],
                y=errors[-DISPLAY_WINDOW:],
                mode='lines',
                name='Error'
            ))

            if threshold > 0:
                fig_err.add_hline(y=threshold, line_dash="dash", line_color="red")

            fig_err.update_layout(title="Reconstruction Error")

            st.plotly_chart(fig_err, width="stretch")

            # ================= METRICS =================
            col1, col2 = st.columns(2)
            col1.metric("Total Points", len(timestamps))
            col2.metric("Anomalies Detected", len(anomaly_idx))

        time.sleep(0.05)

    # ================= FINAL =================
    st.success("✅ Streaming Completed (150 points)")

    st.header("📊 Final Summary (Full Data)")

    cols = st.columns(2)

    for idx, f in enumerate(features):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=sensor_data[f],
            mode='lines'
        ))

        anomaly_x = [timestamps[j] for j in anomaly_idx]
        anomaly_y = [sensor_data[f][j] for j in anomaly_idx]

        fig.add_trace(go.Scatter(
            x=anomaly_x,
            y=anomaly_y,
            mode='markers',
            marker=dict(color='red', size=6)
        ))

        fig.update_layout(title=f"{f.capitalize()} Sensor (Final)")

        cols[idx % 2].plotly_chart(fig, width="stretch")

    # FINAL ERROR GRAPH
    fig_err = go.Figure()

    fig_err.add_trace(go.Scatter(
        x=timestamps,
        y=errors,
        mode='lines'
    ))

    if threshold > 0:
        fig_err.add_hline(y=threshold, line_dash="dash", line_color="red")

    fig_err.update_layout(title="Final Reconstruction Error")

    st.plotly_chart(fig_err, width="stretch")