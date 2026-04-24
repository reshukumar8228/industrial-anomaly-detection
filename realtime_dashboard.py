import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json

# ================= CONFIG =================
st.set_page_config(layout="wide")

DATA_PATH = "data/high_anomaly_dataset.csv"
MODEL_PATH = "models/lstm_autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"

WINDOW_SIZE = 10
MAX_POINTS = 150
DISPLAY_WINDOW = 100

features = ["pressure", "temperature", "flow", "level", "vibration"]

units = {
    "pressure": "bar",
    "temperature": "°C",
    "flow": "L/min",
    "level": "cm",
    "vibration": "mm/s"
}

# ================= LOAD =================
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH, compile=False)

# ================= METRICS =================
metrics = {}
try:
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
except:
    pass

accuracy = metrics.get("accuracy", 0)
precision = metrics.get("precision", metrics.get("f1_score", 0))
recall = metrics.get("recall", 0)
f1 = metrics.get("f1_score", 0)

# ================= UI =================
st.title("🚀 Industrial Anomaly Detection Dashboard")

with st.sidebar:
    threshold_percentile = st.slider("Threshold Percentile", 80, 99, 85)
    run = st.button("Run Analysis")

# ================= SOUND =================
def play_sound():
    st.markdown("""
    <audio autoplay>
    <source src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# ================= HELPERS =================
def anomaly_context(rate):
    if rate < 5:
        return "Low"
    elif rate < 15:
        return "Moderate"
    else:
        return "High"

def severity_color(sev):
    return {
        "Normal": "🟢",
        "Medium": "🟠",
        "High": "🔴"
    }[sev]

# ================= RUN =================
if run:

    st.info("🟢 Live Streaming Started...")

    buffer, errors, anomaly_idx, severity_list = [], [], [], []
    sensor_data = {f: [] for f in features}
    timestamps = []

    threshold = 0

    # placeholders
    kpi = st.empty()
    status = st.empty()
    charts = st.empty()
    alert_box = st.empty()

    for i in range(min(len(df), MAX_POINTS)):

        row = df.iloc[i]
        t = row["timestamp"]

        timestamps.append(t)
        for f in features:
            sensor_data[f].append(row[f])

        buffer.append(row[features])
        if len(buffer) > WINDOW_SIZE:
            buffer.pop(0)

        if len(buffer) == WINDOW_SIZE:

            seq_df = pd.DataFrame(buffer, columns=features)
            seq_scaled = scaler.transform(seq_df)
            seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, -1)

            recon = model.predict(seq_scaled, verbose=0)
            error = np.mean((seq_scaled - recon) ** 2)

            errors.append(error)

            if len(errors) > 20:
                threshold = np.percentile(errors, threshold_percentile)

            # ===== SEVERITY =====
            if threshold > 0:
                if error > 2 * threshold:
                    severity = "High"
                elif error > threshold:
                    severity = "Medium"
                else:
                    severity = "Normal"
            else:
                severity = "Normal"

            severity_list.append(severity)

            # ===== SENSOR CONTRIBUTION =====
            sensor_error = np.mean((seq_scaled - recon) ** 2, axis=(1, 2))
            dominant_sensor = features[np.argmax(np.var(seq_scaled[0], axis=0))]

            # ===== KPI (CARD STYLE) =====
            if threshold > 0:

                confidence = max(0, min(100, (1 - error / (2 * threshold)) * 100))

                with kpi.container():
                    st.subheader("📊 KPI Summary")

                    c1, c2, c3, c4 = st.columns(4)
                    c5, c6, c7, c8 = st.columns(4)

                    c1.metric("Accuracy", f"{accuracy*100:.2f}%")
                    c2.metric("Precision", f"{precision:.2f}")
                    c3.metric("Recall", f"{recall:.2f}")
                    c4.metric("F1 Score", f"{f1:.2f}")

                    c5.metric("Total Points", len(timestamps))
                    c6.metric("Anomalies", len(anomaly_idx))
                    c7.metric("Threshold", f"{threshold:.4f}")
                    c8.metric("Confidence", f"{confidence:.2f}%")

                    st.caption(f"Confidence based on distance from threshold")

                    st.write(f"Threshold Percentile: {threshold_percentile}%")

            # ===== STATUS =====
            sys_state = "🟢 Live" if severity == "Normal" else "🔴 Fault Detected"

            with status.container():
                st.subheader("📡 System Status")
                st.write(f"Status: {sys_state}")
                st.write(f"Active Sensors: {len(features)}")
                st.write(f"Last Update: {t}")

            # ===== ALERT =====
            if severity != "Normal":
                anomaly_idx.append(i)
                play_sound()

                alert_box.warning(
                    f"""
⚠ {severity} Anomaly  
Sensor: {dominant_sensor.capitalize()}  
Error: {error:.4f}  
Time: {t}
                    """
                )

        else:
            errors.append(0)
            severity_list.append("Normal")

        # ================= CHARTS =================
        with charts.container():

            cols = st.columns(2)

            for idx, f in enumerate(features):

                fig = go.Figure()

                x = timestamps[-DISPLAY_WINDOW:]
                y = sensor_data[f][-DISPLAY_WINDOW:]

                fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

                mean = np.mean(y)
                std = np.std(y)

                fig.add_hrect(
                    y0=mean - std,
                    y1=mean + std,
                    fillcolor="cyan",
                    opacity=0.25
                )

                # anomalies
                ax, ay = [], []
                for j in range(len(x)):
                    gi = len(timestamps) - len(x) + j
                    if gi in anomaly_idx:
                        ax.append(x[j])
                        ay.append(y[j])

                fig.add_trace(go.Scatter(
                    x=ax,
                    y=ay,
                    mode='markers',
                    marker=dict(color="red", size=7)
                ))

                fig.update_layout(title=f"{f} ({units[f]})")

                cols[idx % 2].plotly_chart(fig, width="stretch")

            # ===== ERROR GRAPH (IMPROVED) =====
            fig_err = go.Figure()

            fig_err.add_trace(go.Scatter(
                x=timestamps[-DISPLAY_WINDOW:],
                y=errors[-DISPLAY_WINDOW:]
            ))

            if threshold > 0:
                fig_err.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold:.4f})"
                )

            fig_err.update_layout(
                title="Reconstruction Error",
                yaxis_title="Reconstruction Error"
            )

            st.plotly_chart(fig_err, width="stretch")

        time.sleep(0.05)

    # ================= FINAL SUMMARY =================
    st.markdown("---")
    st.success("✅ Streaming Completed")

    st.header("📊 Final Summary")

    total = len(timestamps)
    anomalies = len(anomaly_idx)
    rate = (anomalies / total) * 100 if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Points", total)
    c2.metric("Anomalies", anomalies)
    c3.metric("Anomaly Rate", f"{rate:.2f}%")

    # ===== FULL GRAPHS =====
    st.subheader("📈 Full Sensor Analysis")

    cols = st.columns(2)

    for idx, f in enumerate(features):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=sensor_data[f],
            mode='lines'
        ))

        ax = [timestamps[i] for i in anomaly_idx]
        ay = [sensor_data[f][i] for i in anomaly_idx]

        fig.add_trace(go.Scatter(
            x=ax,
            y=ay,
            mode='markers',
            marker=dict(color="red", size=7)
        ))

        fig.update_layout(title=f"{f} ({units[f]})")

        cols[idx % 2].plotly_chart(fig, width="stretch")

    # ===== FULL ERROR =====
    st.subheader("⚠️ Reconstruction Error (Full)")

    fig_err = go.Figure()

    fig_err.add_trace(go.Scatter(
        x=timestamps,
        y=errors
    ))

    if threshold > 0:
        fig_err.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red"
        )

    st.plotly_chart(fig_err, width="stretch")

    # ===== EXPORT =====
    df_export = pd.DataFrame({
        "timestamp": timestamps,
        "error": errors,
        "severity": severity_list
    })

    st.download_button(
        "⬇ Download Full Results",
        df_export.to_csv(index=False),
        "final_results.csv"
    )

    # ===== MODEL INFO =====
    st.markdown("---")
    st.subheader("🧪 Model Info")
    st.write("Model: LSTM Autoencoder")
    st.write(f"Window Size: {WINDOW_SIZE}")
    st.write(f"Threshold Percentile: {threshold_percentile}%")

    # ===== EXPLANATION =====
    st.markdown("---")
    st.subheader("🔍 Explanation")
    st.write("Low error → Normal")
    st.write("Medium error → Warning")
    st.write("High error → Critical anomaly")