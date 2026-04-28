import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import json
import os

# ================= CONFIG =================
st.set_page_config(layout="wide")

DATA_PATH = "data/boiler_dataset.csv"
MODEL_PATH = "models/lstm_autoencoder.keras"
SCALER_PATH = "models/scaler.pkl"

WINDOW_SIZE = 10
MAX_POINTS = 200
DISPLAY_WINDOW = 200

# ================= LOAD =================
df = pd.read_csv(DATA_PATH)

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
elif 'date' in df.columns:
    df['timestamp'] = pd.to_datetime(df['date'])
else:
    raise ValueError("No timestamp/date column found")
df = df.sort_values('timestamp')

features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'timestamp' in features:
    features.remove('timestamp')

# For UI clarity, only show first 5 sensors
display_features = features[:5]

scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH, compile=False)

# ================= METRICS =================
metrics = {}
try:
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
except:
    pass

# ================= UI =================
st.title("🚀 Industrial Anomaly Detection Dashboard")

with st.sidebar:
    threshold_percentile = st.slider("Threshold Percentile", 80, 99, 85)
    selected_sensor = st.selectbox("Select Sensor", features)
    run = st.button("Run Analysis")

    st.markdown("---")
    st.subheader("📉 Training Loss")
    if os.path.exists("reports/training_loss.png"):
        st.image("reports/training_loss.png", use_container_width=True)
    else:
        st.info("Run train.py to generate training loss graph.")


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

# ================= RUN =================
if run:

    st.info("🟢 Live Streaming Started...")

    buffer, errors, anomaly_idx, severity_list = [], [], [], []
    sensor_data = {f: [] for f in features}
    timestamps = []

    # placeholders
    st.markdown("### 📊 KPI Summary")
    kpi = st.empty()
    status = st.empty()
    charts = st.empty()
    alert_box = st.empty()

    start_idx = np.random.randint(0, len(df) - MAX_POINTS)

    for i in range(start_idx, start_idx + MAX_POINTS):

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

            recon = model(seq_scaled, training=False).numpy()

            # 🔥 FIXED ERROR FORMULA (balanced)
            error = np.mean((seq_scaled - recon) ** 2)
            errors.append(error)

            if len(errors) == 0:
                st.warning("No data available to compute threshold")
                st.stop()

            threshold = np.percentile(errors, threshold_percentile)

            # ===== SEVERITY =====
            if error > 2 * threshold:
                severity = "High"
            elif error > threshold:
                severity = "Medium"
            else:
                severity = "Normal"

            severity_list.append(severity)

            # ===== SENSOR CONTRIBUTION =====
            feature_errors = np.mean((seq_scaled[0] - recon[0]) ** 2, axis=0)
            dominant_sensor = features[np.argmax(feature_errors)]

            # ===== KPI =====
            confidence = min(100, (error / threshold) * 100) if threshold > 1e-9 else 0

            with kpi.container():
                c1, c2, c3 = st.columns(3)
                c4, c5 = st.columns(2)

                c1.metric("Total Data Points", len(timestamps))
                c2.metric("Anomalies Detected", len(anomaly_idx))
                
                anomaly_rate = (len(anomaly_idx) / len(timestamps)) * 100 if len(timestamps) > 0 else 0
                c3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

                c4.metric("Threshold Value", f"{threshold:.6f}")
                c5.metric("Threshold Percentile", f"{threshold_percentile}%")

            # ===== STATUS =====
            anomalies_detected = len(anomaly_idx)
            if anomalies_detected > 0:
                sys_state = "🔴 Fault Detected"
            else:
                sys_state = "🟢 Normal"

            with status.container():
                st.subheader("📡 System Status")
                st.write(f"Status: {sys_state}")
                st.write(f"Active Sensors: {len(features)}")
                st.write(f"Last Update: {t}")

            # ===== ALERT =====
            if severity != "Normal":
                anomaly_idx.append(len(timestamps) - 1)
                play_sound()

                alert_box.warning(
                    f"""
⚠ {severity} Anomaly  
Sensor: {dominant_sensor.capitalize()}  
Error: {error:.4f}  
Threshold: {threshold:.4f}  
Confidence: {confidence:.2f}%  
Time: {t}
                    """
                )

        else:
            continue

        # ================= CHARTS =================
        with charts.container():

            st.subheader(f"🔍 Detailed View: {selected_sensor}")
            fig_sel = go.Figure()

            x_sel = timestamps[-DISPLAY_WINDOW:]
            y_sel = sensor_data[selected_sensor][-DISPLAY_WINDOW:]

            fig_sel.add_trace(go.Scatter(x=x_sel, y=y_sel, mode='lines'))

            mean_sel = np.mean(y_sel)
            std_sel = np.std(y_sel)
            fig_sel.add_hrect(y0=mean_sel - std_sel, y1=mean_sel + std_sel, fillcolor="cyan", opacity=0.25)

            ax_sel, ay_sel = [], []
            for j in range(len(x_sel)):
                gi = len(timestamps) - len(x_sel) + j
                if gi in anomaly_idx:
                    ax_sel.append(x_sel[j])
                    ay_sel.append(y_sel[j])

            fig_sel.add_trace(go.Scatter(x=ax_sel, y=ay_sel, mode='markers', marker=dict(color="red", size=7)))
            fig_sel.update_layout(title=f"{selected_sensor}")
            st.plotly_chart(fig_sel, width="stretch", key=f"plot_sel_{i}")

            st.subheader("📊 Overview (First 5 Sensors)")

            cols = st.columns(2)

            for idx, f in enumerate(display_features):

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

                fig.update_layout(title=f"{f}")

                cols[idx % 2].plotly_chart(fig, width="stretch", key=f"plot_{f}_{i}")

            # ===== ERROR GRAPH =====
            fig_err = go.Figure()

            min_len = min(len(timestamps), len(errors))
            x_vals = timestamps[-min_len:]
            y_vals = errors[-min_len:]

            fig_err.add_trace(go.Scatter(
                x=x_vals[-DISPLAY_WINDOW:],
                y=y_vals[-DISPLAY_WINDOW:]
            ))

            fig_err.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold:.4f})"
            )

            fig_err.update_layout(
                title="Reconstruction Error",
                xaxis_title="Time",
                yaxis_title="Reconstruction Error"
            )

            st.plotly_chart(fig_err, width="stretch", key=f"plot_err_{i}")

        time.sleep(0.05)

    # ================= FINAL SUMMARY =================
    st.success("✅ Streaming Completed")

    st.markdown("## 📊 Final Summary")

    total = len(timestamps)
    anomalies = len(anomaly_idx)
    rate = (anomalies / total) * 100 if total > 0 else 0

    context = anomaly_context(rate)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Points", total)
    c2.metric("Anomalies", anomalies)
    c3.metric("Anomaly Rate", f"{rate:.2f}% ({context})")

    # ===== FULL GRAPHS =====
    st.subheader(f"🔍 Detailed View: {selected_sensor} (Full)")

    fig_sel_full = go.Figure()
    fig_sel_full.add_trace(go.Scatter(x=timestamps, y=sensor_data[selected_sensor], mode='lines'))

    ax_sel_full = [timestamps[i] for i in anomaly_idx]
    ay_sel_full = [sensor_data[selected_sensor][i] for i in anomaly_idx]

    fig_sel_full.add_trace(go.Scatter(
        x=ax_sel_full,
        y=ay_sel_full,
        mode='markers',
        marker=dict(color="red", size=7)
    ))
    fig_sel_full.update_layout(title=f"{selected_sensor}")
    st.plotly_chart(fig_sel_full, width="stretch", key="main_chart_sel_full")

    st.subheader("📈 Full Sensor Analysis (Overview)")

    cols = st.columns(2)

    for idx, f in enumerate(display_features):

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

        fig.update_layout(title=f"{f}")

        cols[idx % 2].plotly_chart(fig, width="stretch", key=f"plot_{f}_{idx}_full")

    # ===== ERROR GRAPH FULL =====
    st.subheader("⚠️ Reconstruction Error (Full)")

    errors = [e for e in errors if not np.isnan(e)]
    min_len_full = min(len(timestamps), len(errors))

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=timestamps[-min_len_full:], y=errors[-min_len_full:]))

    fig_err.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold:.4f})"
    )

    st.plotly_chart(fig_err, width="stretch", key="main_chart_err_full")

    # ===== EXPORT =====
    min_len = min(len(timestamps), len(errors), len(severity_list))

    df_export = pd.DataFrame({
        "timestamp": timestamps[-min_len:],
        "error": errors[-min_len:],
        "severity": severity_list[-min_len:]
    })

    st.download_button(
        "⬇ Download Full Results",
        df_export.to_csv(index=False),
        "final_results.csv"
    )

    # ===== MODEL INFO =====
    st.subheader("🧪 Model Info")
    st.write("Model: LSTM Autoencoder")
    st.write(f"Window Size: {WINDOW_SIZE}")
    st.write(f"Threshold Percentile: {threshold_percentile}%")
    st.write(f"Threshold Value: {threshold:.4f}")

    # ===== EXPLANATION =====
    st.subheader("🔍 Explanation")
    st.write("Low error → Normal")
    st.write("Medium error → Warning")
    st.write("High error → Critical anomaly")