import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

# Time
timestamps = pd.date_range(start="2024-01-01", periods=n, freq="min")

# Normal behavior
pressure = np.random.normal(20, 0.5, n)
temperature = np.random.normal(35, 0.5, n)
flow = np.random.normal(100, 1, n)
level = np.random.normal(70, 0.5, n)
vibration = np.random.normal(0.02, 0.002, n)

label = np.zeros(n)

# 🔥 Inject multiple anomaly types

# 1. Spikes
for i in range(300, 350):
    pressure[i] += np.random.uniform(5, 10)
    temperature[i] += np.random.uniform(5, 10)
    flow[i] += np.random.uniform(15, 25)
    vibration[i] += np.random.uniform(0.05, 0.1)
    label[i] = 1

# 2. Sudden drops
for i in range(700, 750):
    pressure[i] -= np.random.uniform(5, 8)
    flow[i] -= np.random.uniform(10, 20)
    level[i] -= np.random.uniform(5, 10)
    label[i] = 1

# 3. Drift anomaly
for i in range(1000, 1100):
    pressure[i] += (i - 1000) * 0.05
    temperature[i] += (i - 1000) * 0.03
    label[i] = 1

# 4. High vibration burst
for i in range(1300, 1350):
    vibration[i] += np.random.uniform(0.1, 0.2)
    label[i] = 1

# 5. Random anomalies
for i in range(1600, 1700):
    pressure[i] = np.random.uniform(10, 35)
    temperature[i] = np.random.uniform(25, 50)
    flow[i] = np.random.uniform(80, 130)
    vibration[i] = np.random.uniform(0.01, 0.3)
    label[i] = 1

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "pressure": pressure,
    "temperature": temperature,
    "flow": flow,
    "level": level,
    "vibration": vibration,
    "label": label
})

# Save
df.to_csv("data/high_anomaly_dataset.csv", index=False)

print("✅ High anomaly dataset created!")