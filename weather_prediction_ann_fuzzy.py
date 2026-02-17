# ============================================================
# Weather Prediction using ANN + Fuzzy Logic (Multi-Output)
# Author: Charles Prathipati
# Description:
# Hybrid soft computing model for next-day weather forecasting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "weather_prediction_dataset.csv"

FEATURES = [
    'BASEL_temp_mean',
    'BASEL_humidity',
    'DE_BILT_wind_speed',
    'BASEL_pressure'
]

EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


# -----------------------------
# Load and Prepare Data
# -----------------------------
df = pd.read_csv(DATASET_PATH)
df = df[FEATURES].dropna()

# Create next-day prediction setup
X_raw = df.iloc[:-1]
y_raw = df.iloc[1:]

# Separate scalers for inputs and outputs
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)


# -----------------------------
# Fuzzy Logic: Temperature
# -----------------------------
temp_range = np.arange(0, 51, 1)

temp_low = fuzz.trimf(temp_range, [0, 0, 25])
temp_medium = fuzz.trimf(temp_range, [15, 25, 35])
temp_high = fuzz.trimf(temp_range, [30, 50, 50])


def fuzzify_temperature(temp):
    return [
        fuzz.interp_membership(temp_range, temp_low, temp),
        fuzz.interp_membership(temp_range, temp_medium, temp),
        fuzz.interp_membership(temp_range, temp_high, temp)
    ]


# Generate fuzzy temperature inputs
fuzzy_temps = np.array([
    fuzzify_temperature(temp)
    for temp in X_raw['BASEL_temp_mean'].values
])

# Combine fuzzy temperature with other normalized features
other_features = X_scaled[:, 1:]  # exclude temperature
X_final = np.hstack((fuzzy_temps, other_features))


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y_scaled,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# -----------------------------
# ANN Model
# -----------------------------
model = Sequential()
model.add(Dense(12, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))  # multi-output regression

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# -----------------------------
# Training
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1
)


# -----------------------------
# Training Curve
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Evaluation
# -----------------------------
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")


# -----------------------------
# Prediction (Proper Inverse Scaling)
# -----------------------------
y_pred_scaled = model.predict(X_test)

y_pred_real = y_scaler.inverse_transform(y_pred_scaled)
y_test_real = y_scaler.inverse_transform(y_test)


# -----------------------------
# Prediction vs Actual Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(y_test_real[:50, 0], label='Actual Temperature')
plt.plot(y_pred_real[:50, 0], label='Predicted Temperature')
plt.xlabel('Sample')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Single Sample Inference
# -----------------------------
sample = X_raw.iloc[100]

sample_df = pd.DataFrame([sample], columns=FEATURES)
sample_scaled = x_scaler.transform(sample_df)

fuzzy_temp = fuzzify_temperature(sample['BASEL_temp_mean'])
other_feats = sample_scaled[0][1:]

sample_input = np.hstack((fuzzy_temp, other_feats)).reshape(1, -1)

sample_pred_scaled = model.predict(sample_input)
sample_pred_real = y_scaler.inverse_transform(sample_pred_scaled)[0]


# -----------------------------
# Output
# -----------------------------
print("\nModel Inference: Next-Day Weather Forecast")
print(f"Temperature : {sample_pred_real[0]:.2f} °C")
print(f"Humidity    : {sample_pred_real[1]:.2f} %")
print(f"Wind Speed  : {sample_pred_real[2]:.2f} km/h")
print(f"Pressure    : {sample_pred_real[3]:.2f} hPa")
