import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Data Simulation ---
# In a real-world scenario, this data would come from sensors, weather APIs, and utility grids.
# We simulate 365 days of hourly data (365 * 24 = 8760 time steps).

def simulate_data(timesteps):
    """Generates synthetic time-series data for model training."""
    np.random.seed(42)

    # 1. Primary Features (Inputs)
    hours = np.linspace(0, timesteps - 1, timesteps)
    
    # Seasonality for Temperature (high in summer, low in winter)
    temp = 15 + 10 * np.sin(hours * 2 * np.pi / 8760) + np.random.normal(0, 1, timesteps)
    
    # Diurnal pattern for Grid Load (high during day, low at night)
    load_base = 30 + 10 * np.sin(hours * 2 * np.pi / 24)
    grid_load = load_base + np.random.normal(0, 2, timesteps)
    
    # Wind Speed (random/intermittent renewable generation factor)
    wind_speed = 5 + 3 * np.random.rand(timesteps)
    
    # 2. Target Variable (Output): Carbon Intensity (CI in gCO2/kWh)
    # CI is primarily driven by load and renewables (wind).
    # Higher load generally means more fossil fuels (higher CI).
    # Higher wind means less fossil fuels (lower CI).
    
    # Base CI is 400 gCO2/kWh, modulated by load (positive effect) and wind (negative effect)
    carbon_intensity = 400 + 0.5 * grid_load - 15 * wind_speed + np.random.normal(0, 5, timesteps)
    
    # Clamp CI to realistic range
    carbon_intensity = np.clip(carbon_intensity, 200, 550) 
    
    data = pd.DataFrame({
        'Temperature': temp,
        'Grid_Load': grid_load,
        'Wind_Speed': wind_speed,
        'Carbon_Intensity': carbon_intensity
    })
    
    return data

# --- 2. Preprocessing for LSTM ---

def create_sequences(data, lookback):
    """
    Converts time-series data into sequences for LSTM training.
    
    The LSTM needs data in the format [samples, timesteps, features].
    Each sample will contain 'lookback' hours of input data to predict 
    the Carbon Intensity one hour ahead.
    """
    X, y = [], []
    # Drop the target column from features for input X
    features = data.drop(columns=['Carbon_Intensity'])
    target = data['Carbon_Intensity'].values
    
    for i in range(len(data) - lookback - 1):
        # Current sequence: 'lookback' hours of features
        X.append(features[i:(i + lookback)].values)
        # Target: Carbon Intensity one hour after the sequence ends
        y.append(target[i + lookback])
        
    return np.array(X), np.array(y)

# --- Main Execution ---

# Configuration
TIMESTEPS = 8760  # One year of hourly data
LOOKBACK_HOURS = 24  # Use the last 24 hours of data to predict the next hour
TEST_SPLIT = 0.2    # 20% of data for testing

# 1. Generate Data
df = simulate_data(TIMESTEPS)
print(f"Simulated Data Shape: {df.shape}")

# 2. Normalize Data
# We use a separate scaler for the target variable to easily inverse-transform predictions
feature_scaler = MinMaxScaler()
df_scaled = pd.DataFrame(feature_scaler.fit_transform(df), columns=df.columns)

target_scaler = MinMaxScaler()
# Fit and transform the target column separately
df_scaled['Carbon_Intensity'] = target_scaler.fit_transform(df[['Carbon_Intensity']])

# 3. Create LSTM Sequences
X, y = create_sequences(df_scaled, LOOKBACK_HOURS)
print(f"Input Sequences (X) shape: {X.shape}")
print(f"Target Array (y) shape: {y.shape}")

# 4. Split Data (Training and Testing)
split_point = int(X.shape[0] * (1 - TEST_SPLIT))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 5. Build the LSTM Model (Neural Network)
model = Sequential([
    # Input layer expects a sequence of LOOKBACK_HOURS steps with 3 features
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2), 
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1) # Output layer for the single predicted value (Carbon Intensity)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
print("\n--- Model Summary ---")
model.summary()

# 6. Train the Model 
print("\n--- Training Model (LSTM) ---")
# EarlyStopping helps prevent overfitting and saves computation time
es_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test), 
    callbacks=[es_callback],
    verbose=1
)

# 7. Evaluate and Predict
print("\n--- Making Predictions ---")
# Predict on the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to get actual Carbon Intensity values (gCO2/kWh)
# The prediction output is 2D, but the scaler expects a specific format
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE in the original units
mae = np.mean(np.abs(y_actual - y_pred))

print(f"\nModel Mean Absolute Error (MAE) on Test Set: {mae:.2f} gCO2/kWh")

# --- Application (Optimization) ---

# Get the prediction for the very last test sample (the latest forecast)
latest_ci_forecast = y_pred[-1][0]
print(f"Latest Predicted Carbon Intensity (CI_pred) for next hour: {latest_ci_forecast:.2f} gCO2/kWh")

# Decision Logic for SDG 13:
LOW_CARBON_THRESHOLD = 350 # Example threshold

if latest_ci_forecast < LOW_CARBON_THRESHOLD:
    print("\n--- Optimization Recommendation (Climate Action) ---")
    print("Recommendation: Carbon Intensity is low. This is the optimal time for energy-intensive tasks (e.g., EV charging, industrial processes) to run and draw power from cleaner sources.")
else:
    print("\n--- Optimization Recommendation (Climate Action) ---")
    print("Recommendation: Carbon Intensity is high. Non-essential energy consumption should be shifted to a later period when CI is forecasted to drop.")
