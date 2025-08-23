import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
def regression_model():
    df = pd.read_csv("/Users/salonisahal/Stock-Price-Prediction/Data/featured_data.csv")
# Prepare Data
# -----------------------------
# Assume df contains: [date, open, high, low, close, volume, ... , target]
    features = df.drop(columns=["stock_id","date", "target", "Target_Movement"])  # keep only numeric features
    target = df['target'].values.reshape(-1, 1)

# scale features
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(features)

# scale target
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(target)

# reshape for LSTM: (samples, timesteps, features)
# here timesteps = 1 (can be increased for sequence prediction)
    X = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y = y_scaled

# train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("X_train shape:", X_train.shape)  # (samples, timesteps, features)
    print("y_train shape:", y_train.shape)

# -----------------------------
# Build LSTM Model
# -----------------------------
    model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # regression output
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.summary()

# -----------------------------
# Train
# -----------------------------
    history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
    )

# -----------------------------
# Evaluate
# -----------------------------
    y_pred = model.predict(X_test)

# inverse transform target and predictions
    y_pred = target_scaler.inverse_transform(y_pred)
    y_test = target_scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📉 Regression Results")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
    return
