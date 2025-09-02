import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import joblib

# -----------------------------
# Encode Stock ID
# -----------------------------
df= pd.read_csv("Data/all_stocks.csv")
stock_encoder = LabelEncoder()
df["stock_id_enc"] = stock_encoder.fit_transform(df["stock_id"])  # integer encode

# -----------------------------
# Prepare Features & Target
# -----------------------------
features = df.drop(columns=["stock_id", "date", "target", "Target_Movement"])
target = df["target"].values.reshape(-1, 1)

# scale features
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(features)

# scale target
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(target)

# reshape numeric features for LSTM
X_num = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
y = y_scaled

# stock ids
X_stock = df["stock_id_enc"].values

# -----------------------------
# Train-Test Split
# -----------------------------
split = int(0.8 * len(X_num))
X_num_train, X_num_test = X_num[:split], X_num[split:]
X_stock_train, X_stock_test = X_stock[:split], X_stock[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Build Model with Stock Embedding
# -----------------------------
# Inputs
num_input = Input(shape=(X_num_train.shape[1], X_num_train.shape[2]))  # numeric features
stock_input = Input(shape=(1,))  # stock ID

# Stock Embedding
embedding_dim = 16  # size of embedding vector (tunable)
stock_emb = Embedding(input_dim=len(stock_encoder.classes_), output_dim=embedding_dim)(stock_input)
stock_emb = Flatten()(stock_emb)  # shape: (batch, embedding_dim)

# LSTM on numeric features
x = LSTM(64, return_sequences=True)(num_input)
x = Dropout(0.2)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dropout(0.2)(x)

# Concatenate embedding with LSTM output
combined = Concatenate()([x, stock_emb])

# Dense layers
out = Dense(64, activation="relu")(combined)
out = Dense(1)(out)  # regression output

# Model
model = Model(inputs=[num_input, stock_input], outputs=out)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
model.summary()

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    [X_num_train, X_stock_train], y_train,
    validation_data=([X_num_test, X_stock_test], y_test),
    epochs=5,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict([X_num_test, X_stock_test])

# inverse transform target and predictions
y_pred = target_scaler.inverse_transform(y_pred)
y_test = target_scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📉 Regression Results")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

joblib.dump(feature_scaler, "model/feature_scaler.pkl")
joblib.dump(target_scaler, "model/target_scaler.pkl")
joblib.dump(stock_encoder, "model/stock_encoder.pkl")
model.save('model/LSTM_model_stock.keras')