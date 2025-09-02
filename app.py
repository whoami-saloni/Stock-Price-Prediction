import gradio as gr
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model and scalers
# -----------------------------
lstm_model = load_model("model/LSTM_model_stock.keras")
feature_scaler = joblib.load("model/feature_scaler.pkl")
target_scaler = joblib.load("model/target_scaler.pkl")
stock_encoder = joblib.load("model/stock_encoder.pkl")

# Load dataset
df = pd.read_csv("Data/all_stocks.csv")

# Map stock IDs to encoded values
label_map = {label: i for i, label in enumerate(stock_encoder.classes_)}
df["stock_id_enc"] = df["stock_id"].map(label_map).fillna(-1).astype(int)

# -----------------------------
# Predict next N days with rolling window
# -----------------------------
def predict_next_days(model, last_known_features, stock_df, stock_id_value, target_scaler, days=22):
    predictions = []
    current_input = last_known_features.copy()
    stock_id_input = np.array([[stock_id_value]])
    
    # Calculate historical volatility for noise
    hist_std = stock_df['target'].pct_change().std()
    hist_std = hist_std if not np.isnan(hist_std) else 0.01  # fallback

    for _ in range(days):
        # Predict next scaled price
        next_scaled = model.predict([current_input, stock_id_input], verbose=0)
        next_price = target_scaler.inverse_transform(next_scaled)[0, 0]
        predictions.append(next_price)

        # Update rolling input realistically
        new_row = current_input[:, -1, :].copy()
        for i, col_name in enumerate(feature_scaler.feature_names_in_):
            if "close" in col_name.lower():
                new_row[0, i] = next_price
            elif "open" in col_name.lower():
                new_row[0, i] = predictions[-1]  # previous close
            elif "high" in col_name.lower():
                new_row[0, i] = max(new_row[0, i], next_price)
            elif "low" in col_name.lower():
                new_row[0, i] = min(new_row[0, i], next_price)
            else:
                # Add realistic noise based on historical volatility
                new_row[0, i] *= 1 + np.random.normal(0, hist_std)
                

        # Shift window
        current_input = np.concatenate([current_input[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)

    return predictions

# -----------------------------
# Gradio prediction function
# -----------------------------
def predict_stock(stock_name):
    stock_df = df[df["stock_id"] == stock_name].copy()
    if stock_df.empty:
        return None, f"Stock '{stock_name}' not found!"

    # Prepare features
    features = stock_df.drop(columns=["stock_id", "date", "target", "Target_Movement"], errors='ignore')
    for col in feature_scaler.feature_names_in_:
        if col not in features.columns:
            features[col] = 0
    X_scaled = feature_scaler.transform(features)

    # Prepare last timesteps for LSTM
    timesteps = lstm_model.input_shape[0][1]
    if X_scaled.shape[0] < timesteps:
        pad = np.zeros((timesteps - X_scaled.shape[0], X_scaled.shape[1]))
        X_scaled = np.vstack([pad, X_scaled])
    last_features = X_scaled[-timesteps:].reshape(1, timesteps, -1)

    stock_id_value = stock_df["stock_id_enc"].iloc[-1]

    # Predict next 100 days
    preds = predict_next_days(lstm_model, last_features, stock_df, stock_id_value, target_scaler, days=22)

    # Create DataFrame
    table_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(22)],
        "Predicted Price": preds
    })

    # Plot prices
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(table_df["Day"], table_df["Predicted Price"], marker='o', color='blue', markersize=3)
    ax.set_title(f"Predicted Prices for {stock_name} (Next 100 Days)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.grid(True)
    plt.xticks(rotation=45)

    return fig, table_df

# -----------------------------
# Launch Gradio UI
# -----------------------------
stock_list = df["stock_id"].unique().tolist()

gr.Interface(
    fn=predict_stock,
    inputs=gr.Dropdown(stock_list, label="Select Stock"),
    outputs=[gr.Plot(label="Price Chart"), gr.Dataframe(label="Predicted Prices")],
    title="Stock Price Prediction (Next 100 Days)",
    description="Select a stock to see LSTM-predicted prices for the next 100 days."
).launch(debug=True)
