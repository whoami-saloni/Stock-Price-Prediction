import pandas as pd
import numpy as np

# ---------- Example Backtesting Function ----------
def backtest_strategy(df, initial_capital=100000):
    

    df = df.copy()
    df["signal"] = 0  # default hold
    
    # Generate trading signals
    df.loc[(df["target"] > df["close"]) & (df["Target_Movement"] == 1), "signal"] = 1   # Buy
    df.loc[(df["target"] < df["close"]) & (df["Target_Movement"] == 0), "signal"] = -1  # Sell

    # Shift signals (buy today, action tomorrow)
    df["position"] = df["signal"].shift().fillna(0)

    # Daily returns
    df["return"] = df["close"].pct_change().fillna(0)

    # Strategy returns
    df["strategy_return"] = df["position"] * df["return"]

    # Portfolio growth
    df["portfolio_value"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    # Metrics
    sharpe_ratio = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(252)
    cum_return = df["portfolio_value"].iloc[-1] / initial_capital - 1
    max_drawdown = ((df["portfolio_value"].cummax() - df["portfolio_value"]) / df["portfolio_value"].cummax()).max()
    win_rate = (df["strategy_return"] > 0).mean()

    report = {
        "Cumulative Return": round(cum_return * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Win Rate": round(win_rate * 100, 2)
    }

    


# ---------- Example Usage ----------
# Assume df_stock is your DataFrame with predictions
    df_stock = pd.read_csv("/content/drive/MyDrive/NSE/angelone_historical3_NSE_EGOLD-EQ_ONE_DAY.csv")

# Run backtest
    result_df, metrics = backtest_strategy(df)

    print(metrics)
    result_df[["date","stock_id","close","signal","portfolio_value"]].head()
    result_df.to_csv("/Users/salonisahal/Stock-Price-Prediction/Data/backtest_results.csv", index=False)
    return