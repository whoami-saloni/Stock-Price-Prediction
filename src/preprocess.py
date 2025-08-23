import pandas as pd
def preprocessing():
    df="/Users/salonisahal/Stock-Price-Prediction/Data/all_stocks_data.csv"
    indicator_cols = [
    'SMMA_5', 'SMMA_20', 'SMMA_40', 'SMMA_150',
        'SMMA_diff_5_20', 'SMMA_diff_20_40', 'SMMA_diff_40_150'
    ]

# First forward fill (propagate last valid value downwards)
    df[indicator_cols] = df[indicator_cols].ffill()

# Then backward fill (fill remaining NaNs at the top with next valid value)
    df[indicator_cols] = df[indicator_cols].bfill()

# Check if any nulls remain
    print(df[indicator_cols].isnull().sum())
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")  
    df = df.drop_duplicates()
    # Create Target column (next day's close)
    df['target'] = df['close'].shift(-1)

# Calculate Target Return
    df['Target_Return'] = (df['target'] - df['close']) / df['close']

# Define Target Movement (1 = up, -1 = down, 0 = no change)
    df['Target_Movement'] = df['Target_Return'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["Target_Movement"] = df["Target_Movement"].map({-1: 0, 0: 1, 1: 2})
    df = df.drop(df.index[-1])
    df.isnull().sum()
# Last row will have NaN since there's no "next day"
    print(df.head())
    df.to_csv("/Users/salonisahal/Stock-Price-Prediction/Data/preprocessed_data.csv", index=False)
    
    return