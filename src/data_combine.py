import os
import glob
import pandas as pd

def combine_stock_data():
# 🔹 Path to folder containing all stock CSVs
    folder_path = "/content/drive/MyDrive/NSE"   # change to your folder

# 🔹 Get list of all CSV files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 🔹 Empty list to hold DataFrames
    df_list = []

    for file in csv_files:
        s=file[52:]
        stock_id = s.replace("-EQ_ONE_DAY.csv", "")
    
    # Read CSV
        df = pd.read_csv(file)
    
    # Add stock_id column
        df["stock_id"] = stock_id
        print(stock_id)
    # Append to list
        df_list.append(df)

# 🔹 Combine all into one DataFrame
    all_stocks_df = pd.concat(df_list, ignore_index=True)

# 🔹 Sort by stock_id + date (important for time series)
    all_stocks_df["date"] = pd.to_datetime(all_stocks_df["date"])
    all_stocks_df = all_stocks_df.sort_values(by=["stock_id", "date"])

    print("Shape of combined dataset:", all_stocks_df.shape)
    print(all_stocks_df.head())
    all_stocks_df.to_csv("/Users/salonisahal/Stock-Price-Prediction/Data/all_stocks_data.csv", index=False)
    return all_stocks_df
