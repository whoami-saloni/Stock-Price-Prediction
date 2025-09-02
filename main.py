from src.data_combine import combine_stock_data
from src.preprocess import preprocessing
from src.Feature_Engineering import feature_engineering
from src.Classification_Model import classification_model
from src.Regression_Model import regression_model
from src.Back_Testing import backtest_strategy  

if __name__ == "__main__":
    print("🔄 Combining stock data...")
    combine_stock_data()

    print("\n🔄 Preprocessing data...")
    preprocessing()

    print("\n🔄 Performing feature engineering...")
    feature_engineering()

    print("\n🔄 Training classification model...")
    classification_model()

    print("\n🔄 Training regression model...")
    regression_model()

    print("\n🔄 Running backtesting...")
    backtest_strategy()

    