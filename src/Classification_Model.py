from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def classification_model():
    df = pd.read_csv("/Users/salonisahal/Stock-Price-Prediction/Data/featured_data.csv")
# Prepare data
    X = df.drop(columns=['stock_id','date', 'target', 'Target_Return', 'Target_Movement'])
    y = df['Target_Movement']

# Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

# Initialize XGBoost classifier
    model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softmax",   # directly predict class labels
    num_class=3
    )

# Train
    model.fit(X_train, y_train)

# Predict
    y_pred = model.predict(X_test)

# Evaluation
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance plot
    plt.figure(figsize=(10,6))
    plot_importance(model, max_num_features=10)
    plt.show()
    return