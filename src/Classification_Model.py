import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df= pd.read_csv("Data/all_stocks.csv")
# Encode stock_id as integers
encoder = LabelEncoder()
df['stock_id_encoded'] = encoder.fit_transform(df['stock_id'])

# Create stock embeddings (random init, can also be pretrained with NN)
embedding_dim = 8  # you can tune this
num_stocks = df['stock_id_encoded'].nunique()

# Random embeddings for each stock_id
stock_embeddings = np.random.randn(num_stocks, embedding_dim)

# Map each stock_id to its embedding vector
embedding_matrix = df['stock_id_encoded'].map(lambda x: stock_embeddings[x])

# Convert to DataFrame and expand embedding columns
embedding_df = pd.DataFrame(embedding_matrix.tolist(),
                            index=df.index,
                            columns=[f"stock_emb_{i}" for i in range(embedding_dim)])

# Merge with original features
df_emb = pd.concat([df, embedding_df], axis=1)
# Prepare data with embeddings
X = df_emb.drop(columns=['stock_id','date', 'target', 'Target_Return', 'Target_Movement'])
y = df_emb['Target_Movement']

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softmax",
    num_class=3
)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
model.save_model("model/xgb_stock.json")