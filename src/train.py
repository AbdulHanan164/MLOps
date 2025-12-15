import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
data = pd.read_csv("data/dataset.csv")

X = data[["feature1", "feature2"]]
y = data["target"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model trained and saved successfully")
