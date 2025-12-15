from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def root():
    return {"status": "Model API running"}

@app.post("/predict")
def predict(feature1: float, feature2: float):
    X = np.array([[feature1, feature2]])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
