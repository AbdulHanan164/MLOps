from fastapi import FastAPI
import joblib
import numpy as np
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

@app.get("/")
def root():
    return {"status": "Model API running"}

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}

@app.post("/predict")
def predict(feature1: float, feature2: float):
    try:
        logger.info(f"Prediction request | feature1={feature1}, feature2={feature2}")

        X = np.array([[feature1, feature2]])
        prediction = model.predict(X)[0]

        logger.info(f"Prediction result | prediction={int(prediction)}")

        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed | error={str(e)}")
        return {"error": "Prediction failed"}
