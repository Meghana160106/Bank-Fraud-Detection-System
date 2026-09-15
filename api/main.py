from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Bank Fraud Detection API")

model = joblib.load("models/xgboost_optuna.pkl")

@app.get("/")
def home():
    return {"status": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: list[float]):
    data = np.array(transaction).reshape(1, -1)

    probability = float(model.predict_proba(data)[0][1])
    prediction = int(probability >= 0.5)

    risk = (
        "HIGH" if probability >= 0.7
        else "MEDIUM" if probability >= 0.3
        else "LOW"
    )

    return {
        "fraud_probability": round(probability, 4),
        "prediction": "Fraud" if prediction else "Normal",
        "risk_level": risk
    }