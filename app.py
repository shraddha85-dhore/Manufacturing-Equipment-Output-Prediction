from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Manufacturing Equipment Output Prediction API")

model = joblib.load("linear_regression_model.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Manufacturing Equipment Output Prediction API is Running Successfully!"}

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"Predicted Parts Per Hour": float(prediction[0])}
