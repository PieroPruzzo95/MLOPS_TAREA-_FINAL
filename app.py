from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return round(prediction[0], 2)
	

@app.get("/")
async def root():
    return {"message": "Prediction"}

@app.get("/predict")
async def predict(
    experience: int,
    test_score: int,
    interview_score: int
):
    features = [experience, test_score, interview_score]
    prediction = model_pred(features)  # Usa la función model_pred
    return {"prediction": prediction}  
	

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
