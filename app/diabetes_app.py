from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel
from preprocessing_utils import preprocess_data

app = FastAPI(
    title="Diabetes Classifier",
    description="Classify whether an individual has Diabetes or not based on certain features",
    version="0.1",
)


@app.get('/')
def main():
	return {'message': 'This is a model for classifying diabetes'}


class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int


@app.on_event('startup')
def load_artifacts():
    global model
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)


@app.post('/predict')
def predict(data : DiabetesInput):
         # Preprocess the incoming data point
    print("Received data:", data)
    preprocessed_data = preprocess_data(data)

    # Make a prediction using the loaded model
    prediction = model.predict(preprocessed_data)

    # Return the prediction result
    return {'prediction': int(prediction[0])}
