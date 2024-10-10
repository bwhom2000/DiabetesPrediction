import requests

data = {
    "gender": "Female",
    "age": 30,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 22.5,
    "HbA1c_level": 5.7,
    "blood_glucose_level": 90
}


url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=data)
print(response.json())
