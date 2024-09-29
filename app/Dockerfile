FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY [ "model.pkl", "preprocessing_utils.py", "diabetes_app.py", "./"]

# Expose the port on which the app will run
EXPOSE 8080

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "diabetes_app:app", "--host", "0.0.0.0", "--port", "8080"]
