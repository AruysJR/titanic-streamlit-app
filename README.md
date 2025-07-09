# Titanic Survival Prediction App
This is a Streamlit web app that predicts whether a Titanic passenger **would have survived** or **would not have survived**, based on their details like name, age, sex, fare, and more.

## Features
- Predicts survival using a trained **Random Forest classification model**
- It shows prediction confidence percentage
- Displays a pie chart of survival probability
- Keeps a history of all predictions in the session
- Allows download of:
  - A single prediction as CSV
  - The full prediction history as CSV

## Files Included

- `app.py` – The main Streamlit app script
- `titanic_model.pkl` – The trained Titanic classification model
- `requirements.txt` – List of required Python packages

[Live Demo](https://titanic-app-1.streamlit.app/)
