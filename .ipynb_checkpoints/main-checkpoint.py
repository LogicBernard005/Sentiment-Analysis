# import streamlit as st
# import requests

# # flask --app api.py run --port=5000
# prediction_endpoint = "http://127.0.0.1:5000/predict"

# st.title("Text Sentiment Predictor")

# # Text input for sentiment prediction
# user_input = st.text_input("Enter text and click on Predict", "")

# # Prediction on single sentence
# if st.button("Predict"):
#     response = requests.post(prediction_endpoint, data={"text": user_input})
#     response = response.json()
#     st.write(f"Predicted sentiment: {response['prediction']}")



import streamlit as st
import requests

# API endpoint
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Prediction on single sentence
if st.button("Predict"):
    if user_input:
        try:
            response = requests.post(prediction_endpoint, json={"text": user_input})
            response = response.json()
            if "prediction" in response:
                st.write(f"Predicted sentiment: {response['prediction']}")
            else:
                st.write(f"Error: {response.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.write(f"An error occurred: {str(e)}")
    else:
        st.write("Please enter some text to get a prediction.")
