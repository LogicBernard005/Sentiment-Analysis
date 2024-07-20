from flask_cors import CORS
from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():

## Chk this out
    data = request.get_json()
    text = data.get('text')
    
    if text:
        # Placeholder for prediction logic
        prediction = "positive" if "good" in text else "negative"
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'No text provided'}), 400
## Have u chked

    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_dt.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countvectorizer.pkl", "rb"))
    try:
        if "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
