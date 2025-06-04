from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Load model and tokenizer with correct paths
model = tf.keras.models.load_model('models/imdb_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

maxlen = 200

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    return padded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed = preprocess(review)
    prediction = model.predict(processed)[0][0]
    sentiment = 'Positive 😊' if prediction >= 0.5 else 'Negative 😞'
    return render_template('index.html', review=review, sentiment=sentiment, prob=prediction)

if __name__ == '__main__':
    app.run(debug=True)
