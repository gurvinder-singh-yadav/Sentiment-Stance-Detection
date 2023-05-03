import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

stance_model = tf.keras.models.load_model('data/Bi_LSTM_stance.h5')
sentiment_model = tf.keras.models.load_model('data/Bi_LSTM_senti.h5')

def preprocess_text(text):

    max_length = 120
    trunc_type='post'
    padding_type='post'

    # loading
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    text_sequences = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return text_padded

def predict_stance(tweet):
    # Preprocess the tweet text
    preprocessed_tweet = preprocess_text(tweet)
    
    # Make a prediction with the loaded model
    prediction = stance_model.predict(preprocessed_tweet)[0][0]
    
    # Determine the Stance label based on the prediction
    if prediction >= 0.5:
        return ('Believer', prediction)
    else:
        return ('Denier', 1-prediction)
    
def predict_senti(tweet):
    # Preprocess the tweet text
    preprocessed_tweet = preprocess_text(tweet)

    # Make a prediction with the loaded model
    prediction = sentiment_model.predict(preprocessed_tweet)[0]
    
    # Determine the sentiment label based on the prediction
    label = np.argmax(prediction)
    if label == 0:
        return ('Neutral',prediction)
    elif label == 1:
        return ('Positive',prediction)
    else:
        return ('Negative',prediction)


def run(tweet):
    sentiment, senti_score = predict_senti(tweet)
    stance, stance_score = predict_stance(tweet)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment", sentiment, np.max(senti_score).tolist())
    with col2:
        st.metric("Stance", stance, float(stance_score))

st.set_page_config(
    page_title="Sentiment & Stance Detection",
    layout="centered"
)

st.title("Sentiment & Stance Detection")

col1, col2 = st.columns(2)

tweet = st.selectbox(
    "Predict Sentiment & Stance",
    ["Climate Change is a hoax",
     "Scientists suggest that earth is warming up every consecutive year",
     "thermometer in world's coldest village break temperature plunge minus 62 celsius climate change is real",
     "We need renewable energy to save earth"
     ]
)


if st.button("Predict"):
    with st.spinner("Analysing Text"):
        run(tweet)
        st.success("Analysed")





