import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_cleaning import clean_tweet


app = FastAPI()
model = tf.keras.models.load_model("models/deep_learning.h5")
tokenizer = pickle.load(open('models/tokenizer.sav', 'rb'))
encoder = pickle.load(open('models/encoder.sav', 'rb'))

@app.get("/predict", summary="Predict arabic dialect of a text")
def predict(text: str):
    try:
        clean_text = clean_tweet(text)
        text_tokenized = tokenizer.texts_to_sequences(clean_text)
        input = pad_sequences(text_tokenized, maxlen=20, padding='post')
        prediction = model.predict([input])
        prediction = np.argmax(prediction,axis=1)
        prediction = encoder.inverse_transform(prediction)
        return {'text': text, 'clean text': clean_text, 'dialect': prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/")
def home():
    return {"msg": "Welcome to Arabic Dialect Iidentification Application"}