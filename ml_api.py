import pickle
from fastapi import FastAPI, HTTPException
from data_cleaning import clean_tweet


app = FastAPI()
model = pickle.load(open('models/model.sav', 'rb'))

@app.get("/predict", summary="Predict arabic dialect of a text")
def predict(text: str):
    try:
        clean_text = clean_tweet(text)
        prediction = model.predict([clean_text])
        return {'text': text, 'clean text': clean_text, 'dialect': prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/")
def home():
    return {"msg": "Welcome to Arabic Dialect Iidentification Application"}