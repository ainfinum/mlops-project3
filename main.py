import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

from model.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

# uvicorn main:app --reload
# App will be available locally at http://127.0.0.1:8000.


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = pickle.load(open("./model/saved_models/saved_model.pkl", "rb"))
    encoder = pickle.load(open("./model/saved_models/saved_encoder.pkl", "rb"))
    lb = pickle.load(open("./model/saved_models/saved_lb.pkl", "rb"))


@app.get("/")
async def hello():
    return {
        "endpoints": "Available endpoint: POST: /predict - prediction, \
        GET: /docs - documentation"
    }


class Item(BaseModel):
    age: int
    workclass: str
    fnlgt: Optional[int] = 0
    education: str
    educationNum: int
    maritalStatus: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capitalGain: int
    capitalLoss: int
    hoursPerWeek: int
    nativeCountry: str

    class Config:
        schema_extra = {
            "example": {
                "age": 40,
                "workclass": "Private",
                "fnlgt": 54545,
                "education": "Doctorate",
                "educationNum": 16,
                "maritalStatus": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capitalGain": 0,
                "capitalLoss": 0,
                "hoursPerWeek": 40,
                "nativeCountry": "United-States",
            }
        }


@app.post("/predict")
async def predict(item: Item):

    X_categorical = []
    X_continuous = []

    X_categorical.append(
        np.array(
            [
                item.workclass,
                item.education,
                item.maritalStatus,
                item.occupation,
                item.relationship,
                item.race,
                item.sex,
                item.nativeCountry,
            ]
        )
    )

    X_continuous.append(
        np.array(
            [
                item.age,
                item.fnlgt,
                item.educationNum,
                item.capitalGain,
                item.capitalLoss,
                item.hoursPerWeek,
            ]
        )
    )

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    prediction = inference(model, X)
    prediction = prediction.round()

    salary = ">50K" if prediction > 0.5 else "<=50K"
    results = {"salary": salary}
    return results
