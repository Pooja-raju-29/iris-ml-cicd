####FastAPI → framework use to create the API.
####Uvicorn → server that runs FastAPI application.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()

model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {"message": "Iris ML API is running"}


class FlowerData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: FlowerData):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    flower_names = ["setosa", "versicolor", "virginica"]

    predicted_flower = flower_names[prediction[0]]

    return {
        "prediction": predicted_flower
    }



#### to check the port : python -m uvicorn main:app --reload

#### to check if its up and running use python -m uvicorn main:app --reload