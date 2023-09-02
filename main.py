import uvicorn
from fastapi import FastAPI, Query, Response
from typing import Annotated
import mlflow
from mlflow import MlflowClient
import pickle
import pandas as pd

app = FastAPI()

data = pd.read_csv('df_test_cleaned_3000.csv')

# Load Model from mlflow
tracking_URI = "http://127.0.0.1:5000"

if tracking_URI == '':
    mlflow.set_tracking_uri(tracking_URI) 

    client = MlflowClient()
    rm = client.search_registered_models()
    if len(rm) != 0:
        rm_name = rm[0].name
        rm_run_id = rm[0].latest_versions[0].run_id
    else:
        rm_run_id = ''
        print('There is no model registered.')

    model_path = "runs:/" + rm_run_id + "/" + rm_name
    model = mlflow.sklearn.load_model(model_path)
else:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
data["y_pred"] = model.predict(data)
probas = model.predict_proba(data)

@app.get('/')
async def start_page():
    return {'message': "Welcome !"}


@app.get('/group/')
async def customers_stat(feature: str):
    return Response(data[feature])


@app.get('/customer/')
async def predict_id(id: int):
    prediction = int(data["y_pred"][id])
    probability = probas[id].tolist()
    infos = data.iloc[id, :]
    dict_data = {'prediction': prediction,
                 'probability': probability,
                 'infos': infos}
    return dict_data


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)