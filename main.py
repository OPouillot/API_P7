import uvicorn
from fastapi import FastAPI, Query, Response
from typing import Annotated
import mlflow
from mlflow import MlflowClient
import pickle
import pandas as pd

app = FastAPI()

data = pd.read_csv('df_test_cleaned.csv')

## Load Model from mlflow
#tracking_URI = "http://127.0.0.1:5000"
#
#if tracking_URI == '':
#    mlflow.set_tracking_uri(tracking_URI) 
#
#    client = MlflowClient()
#    rm = client.search_registered_models()
#    if len(rm) != 0:
#        rm_name = rm[0].name
#        rm_run_id = rm[0].latest_versions[0].run_id
#    else:
#        rm_run_id = ''
#        print('There is no model registered.')
#
#    model_path = "runs:/" + rm_run_id + "/" + rm_name
#    model = mlflow.sklearn.load_model(model_path)

model = pickle.load(open('model.pkl', 'rb'))

#predictions = model.predict(data)
#probas = model.predict_proba(data)
#data["y_pred"] = predictions


@app.get('/')
async def start_page():
    return {'message': "Welcome !"}


@app.get('/group/')
async def customers_stat(arr_features: Annotated[list[str] | None, Query()] = None):
    sample_false = data[arr_features].loc[data["y_pred"] == 0].sample(1500)
    sample_customers = pd.concat([sample_false, data[arr_features].loc[data["y_pred"] == 1].sample(1500)])
    return Response(sample_customers.to_json(orient="records"), media_type="application/json")


@app.get('/customer/')
async def predict_id(id: int):
    prediction = int(predictions[id])
    probability = probas[id].tolist()
    infos = data.iloc[id, :]
    dict_data = {'prediction': prediction,
                 'probability': probability,
                 'infos': infos}
    return dict_data


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)