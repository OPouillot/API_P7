import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd

tags_metadata = [
    {
        "name": "group",
        "description": "Return all client prediction and a specific feature",
    },
    {
        "name": "feat_imp",
        "description": "Return features importance",
    },
    {
        "name": "customer",
        "description": "Return prediction, probabilities and info about a specific client",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

data = pd.read_csv('df_test_cleaned_3000.csv')

with open('model.pkl', 'rb') as model_file:
    model_pipe = pickle.load(model_file)

@app.get('/')
async def start_page():
    return {'msg': "Welcome to the Projet 7 API !"}


@app.get('/group/', tags=["group"])
async def get_group(feature: str):
    y_pred = pd.Series(data=model_pipe.predict(data))

    return {'feature':data[feature].tolist(),
            'y_pred': y_pred.tolist()}

@app.get('/feat_imp/', tags=["feat_imp"])
async def get_shap():
    # Extraire le modèle du pipeline
    model = None
    for step_name, step_model in model_pipe.named_steps.items():
        if step_name == 'model':
            model = step_model
            break
    # Vérifier si le modèle a été trouvé
    if model is None:
        raise ValueError("Aucun modèle trouvé dans le pipeline")
    
    feat_imp = model.feature_importances_.tolist()
    return {'features_importance': feat_imp}


@app.get('/customer/', tags=["customer"])
async def get_predict(id: int):
    id_features = data.iloc[id, :].values.reshape(1, -1)
    probability = model_pipe.predict_proba(id_features)[0].tolist()
    prediction = int(model_pipe.predict(id_features))
    infos = data.iloc[id, :]
    dict_data = {'prediction': prediction,
                 'probability': probability,
                 'infos': infos}
    return dict_data


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)