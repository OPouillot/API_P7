from sklearn.dummy import DummyClassifier
import pandas as pd
from model_selection import job_score, model_pipe, select_final_model


def test_should_calculate_job_score():
    data = [0,0,0,0,1,1,1,1]
    y_true = [0,0,0,0,1,1,1,1]
    dummy = DummyClassifier()
    dummy.fit(data, y_true)

    assert job_score(dummy, data, y_true) == 10


def test_select_final_model():
    scores = {'best_roc_auc': [1, 0],
              'job_score': [0, 10],
              'best_time': [0,1]}
    mlflow_ids = {'first': 0,
                  'second': 1}
    best_scores = pd.DataFrame(scores)
    # score calcul = time + 2*auc + 2*job = 1 + 2 + 2
    result = {'id': 0,
              'name': 'first',
              'score': 5}
    assert select_final_model(best_scores, mlflow_ids) == result


def test_model_pipe_with_preprocess():
    regressor = DummyClassifier()
    pipe = model_pipe(regressor, preprocess=True)
    
    # Vérification des étapes du pipeline
    assert len(pipe.steps) == 3
    assert pipe.steps[0][0] == 'scaler'
    assert pipe.steps[1][0] == 's'
    assert pipe.steps[2][0] == 'model'
    
    # Vérification du modèle out
    assert pipe.steps[2][1] == regressor


def test_model_pipe_without_preprocess():
    regressor = DummyClassifier()
    pipe = model_pipe(regressor, preprocess=False)
    
    # Vérification des étapes du pipeline
    assert len(pipe.steps) == 1
    assert pipe.steps[0][0] == 'model'
    
    # Vérification du modèle out
    assert pipe.steps[0][1] == regressor
