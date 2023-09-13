import os
import shutil
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (train_test_split,
                                     RandomizedSearchCV,
                                     RepeatedStratifiedKFold)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score)
import mlflow
from mlflow import MlflowClient
import warnings
warnings.filterwarnings("ignore")

def job_score(model, X_test, y_true):
    ''' Function establishing a job score assuming that the cost of an FN is ten times higher than the cost of an FP '''
    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print(confusion_matrix(y_true, y_pred))
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    job_score = (10 * fnr) + fpr
    
    return round(job_score, 3)

def model_pipe(regressor, preprocess=False):
    """ Fonction de définition des modèles, afin d'utiliser le même pipeline de preprocessing pour chaque modèle """
    
    if preprocess is False:
        pipe = Pipeline([
            ('model', regressor)
        ])
    else:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('s', SMOTE(random_state=1)),
            ('model', regressor)
        ])
    return pipe

def log_run(gridsearch: RandomizedSearchCV, model_name: str, tracking_URI: str, metrics={}, tags={}):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        metrics (dict): Dictionary of metrics
        tags (dict): Dictionary of extra data and tags (usually features)
        
    Output:
        run_id (str): Return the ID for saved run
    """
    mlflow.set_tracking_uri(tracking_URI)
    cv_results = gridsearch.cv_results_
    
    with mlflow.start_run() as run:  

        mlflow.log_param("folds", gridsearch.cv)

        print("Logging parameters")
        params = list(gridsearch.param_distributions.keys())
        for param in params:
            mlflow.log_param(param, cv_results["param_%s" % param])

        print("Logging metrics")
        mlflow.log_metrics(metrics)

        print("Logging model")        
        mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name)

        print("Logging extra data related to the experiment")
        mlflow.set_tags(tags) 

        run_id = run.info.run_uuid
        mlflow.end_run()
    print("runID: %s" % run_id)
    
    return run_id

def select_final_model(best_scores, mlflow_ids):
    """Selecting the final model to save from total_score calculated from the scores of each model
    total_score = time + 2*auc + 2*job
    
    Args:
        best_scores (DataFrame): Scores (roc_auc, job and time) for each models
        mlflow_ids (dict): MLFlow runs ids for each models
        
    Output:
        final_model (dict): Return a dict containing the final model information (id, name and score)
    """
    
    total_score = []
    max_time = max(best_scores['best_time'])
    min_time = min(best_scores['best_time'])
    
    for index, model in best_scores.iterrows():
        
        normalized_time_score = (model['best_time'] - max_time) / (min_time - max_time)
        normalized_job_score = 1 - ( model['job_score'] / 10 )
        
        rounded_time_score = round(normalized_time_score, 3)
        rounded_job_score = round(normalized_job_score, 3)
        rounded_roc_score = round(model['best_roc_auc'], 3)
        
        score = (rounded_time_score +
                 2 * rounded_roc_score +
                 2 * rounded_job_score)
        
        total_score.append(score)
    print(total_score)
    final_index = total_score.index(max(total_score))
    
    final_model = {'id': list(mlflow_ids.values())[final_index],
                   'name': list(mlflow_ids)[final_index],
                   'score': total_score[final_index]}
    
    return final_model

if __name__ == '__main__':

    tracking_URI = "http://127.0.0.1:5000"
    
    # import data file
    df_train = pd.read_csv('../df_train_cleaned.csv')
    
    #splitting data to train and test sets
    X = df_train.drop('TARGET', axis=1).values
    y = df_train['TARGET'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.25, random_state=27)
    
    # Define class weights
    class_weights = {0: 1, 1: 9}
    
    # Définition de la grille des hyperparamètres pour chaque modèle
    param_grid_lr = {'model__penalty': ['none', 'l2'],
                 'model__C': [0.001, 0.01, 0.1],
                 'model__max_iter': [500, 1000]}
    param_grid_xgb = {'model__n_estimators': [100, 200],
                  'model__max_depth': [3, 5],
                  'model__min_child_weight': [1, 3],
                  'model__gamma': [0, 0.2],
                  'model__scale_pos_weight' : [class_weights[1]/class_weights[0]]}
    param_grid_lgb = {'model__num_leaves': [31, 50, 100],
                  'model__max_depth': [3, 4, 5],
                  'model__min_child_samples': [20, 50, 100],
                  'model__class_weight' : [class_weights]}
    
    # Création d'une liste de modèles pour pouvoir les itérer dans une boucle for
    models = [
        ('Logistic Regression', model_pipe(LogisticRegression(), preprocess=True), param_grid_lr),
        ('XGBoost', model_pipe(XGBClassifier(), preprocess=False), param_grid_xgb),
        ('LightBoost', model_pipe(LGBMClassifier(), preprocess=False), param_grid_lgb)
         ]
    
    # Définition de la cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    best_models = {}
    best_scores = {}
    mlflow_ids = {}
    
    # Iteration sur la liste des modèles pour effectuer la recherche des meilleurs hyperparamètres pour chacun
    for name, model, param in models:
        print('GridSearchCV pour :', name)
    
        random_search = RandomizedSearchCV(estimator=model,
                                           param_distributions=param,
                                           n_iter=10,
                                           scoring='roc_auc',
                                           cv=cv,
                                           verbose=1,
                                           random_state=1)
        random_search.fit(X_train, y_train)
    
        best_models[name] = random_search.best_estimator_
    
        # Make predictions on the test data
        y_pred = best_models[name].predict_proba(X_test)[:,1]
    
        # METRICS
        # ROC AUC score
        roc_auc_test = roc_auc_score(y_test, y_pred)
        # JOB score
        job = job_score(best_models[name], X_test, y_test)
    
        metrics = {'best_roc_auc': random_search.best_score_,
                   'roc_auc_test': roc_auc_test,
                   'job_score': job,
                   'best_time': random_search.cv_results_['mean_fit_time'][random_search.best_index_]}
        
        best_scores[name] = metrics
    
        # MLFlow Logs
        mlflow_ids[name] = log_run(random_search,
                                   name,
                                   tracking_URI,
                                   metrics=metrics,
                                   tags={'model_name': name})
    
        # Print the best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)
        print("job score: ", job)
    
    # Select final model
    final_model = select_final_model(pd.DataFrame(best_scores).T, mlflow_ids)
    
    # Save final model
    client = MlflowClient()
    rm = client.search_registered_models()
    
    if len(rm) != 0:
        rm_run_id = rm[0].latest_versions[0].run_id
        rm_score = rm[0].latest_versions[0].tags['final_score']
    else:
        rm_run_id = ''
        rm_score = 0
    
    if(final_model['id'] != rm_run_id and final_model['score'] > float(rm_score)):
        print("Saving the best model...")
        model_path = "runs:/" + final_model['id'] + "/" + final_model['name']
    
        # Register the best sklearn run as a model
        result = mlflow.register_model(model_path, final_model['name'], tags={'final_score': final_model['score']})
    
        # save the model in pickle format
        # set path to location for persistence
        sk_model = best_models[final_model['name']]
        sk_path_dir = os.getcwd() + "/Model"
        
        # Si le dossier existe, le supprimer
        if os.path.exists(sk_path_dir) and os.path.isdir(sk_path_dir):
            # Supprimer le dossier et son contenu
            shutil.rmtree(sk_path_dir)
            print("Remplassement du dossier de sauvegarde du model.")
    
        mlflow.sklearn.save_model(sk_model,
                                  sk_path_dir,
                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
        
        shutil.copyfile(sk_path_dir+"/model.pkl", os.getcwd()+"/model.pkl")
        print("Model saved and ready to use.")
        
    else:
        print("No better model was built.")