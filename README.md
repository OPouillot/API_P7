# Projet7

Ce dépôt GitHub fait parti du Projet 7 de la formation Data Scientist OpenClassRooms.
Le projet est composé de plusieurs partie :
* API
* [Dashboard](https://github.com/OPouillot/Dashboard_P7)

## Description du projet

La société financière "Prêt à dépenser", propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un outil de scoring crédit qui calcule la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

Les données originales sont téléchargeables sur Kaggle à [cette adresse](https://www.kaggle.com/competitions/home-credit-default-risk/data)

## API - Arborescence

Ce dépôt compose la partie API. Elle comprends les élements suivants :
* EDA_notebook.ipynb : Notebook d'analyse exploratoire et de préparation des données.
* First_modeling.ipynb : Premiers essais de modèles pour sélectionner le modèle de prédiction.
* model_selection.py : Script de sélection et sauvegarde automatique du meilleur modèle de prédiction.
* main.py : API récupérant le model __model.pkl__, et renvoyant des informations sur le client dont la prédiction et des informations qui seront traitées par le [Dashboard](https://github.com/OPouillot/Dashboard_P7)

## API - Fonctionnement

Pour faire fonctionner l'API, il faut suivre les étapes dans l'ordre :
* Installer l'environnement de développement et installer les librairies via la liste __requirements.txt__
* Télécharger les données kaggle
* Lancer le notebook __EDA_notebook__
* Lancer le script __model_selection.py__
* Suite à un push GitHub, les GitHub actions vont déployer l'API et le modèle sur Azure Web Service à [cette adresse](https://apip7oc.azurewebsites.net/)

__Attention : La version gratuite d'azure ne fournit qu'1Go de RAM, ce qui demande d'avoir un dataset de 3000 lignes MAX !__