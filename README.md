
# 🏦 Projet 7 — Implémentez un modèle de scoring

Ce dépôt contient l’ensemble du travail réalisé pour le Projet 7 d’OpenClassrooms :  
**développer un moteur de scoring, le packager, l’exposer via une API, ajouter un monitoring, et déployer l’ensemble dans le cloud.**

---

## 📌 Sommaire

- [1. Description du projet](#1-description-du-projet)
- [2. Structure du projet](#2-structure-du-projet)
- [3. Déploiement Docker](#3-déploiement-docker)
- [4. API FastAPI (local + Render)](#4-api-fastapi-local--render)
- [5. Application Streamlit](#5-application-streamlit)
- [6. Monitoring (Data Drift)](#6-monitoring-data-drift)
- [7. Tests unitaires](#7-tests-unitaires)
- [8. Installation & utilisation](#8-installation--utilisation)

---

## 1. Description du projet

L’objectif est de développer un modèle capable de prédire la **probabilité de défaut d’un client** (Home Credit Default Risk).

Le projet inclut :

- Préparation des données & Feature Engineering  
- Entraînement et optimisation d’un modèle LightGBM  
- Sauvegarde du pipeline + seuil métier  
- Exposition via API (FastAPI)  
- Tests unitaires (pytest)  
- Monitoring du Data Drift (Evidently.ai)  
- Déploiement cloud (Render)  
- Interface utilisateur via Streamlit  

---

## 2. Structure du projet

```
Projet_7/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── api/
│       └── app.py
│
├── models/
│   ├── final_lightgbm_model.joblib
│   ├── background.csv
│   └── features.json
│
├── streamlit_app/
│   └── streamlit_app.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── projet7_modelisation_mlflow.ipynb
├── data_drift_evidently.ipynb
├── requirements.txt
├── README.md
```

---

## 3. Déploiement Docker

### Construire l’image

```bash
docker build -t projet7-api .
```

### Lancer le conteneur

```bash
docker run --rm -p 8000:8000 projet7-api
```

### Accès local

- http://localhost:8000  
- http://localhost:8000/docs (Swagger)

---

## 4. API FastAPI (local & Render)

### Lancer localement

```bash
uvicorn src.api.app:app --reload
```

### Accès Swagger

- http://127.0.0.1:8000/docs

### Variables d’environnement utiles

```
MODEL_PATH=./models/final_lightgbm_model.joblib
API_URL=https://oc-p7-implementez-un-modele-de-scoring.onrender.com/
```

### Déploiement Render

- Service type : **Web Service**
- Runtime : **Docker**
- Commande : aucune (Render lit le Dockerfile)
- Variables requises :
  - `MODEL_PATH=./models/final_lightgbm_model.joblib`

---

## 5. Application Streamlit

L’app permet :

- de tester l’API déployée  
- d’envoyer un client  
- d’obtenir une prédiction en direct  

Exécution :

```bash
streamlit run streamlit_app/streamlit_app.py
```

Pour se connecter à l’API cloud, changer dans le script :

```
API_URL=https://oc-p7-implementez-un-modele-de-scoring.onrender.com/"
```

---

## 6. Monitoring (Data Drift)

Monitoring réalisé avec **Evidently** :

- Comparaison train/test  
- Détection des colonnes ayant subi un drift  
- Génération d’un rapport HTML  

Exemple de génération :

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=test_df)
report.save_html("drift_report.html")
```

---

## 7. Tests unitaires

Localisation : `tests/test_api.py`

Exécution :

```bash
pytest -q
```

Les tests couvrent :

- santé de l’API  
- prédiction simple  
- comportement en cas d’erreur  

---

## 8. Installation & utilisation

### Cloner le repo

```bash
git clone https://github.com/MargauxDupeyron/OC-P7-Implémentez-un-modèle-de-scoring
cd Projet_7
```

### Créer un environnement & installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.\.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---