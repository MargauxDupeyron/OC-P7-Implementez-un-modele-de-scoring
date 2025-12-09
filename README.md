# 🏦 Projet 7 — Implémentez un modèle de scoring

Ce dépôt contient l’ensemble du travail réalisé pour le **Projet 7 du parcours Data Scientist – OpenClassrooms**.

🎯 Objectif : **mettre en production un modèle de scoring** capable de prédire la probabilité de défaut de clients, via :

- Feature engineering & modèle ML (LightGBM)  
- Pipeline complet de transformation & prédiction  
- API FastAPI dockerisée et déployée sur Render  
- Interface utilisateur Streamlit connectée à l’API  
- Monitoring du Data Drift (Evidently)  
- Tests unitaires (pytest)  
- Déploiement via Docker  

---

## 📌 Sommaire
- [1. Description du projet](#1-description-du-projet)  
- [2. Structure du projet](#2-structure-du-projet)  
- [3. API FastAPI (local & Render)](#3-api-fastapi-local--render)  
- [4. Application Streamlit](#4-application-streamlit)  
- [5. Monitoring du Data Drift](#5-monitoring-du-data-drift)  
- [6. Docker](#6-docker)  
- [7. Tests unitaires](#7-tests-unitaires)  
- [8. Installation & exécution](#8-installation--exécution)  
- [9. Architecture du projet](#9-architecture-du-projet)  
- [10. Résumé](#10-résumé)

---

## 1. Description du projet

Le projet consiste à industrialiser un **moteur de scoring** basé sur le dataset Home Credit, afin d’évaluer le risque de défaut d’un client.

### Modélisation
- Nettoyage et préparation des données  
- Feature engineering avancé  
- Création d’un **FeatureBuilder** (pipeline preprocessing custom)  
- Entraînement et optimisation d’un modèle **LightGBMClassifier**  
- Optimisation du seuil métier  
- Export des artefacts :  
  - `model.pkl` (pipeline complet)  
  - `feature_names.json`  
  - `threshold.json`

### Mise en production
- Développement d’une **API REST** avec FastAPI  
- Déploiement sur Render via Docker  
- Endpoints :  
  - `/` → Healthcheck  
  - `/predict_proba` → scoring  
  - `/shap_explanation` → SHAP local  
  - `/shap_global` → SHAP global  

### Interface Streamlit
L’application permet :
- Sélection du client (index)  
- Envoi des features à l’API Render  
- Affichage :
  - probabilité de défaut  
  - décision (accepté/refusé)  
  - seuil métier  
- Explicabilité :  
  - **SHAP local : waterfall plot**  
  - **SHAP global : summary plot + top features**  
- Comparaison du client avec la population (histogrammes)

### Monitoring
- Analyse du drift entre train/test via **Evidently**  
- Rapport complet exporté dans `reports/`

### Tests unitaires
- Tests API  
- Tests du modèle & des transformations  
- Tests structurels des endpoints FastAPI  

---

## 2. Structure du projet

```
Projet_7/
│
├── data/                     # Données locales 
│   └── processed/
│       └── df_test_sample.csv
│
├── models/                   # Artefacts ML
│   ├── model.pkl
│   ├── feature_names.json
│   └── threshold.json
│
├── src/
│   ├── api/
│   │   ├── app.py            # API FastAPI
│   │   └── schemas.py 
│   └── utils/
│       └── preprocessing.py
│
├── streamlit_app/
│   ├── test_api.ipynb
│   └── streamlit_app.py      # Interface Streamlit
│
├── tests/                    # Tests unitaires pytest
│
├── reports/
│   └── data_drift_report.html
│
├── data_drift_evidently.ipynb 
├── projet7_modelisation_mlflow.ipynb   
├── Dockerfile                # Image Docker (Render)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 3. API FastAPI (local & Render)

### ▶Lancer l’API en local

```bash
uvicorn src.api.app:app --reload
```

### 📄 Documentation interactive  
👉 http://localhost:8000/docs

---

### Version déployée sur Render

API disponible ici :  
👉 **https://oc-p7-implementez-un-modele-de-scoring.onrender.com**

Endpoints :

| Endpoint             | Description |
|---------------------|-------------|
| `/`                 | Healthcheck |
| `/predict_proba`    | Prédiction du score client |
| `/shap_explanation` | SHAP local |
| `/shap_global`      | SHAP global |

---

## 4. Application Streamlit

### ▶Lancement en local

```bash
streamlit run streamlit_app/streamlit_app.py
```

### Utilisation avec l’API Render

Dans `streamlit_app.py` :

```python
API_URL = "https://oc-p7-implementez-un-modele-de-scoring.onrender.com"
```

Fonctionnalités :
- Prédiction en direct  
- Décision métier  
- Visualisation SHAP locale + globale  
- Profil client vs population  

---

## 5. Monitoring du Data Drift

Exemple Evidently :

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=test_df)
report.save_html("reports/data_drift_report.html")
```

---

## 6. Docker

### Construire l’image

```bash
docker build -t projet7-api .
```

### Lancer le conteneur

```bash
docker run -p 8000:8000 -e PORT=8000 projet7-api
```

---

## 7. Tests unitaires

### Lancer les tests

```bash
pytest -q
```

---

## 8. Installation & exécution

```bash
git clone https://github.com/MargauxDupeyron/OC-P7-Implementez-un-modele-de-scoring.git
cd OC-P7-Implementez-un-modele-de-scoring

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 9. Architecture du projet

```
                           ┌────────────────────────────┐
                           │        Utilisateur         │
                           │ (via navigateur Streamlit) │
                           └──────────────┬─────────────┘
                                          │
                                          ▼
                           ┌────────────────────────────┐
                           │     Application Streamlit  │
                           └──────────────┬─────────────┘
                                          │ requêtes HTTP
                                          ▼
                    ┌───────────────────────────────────────────────┐
                    │        API FastAPI (Render, Docker)           │
                    └──────────────┬────────────────────────────────┘
                                   │ charge modèle + pipeline
                                   ▼
                         ┌──────────────────────────────┐
                         │   Modèle ML (LightGBM)        │
                         │   + FeatureBuilder            │
                         │   + feature_names.json        │
                         │   + threshold.json            │
                         └──────────────────────────────┘
```

---

## 10. Résumé

- ✔ API cloud + Docker  
- ✔ Streamlit complet  
- ✔ SHAP local/global  
- ✔ Monitoring drift  
- ✔ Tests unitaires  
- ✔ Pipeline ML industrialisé  

🚀 **Projet totalement fonctionnel et déployé**
