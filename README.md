# 🏦 Projet 7 — Implémentez un modèle de scoring

Ce dépôt contient l’ensemble du travail réalisé pour le **Projet 7 du parcours Data Scientist (OpenClassrooms)**.  
L’objectif : **mettre en production un modèle de scoring pour prédire la probabilité de défaut d’un client**, via :

- 🔧 Feature engineering & modèle ML (LightGBM)  
- ⚙️ Packaging du modèle et pipeline  
- 🚀 API FastAPI déployée sur Render  
- 🖥 Interface utilisateur Streamlit  
- 🔍 Monitoring du data drift (Evidently)  
- 🧪 Tests unitaires (pytest)  
- 🐳 Déploiement via Docker  

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

---

## 1. Description du projet

Le projet consiste à construire et industrialiser un **moteur de scoring** pour évaluer le risque de défaut de paiement des clients d’une institution financière (dataset Home Credit).

Le travail réalisé inclut :

### 🔨 Modélisation
- Feature engineering avancé  
- Pipeline de transformation (FeatureBuilder)  
- Entraînement d’un modèle LightGBM  
- Optimisation du seuil métier  
- Export des artefacts (modèle, features, pipeline)

### 🌐 Mise en production
- Développement d’une **API REST** (FastAPI)
- Déploiement sur **Render** via Docker
- Interface Streamlit permettant :
  - Sélection d’un client
  - Prédiction en direct
  - Explications SHAP locales & globales
  - Visualisation profil client vs population

### 📊 Monitoring
- Analyse automatique du Data Drift avec Evidently.ai

### 🧪 Tests unitaires
- Tests API  
- Tests du FeatureBuilder  
- Tests de robustesse du modèle  

---

## 2. Structure du projet

```
Projet_7/
│
├── data/                     # Données locales (ignorées par Git)
│
├── models/                   # Artefacts ML
│   ├── final_lightgbm_model.joblib
│   ├── featurebuilder.pkl
│   └── feature_names.json
│
├── src/
│   └── api/
│       └── app.py            # API FastAPI
│
├── streamlit_app/
│   └── streamlit_app.py      # Interface utilisateur
│
├── tests/
│   ├── test_api.py
│   ├── test_featurebuilder.py
│   └── conftest.py
│
├── reports/
│   └── drift_report.html     # Rapport Evidently
│
├── Dockerfile                # Déploiement API
├── requirements.txt          # Dépendances
├── pytest.ini                # Configuration des tests
├── README.md
└── .gitignore
```

---

## 3. API FastAPI (local & Render)

### ▶️ Lancer l’API en local

```bash
uvicorn src.api.app:app --reload
```

### 📄 Documentation interactive

Swagger UI :  
👉 http://localhost:8000/docs

### 🌍 Version déployée sur Render

API URL (exemple) :  
👉 https://oc-p7-implementer-un-modele-de-scoring.onrender.com

### Variables d’environnement

```
MODEL_PATH=models/final_lightgbm_model.joblib
```

---

## 4. Application Streamlit

L’application Streamlit permet :

- Sélection d’un client via index  
- Envoi des données à l'API  
- Visualisation prédiction + seuil métier  
- Explications SHAP locales (waterfall + barplot)  
- Explications SHAP globales (summary plot + top 15 features)  
- Analyse du profil client vs population  

### ▶️ Lancement

```bash
streamlit run streamlit_app/streamlit_app.py
```

Pour utiliser l’API cloud, modifier dans le script :

```python
API_URL = "https://oc-p7-implementer-un-modele-de-scoring.onrender.com"
```

---

## 5. Monitoring du Data Drift

Réalisé avec Evidently.ai.

### Exemple d'exécution

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=test_df)
report.save_html("reports/drift_report.html")
```

Le rapport identifie :  
- Colonnes dérivantes  
- Distribution avant/après  
- Scores de drift  

---

## 6. Docker

### 📦 Construire l’image

```bash
docker build -t projet7-api .
```

### ▶️ Lancer le conteneur

```bash
docker run -p 8000:8000 projet7-api
```

L'API est alors accessible sur :  
👉 http://localhost:8000  
👉 http://localhost:8000/docs

---

## 7. Tests unitaires

### ▶️ Exécuter les tests

```bash
pytest -q
```

### Contenu des tests

✔ `test_api.py`  
- API disponible  
- Retour formaté correctement  
- Probabilité valide (0–1)

✔ `test_featurebuilder.py`  
- Le FeatureBuilder transforme correctement un échantillon  
- Colonnes attendues présentes

✔ Tests additionnels recommandés  
- Test sur seuil métier  
- Test valeurs manquantes  
- Test formats non numériques  

---

## 8. Installation & exécution

### 💾 Cloner le projet

```bash
git clone https://github.com/MargauxDupeyron/OC-P7-Implementer-un-modele-de-scoring.git
cd Projet_7
```

### 🧰 Installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate       # Mac/Linux  
.\.venv\Scriptsctivate        # Windows

pip install -r requirements.txt
```

---

## 🎯 Résumé

Ce projet couvre l’ensemble du cycle ML :

- ✔ Industrialisation d’un modèle ML  
- ✔ Mise en production via FastAPI  
- ✔ Interface utilisateur Streamlit  
- ✔ Monitoring du data drift  
- ✔ Dockerisation  
- ✔ Tests unitaires  
- ✔ Déploiement cloud Render  

---

