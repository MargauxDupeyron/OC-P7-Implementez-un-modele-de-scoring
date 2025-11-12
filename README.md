# Projet Home Credit – Score de risque client

## 🎯 Objectif

Ce projet vise à développer un modèle de scoring de crédit pour l’entreprise **Prêt à dépenser**.
L’objectif est de **prédire le risque de défaut d’un client** et de mettre ce modèle à disposition
sous forme d’**API déployée sur le Cloud**, afin que les applications métiers puissent interroger
le moteur de scoring.

## 🗂 Organisation du dépôt

- `notebooks/`
  - `projet7_modelisation_mlflow.ipynb` : prétraitement, modélisation, MLflow, SHAP.
  - `projet7__data_drift_evidently.ipynb` : génération du rapport de data drift (Evidently).
  - `projet7_test_api.ipynb` : appels de test à l’API de scoring.
- `src/`
  - `api/main.py` : API de scoring (Flask/FastAPI).
  - `preprocessing/preprocess.py` : fonctions de préparation des données pour l’API.
  - `models/load_model.py` : chargement du modèle (depuis MLflow ou joblib).
  - `utils/business_metrics.py` : score métier, seuil optimal, etc.
- `reports/`
  - `drift_report.html` : rapport HTML Evidently de data drift.
  - `figures/` : graphiques (ROC, courbes business, SHAP, etc.).
- `tests/`
  - `test_preprocess.py`, `test_api.py` : tests unitaires (pytest).
- `.github/workflows/ci_cd.yml` : pipeline CI/CD (tests + déploiement API).
- `requirements.txt` : dépendances Python.
- `.gitignore` : fichiers/dossiers exclus du versioning.

## ⚙️ Installation

```bash
git clone <URL_DU_REPO>
cd home-credit-scoring

# Création de l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate

pip install -r requirements.txt
