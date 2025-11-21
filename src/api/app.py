import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1) Chargement des variables d'environnement (.env à la racine)
# -------------------------------------------------------------------
load_dotenv()

# Chemin vers le modèle (par défaut : models/final_lightgbm_model.joblib)
MODEL_PATH = os.getenv("MODEL_PATH", "models/final_lightgbm_model.joblib")

# Objets globaux
_model = None          # pipeline sklearn (imputer + LGBM)
_threshold = 0.5       # seuil métier
_model_metrics = None  # dict des métriques enregistrées
_model_name = "LightGBM_final"

# -------------------------------------------------------------------
# 2) Schémas Pydantic (request / response)
# -------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """
    Requête de prédiction :
    - features : dictionnaire {nom_feature: valeur}
    """
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """
    Réponse de prédiction :
    - probability : proba de défaut (classe 1)
    - prediction  : 0 / 1 selon le seuil métier
    - threshold   : seuil utilisé
    - model_name  : petit label pour info
    """
    probability: float
    prediction: int
    threshold: float
    model_name: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


# -------------------------------------------------------------------
# 3) Fonction de chargement du modèle
# -------------------------------------------------------------------
def load_model():
    """Charge le modèle depuis le fichier joblib."""
    global _model, _threshold, _model_metrics

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Fichier de modèle introuvable : {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    # Tu as sauvegardé un dict {"model": ..., "threshold": ..., "metrics": ...}
    if isinstance(bundle, dict):
        _model = bundle.get("model", None)
        _threshold = float(bundle.get("threshold", 0.5))
        _model_metrics = bundle.get("metrics", None)
    else:
        # Cas de secours : si jamais tu charges directement le pipeline
        _model = bundle
        _threshold = 0.5
        _model_metrics = None

    if _model is None:
        raise RuntimeError("Impossible de récupérer 'model' depuis le bundle joblib.")

    print(f"[API] Modèle chargé depuis {MODEL_PATH} avec seuil = {_threshold:.3f}")


# -------------------------------------------------------------------
# 4) Création de l'appli FastAPI
# -------------------------------------------------------------------
app = FastAPI(
    title="Home Credit Default Risk - Scoring API",
    description="API de scoring pour le projet 7 OpenClassrooms (LightGBM).",
    version="1.0.0",
)


# -------------------------------------------------------------------
# 5) Évènement de démarrage : on charge le modèle une seule fois
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        # On logge l'erreur dans la console, l'API démarrera quand même mais /health l’indiquera
        print(f"[API] Erreur au chargement du modèle : {e}")


# -------------------------------------------------------------------
# 6) Endpoints
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint de santé simple pour vérifier que l'API tourne et que le modèle est chargé."""
    is_loaded = _model is not None
    return HealthResponse(
        status="ok" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint principal de prédiction.
    On attend un JSON du type :
    {
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.3,
            ...
        }
    }
    """

    if _model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    # 1) Conversion dict -> DataFrame (1 ligne)
    try:
        df = pd.DataFrame([request.features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing des features : {e}")

    # 2) Prédiction proba (classe 1 = défaut)
    try:
        proba = _model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

    # 3) Application du seuil métier
    pred = int(proba >= _threshold)

    return PredictionResponse(
        probability=float(proba),
        prediction=pred,
        threshold=float(_threshold),
        model_name=_model_name,
        message="Succès de la prédiction."
    )

