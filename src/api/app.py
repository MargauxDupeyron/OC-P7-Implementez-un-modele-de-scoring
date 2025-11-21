import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1) Chargement des variables d'environnement (.env à la racine)
# -------------------------------------------------------------------
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_lightgbm_model.joblib")

# Objets globaux (chargés au démarrage)
_model = None          # pipeline sklearn (imputer + LGBM)
_threshold = 0.5       # seuil métier
_model_name = "LightGBM_final"

def normalize_feature_name(name: str) -> str:
    """
    Reproduit le nettoyage des noms de colonnes utilisé à l'entraînement.
    (à adapter si besoin en fonction de ton notebook de modélisation)
    """
    return (
        name
        .replace(" ", "_")   # espaces -> _
        .replace("-", "_")   # tirets -> _
        .replace(":", "")    # on enlève les ":" (comme dans Cash X-Sell: high)
    )


# -------------------------------------------------------------------
# 2) Schémas Pydantic
# -------------------------------------------------------------------
class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
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
    global _model, _threshold

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Fichier de modèle introuvable : {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        _model = bundle.get("model", None)
        _threshold = float(bundle.get("threshold", 0.5))
    else:
        _model = bundle
        _threshold = 0.5

    if _model is None:
        raise RuntimeError("Impossible de récupérer 'model' depuis le bundle joblib.")

    print(f"[API] Modèle chargé depuis {MODEL_PATH} avec seuil = {_threshold:.3f}")


# -------------------------------------------------------------------
# 4) App FastAPI
# -------------------------------------------------------------------
app = FastAPI(
    title="Home Credit Default Risk - Scoring API",
    description="API de scoring pour le projet 7 OpenClassrooms (LightGBM).",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"[API] Erreur au chargement du modèle : {e}")


# -------------------------------------------------------------------
# 5) Endpoints
# -------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
def root():
    """Petit endpoint d'accueil + état du modèle."""
    is_loaded = _model is not None
    return HealthResponse(
        status="ok" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
        model_path=MODEL_PATH,
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    is_loaded = _model is not None
    return HealthResponse(
        status="ok" if is_loaded else "model_not_loaded",
        model_loaded=is_loaded,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    # 1) Normaliser les noms de features (pour coller à ceux vus au fit)
    try:
        norm_features = {
            normalize_feature_name(k): v
            for k, v in request.features.items()
        }
        df = pd.DataFrame([norm_features])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing des features : {e}")

    # 2) Prédiction proba (classe 1 = défaut)
    try:
        proba = _model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

    pred = int(proba >= _threshold)

    return PredictionResponse(
        probability=float(proba),
        prediction=pred,
        threshold=float(_threshold),
        model_name=_model_name,
        message="Succès de la prédiction."
    )

