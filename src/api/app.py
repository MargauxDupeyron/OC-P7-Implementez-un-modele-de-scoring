import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import traceback
from src.utils.preprocessing import cleanup_inf_to_nan



# =====================================================
# 🔧 CONFIGURATION DES CHEMINS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"          # ✔ pipeline final
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


# =====================================================
# 🚀 INITIALISATION FASTAPI
# =====================================================
app = FastAPI(title="Home Credit API", version="2.0")


# =====================================================
# 📥 INPUT UTILISATEUR
# =====================================================
class ClientData(BaseModel):
    data: dict


# =====================================================
# 📦 CHARGEMENT DES RESSOURCES
# =====================================================
print("📂 Chargement du modèle pipeline…")
model = joblib.load(MODEL_PATH)     # 👉 Pipeline complet
print("   ✔ Pipeline chargé :", type(model))

print("📂 Chargement features…")
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)
print("   ✔", len(feature_names), "features")

print("📂 Chargement du seuil métier…")
with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)
threshold = threshold_data["threshold"]
print("   ✔ Seuil =", threshold)


# =====================================================
# 🌳 SHAP INITIALISATION
# =====================================================
print("📂 Initialisation SHAP…")

# 👉 On récupère le vrai modèle LGBM à l’intérieur du pipeline
model_lgbm = model.named_steps["model"]

explainer = shap.TreeExplainer(model_lgbm)
print("   ✔ SHAP initialisé")


# =====================================================
# 🏁 ROUTE RACINE
# =====================================================
@app.get("/")
def root():
    return {
        "message": "API Home Credit OK",
        "model_loaded": True,
        "n_features": len(feature_names),
        "decision_threshold": threshold
    }


# =====================================================
# 🔧 PREPARATION DES FEATURES
# =====================================================
def prepare_features(input_dict: dict) -> pd.DataFrame:

    df = pd.DataFrame([input_dict])

    # 👉 On applique EXACTEMENT les colonnes attendues par le modèle final
    df = df.reindex(columns=feature_names, fill_value=0)

    print("📌 Colonnes envoyées au modèle :", list(df.columns))
    print("📌 Shape =", df.shape)

    return df


# =====================================================
# 🔥 ENDPOINT PREDICTION
# =====================================================
@app.post("/predict_proba")
def predict(payload: ClientData):

    try:
        # 1) On prépare les features
        df_prepared = prepare_features(payload.data)

        # 2) Le pipeline s’occupe du reste (cleanup + imputer + LGBM)
        proba = float(model.predict_proba(df_prepared)[:, 1])

        decision = 1 if proba >= threshold else 0

        return {
            "probability_default": proba,
            "decision": decision,
            "threshold_used": threshold
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Erreur modèle : {str(e)}")


# =====================================================
# 📊 ENDPOINT SHAP LOCAL
# =====================================================
@app.post("/shap_explanation")
def shap_local(payload: ClientData):

    try:
        # -----------------------
        # Safe float converter
        # -----------------------
        def safe_float(x) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        # 1) Input
        input_data = payload.data

        # 2) FeatureBuilder + reorder
        df_prepared = prepare_features(input_data)

        # 3) SHAP values
        shap_values = explainer.shap_values(df_prepared)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 4) Expected value
        raw_expected = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        )
        expected_value = safe_float(raw_expected)

        # 5) SHAP + features cleaned
        shap_values_clean: list[float] = [
            safe_float(v) for v in shap_values[0].tolist()
        ]

        feature_values_clean: list[float] = [
            safe_float(v) for v in df_prepared.iloc[0].tolist()
        ]

        feature_names_clean: list[str] = df_prepared.columns.tolist()

        # 6) JSON
        explanation = {
            "expected_value": expected_value,
            "shap_values": shap_values_clean,
            "features": feature_values_clean,
            "feature_names": feature_names_clean
        }

        return explanation

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur SHAP : {str(e)}"
        )


