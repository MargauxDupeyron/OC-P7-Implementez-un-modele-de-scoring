import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import traceback


# =============================================
# 🔧 CONFIGURATION DES CHEMINS
# =============================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "final_lightgbm_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.json"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"
FEATUREBUILDER_PATH = BASE_DIR / "models" / "featurebuilder.pkl"

# =============================================
# 🚀 INITIALISATION FASTAPI
# =============================================
app = FastAPI(title="Home Credit API", version="1.0")


# =============================================
# 📥 SCHEMA D'ENTRÉE (simple)
# =============================================
class ClientData(BaseModel):
    data: dict


# =============================================
# 📦 CHARGEMENT DES RESSOURCES
# =============================================

print("📂 Chargement du modèle…")
model = joblib.load(MODEL_PATH)   # 👉 LightGBM pur
print("   ✔ Modèle OK :", type(model))

print("📂 Chargement FeatureBuilder…")
feature_builder = joblib.load(FEATUREBUILDER_PATH)
print("   ✔ FeatureBuilder OK")

print("📂 Chargement features…")
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)
print("   ✔", len(feature_names), "features chargées")

print("📂 Chargement du seuil métier…")
with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)
threshold = threshold_data["threshold"]
print("   ✔ Seuil utilisé =", threshold)

# =============================================
# 🌳 SHAP INITIALISATION
# =============================================
print("📂 Initialisation SHAP…")
explainer = shap.TreeExplainer(model)   # 👉 LightGBM pur
print("   ✔ SHAP prêt")


# =============================================
# 🏁 ROUTE RACINE
# =============================================
@app.get("/")
def root():
    return {
        "message": "API Home Credit OK",
        "model_loaded": True,
        "n_features": len(feature_names),
        "decision_threshold": threshold
    }


# =============================================
# 🔧 FONCTION PRÉPARATION FEATURES
# =============================================
def prepare_features(input_dict: dict) -> pd.DataFrame:

    df = pd.DataFrame([input_dict])

    # 1) FeatureBuilder : ajoute/retire colonnes
    df_clean = feature_builder.transform(df)

    # 2) Réordonnancement exact
    df_clean = df_clean[feature_names]

    print("📌 Colonnes envoyées au modèle :", df_clean.columns.tolist())
    print("📌 Nombre colonnes :", df_clean.shape[1])

    return df_clean


# =============================================
# 🔥 ENDPOINT PREDICTION
# =============================================
@app.post("/predict_proba")
def predict_proba(payload: ClientData):

    try:
        input_data = payload.data
        df_prepared = prepare_features(input_data)

        # LightGBM pur → predict_proba OK
        proba = float(model.predict_proba(df_prepared)[:, 1])

        decision = 1 if proba >= threshold else 0

        return {
            "probability_default": proba,
            "decision": decision,
            "threshold_used": threshold
        }

    except Exception as e:
        print("\n🔥 ERREUR DÉTAILLÉE :")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur modèle : {str(e)}")


# =============================================
# 📊 ENDPOINT SHAP LOCAL (VERSION PROPRE & ROBUSTE)
# =============================================
@app.post("/shap_explanation")
def shap_local(payload: ClientData):

    try:
        # 1) Extraction des données envoyées
        input_data = payload.data

        # 2) Préparation des features (FeatureBuilder + réordonnancement)
        df_prepared = prepare_features(input_data)

        # 3) Calcul des SHAP values
        shap_values = explainer.shap_values(df_prepared)

        # Pour modèles binaires LightGBM → shap_values = [classe0, classe1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 4) Expected value (LightGBM renvoie parfois une liste)
        if isinstance(explainer.expected_value, list):
            expected_val = explainer.expected_value[1]
        else:
            expected_val = explainer.expected_value

        # 5) Construction du JSON de réponse
        explanation = {
            "expected_value": float(expected_val),
            "shap_values": shap_values[0].tolist(),
            "features": df_prepared.iloc[0].tolist(),
            "feature_names": df_prepared.columns.tolist()
        }

        return explanation

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur SHAP : {str(e)}"
        )
