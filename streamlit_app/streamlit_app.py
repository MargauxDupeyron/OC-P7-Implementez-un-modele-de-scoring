import os
import math
import base64
import requests
import pandas as pd
import numpy as np
import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
import json
import re

from dotenv import load_dotenv
from PIL import Image

# -------------------------------------------------------------------
# 1. CONFIGURATION DE BASE
# -------------------------------------------------------------------
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TEST_PATH = os.getenv("TEST_DATA_PATH", "data/processed/test_mean_imputed.csv")

# Charger pipeline final (imputer + modèle)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/final_lightgbm_model.joblib"))

model_dict = joblib.load(PIPELINE_PATH)
final_lgbm = model_dict["model"]

# Pipeline
imputer = final_lgbm.named_steps["imputer"]
model = final_lgbm.named_steps["model"]

# Explainer SHAP
explainer = shap.TreeExplainer(model)

# -------------------------------------------------------------------
# 2. FONCTIONS UTILES
# -------------------------------------------------------------------

def normalize_colname(col):
    """⭐ Normalisation complète des colonnes pour matcher le modèle."""
    col = col.strip()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace(":", "_")
    col = col.replace("/", "_")
    col = re.sub(r"__+", "_", col)
    return col

def clean_sample_for_api(sample: dict) -> dict:
    """Remplace les NaN par None pour JSON."""
    return {
        k: (None if (isinstance(v, float) and math.isnan(v)) else v)
        for k, v in sample.items()
    }

@st.cache_data
def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    # ⭐ Normalisation juste après chargement
    df.columns = [normalize_colname(c) for c in df.columns]
    return df

def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Charger les features utilisées au moment du fit
FEATURES_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/features.json"))
with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

def align_features(df, features_expected):
    """⭐ Alignement + normalisation obligatoire."""
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Ajouter colonnes manquantes
    for col in features_expected:
        if col not in df.columns:
            df[col] = np.nan

    # Retirer colonnes inutiles
    df = df[features_expected]
    return df

# -------------------------------------------------------------------
# 3. CONFIG STREAMLIT
# -------------------------------------------------------------------
st.set_page_config(page_title="Home Credit – API de scoring", layout="wide")

# -------------------------------------------------------------------
# 4. BANNIÈRE
# -------------------------------------------------------------------
BANNER_PATH = os.path.join(APP_DIR, "banner.png")

if os.path.exists(BANNER_PATH):
    img_base64 = load_image_base64(BANNER_PATH)
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_base64}" style="width:600px; margin-bottom:20px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Image non trouvée : {BANNER_PATH}")

# bandeau API
st.markdown(
    f"""
    <div style="padding:0.6rem 1rem; border-radius:0.5rem;
                background:linear-gradient(90deg,#0f4c75,#3282b8);
                color:white; margin-bottom:1rem;">
        <strong>🔌 API utilisée :</strong> {API_URL}
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Home Credit – Démo de scoring & explicabilité")

# -------------------------------------------------------------------
# 5. CHARGEMENT DES DONNÉES
# -------------------------------------------------------------------
try:
    df_test = load_test_data(TEST_PATH)
except Exception as e:
    st.error(f"❌ Erreur chargement test : {e}")
    st.stop()

n_rows = len(df_test)

# -------------------------------------------------------------------
# 6. SELECTION DU CLIENT
# -------------------------------------------------------------------
st.markdown("## 1️⃣ Sélection du client")

col_left, col_right = st.columns([2, 1])

with col_left:
    tab1, tab2 = st.tabs(["🔢 Saisie manuelle", "🎚️ Slider"])

    with tab1:
        idx = st.number_input(
            "Numéro de client (index)",
            min_value=0, max_value=n_rows - 1, value=0
        )

    with tab2:
        idx = st.slider("Choix visuel de l’index", 0, n_rows - 1, 0)

with col_right:
    st.metric("Nombre total de clients (test)", n_rows)

sample = df_test.iloc[idx]  # ⭐ Colonnes déjà normalisées plus haut

st.markdown("### 🔍 Détails du client")
st.dataframe(sample.to_frame(name="valeur"), use_container_width=True)

# -------------------------------------------------------------------
# 7. STATUT API
# -------------------------------------------------------------------
st.markdown("## 2️⃣ Statut de l’API")

try:
    health = requests.get(f"{API_URL}/health", timeout=20)
    if health.status_code == 200:
        st.success("API opérationnelle ✔️")
    else:
        st.warning(f"API répond avec le code {health.status_code}")
except:
    st.error("❌ API non joignable")

# -------------------------------------------------------------------
# 8. PREDICTION & SHAP
# -------------------------------------------------------------------
st.markdown("## 3️⃣ Prédiction & Explicabilité")

if st.button("🚀 Lancer la prédiction pour ce client"):
    with st.spinner("Calcul en cours..."):

        # ---- API PRED ----
        clean_features = clean_sample_for_api(sample.to_dict())
        payload = {"features": clean_features}

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            result = resp.json()
        except Exception as e:
            st.error(f"❌ Erreur API : {e}")
            st.stop()

        prob = result.get("probability")
        pred = result.get("prediction")
        thr = result.get("threshold")

        tab_pred, tab_local, tab_global = st.tabs(
            ["🧮 Prédiction", "🔍 SHAP locale", "🌍 SHAP globale"]
        )

        # ---- PRED ----
        with tab_pred:
            st.subheader("📌 Résultat")
            st.metric("Probabilité", f"{prob:.3f}")
            st.metric("Décision", "Risque" if pred else "Accepté")
            st.metric("Seuil métier", f"{thr:.3f}")

        # ---- SHAP LOCALE ----
        with tab_local:
            st.subheader("🔍 SHAP locale")

            x_client = align_features(sample.to_frame().T, FEATURES)
            x_client_imp = imputer.transform(x_client)

            shap_client = explainer(x_client_imp)

            fig_force = shap.force_plot(
                shap_client.base_values[0],
                shap_client.values[0],
                x_client,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force)

        # ---- SHAP GLOBALE ----
        with tab_global:
            st.subheader("🌍 SHAP globale")

            X_sample = df_test.sample(n=1500, random_state=42)
            X_sample = align_features(X_sample, FEATURES)
            X_sample_imp = imputer.transform(X_sample)

            shap_values = explainer(X_sample_imp)

            fig_summary = plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values.values,
                X_sample,
                feature_names=X_sample.columns.tolist(),
                plot_type="violin",
                max_display=20,
                show=False
            )
            st.pyplot(fig_summary)

            # barplot global
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            df_global = (
                pd.DataFrame({
                    "feature": X_sample.columns,
                    "mean_abs_shap": mean_abs_shap
                })
                .sort_values("mean_abs_shap", ascending=False)
                .head(15)
                .iloc[::-1]
            )

            fig_gbar, ax = plt.subplots(figsize=(8, 6))
            ax.barh(df_global["feature"], df_global["mean_abs_shap"])
            ax.set_title("Top 15 – importance globale SHAP")
            st.pyplot(fig_gbar)


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Projet 7 – API FastAPI · Streamlit · LightGBM · SHAP · Docker")

