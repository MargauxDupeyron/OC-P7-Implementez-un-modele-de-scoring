import os, sys
import json
import math
import base64
import requests
import pandas as pd
import numpy as np
import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
import re
import urllib.parse

from dotenv import load_dotenv

print("=== DEBUG ENV ===")
print("Current working directory:", os.getcwd())
print("Env API_URL:", os.getenv("API_URL"))
print("Files in CWD:", os.listdir())
print("==================")

# =========================================================
# 1. INIT CONFIG
# =========================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

dotenv_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- API_URL sécurisée (fix définitif) ---
API_URL = os.getenv("API_URL")
if API_URL is None or API_URL.strip() == "":
    API_URL = "https://oc-p7-implementez-un-modele-de-scoring.onrender.com"

API_URL = API_URL.strip()
print("API_URL FINAL =", API_URL)

DATA_PATH = "data/processed/df_test.csv"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 2. LOAD FEATURES
# =========================================================

FEATURES_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/feature_names.json"))
with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

# =========================================================
# 3. UTILS
# =========================================================

def clean_for_api(d):
    """Assure que toutes les valeurs sont JSON-compatibles."""
    clean = {}
    for k, v in d.items():
        try:
            val = float(v)
            if not np.isfinite(val):
                val = 0.0
        except:
            val = 0.0
        clean[k] = val
    return clean


@st.cache_data
def load_test_data(path):
    df = pd.read_csv(path)
    return df.reindex(columns=FEATURE_NAMES, fill_value=0)


def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================================================
# 4. STREAMLIT UI SETUP
# =========================================================

st.set_page_config(page_title="Home Credit Scoring", layout="wide")

# Banner
BANNER_PATH = os.path.join(APP_DIR, "banner.png")
if os.path.exists(BANNER_PATH):
    img64 = load_image_base64(BANNER_PATH)
    st.markdown(
        f"""<div style="text-align:center;">
            <img src="data:image/png;base64,{img64}" style="width:600px;">
        </div>""",
        unsafe_allow_html=True
    )

# API info
st.markdown(
    f"""<div style="padding:0.6rem; background:#0f4c75; color:white; border-radius:8px;">
        API utilisée : <strong>{API_URL}</strong>
    </div>""",
    unsafe_allow_html=True
)

st.title("🏦 Home Credit – Scoring & Explicabilité")

# =========================================================
# 5. LOAD DATA
# =========================================================

try:
    df_test = load_test_data(DATA_PATH)
except Exception as e:
    st.error(f"Erreur chargement des données : {e}")
    st.stop()

n_rows = len(df_test)

# =========================================================
# 6. Sélection du client
# =========================================================

st.subheader("1️⃣ Sélection du client")

if "idx" not in st.session_state:
    st.session_state.idx = 0

max_idx = n_rows - 1

st.markdown(
    f"""
    <div style="
        padding: 12px;
        border: 1px solid #DDD;
        border-radius: 8px;
        background-color: #FAFAFA;
        margin-bottom: 10px;
    ">
        <strong>Saisir un index client (entre 0 et {max_idx}) :</strong>
    </div>
    """,
    unsafe_allow_html=True
)

idx_text = st.text_input("Index client :", value=str(st.session_state.idx))

try:
    idx = int(idx_text)
    idx = max(0, min(idx, max_idx))
except:
    st.warning("Veuillez saisir un entier valide.")
    idx = st.session_state.idx

st.session_state.idx = idx

sample = df_test.iloc[idx]

st.write(f"### Client sélectionné : **#{idx}**")
df_display = pd.DataFrame({
    "feature": sample.index,
    "valeur": sample.astype(str).values
})
st.dataframe(df_display, use_container_width=True)

# =========================================================
# 7. API STATUS
# =========================================================

st.subheader("2️⃣ Statut de l’API")

try:
    # Construction URL 100% fiable
    health_url = urllib.parse.urljoin(API_URL.rstrip("/") + "/", "/")
    st.write("🔍 URL appelée :", health_url)

    r = requests.get(health_url)

    if r.status_code == 200:
        st.success("API opérationnelle ✔️")
    else:
        st.warning(f"L’API répond : {r.status_code}")
        st.code(r.text)

except Exception as e:
    st.error(f"❌ API non joignable : {e}")
    st.stop()

# =========================================================
# 8. PREDICTION
# =========================================================

st.subheader("3️⃣ Prédiction & Explicabilité")

payload = {"data": clean_for_api(sample.to_dict())}

if st.button("🚀 Lancer la prédiction"):

    with st.spinner("Calcul en cours..."):

        try:
            resp = requests.post(f"{API_URL}/predict_proba", json=payload)
            pred = resp.json()
        except Exception as e:
            st.error(f"Erreur API : {e}")
            st.stop()

        proba = pred["probability_default"]
        decision = pred["decision"]
        thr = pred["threshold_used"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité défaut", f"{proba:.3f}")
        c2.metric("Décision", "❌ Risque" if decision else "✔️ Accepté")
        c3.metric("Seuil métier", thr)

        # SHAP Local
        shap_api = requests.post(f"{API_URL}/shap_explanation", json=payload).json()

        shap_values = np.array(shap_api["shap_values"])
        expected_value = shap_api["expected_value"]
        feature_values = shap_api["features"]
        feature_names = shap_api["feature_names"]

        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=np.array(feature_values),
            feature_names=feature_names
        )

        tab_profile, tab_local, tab_global = st.tabs([
            "🧑‍💼 Profil client",
            "🔍 SHAP Local",
            "🌍 SHAP Global"
        ])

        with tab_profile:

            st.markdown("### Comparaison du client vs population")

            num_cols = df_test.select_dtypes(include=["float", "int"]).columns[:10]

            col1, col2 = st.columns(2)

            for i, col in enumerate(num_cols):
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(df_test[col].dropna(), bins=30, alpha=0.6, label="Population")
                ax.axvline(sample[col], color="red", linewidth=2, label="Client")
                ax.set_title(col)
                ax.legend()

                if i < 5:
                    col1.pyplot(fig)
                else:
                    col2.pyplot(fig)

                plt.close(fig)

        # SHAP LOCAL
        with tab_local:
            st.markdown("### SHAP – Importance locale du client")
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(explanation, max_display=15, show=False)
            st.pyplot(fig)
            plt.close(fig)

        # SHAP GLOBAL
        with tab_global:
            st.markdown("### 🌍 Importance globale (SHAP)")
            shap_global_api = requests.get(f"{API_URL}/shap_global").json()

            if "feature_names" not in shap_global_api:
                st.error("Erreur API : 'feature_names' absent.")
                st.write("Réponse brute :", shap_global_api)
                st.stop()

            feature_names = shap_global_api["feature_names"]
            shap_global = np.array(shap_global_api["shap_values"], dtype=float)
            X_global = np.array(shap_global_api["data"], dtype=float)

            mean_abs = np.abs(shap_global).mean(axis=0)

            df_import = (
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": mean_abs
                })
                .sort_values("importance", ascending=False)
                .head(15)
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Summary Plot")
                fig1 = plt.figure(figsize=(5, 4))
                shap.summary_plot(shap_global, X_global, feature_names=feature_names, max_display=15, show=False)
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                st.markdown("### Top 15 variables")
                fig2, ax = plt.subplots(figsize=(5, 4))
                ax.barh(df_import["feature"][::-1], df_import["importance"][::-1], color="#2980b9")
                ax.set_title("Importance globale (Top 15)")
                st.pyplot(fig2)
                plt.close(fig2)

