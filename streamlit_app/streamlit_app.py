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

from dotenv import load_dotenv


# =========================================================
# 1. INIT CONFIG
# =========================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

load_dotenv()

API_URL = os.getenv("API_URL")

DATA_PATH = "data/processed/df_filtre.csv"

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/final_lightgbm_model.joblib"))
FEATURES_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/feature_names.json"))
FEATUREBUILDER_PATH = os.path.abspath(os.path.join(APP_DIR, "../models/featurebuilder.pkl"))


# =========================================================
# 2. LOAD MODEL + FEATURES
# =========================================================

model = joblib.load(MODEL_PATH)
feature_builder = joblib.load(FEATUREBUILDER_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

explainer = shap.TreeExplainer(model)


# =========================================================
# 3. UTILS
# =========================================================

def normalize_col(c):
    return re.sub(r"[^A-Za-z0-9_]", "_", c.strip())


@st.cache_data
def load_test_data(path):
    df = pd.read_csv(path)
    df.columns = [normalize_col(c) for c in df.columns]
    return df


def clean_for_api(d):
    """Transforme NaN → None pour API JSON."""
    return {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in d.items()}


def prepare_features(df_row):
    df = pd.DataFrame([df_row])
    df2 = feature_builder.transform(df)
    return df2.reindex(columns=FEATURE_NAMES)


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


# =============================
# 6. Sélection du client
# =============================

st.subheader("1️⃣ Sélection du client")

# 🔹 état persistant
if "idx" not in st.session_state:
    st.session_state.idx = 0

max_idx = n_rows - 1  # ⭐ valeur réelle max

# Petit encadré esthétique
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

# Champ pour saisir l'index
idx_text = st.text_input("Index client :", value=str(st.session_state.idx))

# Validation
try:
    idx = int(idx_text)
    if idx < 0:
        idx = 0
    if idx > max_idx:
        idx = max_idx
except:
    st.warning("Veuillez saisir un entier valide.")
    idx = st.session_state.idx  # reste sur l’ancien

# Sauvegarde dans session_state
st.session_state.idx = idx

# Sélection du client
sample = df_test.iloc[idx]

st.write(f"### 🔍 Client sélectionné : **#{idx}**")
st.dataframe(sample.to_frame("valeur"), use_container_width=True)


# =========================================================
# 7. API STATUS
# =========================================================

st.subheader("2️⃣ Statut de l’API")

try:
    r = requests.get(f"{API_URL}/")
    if r.status_code == 200:
        st.success("API opérationnelle ✔️")
    else:
        st.warning(f"L’API répond : {r.status_code}")
except:
    st.error("❌ API non joignable")
    st.stop()


# =========================================================
# 8. PREDICTION
# =========================================================

st.subheader("3️⃣ Prédiction & Explicabilité")

payload = {"data": clean_for_api(sample.to_dict())}

if st.button("🚀 Lancer la prédiction"):

    with st.spinner("Calcul en cours..."):

        # -------------------------
        # Call API
        # -------------------------
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

        # =========================================================
        # SHAP FROM API
        # =========================================================
        shap_api = requests.post(f"{API_URL}/shap_explanation", json=payload).json()

        # L'API renvoie directement la liste des 84 valeurs
        shap_values = np.array(shap_api["shap_values"])


        expected_value = shap_api["expected_value"]
        feature_values = shap_api["features"]
        feature_names = shap_api["feature_names"]

        df_explain = pd.DataFrame([feature_values], columns=feature_names)

        # =========================================================
        # TABS : Profil Client | SHAP Local | SHAP Global
        # =========================================================

        tab_profile, tab_local, tab_global = st.tabs([
            "🧑‍💼 Profil client",
            "🔍 SHAP Local",
            "🌍 SHAP Global"
        ])

        # ================================
        # 🧑‍💼 ONGLET PROFIL CLIENT
        # ================================
        with tab_profile:

            st.markdown("### Comparaison du client vs population")

            # Sélectionner les 10 premières variables numériques
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


        # ================================
        # 🔍 ONGLET SHAP LOCAL
        # ================================
        with tab_local:

            st.markdown("### SHAP – Importance locale du client")

            # SHAP values déjà reçus depuis l'API
            shap_values_np = np.array(shap_values).reshape(-1)

            explanation = shap.Explanation(
                values=shap_values_np,
                base_values=expected_value,
                data=df_explain.values[0],
                feature_names=feature_names
            )

            fig = plt.figure(figsize=(6, 4))
            shap.plots.waterfall(explanation, max_display=15, show=False)  # ⭐ compact & professionnel
            st.pyplot(fig)
            plt.close(fig)



        # ================================
        # 🌍 ONGLET SHAP GLOBAL
        # ================================
        with tab_global:

            st.markdown("### 🌍 Importance globale (SHAP)")

            # -----------------------------
            # 1. Préparation des données
            # -----------------------------
            X_sample = df_test.sample(1000, random_state=42)
            X_sample_fb = feature_builder.transform(X_sample)
            X_sample_fb = X_sample_fb.reindex(columns=FEATURE_NAMES)

            shap_global = explainer.shap_values(X_sample_fb)
            if isinstance(shap_global, list):  # modèle binaire
                shap_global = shap_global[1]

            mean_abs = np.abs(shap_global).mean(axis=0)
            df_import = (
                pd.DataFrame({
                    "feature": FEATURE_NAMES,
                    "importance": mean_abs
                })
                .sort_values("importance", ascending=False)
                .head(15)
            )

            # -----------------------------
            # 2. Deux colonnes côte à côte
            # -----------------------------
            col1, col2 = st.columns(2)

            # 🟦 Colonne 1 : Summary Plot
            with col1:
                st.markdown("### Summary Plot")

                fig1 = plt.figure(figsize=(5, 4))
                shap.summary_plot(shap_global, X_sample_fb, max_display=15, show=False)
                st.pyplot(fig1)
                plt.close(fig1)

            # 🟥 Colonne 2 : Barplot Top 15
            with col2:
                st.markdown("### Top 15 variables")

                fig2, ax = plt.subplots(figsize=(5, 4))
                ax.barh(df_import["feature"][::-1], df_import["importance"][::-1], color="#2980b9")
                ax.set_title("Importance globale (Top 15)")
                st.pyplot(fig2)
                plt.close(fig2)
