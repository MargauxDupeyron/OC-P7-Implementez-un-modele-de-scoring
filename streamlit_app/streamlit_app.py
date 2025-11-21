import os
import math
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1. Config de base
# -------------------------------------------------------------------
load_dotenv()

# URL de l'API (même logique que dans le notebook de test)
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Chemin vers le fichier de test
TEST_PATH = os.getenv(
    "TEST_DATA_PATH",
    "data/processed/test_mean_imputed.csv"
)

@st.cache_data
def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # On retire TARGET si elle est présente
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    return df


def clean_sample_for_api(sample: dict) -> dict:
    """Remplace les NaN par None (JSON-friendly)"""
    return {
        k: (None if (isinstance(v, float) and math.isnan(v)) else v)
        for k, v in sample.items()
    }


# -------------------------------------------------------------------
# 2. Interface Streamlit
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Home Credit - Demo API",
    layout="wide"
)

st.title("🏦 Home Credit – Démo de l’API de scoring")
st.markdown(
    """
    Cette mini-application Streamlit permet de :
    - sélectionner un client dans le **jeu de données de test**  
    - appeler l’endpoint **`/predict`** de l’API FastAPI  
    - afficher la **probabilité de défaut** et la **décision** (0/1) selon le seuil métier.  
    """
)

# -------------------------------------------------------------------
# 3. Chargement des données de test
# -------------------------------------------------------------------
st.sidebar.header("Paramètres")

try:
    df_test = load_test_data(TEST_PATH)
except Exception as e:
    st.error(f"Erreur lors du chargement de `{TEST_PATH}` : {e}")
    st.stop()

n_rows = len(df_test)
st.sidebar.write(f"Nombre de clients dans le test : **{n_rows}**")

# Sélection de l’index du client
idx = st.sidebar.number_input(
    "Index du client",
    min_value=0,
    max_value=max(0, n_rows - 1),
    value=0,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.write(f"API URL : `{BASE_URL}`")

st.subheader("1️⃣ Client sélectionné")

sample = df_test.iloc[idx]
st.write(f"**Index sélectionné :** {idx}")
st.dataframe(sample.to_frame(name="valeur"))

# -------------------------------------------------------------------
# 4. Appel à l’API /health (optionnel mais pratique)
# -------------------------------------------------------------------
st.subheader("2️⃣ Statut de l’API")

try:
    health_resp = requests.get(f"{BASE_URL}/health", timeout=5)
    if health_resp.status_code == 200:
        data = health_resp.json()
        status = data.get("status", "unknown")
        model_loaded = data.get("model_loaded", False)
        st.success(f"API OK – status: `{status}`, model_loaded: `{model_loaded}`")
    else:
        st.error(f"API répond avec le code {health_resp.status_code}")
except Exception as e:
    st.error(f"Impossible de joindre l’API : {e}")

# -------------------------------------------------------------------
# 5. Préparation du payload et appel à /predict
# -------------------------------------------------------------------
st.subheader("3️⃣ Prédiction via l’API")

if st.button("Lancer la prédiction pour ce client"):
    with st.spinner("Appel à l’API en cours..."):
        clean_features = clean_sample_for_api(sample.to_dict())
        payload = {"features": clean_features}

        try:
            resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
        else:
            if resp.status_code != 200:
                st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
            else:
                result = resp.json()
                prob = result.get("probability")
                pred = result.get("prediction")
                thr = result.get("threshold")
                model_name = result.get("model_name")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probabilité de défaut", f"{prob:.3f}")
                with col2:
                    st.metric("Seuil métier", f"{thr:.3f}")
                with col3:
                    st.metric("Décision (0=OK, 1=risqué)", str(pred))

                # Message lisible
                if pred == 1:
                    st.error("⚠️ Le client est **classé à risque** selon le seuil métier.")
                else:
                    st.success("✅ Le client est **classé acceptable** selon le seuil métier.")

                st.markdown(f"_Modèle utilisé : **{model_name}**_")
                st.json(result)
else:
    st.info("Clique sur **« Lancer la prédiction pour ce client »** pour interroger l’API.")


st.markdown("---")
st.caption("Projet 7 – API de scoring Home Credit avec FastAPI + Streamlit.")