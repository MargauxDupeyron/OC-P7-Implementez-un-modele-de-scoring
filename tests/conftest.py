"""
Fixtures utilitaires utilisées dans les tests de l'API Home Credit.

Ce module fournit plusieurs fixtures PyTest permettant de :
- charger la liste officielle des features utilisées par le pipeline final,
- construire un échantillon minimal de données d'entrée pour les endpoints,
- vérifier la disponibilité de l'API avant d'exécuter les tests d'intégration.

Ces fixtures sont utilisées dans les tests fonctionnels (/predict_proba et 
/shap_explanation) afin de garantir la cohérence des données envoyées et la
répétabilité des scénarios de test. Elles facilitent également le mock ou la
configuration future d'autres tests si l'API évolue.

Ce fichier remplace les anciennes fixtures liées au FeatureBuilder ou au 
modèle LightGBM brut, maintenant obsolètes depuis que le pipeline final est
déployé sous la forme d’un modèle unique (model.pkl) et consommé directement
par l'API.
"""

import pytest
import json
import requests


# ----- Fixture : chemin API -----
@pytest.fixture
def api_url():
    return "http://localhost:8000"


# ----- Fixture : vérifier si l'API tourne -----
@pytest.fixture
def api_available(api_url):
    try:
        requests.get(api_url, timeout=1)
        return True
    except:
        return False


# ----- Fixture : charger les noms des features -----
@pytest.fixture
def feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)


# ----- Fixture : exemple d'entrée -----
@pytest.fixture
def sample_features():
    """Échantillon minimal utilisé dans plusieurs tests."""
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.3,
        "EXT_SOURCE_3": 0.2,
        "AMT_ANNUITY": 20000,
        "DAYS_EMPLOYED": -1200
    }
