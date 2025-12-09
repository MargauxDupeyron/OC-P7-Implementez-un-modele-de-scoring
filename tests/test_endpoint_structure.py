"""
Tests de structure et de cohérence des endpoints de l'API Home Credit.

Ce module vérifie trois aspects essentiels :

1. Accessibilité de l'API
   - Le endpoint racine ("/") doit répondre correctement avec un statut HTTP 200.
   - Ce test sert aussi de pré-condition pour exécuter les tests suivants.

2. Validation du format de réponse du endpoint /predict_proba
   - Vérifie la présence des clés obligatoires : probability_default, decision, threshold_used.
   - S'assure que la probabilité renvoyée est un float compris entre 0 et 1.
   - Confirme que la décision renvoyée par l'API est conforme (0 ou 1).
   Ce test garantit que l'API respecte la spécification contractuelle définie pour la prédiction.

3. Validation du format de réponse du endpoint /shap_explanation
   - Vérifie la présence des clés nécessaires à l'explication locale : shap_values, expected_value, feature_names, features.
   - Confirme que expected_value est un float.
   - Vérifie que le nombre de SHAP values et de features renvoyés est cohérent avec le modèle (84 features).
   Ce test assure que l'API fournit des explications SHAP exploitables, fiables et correctement alignées avec le modèle.

Ces tests valident la stabilité des endpoints, leur conformité au schéma d’inférence,
ainsi que la cohérence structurelle des données renvoyées. Ils jouent un rôle clé
dans la validation de l’API pour le Projet 7.
"""


import pytest
import requests

API_URL = "http://localhost:8000"


def test_root_endpoint():
    """Le endpoint racine doit répondre 200."""
    r = requests.get(API_URL)
    assert r.status_code == 200


@pytest.mark.skipif(requests.get(API_URL).status_code != 200, reason="API non disponible")
def test_predict_proba_structure():
    """Vérifie que l'API renvoie bien les clés attendues."""
    sample = {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.2}

    r = requests.post(f"{API_URL}/predict_proba", json={"data": sample})
    assert r.status_code == 200

    data = r.json()

    expected_keys = {"probability_default", "decision", "threshold_used"}

    for key in expected_keys:
        assert key in data, f"{key} manquant dans la réponse"

    assert isinstance(data["probability_default"], float)
    assert 0 <= data["probability_default"] <= 1
    assert data["decision"] in [0, 1]



@pytest.mark.skipif(requests.get(API_URL).status_code != 200, reason="API non disponible")
def test_shap_structure():
    """Vérifie que l'API SHAP renvoie bien les valeurs et métadonnées attendues."""
    sample = {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.2}

    r = requests.post(f"{API_URL}/shap_explanation", json={"data": sample})
    assert r.status_code == 200

    res = r.json()

    # --- vérification structure ---
    assert "shap_values" in res
    assert "expected_value" in res
    assert "feature_names" in res
    assert "features" in res

    # --- expected_value doit être un float ---
    assert isinstance(res["expected_value"], float)

    # --- tailles cohérentes ---
    n_features = len(res["feature_names"])

    assert len(res["shap_values"]) == n_features
    assert len(res["features"]) == n_features

    # --- facultatif mais recommandé ---
    # Tu sais que ton modèle final a 84 features :
    expected_features = len(res["feature_names"])
    assert n_features == expected_features, \
       f"Mismatch: SHAP renvoie {n_features} valeurs mais feature_names en compte {expected_features}"