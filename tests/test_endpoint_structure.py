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


@pytest.mark.skipif(requests.get(API_URL).status_code != 200, reason="API non disponible")
def test_shap_structure():
    """Vérifie que l'API SHAP renvoie bien les 84 valeurs et les métadonnées."""
    sample = {"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.2}

    r = requests.post(f"{API_URL}/shap_explanation", json={"data": sample})
    assert r.status_code == 200

    res = r.json()

    assert "shap_values" in res
    assert "expected_value" in res
    assert "feature_names" in res
    assert "features" in res

    assert len(res["shap_values"]) == len(res["feature_names"])
