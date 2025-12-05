import pytest
import requests

API_URL = "http://localhost:8000"

def api_available():
    try:
        requests.get(API_URL, timeout=1)
        return True
    except:
        return False

@pytest.mark.skipif(not api_available(), reason="API non disponible")
def test_api_predict():
    """Test de l'API FastAPI pour vérifier /predict_proba."""
    sample = {
        "AMT_ANNUITY": 20000,
        "EXT_SOURCE_1": 0.45,
        "EXT_SOURCE_2": 0.33,
        "EXT_SOURCE_3": 0.50,
    }

    response = requests.post(f"{API_URL}/predict_proba",
                             json={"data": sample})

    assert response.status_code == 200

    data = response.json()

    assert "probability_default" in data
    assert 0 <= data["probability_default"] <= 1
    assert "decision" in data
