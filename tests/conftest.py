import pytest
import json
import joblib
import requests

@pytest.fixture
def featurebuilder():
    return joblib.load("models/featurebuilder.pkl")

@pytest.fixture
def feature_names():
    with open("models/feature_names.json", "r") as f:
        return json.load(f)

@pytest.fixture
def model():
    return joblib.load("models/final_lightgbm_model.joblib")

@pytest.fixture
def sample_features():
    return {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.3,
        "EXT_SOURCE_3": 0.2,
        "AMT_ANNUITY": 20000,
        "DAYS_EMPLOYED": -1200
    }

@pytest.fixture
def api_url():
    return "http://localhost:8000"

@pytest.fixture
def api_available(api_url):
    try:
        requests.get(api_url, timeout=1)
        return True
    except:
        return False