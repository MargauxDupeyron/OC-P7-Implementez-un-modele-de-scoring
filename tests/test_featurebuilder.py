import joblib
import pandas as pd
import json
import os

def test_featurebuilder_transform():
    """Test unitaire sur le FeatureBuilder (obligatoire pour OC)."""

    # Chargement du builder
    fb = joblib.load("models/featurebuilder.pkl")

    # Chargement des features finales
    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)

    # Exemple d'entrée minimale
    sample = pd.DataFrame([{
        "AMT_ANNUITY": 10000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.3,
    }])

    # Transformation
    transformed = fb.transform(sample)

    # Tests
    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == feature_names
    assert transformed.shape[1] == len(feature_names)
