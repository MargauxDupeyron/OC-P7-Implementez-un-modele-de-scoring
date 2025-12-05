import joblib
import numpy as np
import pytest
import os

MODEL_PATH = "models/final_lightgbm_model.joblib"

def test_model_loading():
    """Vérifie que le modèle se charge correctement."""
    assert os.path.exists(MODEL_PATH), "Le modèle n'existe pas."
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predict_shape():
    """Vérifie que predict_proba renvoie une probabilité valide."""
    model = joblib.load(MODEL_PATH)

    # 84 features en entrée : on envoie quelque chose de neutre
    X_dummy = np.zeros((1, model.n_features_))

    proba = model.predict_proba(X_dummy)
    
    assert proba.shape == (1, 2)
    assert 0 <= proba[0][1] <= 1
