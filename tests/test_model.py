"""
Tests unitaires du modèle entraîné (pipeline final).

Ce fichier vérifie deux éléments essentiels :
1. Le pipeline ML (model.pkl) se charge correctement depuis le disque.
2. Le pipeline est capable de produire une probabilité valide via predict_proba,
   lorsque l'on lui fournit un échantillon minimal contenant exactement les
   features attendues.

Ces tests garantissent que :
- le modèle n'est pas corrompu,
- les features sont cohérentes avec celles utilisées à l'entraînement,
- la prédiction fonctionne de manière stable (probabilité ∈ [0, 1]).

Ce module fait partie des tests de robustesse du modèle pour le Projet 7.
"""

import joblib
import numpy as np
import pytest
import json
import pandas as pd
import os

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.json"


def test_model_loading():
    """Vérifie que le pipeline se charge correctement."""
    assert os.path.exists(MODEL_PATH), "Le modèle n'existe pas."
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predict_shape():
    """Vérifie que le pipeline renvoie une probabilité valide."""
    
    # Chargement du pipeline
    model = joblib.load(MODEL_PATH)

    # Chargement des features attendues
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    # Création d'un échantillon minimal rempli de zéros
    X_dummy = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

    # Prédiction
    proba = model.predict_proba(X_dummy)

    assert proba.shape == (1, 2)
    assert 0 <= proba[0][1] <= 1

