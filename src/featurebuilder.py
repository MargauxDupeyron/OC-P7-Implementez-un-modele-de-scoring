# featurebuilder.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Version robuste pour production.

    - Aligne les colonnes sur celles du training (features_keep)
    - Ajoute les colonnes manquantes (remplies à 0)
    - Supprime les colonnes inconnues
    - Gère les NaN / inf / -inf
    """

    def __init__(self, features_keep):
        self.features_keep = features_keep
        self.final_columns = list(features_keep)

    def fit(self, X, y=None):
        # On fige seulement l’ordre des colonnes du training
        self.final_columns = list(self.features_keep)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureBuilder.transform attend un DataFrame pandas.")

        df = X.copy()

        # Ajout des colonnes manquantes
        for col in self.final_columns:
            if col not in df.columns:
                df[col] = 0

        # Suppression des colonnes inattendues
        df = df[self.final_columns]

        # Nettoyage valeurs aberrantes
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Conversion en float (LightGBM friendly)
        df = df.astype(float)

        return df
