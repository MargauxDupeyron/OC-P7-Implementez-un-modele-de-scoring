import pandas as pd
import numpy as np 

def cleanup_inf_to_nan(X):
    """Uniformise les données numériques avant imputation.
    Remplace inf/-inf par NaN et force float64 pour l'imputation.
    """
    if isinstance(X, pd.DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.astype(np.float64)
    X = X.astype(np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X