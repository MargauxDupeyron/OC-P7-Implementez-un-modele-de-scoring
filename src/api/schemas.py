from pydantic import BaseModel
from typing import Optional

class ClientData(BaseModel):
    """
    Contient toutes les features nécessaires pour un client.
    Optionnel : si une feature manque → FeatureBuilder la remplira à 0.
    """

    # On accepte n'importe quelle feature => dict libre
    # (plus simple pour OC)
    data: dict
    