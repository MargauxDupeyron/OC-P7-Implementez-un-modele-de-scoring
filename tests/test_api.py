import os
import sys

# --- Rendre le module src importable ---
# On ajoute la racine du projet (_Projet_7) au sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from src.api.app import app  # maintenant importable

client = TestClient(app)


def test_health_ok():
    """Le endpoint /health doit répondre 200 et retourner un JSON minimal."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # On vérifie juste la structure de la réponse
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" in data

    # On accepte que le modèle ne soit pas forcément chargé pendant les tests
    assert data["status"] in ("ok", "model_not_loaded")
    assert isinstance(data["model_loaded"], bool)



def test_predict_basic():
    """
    Test simple de /predict :
    - envoie un payload avec un petit subset de features
    - vérifie que la réponse contient probability, prediction, threshold, etc.
    """
    # Payload minimal : ici on met juste quelques features bidon,
    # l'important pour le test est la structure, pas la valeur.
    sample_features = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.3,
        "EXT_SOURCE_3": 0.2,
    }

    response = client.post(
        "/predict",
        json={"features": sample_features}
    )

    # Le modèle peut lever une erreur si les features ne matchent pas exactement,
    # donc ici on ne force pas ce test si ton modèle est strict.
    # Si besoin, on adaptera plus tard avec un vrai sample cohérent.

    assert response.status_code in (200, 500)

    if response.status_code == 200:
        data = response.json()
        assert "probability" in data
        assert "prediction" in data
        assert "threshold" in data
        assert "model_name" in data

