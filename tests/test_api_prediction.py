"""
Test fonctionnel du endpoint /predict_proba de l'API Home Credit.

Ce module vérifie le comportement de l'API en condition réelle, en envoyant une
requête HTTP à l'endpoint /predict_proba. Il s'agit d'un test d'intégration,
car il contrôle non seulement la logique du modèle mais aussi :

- le bon lancement de l'API FastAPI,
- la capacité de l'API à recevoir et interpréter une requête JSON,
- la cohérence de la réponse selon le contrat d’inférence défini.

Plus précisément, le test contient :

1. Une vérification préalable que l’API est disponible.
   Cela permet d'éviter les faux négatifs lorsque le serveur n'est pas en cours d'exécution.

2. Un appel réel à l'endpoint /predict_proba avec un échantillon minimal.
   Cet échantillon ne contient que quelques features, ce qui permet de valider
   la robustesse du mécanisme interne de réindexation des features.

3. Des assertions sur la structure et la validité du résultat :
   - presence des clés probability_default, decision et threshold_used,
   - probability_default est bien un float dans l’intervalle [0, 1],
   - decision ∈ {0, 1},
   - le seuil métier (threshold_used) est correctement renvoyé.

Ce test garantit que l’API renvoie une prédiction cohérente, respectant le
schéma prévu, et que la couche inference (FastAPI + pipeline ML) fonctionne
correctement de bout en bout. Il constitue un élément clé de la validation
fonctionnelle du Projet 7.
"""


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
    """Test de /predict_proba avec un échantillon minimal."""

    sample = {
        "AMT_ANNUITY": 20000,
        "EXT_SOURCE_1": 0.45,
        "EXT_SOURCE_2": 0.33,
        "EXT_SOURCE_3": 0.50,
    }

    response = requests.post(
        f"{API_URL}/predict_proba",
        json={"data": sample},
        timeout=2
    )

    assert response.status_code == 200

    data = response.json()

    # --- Vérifications de base ---
    assert "probability_default" in data
    assert isinstance(data["probability_default"], float)
    assert 0 <= data["probability_default"] <= 1

    assert "decision" in data
    assert data["decision"] in [0, 1]

    # --- NOUVEAU : vérifier le seuil métier ---
    assert "threshold_used" in data
    assert isinstance(data["threshold_used"], float)
