# ===========================
# 1. Image de base
# ===========================
FROM python:3.11-slim

# Ne pas générer de .pyc et forcer le flush stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===========================
# 2. Dépendances système
# ===========================
# LightGBM a besoin de quelques libs (libgomp1).
# build-essential sert à compiler certains paquets si besoin.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ===========================
# 3. Dossier de travail
# ===========================
WORKDIR /app

# ===========================
# 4. Installation des dépendances Python
# ===========================
# On copie uniquement le fichier de requirements pour profiter du cache Docker
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# 5. Copie du code dans l'image
# ===========================
COPY . .

# ===========================
# 6. Variables d'environnement
# ===========================
# Chemin du modèle dans l'image Docker
ENV MODEL_PATH="models/model.pkl"

# Port d'écoute de l'API
ENV PORT=8000

# ===========================
# 7. Commande de lancement
# ===========================
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port $PORT"]
