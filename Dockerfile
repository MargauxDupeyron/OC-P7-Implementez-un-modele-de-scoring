# # ===========================
# 1. Image de base
# ===========================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ===========================
# 2. Dépendances système
# ===========================
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
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# 5. Copie du code
# ===========================
COPY . .

# ===========================
# 6. Copie des fichiers nécessaires à SHAP Global
# ===========================
COPY data/processed/df_test_sample.csv /app/data/processed/df_test_sample.csv

# S'assurer que le dossier existe
RUN mkdir -p /app/data/processed

# ===========================
# 7. Variables d'environnement
# ===========================
ENV MODEL_PATH="models/model.pkl"
ENV PORT=8000

# ===========================
# 8. Commande de lancement
# ===========================
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port $PORT"]
