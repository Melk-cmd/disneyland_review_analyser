# Image de base : Python 3.11 allégé
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# On installe les bibliothèques (versions fixées dans requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code (train.py) n'est pas copié ici : MLflow l'ajoute lui-même au lancement
