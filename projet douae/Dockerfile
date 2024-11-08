# Utiliser l'image de base Python
FROM python:3.10


# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier les fichiers nécessaires dans l'image Docker
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier les modèles et autres fichiers nécessaires dans le container
COPY ./artefacts /app/artefacts

# Exposer le port sur lequel l'application Flask va écouter
EXPOSE 5000

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
