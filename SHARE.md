# Partage du projet boosted-App

Ce document explique comment vos camarades peuvent récupérer, installer et lancer rapidement le projet `boosted-App`.

## 1) Récupération

- Depuis GitHub (recommandé si le dépôt est public) :

  ```bash
  git clone https://github.com/benfai14-oss/boosted-App.git
  cd boosted-App
  ```

- Si vous recevez une archive ZIP (fichier `boosted-App-share.zip`) :

  - Décompressez l'archive puis entrez dans le dossier décompressé.

## 2) Environnement Python

Le projet est en Python. Il contient un fichier `requirements.txt` listant les dépendances.

1. (Recommandé) Créez un environnement virtuel :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

3. (Optionnel) Si vous utilisez `pip` dans un environnement qui ne supporte pas `python3` directement, remplacez `python3` par `python`.

## 3) Démarrage rapide

- Lancer le script principal (exemple) :

  ```bash
  python main.py
  ```

- Si le projet inclut une interface Streamlit (vérifiez `README.md` ou `app.py`) :

  ```bash
  streamlit run app.py
  ```

## 4) Données

- Les données (raw/processed) sont incluses dans le dossier `data/`. Si certaines sources doivent être téléchargées (API keys, gros fichiers), elles sont documentées dans le `README.md`.

## 5) Partage via GitHub (si vous voulez pousser)

- Si vous souhaitez créer un dépôt GitHub et pousser le projet (nécessite `gh` ou accès GitHub) :

  ```bash
  # créer le repo localement (optionnel)
  git init
  git add .
  git commit -m "Initial commit"

  # Créer le repo distant (si vous avez l'outil gh)
  gh repo create boosted-App --public --source=. --remote=origin --push
  ```

  Si vous n'avez pas `gh`, créez le repo sur github.com puis ajoutez le remote et poussez :

  ```bash
  git remote add origin https://github.com/<votre-compte>/boosted-App.git
  git branch -M main
  git push -u origin main
  ```

## 6) Conseils pour les camarades

- Si vous rencontrez des problèmes d'installation, partagez la sortie d'erreur.
- N'oubliez pas d'activer l'environnement virtuel avant d'installer.
- Pour contributions : forkez et ouvrez une pull request.

## 7) Contact

Pour toute question, contactez l'auteur du projet ou l'équipe pédagogique.

---

Fichier généré automatiquement pour faciliter le partage.
