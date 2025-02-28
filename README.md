# Gen_IA-Outil_de_recherche_contextuelle-Chatbot

# Outil de recherche contextuelle 💻📚

Cet instrument est une ressource d'information conviviale pensée pour simplifier la quête d'informations. 
Les utilisateurs ont la possibilité d'entrer des URL d'articles et de poser des questions afin d'obtenir des renseignements appropriés.

![Logo du projet](images/logo.png)

## Fonctionnalités de l'application

- Extraction automatique du contenu des articles à partir des URL fournies.
- Division intelligente du texte en segments pour un traitement plus efficace.
- Génération d’embeddings via Hugging Face (`all-MiniLM-L6-v2`) pour une meilleure compréhension sémantique.
- Indexation et stockage des embeddings dans FAISS pour une recherche rapide.
- Possibilité de poser des questions en langage naturel sur les articles traités.
- Affichage des résultats avec les sources d'information correspondantes.

## Installation

1. Clonez ce dépôt sur votre machine locale en exécutant :

   ```bash
   git clone https://github.com/votre-repo.git
   ```

2. Accédez au répertoire du projet :

   ```bash
   cd votre-repo
   ```

3. Installez les dépendances requises avec `pip` :

   ```bash
   pip install -r requirements.txt
   ```

4. Configurez votre clé API en créant un fichier `.env` à la racine du projet et en ajoutant votre clé API :

   ```env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

1. Exécutez l'application Streamlit avec la commande suivante :

   ```bash
   streamlit run main.py
   ```


2. L'application web s'ouvrira automatiquement dans votre navigateur.


3. Dans les premières barres, vous pourrez entrer les URL des articles à analyser.


4. Lancez le traitement des articles en cliquant sur **"Lancer la recherche"**.


5. L'application effectuera le chargement des articles, la segmentation du texte, la génération d’embeddings et l’indexation dans FAISS.


6. Une fois le traitement terminé, posez une question et obtenez une réponse basée sur le contenu des articles.

## Structure du projet

1. main.py : Le script principal de l'application Streamlit.


2. requirements.txt : Une liste des paquets Python requis pour le projet.


3. .env : Fichier de configuration pour stocker votre clé API.
