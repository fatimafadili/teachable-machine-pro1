🤖 Machine Teachable Pro - Auto ML/DL Image Builder
Python Streamlit TensorFlow Scikit-learn License

🌍 Vue d'Ensemble
Machine Teachable Pro est une application web complète développée avec Streamlit, permettant de créer, entraîner, évaluer et déployer des modèles de Machine Learning (ML) et Deep Learning (DL) pour la classification d'images de manière entièrement automatique.

🎯 Caractéristiques Principales
✅ Interface No-Code - Aucune programmation requise
✅ Téléchargement Automatique des datasets populaires
✅ Multi-Algorithmes - 4 modèles ML + 3 architectures DL
✅ Évaluation Complète - Métriques détaillées et visualisations
✅ Déploiement Immédiat - Test en temps réel avec vos images
🚀 Démarrage Rapide
Prérequis
Python 3.8 ou supérieur
Pip (gestionnaire de packages Python)
Installation
# 1. Cloner le repository
git clone https://github.com/votre-username/machine-teachable-pro.git
cd machine-teachable-pro

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur Windows:
.\venv\Scripts\activate
# Sur Mac/Linux:
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Lancer l'application
streamlit run app.py

📦 Dépendances Principales
Les dépendances sont automatiquement installées via requirements.txt:

streamlit>=1.28.0 - Interface web

tensorflow>=2.13.0 - Deep Learning

scikit-learn>=1.3.0 - Machine Learning

pandas>=2.0.0 - Manipulation de données

numpy>=1.24.0 - Calculs scientifiques

matplotlib>=3.7.0 - Visualisations

Pillow>=10.0.0 - Traitement d'images

📊 Fonctionnalités Détaillées
🗃️ Datasets Intégrés
L'application télécharge automatiquement 4 datasets populaires :

Dataset	Classes	Images	Description
CIFAR-10	10	~10,000	Objets divers (avions, voitures, animaux...)
MNIST	10	~10,000	Chiffres manuscrits
Fashion MNIST	10	~10,000	Articles de mode
Cats vs Dogs	2	~2,000	Classification binaire chats/chiens
🤖 Algorithmes Supportés
Machine Learning Classique
🎯 Random Forest - Forêts aléatoires

📈 Logistic Regression - Régression logistique

🔍 SVM - Machines à vecteurs de support

📍 K-Nearest Neighbors - Plus proches voisins

Deep Learning
🧠 CNN Simple - Architecture convolutionnelle basique

🔄 MLP - Perceptron multicouche

🏗️ CNN VGG-like - Architecture avancée type VGG

📈 Métriques d'Évaluation
✅ Accuracy - Précision globale

📊 Matrice de Confusion - Visualisation des performances

🎯 Rapport de Classification - Precision, Recall, F1-Score

📉 Courbes d'Apprentissage - Suivi de l'entraînement (DL)

🎮 Guide d'Utilisation
Étape 1: 📥 Téléchargement des Données
Allez dans l'onglet "AUTO DATA"

Cliquez sur "JE TÉLÉCHARGE TOUT !"

Les datasets sont automatiquement téléchargés et préparés

Étape 2: 🔧 Prétraitement
Sélectionnez votre dataset dans l'onglet Preprocess

Cliquez sur "Prétraiter le Dataset"

Visualisez les images prétraitées

Étape 3: 🎯 Entraînement
Choisissez entre ML Classique ou Deep Learning

Sélectionnez votre algorithme préféré

Cliquez sur "Lancer l'Entraînement Auto"

Observez les résultats en temps réel

Étape 4: 📊 Évaluation
Analysez la matrice de confusion

Consultez les métriques détaillées

Visualisez les courbes d'apprentissage (DL)

Étape 5: 🚀 Déploiement
Uploader une image dans l'onglet Deploy

Obtenez la prédiction instantanée

Visualisez les probabilités par classe

⚙️ Configuration
Paramètres Disponibles
Taille des images : 64×64, 128×128, 224×224 pixels

Mode couleur : Niveaux de gris ou RGB

Split train/test : 10% à 40%

Architectures : 7 algorithmes différents

Structure du Projet
text
machine-teachable-pro/
├── app.py                 # Application principale
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
├── data/                 # Dossiers des datasets (auto-généré)
├── models/               # Modèles sauvegardés (auto-généré)
└── exports/              # Exports des résultats (auto-généré)
🎯 Cas d'Usage
🏫 Éducation
Apprentissage des concepts ML/DL sans codage

Expérimentation avec différents algorithmes

Visualisation des résultats d'entraînement

🔬 Prototypage Rapide
Validation de concepts de classification

Benchmark d'algorithmes

Tests préliminaires de modèles

🎨 Projets Personnels
Classification d'images personnalisées

Expérimentation créative

Apprentissage pratique de l'IA

📊 Performances Typiques
Dataset	Algorithmes	Accuracy Typique
MNIST	CNN Simple	95-98%
CIFAR-10	Random Forest	70-85%
Fashion MNIST	SVM	85-92%
Cats vs Dogs	CNN VGG-like	90-95%
