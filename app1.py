# app.py
import os
import io
import hashlib
import json
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Tuple, List, Dict

import streamlit as st
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from skimage.feature import hog
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ML & DL libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------------------
# Configuration de base
# ---------------------------
st.set_page_config(page_title="Auto ML/DL Image Builder", layout="wide")

# ---------------------------
# Utils
# ---------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "exports"
for d in (DATA_DIR, MODELS_DIR, EXPORT_DIR):
    d.mkdir(exist_ok=True)

# Paramètres par défaut
DEFAULT_PROJECT_NAME = "auto_dataset_project"
DEFAULT_TARGET_SIZE = (64, 64)
DEFAULT_GRAY_MODE = True
DEFAULT_TEST_SIZE = 20

def hash_file_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def load_image(p: Path, target_size=(64,64), gray=True):
    try:
        img = Image.open(p).convert("RGB")
        img = img.resize(target_size)
        if gray:
            img = img.convert("L")
            arr = np.array(img).reshape(target_size + (1,))
        else:
            arr = np.array(img)
        arr = arr.astype("float32") / 255.0
        return arr
    except Exception as e:
        return None

def extract_hog_features(image_array):
    arr = (image_array.squeeze() * 255).astype("uint8")
    feat = hog(arr, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    return feat

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def find_image_files(base_dir: Path) -> List[Path]:
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(base_dir.rglob(f"*{ext}"))
        image_files.extend(base_dir.rglob(f"*{ext.upper()}"))
    return image_files

def get_class_from_path(file_path: Path, base_dir: Path) -> str:
    relative_path = file_path.relative_to(base_dir)
    if len(relative_path.parts) > 1:
        return relative_path.parts[0]
    else:
        return "unknown"

# ---------------------------
# DATASET DOWNLOADERS - AUTOMATIQUES
# ---------------------------
def download_cifar10_auto(project_data_dir: Path):
    """Télécharge CIFAR-10 automatiquement"""
    try:
        st.info("📥 Téléchargement CIFAR-10...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        for subset, X, Y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            for i in range(min(1000, len(X))):  # 1000 images par classe max
                lab = labels[int(Y[i][0])]
                class_dir = project_data_dir / "cifar10" / lab
                class_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(X[i]).save(class_dir / f"{lab}_{i:04d}.png")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur CIFAR-10: {e}")
        return False

def download_cats_vs_dogs_auto(project_data_dir: Path):
    """Télécharge Cats vs Dogs automatiquement"""
    try:
        st.info("📥 Téléchargement Cats vs Dogs...")
        
        # Télécharger depuis TensorFlow Datasets
        import tensorflow_datasets as tfds
        
        # Charger le dataset
        dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
        
        train_data = dataset['train']
        
        # Créer les dossiers
        cats_dir = project_data_dir / "cats_vs_dogs" / "cats"
        dogs_dir = project_data_dir / "cats_vs_dogs" / "dogs"
        cats_dir.mkdir(parents=True, exist_ok=True)
        dogs_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les images
        count = 0
        for image, label in train_data.take(2000):  # 2000 images max
            if count >= 2000:
                break
                
            image_np = image.numpy()
            label_np = label.numpy()
            
            if label_np == 0:  # Cat
                img_path = cats_dir / f"cat_{count:04d}.jpg"
            else:  # Dog
                img_path = dogs_dir / f"dog_{count:04d}.jpg"
            
            Image.fromarray(image_np).save(img_path)
            count += 1
        
        return True
        
    except Exception as e:
        st.warning(f"⚠️ Cats vs Dogs nécessite tensorflow-datasets. Utilisation de CIFAR-10. Error: {e}")
        return False

def download_mnist_auto(project_data_dir: Path):
    """Télécharge MNIST automatiquement"""
    try:
        st.info("📥 Téléchargement MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # MNIST est en niveaux de gris, convertir en RGB pour la cohérence
        for subset, X, Y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            for i in range(min(1000, len(X))):  # 1000 images par classe max
                lab = str(int(Y[i]))
                class_dir = project_data_dir / "mnist" / lab
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Convertir en PIL Image et sauvegarder
                img = Image.fromarray(X[i]).convert('L')
                img = img.resize((64, 64))  # Redimensionner pour cohérence
                img.save(class_dir / f"{lab}_{i:04d}.png")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur MNIST: {e}")
        return False

def download_fashion_mnist_auto(project_data_dir: Path):
    """Télécharge Fashion MNIST automatiquement"""
    try:
        st.info("📥 Téléchargement Fashion MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        fashion_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        for subset, X, Y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            for i in range(min(800, len(X))):  # 800 images par classe max
                lab = fashion_labels[int(Y[i])]
                class_dir = project_data_dir / "fashion_mnist" / lab
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Convertir et sauvegarder
                img = Image.fromarray(X[i]).convert('L')
                img = img.resize((64, 64))
                img.save(class_dir / f"{lab}_{i:04d}.png")
        
        return True
    except Exception as e:
        st.error(f"❌ Erreur Fashion MNIST: {e}")
        return False

def auto_download_all_datasets(project_data_dir: Path):
    """Télécharge automatiquement tous les datasets"""
    st.info("🚀 **DÉMARRAGE DU TÉLÉCHARGEMENT AUTOMATIQUE**")
    
    datasets_status = {}
    
    # Essayer différents datasets dans l'ordre
    datasets_to_try = [
        ("CIFAR-10", download_cifar10_auto),
        ("MNIST", download_mnist_auto), 
        ("Fashion MNIST", download_fashion_mnist_auto),
        ("Cats vs Dogs", download_cats_vs_dogs_auto)
    ]
    
    for dataset_name, download_func in datasets_to_try:
        with st.spinner(f"Téléchargement {dataset_name}..."):
            success = download_func(project_data_dir)
            datasets_status[dataset_name] = success
            if success:
                st.success(f"✅ {dataset_name} - Terminé!")
            else:
                st.warning(f"⚠️ {dataset_name} - Échec")
    
    return datasets_status

# ---------------------------
# Initialisation des variables de session
# ---------------------------
if "project_name" not in st.session_state:
    st.session_state.project_name = DEFAULT_PROJECT_NAME
if "target_size" not in st.session_state:
    st.session_state.target_size = DEFAULT_TARGET_SIZE
if "gray_mode" not in st.session_state:
    st.session_state.gray_mode = DEFAULT_GRAY_MODE
if "test_size" not in st.session_state:
    st.session_state.test_size = DEFAULT_TEST_SIZE

# ---------------------------
# Main UI
# ---------------------------
st.title("🤖 AUTO ML/DL - Téléchargement Automatique")
st.markdown("### 🎯 **JE TÉLÉCHARGE TOUT AUTOMATIQUEMENT !**")

# Paramètres dans la sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    project_name = st.text_input("Nom du projet", value=DEFAULT_PROJECT_NAME)
    target_size = st.selectbox("Taille des images", [(64, 64), (128, 128), (224, 224)], index=0)
    gray_mode = st.checkbox("Mode niveaux de gris", value=True)
    test_size = st.slider("Pourcentage de test (%)", 10, 40, 20)

# Mettre à jour les variables de session
st.session_state.project_name = project_name
st.session_state.target_size = target_size
st.session_state.gray_mode = gray_mode
st.session_state.test_size = test_size

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 AUTO DATA", "🔧 Preprocess", "🎯 Train", "📊 Evaluate", "🚀 Deploy"])

# ---------------------------
# TAB 1: AUTO DATA - Je télécharge tout !
# ---------------------------
with tab1:
    st.header("🚀 **TÉLÉCHARGEMENT AUTOMATIQUE**")
    st.markdown("Je vais télécharger **CIFAR10, MNIST, Fashion MNIST, Cats vs Dogs** automatiquement !")
    
    project_data_dir = DATA_DIR / project_name
    
    # Bouton de téléchargement automatique
    if st.button("🎯 CLIQUEZ ICI - JE TÉLÉCHARGE TOUT !", type="primary", use_container_width=True):
        with st.spinner("🔄 Téléchargement en cours... Cela peut prendre quelques minutes"):
            # Nettoyer le dossier existant
            if project_data_dir.exists():
                shutil.rmtree(project_data_dir)
            project_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Télécharger tous les datasets
            status = auto_download_all_datasets(project_data_dir)
            
            # Afficher le résumé
            st.success("🎉 **TÉLÉCHARGEMENT TERMINÉ !**")
            
            # Statistiques
            total_images = 0
            for dataset in ["cifar10", "mnist", "fashion_mnist", "cats_vs_dogs"]:
                dataset_path = project_data_dir / dataset
                if dataset_path.exists():
                    images = find_image_files(dataset_path)
                    total_images += len(images)
                    st.info(f"📊 {dataset}: {len(images)} images")
            
            st.metric("📈 Total Images Téléchargées", total_images)
    
    # Afficher le contenu téléchargé
    if project_data_dir.exists():
        st.subheader("📁 Contenu Téléchargé")
        
        datasets = ["cifar10", "mnist", "fashion_mnist", "cats_vs_dogs"]
        for dataset in datasets:
            dataset_path = project_data_dir / dataset
            if dataset_path.exists():
                images = find_image_files(dataset_path)
                classes = set()
                for img in images:
                    cls = get_class_from_path(img, dataset_path)
                    classes.add(cls)
                
                st.success(f"**{dataset.upper()}**: {len(images)} images, {len(classes)} classes")
                
                # Aperçu
                st.write(f"Aperçu {dataset}:")
                cols = st.columns(4)
                for i, img_file in enumerate(images[:4]):
                    with cols[i % 4]:
                        try:
                            img = Image.open(img_file).resize((80, 80))
                            st.image(img, caption=f"{get_class_from_path(img_file, dataset_path)}")
                        except:
                            pass

# ---------------------------
# TAB 2: Preprocess - Automatique
# ---------------------------
with tab2:
    st.header("2. 🔧 Prétraitement Automatique")
    
    project_data_dir = DATA_DIR / project_name
    
    if not project_data_dir.exists():
        st.warning("⚠️ Cliquez d'abord sur 'JE TÉLÉCHARGE TOUT' dans l'onglet DATA")
    else:
        # Sélection du dataset
        datasets_available = []
        for dataset in ["cifar10", "mnist", "fashion_mnist", "cats_vs_dogs"]:
            if (project_data_dir / dataset).exists():
                datasets_available.append(dataset)
        
        selected_dataset = st.selectbox("📂 Choisissez le dataset à utiliser:", datasets_available)
        
        if selected_dataset:
            dataset_path = project_data_dir / selected_dataset
            files = find_image_files(dataset_path)
            
            if not files:
                st.error("❌ Aucune image trouvée dans ce dataset!")
            else:
                # Détection automatique des classes
                class_names = set()
                for file_path in files:
                    class_name = get_class_from_path(file_path, dataset_path)
                    class_names.add(class_name)
                
                class_names = sorted(list(class_names))
                st.write(f"**🏷️ Classes détectées:** {class_names}")
                st.write(f"**📊 Total images:** {len(files)}")
                
                # Bouton de prétraitement automatique
                if st.button("🔄 Prétraiter le Dataset", type="primary"):
                    with st.spinner("🔄 Prétraitement en cours..."):
                        X, y = [], []
                        label_map = {}
                        
                        # Créer le mapping des labels
                        for i, class_name in enumerate(sorted(class_names)):
                            label_map[class_name] = i
                        
                        # Charger les images
                        progress_bar = st.progress(0)
                        successful_loads = 0
                        
                        for idx, file_path in enumerate(files):
                            class_name = get_class_from_path(file_path, dataset_path)
                            arr = load_image(file_path, target_size=target_size, gray=gray_mode)
                            
                            if arr is not None:
                                X.append(arr)
                                y.append(label_map[class_name])
                                successful_loads += 1
                            
                            if idx % 50 == 0:  # Mettre à jour moins fréquemment pour plus de rapidité
                                progress_bar.progress(min((idx + 1) / len(files), 1.0))
                        
                        if successful_loads > 0:
                            X = np.array(X)
                            y = np.array(y)
                            
                            st.session_state["X"] = X
                            st.session_state["y"] = y
                            st.session_state["label_map"] = label_map
                            st.session_state["dataset_name"] = selected_dataset
                            st.session_state["class_names"] = class_names
                            
                            st.success(f"✅ Prétraitement terminé!")
                            st.write(f"📐 **X shape:** {X.shape}")
                            st.write(f"🎯 **y shape:** {y.shape}")
                            st.write(f"🏷️ **Classes:** {len(class_names)}")
                            
                            # Aperçu des données
                            st.subheader("👀 Aperçu des données prétraitées")
                            cols = st.columns(5)
                            for i in range(min(5, len(X))):
                                with cols[i]:
                                    if gray_mode:
                                        img_display = (X[i].squeeze() * 255).astype(np.uint8)
                                    else:
                                        img_display = (X[i] * 255).astype(np.uint8)
                                    st.image(img_display, caption=f"Class: {class_names[y[i]]}")
                        else:
                            st.error("❌ Aucune image n'a pu être chargée!")

# ---------------------------
# TAB 3: Train - Automatique
# ---------------------------
with tab3:
    st.header("3. 🎯 Entraînement Automatique")
    
    if "X" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord prétraiter un dataset dans l'onglet précédent")
    else:
        X = st.session_state["X"]
        y = st.session_state["y"]
        label_map = st.session_state["label_map"]
        dataset_name = st.session_state.get("dataset_name", "unknown")
        class_names = st.session_state.get("class_names", list(label_map.keys()))
        n_classes = len(label_map)
        
        st.write(f"**📊 Dataset:** {dataset_name}")
        st.write(f"**📐 Shape:** {X.shape[0]} images, {X.shape[1:]}")
        st.write(f"**🏷️ Classes:** {n_classes}")
        st.write(f"**📋 Labels:** {label_map}")
        
        # Sélection rapide du modèle
        col1, col2 = st.columns(2)
        with col1:
            algorithm_type = st.selectbox("Méthode", ["Classical ML", "Deep Learning"])
        with col2:
            if algorithm_type == "Classical ML":
                classifier_name = st.selectbox("Algorithme", ["RandomForest", "LogisticRegression", "SVM", "KNN"])
            else:
                classifier_name = st.selectbox("Architecture", ["CNN_simple", "MLP", "CNN_vgg_like"])
        
        if st.button("🚀 Lancer l'Entraînement Auto", type="primary"):
            with st.spinner("🎯 Entraînement en cours..."):
                if algorithm_type == "Classical ML":
                    # Préparation des features
                    X_feats = X.reshape((X.shape[0], -1))
                    
                    # Split train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_feats, y, test_size=test_size/100.0, stratify=y, random_state=42
                    )
                    
                    st.write(f"📈 Données d'entraînement: {X_train.shape[0]} samples")
                    
                    # Sélection et entraînement du modèle
                    if classifier_name == "RandomForest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif classifier_name == "LogisticRegression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif classifier_name == "SVM":
                        model = SVC(probability=True, random_state=42)
                    else:  # KNN
                        model = KNeighborsClassifier(n_neighbors=5)
                    
                    model.fit(X_train, y_train)
                    
                    # Évaluation
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    accuracy = np.mean(y_pred == y_test)
                    
                    # Rapport de classification détaillé
                    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                    
                    st.success(f"✅ **Modèle ML entraîné ! Accuracy = {accuracy:.2%}**")
                    
                    # Affichage du rapport de classification
                    st.subheader("📊 Rapport de Classification Détaillé")
                    
                    # Créer un DataFrame pour un affichage plus propre
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Afficher le tableau des métriques
                    st.dataframe(report_df.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}', 
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}'
                    }).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))
                    
                    # Matrice de confusion avec seaborn pour un meilleur visuel
                    st.subheader("🎯 Matrice de Confusion")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, 
                               yticklabels=class_names,
                               ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vraies valeurs')
                    ax.set_title('Matrice de Confusion')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                    
                    # Sauvegarde
                    model_path = MODELS_DIR / f"{dataset_name}_{classifier_name}.pkl"
                    joblib.dump({"model": model, "label_map": label_map}, model_path)
                    st.session_state["trained_model"] = {"type": "sklearn", "path": str(model_path)}
                    st.session_state["test_data"] = {"X_test": X_test, "y_test": y_test, "y_pred": y_pred}
                    st.info(f"💾 Modèle sauvegardé: {model_path}")
                
                else:  # Deep Learning
                    # Préparation des données
                    X_dl = X if not gray_mode else np.concatenate([X, X, X], axis=-1) if X.shape[-1]==1 else X
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_dl, y, test_size=test_size/100.0, stratify=y, random_state=42
                    )
                    
                    # Construction du modèle
                    if classifier_name == "CNN_simple":
                        model = models.Sequential([
                            layers.Input(shape=X_train.shape[1:]),
                            layers.Conv2D(32, 3, activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(64, 3, activation='relu'),
                            layers.MaxPooling2D(),
                            layers.Flatten(),
                            layers.Dense(128, activation='relu'),
                            layers.Dropout(0.5),
                            layers.Dense(n_classes, activation='softmax')
                        ])
                    elif classifier_name == "CNN_vgg_like":
                        model = models.Sequential([
                            layers.Input(shape=X_train.shape[1:]),
                            layers.Conv2D(32, 3, activation='relu', padding='same'),
                            layers.Conv2D(32, 3, activation='relu', padding='same'),
                            layers.MaxPooling2D(),
                            layers.Conv2D(64, 3, activation='relu', padding='same'),
                            layers.Conv2D(64, 3, activation='relu', padding='same'),
                            layers.MaxPooling2D(),
                            layers.Flatten(),
                            layers.Dense(256, activation='relu'),
                            layers.Dropout(0.5),
                            layers.Dense(n_classes, activation='softmax')
                        ])
                    else:  # MLP
                        model = models.Sequential([
                            layers.Flatten(input_shape=X_train.shape[1:]),
                            layers.Dense(512, activation='relu'),
                            layers.Dropout(0.3),
                            layers.Dense(256, activation='relu'),
                            layers.Dropout(0.3),
                            layers.Dense(n_classes, activation='softmax')
                        ])
                    
                    model.compile(optimizer='adam', 
                                loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy'])
                    
                    # Entraînement
                    history = model.fit(X_train, y_train, 
                                      validation_data=(X_test, y_test),
                                      epochs=15, 
                                      batch_size=32, 
                                      verbose=1)
                    
                    # Évaluation
                    y_pred_proba = model.predict(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    
                    # Rapport de classification pour DL
                    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                    
                    st.success(f"✅ **Modèle DL entraîné ! Accuracy = {accuracy:.2%}**")
                    
                    # Affichage du rapport de classification
                    st.subheader("📊 Rapport de Classification Détaillé")
                    
                    # Créer un DataFrame pour un affichage plus propre
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Afficher le tableau des métriques
                    st.dataframe(report_df.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}', 
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}'
                    }).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']))
                    
                    # Matrice de confusion
                    st.subheader("🎯 Matrice de Confusion")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=class_names, 
                               yticklabels=class_names,
                               ax=ax)
                    ax.set_xlabel('Prédictions')
                    ax.set_ylabel('Vraies valeurs')
                    ax.set_title('Matrice de Confusion')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                    
                    # Graphiques d'entraînement
                    st.subheader("📈 Courbes d'Apprentissage")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(history.history['loss'], label='Train Loss')
                    ax1.plot(history.history['val_loss'], label='Val Loss')
                    ax1.set_title('Loss')
                    ax1.legend()
                    
                    ax2.plot(history.history['accuracy'], label='Train Accuracy')
                    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
                    ax2.set_title('Accuracy')
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                    # Sauvegarde
                    model_path = MODELS_DIR / f"{dataset_name}_{classifier_name}.h5"
                    model.save(model_path)
                    st.session_state["trained_model"] = {"type": "keras", "path": str(model_path)}
                    st.session_state["history"] = history.history
                    st.session_state["test_data"] = {"X_test": X_test, "y_test": y_test, "y_pred": y_pred}
                    st.info(f"💾 Modèle DL sauvegardé: {model_path}")

# ---------------------------
# TAB 4: Evaluate
# ---------------------------
with tab4:
    st.header("4. 📊 Évaluation Détaillée du Modèle")
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Aucun modèle entraîné trouvé")
    else:
        model_info = st.session_state["trained_model"]
        st.success(f"✅ Modèle chargé: {model_info['path']}")
        
        # Afficher les métriques détaillées si disponibles
        if "test_data" in st.session_state:
            test_data = st.session_state["test_data"]
            y_test = test_data["y_test"]
            y_pred = test_data["y_pred"]
            class_names = st.session_state.get("class_names", [])
            
            # Calculer l'accuracy
            accuracy = np.mean(y_pred == y_test)
            
            # Afficher le message principal
            st.success(f"✅ **Modèle ML entraîné ! Accuracy = {accuracy:.2%}**")
            
            # Rapport de classification complet
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            
            # Afficher dans un format plus lisible
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Métriques par Classe")
                metrics_df = pd.DataFrame(report).transpose()
                
                # Nettoyer l'affichage
                display_df = metrics_df[['precision', 'recall', 'f1-score', 'support']].round(3)
                st.dataframe(display_df.style.background_gradient(cmap='Blues'))
            
            with col2:
                st.subheader("📊 Résumé Global")
                
                # Métriques globales
                accuracy = report['accuracy']
                macro_avg = report['macro avg']
                weighted_avg = report['weighted avg']
                
                st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                st.metric("📊 Precision (macro)", f"{macro_avg['precision']:.3f}")
                st.metric("🔍 Recall (macro)", f"{macro_avg['recall']:.3f}")
                st.metric("⚖️ F1-Score (macro)", f"{macro_avg['f1-score']:.3f}")
        
        # Afficher les courbes d'apprentissage pour les modèles DL
        if model_info["type"] == "keras" and "history" in st.session_state:
            history = st.session_state["history"]
            
            st.subheader("📈 Courbes d'Apprentissage")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(history['loss'], label='Train Loss')
            ax1.plot(history['val_loss'], label='Val Loss')
            ax1.set_title('Loss')
            ax1.legend()
            
            ax2.plot(history['accuracy'], label='Train Accuracy')
            ax2.plot(history['val_accuracy'], label='Val Accuracy')
            ax2.set_title('Accuracy')
            ax2.legend()
            
            st.pyplot(fig)

# ---------------------------
# TAB 5: Deploy
# ---------------------------
with tab5:
    st.header("5. 🚀 Testez Votre Modèle!")
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Entraînez d'abord un modèle dans l'onglet Train")
    else:
        model_info = st.session_state["trained_model"]
        label_map = st.session_state.get("label_map", {})
        class_names = st.session_state.get("class_names", list(label_map.keys()))
        
        st.subheader("🎯 Faites une Prédiction")
        
        uploaded_file = st.file_uploader("📤 Uploader une image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Afficher l'image
            img = Image.open(uploaded_file)
            st.image(img, caption="📷 Image uploadée", width=300)
            
            # Prétraiter l'image
            img_resized = img.resize(target_size)
            if gray_mode:
                img_resized = img_resized.convert("L")
                img_array = np.array(img_resized).reshape((1,) + target_size + (1,))
                # Pour les modèles DL qui attendent 3 canaux
                if model_info["type"] == "keras":
                    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
            else:
                img_array = np.array(img_resized).reshape((1,) + target_size + (3,))
            
            img_array = img_array.astype("float32") / 255.0
            
            # Faire la prédiction
            if model_info["type"] == "sklearn":
                model_data = joblib.load(model_info["path"])
                model = model_data["model"]
                img_flat = img_array.reshape(1, -1)
                prediction = model.predict(img_flat)[0]
                probabilities = model.predict_proba(img_flat)[0]
            else:  # Keras
                model = keras.models.load_model(model_info["path"])
                probabilities = model.predict(img_array)[0]
                prediction = np.argmax(probabilities)
            
            # Afficher le résultat
            label_rev = {v: k for k, v in label_map.items()}
            predicted_class = label_rev.get(prediction, f"Class {prediction}")
            confidence = probabilities[prediction] if model_info["type"] == "sklearn" else probabilities[prediction]
            
            st.success(f"**🎯 Prédiction:** {predicted_class}")
            st.info(f"**📊 Confiance:** {confidence:.3f}")
            
            # Afficher toutes les probabilités sous forme de graphique
            st.subheader("📈 Probabilités par Classe")
            
            # Créer un graphique en barres des probabilités
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(class_names))
            ax.barh(y_pos, probabilities, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names)
            ax.set_xlabel('Probabilité')
            ax.set_title('Distribution des Probabilités par Classe')
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(probabilities):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            st.pyplot(fig)
            
            # Afficher aussi sous forme de tableau
            prob_df = pd.DataFrame({
                'Classe': class_names,
                'Probabilité': probabilities
            }).sort_values('Probabilité', ascending=False)
            
            st.dataframe(prob_df.style.format({'Probabilité': '{:.3f}'}))

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("### 🎉 **SYSTÈME AUTOMATIQUE ACTIVÉ !**")
st.markdown("""
**Fonctionnalités Automatiques:**
- ✅ **Téléchargement auto** de CIFAR10, MNIST, Fashion MNIST, Cats vs Dogs
- ✅ **Prétraitement auto** des images
- ✅ **Entraînement auto** avec différents modèles
- ✅ **Évaluation auto** des performances
- ✅ **Interface simple** - juste cliquer sur les boutons!

**Comment utiliser:**
1. 📥 **Onglet DATA** - Cliquez sur "JE TÉLÉCHARGE TOUT"
2. 🔧 **Onglet Preprocess** - Choisissez dataset et cliquez "Prétraiter"
3. 🎯 **Onglet Train** - Choisissez modèle et cliquez "Entraîner"
4. 🚀 **Onglet Deploy** - Testez avec vos images!
""")

st.sidebar.markdown("---")
st.sidebar.success("""
**🤖 MODE AUTOMATIQUE**
Je télécharge tout:
- CIFAR-10 ✅
- MNIST ✅  
- Fashion MNIST ✅
- Cats vs Dogs ✅
Juste exécutez et suivez les onglets!
""")