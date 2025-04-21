import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from PIL import Image
import torch
from ultralytics import YOLO
import yaml

# === Page config ===
st.set_page_config(page_title="Dashboard CIFAR-10 & YOLOv11-cls", page_icon="🧠", layout="wide")

# === Classes CIFAR-10 ===
cifar10_classes = ["Avion", "Automobile", "Oiseau", "Chat", "Cerf", 
                   "Chien", "Grenouille", "Cheval", "Bateau", "Camion"]

# === Onglets ===
tabs = st.tabs(["📊 Analyse Exploratoire", "🤖 Prédiction YOLOv11-cls"])

# =============================
# Onglet 1 : Analyse Exploratoire
# =============================
with tabs[0]:
    st.title("📸 Analyse Exploratoire du Dataset CIFAR-10")

    # Chargement des données
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Affichage infos générales
    st.header("📁 Informations sur le Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Images d'entraînement", x_train.shape[0])
        st.metric("Dimensions", f"{x_train.shape[1]}x{x_train.shape[2]}")
    with col2:
        st.metric("Images de test", x_test.shape[0])
        st.metric("Nombre de classes", len(cifar10_classes))

    # Exemples d’images : 1 par classe
    st.header("🔍 Exemples d’images (une par classe)")
    st.markdown("Les images ci-dessous illustrent un exemple typique pour chaque classe du dataset CIFAR-10.")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    shown_classes = set()
    i = 0
    for idx, (img, label) in enumerate(zip(x_train, y_train.flatten())):
        class_id = int(label)
        if class_id not in shown_classes:
            ax = axes.flat[i]
            ax.imshow(img)
            ax.set_title(f"{cifar10_classes[class_id]}", fontsize=10)
            ax.axis("off")
            shown_classes.add(class_id)
            i += 1
        if i >= 10:
            break
    st.pyplot(fig)
    st.caption("Figure : un exemple visuel pour chaque classe CIFAR-10 (images informatives).")

    # Distribution des classes
    st.header("📊 Distribution des classes")
    st.markdown("Le graphique ci-dessous montre combien d’images sont présentes pour chaque classe dans l'ensemble d'entraînement.")
    class_counts = np.bincount(y_train.flatten())
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(cifar10_classes, class_counts, color="#1b1f23")  # contraste renforcé
    ax2.set_title("Répartition des classes dans les données d'entraînement", fontsize=14)
    ax2.set_xlabel("Classe", fontsize=12)
    ax2.set_ylabel("Nombre d’images", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    st.caption("Figure : histogramme de la fréquence des classes.")

# =============================
# Onglet 2 : Prédiction avec YOLOv11-cls
# =============================
with tabs[1]:
    st.title("🤖 Prédiction d’image avec YOLOv11n-cls")

    # Chargement des classes depuis data.yaml
    try:
        with open("C:/Users/jouvi/OCR - IML/P7 - Developpez une preuve de concept/cifar10.yaml", "r") as f:
            dataset_yaml = yaml.safe_load(f)
            class_names = dataset_yaml["names"]
    except Exception as e:
        st.error("Erreur lors du chargement de 'data.yaml'.")
        st.text(str(e))
        st.stop()

    # Chargement du modèle
    try:
        model = YOLO("C:/Users/jouvi/OCR - IML/P7 - Developpez une preuve de concept/models/best_overall.pt")
    except Exception as e:
        st.error("Erreur lors du chargement du modèle.")
        st.text(str(e))
        st.stop()

    # Upload de l’image
    st.header("📥 Importer une image à classer")
    st.markdown(
        "Veuillez charger une image **au format JPEG ou PNG**. "
        "L'image doit représenter un objet ou un animal reconnaissable."
    )
    uploaded_file = st.file_uploader(
        "📸 Sélectionnez une image à prédire :", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Image utilisateur chargée – utilisée pour la prédiction.", width=200)

            # Inference
            with st.spinner("🔎 Prédiction en cours..."):
                results = model.predict(image, imgsz=32, conf=0.25)
                class_id = int(results[0].probs.top1)
                confidence = float(results[0].probs.top1conf)
                class_name = class_names[class_id]

            # Affichage du résultat
            st.success(f"✅ Classe prédite : **{class_name}**")
            st.caption(f"Taux de confiance du modèle : **{confidence:.2%}**")

        except Exception as e:
            st.error("Une erreur est survenue pendant la prédiction.")
            st.text(str(e))