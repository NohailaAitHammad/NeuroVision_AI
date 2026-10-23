
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import json
from tensorflow import keras


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
      page_title="NeuroVision AI",
      page_icon="🧠",
      layout="wide",
      initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/best_model.keras")
IMAGE_SIZE = (224, 224)

# IMPORTANT : cet ordre doit correspondre à ton entraînement.
CLASS_NAMES = [
      "Glioma",
      "Méningioma",
      "Non Tumor",
      "Pituitary",
]

CLASS_COLORS = {
      "Glioma": "#D97757",
      "Méningioma": "#5487D9",
      "Non Tumor": "#119B88",
      "Pituitary": "#8A70D6",
}

# =========================================================
# 2. STYLE PROFESSIONNEL
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
      background: #F4F7FB;
      color: #172B43;
      font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
      background: #10253F;
}

[data-testid="stSidebar"] * {
      color: #F0F5FC;
}

.main .block-container {
      max-width: 1350px;
      padding-top: 2rem;
      padding-bottom: 3rem;
}

.hero {
      background: linear-gradient(120deg, #10253F, #174765, #087E80);
      padding: 2.2rem;
      border-radius: 22px;
      color: white;
      margin-bottom: 1.5rem;
}

.hero h1 {
      color: white;
      font-size: 2.5rem;
      margin-bottom: .5rem;
}

.hero p {
      color: #DCEAF5;
      font-size: 1rem;
}

.eyebrow {
      color: #71E1D2;
      font-weight: 700;
      letter-spacing: 2px;
      font-size: .75rem;
}

.metric {
      background: white;
      border: 1px solid #E3EAF2;
      padding: 1.2rem;
      border-radius: 16px;
      min-height: 110px;
}

.metric-label {
      color: #718096;
      font-size: .85rem;
}

.metric-value {
      color: #10253F;
      font-size: 1.65rem;
      font-weight: 700;
      margin-top: .4rem;
}

div.stButton > button {
      background: #0D9488;
      color: white;
      border: none;
      border-radius: 10px;
      font-weight: 700;
      min-height: 45px;
}

div.stButton > button:hover {
      background: #08786F;
      color: white;
}

.notice {
      background: #FFF6DF;
      color: #77571B;
      padding: 1rem;
      border-radius: 12px;
      border: 1px solid #F1E1B7;
}

</style>
""", unsafe_allow_html=True)


def show_hero(tag, title, description):
      st.markdown(
            f"""
            <div class="hero">
                  <div class="eyebrow">{tag}</div>
                  <h1>{title}</h1>
                  <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
      )


def show_metric(label, value, detail):
      st.markdown(
            f"""
            <div class="metric">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-label">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
      )


# =========================================================
# 3. CHARGEMENT DU MODÈLE
# =========================================================

@st.cache_resource
def load_model():
      return keras.models.load_model(
            str(MODEL_PATH),
            compile=False,
      )


# =========================================================
# 4. PRÉTRAITEMENT ET PRÉDICTION
# =========================================================

def preprocess_image(image):
      """
      Cette fonction doit reproduire le prétraitement
      utilisé pendant l'entraînement du CNN.
      """
      image = image.convert("RGB")
      image = image.resize(IMAGE_SIZE)

      array = np.asarray(image, dtype=np.float32)
      array = array / 255.0

      return np.expand_dims(array, axis=0)


def predict_image(image):
      if not MODEL_PATH.exists():
            raise FileNotFoundError(
                  f"Modèle introuvable : {MODEL_PATH}"
            )

      model = load_model()
      batch = preprocess_image(image)

      output = np.asarray(
            model.predict(batch, verbose=0)
      ).reshape(-1)

      if len(output) != len(CLASS_NAMES):
            raise ValueError(
                  "Le modèle ne retourne pas quatre sorties. "
                  "Vérifie son architecture et CLASS_NAMES."
            )

      # Si la sortie est déjà une distribution softmax,
      # on la conserve. Sinon, on applique softmax aux logits.
      if (
            np.all(output >= 0)
            and np.all(output <= 1)
            and np.isclose(output.sum(), 1.0, atol=0.02)
      ):
            probabilities = output
      else:
            shifted = output - np.max(output)
            exp_values = np.exp(shifted)
            probabilities = exp_values / exp_values.sum()

      best_index = int(np.argmax(probabilities))

      return CLASS_NAMES[best_index], probabilities


# =========================================================
# 5. INITIALISATION DE L'HISTORIQUE
# =========================================================

if "history" not in st.session_state:
      st.session_state.history = []


# =========================================================
# 6. MENU DE NAVIGATION
# =========================================================

with st.sidebar:
      st.markdown("# 🧠 NeuroVision AI")
      st.caption("Deep Learning · Computer Vision")
      st.divider()

      page = st.radio(
            "NAVIGATION",
            [
                  "Tableau de bord",
                  "Analyser une IRM",
                  "Historique",
                  "Performance du modèle",
                  "À propos",
            ],
      )

      st.divider()
      st.caption("Prototype pédagogique")
      st.caption(
            "Classification expérimentale d'images IRM."
      )

      if MODEL_PATH.exists():
            st.success("Fichier modèle détecté")
      else:
            st.warning("Fichier modèle absent")


# =========================================================
# 7. TABLEAU DE BORD
# =========================================================

if page == "Tableau de bord":

      show_hero(
            "INTELLIGENCE ARTIFICIELLE · IMAGERIE MÉDICALE",
            "NeuroVision AI",
            "Une application interactive de classification "
            "d'images IRM cérébrales basée sur un réseau "
            "de neurones convolutif.",
      )

      c1, c2, c3, c4 = st.columns(4)

      with c1:
            show_metric("Architecture", "CNN", "Deep Learning")

      with c2:
            show_metric("Classes", "04", "Catégories configurées")

      with c3:
            show_metric("Entrée", "224 × 224", "Taille configurée")

      with c4:
            show_metric("Framework", "Keras", "TensorFlow")

      st.write("")
      left, right = st.columns([1.2, 0.8], gap="large")

      with left:
            st.subheader("Comment fonctionne l'application ?")

            st.markdown("""
            **01 — Importer**

            Sélectionner une image IRM cérébrale.

            **02 — Prétraiter**

            Convertir l'image en RGB, redimensionner et normaliser.

            **03 — Prédire**

            Transmettre l'image au CNN entraîné.

            **04 — Interpréter**

            Visualiser la classe prédite et les scores retournés.
            """)

      with right:
            st.subheader("Catégories configurées")

            for name in CLASS_NAMES:
                  st.markdown(f"- {name}")

            st.info(
                  "Les résultats dépendent du modèle entraîné, "
                  "de ses données et de son prétraitement."
            )

      st.markdown(
            """
            <div class="notice">
            <b>Limite importante :</b> cette application est un
            prototype pédagogique. Elle ne remplace pas un
            radiologue et ne permet pas d'établir un diagnostic.
            </div>
            """,
            unsafe_allow_html=True,
      )


# =========================================================
# 8. ANALYSE D'UNE IMAGE IRM
# =========================================================

elif page == "Analyser une IRM":

      show_hero(
            "ANALYSE INTERACTIVE",
            "Analyser une image",
            "Importe une image pour explorer la sortie du modèle CNN.",
      )

      uploaded_file = st.file_uploader(
            "Importer une image IRM",
            type=["png", "jpg", "jpeg", "bmp"],
      )

      if uploaded_file is not None:

            try:
                  image = Image.open(uploaded_file).convert("RGB")

                  left, right = st.columns(2, gap="large")

                  with left:
                        st.subheader("Image importée")
                        st.image(
                              image,
                              use_container_width=True,
                        )
                        st.caption(
                              f"Dimensions : {image.width} × {image.height}"
                        )

                  with right:
                        st.subheader("Analyse du modèle")
                        st.write(
                              "L'image sera prétraitée avant son passage "
                              "dans le réseau de neurones."
                        )

                  if st.button(
                        "Lancer la prédiction",
                        type="primary",
                        use_container_width=True,
                  ):
                        with st.spinner("Analyse en cours..."):
                              try:
                                    predicted_class, probabilities = (
                                          predict_image(image)
                                    )

                                    result = {
                                          "date": datetime.now().strftime(
                                                "%Y-%m-%d %H:%M:%S"
                                          ),
                                          "filename": uploaded_file.name,
                                          "prediction": predicted_class,
                                          "confidence": float(
                                                np.max(probabilities)
                                          ),
                                          "scores": {
                                                CLASS_NAMES[i]: float(
                                                probabilities[i]
                                                )
                                                for i in range(len(CLASS_NAMES))
                                          },
                                    }

                                    st.session_state.last_result = result
                                    st.session_state.history.insert(
                                          0, result.copy()
                                    )

                              except Exception as error:
                                    st.error(str(error))

                  result = st.session_state.get("last_result")

                  if (
                  result is not None
                  and result["filename"] == uploaded_file.name
                  ):
                        st.divider()
                        st.subheader("Résultat de la prédiction")

                        a, b = st.columns([0.8, 1.2], gap="large")

                        with a:
                              st.markdown("### Classe la plus probable")
                              st.success(result["prediction"])

                              st.metric(
                                    "Score de sortie maximal",
                                    f"{result['confidence']:.1%}",
                              )

                              st.caption(
                                    "Ce score n'est pas une probabilité "
                                    "clinique validée."
                              )

                        with b:
                              st.markdown("### Scores par catégorie")

                              scores_df = pd.DataFrame({
                                    "Classe": list(result["scores"].keys()),
                                    "Score": list(result["scores"].values()),
                              })

                              st.bar_chart(
                                    scores_df.set_index("Classe"),
                                    horizontal=True,
                              )

                        st.caption(
                              f"Date de l'analyse : {result['date']}"
                        )

            except Exception:
                        st.error(
                        "Impossible de lire cette image. "
                        "Essaie un autre fichier."
                        )

      else:
            st.info(
                  "Commence par importer une image PNG, JPG, JPEG ou BMP."
            )


# =========================================================
# 9. HISTORIQUE
# =========================================================

elif page == "Historique":

      show_hero(
            "SUIVI DES ANALYSES",
            "Historique",
            "Consulte les analyses réalisées pendant cette session.",
      )

      if not st.session_state.history:
            st.info("Aucune analyse effectuée pour le moment.")

      else:
            rows = []

            for item in st.session_state.history:
                  rows.append({
                  "Date": item["date"],
                  "Image": item["filename"],
                  "Classe prédite": item["prediction"],
                  "Score maximal": f"{item['confidence']:.1%}",
                  })

            history_df = pd.DataFrame(rows)

            st.dataframe(
                  history_df,
                  use_container_width=True,
                  hide_index=True,
            )

            st.download_button(
                  "Exporter l'historique CSV",
                  data=history_df.to_csv(
                  index=False
                  ).encode("utf-8-sig"),
                  file_name="neurovision_history.csv",
                  mime="text/csv",
            )

            if st.button("Effacer l'historique"):
                  st.session_state.history = []
                  st.rerun()


# =========================================================
# 10. PERFORMANCE DU MODÈLE
# =========================================================

elif page == "Performance du modèle":

      show_hero(
            "ÉVALUATION",
            "Performance du CNN",
            "Présente les résultats mesurés sur ton jeu de test.",
      )

      metrics_path = Path("metrics.json")

      if metrics_path.exists():


            try:
                  metrics = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                  )

                  metric_labels = {
                        "accuracy": "Accuracy",
                        "precision": "Précision macro",
                        "recall": "Recall macro",
                        "f1_score": "F1-score macro",
                  }

                  curves = metrics.get("learning_curves", {})

                  cols = st.columns(4)

                  for index, (key, label) in enumerate(
                        metric_labels.items()
                        ):
                        if key in metrics:
                              value = float(metrics[key])

                              with cols[index]:
                                    show_metric(
                                    label,
                                    f"{value:.1%}",
                                    "Résultat fourni",
                              )

                  if "confusion_matrix" in metrics:
                        st.subheader("Matrice de confusion")

                        matrix = pd.DataFrame(
                              metrics["confusion_matrix"],
                              index=CLASS_NAMES,
                              columns=CLASS_NAMES,
                        )

                        st.dataframe(
                              matrix,
                              use_container_width=True,
                        )

                  if "classification_report" in metrics:
                        st.subheader("Courbes d'apprentissage")

                        if "accuracy" in curves and "val_accuracy" in curves:
                              st.markdown("**Accuracy — entraînement et validation**")
                              st.line_chart({
                                    "Entraînement": curves["accuracy"],
                                    "Validation": curves["val_accuracy"]
                              })

                        if "loss" in curves and "val_loss" in curves:
                              st.markdown("**Loss — entraînement et validation**")
                              st.line_chart({
                                    "Entraînement": curves["loss"],
                                    "Validation": curves["val_loss"]
                              })

                        if not curves:
                              st.info("Les courbes ne sont pas disponibles dans metrics.json.")

            except Exception as error:
                  st.error(
                        f"Erreur de lecture de metrics.json : {error}"
                  )

      else:
            st.info(
                  "Ajoute un fichier metrics.json contenant les "
                  "métriques réellement calculées sur le jeu de test."
            )

            st.markdown("""
            Pour compléter cette page, calcule notamment :

            - Accuracy
            - Précision, recall et F1-score par classe
            - Matrice de confusion
            - Éventuellement les courbes d'apprentissage
            """)


# =========================================================
# 11. À PROPOS
# =========================================================

elif page == "À propos":

      show_hero(
            "PRÉSENTATION TECHNIQUE",
            "À propos de NeuroVision AI",
            "Un projet de portfolio consacré au Deep Learning et "
            "à la vision par ordinateur.",
      )

      st.subheader("Technologies utilisées")

      st.markdown("""
      - **Python** : langage de développement.
      - **TensorFlow / Keras** : construction et inférence du CNN.
      - **Pillow / NumPy** : traitement des images.
      - **Pandas** : organisation et export des résultats.
      - **Streamlit** : interface web interactive.
      - **Git / GitHub** : versionnement et présentation du projet.
      """)

      st.subheader("Pipeline du projet")

      st.code(
            "Images IRM → Prétraitement → CNN entraîné "
            "→ Scores des classes → Interface Streamlit",
            language="text",
      )

      st.markdown(
            """
            <div class="notice">
            Cette démonstration est destinée à l'apprentissage et
            à la présentation technique. Elle n'est pas un dispositif
            médical validé.
            </div>
            """,
            unsafe_allow_html=True,
      )


# =========================================================
# 12. PIED DE PAGE
# =========================================================

st.divider()

st.caption(
      "NeuroVision AI · Projet étudiant de Deep Learning · "
      "Prototype pédagogique uniquement."
      )
