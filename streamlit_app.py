from pathlib import Path
import os
import re

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from mlflow.tracking import MlflowClient
from PIL import Image


# Configuration minimale
st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="centered")
PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_DB_PATH = Path(os.getenv("MLFLOW_DB_PATH", str(PROJECT_ROOT / "mlflow.db"))).resolve()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "NeuroVision_AI_Champion")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Méningioma", "Non Tumor", "Pituitary"]


@st.cache_resource
def load_champion():
    """Charge le champion MLflow, avec recours aux artefacts locaux du même modèle."""
    if not MLFLOW_DB_PATH.exists() and TRACKING_URI.startswith("sqlite:"):
        raise FileNotFoundError(f"Base MLflow introuvable : {MLFLOW_DB_PATH}")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    run = client.get_run(version.run_id) if version.run_id else None
    metadata = " ".join(
        [str(getattr(version, "description", "") or ""), str(getattr(run.info, "run_name", "") if run else "")]
        + [f"{k} {v}" for k, v in (run.data.params.items() if run else [])]
        + [f"{k} {v}" for k, v in (run.data.tags.items() if run else [])]
        + [f"{k} {v}" for k, v in (version.tags or {}).items()]
    ).lower()

    # Identifier de l'architecture depuis les informations associées au run.
    if any(x in metadata for x in ("fine_tuning", "fine-tuning", "finetuning", "fine tuning")):
        model_kind = "fine_tuning"
        preprocessing = "mobilenetv2"
    elif any(x in metadata for x in ("transfer_learning", "transfer learning", "mobilenetv2", "mobilenet_v2")):
        model_kind = "transfer_learning"
        preprocessing = "mobilenetv2"
    elif "cnn" in metadata:
        model_kind = "cnn"
        preprocessing = "cnn"
    else:
        model_kind = os.getenv("MLFLOW_CHAMPION_KIND", "").lower()
        preprocessing = "mobilenetv2" if model_kind in {"transfer_learning", "fine_tuning"} else "cnn"

    errors = []

    # 1) Méthode normale : chargement de l'alias champion depuis le registre MLflow.
    try:
        model = mlflow.tensorflow.load_model(MODEL_URI)
        return model, preprocessing, f"MLflow champion · version {version.version}"
    except Exception as exc:
        errors.append(f"Chargement via l'alias MLflow : {exc}")

    # 2) Sur MLflow récent, la source peut ressembler à models:/m-<id>.
    #    Chercher les artefacts de ce modèle-id dans le dossier mlruns local.
    source = str(version.source or "")
    match = re.search(r"models:/([^/]+)", source)
    if match:
        model_id = match.group(1)
        for artifact_dir in PROJECT_ROOT.glob(f"mlruns/**/models/{model_id}/artifacts"):
            if artifact_dir.is_dir():
                try:
                    model = mlflow.tensorflow.load_model(str(artifact_dir))
                    return model, preprocessing, f"Champion MLflow local · version {version.version}"
                except Exception as exc:
                    errors.append(f"Artefacts locaux {artifact_dir}: {exc}")

    # 3) Dernier recours : fichier .keras local correspondant au type du champion.
    #    On ne choisit pas arbitrairement un autre type si le type est inconnu.
    if model_kind in {"cnn", "transfer_learning", "fine_tuning"}:
        local_model = PROJECT_ROOT / "models" / model_kind / "best_model.keras"
        if local_model.is_file():
            try:
                model = tf.keras.models.load_model(local_model)
                return model, preprocessing, f"Fichier local du champion · {model_kind}"
            except Exception as exc:
                errors.append(f"Chargement de {local_model}: {exc}")

    detail = "\n".join(errors[-3:])
    raise RuntimeError(
        "Le champion MLflow est enregistré, mais ses fichiers modèle ne sont pas accessibles localement. "
        "Vérifie que les artefacts du champion sont présents dans mlruns ou que le fichier "
        "models/<type>/best_model.keras correspond bien au champion.\n\n" + detail
    )


def prepare_image(image, preprocessing):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    if preprocessing == "mobilenetv2":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    elif preprocessing == "cnn":
        array = array / 255.0
    return np.expand_dims(array, axis=0)


st.title("🧠 NeuroVision AI")
st.write("Importe une image IRM cérébrale et lance la prédiction avec le champion MLflow.")

uploaded_file = st.file_uploader("Choisir une image IRM", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        if st.button("Lancer la prédiction", type="primary", use_container_width=True):
            with st.spinner("Chargement du champion et analyse de l’image…"):
                model, preprocessing, model_source = load_champion()
                prediction = np.asarray(model.predict(prepare_image(image, preprocessing), verbose=0)).reshape(-1)

                if prediction.size != len(CLASS_NAMES):
                    st.error(f"Le modèle renvoie {prediction.size} sorties au lieu de {len(CLASS_NAMES)}.")
                else:
                    # Les modèles de classification habituels retournent déjà des probabilités softmax.
                    if np.all(prediction >= 0) and np.all(prediction <= 1) and np.isclose(prediction.sum(), 1.0, atol=0.02):
                        probabilities = prediction
                    else:
                        shifted = prediction - np.max(prediction)
                        exp_values = np.exp(shifted)
                        probabilities = exp_values / exp_values.sum()

                    best_index = int(np.argmax(probabilities))
                    st.success(f"Classe prédite : **{CLASS_NAMES[best_index]}**")
                    st.metric("Score maximal", f"{probabilities[best_index]:.1%}")
                    st.bar_chart(pd.DataFrame({"Classe": CLASS_NAMES, "Score": probabilities}).set_index("Classe"))
                    st.caption(f"Modèle utilisé : {model_source}")
                    st.caption("Prototype pédagogique : ce résultat ne constitue pas un diagnostic médical.")
    except Exception as exc:
        st.error(f"Impossible de réaliser la prédiction : {exc}")
else:
    st.info("Importe une image pour commencer.")
