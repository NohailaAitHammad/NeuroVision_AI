from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.config import CLASS_NAMES, OUTPUT_DIR

def plot_learning_curves(history):

      """Sauvegarde les courbes accuracy et loss dans outputs/cnn_baseline."""

      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
      epochs = range(1, len(history.history["val_accuracy"]) + 1)

      plt.figure()
      plt.plot(epochs, history.history["accuracy"], label="accuracy")
      plt.plot(epochs, history.history["val_accuracy"], label="val_accuracy")
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy")
      plt.ylim([0, 1])
      plt.legend(loc="lower right")
      plt.title("Visualisation de la courbe d'apprentissage - Accuracy")
      plt.savefig(OUTPUT_DIR / "accuracy_visualisation.png")
      plt.close()

      plt.figure()
      plt.plot(epochs, history.history["loss"], label="loss")
      plt.plot(epochs, history.history["val_loss"], label="val_loss")
      plt.xlabel("Epoch")
      plt.ylabel("loss")
      plt.ylim([0, 2])
      plt.title("Visualisation de la courbe d'apprentissage - Loss")
      plt.legend(loc="lower right")
      plt.savefig(OUTPUT_DIR / "loss_visualisation.png")
      plt.close()


def plot_confusion_matrix(y_true, y_pred):

      """Affiche et sauvegarde la matrice de confusion."""

      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
      cm = confusion_matrix(y_true, y_pred)
      display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=CLASS_NAMES
      )
      display.plot(cmap="Blues", xticks_rotation=45)
      plt.title("Matrice de confusion CNN")
      plt.tight_layout()
      plt.savefig(OUTPUT_DIR / "matrice_confusion.png")
      plt.close()

def save_example_prediction(images_all, y_true, y_pred, index=None):

      """Sauvegarde un exemple d'image et sa vraie classe/prédiction."""
      
      if index is None:
            index = len(y_true) - 3

      plt.figure(figsize=(5, 5))
      plt.imshow(images_all[index])
      plt.title(
            f"Vraie classe : {CLASS_NAMES[int(y_true[index])]}\n"
            f"Classe prédite : {CLASS_NAMES[int(y_pred[index])]}"
      )
      plt.axis("off")
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
      plt.savefig(OUTPUT_DIR / "tru_predict_image.png")
      plt.close()