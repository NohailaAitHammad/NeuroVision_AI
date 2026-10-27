from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
      accuracy_score,
      classification_report,
      confusion_matrix,
      ConfusionMatrixDisplay,
      precision_score,
      recall_score,
      f1_score
)
from config import CONFUSION_MATRIX_FILENAME, METRICS_FILENAME


def evaluate_model(model, test_ds, class_names, output_dir):
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)

      #1 Recuperer les vrais labels
      y_true = np.concatenate([
            labels.numpy() for _, labels in test_ds
      ])

      # 2 Recuperer les probabilites

      y_prob = model.predict(test_ds, verbose=1)

      # 3 Transformer les probalibites en classes predites

      y_pred = np.argmax(y_prob, axis=1)

      # Verifier la correspondance entre les classes et les indices

      class_labels = list(range(len(class_names)))

      # 4 Calculer les metriques globals

      metrics = {
            "accuracy" : float(accuracy_score(y_true=y_true, y_pred=y_pred)),
            "precision_macro" : float(
                  precision_score(
                        y_true=y_true,
                        y_pred=y_pred,labels=class_labels, average='macro', zero_division=0)),

            "recall_macro" : float(
            recall_score(
                  y_true=y_true,
                  y_pred=y_pred,
                  labels=class_labels,
                  average='macro',
                  zero_division=0)),
            "f1_weighted" : float(
            f1_score(
                  y_true=y_true,
                  y_pred=y_pred,
                  labels=class_labels,
                  average='weighted',
                  zero_division=0)),
      }

      # 5 Rapport detailler par class

      raport = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
      )

      print("\n=== Métriques globales ===")
      for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

      print("\n=== Rapport par classe ===")
      print(
            classification_report(
                  y_true,
                  y_pred,
                  labels=class_labels,
                  target_names=class_names,
                  zero_division=0
            )
      )

      # 6 Metrice de confusion
      cm= confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=class_labels
      )

      display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

      fig, ax = plt.subplots(figsize=(8,7))
      display.plot(
            ax=ax,
            cmap='Blues',
            values_format='d',
            colorbar=False
      )

      ax.set_title("Matrice de confusion")
      fig.tight_layout()
      fig.savefig(
            output_dir / CONFUSION_MATRIX_FILENAME, dpi=150,
            bbox_inches='tight'
      )
      plt.show()
      plt.close(fig)

      # 7 Sauvegarder les résultats en JSON
      results = {
            "metrics": metrics,
            "classification_report": raport,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "test_samples": int(len(y_true))
      }

      with open(output_dir / METRICS_FILENAME, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

      print(f"\nResultats sauvgardés dans : {output_dir}")

      return results