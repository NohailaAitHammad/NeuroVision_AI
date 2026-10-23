import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
      accuracy_score, precision_score, recall_score, f1_score,
      confusion_matrix, classification_report
)

from src.config import CLASS_NAMES, OUTPUT_DIR


def evaluate_and_save_metrics(model, Test_ds, history=None):
      """Évalue le modèle et enregistre les métriques au format JSON."""

      y_true = np.concatenate([
            labels.numpy() for images, labels in Test_ds
      ])

      if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

      y_prob = model.predict(Test_ds)
      y_pred = np.argmax(y_prob, axis=1)

      metrics = {
            "model": "cnn",
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                  precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                  recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_score": float(
                  f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                  y_true, y_pred, labels=range(len(CLASS_NAMES))
            ).tolist(),
            "class_names": CLASS_NAMES,
            "classification_report": classification_report(
                  y_true, y_pred, labels=range(len(CLASS_NAMES)),
                  target_names=CLASS_NAMES, output_dict=True, zero_division=0
            )
      }

      if history is not None:
            metrics["learning_curves"] = {
                  key: [float(value) for value in values]
                  for key, values in history.history.items()
            }

      metrics_dir = OUTPUT_DIR / "metrics"
      metrics_dir.mkdir(parents=True, exist_ok=True)
      output_path = metrics_dir / "metrics_cnn.json"

      with output_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4, ensure_ascii=False)

      print(f"Métriques sauvegardées dans : {output_path.resolve()}")
      
      return metrics
