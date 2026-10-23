import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, BackupAndRestore


from src.config import EPOCHS, OUTPUT_DIR, MODEL_DIR, BACKUP_DIR
from src.models.mobilenetv2 import build_fine_tuning, build_data_augmentation, fine_tuning


def train_fine_tuning(Train_ds, Val_ds):
      """Compile et entraîne le FineTuning de base."""

      model, base_model = build_fine_tuning(build_data_augmentation())

      model.compile(
                  optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"]
            )
      loss0, accuracy0 = model.evaluate(Val_ds)
      print("initial loss {:.2f}".format(loss0))
      print("initial accuracy {:.2f}".format(accuracy0))

      MODEL_DIR.mkdir(parents=True, exist_ok=True)
      BACKUP_DIR.mkdir(parents=True, exist_ok=True)
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

      # checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
      #             filepath=str(MODEL_DIR / "best_model.keras"),
      #             monitor="val_loss",
      #             mode="min",
      #             save_best_only=True,
      #             save_weights_only=False,
      #             verbose=1
      #       )
      # callback = EarlyStopping(
      #       monitor="val_loss", patience=4, restore_best_weights=True
      # )
      # backup = BackupAndRestore(
      #       str(BACKUP_DIR), save_freq="epoch",
      #       double_checkpoint=True, delete_checkpoint=False
      # )

      debut = time.time()
      history = model.fit(
            Train_ds,
            epochs=EPOCHS,
            validation_data=Val_ds,
            batch_size=32
      )
      fin = time.time()

      duree = fin - debut
      print(f"Durée de l'entraînement : {duree:.2f} secondes")
      print(f"Durée en minutes : {duree / 60:.2f} minutes")

      model = fine_tuning(model, base_model)
      model.compile(
                        optimizer="adam",
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"]
                  )

      loss0, accuracy0 = model.evaluate(Val_ds)
      print("initial loss {:.2f}".format(loss0))
      print("initial accuracy {:.2f}".format(accuracy0))

      MODEL_DIR.mkdir(parents=True, exist_ok=True)
      BACKUP_DIR.mkdir(parents=True, exist_ok=True)
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

      checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
                  filepath=str(MODEL_DIR / "best_model.keras"),
                  monitor="val_loss",
                  mode="min",
                  save_best_only=True,
                  save_weights_only=False,
                  verbose=1
            )
      callback = EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
      )
      backup = BackupAndRestore(
            str(BACKUP_DIR), save_freq="epoch",
            double_checkpoint=True, delete_checkpoint=False
      )

      debut = time.time()
      history = model.fit(
            Train_ds,
            epochs=EPOCHS,
            validation_data=Val_ds,
            batch_size=32,
            callbacks=[callback, backup, checkpoint_best]
      )
      fin = time.time()

      duree = fin - debut
      print(f"Durée de l'entraînement : {duree:.2f} secondes")
      print(f"Durée en minutes : {duree / 60:.2f} minutes")

      return model, history