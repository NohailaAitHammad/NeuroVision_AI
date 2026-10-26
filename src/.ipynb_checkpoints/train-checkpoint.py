import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import time
from config import TRANSFER_LEARNING_RATE

def compile_mode(model, learning_rate=TRANSFER_LEARNING_RATE):

      model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
      )

      return model

def evaluate_initial_model(model, val_ds):
      loss, accuracy = model.evaluate(val_ds, verbose=1)

      print(f"Loss initiale : {loss:.4f}")
      print(f"Accuracu initiale : {accuracy:.4f}")

      return {
            "initial_val_loss": float(loss),
            "initial_val_accuracy": float(accuracy)
      }


def train_model(model, train_ds, val_ds, epochs, callbacks):

      debut = time.time()

      history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
      )

      duree = time.time() - debut

      print(f"Durée : {duree:.2f} secondes")
      print(f"Durée : {duree /60:.2f} minutes")

      return history, duree

