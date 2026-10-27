import numpy as np
import pandas as pd
import kagglehub
from pathlib import Path
import cv2
from collections import Counter

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from config import DATASET_CACHE_PATH, DATASET_SLUG, LABELS_BY_FOLDER, SUPPORTED_IMAGE_EXTENSIONS, IMAGE_SIZE, RANDOM_SEED, BATCH_SIZE


def charger_dataset_path():
      
      dataset_path = DATASET_CACHE_PATH

      if dataset_path.exists() and any(dataset_path.iterdir()):
            print("Dataset déjà présent dans le cache.")
            print("Chemin :", dataset_path)
            print("Contenu :", [p.name for p in dataset_path.iterdir()])
      else:
            print("Dataset absent ou dossier vide.")
            dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG)) / "Data"

      print("Path :", dataset_path)
      return Path(dataset_path)


def check_data(path, extensions_valides=SUPPORTED_IMAGE_EXTENSIONS, labels_par_dossier=LABELS_BY_FOLDER, image_size=IMAGE_SIZE):

      images = []
      labels = []

      num_skipped = 0
      num_corrompu = 0

      data_dir = Path(path)
      extensions_valides = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions_valides
      }

      for folder in data_dir.iterdir():
            if not folder.is_dir():
                  continue

            # Ignorer les dossiers qui ne correspondent pas aux classes
            if folder.name not in labels_par_dossier:
                  continue

            for file in folder.iterdir():
                  if not file.is_file():
                        continue

                  # Vérifier l'extension avant de décoder l'image
                  if file.suffix.lower() not in extensions_valides:
                        num_skipped += 1
                        continue

                  image = cv2.imread(str(file))

                  if image is None:
                        num_corrompu += 1
                        continue

                  # BGR -> RGB
                  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                  # Redimensionnement pour MobileNetV2
                  image = cv2.resize(image, image_size)

                  images.append(image)
                  labels.append(labels_par_dossier[folder.name])

      print(f"Fichiers ignorés : {num_skipped}")
      print(f"Images corrompues : {num_corrompu}")
      print(f"Images chargées : {len(images)}")
      print(f"Labels chargés : {len(labels)}")

      return images, labels


def preparing_data(images, labels,image_size=IMAGE_SIZE):

      images_resized =[]
      
      for image in images:
            resized = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
            images_resized.append(resized)
      
      X = np.asarray(images_resized, dtype=np.uint8)
      
      y = np.asarray(labels, dtype=np.int32)

      if len(X) != len(y):
            raise ValueError(
                  "Le nombre d'images doit correspondre au nombre de labels."
            )
      
      return X, y

def spliting_data(X, y, random_state=RANDOM_SEED):

      X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=random_state)

      X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5,  stratify=y_tmp, random_state=random_state)

      print(X_train.shape, X_test.shape, X_val.shape)

      print(y_train.shape, y_test.shape, y_val.shape)

      return  X_train, X_test, X_val, y_train, y_test, y_val

def preprocess(image,label, cnn=True):

      image = tf.cast(image, tf.float32)

      # Normalisation
      if cnn:
            image = tf.cast(image, tf.float32) / 255.0
      else:
            image = preprocess_input(image)

      label = tf.cast(label, tf.int32)

      return image, label

def prep(X, y, training=True, cnn=True,  random_state=RANDOM_SEED, batch_size=BATCH_SIZE):

      X_ds = tf.data.Dataset.from_tensor_slices((X, y))

      if training:
            X_ds = X_ds.shuffle(buffer_size=len(X), seed=random_state, reshuffle_each_iteration=True)

      X_ds = X_ds.map(
          lambda image, label: preprocess(image, label, cnn=cnn),
        num_parallel_calls=tf.data.AUTOTUNE,)

      X_ds = X_ds.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

      return X_ds

def print_class_counts(labels, labels_par_dossier=LABELS_BY_FOLDER):
      effectifs = Counter(labels)
      noms_classes = list(labels_par_dossier.keys())
      nombres_images = [
            effectifs[labels_par_dossier[nom]] for nom in noms_classes
      ]

      for nom, nombre in zip(noms_classes, nombres_images):
            print(f"{nom} : {nombre} images")

      return noms_classes, nombres_images

