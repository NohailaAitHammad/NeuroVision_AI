import numpy as np
import pandas as pd
import kagglehub
import pathlib
from pathlib import Path
import cv2
from collections import Counter
from src.config import (
      DATASET_CACHE_PATH,
      DATASET_SLUG,
      LABELS_BY_FOLDER,
      VALID_EXTENSIONS,
)



def charger_dataset_path():
      """Retourne le dataset en cache ou le télécharge avec kagglehub."""
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


def verifier_chargement(dataset_path, extensions_valides=VALID_EXTENSIONS, labels_par_dossier=LABELS_BY_FOLDER):

      """Charge les images valides et leurs labels,en conservant la logique initiale."""

      num_skiped = 0
      num_corrompu = 0
      images= []
      labels = []
      try:
            data_dir = pathlib.Path(dataset_path)
            for folder in data_dir.iterdir():
                  if folder.is_dir():
                        for file in folder.iterdir():
                              if not file.is_file():
                                    continue
                              
                              if file.is_file() and not file.name.lower().endswith(extensions_valides):
                                    num_skiped +=1
                                    continue
                              
                              else:
                                    image = cv2.imread(str(file))
            
                                    if image is None:
                                          num_corrompu +=1
                                          continue
                                    
                                    else:
                                          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                          images.append(image)
                                          label = labels_par_dossier[folder.name]
                                          labels.append(label)
            
            print('nombre de fichiers invlaid', num_skiped)
            print('nombre de fichiers corrompus', num_corrompu)
      except (FileNotFoundError):
            print("Erreur : Le fichier n'existe pas.")
      except Exception as e:
            print(f"Erreur inattendue : {e}")

      return images, labels

def preparer_data(image_size, images, labels):

      """Redimensionne les images et convertit les listes en tableaux NumPy."""

      images_resized =[]

      for image in images:
            resized = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
            images_resized.append(resized)
      
      images_array_2d = np.array(images_resized)
      
      labeles_array_2d = np.array(labels)
      
      return images_array_2d, labeles_array_2d


def print_class_counts(labels, labels_par_dossier=LABELS_BY_FOLDER):
      effectifs = Counter(labels)
      noms_classes = list(labels_par_dossier.keys())
      nombres_images = [
            effectifs[labels_par_dossier[nom]] for nom in noms_classes
      ]

      for nom, nombre in zip(noms_classes, nombres_images):
            print(f"{nom} : {nombre} images")

      return noms_classes, nombres_images

