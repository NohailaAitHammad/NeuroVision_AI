import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models



def  build_cnn(data_augmentation):

      """Construit le CNN de base comme dans le notebook."""
      
      # API Sequentiel
      model = models.Sequential()
      model.add(layers.Input(shape=(224, 224, 3)))
      model.add(data_augmentation)
      model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.Flatten())
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(4))

      return model

def  build_data_augmentation():
      """Retourne les transformations du notebook."""
      return tf.keras.Sequential([
            layers.RandomRotation(0.05),
            layers.RandomZoom(
                  height_factor=(-0.2, 0.2),
                  width_factor=(-0.2, 0.2)
            ),
            layers.RandomTranslation(
                  height_factor=0.05,
                  width_factor=0.05
            ),
            layers.RandomZoom(0.1)
      ], name="data_augmentation")
