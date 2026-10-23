import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_SIZE

def  build_transfer_learning(data_augmentation):


      # Api Fonctionnel
      IMG_SHAPE = IMAGE_SIZE + (3,)
      base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

      base_model.trainable = False

      global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

      prediction_layer = tf.keras.layers.Dense(4, activation="softmax")

      preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

      inputs = tf.keras.Input(shape=(224, 224, 3))
      x = data_augmentation(inputs)
      x = preprocess_input(x)
      x = base_model(x, training=False)
      x  = global_average_layer(x)
      x = tf.keras.layers.Dropout(0.2)(x)
      outputs = prediction_layer(x)
      model = tf.keras.Model(inputs, outputs)

      return model