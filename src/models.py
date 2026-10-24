import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import time
from config import IMAGE_SIZE, RANDOM_SEED, BATCH_SIZE, DROPOUT_RATE

def build_data_augmentation():

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

def  build_cnn(data_augmentation, image_size=IMAGE_SIZE):
      
      # API Sequentiel
      model = models.Sequential()
      model.add(layers.Input(shape=image_size + (3,)))
      model.add(data_augmentation)
      model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_size + (3,)))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.Flatten())
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(4))

      return model

def build_model(image_size=IMAGE_SIZE,dropout_rate=DROPOUT_RATE, num_classes=4):

      data_augmentation= build_data_augmentation()

      base_model = tf.keras.applications.MobileNetV2(
            input_shape=image_size + (3,), include_top=False,
            weights='imagenet'
      )

      #Transfert_Learning : geler le backbone
      base_model.trainable = False

      inputs = tf.keras.Input(shape=(image_size +(3,)))
      x = data_augmentation(inputs)

      # Les images sont déjà prétraitées dans Train_ds
      x = base_model(x, training=False)

      x = layers.GlobalAveragePooling2D()(x)

      x= tf.keras.layers.Dropout(dropout_rate)(x)

      #Sortie en probabilites pour les 4 classes
      outputs = layers.Dense(num_classes, activation='softmax')(x)

      model = tf.keras.Model(inputs, outputs)

      return model, base_model



def degler_layers(base_model, num_layers=30):

      #Degeler le model pour permettre le fine-tuning
      base_model.trainable = True

      #Garder les premiers courches gelées
      for layer in base_model.layers[:-num_layers]:
            layer.trainable = False

      return base_model