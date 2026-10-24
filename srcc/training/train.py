import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import time


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

def build_model(image_size=(224, 224),dropout_rate=0.2, num_classes=4):

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

      x= tf.keras.layers.Dropout(0.2)(x)

      #Sortie en probabilites pour les 4 classes
      outputs = layers.Dense(num_classes, activation='softmax')(x)

      model = tf.keras.Model(inputs, outputs)

      return model, base_model

def compile_mode(model, learning_rate=1e-5):
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

def create_callbacks(output_dir, backup_dir):
      output_dir = Path(output_dir)
      backup_dir = Path(backup_dir)

      output_dir.mkdir(parents=True, exist_ok=True)
      backup_dir.mkdir(parents=True, exist_ok=True)

      checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
      )

      early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
      )

      backup = tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            save_freq='epoch',
            double_checkpoint=True,
            delete_checkpoint=False
      )

      return [checkpoint_best, early_stopping, backup]

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

      return (history, duree)


def degler_layers(base_model, num_layers=30):

      #Degeler le model pouir permettre le fine-tuning
      base_model.trainable = True

      #Garder les premiers courches gelées
      for layer in base_model.layers[:-num_layers]:
            layer.trainable = False

      return base_model