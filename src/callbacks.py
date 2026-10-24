import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import time
from config import EARLY_STOPPING_PATIENCE, BEST_MODEL_FILENAME

def create_callbacks(output_dir, backup_dir):

      checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / BEST_MODEL_FILENAME),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
      )

      early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
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
