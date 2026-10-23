import tensorflow as tf

from src.config import BATCH_SIZE


def preprocessing(X, y, training=True):

      """Crée un tf.data.Dataset, normalise en [0, 1], batch et précharge."""

      X_ds = tf.data.Dataset.from_tensor_slices((X, y))
      
      if training:
            X_ds = X_ds.shuffle(buffer_size=X_ds.cardinality())

      X_ds = X_ds.batch(batch_size=BATCH_SIZE)
      
      X_ds = X_ds.map(
            lambda images, labels: (
                  tf.cast(images, tf.float32) / 255.0,
                  labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
      )
      
      return X_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def preprocessing_v2(X, y, training=True):

      """Crée un tf.data.Dataset, batch et précharge."""

      X_ds = tf.data.Dataset.from_tensor_slices((X, y))
      
      if training:
            X_ds = X_ds.shuffle(buffer_size=X_ds.cardinality())

      X_ds = X_ds.batch(batch_size=BATCH_SIZE)
      
      X_ds = X_ds.map(
            lambda images, labels: (
                  tf.cast(images, tf.float32),
                  labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
      )
      
      return X_ds.prefetch(buffer_size=tf.data.AUTOTUNE)