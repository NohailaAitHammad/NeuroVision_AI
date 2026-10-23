import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import IMAGE_SIZE, LABELS_BY_FOLDER
from src.data.loading import charger_dataset_path,verifier_chargement, preparer_data, print_class_counts
from src.data.splitting import spliting_data
from src.data.pipeline import preprocessing
from src.training.train_cnn import train_cnn
from src.evaluation.metrics import evaluate_and_save_metrics
from src.evaluation.plots import plot_learning_curves, plot_confusion_matrix, save_example_prediction
from src.config import OUTPUT_DIR, MODEL_DIR, RANDOM_STATE

def main():
      #Chargement 
      dataset_path = charger_dataset_path
      #Verification
      images, labels = verifier_chargement(dataset_path)
      #Transformation à 2d arrays
      images_array_2d, labeles_array_2d = preparer_data(images, labels, IMAGE_SIZE)

      # Aperçu des premières images : les images ont déjà été converties en RGB.
      plt.figure(figsize=(10, 10))
      n_images = min(9, len(images))
      for i in range(n_images):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            nom_class = next(
                  (k for k, v in LABELS_BY_FOLDER.items() if v == labels[i]), None
            )
            plt.title(nom_class)
      plt.tight_layout()
      OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
      plt.savefig(OUTPUT_DIR / "example1.png")
      plt.close()

      noms_classes, nombres_images = print_class_counts(labels)
      plt.figure()
      plt.bar(noms_classes, nombres_images)
      plt.title("Nombre d'images par classe")
      plt.xlabel("Classes")
      plt.ylabel("Nombre d'images")
      plt.savefig(OUTPUT_DIR / "image_par_class.png")
      plt.close()

      X_train, X_test, X_val, y_train, y_test, y_val = spliting_data(
            images_array_2d, labeles_array_2d, RANDOM_STATE
      )

      print("X_train :", type(X_train))
      print("Shape :", X_train.shape)
      print("Type des données :", X_train.dtype)
      print("Taille en mémoire :", round(X_train.nbytes / 1024**3, 2), "GB")
      print("y_train :", np.asarray(y_train).shape)
      print("GPU détecté :", tf.config.list_physical_devices("GPU"))


      Train_ds = preprocessing(X_train, y_train, training=True)
      Test_ds = preprocessing(X_test, y_test, training=False)
      Val_ds = preprocessing(X_val, y_val, training=False)

      for image_batch, label_batch in Train_ds.take(1):
            print(image_batch.shape)
            print(label_batch.shape)
      for batch_images, batch_labels in Train_ds.take(1):
            print("Forme des images :", batch_images.shape)
            print("Valeur minimale :", float(tf.reduce_min(batch_images)))
            print("Valeur maximale :", float(tf.reduce_max(batch_images)))
            print("Labels :", batch_labels[:10].numpy())

      model, history = train_cnn(Train_ds, Val_ds)

      plot_learning_curves(history)

      test_loss, test_acc = model.evaluate(Test_ds, verbose=2)
      print("Test loss :", test_loss)
      print("Test accuracy :", test_acc)

      y_prob = model.predict(Test_ds)
      y_pred = np.argmax(y_prob, axis=1)
      y_true = np.concatenate([batch_labels.numpy() for _, batch_labels in Test_ds])


      plot_confusion_matrix(y_true, y_pred)
      evaluate_and_save_metrics(model, Test_ds, history)

      images_all = np.concatenate([batch_images.numpy() for batch_images, _ in Test_ds])
      save_example_prediction(images_all, y_true, y_pred)

      best_model_path = MODEL_DIR / "best_model.keras"
      best_model = tf.keras.models.load_model(best_model_path)
      best_model.summary()
      best_model.evaluate(Test_ds)



if __name__ == "__main__":
      main()