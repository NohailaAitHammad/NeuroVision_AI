from pathlib import Path
import mlflow
from config import MLFLOW_TRACKING_URI
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


def log_transfer_learning_experiment(model_path, results, training_duration, epochs, batch_size, learning_rate,run_name,  artifact_dict=None, ):
      mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
      mlflow.set_experiment('NeuroVision AI')

      with mlflow.start_run(run_name=run_name) as run:

            #Parameteres de l'experience
            mlflow.log_params({
                  "model" : run_name,
                  "strategy" : run_name + "Strategy",
                  "image_size" : "224x224",
                  "num_classes" : 4,
                  "epochs_requested" : epochs,
                  "Batch_size" : batch_size,
                  "learning_rate" : learning_rate,
                  "optimizer" : "Adam",
                  "loss" : "SparseCategoricalCrossentropy",
                  "augmentation" : True
            })

            #Metriques globales calculées sur le jeu de test

            for name, value in results['metrics'].items():
                  mlflow.log_metric(name, float(value))

            mlflow.log_metric("training_duration_seconds", float(training_duration))

            # Rapport , matrice de confusion et autres fichiers : 
            if artifact_dict:
                  for directory in artifact_dict:
                        directory = Path(directory)
                        if directory.exists():
                              mlflow.log_artifacts(str(directory))

            # Enregistrer le model existant sans réentrainer

            if Path(model_path).exists():
                  mlflow.log_artifact(str(model_path), artifact_path="model_files")
            else:
                  raise FileNotFoundError(
                        f"Mode introuvable : {model_path}"
                  )
            print("Run Mlflow enregistré : ", run.info.run_id)


def enregistrer_champion_mlflow(best_model_name, best_model, best_cv_f1):

      mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

      mlflow.set_experiment('NeuroVision AI')

      model_registry_name = 'NeuroVision AI'

      #enregistrer me pipeline champion dans un nouveau run

      with mlflow.start_run(run_name=f"champion_{best_model_name}") as run :

            mlflow.log_param('champion_model', best_model_name)

            mlflow.log_metric('cv_f1_macro', float(best_cv_f1))

            mlflow.sklearn.log_model(
                  sk_model=best_model,
                  name="model",
                  serialization_format='cloudpickle'
            )

            champion_run_id = run.info.run_id

      # Enregistrer le model dans le model Registry

      model_uri = f"runs:/{champion_run_id}/model"

      model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
      )

      # Attribuer l'alisa champion a cette version

      client = MlflowClient()

      client.set_registered_model_alias(
            name=model_registry_name,
            alias='champion',
            version=model_version.version
      )

      print("\nChampion enregistrer avec success")
      print("Model : ", best_model_name)
      print('CV_F1', best_cv_f1)
      print('Version MLflow', model_version.version)
      print('Alisas : champion')


def enregistrer_champion_mlflow(
      best_model_name,
      best_model,
      best_cv_f1
):
      mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
      mlflow.set_experiment("NeuroVision AI")

      model_registry_name = "NeuroVision_AI_Champion"

      with mlflow.start_run(
            run_name=f"champion_{best_model_name}"
      ) as run:

            mlflow.log_param("champion_model", best_model_name)
            mlflow.log_metric("cv_f1_macro", float(best_cv_f1))

            # Enregistrer le modèle TensorFlow/Keras
            mlflow.tensorflow.log_model(
                  model=best_model,
                  name="model"
            )

            champion_run_id = run.info.run_id

      # Enregistrer une nouvelle version dans le Registry
      model_uri = f"runs:/{champion_run_id}/model"

      model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
      )

      # Attribuer l'alias champion à cette version
      client = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI
      )

      client.set_registered_model_alias(
            name=model_registry_name,
            alias="champion",
            version=model_version.version
      )

      print("Champion enregistré avec succès")
      print("Modèle :", best_model_name)
      print("CV F1 macro :", best_cv_f1)
      print("Version MLflow :", model_version.version)
      print("Alias : champion")

      return model_version
