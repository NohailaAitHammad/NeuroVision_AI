from pathlib import Path
import mlflow
from config import MLFLOW_TRACKING_URI

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
