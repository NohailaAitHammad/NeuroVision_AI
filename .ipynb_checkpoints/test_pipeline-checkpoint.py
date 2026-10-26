import tensorflow as tf
from src.data_preprocessing import charger_dataset_path, check_data, preparing_data, spliting_data,prep
from src.models import *
from src.callbacks import *
from src.train import *
from src.metrics import *
from src.plots import *
from config import * 
from src.mlflow_tracking import * 

scores = {}
models = {}

model_paths = {
      "CNN": CNN_MODELS_DIR / BEST_MODEL_FILENAME,
      "Transfer_Learning": TRANSFER_MODELS_DIR / BEST_MODEL_FILENAME,
      "Fine_Tuning": FINE_TUNING_MODELS_DIR / BEST_MODEL_FILENAME,
}

dataset_path = charger_dataset_path()

images, labels = check_data(dataset_path,SUPPORTED_IMAGE_EXTENSIONS ,  LABELS_BY_FOLDER)


images_array_2d, labeles_array_2d = preparing_data(images=images, labels=labels,image_size=IMAGE_SIZE)

X_train, X_test, X_val, y_train, y_test, y_val = spliting_data(images_array_2d, labeles_array_2d, random_state=RANDOM_SEED)

Train_ds =  prep(X_train, y_train, training=True)
Test_ds = prep(X_test, y_test, training=False)
Val_ds = prep(X_val, y_val, training=False)
print(Train_ds, Test_ds, Val_ds)

def build_artifact_dict(output_dir):
    """
    Construit un dictionnaire des artefacts déjà générés.
    Format : nom_du_fichier -> chemin_du_fichier.
    """
    output_dir = Path(output_dir)
    artifacts = {}

    if not output_dir.exists():
        return artifacts

    for file_path in output_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {
            ".png", ".json", ".csv", ".txt"
        }:
            artifacts[file_path.name] = str(file_path)

    return artifacts

##### CNN
# 1. Construire le modèle
cnn_model = build_cnn(IMAGE_SIZE)

# 2. Compiler
cnn_model = compile_mode(cnn_model, learning_rate=CNN_LEARNING_RATE)

# 3. Vérifier l'architecture
cnn_model.summary()

# 4. Évaluer avant entraînement
# initial_metrics = evaluate_initial_model(model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(CNN_MODELS_DIR, CNN_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      cnn_model,
      Train_ds,
      Val_ds,
      epochs=EPOCHS,
      callbacks=callbacks
)

# Visualisation de l'entraînement
plot_training_history(
      history,
      output_dir=CNN_ARTIFACTS_DIR
)

# Évaluation finale sur le jeu de test
results = evaluate_model(
      model=cnn_model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=CNN_ARTIFACTS_DIR
)

scores['CNN'] = results['metrics']['f1_weighted']
models['CNN'] = cnn_model

cnn_model_path = CNN_MODELS_DIR / BEST_MODEL_FILENAME

if not cnn_model_path.exists():
    cnn_model_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_model.save(cnn_model_path)

cnn_artifacts = build_artifact_dict(CNN_ARTIFACTS_DIR)

# mlflow
log_transfer_learning_experiment(str(CNN_MODELS_DIR/BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,CNN_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_CNN, artifact_dict=cnn_artifacts)

print('CNN run successfely ')


###### Transfer Learning


# 1. Construire le modèle
transfer_model, base_model= build_model()

# 2. Compiler
transfer_model = compile_mode(transfer_model, learning_rate=TRANSFER_LEARNING_RATE)

# 3. Vérifier l'architecture
transfer_model.summary()

# 4. Évaluer avant entraînement
initial_metrics = evaluate_initial_model(transfer_model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(TRANSFER_MODELS_DIR, TRANSFER_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      transfer_model,
      Train_ds,
      Val_ds,
      epochs=EPOCHS,
      callbacks=callbacks
)

# Visualisation de l'entraînement
plot_training_history(
      history,
      output_dir=TRANSFER_ARTIFACTS_DIR
)

# Évaluation finale sur le jeu de test
results = evaluate_model(
      model=transfer_model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=TRANSFER_ARTIFACTS_DIR
)

scores['Transfer_Learning'] = results['metrics']['f1_weighted']
models['Transfer_Learning'] = transfer_model

transfer_learning_model_path = TRANSFER_MODELS_DIR / BEST_MODEL_FILENAME

if not transfer_learning_model_path.exists():
    transfer_learning_model_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.save(transfer_learning_model_path)

transfer_leanring_artifacts = build_artifact_dict(TRANSFER_ARTIFACTS_DIR)

# Garder le modèle Transfer Learning indépendant
transfer_model.save(
      TRANSFER_MODELS_DIR / BEST_MODEL_FILENAME
)

# Créer une copie indépendante pour le Fine-Tuning
fine_tuning_model = tf.keras.models.clone_model(
      transfer_model
)

fine_tuning_model.set_weights(
      transfer_model.get_weights()
)

# mlflow
log_transfer_learning_experiment(str(TRANSFER_MODELS_DIR / BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,TRANSFER_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_TRANSFER, artifact_dict=transfer_leanring_artifacts)

print('Transfer Learning run successfely ')
###### Fine Tuning

# Récupérer la base MobileNetV2 du modèle copié
base_model_ft = fine_tuning_model.get_layer("mobilenetv2_1.00_224")

# Dégeler les dernières couches
base_model_ft = degler_layers(
      base_model_ft,
      num_layers=30
)
# 2. Compiler
fine_tuning_model = compile_mode(fine_tuning_model, learning_rate=FINE_TUNING_LEARNING_RATE)

# 3. Vérifier l'architecture
fine_tuning_model.summary()


# 4. Évaluer avant entraînement
initial_metrics = evaluate_initial_model(fine_tuning_model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(FINE_TUNING_MODELS_DIR, FINE_TUNING_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      fine_tuning_model,
      Train_ds,
      Val_ds,
      epochs=EPOCHS,
      callbacks=callbacks
)

# Visualisation de l'entraînement
plot_training_history(
      history,
      output_dir=FINE_TUNING_ARTIFACTS_DIR
)

# Évaluation finale sur le jeu de test
results = evaluate_model(
      model=fine_tuning_model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=FINE_TUNING_ARTIFACTS_DIR
)
scores['Fine_Tuning'] = results['metrics']['f1_weighted']
models['Fine_Tuning'] = fine_tuning_model

fine_tuning_model_path = FINE_TUNING_MODELS_DIR / BEST_MODEL_FILENAME

if not fine_tuning_model_path.exists():
    fine_tuning_model_path.parent.mkdir(parents=True, exist_ok=True)
    fine_tuning_model.save(fine_tuning_model_path)

fine_tuning_artifacts = build_artifact_dict(FINE_TUNING_ARTIFACTS_DIR)

# Garder le modèle Transfer Learning indépendant
transfer_model.save(
      TRANSFER_MODELS_DIR / BEST_MODEL_FILENAME
)
# mlflow
log_transfer_learning_experiment(str(FINE_TUNING_MODELS_DIR/BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,FINE_TUNING_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_FINE_TUNING, artifact_dict=fine_tuning_artifacts)

print('Fine Tuning  run successfely ')
champion_name = max(scores, key=scores.get)
champion_f1 = scores[champion_name]

print("Champion :", champion_name)
print("F1 macro :", champion_f1)

champion_path = model_paths[champion_name]
if not champion_path.exists():
      raise FileNotFoundError(
            f"Modèle introuvable : {champion_path}"
      )

# Charger le modèle sauvegardé correspondant
champion_model = tf.keras.models.load_model(
      champion_path
)
enregistrer_champion_mlflow(best_model_name=champion_name, best_model=champion_model, best_cv_f1=champion_f1)

champion = mlflow.tensorflow.load_model(
      "models:/NeuroVision_AI_Champion@champion"
)
print('Champion mlflow run successfely ')