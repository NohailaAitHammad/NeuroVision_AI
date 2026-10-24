import tensorflow as tf
from src.data_preprocessing import charger_dataset_path, check_data, preparing_data, spliting_data,prep
from src.models import *
from src.callbacks import *
from src.train import *
from src.metrics import *
from src.plots import *
from config import * 
from src.mlflow_tracking import * 

dataset_path = charger_dataset_path()

images, labels = check_data(dataset_path,SUPPORTED_IMAGE_EXTENSIONS ,  LABELS_BY_FOLDER)


images_array_2d, labeles_array_2d = preparing_data(images=images, labels=labels,image_size=IMAGE_SIZE)

X_train, X_test, X_val, y_train, y_test, y_val = spliting_data(images_array_2d, labeles_array_2d, random_state=RANDOM_SEED)

Train_ds =  prep(X_train, y_train, training=True)
Test_ds = prep(X_test, y_test, training=False)
Val_ds = prep(X_val, y_val, training=False)
print(Train_ds, Test_ds, Val_ds)


##### CNN

# 1. Construire le modèle
model = build_cnn()

# 2. Compiler
model = compile_mode(model, learning_rate=CNN_LEARNING_RATE)

# 3. Vérifier l'architecture
model.summary()

# 4. Évaluer avant entraînement
# initial_metrics = evaluate_initial_model(model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(CNN_MODELS_DIR, CNN_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      model,
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
      model=model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=CNN_ARTIFACTS_DIR
)

# mlflow
log_transfer_learning_experiment(str(CNN_MODELS_DIR/BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,CNN_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_CNN, artifact_dict=True)



###### Transfer Learning


# 1. Construire le modèle
model, base_model= build_model()

# 2. Compiler
model = compile_mode(model, learning_rate=TRANSFER_LEARNING_RATE)

# 3. Vérifier l'architecture
model.summary()

# 4. Évaluer avant entraînement
initial_metrics = evaluate_initial_model(model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(TRANSFER_MODELS_DIR, TRANSFER_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      model,
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
      model=model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=TRANSFER_ARTIFACTS_DIR
)
# mlflow
log_transfer_learning_experiment(str(MLFLOW_EXPERIMENT_TRANSFER/BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,TRANSFER_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_TRANSFER, artifact_dict=True)

###### Fine Tuning

base_model = degler_layers(base_model, num_layers=30)

# 2. Compiler
model = compile_mode(model, learning_rate=FINE_TUNING_LEARNING_RATE)

# 3. Vérifier l'architecture
model.summary()


# 4. Évaluer avant entraînement
initial_metrics = evaluate_initial_model(model, Val_ds)

# 5. Créer les callbacks
callbacks = create_callbacks(FINE_TUNING_MODELS_DIR, FINE_TUNING_BACKUP_DIR)

# 6. Entraîner
history, training_duration = train_model(
      model,
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
      model=model,
      test_ds=Test_ds,
      class_names=CLASS_NAMES,
      output_dir=FINE_TUNING_ARTIFACTS_DIR
)
# mlflow
log_transfer_learning_experiment(str(MLFLOW_EXPERIMENT_FINE_TUNING/BEST_MODEL_FILENAME), results,training_duration, EPOCHS, BATCH_SIZE,FINE_TUNING_LEARNING_RATE, run_name=MLFLOW_EXPERIMENT_FINE_TUNING, artifact_dict=True)