from pathlib import Path


# Chemins du projet

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data"             
DATASET_CACHE_PATH =  Path(
      "/root/.cache/kagglehub/datasets/devana2004/dataset/versions/1/Data"
)
DATASET_SLUG = "devana2004/dataset"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# Dossiers par approche
CNN_MODELS_DIR = MODELS_DIR / "cnn"
TRANSFER_MODELS_DIR = MODELS_DIR / "transfer_learning"
FINE_TUNING_MODELS_DIR = MODELS_DIR / "fine_tuning"

CNN_ARTIFACTS_DIR = ARTIFACTS_DIR / "cnn"
TRANSFER_ARTIFACTS_DIR = ARTIFACTS_DIR / "transfer_learning"
FINE_TUNING_ARTIFACTS_DIR = ARTIFACTS_DIR / "fine_tuning"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Sauvegardes pour BackupAndRestore 
CNN_BACKUP_DIR = LOGS_DIR / "backup" / "cnn"
TRANSFER_BACKUP_DIR = LOGS_DIR / "backup" / "transfer_learning"
FINE_TUNING_BACKUP_DIR = LOGS_DIR / "backup" / "fine_tuning"

# Creation
OUTPUT_DIRS = (
      MODELS_DIR,
      ARTIFACTS_DIR,
      LOGS_DIR,
      MLFLOW_DIR,
      CNN_MODELS_DIR,
      TRANSFER_MODELS_DIR,
      FINE_TUNING_MODELS_DIR,
      CNN_ARTIFACTS_DIR,
      TRANSFER_ARTIFACTS_DIR,
      FINE_TUNING_ARTIFACTS_DIR,
      REPORTS_DIR,
      CNN_BACKUP_DIR,
      TRANSFER_BACKUP_DIR,
      FINE_TUNING_BACKUP_DIR,
)
for directory in OUTPUT_DIRS:
      directory.mkdir(parents=True, exist_ok=True)

# Dataset et classes

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS_BY_FOLDER = {name: index for index, name in enumerate(CLASS_NAMES)}
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")



# Entrainement

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30


CNN_LEARNING_RATE = 1e-3
TRANSFER_LEARNING_RATE = 1e-4
FINE_TUNING_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.2
EARLY_STOPPING_PATIENCE = 4

# Noms de fichiers et MLflow

BEST_MODEL_FILENAME = "best_model.keras"
METRICS_FILENAME = "metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
ACCURACY_PLOT_FILENAME = "accuracy.png"
LOSS_PLOT_FILENAME = "loss.png"

MLFLOW_EXPERIMENT_CNN = "NeuroVision_CNN"
MLFLOW_EXPERIMENT_TRANSFER = "NeuroVision_TransferLearning"
MLFLOW_EXPERIMENT_FINE_TUNING = "NeuroVision_FineTuning"

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
