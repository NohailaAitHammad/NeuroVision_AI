from pathlib import Path

# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Kaggle (adapter si nécessaire sur une autre machine)
DATASET_CACHE_PATH = Path(
      "/root/.cache/kagglehub/datasets/devana2004/dataset/versions/1/Data"
)
DATASET_SLUG = "devana2004/dataset"

# Dossiers des résultats
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn_baseline"
MODEL_DIR = PROJECT_ROOT / "models" / "cnn_baseline"
BACKUP_DIR = OUTPUT_DIR / "backup"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Paramètre s conservés du notebook
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 1
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

LABELS_BY_FOLDER = {
      "glioma": 0,
      "meningioma": 1,
      "notumor": 2,
      "pituitary": 3,
}

CLASS_NAMES = [
      "Gliome",
      "Méningiome",
      "Aucune tumeur",
      "Tumeur hypophysaire",
]