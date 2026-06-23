from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

IMAGE_SIZE = (64, 64)
OPENMP_INPUT_SIZE = (128, 128)
PREPROCESSING_STRATEGY = "lower_face"
CLASS_NAMES = ["no_yawn", "yawn"]
CLASS_TO_LABEL = {"no_yawn": 0, "yawn": 1}
LABEL_TO_CLASS = {0: "no_yawn", 1: "yawn"}

RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 30
AUGMENT_TRAINING = True
L2_REGULARIZATION = 0.0001
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 32
