from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled by the Streamlit UI.
    cv2 = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled by the Streamlit UI.
    tf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
IMAGE_SIZE = (64, 64)

MODEL_CANDIDATES = (
    MODELS_DIR / "best_model",
    MODELS_DIR / "final_model",
)


class MissingDependencyError(RuntimeError):
    """Raised when a required runtime dependency is not installed."""


def check_runtime_dependencies() -> list[str]:
    missing = []
    if tf is None:
        missing.append("TensorFlow")
    if cv2 is None:
        missing.append("OpenCV")
    return missing


def read_default_threshold() -> float:
    threshold_path = METRICS_DIR / "best_threshold.txt"
    if not threshold_path.exists():
        return 0.5

    try:
        return float(threshold_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0.5


def load_trained_model():
    if tf is None:
        raise MissingDependencyError(
            "TensorFlow no esta instalado. Instala las dependencias antes de ejecutar la app."
        )

    load_errors = []
    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue

        try:
            return tf.saved_model.load(str(model_path)), model_path
        except Exception as exc:  # pragma: no cover - depends on local model files.
            load_errors.append(f"{model_path}: {exc}")

    if load_errors:
        details = "\n".join(load_errors)
        raise RuntimeError(f"No fue posible cargar los modelos disponibles:\n{details}")

    candidates = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"No se encontro un modelo entrenado en: {candidates}")


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    if cv2 is None:
        raise MissingDependencyError(
            "OpenCV no esta instalado. Instala opencv-python para procesar imagenes."
        )

    image_bytes = uploaded_file.getvalue()
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("La imagen no pudo leerse. Prueba con un archivo JPG, PNG, BMP o WEBP.")
    return image


def preprocess_image_array(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise MissingDependencyError(
            "OpenCV no esta instalado. Instala opencv-python para procesar imagenes."
        )

    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, IMAGE_SIZE)
    normalized = resized.astype("float32") / 255.0
    input_vector = normalized.flatten().reshape(1, -1)
    return input_vector, normalized


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise MissingDependencyError(
            "OpenCV no esta instalado. Instala opencv-python para procesar imagenes."
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_probability(model, input_vector: np.ndarray) -> float:
    if tf is None:
        raise MissingDependencyError(
            "TensorFlow no esta instalado. Instala las dependencias antes de ejecutar la app."
        )

    tensor = tf.convert_to_tensor(input_vector, dtype=tf.float32)
    output = model(tensor).numpy().reshape(-1)
    return float(output[0])


def classify_probability(probability: float, threshold: float) -> str:
    return "Bostezo detectado" if probability >= threshold else "No se detecta bostezo"

