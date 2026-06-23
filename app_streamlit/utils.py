from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled by the Streamlit UI.
    cv2 = None

try:
    from src.config import HIDDEN_LAYER_1, HIDDEN_LAYER_2, IMAGE_SIZE
    from src.preprocessing import preprocess_image_array as _project_preprocess_image_array
except ImportError:  # pragma: no cover - fallback for isolated execution.
    IMAGE_SIZE = (64, 64)
    HIDDEN_LAYER_1 = 64
    HIDDEN_LAYER_2 = 32
    _project_preprocess_image_array = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_WEIGHTS_PATH = PROJECT_ROOT / "models" / "cuda_weights.bin"


class MissingDependencyError(RuntimeError):
    """Raised when a runtime dependency needed by the UI is unavailable."""


class CudaMlpInference:
    """Inferencia NumPy con los pesos que produjo el entrenamiento CUDA."""

    def __init__(self, w1, b1, w2, b2, w3, b3):
        self.w1, self.b1 = w1, b1
        self.w2, self.b2 = w2, b2
        self.w3, self.b3 = w3, b3

    def predict(self, input_vector: np.ndarray) -> np.ndarray:
        hidden_1 = np.maximum(input_vector @ self.w1 + self.b1, 0.0)
        hidden_2 = np.maximum(hidden_1 @ self.w2 + self.b2, 0.0)
        logits = hidden_2 @ self.w3 + self.b3
        return 1.0 / (1.0 + np.exp(-logits))


def check_runtime_dependencies() -> list[str]:
    return ["OpenCV"] if cv2 is None else []


def load_trained_model():
    """Carga una vez los pesos CUDA que usan imagen, foto y camara en vivo."""
    if not CUDA_WEIGHTS_PATH.exists():
        raise FileNotFoundError("No se encontro models/cuda_weights.bin. Ejecuta primero python -m src.train.")

    with CUDA_WEIGHTS_PATH.open("rb") as binary_file:
        magic, input_dim, hidden_1, hidden_2 = struct.unpack("<8sIII", binary_file.read(20))
        expected = (IMAGE_SIZE[0] * IMAGE_SIZE[1], HIDDEN_LAYER_1, HIDDEN_LAYER_2)
        if magic.rstrip(b"\0") != b"CUDAWTS" or (input_dim, hidden_1, hidden_2) != expected:
            raise ValueError("Los pesos CUDA no coinciden con la arquitectura actual.")

        def read_array(size: int) -> np.ndarray:
            return np.frombuffer(binary_file.read(size * 4), dtype="<f4").copy()

        w1 = read_array(input_dim * hidden_1).reshape(input_dim, hidden_1)
        b1 = read_array(hidden_1)
        w2 = read_array(hidden_1 * hidden_2).reshape(hidden_1, hidden_2)
        b2 = read_array(hidden_2)
        w3 = read_array(hidden_2).reshape(hidden_2, 1)
        b3 = read_array(1)
    return CudaMlpInference(w1, b1, w2, b2, w3, b3), CUDA_WEIGHTS_PATH


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if cv2 is None:
        raise MissingDependencyError("OpenCV no esta instalado.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("La imagen no pudo leerse. Prueba con JPG, PNG, BMP o WEBP.")
    return image


def preprocess_image_array(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reutiliza exactamente el preprocesamiento compartido con el entrenamiento."""
    if _project_preprocess_image_array is None:
        raise MissingDependencyError("No se pudo cargar el preprocesamiento del proyecto.")
    return _project_preprocess_image_array(image_bgr)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise MissingDependencyError("OpenCV no esta instalado.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_probability(model, input_vector: np.ndarray) -> float:
    """Devuelve la salida sigmoid: probabilidad de que la imagen sea un bostezo."""
    return float(model.predict(input_vector).reshape(-1)[0])


def classify_probability(probability: float, threshold: float = 0.5) -> str:
    return "Bostezo detectado" if probability >= threshold else "No se detecta bostezo"
