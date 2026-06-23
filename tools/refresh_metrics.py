"""Regenerate readable metric images from an already trained CUDA model without retraining."""

from __future__ import annotations

from pathlib import Path

from app_streamlit.utils import load_trained_model, predict_probability
from src.config import METRICS_DIR
from src.evaluation import compute_metrics, save_confusion_matrix, save_results_report
from src.native_data import load_preprocessed_split
from src.performance_plots import create_performance_plots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "artifacts" / "preprocessed" / "test.bin"


def main() -> None:
    if not TEST_DATA_PATH.exists():
        raise SystemExit("No existe artifacts/preprocessed/test.bin. Ejecuta entrenamiento primero.")

    model, _ = load_trained_model()
    features, labels = load_preprocessed_split(TEST_DATA_PATH)
    probabilities = [predict_probability(model, feature.reshape(1, -1)) for feature in features]
    metrics = compute_metrics(labels, probabilities)
    save_results_report(metrics, METRICS_DIR)
    save_confusion_matrix(labels, probabilities, METRICS_DIR / "confusion_matrix.png")
    create_performance_plots(METRICS_DIR)
    print("Metricas y graficas actualizadas sin entrenar el modelo.")


if __name__ == "__main__":
    main()
