"""Flujo unico: preprocesamiento OpenMP, comparacion CPU/CUDA y exportacion del modelo."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

from .config import METRICS_DIR, MODELS_DIR
from .cuda_weights import export_cuda_model
from .evaluation import compute_metrics, save_confusion_matrix, save_results_report
from .model import predict
from .native_data import load_preprocessed_split
from .performance_plots import create_performance_plots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def executable_path(name: str) -> Path:
    return BUILD_DIR / f"{name}.exe" if os.name == "nt" else BUILD_DIR / name


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_windows_shell(command: str) -> None:
    print(command)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, shell=True)


def compile_native_modules() -> tuple[Path, Path]:
    """Compila los modulos nativos usados por una corrida manual de entrenamiento."""
    BUILD_DIR.mkdir(exist_ok=True)
    openmp = executable_path("openmp_preprocess")
    cuda = executable_path("mlp_cuda")
    run(["gcc", "-O2", "-std=c11", "-fopenmp", "src/openmp_preprocess.c", "-o", str(openmp)])
    if os.name == "nt":
        candidates = glob(r"C:\Program Files*\Microsoft Visual Studio\*\*\Common7\Tools\VsDevCmd.bat")
        if not candidates:
            raise RuntimeError("No se encontró Visual Studio Build Tools para compilar CUDA.")
        developer_prompt = max(candidates)
        command = (
            f'call "{developer_prompt}" -arch=x64 >nul && '
            f'nvcc -O2 src\\mlp_cuda.cu src\\mlp_cuda_kernels.cu -o "{cuda}"'
        )
        run_windows_shell(command)
    else:
        nvcc = shutil.which("nvcc")
        if nvcc is None and Path("/usr/local/cuda/bin/nvcc").exists():
            nvcc = "/usr/local/cuda/bin/nvcc"
        if nvcc is None:
            raise RuntimeError("No se encontro nvcc en WSL. Instala CUDA Toolkit o agrega nvcc al PATH.")
        run([nvcc, "-O2", "src/mlp_cuda.cu", "src/mlp_cuda_kernels.cu", "-o", str(cuda)])
    return openmp, cuda


def reset_model_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    """Ejecuta una corrida completa solo cuando el usuario decide entrenar manualmente."""
    METRICS_DIR.mkdir(exist_ok=True)
    # Exporta las tres divisiones para que OpenMP aplique el mismo preprocesamiento.
    run([sys.executable, "-m", "tools.export_parallel_data"])
    openmp, cuda = compile_native_modules()

    processed_dir = ARTIFACTS_DIR / "preprocessed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Se miden serial y OpenMP sobre train, validation y test antes de entrenar.
    run([
        str(openmp),
        str(ARTIFACTS_DIR / "raw" / "train.bin"),
        str(ARTIFACTS_DIR / "raw" / "validation.bin"),
        str(ARTIFACTS_DIR / "raw" / "test.bin"),
        str(processed_dir / "train.bin"),
        str(processed_dir / "validation.bin"),
        str(processed_dir / "test.bin"),
        str(METRICS_DIR / "openmp_benchmark.md"),
        str(METRICS_DIR / "openmp_benchmark.json"),
    ])
    weights_path = MODELS_DIR / "cuda_weights.bin"
    # CUDA selecciona la mejor epoca con val_loss; CPU repite esas epocas para el speedup.
    run([
        str(cuda),
        str(processed_dir / "train.bin"),
        str(processed_dir / "validation.bin"),
        str(processed_dir / "test.bin"),
        str(METRICS_DIR / "cuda_training.md"),
        str(METRICS_DIR / "cuda_training.json"),
        str(weights_path),
    ])

    # El binario CUDA es el modelo que consume Streamlit; SavedModel queda para integracion externa.
    reset_model_dir(MODELS_DIR / "best_model")
    reset_model_dir(MODELS_DIR / "final_model")
    model = export_cuda_model(weights_path, MODELS_DIR / "final_model")
    shutil.copytree(MODELS_DIR / "final_model", MODELS_DIR / "best_model")
    x_test, y_test = load_preprocessed_split(processed_dir / "test.bin")
    probabilities = predict(model, x_test)
    # Las metricas finales se calculan una sola vez sobre test, nunca sobre validation.
    metrics = compute_metrics(y_test, probabilities)
    save_results_report(metrics, METRICS_DIR)
    save_confusion_matrix(y_test, probabilities, METRICS_DIR / "confusion_matrix.png")
    create_performance_plots(METRICS_DIR)
    print("Pipeline completo: OpenMP -> CUDA -> modelo final -> métricas y gráficas.")


if __name__ == "__main__":
    main()
