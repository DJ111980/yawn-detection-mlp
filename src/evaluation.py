from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CLASS_NAMES


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Convierte probabilidades sigmoid en etiquetas y calcula metricas del conjunto de prueba."""
    y_pred = (y_prob >= threshold).astype("int32").reshape(-1)
    y_true = y_true.reshape(-1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def save_history_plots(history, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in [("accuracy", "accuracy.png"), ("loss", "loss.png")]:
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label=f"train_{metric}", color="#2563eb", linewidth=2.2)
        val_metric = f"val_{metric}"
        if val_metric in history and history[val_metric]:
            plt.plot(history[val_metric], label=val_metric, color="#f97316", linewidth=2.2)
        plt.xlabel("Epocas")
        plt.ylabel(metric)
        plt.title(f"Evolucion de {metric}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def save_confusion_matrix(y_true, y_prob, output_path: str | Path, threshold: float = 0.5):
    """Guarda una matriz legible para explicar falsos positivos y falsos negativos."""
    y_pred = (y_prob >= threshold).astype("int32").reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, square=True, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_results_report(metrics: dict, output_dir: str | Path):
    """Escribe las metricas finales en Markdown para leerlas directamente en VS Code."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {name: float(value) for name, value in metrics.items()}

    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {value:.4f} |"
        for name, value in normalized.items()
    )
    (output_dir / "results.md").write_text(
        "# Resultados de evaluacion\n\n"
        "| Metrica | Valor |\n"
        "|---|---:|\n"
        f"{rows}\n",
        encoding="utf-8",
    )
