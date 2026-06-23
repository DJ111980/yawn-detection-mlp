"""Create clean static charts from the two runtime benchmark JSON reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_performance_plots(metrics_dir: str | Path) -> None:
    metrics_dir = Path(metrics_dir)
    openmp = json.loads((metrics_dir / "openmp_benchmark.json").read_text(encoding="utf-8"))
    cuda = json.loads((metrics_dir / "cuda_training.json").read_text(encoding="utf-8"))

    charts = [
        ("openmp_speedup.png", "Preprocesamiento: serial vs OpenMP", ["Serial", "OpenMP (8 hilos)"], [openmp["serial_ms"], openmp["openmp_ms"]], ["#2563eb", "#16a34a"]),
        ("cuda_speedup.png", "Entrenamiento: CPU vs CUDA", ["CPU", "CUDA GPU"], [cuda["cpu"]["time_ms"], cuda["cuda"]["time_ms"]], ["#2563eb", "#f97316"]),
    ]
    for filename, title, labels, values, colors in charts:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(f"{title} - tiempo total medido")
        ax.set_ylabel("Tiempo (ms)")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, labels=[f"{value:.1f} ms" for value in values], padding=4)
        fig.tight_layout()
        fig.savefig(metrics_dir / filename, dpi=170)
        plt.close(fig)

    epochs = range(1, cuda["epochs"] + 1)
    for metric, filename, title, ylabel in (
        ("loss", "loss.png", "Perdida por epoca", "Binary Cross Entropy"),
        ("accuracy", "accuracy.png", "Accuracy por epoca", "Accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        validation_metric = f"val_{metric}"
        # CUDA se dibuja primero porque ambas curvas de entrenamiento pueden coincidir.
        ax.plot(epochs, cuda["cuda"][metric], label="CUDA entrenamiento", color="#f97316", linewidth=1.9, alpha=0.82, zorder=2)
        ax.plot(epochs, cuda["cpu"][metric], label="CPU entrenamiento", color="#2563eb", linewidth=2.8, linestyle="--", zorder=3)
        if validation_metric in cuda["cuda"]:
            ax.plot(epochs, cuda["cuda"][validation_metric], label="CUDA validacion", color="#dc2626", linewidth=2.0, linestyle=":", zorder=4)
        if validation_metric in cuda["cpu"]:
            ax.plot(epochs, cuda["cpu"][validation_metric], label="CPU validacion", color="#0891b2", linewidth=1.8, linestyle="-.", zorder=4)
        ax.set_title(title)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        differences = [cpu - gpu for cpu, gpu in zip(cuda["cpu"][metric], cuda["cuda"][metric])]
        inset = ax.inset_axes([0.58, 0.14, 0.35, 0.23])
        inset.plot(epochs, differences, color="#7c3aed", linewidth=1.25)
        inset.axhline(0.0, color="#6b7280", linewidth=0.7)
        inset.set_title("CPU - CUDA", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(metrics_dir / filename, dpi=170)
        plt.close(fig)
