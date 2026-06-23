from __future__ import annotations

from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
METRICS_DIR = PROJECT_ROOT / "metrics"
REPORT_PATH = METRICS_DIR / "dataset_summary.md"
PLOT_PATH = METRICS_DIR / "dataset_distribution.png"

SPLITS = ("train", "validation", "test")
CLASSES = ("no_yawn", "yawn")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def collect_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for class_name in CLASSES:
            counts[split][class_name] = count_images(DATASET_DIR / split / class_name)
    return counts


def write_report(counts: dict[str, dict[str, int]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    total_images = sum(sum(classes.values()) for classes in counts.values())

    class_totals = {class_name: sum(counts[split][class_name] for split in SPLITS) for class_name in CLASSES}
    rows = "\n".join(
        f"| {split} | {counts[split]['no_yawn']} | {counts[split]['yawn']} | {sum(counts[split].values())} |"
        for split in SPLITS
    )
    REPORT_PATH.write_text(
        "# Distribucion del dataset\n\n"
        "| Division | No bostezo | Bostezo | Total |\n|---|---:|---:|---:|\n"
        f"{rows}\n\n"
        f"**Total:** {total_images} imagenes.  \\n"
        f"**Total no_yawn:** {class_totals['no_yawn']}.  \\n"
        f"**Total yawn:** {class_totals['yawn']}.\n",
        encoding="utf-8",
    )


def write_plot(counts: dict[str, dict[str, int]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    x_positions = range(len(SPLITS))
    width = 0.35
    no_yawn_counts = [counts[split]["no_yawn"] for split in SPLITS]
    yawn_counts = [counts[split]["yawn"] for split in SPLITS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in x_positions], no_yawn_counts, width, label="no_yawn", color="#2f80ed")
    ax.bar([x + width / 2 for x in x_positions], yawn_counts, width, label="yawn", color="#f2994a")

    ax.set_title("Distribucion del dataset por split y clase")
    ax.set_xlabel("Split")
    ax.set_ylabel("Cantidad de imagenes")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(SPLITS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    return True


def print_summary(counts: dict[str, dict[str, int]], plot_created: bool) -> None:
    class_totals: defaultdict[str, int] = defaultdict(int)
    total_images = 0
    for split in SPLITS:
        split_total = sum(counts[split].values())
        total_images += split_total
        print(f"{split}: {split_total} imagenes")
        for class_name in CLASSES:
            count = counts[split][class_name]
            class_totals[class_name] += count
            print(f"  - {class_name}: {count}")

    print(f"Total: {total_images} imagenes")
    for class_name in CLASSES:
        print(f"Total {class_name}: {class_totals[class_name]}")

    if total_images:
        for class_name in CLASSES:
            percentage = class_totals[class_name] / total_images * 100
            print(f"Porcentaje {class_name}: {percentage:.2f}%")

    difference = abs(class_totals["yawn"] - class_totals["no_yawn"])
    print(f"Diferencia absoluta entre clases: {difference} imagenes")

    minimum_per_class = 300
    for class_name in CLASSES:
        status = "cumple" if class_totals[class_name] >= minimum_per_class else "no cumple"
        print(f"Minimo de {minimum_per_class} imagenes para {class_name}: {status}")

    print(f"Resumen generado: {REPORT_PATH}")
    if plot_created:
        print(f"Grafica generada: {PLOT_PATH}")
    else:
        print("Grafica no generada: matplotlib no esta disponible.")


def main() -> None:
    counts = collect_counts()
    write_report(counts)
    plot_created = write_plot(counts)
    print_summary(counts, plot_created)


if __name__ == "__main__":
    main()
