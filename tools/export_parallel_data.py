"""Prepare cropped real images for the integrated OpenMP preprocessing step."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config import DATASET_DIR, OPENMP_INPUT_SIZE, PREPROCESSING_STRATEGY
from src.preprocessing import VALID_EXTENSIONS, _load_preprocessed_crops


RAW_DIR = PROJECT_ROOT / "artifacts" / "raw"
CLASS_TO_LABEL = {"no_yawn": 0, "yawn": 1}


def iter_images(split: str):
    for class_name, label in CLASS_TO_LABEL.items():
        for path in sorted((DATASET_DIR / split / class_name).rglob("*")):
            if path.suffix.lower() in VALID_EXTENSIONS:
                yield path, label


def export_split(split: str) -> int:
    images, labels = [], []
    for path, label in iter_images(split):
        crop = _load_preprocessed_crops(path, PREPROCESSING_STRATEGY)
        resized = cv2.resize(crop, OPENMP_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        images.append(np.repeat(resized[..., np.newaxis], 3, axis=2))
        labels.append(label)

    if not images:
        raise RuntimeError(f"No hay imágenes legibles en datasets/{split}.")

    pixels = np.ascontiguousarray(np.stack(images), dtype=np.uint8)
    label_array = np.asarray(labels, dtype="<i4")
    output_path = RAW_DIR / f"{split}.bin"
    with output_path.open("wb") as binary_file:
        binary_file.write(struct.pack("<8sIII", b"OMPRAW64", len(pixels), OPENMP_INPUT_SIZE[0], OPENMP_INPUT_SIZE[1]))
        binary_file.write(label_array.tobytes())
        binary_file.write(pixels.tobytes())
    return len(pixels)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    counts = {split: export_split(split) for split in ("train", "validation", "test")}
    (RAW_DIR.parent / "raw_export.json").write_text(
        json.dumps({"input_size": list(OPENMP_INPUT_SIZE), "splits": counts}, indent=2),
        encoding="utf-8",
    )
    print(f"Imágenes reales preparadas para OpenMP: {counts}")


if __name__ == "__main__":
    main()
