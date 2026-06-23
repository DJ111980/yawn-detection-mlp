from pathlib import Path

import cv2
import numpy as np

from .config import AUGMENT_TRAINING, CLASS_TO_LABEL, IMAGE_SIZE, OPENMP_INPUT_SIZE, PREPROCESSING_STRATEGY


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
BLUR_KERNEL = (3, 3)
GAUSSIAN_3X3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype="float32") / 16.0


def _largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def _center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    start_x = (width - side) // 2
    start_y = (height - side) // 2
    return image[start_y : start_y + side, start_x : start_x + side]


def _portrait_fallback_crop(image: np.ndarray) -> np.ndarray:
    """Fallback crop for centered portraits when no face is detected."""
    height, width = image.shape[:2]
    x1 = int(width * 0.12)
    x2 = int(width * 0.88)
    y1 = int(height * 0.08)
    y2 = int(height * 0.78)
    return image[y1:y2, x1:x2]


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _detect_largest_face(gray_image: np.ndarray):
    angles = [0, -12, 12, -24, 24]
    parameter_sets = [
        {"scaleFactor": 1.05, "minNeighbors": 4},
        {"scaleFactor": 1.10, "minNeighbors": 3},
    ]

    for angle in angles:
        candidate = gray_image if angle == 0 else _rotate_image(gray_image, angle)
        equalized = cv2.equalizeHist(candidate)

        for params in parameter_sets:
            faces = FACE_CASCADE.detectMultiScale(equalized, minSize=(35, 35), **params)
            face = _largest_face(faces)
            if face is not None:
                return candidate, face

    return gray_image, None


def crop_mouth_region(gray_image: np.ndarray) -> np.ndarray:
    """Crop a mouth-centered region from the detected face."""
    working_image, face = _detect_largest_face(gray_image)

    if face is None:
        return _portrait_fallback_crop(gray_image)

    x, y, w, h = face
    x1 = x + int(w * 0.10)
    x2 = x + int(w * 0.90)
    y1 = y + int(h * 0.42)
    y2 = y + int(h * 1.04)

    x1 = max(0, x1)
    x2 = min(working_image.shape[1], x2)
    y1 = max(0, y1)
    y2 = min(working_image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return _portrait_fallback_crop(gray_image)

    return working_image[y1:y2, x1:x2]


def crop_lower_face(gray_image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the mouth crop."""
    return crop_mouth_region(gray_image)


def _adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    adjusted = image.astype("float32") * alpha + beta
    return np.clip(adjusted, 0, 255).astype("uint8")


def _translate_image(image: np.ndarray, tx_ratio: float, ty_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, width * tx_ratio], [0, 1, height * ty_ratio]])
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _augment_crop(gray_crop: np.ndarray) -> list[np.ndarray]:
    """Create mild training-only variants without changing the physical dataset."""
    return [
        _rotate_image(gray_crop, -5),
        _rotate_image(gray_crop, 5),
        _translate_image(gray_crop, -0.04, 0.02),
        _translate_image(gray_crop, 0.04, -0.02),
        _adjust_brightness_contrast(gray_crop, 1.12, 8),
        _adjust_brightness_contrast(gray_crop, 0.88, -8),
    ]


def _vectorize_crop(gray_crop: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    standardized_crop = cv2.resize(gray_crop, OPENMP_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.filter2D(
        standardized_crop.astype("float32"),
        cv2.CV_32F,
        GAUSSIAN_3X3,
        borderType=cv2.BORDER_REPLICATE,
    )
    output_width, output_height = image_size
    source_height, source_width = blurred.shape
    x_positions = np.linspace(0, source_width - 1, output_width, dtype="float32")
    y_positions = np.linspace(0, source_height - 1, output_height, dtype="float32")
    x0, y0 = x_positions.astype("int32"), y_positions.astype("int32")
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    wx, wy = x_positions - x0, y_positions - y0
    top = blurred[y0[:, None], x0] * (1 - wx) + blurred[y0[:, None], x1] * wx
    bottom = blurred[y1[:, None], x0] * (1 - wx) + blurred[y1[:, None], x1] * wx
    resized = top * (1 - wy[:, None]) + bottom * wy[:, None]
    normalized = resized.astype("float32") / 255.0
    return normalized.flatten()


def preprocess_image_array(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact project preprocessing to an in-memory BGR image."""
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    crop = _load_preprocessed_crops_from_gray(gray, PREPROCESSING_STRATEGY)
    vector = _vectorize_crop(crop, IMAGE_SIZE)
    return vector.reshape(1, -1), vector.reshape(IMAGE_SIZE)


def _load_preprocessed_crops(image_path: str | Path, strategy: str) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _load_preprocessed_crops_from_gray(gray, strategy)


def _load_preprocessed_crops_from_gray(gray: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "lower_face":
        return crop_lower_face(gray)
    if strategy == "center":
        return _center_crop(gray)
    if strategy == "full":
        return gray
    raise ValueError(f"Estrategia de preprocesamiento no soportada: {strategy}")


def preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    strategy: str = PREPROCESSING_STRATEGY,
) -> np.ndarray:
    """Load, grayscale, crop, smooth, resize, normalize and flatten one image."""
    gray_crop = _load_preprocessed_crops(image_path, strategy)
    return _vectorize_crop(gray_crop, image_size)


def preprocess_image_variants(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    strategy: str = PREPROCESSING_STRATEGY,
) -> list[np.ndarray]:
    """Return the original preprocessed image plus training-only augmented variants."""
    gray_crop = _load_preprocessed_crops(image_path, strategy)
    crops = [gray_crop, *_augment_crop(gray_crop)]
    return [_vectorize_crop(crop, image_size) for crop in crops]


def load_split(
    dataset_dir: str | Path,
    split: str,
    image_size: tuple[int, int] = IMAGE_SIZE,
    augment: bool = False,
):
    """Load one dataset split from folders named yawn and no_yawn."""
    split_dir = Path(dataset_dir) / split
    features, labels, paths = [], [], []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if augment:
                variants = preprocess_image_variants(image_path, image_size)
                features.extend(variants)
                labels.extend([label] * len(variants))
                paths.extend([str(image_path)] * len(variants))
            else:
                features.append(preprocess_image(image_path, image_size))
                labels.append(label)
                paths.append(str(image_path))

    if not features:
        return np.empty((0, image_size[0] * image_size[1]), dtype="float32"), np.array([], dtype="int32"), []

    return np.vstack(features).astype("float32"), np.array(labels, dtype="int32"), paths


def load_paths(paths: list[str], labels: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE, augment: bool = False):
    features, expanded_labels = [], []
    for image_path, label in zip(paths, labels):
        if augment:
            variants = preprocess_image_variants(image_path, image_size)
            features.extend(variants)
            expanded_labels.extend([label] * len(variants))
        else:
            features.append(preprocess_image(image_path, image_size))
            expanded_labels.append(label)

    return np.vstack(features).astype("float32"), np.array(expanded_labels, dtype="int32")


def load_all_dataset(dataset_dir: str | Path, image_size: tuple[int, int] = IMAGE_SIZE):
    """Load train, validation and test splits."""
    return {
        split: load_split(dataset_dir, split, image_size, augment=(split == "train" and AUGMENT_TRAINING))
        for split in ("train", "validation", "test")
    }
