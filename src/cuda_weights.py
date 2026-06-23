"""Convert weights produced by the CUDA trainer into the TensorFlow SavedModel used by Streamlit."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .config import HIDDEN_LAYER_1, HIDDEN_LAYER_2, IMAGE_SIZE
from .model import PureTensorFlowMLP, save_model


HEADER = struct.Struct("<8sIII")


def export_cuda_model(weights_path: str | Path, model_dir: str | Path):
    with Path(weights_path).open("rb") as binary_file:
        magic, input_dim, hidden_1, hidden_2 = HEADER.unpack(binary_file.read(HEADER.size))
        expected_input = IMAGE_SIZE[0] * IMAGE_SIZE[1]
        if magic.rstrip(b"\0") != b"CUDAWTS" or (input_dim, hidden_1, hidden_2) != (expected_input, HIDDEN_LAYER_1, HIDDEN_LAYER_2):
            raise ValueError("Los pesos CUDA no coinciden con la arquitectura actual.")
        def read_array(size: int):
            return np.frombuffer(binary_file.read(size * 4), dtype="<f4").copy()
        w1 = read_array(input_dim * hidden_1).reshape(input_dim, hidden_1)
        b1 = read_array(hidden_1)
        w2 = read_array(hidden_1 * hidden_2).reshape(hidden_1, hidden_2)
        b2 = read_array(hidden_2)
        w3 = read_array(hidden_2).reshape(hidden_2, 1)
        b3 = read_array(1)

    model = PureTensorFlowMLP(input_dim=input_dim)
    for variable, values in zip(model.trainable_variables, (w1, b1, w2, b2, w3, b3)):
        variable.assign(values)
    save_model(model, model_dir)
    return model
