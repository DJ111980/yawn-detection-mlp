"""Adaptador TensorFlow sin Keras para exportar los pesos entrenados en CUDA."""

from __future__ import annotations

import tensorflow as tf

from .config import HIDDEN_LAYER_1, HIDDEN_LAYER_2


class PureTensorFlowMLP(tf.Module):
    """Representa el MLP CUDA en TensorFlow para exportarlo como SavedModel."""

    def __init__(self, input_dim: int = 4096, seed: int = 42, name: str | None = None):
        super().__init__(name=name)
        tf.random.set_seed(seed)
        self.w1 = tf.Variable(tf.random.normal([input_dim, HIDDEN_LAYER_1], stddev=0.05), name="w1")
        self.b1 = tf.Variable(tf.zeros([HIDDEN_LAYER_1]), name="b1")
        self.w2 = tf.Variable(tf.random.normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2], stddev=0.05), name="w2")
        self.b2 = tf.Variable(tf.zeros([HIDDEN_LAYER_2]), name="b2")
        self.w3 = tf.Variable(tf.random.normal([HIDDEN_LAYER_2, 1], stddev=0.05), name="w3")
        self.b3 = tf.Variable(tf.zeros([1]), name="b3")

    @property
    def trainable_variables(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, x):
        # Replica ReLU -> ReLU -> sigmoid para que backend pueda consumir los pesos CUDA.
        hidden_1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.w2) + self.b2)
        return tf.sigmoid(tf.matmul(hidden_2, self.w3) + self.b3)


def predict(model, features):
    """Calcula la probabilidad de bostezo para vectores ya preprocesados."""
    return model(tf.convert_to_tensor(features, dtype=tf.float32)).numpy().reshape(-1)


def save_model(model, output_dir):
    """Exporta el modelo compatible con TensorFlow SavedModel."""
    tf.saved_model.save(model, str(output_dir))
