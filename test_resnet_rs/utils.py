from typing import Tuple

import tensorflow as tf
from psutil import virtual_memory


def get_inference_function(model: tf.keras.Model, input_shape: Tuple[int, int]):
    """Return convertible inference function with provided model."""

    def inference_func(inputs):
        return model(inputs, training=False)

    tensor_spec = tf.TensorSpec(shape=(1, *input_shape, 3), dtype=tf.float32)
    return tf.function(func=inference_func, input_signature=[tensor_spec])


def is_enough_memory(required_ram) -> bool:
    total_ram = virtual_memory().total / (1024.0**3)
    return total_ram >= required_ram
