from tensorflow.keras.layers.experimental.preprocessing import Normalization


def get_preprocessing_layer():
    """Return preprocessing layer for ResNetRS models."""
    return Normalization(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
    )
