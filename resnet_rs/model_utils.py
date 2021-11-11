import sys

import tensorflow as tf


def get_survival_probability(init_rate, block_num, total_blocks):
    """Get survival probability based on block number and initial rate.

    Source:
        https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/detection/modeling/architecture/resnet.py#L34
    """
    return init_rate * float(block_num) / total_blocks


def allow_bigger_recursion(target_limit: int):
    """Increase default recursion limit to create larger models."""
    current_limit = sys.getrecursionlimit()
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)


def fixed_padding(inputs, kernel_size):
    """Pad the input along the spatial dimensions independently of input size.

    This function is copied/modified from original repo:
    https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/resnet/resnet_model.py#L357

    Args:
        inputs: `Tensor` of size `[batch, channels, height, width]` or
            `[batch, height, width, channels]` depending on `data_format`.
        kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
            operations. Should be a positive integer.
    Returns:
        A padded `Tensor` of the same `data_format` with size either intact
        (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # Use ZeroPadding as to avoid TFOpLambda layer
    padded_inputs = tf.keras.layers.ZeroPadding2D(
        padding=((pad_beg, pad_end), (pad_beg, pad_end))
    )(inputs)

    return padded_inputs
