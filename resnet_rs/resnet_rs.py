"""Architecture code for Resnet RS models."""

import sys

import tensorflow as tf

BLOCK_ARGS = {
    50: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 6},
        {"input_filters": 512, "num_repeats": 3},
    ],
    101: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 4},
        {"input_filters": 256, "num_repeats": 23},
        {"input_filters": 512, "num_repeats": 3},
    ],
    152: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 8},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    200: [
        {"input_filters": 64, "num_repeats": 3},
        {"input_filters": 128, "num_repeats": 24},
        {"input_filters": 256, "num_repeats": 36},
        {"input_filters": 512, "num_repeats": 3},
    ],
    270: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 29},
        {"input_filters": 256, "num_repeats": 53},
        {"input_filters": 512, "num_repeats": 4},
    ],
    350: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 36},
        {"input_filters": 256, "num_repeats": 72},
        {"input_filters": 512, "num_repeats": 4},
    ],
    420: [
        {"input_filters": 64, "num_repeats": 4},
        {"input_filters": 128, "num_repeats": 44},
        {"input_filters": 256, "num_repeats": 87},
        {"input_filters": 512, "num_repeats": 4},
    ],
}


def _allow_bigger_recursion(target_limit: int):
    """Increase default recursion limit to create larger models."""
    current_limit = sys.getrecursionlimit()
    if current_limit < target_limit:
        sys.setrecursionlimit(target_limit)


def fixed_padding(inputs, kernel_size):
    """Pad the input along the spatial dimensions independently of input size.

    This function is copied from original repo:
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
    if tf.keras.backend.image_data_format() == "channels_first":
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        )
    else:
        padded_inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )

    return padded_inputs


def Conv2DFixedPadding(filters, kernel_size, strides, name):
    """Conv2D block with fixed padding.

    This is a rewrite of original block:
    https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/resnet/resnet_model.py#L385
    """

    def apply(inputs):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same" if strides == 1 else "valid",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            name=name,
        )(inputs)

    return apply


def STEM(bn_momentum: float, bn_epsilon: float, activation: str):
    """ResNet-D type STEM block."""

    def apply(inputs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

        # First stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=2, name="stem_conv_1"
        )(inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name="stem_batch_norm_1",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_1")(x)

        # Second stem block
        x = Conv2DFixedPadding(
            filters=32, kernel_size=3, strides=1, name="stem_conv_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name="stem_batch_norm_2",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_2")(x)

        # Final Stem block:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=1, name="stem_conv_3"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name="stem_batch_norm_3",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_3")(x)

        # Replace stem max pool:
        x = Conv2DFixedPadding(
            filters=64, kernel_size=3, strides=2, name="stem_conv_4"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name="stem_batch_norm_4",
        )(x)
        x = tf.keras.layers.Activation(activation, name="stem_act_4")(x)
        return x

    return apply


def BottleneckBlock(
    filters: int,
    strides: int,
    use_projection: bool,
    bn_momentum: float,
    bn_epsilon: float,
    activation: str,
    se_ratio: float,
    name: str = "",
):
    """Bottleneck block variant for residual networks with BN after convolutions."""

    def apply(inputs):
        bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

        shortcut = inputs

        if use_projection:
            filters_out = filters * 4
            if strides == 2:
                shortcut = tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding="same",
                    name=name + "projection_pooling",
                )(inputs)
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=1,
                    name=name + "projection_conv",
                )(shortcut)
            else:
                shortcut = Conv2DFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=strides,
                    name=name + "projection_conv",
                )(inputs)

            shortcut = tf.keras.layers.BatchNormalization(
                axis=bn_axis,
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                name=name + "projection_batch_norm",
            )(shortcut)

        # TODO: DROPBLOCK IS USED HERE
        # https://github.com/tensorflow/tpu/blob/298d1fa98638f302ab9df34d9d26bbded7220e8b/models/official/resnet/resnet_model.py#L549

        # First conv layer:
        inputs = Conv2DFixedPadding(
            filters=filters, kernel_size=1, strides=1, name=name + "conv_1"
        )(inputs)
        inputs = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_1",
        )(inputs)
        inputs = tf.keras.layers.Activation(activation, name=name + "act_1")(inputs)

        # Second conv layer:
        inputs = Conv2DFixedPadding(
            filters=filters, kernel_size=3, strides=strides, name=name + "conv_2"
        )(inputs)
        inputs = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_2",
        )(inputs)
        inputs = tf.keras.layers.Activation(activation, name=name + "act_2")(inputs)

        # Third conv layer:
        inputs = Conv2DFixedPadding(
            filters=filters * 4, kernel_size=1, strides=1, name=name + "conv_3"
        )(inputs)
        inputs = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_3",
        )(inputs)

        # TODO: SE block as a separate layer.
        if 0 < se_ratio < 1:
            num_reduced_filters = max(1, int(filters * 4 * se_ratio))

            if tf.keras.backend.image_data_format() == "channels_first":
                spatial_dims = [2, 3]
            else:
                spatial_dims = [1, 2]
            se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)

            # TODO: Global average Pooling instead?
            # se_tensor = tf.keras.layers.GlobalAveragePooling2D(
            # name=name + "se_squeeze")(inputs)
            # if bn_axis == 1:
            #     se_shape = (filters, 1, 1)
            # else:
            #     se_shape = (1, 1, filters)
            # se_tensor = tf.keras.layers.Reshape(
            # (1, 1, -1), name=name + "se_reshape")(se_tensor)

            se_tensor = tf.keras.layers.Conv2D(
                filters=num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                padding="same",
                use_bias=True,
                activation="relu",
                name=name + "se_reduce",
            )(se_tensor)

            se_tensor = tf.keras.layers.Conv2D(
                filters=4 * filters,  # Expand ratio is 1 by default
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                padding="same",
                use_bias=True,
                activation="sigmoid",
                name=name + "se_expand",
            )(se_tensor)

            inputs = tf.keras.layers.multiply(
                [inputs, se_tensor], name=name + "se_excite"
            )

        inputs = tf.keras.layers.Add()([inputs, shortcut])

        return tf.keras.layers.Activation(activation, name=name + "output_act")(inputs)

    return apply


def BlockGroup(
    filters,
    strides,
    se_ratio,
    bn_epsilon,
    bn_momentum,
    num_repeats,
    activation,
    name: str,
):
    """Create one group of blocks for the ResNet model."""

    def apply(inputs):
        # Only the first block per block_group uses projection shortcut and strides.
        x = BottleneckBlock(
            filters=filters,
            strides=strides,
            use_projection=True,
            se_ratio=se_ratio,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
            activation=activation,
            name=name + "block_0_",
        )(inputs)

        for i in range(1, num_repeats):
            x = BottleneckBlock(
                filters=filters,
                strides=1,
                use_projection=False,
                se_ratio=se_ratio,
                activation=activation,
                bn_epsilon=bn_epsilon,
                bn_momentum=bn_momentum,
                name=name + f"block_{i}_",
            )(x)
        return x

    return apply


def ResNetRS(
    depth: int,
    input_shape=(None, None, 3),  # TODO: make this more, keras like.
    bn_momentum=0,  # Todo not 0.9?
    bn_epsilon=1e-5,
    activation: str = "relu",
    se_ratio=0.25,
    dropout_rate=0.25,  # TODO: check dropblock and stochastic depth
    include_top=True,
):
    """Build Resnet-RS model.

    TODO: double transpose trick?
    TODO: extend this docstring.
    TODO: consider (optional) preprocessing layer.
    """
    inputs = tf.keras.Input(input_shape)

    x = STEM(bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, activation=activation)(
        inputs
    )

    for i, args in enumerate(BLOCK_ARGS[depth]):
        x = BlockGroup(
            filters=args["input_filters"],
            activation=activation,
            strides=(1 if i == 0 else 2),
            num_repeats=args["num_repeats"],
            se_ratio=se_ratio,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            name=f"c{i + 2}_",
        )(x)

    # Build HEAD:
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000, name="predictions")(x)

    return tf.keras.Model(inputs=[inputs], outputs=[x])


def ResNetRS50():
    """Build ResNet-RS50 model."""
    return ResNetRS(depth=50)


def ResNetRS101():
    """Build ResNet-RS101 model."""
    return ResNetRS(depth=101)


def ResNetRS152():
    """Build ResNet-RS152 model."""
    return ResNetRS(depth=152)


def ResNetRS200():
    """Build ResNet-RS200 model."""
    return ResNetRS(depth=200)


def ResNetRS270():
    """Build ResNet-RS-270 model."""
    _allow_bigger_recursion(1100)
    return ResNetRS(depth=270)


def ResNetRS350():
    """Build ResNet-RS350 model."""
    _allow_bigger_recursion(1500)
    return ResNetRS(depth=350)


def ResNetRS420():
    """Build ResNet-RS420 model."""
    _allow_bigger_recursion(1700)
    return ResNetRS(depth=420)
