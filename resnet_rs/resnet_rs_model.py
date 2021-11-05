"""Architecture code for Resnet RS models."""

import tensorflow as tf

# TODO: figure out dropblock - where was it used and how
from resnet_rs.block_args import BLOCK_ARGS
from resnet_rs.model_utils import (
    allow_bigger_recursion,
    fixed_padding,
    get_survival_probability,
)


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
    survival_probability: float,
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

        # Drop connect
        if survival_probability:
            inputs = tf.keras.layers.Dropout(
                survival_probability, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(inputs)

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
    survival_probability: float,
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
            survival_probability=survival_probability,
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
                survival_probability=survival_probability,
                name=name + f"block_{i}_",
            )(x)
        return x

    return apply


def ResNetRS(
    depth: int,
    input_shape=(None, None, 3),  # TODO: make this more, keras like.
    bn_momentum=0,  # TODO; get from configs
    bn_epsilon=1e-5,  # TODO; get from configs
    activation: str = "relu",
    se_ratio=0.25,  # TODO; get from configs
    dropout_rate=0.25,  # TODO; get from configs
    drop_connect_rate=0.2,
    include_top=True,
):
    """Build Resnet-RS model.

    TODO: double transpose trick?
    TODO: extend this docstring.
    TODO: Dropblock layer


    """
    inputs = tf.keras.Input(input_shape)

    x = STEM(bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, activation=activation)(
        inputs
    )

    block_args = BLOCK_ARGS[depth]

    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(
            init_rate=drop_connect_rate,
            block_num=i + 2,
            total_blocks=len(block_args) + 1,
        )

        x = BlockGroup(
            filters=args["input_filters"],
            activation=activation,
            strides=(1 if i == 0 else 2),
            num_repeats=args["num_repeats"],
            se_ratio=se_ratio,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            survival_probability=survival_probability,
            name=f"c{i + 2}_",
        )(x)

    # Build HEAD:
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1000, name="predictions")(x)

    return tf.keras.Model(inputs=[inputs], outputs=[x])


def ResNetRS50(include_top=True):
    """Build ResNet-RS50 model."""
    return ResNetRS(depth=50, include_top=include_top, drop_connect_rate=0.0)


def ResNetRS101(include_top=True):
    """Build ResNet-RS101 model."""
    return ResNetRS(depth=101, include_top=include_top, drop_connect_rate=0.0)


def ResNetRS152(include_top=True):
    """Build ResNet-RS152 model."""
    return ResNetRS(depth=152, include_top=include_top, drop_connect_rate=0.0)


def ResNetRS200(include_top=True):
    """Build ResNet-RS200 model."""
    return ResNetRS(depth=200, include_top=include_top, drop_connect_rate=0.1)


def ResNetRS270(include_top=True):
    """Build ResNet-RS-270 model."""
    allow_bigger_recursion(1300)
    return ResNetRS(depth=270, include_top=include_top, drop_connect_rate=0.1)


def ResNetRS350(include_top=True):
    """Build ResNet-RS350 model."""
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350,
        include_top=include_top,
        dropout_rate=0.25,  # or 0.4
        drop_connect_rate=0.1,
    )


def ResNetRS420(include_top=True):
    """Build ResNet-RS420 model."""
    allow_bigger_recursion(1800)
    return ResNetRS(
        depth=420, include_top=include_top, dropout_rate=0.4, drop_connect_rate=0.1
    )


if __name__ == "__main__":
    model = ResNetRS200()
