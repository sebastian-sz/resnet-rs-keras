"""Architecture code for Resnet RS models."""
from typing import Callable, Dict, List, Union

import tensorflow as tf
from absl import logging
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from resnet_rs.block_args import BLOCK_ARGS
from resnet_rs.model_utils import (
    allow_bigger_recursion,
    fixed_padding,
    get_survival_probability,
)

BASE_WEIGHTS_URL = (
    "https://github.com/sebastian-sz/resnet-rs-keras/releases/download/v1.0/"
)

WEIGHT_HASHES = {
    "resnet-rs-101-i160.h5": "544b3434d00efc199d66e9058c7f3379",
    "resnet-rs-101-i160_notop.h5": "82d5b90c5ce9d710da639d6216d0f979",
    "resnet-rs-101-i192.h5": "eb285be29ab42cf4835ff20a5e3b5d23",
    "resnet-rs-101-i192_notop.h5": "f9a0f6b85faa9c3db2b6e233c4eebb5b",
    "resnet-rs-152-i192.h5": "8d72a301ed8a6f11a47c4ced4396e338",
    "resnet-rs-152-i192_notop.h5": "5fbf7ac2155cb4d5a6180ee9e3aa8704",
    "resnet-rs-152-i224.h5": "31a46a92ab21b84193d0d71dd8c3d03b",
    "resnet-rs-152-i224_notop.h5": "dc8b2cba2005552eafa3167f00dc2133",
    "resnet-rs-152-i256.h5": "ba6271b99bdeb4e7a9b15c05964ef4ad",
    "resnet-rs-152-i256_notop.h5": "fa79794252dbe47c89130f65349d654a",
    "resnet-rs-200-i256.h5": "a76930b741884e09ce90fa7450747d5f",
    "resnet-rs-200-i256_notop.h5": "bbdb3994718dfc0d1cd45d7eff3f3d9c",
    "resnet-rs-270-i256.h5": "20d575825ba26176b03cb51012a367a8",
    "resnet-rs-270-i256_notop.h5": "2c42ecb22e35f3e23d2f70babce0a2aa",
    "resnet-rs-350-i256.h5": "f4a039dc3c421321b7fc240494574a68",
    "resnet-rs-350-i256_notop.h5": "6e44b55025bbdff8f51692a023143d66",
    "resnet-rs-350-i320.h5": "7ccb858cc738305e8ceb3c0140bee393",
    "resnet-rs-350-i320_notop.h5": "ab0c1f9079d2f85a9facbd2c88aa6079",
    "resnet-rs-420-i320.h5": "ae0eb9bed39e64fc8d7e0db4018dc7e8",
    "resnet-rs-420-i320_notop.h5": "fe6217c32be8305b1889657172b98884",
    "resnet-rs-50-i160.h5": "69d9d925319f00a8bdd4af23c04e4102",
    "resnet-rs-50-i160_notop.h5": "90daa68cd26c95aa6c5d25451e095529",
}

DEPTH_TO_WEIGHT_VARIANTS = {
    50: [160],
    101: [160, 192],
    152: [192, 224, 256],
    200: [256],
    270: [256],
    350: [256, 320],
    420: [320],
}


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


def SE(in_filters: int, se_ratio: float, name: str, expand_ratio: int = 1):
    """Squeeze and Excitation block."""

    def apply(inputs):
        if tf.keras.backend.image_data_format() == "channels_first":
            spatial_dims = [2, 3]
        else:
            spatial_dims = [1, 2]
        se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)

        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))

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
            filters=4 * in_filters * expand_ratio,  # Expand ratio is 1 by default
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            padding="same",
            use_bias=True,
            activation="sigmoid",
            name=name + "se_expand",
        )(se_tensor)

        return tf.keras.layers.multiply([inputs, se_tensor], name=name + "se_excite")

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

        # First conv layer:
        x = Conv2DFixedPadding(
            filters=filters, kernel_size=1, strides=1, name=name + "conv_1"
        )(inputs)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_1",
        )(x)
        x = tf.keras.layers.Activation(activation, name=name + "act_1")(x)

        # Second conv layer:
        x = Conv2DFixedPadding(
            filters=filters, kernel_size=3, strides=strides, name=name + "conv_2"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_2",
        )(x)
        x = tf.keras.layers.Activation(activation, name=name + "act_2")(x)

        # Third conv layer:
        x = Conv2DFixedPadding(
            filters=filters * 4, kernel_size=1, strides=1, name=name + "conv_3"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name=name + "batch_norm_3",
        )(x)

        if 0 < se_ratio < 1:
            x = SE(filters, se_ratio=se_ratio, name=name)(x)

        # Drop connect
        if survival_probability:
            x = tf.keras.layers.Dropout(
                survival_probability, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)

        x = tf.keras.layers.Add()([x, shortcut])

        return tf.keras.layers.Activation(activation, name=name + "output_act")(x)

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
    input_shape=(None, None, 3),
    bn_momentum=0,
    bn_epsilon=1e-5,
    activation: str = "relu",
    se_ratio=0.25,
    dropout_rate=0.25,
    drop_connect_rate=0.2,
    include_top=True,
    block_args: List[Dict[str, int]] = None,
    model_name="resnet-rs",
    pooling=None,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    classifier_activation: Union[str, Callable] = "softmax",
):
    """Build Resnet-RS model, given provided parameters.

    Parameters:
        :param depth: Depth of ResNet network.
        :param dropout_rate: dropout rate before final classifier layer.
        :param bn_momentum: Momentum parameter for Batch Normalization layers.
        :param bn_epsilon: Epsilon parameter for Batch Normalization layers.
        :param activation: activation function.
        :param block_args: list of dicts, parameters to construct block modules.
        :param se_ratio: Squeeze and Excitation layer ratio.
        :param model_name: name of the model.
        :param drop_connect_rate: dropout rate at skip connections.
        :param include_top: whether to include the fully-connected layer at the top of
        the network.
        :param weights: one of `None` (random initialization), `'imagenet'`
            (pre-training on ImageNet), or the path to the weights file to be loaded.
            Note: one model can have multiple imagenet variants depending on
            input shape it was trained with. For input_shape 224x224 pass
            `imagenet-i224` as argument. By default, highest input shape weights are
            downloaded.
        :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
        :param input_shape: optional shape tuple. It should have exactly 3 inputs
            channels, and width and height should be no smaller than 32.
            E.g. (200, 200, 3) would be one valid value.
        :param  pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        :param classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        :param classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `tf.keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`, or invalid input
            shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    # Validate parameters
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == "imagenet":
        max_input_shape = max(available_weight_variants)
        logging.warning(
            f"Received `imagenet` argument without "
            f"explicit weights input size. Picking weights trained with "
            f"biggest available shape: imagenet-i{max_input_shape}"
        )
        weights = f"{weights}-i{max_input_shape}"

    weights_allow_list = [f"imagenet-i{x}" for x in available_weight_variants]
    if not (weights in {*weights_allow_list, None} or file_io.file_exists_v2(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `'imagenet'` "
            "(pre-training on ImageNet, with highest available input shape),"
            " or the path to the weights file to be loaded. "
            f"For ResNetRS{depth} the following weight variants are "
            f"available {weights_allow_list} (default=highest)."
            f" Received weights={weights}"
        )

    if weights in weights_allow_list and include_top and classes != 1000:
        raise ValueError(
            f"If using `weights` as `'imagenet'` or any of {weights_allow_list} with "
            f"`include_top` as true, `classes` should be 1000. "
            f"Received classes={classes}"
        )

    # Define input tensor
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Build stem
    x = STEM(bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, activation=activation)(
        img_input
    )

    # Build blocks
    if block_args is None:
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

    # Build head:
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Download weights
    if weights in weights_allow_list:
        weights_input_shape = weights.split("-")[-1]  # e. g. "i160"
        weights_name = f"{model_name}-{weights_input_shape}"
        if not include_top:
            weights_name += "_notop"

        filename = f"{weights_name}.h5"
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = tf.keras.utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir="models",
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNetRS50(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS50 model."""
    return ResNetRS(
        depth=50,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-50",
    )


def ResNetRS101(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS101 model."""
    return ResNetRS(
        depth=101,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-101",
    )


def ResNetRS152(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS152 model."""
    return ResNetRS(
        depth=152,
        include_top=include_top,
        drop_connect_rate=0.0,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-152",
    )


def ResNetRS200(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS200 model."""
    return ResNetRS(
        depth=200,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-200",
    )


def ResNetRS270(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS-270 model."""
    allow_bigger_recursion(1300)
    return ResNetRS(
        depth=270,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.25,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-270",
    )


def ResNetRS350(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS350 model."""
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350,
        include_top=include_top,
        drop_connect_rate=0.1,
        dropout_rate=0.4,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-350",
    )


def ResNetRS420(
    include_top=True,
    weights="imagenet",
    classes=1000,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
):
    """Build ResNet-RS420 model."""
    allow_bigger_recursion(1800)
    return ResNetRS(
        depth=420,
        include_top=include_top,
        dropout_rate=0.4,
        drop_connect_rate=0.1,
        weights=weights,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        model_name="resnet-rs-420",
    )
