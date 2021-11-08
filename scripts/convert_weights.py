import tensorflow as tf
from absl import app, flags, logging

from resnet_rs.resnet_rs_model import (
    BLOCK_ARGS,
    ResNetRS50,
    ResNetRS101,
    ResNetRS152,
    ResNetRS200,
    ResNetRS270,
    ResNetRS350,
    ResNetRS420,
)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "depth",
    enum_values=["50", "101", "152", "200", "270", "350", "420"],
    default="50",
    help="Which resnet to convert.",
)
flags.DEFINE_string("ckpt", default="", help="Path to directory with ckpt files.")
flags.DEFINE_string("output", default="", help="How to name converted .h5 weights.")
flags.DEFINE_bool(
    "include_top", default=True, help="Whether to include_top in model creation."
)
flags.DEFINE_bool(
    "use_ema",
    default=False,
    help="Whether to export Exponential moving average variables.",
)
flags.DEFINE_bool(
    "use_momentum", default=False, help="Whether to export Momentum variables."
)

DEPTH_TO_MODEL = {
    50: ResNetRS50,
    101: ResNetRS101,
    152: ResNetRS152,
    200: ResNetRS200,
    270: ResNetRS270,
    350: ResNetRS350,
    420: ResNetRS420,
}


def main(argv_):
    """Convert tpu checkpoint files to Keras .h5 weights."""
    if FLAGS.use_ema and FLAGS.use_momentum:
        raise ValueError(
            "Received both EMA and Momentum True. "
            "Please use either EMA or Momentum or leave both off."
        )
    depth = int(FLAGS.depth)
    model = DEPTH_TO_MODEL[depth](
        include_top=FLAGS.include_top, weights=None, input_shape=(None, None, 3)
    )
    block_args = BLOCK_ARGS[depth]

    logging.info("Creating variable mapping...")
    variable_names_map = make_variables_map(
        model_variables=model.variables, block_args=block_args
    )

    logging.info("Starting conversion...")
    progbar = tf.keras.utils.Progbar(target=len(model.variables))
    for v in model.variables:
        tf_variable_name = variable_names_map[v.name]
        if FLAGS.use_ema:
            tf_variable_name += "/ExponentialMovingAverage"
        elif FLAGS.use_momentum:
            if not tf_variable_name.endswith(("moving_mean", "moving_variance")):
                tf_variable_name += "/Momentum"

        original_variable = tf.train.load_variable(FLAGS.ckpt, tf_variable_name)
        v.assign(original_variable)
        progbar.add(1)

    model.save_weights(FLAGS.output, save_format="h5")
    logging.info(f"OK. Weights saved at: {FLAGS.output}")


def make_variables_map(model_variables, block_args):
    """Map each model variable to ckpt variable."""
    sorted_block_names = _get_sorted_block_names(
        model_variables=model_variables, block_args=block_args
    )

    variables_map = {}
    variables_map.update(_make_stem_map())
    variables_map.update(_make_head_map())
    variables_map.update(_make_block_map(sorted_block_names))
    return variables_map


def _get_sorted_block_names(model_variables, block_args):
    """Hardcoded block order because it is the same across models."""
    block_types = sorted(
        list({x.name.split("_")[0] for x in model_variables if "block" in x.name})
    )
    num_repeats = [x["num_repeats"] for x in block_args]

    sorted_block_names = []
    for block, repeats in zip(block_types, num_repeats):
        for i in range(repeats):
            if i == 0:
                sorted_block_names.append(f"{block}_block_{i}_projection_conv")
                sorted_block_names.append(f"{block}_block_{i}_projection_batch_norm")
            sorted_block_names.append(f"{block}_block_{i}_conv_1")
            sorted_block_names.append(f"{block}_block_{i}_batch_norm_1")
            sorted_block_names.append(f"{block}_block_{i}_conv_2")
            sorted_block_names.append(f"{block}_block_{i}_batch_norm_2")
            sorted_block_names.append(f"{block}_block_{i}_conv_3")
            sorted_block_names.append(f"{block}_block_{i}_batch_norm_3")
            sorted_block_names.append(f"{block}_block_{i}_se_reduce")
            sorted_block_names.append(f"{block}_block_{i}_se_expand")

    return sorted_block_names


def _make_stem_map():
    """Hardcoded STEM variables mapping."""
    return {
        "stem_conv_1/kernel:0": "conv2d/kernel",
        "stem_batch_norm_1/gamma:0": "batch_normalization/gamma",
        "stem_batch_norm_1/beta:0": "batch_normalization/beta",
        "stem_batch_norm_1/moving_mean:0": "batch_normalization/moving_mean",
        "stem_batch_norm_1/moving_variance:0": "batch_normalization/moving_variance",
        "stem_conv_2/kernel:0": "conv2d_1/kernel",
        "stem_batch_norm_2/gamma:0": "batch_normalization_1/gamma",
        "stem_batch_norm_2/beta:0": "batch_normalization_1/beta",
        "stem_batch_norm_2/moving_mean:0": "batch_normalization_1/moving_mean",
        "stem_batch_norm_2/moving_variance:0": "batch_normalization_1/moving_variance",
        "stem_conv_3/kernel:0": "conv2d_2/kernel",
        "stem_batch_norm_3/gamma:0": "batch_normalization_2/gamma",
        "stem_batch_norm_3/beta:0": "batch_normalization_2/beta",
        "stem_batch_norm_3/moving_mean:0": "batch_normalization_2/moving_mean",
        "stem_batch_norm_3/moving_variance:0": "batch_normalization_2/moving_variance",
        "stem_conv_4/kernel:0": "conv2d_3/kernel",
        "stem_batch_norm_4/gamma:0": "batch_normalization_3/gamma",
        "stem_batch_norm_4/beta:0": "batch_normalization_3/beta",
        "stem_batch_norm_4/moving_mean:0": "batch_normalization_3/moving_mean",
        "stem_batch_norm_4/moving_variance:0": "batch_normalization_3/moving_variance",
    }


def _make_head_map():
    """Hardcoded Head variables mapping."""
    return {"predictions/kernel:0": "dense/kernel", "predictions/bias:0": "dense/bias"}


def _make_block_map(keras_block_names):
    """Create block variables mapping."""
    last_conv_number = 4
    last_bn_number = 4

    block_map = {}

    for block_name in keras_block_names:

        if "conv" in block_name:
            block_map.update(
                {f"{block_name}/kernel:0": f"conv2d_{last_conv_number}/kernel"}
            )
            last_conv_number += 1

        elif "se" in block_name:
            block_map.update(
                {f"{block_name}/kernel:0": f"conv2d_{last_conv_number}/kernel"}
            )
            block_map.update(
                {f"{block_name}/bias:0": f"conv2d_{last_conv_number}/bias"}
            )
            last_conv_number += 1

        elif "batch_norm" in block_name:
            block_map.update(
                {f"{block_name}/gamma:0": f"batch_normalization_{last_bn_number}/gamma"}
            )
            block_map.update(
                {f"{block_name}/beta:0": f"batch_normalization_{last_bn_number}/beta"}
            )
            block_map.update(
                {
                    f"{block_name}/moving_mean:0": f"batch_normalization_"
                    f"{last_bn_number}/moving_mean"
                }
            )
            block_map.update(
                {
                    f"{block_name}/moving_variance:0": f"batch_normalization_"
                    f"{last_bn_number}/moving_variance"
                }
            )

            last_bn_number += 1

    return block_map


if __name__ == "__main__":
    app.run(main)
