import tensorflow as tf
from absl import app, flags
from external.imagenet_input import ImageNetInput

from resnet_rs import (
    ResNetRS50,
    ResNetRS101,
    ResNetRS152,
    ResNetRS200,
    ResNetRS270,
    ResNetRS350,
    ResNetRS420,
    get_preprocessing_layer,
)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "depth",
    default="50",
    enum_values=["50", "101", "152", "200", "270", "350", "420"],
    help="Resnet model depth.",
)
flags.DEFINE_integer("image_size", default=224, help="Image size used for eval.")
flags.DEFINE_string(
    "weights", default="imagenet", help="Path to weights or pretrained weights variant."
)
flags.DEFINE_string(
    "data_dir",
    default="/workspace/tfrecords/validation",
    help="Path to validation tfrecords.",
)
flags.DEFINE_integer("batch_size", default=16, help="Batch size for eval.")


VARIANT_TO_MODEL = {
    50: ResNetRS50,
    101: ResNetRS101,
    152: ResNetRS152,
    200: ResNetRS200,
    270: ResNetRS270,
    350: ResNetRS350,
    420: ResNetRS420,
}


def main(argv_):
    """Run Imagenet eval job."""
    FLAGS.depth = int(FLAGS.depth)

    # Load model:
    model = VARIANT_TO_MODEL[FLAGS.depth](
        weights=FLAGS.weights, input_shape=(FLAGS.image_size, FLAGS.image_size, 3)
    )

    # Load data
    params = {"batch_size": FLAGS.batch_size}
    val_dataset = ImageNetInput(
        data_dir=FLAGS.data_dir,
        is_training=False,
        image_size=FLAGS.image_size,
        transpose_input=False,
        use_bfloat16=False,
    ).input_fn(params=params)

    layer = get_preprocessing_layer()

    val_dataset = val_dataset.map(
        lambda img, label: (layer(img), tf.one_hot(label, 1000)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Run eval:
    top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1")
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    progbar = tf.keras.utils.Progbar(target=50000 // FLAGS.batch_size)

    for idx, (images, y_true) in enumerate(val_dataset):
        y_pred = model(images, training=False)

        top1.update_state(y_true=y_true, y_pred=y_pred)
        top5.update_state(y_true=y_true, y_pred=y_pred)

        progbar.update(
            idx, [("top1", top1.result().numpy()), ("top5", top5.result().numpy())]
        )

    print()
    print(f"TOP1: {top1.result().numpy()}.  TOP5: {top5.result().numpy()}")


if __name__ == "__main__":
    app.run(main)
