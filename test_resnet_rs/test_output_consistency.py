import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from resnet_rs import (
    ResNetRS50,
    ResNetRS101,
    ResNetRS152,
    ResNetRS200,
    ResNetRS270,
    ResNetRS350,
    ResNetRS420,
)
from root_dir import ROOT_DIR

OUTPUT_CONSISTENCY_TEST_PARAMS = [
    {
        "testcase_name": "50-i160",
        "model_fn": ResNetRS50,
        "weights_arg": "imagenet-i160",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs50_i160_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "101-i160",
        "model_fn": ResNetRS101,
        "weights_arg": "imagenet-i160",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs101_i160_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "101-i192",
        "model_fn": ResNetRS101,
        "weights_arg": "imagenet-i192",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs101_i192_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i192",
        "model_fn": ResNetRS152,
        "weights_arg": "imagenet-i192",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i192_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i224",
        "model_fn": ResNetRS152,
        "weights_arg": "imagenet-i224",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i224_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i256",
        "model_fn": ResNetRS152,
        "weights_arg": "imagenet-i256",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "200-i256",
        "model_fn": ResNetRS200,
        "weights_arg": "imagenet-i256",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs200_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "270-i256",
        "model_fn": ResNetRS270,
        "weights_arg": "imagenet-i256",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs270_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "350-i256",
        "model_fn": ResNetRS350,
        "weights_arg": "imagenet-i256",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs350_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "350-i320",
        "model_fn": ResNetRS350,
        "weights_arg": "imagenet-i320",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs350_i320_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "420-i320",
        "model_fn": ResNetRS420,
        "weights_arg": "imagenet-i320",
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs420_i320_original_logits_ema.npy",
        ),
    },
]


class TestKerasVSOriginalOutputConsistency(parameterized.TestCase):
    image_path = os.path.join(ROOT_DIR, "test_resnet_rs/assets/panda.jpg")
    input_shape = (224, 224)
    crop_padding = 32
    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(OUTPUT_CONSISTENCY_TEST_PARAMS)
    def test_output_consistency(
        self, model_fn: Callable, weights_arg: str, original_outputs: str
    ):
        model = model_fn(weights=weights_arg, classifier_activation=None)

        input_tensor = self._load_and_preprocess_array()
        output = model(input_tensor, training=False)

        original_out = np.load(original_outputs)

        tf.debugging.assert_near(output, original_out)

    def _decode_and_center_crop(self, image_bytes, image_size):
        # https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/resnet/resnet_preprocessing.py#L109
        shape = tf.image.extract_jpeg_shape(image_bytes)
        image_height = shape[0]
        image_width = shape[1]

        padded_center_crop_size = tf.cast(
            (
                (image_size / (image_size + self.crop_padding))
                * tf.cast(tf.minimum(image_height, image_width), tf.float32)
            ),
            tf.int32,
        )

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        crop_window = tf.stack(
            [
                offset_height,
                offset_width,
                padded_center_crop_size,
                padded_center_crop_size,
            ]
        )
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
        image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
        return image

    def _load_and_preprocess_array(self):
        image_size = self.input_shape[0]  # height = width
        bytes = tf.io.read_file(self.image_path)
        image = self._decode_and_center_crop(bytes, image_size)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image -= tf.constant(self.mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(self.stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
        return tf.expand_dims(image, axis=0)


if __name__ == "__main__":
    absltest.main()
