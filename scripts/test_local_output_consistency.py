import os
from typing import Callable

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from resnet_rs.resnet_rs_model import (
    ResNetRS50,
    ResNetRS101,
    ResNetRS152,
    ResNetRS200,
    ResNetRS270,
    ResNetRS350,
    ResNetRS420,
)
from root_dir import ROOT_DIR

# TODO: refactor this

OUTPUT_CONSISTENCY_TEST_PARAMS = [
    {
        "testcase_name": "50-i160",
        "model_fn": ResNetRS50,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-50-i160.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs50_i160_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "101-i160",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-101-i160.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs101_i160_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "101-i192",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-101-i192.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs101_i192_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i192",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i192.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i192_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i224",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i224.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i224_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "152-i256",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs152_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "200-i256",
        "model_fn": ResNetRS200,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-200-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs200_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "270-i256",
        "model_fn": ResNetRS270,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-270-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs270_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "350-i256",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-350-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs350_i256_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "350-i320",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-350-i320.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs350_i320_original_logits_ema.npy",
        ),
    },
    {
        "testcase_name": "420-i320",
        "model_fn": ResNetRS420,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-420-i320.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "test_resnet_rs/assets/original_outputs/"
            "resnetrs420_i320_original_logits_ema.npy",
        ),
    },
]

FEATURE_EXTRACTION_TEST_PARAMS = [
    {
        "testcase_name": "fe-101-i160",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-101-i160_notop.h5"),
    },
    {
        "testcase_name": "fe-101-i192",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-101-i192_notop.h5"),
    },
    {
        "testcase_name": "fe-152-i192",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i192_notop.h5"),
    },
    {
        "testcase_name": "fe-152-i224",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i224_notop.h5"),
    },
    {
        "testcase_name": "fe-152-i256",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-152-i256_notop.h5"),
    },
    {
        "testcase_name": "fe-200-i256",
        "model_fn": ResNetRS200,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-200-i256_notop.h5"),
    },
    {
        "testcase_name": "fe-270-i256",
        "model_fn": ResNetRS270,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-270-i256_notop.h5"),
    },
    {
        "testcase_name": "fe-350-i256",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-350-i256_notop.h5"),
    },
    {
        "testcase_name": "fe-350-i320",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-350-i320_notop.h5"),
    },
    {
        "testcase_name": "fe-50-i160",
        "model_fn": ResNetRS50,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-50-i160_notop.h5"),
    },
    {
        "testcase_name": "fe-420-i320",
        "model_fn": ResNetRS420,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnet-rs-420-i320_notop.h5"),
    },
]


class TestLocalOutputConsistency(parameterized.TestCase):
    IMAGE_PATH = os.path.join(ROOT_DIR, "test_resnet_rs/assets/panda.jpg")
    INPUT_SHAPE = (224, 224)
    CROP_PADDING = 32
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    rng = tf.random.Generator.from_seed(1234)

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(OUTPUT_CONSISTENCY_TEST_PARAMS)
    def test_output_consistency(self, model_fn, weights_path, original_outputs):
        model = model_fn(weights=weights_path, classifier_activation=None)

        input_tensor = self._load_and_preprocess_array(self.INPUT_SHAPE[0])
        output = model(input_tensor, training=False)

        original_out = np.load(original_outputs)

        tf.debugging.assert_near(output, original_out)

    def _load_and_preprocess_array(self, image_size):
        bytes = tf.io.read_file(self.IMAGE_PATH)
        image = self._decode_and_center_crop(bytes, image_size)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image -= tf.constant(self.MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(self.STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        return tf.expand_dims(image, axis=0)

    def _decode_and_center_crop(self, image_bytes, image_size):
        """Crops to center of image with padding then scales image_size."""

        shape = tf.image.extract_jpeg_shape(image_bytes)
        image_height = shape[0]
        image_width = shape[1]

        padded_center_crop_size = tf.cast(
            (
                (image_size / (image_size + self.CROP_PADDING))
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

    @parameterized.named_parameters(FEATURE_EXTRACTION_TEST_PARAMS)
    def test_feature_extraction_with_converted_weights(
        self, model_fn: Callable, weights_path: str
    ):
        model = model_fn(include_top=False)
        model.load_weights(weights_path)

        mock_input = self.rng.uniform((1, 224, 224, 3))
        expected_output_shape = (1, 7, 7, 2048)

        output = model(mock_input, training=False)

        self.assertEqual(output.shape, expected_output_shape)


if __name__ == "__main__":
    absltest.main()
