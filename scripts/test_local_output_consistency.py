import os

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from resnet_rs.resnet_rs import (
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
    {  # OK
        "testcase_name": "50-i160",
        "model_fn": ResNetRS50,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-50-i160.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs50_i160_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "101-i160",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-101-i160.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs101_i160_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "101-i192",
        "model_fn": ResNetRS101,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-101-i192.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs101_i192_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "152-i192",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-152-i192.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs152_i192_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "152-i224",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-152-i224.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs152_i224_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "152-i256",
        "model_fn": ResNetRS152,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-152-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs152_i256_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "200-i256",
        "model_fn": ResNetRS200,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-200-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs200_i256_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "270-i256",
        "model_fn": ResNetRS270,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-270-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs270_i256_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "350-i256",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-350-i256.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs350_i256_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "350-i320",
        "model_fn": ResNetRS350,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-350-i320.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs350_i320_original_logits.npy",
        ),
    },
    {  # OK
        "testcase_name": "420-i320",
        "model_fn": ResNetRS420,
        "weights_path": os.path.join(ROOT_DIR, "weights/resnetrs-420-i320.h5"),
        "original_outputs": os.path.join(
            ROOT_DIR,
            "tests/assets/original_outputs/resnetrs420_i320_original_logits.npy",
        ),
    },
]


class TestLocalOutputConsistency(parameterized.TestCase):
    IMAGE_PATH = os.path.join(ROOT_DIR, "tests/assets/panda.jpg")
    INPUT_SHAPE = (224, 224)
    CROP_PADDING = 32
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @parameterized.named_parameters(OUTPUT_CONSISTENCY_TEST_PARAMS)
    def test_output_consistency(self, model_fn, weights_path, original_outputs):
        model = model_fn()
        model.load_weights(weights_path)

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


if __name__ == "__main__":
    absltest.main()
