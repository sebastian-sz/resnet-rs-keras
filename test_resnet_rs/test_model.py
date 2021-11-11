"""Unit tests for created models."""
import os
import tempfile
from typing import Callable

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

TEST_PARAMS = [
    {"testcase_name": "rs-50", "model_fn": ResNetRS50},
    {"testcase_name": "rs-101", "model_fn": ResNetRS101},
    {"testcase_name": "rs-152", "model_fn": ResNetRS152},
    {"testcase_name": "rs-200", "model_fn": ResNetRS200},
    {"testcase_name": "rs-270", "model_fn": ResNetRS270},
    {"testcase_name": "rs-350", "model_fn": ResNetRS350},
    {"testcase_name": "rs-420", "model_fn": ResNetRS420},
]


class TestResNetRSModels(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    model_path = os.path.join(tempfile.mkdtemp(), "model.h5")

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_inference(self, model_fn: Callable):
        mock_input = self.rng.uniform((1, 224, 224, 3))

        model = model_fn()
        output = model(mock_input, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertEqual(output.shape, (1, 1000))

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_serialization(self, model_fn: Callable):
        model = model_fn()
        tf.keras.models.save_model(
            model=model, filepath=self.model_path, save_format="h5"
        )

        self.assertTrue(os.path.exists(self.model_path))

        loaded = tf.keras.models.load_model(self.model_path)
        self.assertTrue(isinstance(loaded, tf.keras.Model))

    def test_model_cannot_be_created_with_invalid_weighs_parameter(self):
        invalid_parameters = {"weights": "imagenet-i1920"}

        with self.assertRaises(ValueError):
            ResNetRS50(**invalid_parameters)

    def test_model_cannot_be_created_with_invalid_number_of_classes(self):
        invalid_parameters = {"classes": 1, "include_top": True}

        with self.assertRaises(ValueError):
            ResNetRS50(**invalid_parameters)

    def test_pretrained_model_can_be_created_with_different_input_shape(self):
        input_shape = (160, 160, 3)
        parameters = {
            "weights": "imagenet",
            "include_top": True,
            "input_shape": input_shape,
        }
        mock_input = self.rng.uniform((1, *input_shape))

        model = ResNetRS50(**parameters)
        output = model(mock_input, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertEqual(output.shape, (1, 1000))

    def test_model_can_be_created_with_unspecified_input_shape(self):
        input_shape = (None, None, 3)
        parameters = {
            "weights": "imagenet",
            "include_top": True,
            "input_shape": input_shape,
        }
        mock_input = self.rng.uniform((1, 224, 224, 3))

        model = ResNetRS50(**parameters)
        output = model(mock_input, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertEqual(output.shape, (1, 1000))

    def tearDown(self) -> None:
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def setUp(self):
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    absltest.main()
