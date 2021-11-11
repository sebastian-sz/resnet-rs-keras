from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_resnet_rs.test_model import TEST_PARAMS


class TestEfficientNetV2QATWrap(parameterized.TestCase):
    input_shape = (224, 224, 3)

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_qat_wrapping_entire_model(self, model_fn: Callable):
        try:
            model = model_fn(weights=None, input_shape=self.input_shape)
            tfmot.quantization.keras.quantize_model(model)
        except RuntimeError:
            self.skipTest(
                "The entire model cannot be wrapped in Quantization Aware Training."
                "tensorflow.python.keras.layers.merge.Multiply layer is not supported."
                "This test might succeed, once TF-MOT package will be updated."
                "For more info you can read: "
                "https://github.com/tensorflow/model-optimization/issues/733"
            )


if __name__ == "__main__":
    absltest.main()
