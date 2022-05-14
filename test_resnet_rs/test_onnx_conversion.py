import os
import tempfile
from typing import Callable

import onnxruntime
import tensorflow as tf
import tf2onnx
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from test_resnet_rs.test_model import TEST_PARAMS
from test_resnet_rs.utils import get_inference_function

# Some conversions are RAM hungry and will crash CI on smaller machines. We skip those
# tests, not to break entire CI job.
MODEL_TO_MIN_MEMORY = {  # in GB
    "50": 5,
    "101": 6.5,
    "152": 7.5,
    "200": 7.5,
    "270": 8.5,
    "350": 11.5,
    "420": 14,
}


class TestONNXConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    onnx_model_path = os.path.join(tempfile.mkdtemp(), "model.onnx")
    input_shape = (224, 224, 3)

    _tolerance = 1e-4

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self) -> None:
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_onnx_conversion(self, model_fn: Callable):
        # Skip test if not enough RAM:
        model_variant = self._testMethodName.split("-")[-1]
        if not self._enough_memory_to_convert(model_variant):
            self.skipTest(
                "Not enough memory to convert to onnx. Need at least "
                f"{MODEL_TO_MIN_MEMORY[model_variant]} GB. Skipping... ."
            )

        model = model_fn(
            weights="imagenet", input_shape=self.input_shape, classifier_activation=None
        )

        inference_func = get_inference_function(model, self.input_shape[:2])
        self._convert_onnx(inference_func)

        self.assertTrue(os.path.isfile(self.onnx_model_path))

        # Compare outputs:
        mock_input = self.rng.uniform(shape=(1, *self.input_shape), dtype=tf.float32)
        original_output = model(mock_input, training=False)

        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: mock_input.numpy()}
        onnx_output = onnx_session.run(None, onnx_inputs)

        tf.debugging.assert_near(
            original_output, onnx_output, atol=self._tolerance, rtol=self._tolerance
        )

    @staticmethod
    def _enough_memory_to_convert(model_name: str) -> bool:
        total_ram = virtual_memory().total / (1024.0**3)
        required_ram = MODEL_TO_MIN_MEMORY[model_name]
        return total_ram >= required_ram

    def _convert_onnx(self, inference_func):
        model_proto, _ = tf2onnx.convert.from_function(
            inference_func,
            output_path=self.onnx_model_path,
            input_signature=inference_func.input_signature,
        )
        return model_proto


if __name__ == "__main__":
    absltest.main()
