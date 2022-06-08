import os
import tempfile
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from test_resnet_rs import utils
from test_resnet_rs.test_model import TEST_PARAMS

# Some conversions are RAM hungry and will crash CI on smaller machines. We skip those
# tests, not to break entire CI job.
MODEL_TO_MIN_MEMORY = {  # in GB
    "50": 5,
    "101": 6,
    "152": 7,
    "200": 7.5,
    "270": 10,
    "350": 12.5,
    "420": 14,
}


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")
    input_shape = (224, 224, 3)
    _tolerance = 1e-5

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable):
        # Skip test if not enough RAM:
        model_variant = self._testMethodName.split("-")[-1]
        minimum_required_ram = MODEL_TO_MIN_MEMORY[model_variant]
        if not utils.is_enough_memory(minimum_required_ram):
            self.skipTest(
                "Not enough memory to perform this test. Need at least "
                f"{minimum_required_ram} GB. Skipping... ."
            )

        model = model_fn(weights=None, input_shape=self.input_shape)
        self._convert_and_save_tflite(model, self.input_shape[:2])

        # Check outputs:
        dummy_inputs = self.rng.uniform(shape=(1, *self.input_shape))
        tflite_output = self._run_tflite_inference(dummy_inputs)

        self.assertTrue(isinstance(tflite_output, np.ndarray))
        self.assertEqual(tflite_output.shape, (1, 1000))

    def _convert_and_save_tflite(
        self, model: tf.keras.Model, input_shape: Tuple[int, int]
    ):
        inference_func = utils.get_inference_function(model, input_shape)
        concrete_func = inference_func.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        tflite_model = converter.convert()
        with open(self.tflite_path, "wb") as file:
            file.write(tflite_model)

    def _run_tflite_inference(self, inputs: tf.Tensor) -> np.ndarray:
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], inputs.numpy())
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]["index"])


if __name__ == "__main__":
    absltest.main()
