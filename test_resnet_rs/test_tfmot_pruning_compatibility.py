from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_resnet_rs.test_model import TEST_PARAMS


class TestEfficientNetV2PruningWrapper(parameterized.TestCase):
    input_shape = (224, 224, 3)

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_pruning_entire_model(self, model_fn: Callable):
        model = model_fn(weights=None, input_shape=self.input_shape)
        tfmot.sparsity.keras.prune_low_magnitude(model)


if __name__ == "__main__":
    absltest.main()
