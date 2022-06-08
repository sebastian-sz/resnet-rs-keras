from typing import Callable

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_resnet_rs import utils
from test_resnet_rs.test_model import TEST_PARAMS

# Some tests are RAM hungry and will crash CI on smaller machines. We skip those
# tests, not to break entire CI job.
MODEL_TO_MIN_MEMORY = {  # in GB
    "50": 3.5,
    "101": 3.5,
    "152": 5,
    "200": 6,
    "270": 6.5,
    "350": 7.5,
    "420": 8.5,
}


class TestWeightClusteringWrappers(parameterized.TestCase):
    input_shape = (224, 224, 3)
    centroid_initialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        "number_of_clusters": 3,
        "cluster_centroids_init": centroid_initialization.DENSITY_BASED,
    }

    def setUp(self):
        tf.keras.backend.clear_session()

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_weight_clustering_wrap(self, model_fn: Callable):
        model_variant = self._testMethodName.split("-")[-1]
        minimum_required_ram = MODEL_TO_MIN_MEMORY[model_variant]
        if not utils.is_enough_memory(minimum_required_ram):
            self.skipTest(
                "Not enough memory to perform this test. Need at least "
                f"{minimum_required_ram} GB. Skipping... ."
            )

        model = model_fn(weights=None, input_shape=self.input_shape)
        tfmot.clustering.keras.cluster_weights(model, **self.clustering_params)


if __name__ == "__main__":
    absltest.main()
