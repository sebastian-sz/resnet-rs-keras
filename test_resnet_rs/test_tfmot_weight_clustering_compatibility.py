from typing import Callable

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from test_resnet_rs.test_model import TEST_PARAMS


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
        model = model_fn(weights=None, input_shape=self.input_shape)
        tfmot.clustering.keras.cluster_weights(model, **self.clustering_params)


if __name__ == "__main__":
    absltest.main()
