import tensorflow as tf
from absl.testing import absltest

from resnet_rs.preprocessing_layer import get_preprocessing_layer


class PreprocessingLayerTest(absltest.TestCase):
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    rng = tf.random.Generator.from_non_deterministic_state()

    def test_preprocessing_layer_correctly_rescaling_inputs(self):
        mock_frame = self.rng.uniform((1, 224, 224, 3), maxval=255, dtype=tf.float32)

        layer = get_preprocessing_layer()

        expected_output = self._original_preprocessing(mock_frame)
        layer_output = layer(mock_frame)

        tf.debugging.assert_near(expected_output, layer_output)

    def _original_preprocessing(self, image):
        # https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/resnet/resnet_main.py#L370
        image -= tf.constant(self.MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(self.STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        return image


if __name__ == "__main__":
    absltest.main()
