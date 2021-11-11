from typing import List

from absl.testing import absltest, parameterized

from resnet_rs.block_args import BLOCK_ARGS
from resnet_rs.model_utils import get_survival_probability

SURVIVAL_PROBABILITY_PARAMS = [
    {
        "testcase_name": "resnetrs-50",
        "depth": 50,
        "init_rate": 0.0,
        "expected_survival_probabilities": [0.0] * 4,
    },
    {
        "testcase_name": "resnetrs-101",
        "depth": 101,
        "init_rate": 0.0,
        "expected_survival_probabilities": [0.0] * 4,
    },
    {
        "testcase_name": "resnetrs-152",
        "depth": 152,
        "init_rate": 0.0,
        "expected_survival_probabilities": [0.0] * 4,
    },
    {
        "testcase_name": "resnetrs-200",
        "depth": 200,
        "init_rate": 0.1,
        "expected_survival_probabilities": [0.04, 0.06, 0.08, 0.1],
    },
    {
        "testcase_name": "resnetrs-270",
        "depth": 270,
        "init_rate": 0.1,
        "expected_survival_probabilities": [0.04, 0.06, 0.08, 0.1],
    },
    {
        "testcase_name": "resnetrs-350",
        "depth": 350,
        "init_rate": 0.1,
        "expected_survival_probabilities": [0.04, 0.06, 0.08, 0.1],
    },
    {
        "testcase_name": "resnetrs-420",
        "depth": 420,
        "init_rate": 0.1,
        "expected_survival_probabilities": [0.04, 0.06, 0.08, 0.1],
    },
]


class TestModelArgs(parameterized.TestCase):
    @parameterized.named_parameters(SURVIVAL_PROBABILITY_PARAMS)
    def test_correct_survival_probability_for_each_block_group(
        self, depth: int, init_rate: float, expected_survival_probabilities: List[int]
    ):
        block_args = BLOCK_ARGS[depth]
        num_layers = len(block_args) + 1

        actual_probabilities = []
        for i, _ in enumerate(block_args):
            block_num = i + 2
            survival_probability = get_survival_probability(
                init_rate=init_rate, block_num=block_num, total_blocks=num_layers
            )
            survival_probability = round(survival_probability, 2)
            actual_probabilities.append(survival_probability)

        self.assertListEqual(expected_survival_probabilities, actual_probabilities)


if __name__ == "__main__":
    absltest.main()
