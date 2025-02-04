import unittest

import numpy as np

import experiment_core


class TestGenerateIdxSet(unittest.TestCase):
    def test_generate_idx_set__happy_path(self):
        got_sequence = experiment_core.generate_idx_set(
            n_all_elements=26, n_elements_to_choose=9
        )
        self.assertEqual(len(got_sequence), 9)

    def test_too_many_elements_sampled(self):
        with self.assertRaises(ValueError):
            experiment_core.generate_idx_set(n_all_elements=26, n_elements_to_choose=27)

    def test_generate_idx_set__determinism(self):
        got_sequence1 = experiment_core.generate_idx_set(
            n_all_elements=26, n_elements_to_choose=9, rng=np.random.default_rng(9)
        )
        got_sequence2 = experiment_core.generate_idx_set(
            n_all_elements=26, n_elements_to_choose=9, rng=np.random.default_rng(9)
        )
        self.assertEqual(got_sequence1, got_sequence2)

    def test_generate_idx_set__non_determinism(self):
        got_sequence1 = experiment_core.generate_idx_set(
            n_all_elements=26, n_elements_to_choose=9, rng=np.random.default_rng(9)
        )
        got_sequence2 = experiment_core.generate_idx_set(
            n_all_elements=26, n_elements_to_choose=9, rng=np.random.default_rng(13)
        )
        self.assertNotEqual(got_sequence1, got_sequence2)

    def test_generate_idx_set__invalid_args(self):
        cases = [
            (
                "negative total element count",
                -1,
                9,
                "number of elements in the dataset",
            ),
            ("0 total element count", 0, 9, "number of elements in the dataset"),
            (
                "negative chosen element count",
                9,
                -1,
                "number of elements to choose should",
            ),
            ("0 chosen element count", 9, 0, "number of elements to choose should"),
        ]

        for subtest_name, n_all_elements, n_elements_to_choose, error_part in cases:
            with self.subTest(subtest_name):
                with self.assertRaises(ValueError) as cm:
                    experiment_core.generate_idx_set(
                        n_all_elements=n_all_elements,
                        n_elements_to_choose=n_elements_to_choose,
                    )

                    self.assertIn(error_part, str(cm))
