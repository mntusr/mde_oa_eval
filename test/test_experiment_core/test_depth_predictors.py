import unittest

import experiment_core


class TestDepthPredictors(unittest.TestCase):
    def test_supported_models(self):
        self.assertEqual(
            experiment_core.SUPPORTED_MODELS,
            ("dummy", "depth_anything_v2", "zoe_depth"),
        )
