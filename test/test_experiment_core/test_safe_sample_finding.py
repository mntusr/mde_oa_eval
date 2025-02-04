import unittest
from typing import SupportsIndex
from unittest import mock

import depth_tools
import numpy as np

import experiment_core


class TestSafeSampleFinding(unittest.TestCase):
    def test_collect_safe_samples__happy_path(self):
        required_distance = 1
        relevant_volume = depth_tools.DepthClip((required_distance / 2, 30))

        depths = np.zeros((4, 1, 30, 25), dtype=np.float32)
        depths[0, :, :, 5:] = required_distance * 2 / 3
        depths[0, :, :10, 5:] = 11
        depths[1, :, :, 5:] = 10
        depths[2, :, :, 5:] = 15
        depths[3, :, :, 5:] = 100

        depth_masks = np.full(depths.shape, True)
        depth_masks[:, :, :, :6] = False

        expected_safe_samples = [1, 2]

        def dataset_getitem(idx: SupportsIndex) -> depth_tools.Sample:
            return {
                "depth": depths[idx],
                "mask": depth_masks[idx],
                "rgb": mock.Mock(),
                "camera": depth_tools.CameraIntrinsics(
                    f_x=100, f_y=100, c_x=10, c_y=10
                ),
                "name": f"sample_{idx}",
            }

        dataset = mock.Mock(name="dataset")
        dataset.__getitem__ = mock.Mock(
            name="dataset.__getitem__", side_effect=dataset_getitem
        )
        dataset.__len__ = mock.Mock(
            name="dataset.__len__", side_effect=lambda: len(depths)
        )

        actual_safe_samples = experiment_core.collect_safe_samples(
            dataset=dataset,
            required_distance=required_distance,
            clip_volume=relevant_volume,
            report_progress=False,
        )

        self.assertEqual(expected_safe_samples, actual_safe_samples)

    def test_collect_safe_samples__invalid_required_distance(self):
        with self.assertRaises(ValueError):
            experiment_core.collect_safe_samples(
                dataset=mock.Mock("dataset"),
                required_distance=-1,
                clip_volume=depth_tools.DepthClip((0.1, 10)),
                report_progress=False,
            )
