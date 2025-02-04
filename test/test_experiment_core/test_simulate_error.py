import unittest
from unittest import mock

import depth_tools
import npy_unittest
import numpy as np
from matplotlib import pyplot as plt

import experiment_core


class TestDepthDatasetWithError(npy_unittest.NpyTestCase):
    def setUp(self):
        self.blur_size = 7

        self.initial_shape = (1, 326, 411)

        sample0_depth = np.ones(self.initial_shape, dtype=np.float32)
        sample0_depth[:, :, 20:300] = 15
        sample0_mask = np.full(self.initial_shape, True, dtype=np.bool_)
        sample0_mask[:, :, 53:120] = False

        sample1_depth = sample0_depth.copy()
        sample1_depth[:, :, 53:120] = 0
        sample1_mask = sample0_mask.copy()

        rng = np.random.default_rng(57)
        sample0_rgb = rng.uniform(
            0, 1, (3, self.initial_shape[0], self.initial_shape[1])
        )
        sample1_rgb = rng.uniform(
            0, 1, (3, self.initial_shape[0], self.initial_shape[1])
        )

        self.original_samples: list[depth_tools.Sample] = [
            {
                "depth": sample0_depth,
                "mask": sample0_mask,
                "name": "sample0",
                "rgb": sample0_rgb,
                "camera": depth_tools.CameraIntrinsics(1, 2, 3, 4),
            },
            {
                "depth": sample1_depth,
                "mask": sample1_mask,
                "name": "sample1",
                "rgb": sample1_rgb,
                "camera": depth_tools.CameraIntrinsics(5, 6, 7, 8),
            },
        ]

        self.original_dataset_mock = mock.Mock("original_dataset")
        self.original_dataset_mock.__len__ = mock.Mock(
            "original_dataset.__len__", side_effect=lambda: len(self.original_samples)
        )
        self.original_dataset_mock.__getitem__ = mock.Mock(
            "original_dataset.__getitem__",
            side_effect=lambda idx: self.original_samples[idx],
        )

        self.tested_dataset = experiment_core.DepthDatasetWithError(
            dataset=self.original_dataset_mock,
            blur_size=self.blur_size,
            multiplicative_noise_fn=self.multiplicative_noise_fn,
        )

    def multiplicative_noise_fn(self, idx: int) -> np.ndarray:
        return np.full(self.initial_shape, idx + 1, dtype=np.float32)

    def test_len(self):
        self.assertEqual(len(self.tested_dataset), 2)

    def test_shape(self):
        self.assertEqual(self.initial_shape, self.tested_dataset[0]["depth"].shape)

    def test_ignoring_mask(self):
        sample0 = self.tested_dataset[0]
        sample1 = self.tested_dataset[1]

        self.assertAllclose(
            sample0["depth"][sample0["mask"]] * 2, sample1["depth"][sample1["mask"]]
        )

    def test_intact_things(self):
        for i in [0, 1]:
            blurred_sample = self.tested_dataset[i]
            self.assertAllclose(blurred_sample["rgb"], self.original_samples[i]["rgb"])
            self.assertArrayEqual(
                blurred_sample["mask"], self.original_samples[i]["mask"]
            )
            self.assertEqual(
                blurred_sample["camera"], self.original_samples[i]["camera"]
            )

    def test_blur_application(self):
        sample1 = self.tested_dataset[1]

        # plt.imshow(sample1["depth"][0])
        # plt.show(block=True)

        self.assertGreater(
            sample1["depth"][0, 9, 19], self.original_samples[1]["depth"][0, 9, 19] * 2
        )
        self.assertLess(
            sample1["depth"][0, 9, 21], self.original_samples[1]["depth"][0, 9, 21] * 2
        )

    def test_blur_size(self):
        self.assertEqual(self.tested_dataset.blur_size, self.blur_size)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            experiment_core.DepthDatasetWithError(
                dataset=mock.Mock(),
                blur_size=0,
                multiplicative_noise_fn=self.multiplicative_noise_fn,
            )

    def test_no_multiplicative_noise(self):
        dataset_without_multiplicative_noise = experiment_core.DepthDatasetWithError(
            dataset=self.original_dataset_mock,
            multiplicative_noise_fn=None,
            blur_size=self.blur_size,
        )

        for i in [0, 1]:
            sample_with_multip_noise = self.tested_dataset[i]
            sample_without_multip_noise = dataset_without_multiplicative_noise[i]

            # plt.subplot(1, 2, 1)
            # plt.imshow(sample_with_multip_noise["depth"][0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(sample_without_multip_noise["depth"][0] * (i + 1))
            # plt.show(block=True)
            # plt.close()

            # plt.imshow(
            #     abs(
            #         sample_with_multip_noise["depth"][0]
            #         - sample_without_multip_noise["depth"][0] * (i + 1)
            #    )
            # )
            # plt.show(block=True)
            # plt.close()

            self.assertAllclose(
                sample_with_multip_noise["depth"],
                sample_without_multip_noise["depth"] * (i + 1),
            )
