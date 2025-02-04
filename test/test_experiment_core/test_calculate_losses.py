from typing import Literal, SupportsIndex
from unittest import mock

import depth_tools
import npy_unittest
import numpy as np
import torch
from matplotlib import pyplot as plt

import experiment_core


class TestCalculateLosses(npy_unittest.NpyTestCase):
    def setUp(self):
        self.required_distance = 3.2
        self.safety_distance = 0.7
        self.im_width = 600
        self.im_height = 400

        self.intrinsics = depth_tools.CameraIntrinsics(
            f_x=100, f_y=100, c_x=300, c_y=200
        )
        self.depth_clip = depth_tools.DepthClip((1, 20))
        """
        The minimum is LESS than the required distance.
        """

    def test_calculate_losses__happy_path(self):
        rng = np.random.default_rng(60)
        n_samples = 6

        # declare depth values and masks
        preds = np.zeros(
            (n_samples, 1, self.im_height, self.im_width), dtype=np.float32
        )
        pre_clip_masks = (
            rng.uniform(0, 1, (n_samples, 1, self.im_height, self.im_width)) > 0.5
        )
        gt_depths = np.zeros(
            (n_samples, 1, self.im_height, self.im_width), dtype=np.float32
        )

        # define different depth values, each value is actually a separate test case
        # simplest case
        gt_depths[0] = self.depth_valid_range(0.2, "gt")
        preds[0] = self.depth_valid_range(0.1, "pred")
        # check the effect of depth clipping
        gt_depths[1] = self.depth_valid_range(0.1, "gt")
        self.set_fmit_inplace(
            arr=gt_depths,
            mask=pre_clip_masks,
            sample_idx=1,
            value=self.depth_valid_range(100, "gt"),
        )
        preds[1] = self.depth_valid_range(0.3, "pred")
        self.set_fmit_inplace(
            arr=preds,
            mask=pre_clip_masks,
            sample_idx=1,
            value=self.depth_valid_range(0.99, "gt"),
        )
        # simplest case again
        gt_depths[2] = self.depth_valid_range(0.95, "gt")
        preds[2] = self.depth_valid_range(0.5, "pred")
        # simplest case 3
        gt_depths[3] = self.depth_valid_range(0.5, "gt")
        preds[3] = self.depth_valid_range(0.8, "pred")
        # dilation is not implemented
        gt_depths[4] = self.depth_valid_range(0.7, "gt")
        self.set_fmit_inplace(
            arr=gt_depths,
            mask=pre_clip_masks,
            sample_idx=4,
            value=self.depth_valid_range(0, "gt"),
        )
        preds[4] = self.depth_valid_range(0.6, "pred")
        # dilation is not implemented for predicted depth, the given pixels should be rejected from the volumetric loss (however this should not produce a warning)
        gt_depths[5] = self.depth_valid_range(0.9, "gt")
        preds[5] = self.depth_valid_range(0.1, "pred")
        self.set_fmit_inplace(
            arr=gt_depths,
            mask=pre_clip_masks,
            sample_idx=5,
            value=self.depth_valid_range(0, "pred"),
        )

        # plt.imshow(gt_depths[4, 0])
        # plt.show(block=True)
        # plt.close()

        # modify the predictions to make sure that the original depth mask has effect
        gt_depths[~pre_clip_masks] = self.depth_valid_range(0.01, "gt")
        preds[~pre_clip_masks] = self.depth_valid_range(200, "pred")

        # calculate the expected loss values
        # calculate the masks after clipping
        post_clip_masks = self.depth_clip.on_mask(
            gt_depth=gt_depths, mask=pre_clip_masks
        )

        cases = [
            ("both", True, True),
            ("only_classic", True, False),
            ("only_volumetric", False, True),
        ]

        for case_name, calculate_classic_losses, calculate_volumetric_losses in cases:
            with self.subTest(case_name):
                # calculate the actual results
                with self.assertNoLogs() as l:
                    actual_results = experiment_core.calculate_losses(
                        predictions=self.new_prediction_cache_mock(preds),
                        area_change_distances=(
                            {
                                "required_distance": self.required_distance,
                                "safety_distance": self.safety_distance,
                            }
                            if calculate_volumetric_losses
                            else None
                        ),
                        dataset=self.new_dataset_mock(
                            camera=self.intrinsics,
                            depth_masks=pre_clip_masks,
                            depths=gt_depths,
                        ),
                        calculate_classic_losses=calculate_classic_losses,
                        device=torch.device("cpu"),
                        eval_volume=self.depth_clip,
                        sample_indices=None,
                        show_progress=False,
                    )

                expected_result_cols: dict[str, np.ndarray] = dict()

                if calculate_volumetric_losses:
                    # initialize the arrays
                    expected_lost_volumes = np.zeros(n_samples, dtype=np.float32)
                    expected_dangerous_volumes = np.zeros(n_samples, dtype=np.float32)
                    expected_sample_indices = np.zeros(n_samples, dtype=np.int32)
                    # calculate the expected volumetric loss values
                    for i in range(n_samples):
                        with self.assertNoLogs():
                            (
                                dangerous_volume,
                                lost_volume,
                            ) = experiment_core.get_volume_changes(
                                aligned_pred=torch.from_numpy(preds[i]),
                                area_change_distances={
                                    "required_distance": self.required_distance,
                                    "safety_distance": self.safety_distance,
                                },
                                gt_depth=torch.from_numpy(gt_depths[i]),
                                mask=torch.from_numpy(post_clip_masks[i]),
                                subsampling=None,
                                camera=self.intrinsics,
                            )
                            expected_lost_volumes[i] = lost_volume
                            expected_dangerous_volumes[i] = dangerous_volume
                            expected_sample_indices[i] = i

                    expected_result_cols = expected_result_cols | {
                        "lost_volume": expected_lost_volumes,
                        "dangerous_volume": expected_dangerous_volumes,
                    }

                if calculate_classic_losses:
                    # create a dict containing the expected loss values
                    expected_result_cols = expected_result_cols | {
                        "mse": depth_tools.mse_loss(
                            gt=gt_depths, mask=post_clip_masks, pred=preds
                        ),
                        "d1": depth_tools.dx_loss(
                            gt=gt_depths, mask=post_clip_masks, pred=preds, x=1
                        ),
                        "d2": depth_tools.dx_loss(
                            gt=gt_depths, mask=post_clip_masks, pred=preds, x=2
                        ),
                        "d3": depth_tools.dx_loss(
                            gt=gt_depths, mask=post_clip_masks, pred=preds, x=3
                        ),
                    }

                if l is not None:
                    # volumetric loss calculation is not always supported -> check if
                    # proper warning(s) appear
                    self.assertEqual(len(l.output), 1)
                    self.assertIn("4", l.output[0])
                    self.assertIn("tting the dangerous and lost volume", l.output[0])

                # compare the expected and actual loss values
                for col_name, expected_col_value in expected_result_cols.items():
                    self.assertIn(col_name, actual_results.columns)
                    acutal_col_value = actual_results[col_name].to_numpy()
                    self.assertAllclose(
                        expected_col_value, acutal_col_value, equal_nan=True
                    )

                # check if the losses that should not exist, really do not exist
                self.assertEqual(
                    len(actual_results.columns), len(expected_result_cols.keys()) + 2
                )

    def set_fmit_inplace(
        self, arr: np.ndarray, mask: np.ndarray, sample_idx: int, value: float
    ) -> None:
        idxs0, idxs1, idxs2 = np.nonzero(mask[sample_idx])

        arr[sample_idx, idxs0[0], idxs1[0], idxs2[0]] = value

    def new_prediction_cache_mock(self, samples: np.ndarray) -> mock.Mock:
        cache_mock = mock.Mock(name="prediction_cache")
        cache_mock.__len__ = mock.Mock(
            name="prediction_cache.__len__", return_value=len(samples)
        )
        cache_mock.__getitem__ = mock.Mock(
            name="prediction_cache.__getitem__", side_effect=lambda i: samples[i]
        )

        return cache_mock

    def depth_valid_range(
        self, v: float, mode: Literal["gt", "pred", "just_clip"]
    ) -> float:
        clip_range = self.depth_clip.clip_range

        if clip_range is None:
            raise RuntimeError("The depth clip is not set to a range.")

        dilat_min = {
            "gt": self.required_distance,
            "pred": self.required_distance + self.safety_distance,
            "just_clip": clip_range[0],
        }[mode]

        vmin = max(clip_range[0], dilat_min)
        vmax = clip_range[1]

        return (vmax - vmin) * v + vmin

    def new_dataset_mock(
        self,
        depths: np.ndarray,
        depth_masks: np.ndarray,
        camera: depth_tools.CameraIntrinsics,
    ) -> mock.Mock:
        dataset_mock = mock.Mock(name="dataset")
        dataset_mock.__len__ = mock.Mock(
            name="dataset.__len__", return_value=len(depths)
        )

        def dataset_getitem(i: SupportsIndex) -> depth_tools.Sample:
            return {
                "camera": camera,
                "depth": depths[i],
                "mask": depth_masks[i],
                "name": f"sample_{i}",
                "rgb": np.zeros(
                    (3, depths.shape[-2], depths.shape[-1]), dtype=np.float32
                ),
            }

        dataset_mock.__getitem__ = mock.Mock(
            name="dataset_mock.__getitem__", side_effect=dataset_getitem
        )

        return dataset_mock
