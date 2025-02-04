import datetime
import math
import unittest
from typing import TypedDict
from unittest import mock

import depth_tools
import depth_tools.pt
import npy_unittest
import numpy as np
import torch

import experiment_core
from experiment_core._volumetric_loss import (
    _get_pyramid_volumes_from_depth_map_unchecked,
)


class TestExtraLosses(npy_unittest.NpyTestCase):
    def test_get_pyramid_volumes_from_depth_map_unchecked__happy_path(self):
        camera = depth_tools.CameraIntrinsics(f_x=20, f_y=30, c_x=40, c_y=50)
        depth = np.array(
            [
                [1, 3, 5],
                [4, 1, 9],
            ],
            dtype=np.float32,
        )
        depth = np.expand_dims(depth, axis=0)
        bottom_areas = np.zeros_like(depth)
        mask = np.full(depth.shape, True)
        mask[:, 1, 1] = False

        for x in range(3):
            for y in range(2):
                depth_px = depth[:, y, x]
                bottom_area_width = depth_px / camera.f_x
                bottom_area_height = depth_px / camera.f_y
                bottom_areas[:, y, x] = bottom_area_width * bottom_area_height / 3

        expected_volumes = bottom_areas * depth
        expected_volumes = expected_volumes * mask

        with torch.no_grad():
            actual_volumes = _get_pyramid_volumes_from_depth_map_unchecked(
                depth_map=torch.from_numpy(depth),
                cam=camera,
                depth_mask=torch.from_numpy(mask),
            ).numpy()

        self.assertAllclose(actual_volumes, expected_volumes)

    def test_get_pyramid_volumes_from_depth_map_unchecked__focal_length_assumption(
        self,
    ):
        im_width = 200
        im_height = 100
        depth_val = 10
        camera = depth_tools.CameraIntrinsics(f_x=20, f_y=30, c_x=40, c_y=50)
        depth = np.full((1, im_height, im_width), depth_val, dtype=np.float32)
        mask = np.ones(depth.shape, dtype=np.bool)

        proj_mat_inv = camera.get_intrinsic_mat_inv()

        pt_00 = proj_mat_inv @ np.array(
            [
                [0],
                [0],
                [depth_val],
            ],
            dtype=np.float32,
        )

        pt_0h = proj_mat_inv @ np.array(
            [
                [0],
                [im_height * depth_val],
                [depth_val],
            ],
            dtype=np.float32,
        )

        pt_w0 = proj_mat_inv @ np.array(
            [
                [im_width * depth_val],
                [0],
                [depth_val],
            ],
            dtype=np.float32,
        )

        area = abs(pt_w0[0, 0] - pt_00[0, 0]) * abs(pt_0h[1, 0] - pt_00[1, 0])
        height = depth_val

        expected_volume = area * height / 3
        with torch.no_grad():
            actual_volume = (
                _get_pyramid_volumes_from_depth_map_unchecked(
                    cam=camera,
                    depth_map=torch.from_numpy(depth),
                    depth_mask=torch.from_numpy(mask),
                )
                .sum()
                .item()
            )

        self.assertAlmostEqual(expected_volume, actual_volume, places=2)

    def test_get_volume_changes__happy_path(self):
        subtests = [(True, 42, 51), (False, 51, 42)]
        for pred_less_than_gt, pred_values_all, gt_values_all in subtests:
            with self.subTest(f"{pred_less_than_gt=}"):
                gt = np.full((1, 250, 400), gt_values_all, dtype=np.float32)
                pred = np.full((1, 250, 400), pred_values_all, dtype=np.float32)
                mask = np.full(gt.shape, True)
                safety_distance = 3
                required_distance = 6

                cam = depth_tools.CameraIntrinsics(f_x=100, f_y=100, c_x=125, c_y=200)

                (
                    actual_total_dangerous_volume,
                    actual_total_lost_volume,
                ) = experiment_core.get_volume_changes(
                    aligned_pred=torch.from_numpy(pred),
                    gt_depth=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask),
                    camera=cam,
                    area_change_distances={
                        "safety_distance": safety_distance,
                        "required_distance": required_distance,
                    },
                    subsampling=None,
                )

                gt_dilated = gt - required_distance
                pred_dilated = pred - (required_distance + safety_distance)

                gt_volumes = _get_pyramid_volumes_from_depth_map_unchecked(
                    cam=cam,
                    depth_map=torch.from_numpy(gt_dilated),
                    depth_mask=torch.from_numpy(mask),
                ).numpy()
                pred_volumes = _get_pyramid_volumes_from_depth_map_unchecked(
                    cam=cam,
                    depth_map=torch.from_numpy(pred_dilated),
                    depth_mask=torch.from_numpy(mask),
                ).numpy()

                expected_total_dangerous_volume: float = 0
                expected_total_lost_volume: float = 0
                if pred_less_than_gt:
                    expected_total_lost_volume = (
                        (gt_volumes - pred_volumes).sum().item()
                    )
                else:
                    expected_total_dangerous_volume = (
                        (pred_volumes - gt_volumes).sum().item()
                    )

                self.assertAlmostEqual(
                    actual_total_dangerous_volume,
                    expected_total_dangerous_volume,
                    delta=1e-2,
                )
                self.assertAlmostEqual(
                    actual_total_lost_volume, expected_total_lost_volume, delta=1e-2
                )

    def test_get_volume_changes__unsafe_gt(self):
        gt = np.full((1, 600, 800), 70, dtype=np.float32)
        mask = np.full((1, 600, 800), True)
        gt[:, 200, 300] = 1
        camera = depth_tools.CameraIntrinsics(f_x=100, f_y=100, c_x=400, c_y=300)
        safety_distance = 1e-50
        required_distance = 3
        pred = np.full((1, 600, 800), 70, dtype=np.float32)

        with self.assertNoLogs():
            actual_dangerous_volume, actual_lost_volume = (
                experiment_core.get_volume_changes(
                    aligned_pred=torch.from_numpy(pred),
                    area_change_distances={
                        "safety_distance": safety_distance,
                        "required_distance": required_distance,
                    },
                    gt_depth=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask),
                    camera=camera,
                    subsampling=None,
                )
            )

            self.assertFalse(math.isnan(actual_dangerous_volume))
            self.assertFalse(math.isnan(actual_lost_volume))

            self.assertAlmostEqual(actual_lost_volume, 0)
            self.assertGreater(actual_dangerous_volume, 1)

    def test_get_volume_changes__unsafe_pred(self):
        gt = np.full((1, 600, 800), 70, dtype=np.float32)
        mask = np.full((1, 600, 800), True)
        camera = depth_tools.CameraIntrinsics(f_x=100, f_y=100, c_x=400, c_y=300)
        safety_distance = 1e-50
        required_distance = 3
        pred = gt.copy() * 1.5
        pred[:, 200, 300] = 1

        with self.assertNoLogs():
            actual_dangerous_volume, actual_lost_volume = (
                experiment_core.get_volume_changes(
                    aligned_pred=torch.from_numpy(pred),
                    area_change_distances={
                        "safety_distance": safety_distance,
                        "required_distance": required_distance,
                    },
                    gt_depth=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask),
                    camera=camera,
                    subsampling=None,
                )
            )

            self.assertFalse(math.isnan(actual_dangerous_volume))
            self.assertFalse(math.isnan(actual_lost_volume))

            self.assertAlmostEqual(actual_dangerous_volume, 0)
            self.assertGreater(actual_lost_volume, 1)

    def test_get_volume_changes__no_valid_pixel(self):
        safety_distance = 0.1
        required_distance = 3

        gt = np.full((1, 600, 800), required_distance / 2, dtype=np.float32)
        pred = gt * 2

        camera = depth_tools.CameraIntrinsics(f_x=100, f_y=100, c_x=400, c_y=300)

        mask = np.full(gt.shape, False)

        with self.assertLogs():
            actual_dangerous_volume, actual_lost_volume = (
                experiment_core.get_volume_changes(
                    aligned_pred=torch.from_numpy(pred),
                    area_change_distances={
                        "safety_distance": safety_distance,
                        "required_distance": required_distance,
                    },
                    gt_depth=torch.from_numpy(gt),
                    mask=torch.from_numpy(mask),
                    camera=camera,
                    subsampling=None,
                )
            )
            self.assertAlmostEqual(actual_dangerous_volume, 0)
            self.assertAlmostEqual(actual_lost_volume, 0)

    def _run_area_change_test(
        self,
        gt_depth: np.ndarray,
        pred_depth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        """
        Execute `experiment_core.get_area_changes` as part of a test.

        Parameters
        ----------
        gt_depth
            The ground truth depth values. Format: correctly ``Im_Depth``, but it might be different if the goal is to test the error handling.
        pred_depth
            The predicted depth values. Format: correctly ``Im_Depth``, but it might be different if the goal is to test the error handling.
        mask
            The predicted depth values. Format: correctly ``Im_Mask``, but it might be different if the goal is to test the error handling.
        """
        depth_min = np.minimum(gt_depth, pred_depth).min().item()

        cam = depth_tools.CameraIntrinsics(f_x=4, f_y=4, c_x=3.7, c_y=4.2)
        experiment_core.get_volume_changes(
            aligned_pred=torch.from_numpy(pred_depth),
            area_change_distances={
                "safety_distance": depth_min / 3,
                "required_distance": depth_min / 3,
            },
            gt_depth=torch.from_numpy(gt_depth),
            mask=torch.from_numpy(mask),
            camera=cam,
            subsampling={"max_num": 20_000},
        )

    def _get_pyramid_volume(
        self, height: float, bottom_width: float, bottom_height: float
    ) -> float:
        if height < 0:
            raise RuntimeError(f"The height of the pyramid is negative ({height}).")
        if bottom_width < 0:
            raise RuntimeError(
                f"The width of the bottom of the pyramid is negative ({height})."
            )
        if bottom_height < 0:
            raise RuntimeError(
                f"The height of the bottom of the pyramid is negative ({height})."
            )

        return height * bottom_width * bottom_height / 3


class _AreaChangeCalcTestDict(TypedDict):
    lost_volume: float
    dangerous_volume: float
    negative_pixel_count: float
