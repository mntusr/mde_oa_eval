import datetime
import logging
from typing import Literal, TypedDict, cast

import depth_tools
import depth_tools.pt
import numpy as np
import torch
from matplotlib import pyplot as plt


class AreaChangeDistances(TypedDict):
    safety_distance: float
    required_distance: float


def get_volume_changes(
    aligned_pred: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    camera: depth_tools.CameraIntrinsics,
    area_change_distances: AreaChangeDistances,
    subsampling: depth_tools.PointSubsamplingConf | None,
) -> tuple[float, float]:
    """
    Calculate the pyramid volume changes due to the changed depth.

    Without the validation and masking parts, the pseudocode of the function looks like the following: ::

        # make sure that pred+required_distance+safety_distance > 0
        #   We will assume this for the rest of the code,
        #   so we mask out the pixels for which this does not hold.
        #   We store the number of the mentioned pixels at `negative_aligned_count`

        dilated_gt = dilate_depth(gt, r=required_distance)
        dilated_pred = dilate_depth(pred, r=required_distance+safety_distance)

        gt_volumes = get_pyramid_volumes_from_depths(dilated_gt)
        depth_volumes = get_pyramid_volumes_from_depths(dilated_pred)

        lost_volumes = sum(g-p for g, p in zip(gt_volumes, depth_volumes) if g > p)
        dangerous_volumes = sum(g-p for g, p in zip(gt_volumes, depth_volumes) if p > g)

        return dangerous_volumes, lost_volumes

    Parameters
    ----------
    aligned_pred
        The aligned depth prediction. Format: ``Im_Depth``
    sample
        The sample containing the ground truth and the camera.
    cam
        The camera properties.
    area_change_distances
        The distances that affect the performance. ``safety_distance`` describes the fact that the estimation uncertanity is compensated by keeping an extra distance from the estimated volumes. ``required_distance`` describes the distance needed due to the real-world size of the UAV.
    subsampling
        A dict that configures the point subsampling for the approximation of the dilation.

    Returns
    -------
    dangerous_volume
        The total volume, where the predicted depth is greater than the ground truth depth.
    lost_volume
        The total volume, where the predicted depth is smaller than the ground truth depth.
    """
    _verfiy_area_change_array_args(pred=aligned_pred, gt=gt_depth, mask=mask)

    if area_change_distances["required_distance"] < 0:
        raise ValueError(
            f'The required distance ({area_change_distances["required_distance"]}) is negative.'
        )
    if area_change_distances["safety_distance"] < 0:
        raise ValueError(
            f'The safety distance ({area_change_distances["safety_distance"]}) is negative.'
        )

    # calculate the half size of the cubes inserted during dilation
    r_gt = area_change_distances["required_distance"]
    r_pred = (
        area_change_distances["required_distance"]
        + area_change_distances["safety_distance"]
    )

    # warn the user if no pixel left
    if not torch.any(mask):
        logging.warning("There is no valid pixel inside the evaluation volume.")
        return 0, 0

    # do the dilation on the ground truth depth
    gt_dilated = torch.from_numpy(
        depth_tools.fast_dilate_depth_map(
            depth_mask=mask.detach().cpu().numpy(),
            r=r_gt,
            depth_map=gt_depth.detach().cpu().numpy(),
            intrinsics=camera,
            occlusion_subsampling=subsampling,
        )
    ).to(mask.device)
    # do the dilation on the depth prediction
    pred_dilated = torch.from_numpy(
        depth_tools.fast_dilate_depth_map(
            depth_mask=mask.detach().cpu().numpy(),
            r=r_pred,
            depth_map=aligned_pred.detach().cpu().numpy(),
            intrinsics=camera,
            occlusion_subsampling=subsampling,
        )
    ).to(mask.device)

    # plt.subplot(1, 2, 1)
    # plt.imshow(pred_dilated[0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(gt_dilated[0])
    # plt.show(block=True)
    # plt.close()

    # calculate the pyramid volumes
    pred_volumes = _get_pyramid_volumes_from_depth_map_unchecked(
        depth_map=pred_dilated, depth_mask=mask, cam=camera
    )
    gt_volumes = _get_pyramid_volumes_from_depth_map_unchecked(
        depth_map=gt_dilated, depth_mask=mask, cam=camera
    )
    # dvm = dangerous volume mask
    dvm = (pred_dilated > gt_dilated) & mask

    dangerous_volumes = torch.zeros_like(aligned_pred)
    dangerous_volumes[dvm] = pred_volumes[dvm] - gt_volumes[dvm]

    # lvm = lost volume mask
    lvm = (pred_dilated < gt_dilated) & mask

    lost_volumes = torch.zeros_like(aligned_pred)
    lost_volumes[lvm] = gt_volumes[lvm] - pred_volumes[lvm]

    # plt.imshow(dvm.numpy()[0], vmin=False, vmax=True)
    # plt.show(block=True)
    # plt.close()

    dangerous_volume = dangerous_volumes.sum(dim=(-1, -2))
    lost_volume = lost_volumes.sum(dim=(-1, -2))

    return (
        dangerous_volume.item(),
        lost_volume.item(),
    )


def _verfiy_area_change_array_args(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> None:
    """
    Raise `ValueError` if the arguments do not have the correct format.

    Parameters
    ----------
    pred
        The predicted depth. Format: ``Im_Scalar``
    gt
        The ground truth depth. Format: ``Im_Depth``
    mask
        The mask that selects the valid pixels. Format: ``Im_Mask``
    """
    if len(pred.shape) != 3:
        raise ValueError(
            'The argument "pred" does not have format "Im_Depth", because it is not 3-dimensional.'
        )
    if not pred.is_floating_point():
        raise ValueError(
            'The argument "pred" does not have format "Im_Depth", because it does not have floating point dtype.'
        )
    if pred.shape[0] != 1:
        raise ValueError(
            'The argument "pred" does not have format "Im_Depth", because its number of channels is not equal to 1.'
        )

    if len(gt.shape) != 3:
        raise ValueError(
            'The argument "gt" does not have format "Im_Depth", because it is not 3-dimensional.'
        )
    if not gt.is_floating_point():
        raise ValueError(
            'The argument "gt" does not have format "Im_Depth", because it does not have floating point dtype.'
        )
    if gt.shape[0] != 1:
        raise ValueError(
            'The argument "gt" does not have format "Im_Depth", because its number of channels is not equal to 1.'
        )

    if len(mask.shape) != 3:
        raise ValueError(
            'The argument "mask" does not have format "Im_Mask", because it is not 3-dimensional.'
        )
    if not (mask.dtype == torch.bool):
        raise ValueError(
            'The argument "mask" does not have format "Im_Mask", because it does not have boolean point dtype.'
        )
    if gt.shape[0] != 1:
        raise ValueError(
            'The argument "mask" does not have format "Im_Mask", because its number of channels is not equal to 1.'
        )

    if mask.shape != gt.shape:
        raise ValueError(
            f"The shape of the mask ({mask.shape}) and the ground truth depth map ({gt.shape}) is different."
        )

    if mask.shape != pred.shape:
        raise ValueError(
            f"The shape of the mask ({mask.shape}) and the predicted depth map ({pred.shape}) is different."
        )


def _get_pyramid_volumes_from_depth_map_unchecked(
    depth_map: torch.Tensor, depth_mask: torch.Tensor, cam: depth_tools.CameraIntrinsics
) -> torch.Tensor:
    """
    Get the volumes of the pyramids for each ray. The bottoms of the
    pyramids are parallel to the image plane.

    The function does not check its arguments.

    Parameters
    ----------
    depth_map
        The depth map. Format: ``Im_Depth``
    depth_mask
        The mask that selects the valid pixels from the depth map. Format: ``Im_Mask``
    cam
        The camera intrinsics.
    """
    bottom_width_s = depth_map / torch.full(
        depth_map.shape, abs(cam.f_x), dtype=depth_map.dtype, device=depth_map.device
    )
    bottom_height_s = depth_map / torch.full(
        depth_map.shape, abs(cam.f_y), dtype=depth_map.dtype, device=depth_map.device
    )

    bottom_area_s = bottom_width_s * bottom_height_s

    total_volume_s = bottom_area_s * depth_map / 3

    total_volume_s[~depth_mask] = 0

    return total_volume_s
