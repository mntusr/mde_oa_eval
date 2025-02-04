import logging
import os
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    SupportsIndex,
    TypedDict,
)

import depth_tools
import depth_tools.pt
import numpy as np
import pandas as pd
import torch
from depth_tools import Dataset, DepthClip, Sample
from tqdm import tqdm

from ._depth_predictors import DepthPredictionCache
from ._volumetric_loss import AreaChangeDistances, get_volume_changes


class _AreaChangeDistancesWithSubsampling(AreaChangeDistances):
    subsample: NotRequired[depth_tools.PointSubsamplingConf]


def calculate_losses(
    predictions: "DepthPredictions",
    dataset: Dataset,
    show_progress: bool,
    eval_volume: DepthClip,
    device: torch.device,
    area_change_distances: _AreaChangeDistancesWithSubsampling | None,
    sample_indices: Sequence[int] | None = None,
    calculate_classic_losses: bool = True,
) -> pd.DataFrame:
    """
    Calculate the loss functions on a given dataset.

    In more detail, the function uses the following pseudocode to calculate the losses. For the sake of bevity, the sample metadata storing is omitted. ::

        if sample_indices is None:
            sample_indices = range(len(dataset))

        all_losses = []

        for sample_idx in sample_indices:
            sample = dataset[sample_idx]

            mask = depth_clip(sample.mask)
            pred_depth = prediction_cache[sample_idx]

            if calculate_classic_losses:
                classic_losses = calculate_traditional_losses(pred_depth, sample.depth, mask)
            else:
                classic_losses = dict()

            if area_change_distances is not None:
                volumetric_losses = calculate_volumetric_losses(
                    pred_depth,
                    sample.depth,
                    mask,
                    sample.camera
                )
            else:
                volumetric_losses = dict()

            all_losses.append(classic_losses | volumetric_losses)

        return to_dataframe(all_losses)

    The columns of the returned data frame:

    * Always present
      * ``rgb_path``: The paths of the RGB images.
      * ``sample_idx``: The indices of the samples.
    * Present if the classic losses are calculated:
      * ``mse``: The calculated MSE loss for the sample.
      * ``d1``: The calculated d1 loss for the sample.
      * ``d2``: The calculated d2 loss for the sample.
      * ``d3``: The calculated d3 loss for the sample.
    * Present if the extra losses are calculated:
      * ``dangerous_volume``: The total volumes, where the predicted depths are greater than the ground truth depths.
      * ``lost_volume``: The total volumes, where the predicted depths are smaller than the ground truth depths.

    Parameters
    ----------
    model
        The depth prediction model.
    dataset
        The dataset on which the loss should be calculated.
    show_progress
        If true, then a progress bar is shown during the loss calculation.
    eval_volume
        This is the volume in which the depth predictions are evaluated.
    area_loss
        The configuration for the area losses. If it is None, then the area losses are not calculated.
    sample_indices
        If it is not None, then only these samples will be used.
    calculate_classic_losses
        If true, then the classic losses, like RMSE, d1, ... are calculated.

    Returns
    -------
    v
        The data frame that contains a row for each sample.

    Raises
    ------
    TBD
    """

    # initialize everything needed for the iteration
    result_columns: dict[str, list[Any]] = {}
    sample_idx_iter = _get_sample_indices(dataset, sample_indices)
    wrapped_dataset = depth_tools.pt.DatasetWrapper(dataset, device=device)

    # calculate the losses for each sample
    with torch.no_grad():
        for sample_idx in tqdm(sample_idx_iter, disable=not show_progress):
            # load the sample
            sample = wrapped_dataset[sample_idx]
            gt_depth = sample["depth"]
            depth_pred: torch.Tensor = torch.from_numpy(predictions[sample_idx]).to(
                device
            )

            # TODO add alignment regardless (since common)

            # apply depth clipping
            mask_clipped = depth_tools.pt.depth_clip_on_mask(
                clip=eval_volume, gt_depth=sample["depth"], mask=sample["mask"]
            )
            pred_clipped = depth_tools.pt.depth_clip_on_aligned_pred(
                clip=eval_volume,
                aligned_preds=depth_pred,
            )

            # do the classic loss calculation
            if calculate_classic_losses:
                classic_loss_dict = _get_classic_loss_dict(
                    gt_depth=gt_depth, mask=mask_clipped, aligned_pred=pred_clipped
                )
            else:
                classic_loss_dict = dict()

            # do the volumetric loss calculation
            if area_change_distances is not None:
                volume_change_loss_dict = _get_volume_change_loss_dict(
                    aligned_pred=pred_clipped,
                    area_change_distances=area_change_distances,
                    camera=sample["camera"],
                    gt_depth=gt_depth,
                    mask=mask_clipped,
                )
            else:
                volume_change_loss_dict = dict()

            # store everything
            _append_all_in_dict(
                dict_to_mutate=result_columns,
                value_dicts=[
                    classic_loss_dict,
                    volume_change_loss_dict,
                    {"rgb_path": sample["name"], "sample_idx": sample_idx},
                ],
            )

    return pd.DataFrame(result_columns)


class DepthPredictions(Protocol):
    """
    An object with a getitem method that gives back a depth prediction for a sample index.

    The format of the return value of the getitem method: ``Im_Depth``

    The getitem method raises some exception (not defined by this protocol) if there is no depth prediction for a sample index.
    """

    def __getitem__(self, idx: SupportsIndex, /) -> np.ndarray: ...


def _append_all_in_dict(
    dict_to_mutate: dict[str, list[Any]], value_dicts: Iterable[dict[str, Any]]
) -> None:
    """
    The function takes a dict of lists, then appends the proper values to the lists for each value. If the dict to mutate does not contain the proper keys, then the function creates it.

    Parameters
    ----------
    dict_to_mutate
        The dictionary that contains the lists to modify.
    value_dicts
        The values to append.
    """
    for value_dict in value_dicts:
        for key, value in value_dict.items():
            if key not in dict_to_mutate.keys():
                dict_to_mutate[key] = []
            dict_to_mutate[key].append(value)


def _get_sample_indices(
    dataset: depth_tools.Dataset, manual_indices: Sequence[int] | None
) -> np.ndarray:
    """
    Get the sample indices of a dataset. If the indices are manually specified, then they are converted to a numpy array. If they are not specified, then we use ``np.arange(len(dataset))`` to create the indices.

    Parameters
    ----------
    dataset
        The dataset for the indices.
    manual_indices
        The manually specified indices. The function does not check if they are out of bounds.
    """
    if manual_indices is None:
        sample_idx_iter = np.arange(len(dataset))
    else:
        sample_idx_iter = np.array(manual_indices)
    return sample_idx_iter


def _get_volume_change_loss_dict(
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    aligned_pred: torch.Tensor,
    camera: depth_tools.CameraIntrinsics,
    area_change_distances: _AreaChangeDistancesWithSubsampling,
) -> dict[str, float]:
    """
    Get the dictionary that contains the volumetric loss values.

    The function shows a warning if the dilation is not supported on the ground truth depth map.

    The keys of the dictionary:

    * ``dangerous_volume``: The total dangerous volume.
    * ``lost_volume``: The total lost volume.

    Parameters
    ----------
    gt_depth
        The grund truth depth. Format: ``Im_Depth``
    mask
        The mask that selects the valid depth values. Format: ``Im_Mask``
    aligned_pred
        The aligned depth prediction. Format: ``Im_Depth``
    camera
        The camera.
    sample_idx
        The index of the sample.
    area_change_distances
        A dictionary that configures the volumetric loss.

    Returns
    -------
    v
        The dictionary containing the losses. If the dilation is not implemented for the ground truth depths, then the dangerous and lost volume will be nan, and the negative aligned count will be 0.

    Raises
    ------
    ValueError
        If the format of the given arrays is incorrect.

        If the required distance of the safety distance is negative.
    NotImplementedError
        If any of the focal lengths of the camera is negative.
    """

    subsampling_conf = area_change_distances.get("subsample", None)
    dangerous_volume, lost_volume = get_volume_changes(
        aligned_pred=aligned_pred,
        gt_depth=gt_depth,
        mask=mask,
        camera=camera,
        area_change_distances=area_change_distances,
        subsampling=subsampling_conf,
    )

    return {"dangerous_volume": dangerous_volume, "lost_volume": lost_volume}


def _get_classic_loss_dict(
    gt_depth: torch.Tensor,
    mask: torch.Tensor,
    aligned_pred: torch.Tensor,
) -> dict[str, float]:
    """
    Get a dictionary that contains the loss values for the given sample and prediction.

    The keys of the dictionary:

    * `mse`: The MSE loss.
    * `d1`: The d1 loss.
    * `d2`: The d2 loss.
    * `d3`: The d3 loss.

    Parameters
    ----------
    sample
        The sample.
    aligned_pred
        The aligned depth prediction. Format: ``Im_Depth``
    eval_volume
        The volume in which the depth predictions are evaluated. In other words, the volume in which depth clipping is applied.

    Returns
    -------
    v
        The dictionary containing the losses.
    """

    classic_losses: dict[str, float] = {
        "d1": depth_tools.pt.dx_loss(
            pred=aligned_pred,
            gt=gt_depth,
            mask=mask,
            verify_args=True,
            x=1,
        ).item(),
        "d2": depth_tools.pt.dx_loss(
            pred=aligned_pred,
            gt=gt_depth,
            mask=mask,
            verify_args=True,
            x=2,
        ).item(),
        "d3": depth_tools.pt.dx_loss(
            pred=aligned_pred,
            gt=gt_depth,
            mask=mask,
            verify_args=True,
            x=3,
        ).item(),
        "mse": depth_tools.pt.mse_loss(
            pred=aligned_pred,
            gt=gt_depth,
            mask=mask,
            verify_args=True,
        ).item(),
    }

    return classic_losses
