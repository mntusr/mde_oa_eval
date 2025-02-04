import operator
from typing import Iterable, SupportsIndex

import depth_tools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from typing_extensions import TypeVar


def collect_safe_samples(
    *,
    dataset: depth_tools.Dataset,
    clip_volume: depth_tools.DepthClip,
    required_distance: float,
    report_progress: bool = True,
    indices: Iterable[SupportsIndex] | None = None,
) -> list[int]:
    """
    Get the list of the samples where the camera is not ON or BEHIND the post-depth-clip depth map after dilation. Note that this assumes a dilation where a sphere is drawn around each point with radius.

    Parameters
    ----------
    dataset
        The dataset that contains all samples.
    clip_volume
        The volume to clip.
    required_distance
        The required distance.
    safety_distance
        The safety distance.
    report_progress
        If true, then a tqdm progress bar is shown.
    indices
        If given, then only these samples will be checked and possibly selected.

    Returns
    -------
    v
        The sample indices. Index oder: If the indices iterable present: the same as in that iterable; if not present: ascending.

    Raises
    ------
    ValueError
        If the required distance of the safety distance is negative.
    """
    if required_distance < 0:
        raise ValueError(
            f"The required distance should be non-negative. Current value: {required_distance}"
        )

    if indices is None:
        idx_list = list(range(len(dataset)))
    else:
        idx_list = list(indices)

    result: list[int] = []
    for i in tqdm(idx_list, disable=not report_progress):
        i = operator.index(i)

        sample = dataset[i]

        post_clip_mask = clip_volume.on_mask(
            gt_depth=sample["depth"], mask=sample["mask"]
        )

        if np.any(post_clip_mask):
            point_cloud = depth_tools.depth_2_point_cloud(
                depth_map=sample["depth"],
                depth_mask=post_clip_mask,
                intrinsics=sample["camera"],
                out_coord_sys=depth_tools.CoordSys.LH_YUp,
            )
            distances = np.linalg.norm(point_cloud, axis=1, ord=2)

            if np.all(distances > required_distance):
                result.append(i)

    return result
